# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet (3.12.13)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import json
from pathlib import Path

from loguru import logger as L
import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data import add_metadata_features
from src.models.causal6 import run_acoustic_searchlight, _resolve_target, _has_enough_per_class
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS

# %%
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
epp_path = "outputs/causal46_joined/early_perceptual_projection/all_sites.csv"
reg_lambda_winners_path = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json"

epoch_tmin = -0.4
epoch_sfreq = 100

n_folds = 5
cv_random_state = 42
device = "cpu"
tol = 1e-6
max_iter = 50

# %%
b4_per_cell = pd.read_parquet(b4_per_cell_path)

# %%
epp = pd.read_csv(epp_path)
epp["significant"] = epp.q_one_tailed < 0.05
epp["significant_uncorrected"] = epp.p_one_tailed < 0.05

# %%
reg_lambda = json.loads(Path(reg_lambda_winners_path).read_text())["reg_lambda_acoustic"]

# %%
epochs_dict = {}
for p in Path("outputs/epochs_preprocessed").glob("*.fif"):
    ep = mne.read_epochs(p, verbose=False)
    ep.metadata = add_metadata_features(ep.metadata)
    epochs_dict[p.stem.rstrip("_epo")] = ep

# %%
to_study = pd.merge(
    epp[["subject", "electrode_idx", "phoneme_pair"]],
    b4_per_cell,
    how="left", on=["subject", "electrode_idx", "phoneme_pair"]
)

# %%
import polars as pl
from sklearn.model_selection import StratifiedKFold
import torch

from src.models.decoding_gpu import batched_roc_auc, compute_balanced_sample_weight, fit_batched_l2_logreg, standardise_per_batch


def _fit_batched_cv(
    X: np.ndarray,            # (n_trials, B, d) — B problems on the same trials
    y: np.ndarray,             # (n_trials,)      — labels
    epoch_idxs: np.ndarray,    # (n_trials,)      — original metadata index per trial

    X_ho_test: np.ndarray,       # (n_test_trials, B, d) — B problems on the same trials
    y_ho_test: np.ndarray,        # (n_test_trials,)      — labels
    
    problem_meta: pl.DataFrame,  # (B rows) — identifying columns for each problem
    *,
    reg_lambda: float,
    n_folds: int,
    cv_random_state: int,
    device: str,
    dtype: torch.dtype,
    tol: float,
    max_iter: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Run StratifiedKFold outer CV over B batched binary LogReg problems.

    Each fold:
      - per-problem standardise X on the train fold
      - batched GPU fit on train
      - predict on test fold, compute per-problem AUC

    Args:
        X: (n_trials, B, d) feature tensor. Layout is (trials, problems, features)
            so splitting by trial indices is a straight slice on dim 0.
        y: (n_trials,) binary labels. Same labels across the B problems.
        epoch_idxs: (n_trials,) original pandas-index values into the source
            metadata frame.
        problem_meta: polars DataFrame with B rows. Its columns become the
            identifying keys in the returned DataFrames (e.g. electrode_idx, smin, smax).

    Returns:
        scores: one row per (problem × fold), columns from problem_meta + fold,
            test_roc_auc, n_train, n_test, n_iter, converged.
        predictions: one row per (problem × fold × test-trial), columns from problem_meta
            + fold, epoch_idx, decoder_target, decoder_proba.
        coefficients: one row per (problem × fold), columns from problem_meta + fold,
            coef (list[f32]), mean (list[f32]), scale (list[f32]).
    """
    n_trials, B, d = X.shape
    assert y.shape == (n_trials,)
    assert epoch_idxs.shape == (n_trials,)
    assert problem_meta.height == B, (
        f"problem_meta has {problem_meta.height} rows, expected B={B}"
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)
    y_gpu = torch.tensor(y.astype(np.float64), dtype=dtype, device=device)
    # Upload the whole feature tensor to GPU once, already transposed so the
    # problem dim is first. Each fold then slices rows via torch.index_select
    # on-GPU instead of numpy-fancy-index-copy + transpose.copy() + upload per
    # fold (~65s across the sweep: 37s numpy copy + 28s torch.tensor upload).
    X_gpu = torch.tensor(
        X.transpose(1, 0, 2).copy(), dtype=dtype, device=device
    )  # (B, n_trials, d)

    X_ho_te_gpu = torch.tensor(
        X_ho_test.transpose(1, 0, 2).copy(), dtype=dtype, device=device
    )  # (B, n_ho_te, d)

    scores_frames: list[pl.DataFrame] = []
    predictions_frames: list[pl.DataFrame] = []
    coefficients_frames: list[pl.DataFrame] = []

    problem_idx = np.arange(B, dtype=np.int64)
    problem_meta_with_idx = problem_meta.with_row_index("_problem_idx").with_columns(
        pl.col("_problem_idx").cast(pl.Int64)
    )

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_trials), y)):
        n_tr, n_te = len(train_idx), len(test_idx)

        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=device)

        X_train_t = X_gpu.index_select(1, train_idx_t)  # (B, n_tr, d)
        X_test_t = X_gpu.index_select(1, test_idx_t)    # (B, n_te, d)
        y_train = (
            y_gpu.index_select(0, train_idx_t)
            .unsqueeze(0).expand(B, -1).contiguous()
        )
        mask_tr = torch.ones(B, n_tr, dtype=dtype, device=device)

        X_tr_std, X_te_std, mean, scale = standardise_per_batch(X_train_t, mask_tr, X_test_t)
        X_ho_te_std = standardise_per_batch(X_train_t, mask_tr, X_ho_te_gpu)[1]
        sw_tr = compute_balanced_sample_weight(y_train, mask_tr)

        beta, n_iter, conv = fit_batched_l2_logreg(
            X_tr_std, y_train, mask_tr, sw_tr,
            reg_lambda=reg_lambda, tol=tol, max_iter=max_iter,
        )

        z_te = torch.einsum("bnd,bd->bn", X_te_std, beta)
        proba_te_gpu = torch.sigmoid(z_te)  # (B, n_te)

        # Vectorised per-problem AUCs on GPU; NaN when only one class is present.
        y_te = y[test_idx]  # numpy slice — still needed for the predictions frame
        y_te_gpu = y_gpu.index_select(0, test_idx_t)
        aucs_gpu = batched_roc_auc(proba_te_gpu, y_te_gpu)

        proba_te = proba_te_gpu.cpu().numpy()
        aucs = aucs_gpu.cpu().numpy()

        # also compute on held-out test test
        z_ho_te = torch.einsum("bnd,bd->bn", X_ho_te_std, beta)
        proba_ho_te_gpu = torch.sigmoid(z_ho_te)  # (B, n_ho_te)
        y_ho_te_gpu = torch.tensor(y_ho_test.astype(np.float64), dtype=dtype, device=device).unsqueeze(0).expand(B, -1).contiguous()
        aucs_ho_gpu = batched_roc_auc(proba_ho_te_gpu, y_ho_te_gpu)
        aucs_ho = aucs_ho_gpu.cpu().numpy()

        # --- scores ---
        scores_frames.append(pl.DataFrame({
            "_problem_idx": problem_idx,
            "fold": np.full(B, fold, dtype=np.int32),
            "test_roc_auc": aucs,

            "ho_test_roc_auc": aucs_ho,
            "ho_n_test": np.full(B, len(y_ho_test), dtype=np.int64),

            "n_train": np.full(B, n_tr, dtype=np.int64),
            "n_test": np.full(B, n_te, dtype=np.int64),
            "n_iter": n_iter.cpu().numpy().astype(np.int32),
            "converged": conv.cpu().numpy(),
        }))

        # --- coefficients ---
        # Use fixed-width pl.Array columns so polars can ingest the 2D numpy
        # arrays wholesale. pl.List + .tolist() would force element-by-element
        # type inference over B*d floats (~55s across the sweep).
        beta_np = beta.cpu().numpy().astype(np.float32)
        mean_np = mean.cpu().numpy().astype(np.float32)
        scale_np = scale.cpu().numpy().astype(np.float32)
        coefficients_frames.append(pl.DataFrame([
            pl.Series("_problem_idx", problem_idx, dtype=pl.Int64),
            pl.Series("fold", np.full(B, fold, dtype=np.int32), dtype=pl.Int32),
            pl.Series("coef", beta_np, dtype=pl.Array(pl.Float32, d)),
            pl.Series("mean", mean_np, dtype=pl.Array(pl.Float32, d)),
            pl.Series("scale", scale_np, dtype=pl.Array(pl.Float32, d)),
        ]))

        # --- predictions ---
        # Long format: (B × n_te) rows per fold
        predictions_frames.append(pl.DataFrame({
            "_problem_idx": np.repeat(problem_idx, n_te),
            "fold": np.full(B * n_te, fold, dtype=np.int32),
            "epoch_idx": np.tile(epoch_idxs[test_idx], B),
            "decoder_target": np.tile(y_te, B).astype(np.int8),
            "decoder_proba": proba_te.reshape(-1),
        }))

    # Concat across folds, join problem_meta back in to replace _problem_idx with actual keys
    scores = pl.concat(scores_frames).join(
        problem_meta_with_idx, on="_problem_idx", how="left"
    ).drop("_problem_idx")
    predictions = pl.concat(predictions_frames).join(
        problem_meta_with_idx, on="_problem_idx", how="left"
    ).drop("_problem_idx")
    coefficients = pl.concat(coefficients_frames).join(
        problem_meta_with_idx, on="_problem_idx", how="left"
    ).drop("_problem_idx")
    # Cast the fixed-width Array cols back to List so callers can vstack
    # results across problems with different feature counts (e.g. full vs
    # baseline model in _run_behavior_core).
    coefficients = coefficients.with_columns(
        pl.col("coef").cast(pl.List(pl.Float32)),
        pl.col("mean").cast(pl.List(pl.Float32)),
        pl.col("scale").cast(pl.List(pl.Float32)),
    )
    return scores, predictions, coefficients


# %%
from typing import Literal, Sequence

import torch


def run_acoustic_cross_decoder(
    epochs: mne.Epochs,
    subject: str,
    phoneme_pair: str,
    word_end: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    reg_lambda: float,
    target: Literal["categorical_acoustic_cue", "subject_specific_acoustics"]
        = "categorical_acoustic_cue",
    resampled_steps: tuple[int, ...] = (1, 6),
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
):
    assert epochs.metadata is not None
    md = epochs.metadata

    alternate_word_end = next(iter(set(PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]) - {word_end}))

    target_mask = (md.word_end == word_end).values
    alternate_mask = (md.word_end == alternate_word_end).values

    data = epochs.get_data(picks=list(electrode_idxs))

    n_electrodes = len(electrode_idxs)
    n_windows = windows.shape[0]
    win_size = int(windows[0, 1] - windows[0, 0])
    assert (windows[:, 1] - windows[:, 0] == win_size).all()

    # Restrict to unambiguous stimulus steps by default (project convention).
    resampled_mask = md.resampled.isin(resampled_steps).values

    scores_all, preds_all, coefs_all = [], [], []

    target_selection = (md.phoneme_pair == phoneme_pair).values & target_mask & resampled_mask
    if target_selection.sum() == 0:
        return None, None, None
    alternate_selection = (md.phoneme_pair == phoneme_pair).values & alternate_mask & resampled_mask

    y = _resolve_target(md, target, phoneme_pair, target_selection)
    if not _has_enough_per_class(y, n_folds):
        counts = np.bincount(y.astype(np.int64)) if len(np.unique(y)) > 0 else []
        L.warning(
            f"[acoustic][{subject}/{phoneme_pair}] skipping: insufficient "
            f"class balance for StratifiedKFold(n_splits={n_folds}); "
            f"class counts = {list(counts)}"
        )
        return None, None, None

    epoch_idxs_sel = md.index[target_selection].to_numpy()
    X_sel = data[target_selection]  # (n_trials, n_electrodes, n_samples)
    X_test = data[alternate_selection]
    y_test = _resolve_target(md, target, phoneme_pair, alternate_selection)
    n_trials = X_sel.shape[0]

    B = n_electrodes * n_windows
    X_batch = np.empty((n_trials, B, win_size), dtype=np.float64)
    X_test_batch = np.empty((len(y_test), B, win_size), dtype=np.float64)
    elec_per = np.empty(B, dtype=np.int64)
    smin_per = np.empty(B, dtype=np.int64)
    smax_per = np.empty(B, dtype=np.int64)
    b = 0
    for e_idx, electrode_idx in enumerate(electrode_idxs):
        for smin, smax in windows:
            X_batch[:, b, :] = X_sel[:, e_idx, smin:smax]
            X_test_batch[:, b, :] = X_test[:, e_idx, smin:smax]

            elec_per[b] = int(electrode_idx)
            smin_per[b] = int(smin)
            smax_per[b] = int(smax)
            b += 1

    problem_meta = pl.DataFrame({
        "subject": [subject] * B,
        "phoneme_pair": [phoneme_pair] * B,
        "electrode_idx": elec_per,
        "smin": smin_per,
        "smax": smax_per,
    })

    scores, preds, coefs = _fit_batched_cv(
        X_batch, y,
        epoch_idxs_sel,
        X_test_batch, y_test,
        problem_meta,
        reg_lambda=reg_lambda,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )
    scores_all.append(scores)
    preds_all.append(preds)
    coefs_all.append(coefs)

    if not scores_all:
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    return (
        pl.concat(scores_all).with_columns(pl.lit(target).alias("target")),
        pl.concat(preds_all).with_columns(
            (pl.col("decoder_proba") > 0.5).cast(pl.Int8).alias("decoder_prediction"),
            pl.lit(target).alias("target"),
        ),
        pl.concat(coefs_all).with_columns(pl.lit(target).alias("target")),
    )


# %%
dec_results = []

# Run acoustic decoding by training on the target word end
# and evaluating on both (1) held-out trials with the target word-end and
# (2) held-out trials with the alternate word-end.
#
# We expect (1) to succeed (generalizable decoder) but (2) to fail
# because the acoustic contrast is specific to one word end.
for _, row in tqdm(to_study.iterrows(), total=len(to_study)):
    subject = row["subject"]
    electrode_idx = row["electrode_idx"]
    phoneme_pair = row["phoneme_pair"]
    word_end = row["word_end"]
    other_word_end = next(iter(set(PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]) - {word_end}))

    phon_smin = row["phon_smin"]
    phon_smax = row["phon_smax"]

    dec_smin = phon_smax
    dec_smax = int(round((OFFSET_DICT[word_end] + 0.1 - epoch_tmin) * epoch_sfreq))
    dec_smax_other = int(round((OFFSET_DICT[other_word_end] + 0.1 - epoch_tmin) * epoch_sfreq))

    ep_i = epochs_dict[subject]

    scores, _, _ = run_acoustic_cross_decoder(
        ep_i,
        subject=subject,
        electrode_idxs=[electrode_idx],
        phoneme_pair=phoneme_pair,
        word_end=word_end,
        windows=np.array([(dec_smin, dec_smax)]),
        target="categorical_acoustic_cue",
        reg_lambda=reg_lambda,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
        device=device,
        tol=tol,
        max_iter=max_iter,
    )

    dec_results.append(
        scores
        .to_pandas()
        .assign(
            subject=subject,
            word_end=word_end,
            phon_smin=phon_smin,
            phon_smax=phon_smax,
            dec_smin=dec_smin,
            dec_smax=dec_smax,
        )
    )

# %%
dec_results_df = pd.merge(
    pd.concat(dec_results),
    epp,
    on=["subject", "electrode_idx", "phoneme_pair"]
)

# %%
dec_results_df["test_vs"] = dec_results_df.test_roc_auc - dec_results_df.ho_test_roc_auc
dec_results_df["lexical_evidence"] = 1 - (dec_results_df.word_end.str[0] == dec_results_df.phoneme_pair.str[0]).astype(int)

# %%
# Expectation: test_vs > 0 because the acoustic contrast is specific to one word end
sns.displot(data=dec_results_df.query("significant_uncorrected").groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"]).test_vs.mean(),
            kind="kde", fill=True)
plt.axvline(x=0, color="k", ls="--", lw=1)

# %%
# No strong expectation here. Sites might show an acoustic context effect at both
# word ends (that's what this plot looks for) -- but we expect it won't generalize
# (not what this plot looks for; that's tested above).
sns.scatterplot(
    data=dec_results_df.groupby(["subject", "phoneme_pair", "electrode_idx", "lexical_evidence"]).test_roc_auc.mean().unstack(),
    x=0, y=1
)
plt.axvline(x=0.5, color="k", ls="--", lw=1)
plt.axhline(y=0.5, color="k", ls="--", lw=1)
plt.xlabel("AUC decoding: lexical evidence 0")
plt.ylabel("AUC decoding: lexical evidence 1")
