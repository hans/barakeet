# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
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
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from src.data import add_metadata_features
from src.models.causal6 import run_acoustic_searchlight, _resolve_target, _has_enough_per_class
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS, WORD_PHASE_DF

# %% tags=["parameters"]
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
epp_path = "outputs/causal46_joined/early_perceptual_projection/all_sites.csv"
reg_lambda_winners_path = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json"
outdir = "outputs/causal46_joined/acoustic_late"

epoch_tmin = -0.4
epoch_sfreq = 100

# smin selection for the late acoustic decoder window:
#   "trough" (new)  -> start past the per-site trough of the mean activation trace
#                      (decay->rise turning point of the early acoustic response),
#                      mirroring the trough gate in plot_for_paper.
#   "pod"    (old)  -> the previous heuristic, min(phon_smax + 20, POD sample).
# Flip to "pod" to reproduce the pre-trough behavior exactly.
smin_mode = "trough"

n_folds = 5
cv_random_state = 42
device = "cuda:1"
tol = 1e-6
max_iter = 50
null_max_iter = 15

# Permutation-null significance (single test per site; no searchlight).
n_permutations = 5000
permutation_seed = 0
permutation_chunk_size = 5000

# %%
Path(outdir).mkdir(parents=True, exist_ok=True)

# %%
b4_per_cell = pd.read_parquet(b4_per_cell_path)

# %%
epp = pd.read_csv(epp_path)
epp["significant"] = epp.q_one_tailed < 0.05
epp["significant_uncorrected"] = epp.p_one_tailed < 0.05

# %%
reg_lambda = json.loads(Path(reg_lambda_winners_path).read_text())["reg_lambda_acoustic"]
reg_lambda *= 1000 # DEV

# %%
epochs_dict = {}
for p in Path("outputs/epochs_preprocessed").glob("*.fif"):
    ep = mne.read_epochs(p, verbose=False)
    ep.metadata = add_metadata_features(ep.metadata)
    epochs_dict[p.stem.rstrip("_epo")] = ep

# %%
pod_df = WORD_PHASE_DF.query("phase == 'pod'").rename(columns={'start': 'pod'}).drop(columns=["end", "phase"]).set_index("word").pod
pod_df

# %%
to_study = pd.merge(
    (
        epp
        [epp.early_response_class.isin(("acoustic_only", "type2_aligned"))]
        [["subject", "electrode_idx", "phoneme_pair", "early_response_class"]]
    ),
    b4_per_cell,
    how="left", on=["subject", "electrode_idx", "phoneme_pair"]
)

# %%
to_study.early_response_class.value_counts()

# %%
# Per-site trough of the mean activation trace. The acoustic decoder's late
# window starts *past* this trough — the decay->rise turning point of the early
# acoustic response — mirroring the trough gate in plot_for_paper
# (find_early_peak_and_trough). The trace is the mean HGA over the decoder's own
# training trials (target word-end, unambiguous steps 1 & 6), pooled over both
# acoustic classes and hence label-blind, so anchoring the window edge on it
# introduces no acoustic-contrast leakage. Detection is invariant to a constant
# vertical shift (diff/prominence/diff-MAD), so raw get_data (no baseline) gives
# the same trough as the baseline-corrected trace used upstream.
trough_window_size = 5   # must match acoustic_bootstrap window_size
trough_stride = 5        # must match acoustic_bootstrap stride
SAMPLE_T0 = int(round((0.0 - epoch_tmin) * epoch_sfreq))  # t=0 (word onset)
WORD_END_TAIL_SAMPLES = 20  # +200 ms tail past word offset (search ceiling)
SMOOTH_WINDOWS = 3       # boxcar width on the pooled trace, in windows
SUSTAIN = 2              # consecutive positive-derivative samples to accept a crossing
PEAK_MIN_PROM_SD = 2.0   # first peak must exceed this many noise_sd over the trace min

# Search ceiling: word offset + 200 ms per word_end (`_WE_SMAX`), then pooled to
# a shared per-pair ceiling (`PAIR_SMAX`, the max across the pair's word_ends).
# Mirrors acoustic_bootstrap: a_per_window_by_word_end — the trace plot_for_paper's
# trough gate runs on — uses PAIR_SMAX[pp], so both word_ends span the same grid.
_WE_SMAX = {
    we: int(round((OFFSET_DICT[we] - epoch_tmin) * epoch_sfreq)) + WORD_END_TAIL_SAMPLES
    for we in OFFSET_DICT
}
PAIR_SMAX = {
    pp: max(_WE_SMAX[we] for we in wes)
    for pp, wes in PHONEME_PAIR_TO_WORD_ENDS.items()
}


def _smooth(y, w):
    if w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, w // 2, mode="edge"), k, mode="valid")[:len(y)]


def _diff_mad(y):
    d = np.diff(y)
    if not len(d):
        return np.nan
    return 1.4826 * np.median(np.abs(d - np.median(d)))


def find_trough_sample(mean_trace, *, sample_t0=SAMPLE_T0, search_smax=None,
                       win_size=trough_window_size, stride=trough_stride,
                       smooth=SMOOTH_WINDOWS, sustain=SUSTAIN,
                       min_prom=PEAK_MIN_PROM_SD):
    """First evoked peak of the windowed mean-activation trace, then the following
    sustained decay->rise crossing (the trough). Returns the trough's sample index
    (its window smin) or None if no peak/crossing is found. Ported verbatim from
    `find_early_peak_and_trough` in plot_for_paper: windowed pooled trace (5-sample
    non-overlapping bins from t=0), boxcar-smoothed, first prominent peak, then the
    first derivative zero-crossing sustained for `sustain` windows.

    `search_smax` caps the trace at that sample (the caller passes the shared
    per-pair ceiling `PAIR_SMAX[pp]` = max word offset + 200 ms across the pair,
    matching the trace the original ran on); windows are included while
    `smin + win_size <= search_smax`. None searches to the end of the trace."""
    hi = len(mean_trace) if search_smax is None else min(int(search_smax), len(mean_trace))
    starts = np.arange(sample_t0, hi - win_size + 1, stride)
    if len(starts) < 3:
        return None
    pooled = np.array([mean_trace[s:s + win_size].mean() for s in starts])

    y = _smooth(pooled, smooth)
    sd = _diff_mad(pooled)
    ok_sd = np.isfinite(sd) and sd > 0
    d = np.diff(y)
    prom = (y - y.min()) / sd if ok_sd else np.full_like(y, np.inf)

    i_peak = next((i for i in range(1, len(y) - 1)
                   if d[i - 1] > 0 >= d[i] and prom[i] >= min_prom), None)
    if i_peak is None:
        return None
    i_trough = None
    for i in range(i_peak + 1, len(d) - sustain + 1):
        if d[i] > 0 and all(d[i + j] > 0 for j in range(sustain)):
            i_trough = i
            break
    if i_trough is None:
        return None
    return int(starts[i_trough])


def _site_trough(subject, electrode_idx, phoneme_pair, word_end):
    """Trough sample for one site, computed on the same trial selection the
    acoustic decoder trains on (target word-end, unambiguous steps 1 & 6)."""
    md = epochs_dict[subject].metadata
    sel = ((md.phoneme_pair == phoneme_pair) & (md.word_end == word_end)
           & md.resampled.isin((1, 6))).values
    if sel.sum() == 0:
        return None
    data = epochs_dict[subject].get_data(picks=[electrode_idx])  # (n_trials, 1, n_samples)
    mean_trace = data[sel, 0, :].mean(axis=0)
    return find_trough_sample(mean_trace, search_smax=PAIR_SMAX[phoneme_pair])


# Precompute once per (site × word-end) and carry as a column, so the real
# decoder loop and both permutation-null loops read the *same* window edge
# (identical windows are required for the null p-values to be valid). Under
# smin_mode == "pod" the column is all-NaN, so prepare_decoder_bounds falls
# through to the old POD/phon_smax heuristic for every site.
if smin_mode == "trough":
    to_study["s_trough"] = [
        _site_trough(r.subject, r.electrode_idx, r.phoneme_pair, r.word_end)
        for r in to_study.itertuples()
    ]
    n_no_trough = int(to_study["s_trough"].isna().sum())
    print(f"no trough detected: {n_no_trough} / {len(to_study)} sites "
          "(these fall back to the old POD/phon_smax smin)")
else:
    to_study["s_trough"] = np.nan
    print(f"smin_mode={smin_mode!r}: using the old POD/phon_smax smin for all sites")

# %%
import polars as pl
from sklearn.model_selection import StratifiedKFold
import torch

from src.models.decoding_gpu import batched_roc_auc, compute_balanced_sample_weight, fit_batched_l2_logreg, fit_batched_l2_logreg_perms, standardise_per_batch


def _pooled_oof_auc(oof_proba, labels):
    """Pooled out-of-fold ROC-AUC over ALL held-out predictions jointly, rather
    than averaging per-fold AUCs. `oof_proba` (M, n) holds each trial's single
    held-out probability (one per trial, filled exactly once across folds);
    `labels` (M, n) the matching labels. Returns (M,) AUC. M = B (real) or
    K*B (null). Real and null both route through this same `batched_roc_auc`
    call so the observed and null statistics are computed identically.

    On ~10-trial CV folds, averaging five per-fold AUCs is dominated by
    per-fold sampling noise; pooling the ~48 OOF predictions into one AUC
    roughly halves the per-site SD. Used for the target test only — the
    transfer test already scores the full alternate set every fold.
    """
    return batched_roc_auc(oof_proba, labels)


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

    # Out-of-fold target probabilities: each trial filled once by its test fold,
    # then pooled into a single AUC per problem (see _pooled_oof_auc). NaN-init
    # so a leftover NaN would flag an unfilled trial (indexing bug).
    oof_proba = torch.full((B, n_trials), float("nan"), dtype=dtype, device=device)

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
        oof_proba.index_copy_(1, test_idx_t, proba_te_gpu)  # stash this fold's OOF probas

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

    # Pooled out-of-fold target AUC per problem (one number over all OOF trials),
    # broadcast across the per-fold rows via the join below.
    assert not torch.isnan(oof_proba).any(), "OOF proba has unfilled trials"
    pooled_auc = _pooled_oof_auc(
        oof_proba, y_gpu.unsqueeze(0).expand(B, -1)
    ).cpu().numpy()
    pooled_pl = pl.DataFrame({
        "_problem_idx": problem_idx,
        "test_roc_auc_pooled": pooled_auc,
    })

    # Concat across folds, join problem_meta back in to replace _problem_idx with actual keys
    scores = pl.concat(scores_frames).join(
        pooled_pl, on="_problem_idx", how="left"
    ).join(
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
def prepare_decoder_bounds(phoneme_pair, word_end, phon_smax, s_trough=None):
    other_word_end = next(iter(set(PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]) - {word_end}))

    # smax unchanged: word offset + 100ms, whichever word-end runs later.
    dec_smax = max(
        int(round((OFFSET_DICT[word_end] + 0.1 - epoch_tmin) * epoch_sfreq)),
        int(round((OFFSET_DICT[other_word_end] + 0.1 - epoch_tmin) * epoch_sfreq))
    )

    # smin is conditioned on the per-site trough of the mean activation trace:
    # only look past the trough (the decay->rise turning point of the early
    # acoustic response). Falls back to the old POD/phon_smax heuristic only
    # where no trough was detected (or under smin_mode == "pod").
    if s_trough is not None and np.isfinite(s_trough):
        dec_smin = int(s_trough)
    else:
        dec_smin = int(round(min(phon_smax + 20, (pod_df.loc[word_end] - epoch_tmin) * epoch_sfreq)))

    assert dec_smin < dec_smax, f"empty window: dec_smin={dec_smin} >= dec_smax={dec_smax}"
    return dec_smin, dec_smax


# %%
dec_results = []
dec_coefs = []

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

    dec_smin, dec_smax = prepare_decoder_bounds(phoneme_pair, word_end, phon_smax, row["s_trough"])
    ep_i = epochs_dict[subject]

    scores_i, _, coefs_i = run_acoustic_cross_decoder(
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
        scores_i
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

    dec_coefs.append(
        coefs_i
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
dec_results_df = pd.merge(
    dec_results_df,
    to_study[["subject", "electrode_idx", "phoneme_pair", "word_end", "s_trough"]],
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"]
)

dec_results_df["test_vs"] = dec_results_df.test_roc_auc - dec_results_df.ho_test_roc_auc
dec_results_df["lexical_evidence"] = 1 - (dec_results_df.word_end.str[0] == dec_results_df.phoneme_pair.str[0]).astype(int)
dec_results_df.to_csv(Path(outdir) / "acoustic_late_results.csv", index=False)

# %%
dec_coefs_df = pd.concat(dec_coefs)
dec_coefs_df.to_parquet(Path(outdir) / "acoustic_late_coefs.parquet", index=False)

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

# %% [markdown]
# ## Permutation-null significance test (one test per site)
#
# For each site we refit the acoustic decoder under `n_permutations` shuffles
# of the acoustic label, using the *same* trial selection (target word-end,
# unambiguous steps 1 & 6) and the *same* late window `[phon_smax, dec_smax]`
# as the real decoder above. Splits come from the real (unshuffled) labels, so
# fold sizes match the real run. We null **`test_roc_auc`** — i.e. whether the
# within-word-end acoustic decoder beats chance in the late window. (The
# headline plots key on `test_vs`; testing that difference would need a
# different null. Switch here if that's what's wanted.)

# %%
def _fit_batched_cv_null(
    X: np.ndarray,               # (n_trials, B, d)
    y: np.ndarray,                # (n_trials,) real labels (used for splits only)
    problem_meta: pl.DataFrame,   # (B rows)
    *,
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    reg_lambda: float,
    n_folds: int,
    cv_random_state: int,
    device: str,
    dtype: torch.dtype,
    tol: float,
    max_iter: int,
    X_ho_test: np.ndarray | None = None,   # (n_ho, B, d) — alternate word-end trials
    y_ho_test: np.ndarray | None = None,    # (n_ho,) TRUE (unpermuted) labels
) -> pl.DataFrame:
    """
    Batched CV fit under label permutations — refit-based null.

    Copied and stripped down from `src.models.causal6._fit_batched_cv_permutations`
    (dropped spill/pbar): for each of B problems and K = len(permute_seeds)
    permutations, fit a fresh L2 LogReg on (X, shuffled y) using StratifiedKFold
    splits taken from the *real* labels. Returns test ROC-AUC per
    (problem × fold × permutation).

    If `X_ho_test`/`y_ho_test` are provided (for the `test_vs` null), the same
    permuted-label refit is *also* scored against the alternate word-end trials
    using their TRUE labels, adding a `ho_test_roc_auc` column. Only the training
    labels are permuted; both test sets are evaluated against true labels, so
    `test_roc_auc - ho_test_roc_auc` under this null is centered at ~0.
    """
    compute_ho = X_ho_test is not None
    n_trials, B, _d = X.shape
    K = len(permute_seeds)
    assert y.shape == (n_trials,)
    assert problem_meta.height == B
    if K == 0:
        return pl.DataFrame()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)

    # Pre-shuffle labels deterministically per permutation.
    y_perms = np.empty((K, n_trials), dtype=np.int64)
    y_int = y.astype(np.int64)
    for k, seed in enumerate(permute_seeds):
        y_perms[k] = np.random.default_rng(int(seed)).permutation(y_int)

    X_gpu = torch.tensor(
        X.transpose(1, 0, 2).copy(), dtype=dtype, device=device
    )  # (B, n_trials, d)
    y_perms_gpu = torch.tensor(y_perms.astype(np.float64), dtype=dtype, device=device)

    if compute_ho:
        assert y_ho_test is not None
        n_ho = X_ho_test.shape[0]
        X_ho_gpu = torch.tensor(
            X_ho_test.transpose(1, 0, 2).copy(), dtype=dtype, device=device
        )  # (B, n_ho, d)
        y_ho_gpu = torch.tensor(y_ho_test.astype(np.float64), dtype=dtype, device=device)

    # Out-of-fold target probabilities per permutation, filled once per trial
    # across folds, then pooled into one AUC per (permutation × problem).
    oof_proba = torch.full((K, B, n_trials), float("nan"), dtype=dtype, device=device)

    scores_frames: list[pl.DataFrame] = []
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
        mask_tr_b = torch.ones(B, n_tr, dtype=dtype, device=device)

        # Standardisation is label-independent — compute once per fold.
        X_tr_std_b, X_te_std_b, _, _ = standardise_per_batch(
            X_train_t, mask_tr_b, X_test_t
        )
        if compute_ho:
            # Standardise the alternate word-end trials with this fold's train stats.
            X_ho_std_b = standardise_per_batch(X_train_t, mask_tr_b, X_ho_gpu)[1]  # (B, n_ho, d)

        for chunk_start in range(0, K, permutation_chunk_size):
            chunk_end = min(chunk_start + permutation_chunk_size, K)
            Kc = chunk_end - chunk_start

            y_perm_chunk = y_perms_gpu[chunk_start:chunk_end]                # (Kc, n_trials)
            y_train_kn = y_perm_chunk.index_select(1, train_idx_t)           # (Kc, n_tr)
            y_test_kn = y_perm_chunk.index_select(1, test_idx_t)             # (Kc, n_te)
            y_test_kbn = y_test_kn.unsqueeze(1).expand(Kc, B, n_te)          # broadcast view

            sw_train_kn = compute_balanced_sample_weight(
                y_train_kn,
                torch.ones(Kc, n_tr, dtype=dtype, device=device),
            )                                                                # (Kc, n_tr)
            y_train_kbn = y_train_kn.unsqueeze(1).expand(Kc, B, n_tr)        # broadcast view
            sw_train_kbn = sw_train_kn.unsqueeze(1).expand(Kc, B, n_tr)      # broadcast view

            beta, _, _ = fit_batched_l2_logreg_perms(
                X_tr_std_b, y_train_kbn, mask_tr_b, sw_train_kbn,
                reg_lambda=reg_lambda, tol=tol, max_iter=max_iter,
            )                                                                # (Kc, B, d)

            z_te = torch.einsum("bnd,kbd->kbn", X_te_std_b, beta)            # (Kc, B, n_te)
            proba_te = torch.sigmoid(z_te)
            # Stash OOF probas for this permutation-chunk × fold into the pool.
            oof_proba[chunk_start:chunk_end].index_copy_(2, test_idx_t, proba_te)
            aucs = batched_roc_auc(
                proba_te.reshape(Kc * B, n_te),
                y_test_kbn.reshape(Kc * B, n_te),
            ).cpu().numpy()

            perm_ids = np.repeat(np.arange(chunk_start, chunk_end, dtype=np.int64), B)
            prob_ids = np.tile(problem_idx, Kc)
            chunk_cols = {
                "_problem_idx": prob_ids,
                "fold": np.full(Kc * B, fold, dtype=np.int32),
                "permutation_idx": perm_ids,
                "test_roc_auc": aucs,
            }
            if compute_ho:
                # Score the same permuted-fit decoder on the alternate word-end
                # trials against their TRUE labels (labels not permuted here).
                z_ho = torch.einsum("bnd,kbd->kbn", X_ho_std_b, beta)        # (Kc, B, n_ho)
                proba_ho = torch.sigmoid(z_ho)
                y_ho_kbn = y_ho_gpu.view(1, 1, n_ho).expand(Kc, B, n_ho)     # broadcast view
                chunk_cols["ho_test_roc_auc"] = batched_roc_auc(
                    proba_ho.reshape(Kc * B, n_ho),
                    y_ho_kbn.reshape(Kc * B, n_ho),
                ).cpu().numpy()
            scores_frames.append(pl.DataFrame(chunk_cols))

    # Pooled OOF target AUC per (permutation × problem), scored against the
    # PERMUTED labels (for the target null the label IS what's permuted, so the
    # statistic is the pooled CV score of the shuffled-label refit end-to-end).
    assert not torch.isnan(oof_proba).any(), "OOF proba has unfilled trials"
    y_perms_kbn = y_perms_gpu.unsqueeze(1).expand(K, B, n_trials)  # (K, B, n_trials)
    pooled = _pooled_oof_auc(
        oof_proba.reshape(K * B, n_trials),
        y_perms_kbn.reshape(K * B, n_trials),
    ).cpu().numpy()
    pooled_df = pl.DataFrame({
        "_problem_idx": np.tile(problem_idx, K),
        "permutation_idx": np.repeat(np.arange(K, dtype=np.int64), B),
        "test_roc_auc_pooled": pooled,
    })

    return pl.concat(scores_frames).join(
        pooled_df, on=["_problem_idx", "permutation_idx"], how="left"
    ).join(
        problem_meta_with_idx, on="_problem_idx", how="left"
    ).drop("_problem_idx")


# %%
def run_acoustic_cross_decoder_null(
    epochs: mne.Epochs,
    subject: str,
    phoneme_pair: str,
    word_end: str,
    electrode_idx: int,
    window: tuple[int, int],
    *,
    reg_lambda: float,
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    target: Literal["categorical_acoustic_cue", "subject_specific_acoustics"]
        = "categorical_acoustic_cue",
    resampled_steps: tuple[int, ...] = (1, 6),
    compute_ho: bool = False,
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
):
    """Permutation null for a single site, mirroring `run_acoustic_cross_decoder`
    selection (target word-end + unambiguous steps) and the same window, but
    refitting under shuffled acoustic labels. Coef/pred outputs are dropped.

    With `compute_ho=True` the alternate word-end trials are also scored under
    each permuted-label refit (adds a `ho_test_roc_auc` column), so the caller
    can build the `test_vs = test_roc_auc - ho_test_roc_auc` null."""
    assert epochs.metadata is not None
    md = epochs.metadata

    target_mask = (md.word_end == word_end).values
    resampled_mask = md.resampled.isin(resampled_steps).values
    target_selection = (md.phoneme_pair == phoneme_pair).values & target_mask & resampled_mask
    if target_selection.sum() == 0:
        return None

    y = _resolve_target(md, target, phoneme_pair, target_selection)
    if not _has_enough_per_class(y, n_folds):
        return None

    data = epochs.get_data(picks=[electrode_idx])  # (n_trials, 1, n_samples)
    smin, smax = window
    X_sel = data[target_selection][:, 0, smin:smax]           # (n_trials, win)
    X_batch = X_sel[:, None, :].astype(np.float64)            # (n_trials, B=1, win)

    X_ho_batch = y_ho = None
    if compute_ho:
        alternate_word_end = next(iter(set(PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]) - {word_end}))
        alternate_selection = (
            (md.phoneme_pair == phoneme_pair).values
            & (md.word_end == alternate_word_end).values
            & resampled_mask
        )
        X_ho_batch = data[alternate_selection][:, 0, smin:smax][:, None, :].astype(np.float64)
        y_ho = _resolve_target(md, target, phoneme_pair, alternate_selection)

    problem_meta = pl.DataFrame({
        "subject": [subject],
        "phoneme_pair": [phoneme_pair],
        "electrode_idx": [int(electrode_idx)],
        "smin": [int(smin)],
        "smax": [int(smax)],
    })

    return _fit_batched_cv_null(
        X_batch, y, problem_meta,
        permute_seeds=permute_seeds,
        permutation_chunk_size=permutation_chunk_size,
        reg_lambda=reg_lambda,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
        X_ho_test=X_ho_batch, y_ho_test=y_ho,
    ).with_columns(pl.lit(target).alias("target"))


# %%
permute_seeds = list(range(permutation_seed, permutation_seed + n_permutations))

null_results = []
for _, row in tqdm(to_study.iterrows(), total=len(to_study)):
    subject = row["subject"]
    electrode_idx = row["electrode_idx"]
    phoneme_pair = row["phoneme_pair"]
    word_end = row["word_end"]

    phon_smax = row["phon_smax"]
    dec_smin, dec_smax = prepare_decoder_bounds(phoneme_pair, word_end, phon_smax, row["s_trough"])

    null_scores = run_acoustic_cross_decoder_null(
        epochs_dict[subject],
        subject=subject,
        electrode_idx=electrode_idx,
        phoneme_pair=phoneme_pair,
        word_end=word_end,
        window=(dec_smin, dec_smax),
        target="categorical_acoustic_cue",
        compute_ho=True,
        reg_lambda=reg_lambda,
        permute_seeds=permute_seeds,
        permutation_chunk_size=permutation_chunk_size,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
        device=device,
        tol=tol,
        max_iter=null_max_iter,
    )
    if null_scores is None:
        continue

    null_results.append(null_scores.to_pandas().assign(word_end=word_end))

# %%
# Null statistic per (site × permutation): mean test_roc_auc over folds
# (nanmean — permuted labels can leave a test fold single-class -> NaN AUC),
# matching how the real per-site stat is aggregated below.
null_df = pd.concat(null_results, ignore_index=True)
null_df.to_parquet(Path(outdir) / "acoustic_late_null.parquet")
site_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
# Target test uses the pooled OOF AUC (constant across a permutation's fold rows
# -> `first`); transfer test stays mean-over-folds (full alternate set per fold).
null_per_perm = (
    null_df.groupby(site_keys + ["permutation_idx"])
    .agg(test_roc_auc=("test_roc_auc_pooled", "first"),
         ho_test_roc_auc=("ho_test_roc_auc", "mean"))
    .reset_index()
)

# %%
# Observed statistic: reuse the real per-site mean test_roc_auc from the loop above.
real_per_site = (
    dec_results_df.groupby(site_keys).test_roc_auc.mean().rename("real_test_roc_auc")
)

# One-sided permutation p-value: P(null >= observed), with +1 smoothing.
def _perm_pvalue(g, real_df, col="test_roc_auc", alpha=0.05):
    key = g.name
    obs = real_df.loc[key]
    null_vals = g[col].to_numpy()
    K = len(null_vals)
    p_value = (1 + np.sum(null_vals >= obs)) / (1 + K)
    return pd.Series({
        f"real_{col}": obs,
        "null_mean": np.nanmean(null_vals),
        "n_permutations": K,
        "p_value": p_value,
        "significant": p_value < alpha
    })

# Target test compares the pooled OOF AUC (constant across folds); transfer test
# uses the mean over per-fold AUCs. Both matched to their respective null above.
real_target = dec_results_df.groupby(site_keys).test_roc_auc_pooled.mean()
real_transfer = dec_results_df.groupby(site_keys).ho_test_roc_auc.mean()

sig_target = null_per_perm.groupby(site_keys).apply(_perm_pvalue, real_df=real_target, col="test_roc_auc").reset_index()
sig_transfer = null_per_perm.groupby(site_keys).apply(_perm_pvalue, real_df=real_transfer, col="ho_test_roc_auc").reset_index()

sig_df = pd.merge(
    sig_target[site_keys + ["real_test_roc_auc", "p_value", "null_mean", "n_permutations", "significant"]],
    sig_transfer[site_keys + ["real_ho_test_roc_auc", "p_value", "null_mean", "n_permutations", "significant"]],
    on=site_keys,
    suffixes=("_target", "_transfer")
)

sig_df = pd.merge(
    sig_df,
    to_study[["subject", "electrode_idx", "phoneme_pair", "word_end", "s_trough"]],
    on=site_keys,
    how="left"
)

# only retain examples that have a trough in the mean trace
sig_df = sig_df[~sig_df.s_trough.isna()]

from statsmodels.stats.multitest import multipletests
# FDR-correct both tests separately.
for test in ["target", "transfer"]:
    pvals = sig_df[f"p_value_{test}"].to_numpy()
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    sig_df[f"p_value_{test}_fdr"] = pvals_corrected
    sig_df[f"significant_{test}_fdr"] = reject

sig_df.to_csv(Path(outdir) / "acoustic_late_summary.csv", index=False)

sig_df.sort_values("p_value_target").head(20)

# %%
# Observed AUC vs. permutation-null mean, coloured by significance.
sns.scatterplot(data=sig_df, x="null_mean_target", y="real_test_roc_auc", hue="significant_target")
plt.axline((0.5, 0.5), slope=1, color="k", ls="--", lw=1)
plt.xlabel("null mean AUC")
plt.ylabel("observed AUC")

# %%
# Observed AUC vs. permutation-null mean, coloured by significance.
sns.scatterplot(data=sig_df.query("significant_target"),
                x="real_ho_test_roc_auc", y="real_test_roc_auc")
plt.axline((0.5, 0.5), slope=1, color="k", ls="--", lw=1)
plt.axvline(x=0.5, color="k", ls="--", lw=1)
plt.xlabel("transfer AUC")
plt.ylabel("target AUC")

from scipy import stats
ttest_df = sig_df.query("significant_target")
rvalue, pvalue = stats.pearsonr(ttest_df.real_ho_test_roc_auc, ttest_df.real_test_roc_auc)
print(f"Pearson r = {rvalue:.3f}, p = {pvalue:.3g} for {len(ttest_df)} significant sites")

# %%
g = sns.jointplot(data=sig_df.query("significant_target"),
                  x="real_test_roc_auc", y="real_ho_test_roc_auc",
                  xlim=(0.5, 1), ylim=(0.3, 1),
                  kind="reg",
                  height=3)

# %%
sig_df.sort_values("p_value_target_fdr")

# %%
sig_df.significant_target.sum(), sig_df.significant_target_fdr.sum()

# %%
sig_df.significant_transfer.sum(), sig_df.significant_transfer_fdr.sum()

# %% [markdown]
# ## Examine coefs for sig sites

# %%
coef_study_df = pd.merge(
    sig_df.query("significant_target_fdr"),
    dec_coefs_df.drop(columns=["mean", "scale"]),
    on=site_keys
)

# %%
xs = coef_study_df.explode("coef")
xs["coef2"] = xs.coef ** 2
xs["coef2_of_max"] = xs.groupby(["subject", "phoneme_pair", "electrode_idx", "word_end", "fold"])["coef2"].transform(lambda xs: xs / xs.max())
xs["coef_idx"] = xs.groupby(level=0).cumcount()
xs["dec_sample"] = xs.dec_smin + xs.coef_idx
xs["dec_t"] = xs.dec_sample / epoch_sfreq + epoch_tmin

# merge in PoD information
xs = pd.merge(
    xs,
    WORD_PHASE_DF.query("phase == 'pod'").rename(columns={"start": "pod"})[["word", "pod"]],
    left_on=["word_end"], right_on=["word"],
    how="left"
)
xs["dec_t_from_pod"] = xs.dec_t - xs.pod
xs["dec_t_from_pod_bin"] = pd.cut(xs.dec_t_from_pod, bins=np.arange(-0.1, 0.7, 0.15))

# %%
sns.lineplot(data=xs,#.query("subject == 'EC243' and electrode_idx == 102 and word_end == 'desolate'"),
             x="dec_t", y="coef2_of_max", hue="phoneme_pair")#, hue="fold")

# %%
sns.lineplot(data=xs,#.query("subject == 'EC243' and electrode_idx == 102 and word_end == 'desolate'"),
             x="dec_t_from_pod", y="coef2_of_max", hue="phoneme_pair")#, hue="fold")

# %%
sns.catplot(data=xs,#.query("subject == 'EC243' and electrode_idx == 102 and word_end == 'desolate'"),
             x="dec_t_from_pod_bin", y="coef2_of_max", hue="phoneme_pair", kind="point")#, hue="fold")
