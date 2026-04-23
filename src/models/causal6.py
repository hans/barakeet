"""
causal6 decoder entry points.

Three decoders, one GPU kernel, one outer CV strategy (StratifiedKFold).
Clean parquet-only outputs — no fitted-estimator joblib blobs.

Functions:
    run_acoustic_searchlight(...)
    run_behavior_with_control(...)
    run_behavior_hga_only(...)

Each returns three polars DataFrames:
    - scores: per-fold test AUC + metadata per decoder key
    - predictions: per-trial held-out predictions (one row per epoch_idx per decoder key)
    - coefficients: fitted β per (decoder key, fold), with mean/scale standardisation stats

Callers write these to parquet.

Design invariants:
    - StratifiedKFold(n_splits=5) outer CV, same splits reused across models in a batch
    - Per-training-fold StandardScaler (stats from train only, applied to test)
    - Balanced sample weights (matches sklearn class_weight='balanced')
    - No intercept (matches causal5 convention)
    - One GPU batched call per (subject, phoneme_pair[, word_end group]); each call
      contains all (electrode × window × fold) problems for that grouping.

See notebooks/causal5/behavior_decoding_single_electrode.py and
src/models/decoding.py for the causal5 reference implementations that this
replaces.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import mne
import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger as L
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from src.models.decoding_gpu import (
    batched_roc_auc,
    compute_balanced_sample_weight,
    fit_batched_l2_logreg,
    standardise_per_batch,
)


# ---------------------------------------------------------------------------
# Window enumeration (shared across decoders)
# ---------------------------------------------------------------------------


def make_windows(
    global_min_sample: int,
    global_max_sample: int,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """(N, 2) array of (smin, smax) pairs, inclusive of windows fully within bounds."""
    left = np.arange(global_min_sample, global_max_sample, stride)
    right = left + window_size
    return np.stack([left, right], axis=1)[right <= global_max_sample]


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def _has_enough_per_class(y: np.ndarray, n_folds: int) -> bool:
    """
    StratifiedKFold(n_splits=n_folds) requires at least `n_folds` samples in
    the minority class. Also catches the degenerate single-class case.
    """
    if len(np.unique(y)) != 2:
        return False
    return int(np.bincount(y.astype(np.int64)).min()) >= n_folds


def _resolve_target(
    metadata: pd.DataFrame,
    target: str,
    phoneme_pair: str,
    selection: np.ndarray,
) -> np.ndarray:
    """
    Map metadata + target name to a binary {0, 1} label vector over `selection`.

    Mirrors the target-resolution logic in
    src/models/decoding.py:_prepare_decoding_population.
    """
    md = metadata
    if target == "categorical_acoustic_cue":
        # {-1, +1} in metadata → {0, 1}
        y = (md.categorical_acoustic_cue[selection].values > 0).astype(np.int64)
    elif target == "subject_specific_acoustics":
        y = (md.subject_specific_acoustics[selection].values > 0).astype(np.int64)
    elif target in ("behavior_categorical", "behavior_categorical_forced"):
        y = md.behavior_dummy_forced[selection].values.astype(np.int64)
    else:
        raise ValueError(f"Unsupported target: {target}")
    return y


# ---------------------------------------------------------------------------
# Core batched CV fit — returns long-format polars DataFrames
# ---------------------------------------------------------------------------


def _fit_batched_cv(
    X: np.ndarray,            # (n_trials, B, d) — B problems on the same trials
    y: np.ndarray,             # (n_trials,)      — labels
    epoch_idxs: np.ndarray,    # (n_trials,)      — original metadata index per trial
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

        # --- scores ---
        scores_frames.append(pl.DataFrame({
            "_problem_idx": problem_idx,
            "fold": np.full(B, fold, dtype=np.int32),
            "test_roc_auc": aucs,
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


# ---------------------------------------------------------------------------
# Acoustic searchlight
# ---------------------------------------------------------------------------


def run_acoustic_searchlight(
    epochs: mne.Epochs,
    subject: str,
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
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Acoustic single-electrode searchlight — causal6 replacement for
    src/models/decoding.py:run_decoding_searchlight_single_electrode.

    By convention (see CLAUDE.md), the acoustic response is measured on
    **unambiguous trials only** — stimulus steps where the acoustic cue is
    clean. Defaults to resampled_steps=(1, 6), the two endpoints of the
    continuum, matching causal5's filter at
    notebooks/causal5/acoustic_decoding_single_electrode.py:83.

    Pass `resampled_steps=tuple(range(1, 7))` to include all steps
    (e.g. for the `subject_specific_acoustics` target where the label is
    per-trial acoustics rather than stimulus-step identity).

    Returns (scores, predictions, coefficients) as polars DataFrames.
    """
    assert epochs.metadata is not None
    md = epochs.metadata
    phoneme_pairs = sorted(md.phoneme_pair.dropna().unique())

    X_full = epochs.get_data(picks=list(electrode_idxs))
    n_electrodes = len(electrode_idxs)
    n_windows = windows.shape[0]
    win_size = int(windows[0, 1] - windows[0, 0])
    assert (windows[:, 1] - windows[:, 0] == win_size).all()

    # Restrict to unambiguous stimulus steps by default (project convention).
    resampled_mask = md.resampled.isin(resampled_steps).values

    scores_all, preds_all, coefs_all = [], [], []

    pbar = tqdm(
        phoneme_pairs,
        desc=f"acoustic[{subject}] λ={reg_lambda:g} {target}",
        unit="pp", leave=False,
    )
    for phoneme_pair in pbar:
        selection = (md.phoneme_pair == phoneme_pair).values & resampled_mask
        if selection.sum() == 0:
            continue

        y = _resolve_target(md, target, phoneme_pair, selection)
        if not _has_enough_per_class(y, n_folds):
            counts = np.bincount(y.astype(np.int64)) if len(np.unique(y)) > 0 else []
            L.warning(
                f"[acoustic][{subject}/{phoneme_pair}] skipping: insufficient "
                f"class balance for StratifiedKFold(n_splits={n_folds}); "
                f"class counts = {list(counts)}"
            )
            continue

        epoch_idxs_sel = md.index[selection].to_numpy()
        X_sel = X_full[selection]  # (n_trials, n_electrodes, n_samples)
        n_trials = X_sel.shape[0]

        B = n_electrodes * n_windows
        X_batch = np.empty((n_trials, B, win_size), dtype=np.float64)
        elec_per = np.empty(B, dtype=np.int64)
        smin_per = np.empty(B, dtype=np.int64)
        smax_per = np.empty(B, dtype=np.int64)
        b = 0
        for e_idx, electrode_idx in enumerate(electrode_idxs):
            for smin, smax in windows:
                X_batch[:, b, :] = X_sel[:, e_idx, smin:smax]
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
            X_batch, y, epoch_idxs_sel, problem_meta,
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


# ---------------------------------------------------------------------------
# Behavior decoders (with & without control predictor)
# ---------------------------------------------------------------------------


def _run_behavior_core(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    with_control: bool,
    reg_lambda: float,
    reg_lambda_baseline: Optional[float],
    n_folds: int,
    cv_random_state: int,
    device: str,
    dtype: torch.dtype,
    tol: float,
    max_iter: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Shared core for behavior-with-control and behavior-HGA-only decoders.

    Groups by (phoneme_pair, word_end). For each group:
      - one batched call for all (electrode × window) full-model problems
      - (when with_control) one separate single-problem call for the baseline
        model — shared across all electrodes/windows in the group
    """
    assert epochs.metadata is not None
    md = epochs.metadata
    phoneme_pairs = sorted(md.phoneme_pair.dropna().unique())

    X_full = epochs.get_data(picks=list(electrode_idxs))
    n_electrodes = len(electrode_idxs)
    n_windows = windows.shape[0]
    win_size = int(windows[0, 1] - windows[0, 0])
    assert (windows[:, 1] - windows[:, 0] == win_size).all()

    full_scores, full_preds, full_coefs = [], [], []
    base_scores, base_preds, base_coefs = [], [], []

    # Enumerate (phoneme_pair, word_end) groups up front so tqdm has a total.
    groups: list[tuple[str, str]] = []
    for phoneme_pair in phoneme_pairs:
        pp_mask = (md.phoneme_pair == phoneme_pair).values
        if pp_mask.sum() == 0:
            continue
        for word_end in sorted(md.word_end[pp_mask].dropna().unique()):
            groups.append((phoneme_pair, word_end))

    decoder_label = "behavior_full" if with_control else "behavior_hga_only"
    pbar = tqdm(
        groups,
        desc=f"{decoder_label}[{subject}] λ={reg_lambda:g}",
        unit="group", leave=False,
    )
    for phoneme_pair, word_end in pbar:
        pp_mask = (md.phoneme_pair == phoneme_pair).values
        sel = pp_mask & (md.word_end == word_end).values
        y = _resolve_target(md, "behavior_categorical_forced", phoneme_pair, sel)
        if not _has_enough_per_class(y, n_folds):
            counts = np.bincount(y.astype(np.int64)) if len(np.unique(y)) > 0 else []
            L.warning(
                f"[{decoder_label}][{subject}/{phoneme_pair}/{word_end}] "
                f"skipping: insufficient class balance for "
                f"StratifiedKFold(n_splits={n_folds}); class counts = {list(counts)}"
            )
            continue

        epoch_idxs_sel = md.index[sel].to_numpy()
        X_sel = X_full[sel]
        n_trials = X_sel.shape[0]
        resampled_feat = md.resampled[sel].to_numpy().astype(np.float64).reshape(-1, 1)

        # Full-model batch
        full_d = (1 + win_size) if with_control else win_size
        B_full = n_electrodes * n_windows
        X_full_batch = np.empty((n_trials, B_full, full_d), dtype=np.float64)
        elec_per = np.empty(B_full, dtype=np.int64)
        smin_per = np.empty(B_full, dtype=np.int64)
        smax_per = np.empty(B_full, dtype=np.int64)
        b = 0
        for e_idx, electrode_idx in enumerate(electrode_idxs):
            for smin, smax in windows:
                hga = X_sel[:, e_idx, smin:smax]
                if with_control:
                    X_full_batch[:, b, 0:1] = resampled_feat
                    X_full_batch[:, b, 1:] = hga
                else:
                    X_full_batch[:, b, :] = hga
                elec_per[b] = int(electrode_idx)
                smin_per[b] = int(smin)
                smax_per[b] = int(smax)
                b += 1

        full_meta = pl.DataFrame({
            "subject": [subject] * B_full,
            "phoneme_pair": [phoneme_pair] * B_full,
            "word_end": [word_end] * B_full,
            "electrode_idx": elec_per,
            "smin": smin_per,
            "smax": smax_per,
            "model": ["full"] * B_full,
        })
        fs, fp, fc = _fit_batched_cv(
            X_full_batch, y, epoch_idxs_sel, full_meta,
            reg_lambda=reg_lambda,
            n_folds=n_folds, cv_random_state=cv_random_state,
            device=device, dtype=dtype, tol=tol, max_iter=max_iter,
        )
        full_scores.append(fs)
        full_preds.append(fp)
        full_coefs.append(fc)

        # Baseline model: one problem (resampled only), shared across electrodes/windows
        if with_control:
            X_base = resampled_feat.reshape(n_trials, 1, 1)
            base_meta = pl.DataFrame({
                "subject": [subject],
                "phoneme_pair": [phoneme_pair],
                "word_end": [word_end],
                "electrode_idx": [-1],     # sentinel: not electrode-specific
                "smin": [-1], "smax": [-1],
                "model": ["baseline"],
            })
            bs, bp, bc = _fit_batched_cv(
                X_base, y, epoch_idxs_sel, base_meta,
                reg_lambda=(reg_lambda_baseline if reg_lambda_baseline is not None
                            else reg_lambda),
                n_folds=n_folds, cv_random_state=cv_random_state,
                device=device, dtype=dtype, tol=tol, max_iter=max_iter,
            )
            base_scores.append(bs)
            base_preds.append(bp)
            base_coefs.append(bc)

    # Combine full + baseline
    scores_parts = full_scores + base_scores
    preds_parts = full_preds + base_preds
    coefs_parts = full_coefs + base_coefs
    if not scores_parts:
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    scores = pl.concat(scores_parts)
    preds = pl.concat(preds_parts).with_columns(
        (pl.col("decoder_proba") > 0.5).cast(pl.Int8).alias("decoder_prediction")
    )
    coefs = pl.concat(coefs_parts)
    return scores, preds, coefs


def run_behavior_with_control(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    reg_lambda: float,
    reg_lambda_baseline: Optional[float] = None,
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Behavior decoding with `resampled` control predictor.

    Full model features: [resampled, HGA window]. Baseline features: [resampled].
    Baseline is fit once per (phoneme_pair, word_end, fold) and reused across electrodes
    via the `model` key in the output DataFrames.
    """
    return _run_behavior_core(
        epochs, subject, electrode_idxs, windows,
        with_control=True,
        reg_lambda=reg_lambda, reg_lambda_baseline=reg_lambda_baseline,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )


def run_behavior_hga_only(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    reg_lambda: float,
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Behavior decoding without control predictor: single model on HGA window only.
    Outputs have `model='full'` only (no baseline rows).
    """
    return _run_behavior_core(
        epochs, subject, electrode_idxs, windows,
        with_control=False,
        reg_lambda=reg_lambda, reg_lambda_baseline=None,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )
