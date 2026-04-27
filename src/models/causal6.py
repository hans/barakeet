"""
causal6 decoder entry points.

Five decoders, one GPU kernel, one outer CV strategy (StratifiedKFold).
Clean parquet-only outputs — no fitted-estimator joblib blobs.

Functions:
    run_acoustic_searchlight(...)
    run_behavior_with_control(...)
    run_behavior_hga_only(...)
    run_ganong_with_control(...)
    run_ganong_hga_only(...)

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
    fit_batched_l2_logreg_perms,
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


def audit_class_balance(
    epochs: mne.Epochs,
    subject: str,
    *,
    n_folds: int = 5,
    cv_random_state: int = 42,
    resampled_steps: tuple[int, ...] = (1, 6),
) -> pl.DataFrame:
    """
    Summarize post-stratification per-fold binary-class counts for every
    (decoder × group) the three causal6 decoders will fit on this subject.

    Uses the same `StratifiedKFold(n_splits=n_folds, shuffle=True,
    random_state=cv_random_state)` splitter as `_fit_batched_cv`, so the
    per-fold minority counts reported here are exactly the ones the fit will
    see. Emits `L.warning(...)` for every "low" and "skipped" row.

    Columns:
        subject, decoder ∈ {"acoustic", "behavior_full", "behavior_hga_only",
            "ganong_full", "ganong_hga_only"},
        phoneme_pair, word_end (null for acoustic and ganong rows — both pool
            across completions; populated for behavior rows),
        n_total, n_class_0, n_class_1, min_class,
        will_skip,                       # mirrors _has_enough_per_class(y, n_folds)
        min_test_minority_per_fold,      # null when will_skip
        min_train_minority_per_fold,     # null when will_skip
        status ∈ {"ok", "low", "skipped"}
            "skipped": will_skip (decoder drops the group)
            "low":     min_test_minority_per_fold < 2
                       or min_train_minority_per_fold < 2 * n_folds
            "ok":      otherwise
    """
    assert epochs.metadata is not None
    md = epochs.metadata

    def _summarize(decoder: str, phoneme_pair: str, word_end, y: np.ndarray) -> dict:
        n_total = int(y.size)
        counts = (
            np.bincount(y.astype(np.int64), minlength=2)
            if n_total > 0 else np.zeros(2, dtype=np.int64)
        )
        n_class_0, n_class_1 = int(counts[0]), int(counts[1])
        min_class = int(min(n_class_0, n_class_1))
        will_skip = (n_total == 0) or (not _has_enough_per_class(y, n_folds))

        if will_skip:
            min_test_min: Optional[int] = None
            min_train_min: Optional[int] = None
            status = "skipped"
        else:
            minority_label = int(np.argmin(counts))
            skf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=cv_random_state,
            )
            test_mins: list[int] = []
            train_mins: list[int] = []
            for train_idx, test_idx in skf.split(np.zeros(n_total), y):
                test_mins.append(int((y[test_idx] == minority_label).sum()))
                train_mins.append(int((y[train_idx] == minority_label).sum()))
            min_test_min = min(test_mins)
            min_train_min = min(train_mins)
            if min_test_min < 2 or min_train_min < 2 * n_folds:
                status = "low"
            else:
                status = "ok"

        return {
            "subject": subject,
            "decoder": decoder,
            "phoneme_pair": phoneme_pair,
            "word_end": word_end,
            "n_total": n_total,
            "n_class_0": n_class_0,
            "n_class_1": n_class_1,
            "min_class": min_class,
            "will_skip": will_skip,
            "min_test_minority_per_fold": min_test_min,
            "min_train_minority_per_fold": min_train_min,
            "status": status,
        }

    phoneme_pairs = sorted(md.phoneme_pair.dropna().unique())
    resampled_mask = md.resampled.isin(resampled_steps).values

    rows: list[dict] = []

    # Acoustic: one row per phoneme_pair (unambiguous-step filter).
    for pp in phoneme_pairs:
        sel = (md.phoneme_pair == pp).values & resampled_mask
        if sel.sum() == 0:
            rows.append(_summarize("acoustic", pp, None, np.zeros(0, dtype=np.int64)))
            continue
        y = _resolve_target(md, "categorical_acoustic_cue", pp, sel)
        rows.append(_summarize("acoustic", pp, None, y))

    # Behavior: identical y over (pp, word_end) for both behavior decoders. Emit
    # two rows per group, one per decoder, so the table is readable alongside
    # the per-decoder sweep logs.
    for pp in phoneme_pairs:
        pp_mask = (md.phoneme_pair == pp).values
        if pp_mask.sum() == 0:
            continue
        for we in sorted(md.word_end[pp_mask].dropna().unique()):
            sel = pp_mask & (md.word_end == we).values
            if sel.sum() == 0:
                continue
            y = _resolve_target(md, "behavior_categorical_forced", pp, sel)
            for decoder in ("behavior_full", "behavior_hga_only"):
                rows.append(_summarize(decoder, pp, we, y))

    # Ganong: pooled across completions — one row per phoneme_pair per decoder.
    # word_end = None reflects the pooled group.
    for pp in phoneme_pairs:
        sel = (md.phoneme_pair == pp).values
        if sel.sum() == 0:
            continue
        y = _resolve_target(md, "behavior_categorical_forced", pp, sel)
        for decoder in ("ganong_full", "ganong_hga_only"):
            rows.append(_summarize(decoder, pp, None, y))

    schema = {
        "subject": pl.Utf8,
        "decoder": pl.Utf8,
        "phoneme_pair": pl.Utf8,
        "word_end": pl.Utf8,
        "n_total": pl.Int64,
        "n_class_0": pl.Int64,
        "n_class_1": pl.Int64,
        "min_class": pl.Int64,
        "will_skip": pl.Boolean,
        "min_test_minority_per_fold": pl.Int64,
        "min_train_minority_per_fold": pl.Int64,
        "status": pl.Utf8,
    }
    df = pl.DataFrame(rows, schema=schema).sort(["decoder", "phoneme_pair", "word_end"])

    for row in df.iter_rows(named=True):
        group = row["phoneme_pair"]
        if row["word_end"] is not None:
            group = f"{group}/{row['word_end']}"
        if row["status"] == "skipped":
            L.warning(
                f"[audit][{subject}][{row['decoder']}][{group}] skipped: "
                f"n_total={row['n_total']} class counts = "
                f"[{row['n_class_0']}, {row['n_class_1']}]"
            )
        elif row["status"] == "low":
            L.warning(
                f"[audit][{subject}][{row['decoder']}][{group}] low: "
                f"class counts = [{row['n_class_0']}, {row['n_class_1']}], "
                f"min test/train minority per fold = "
                f"{row['min_test_minority_per_fold']}/"
                f"{row['min_train_minority_per_fold']}"
            )

    return df


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


def _fit_batched_cv_permutations(
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
    pbar: Optional[tqdm] = None,
) -> pl.DataFrame:
    """
    Batched CV fit under label permutations — proper refit-based null.

    For each of B problems and K = len(permute_seeds) permutations, fits a
    fresh L2 LogReg on (X, shuffled y) using StratifiedKFold splits taken
    from the *real* labels. Returns test ROC-AUC per (problem × fold ×
    permutation). Predictions/coefficients are dropped (nulls only need AUC).

    K permutations are fit in chunks of `permutation_chunk_size` per GPU
    call. Within each chunk a single `fit_batched_l2_logreg_perms` call
    handles K_chunk permutations at once — X stays at (B, n, d) and is
    broadcast over the K dimension internally, avoiding the K× memory
    copy the old tiled-X path required.

    Splits use the real y (stratification guarantees balanced fold sizes
    that match the real run's convention). Under permuted labels a fold's
    test set may happen to have zero class variance — those AUCs are NaN
    and get `nanmean`'d out by downstream aggregation.
    """
    n_trials, B, _d = X.shape
    K = len(permute_seeds)
    assert y.shape == (n_trials,)
    assert problem_meta.height == B, (
        f"problem_meta has {problem_meta.height} rows, expected B={B}"
    )
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

        # Standardisation is label-independent — compute once per fold, reuse
        # across every permutation chunk below.
        X_tr_std_b, X_te_std_b, _, _ = standardise_per_batch(
            X_train_t, mask_tr_b, X_test_t
        )

        for chunk_start in range(0, K, permutation_chunk_size):
            chunk_end = min(chunk_start + permutation_chunk_size, K)
            Kc = chunk_end - chunk_start

            # Pull the Kc permutations' label rows for train & test.
            # y is per-permutation (no B dim); we broadcast across B via
            # zero-stride .expand views — no K× materialisation.
            y_perm_chunk = y_perms_gpu[chunk_start:chunk_end]                # (Kc, n_trials)
            y_train_kn = y_perm_chunk.index_select(1, train_idx_t)           # (Kc, n_tr)
            y_test_kn = y_perm_chunk.index_select(1, test_idx_t)             # (Kc, n_te)
            y_train_kbn = y_train_kn.unsqueeze(1).expand(Kc, B, n_tr)        # broadcast view
            y_test_kbn = y_test_kn.unsqueeze(1).expand(Kc, B, n_te)          # broadcast view

            # Class-balanced sample weights depend only on the permuted y
            # (mask is uniform across B). Compute at (Kc, n_tr) and broadcast.
            sw_train_kn = compute_balanced_sample_weight(
                y_train_kn,
                torch.ones(Kc, n_tr, dtype=dtype, device=device),
            )                                                                # (Kc, n_tr)
            sw_train_kbn = sw_train_kn.unsqueeze(1).expand(Kc, B, n_tr)      # broadcast view

            beta, _, _ = fit_batched_l2_logreg_perms(
                X_tr_std_b, y_train_kbn, mask_tr_b, sw_train_kbn,
                reg_lambda=reg_lambda, tol=tol, max_iter=max_iter,
            )                                                                # (Kc, B, d)

            z_te = torch.einsum("bnd,kbd->kbn", X_te_std_b, beta)            # (Kc, B, n_te)
            proba_te = torch.sigmoid(z_te)
            # batched_roc_auc takes 2D tensors; flatten (K, B) into one batch
            # dim. proba_te is contiguous so reshape is a view; y_test_kbn
            # is an expand view and reshape materialises here (~Kc*B*n_te
            # floats — small relative to the fit).
            aucs = batched_roc_auc(
                proba_te.reshape(Kc * B, n_te),
                y_test_kbn.reshape(Kc * B, n_te),
            ).cpu().numpy()

            perm_ids = np.repeat(
                np.arange(chunk_start, chunk_end, dtype=np.int64), B
            )
            prob_ids = np.tile(problem_idx, Kc)
            scores_frames.append(pl.DataFrame({
                "_problem_idx": prob_ids,
                "fold": np.full(Kc * B, fold, dtype=np.int32),
                "permutation_idx": perm_ids,
                "test_roc_auc": aucs,
                "n_train": np.full(Kc * B, n_tr, dtype=np.int64),
                "n_test": np.full(Kc * B, n_te, dtype=np.int64),
            }))

            if pbar is not None:
                pbar.update(Kc)

    scores = pl.concat(scores_frames).join(
        problem_meta_with_idx, on="_problem_idx", how="left"
    ).drop("_problem_idx")
    return scores


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


# ---------------------------------------------------------------------------
# Permutation-test variants
# ---------------------------------------------------------------------------


def run_acoustic_searchlight_permutations(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    reg_lambda: float,
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    target: Literal["categorical_acoustic_cue", "subject_specific_acoustics"]
        = "categorical_acoustic_cue",
    resampled_steps: tuple[int, ...] = (1, 6),
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> pl.DataFrame:
    """
    Permutation-test twin of `run_acoustic_searchlight`.

    For each seed in `permute_seeds`, shuffle trial labels globally and
    refit the full searchlight. Fold splits are drawn from the *real*
    labels (stratification guarantees balanced fold sizes matching the
    real run). Only test ROC-AUC is returned — predictions/coefficients
    are dropped.

    Returns a single scores DataFrame with one row per
    (problem × fold × permutation). Columns:
        subject, phoneme_pair, electrode_idx, smin, smax,
        fold, permutation_idx, test_roc_auc, n_train, n_test, target.
    """
    assert epochs.metadata is not None
    md = epochs.metadata
    phoneme_pairs = sorted(md.phoneme_pair.dropna().unique())

    X_full = epochs.get_data(picks=list(electrode_idxs))
    n_electrodes = len(electrode_idxs)
    n_windows = windows.shape[0]
    win_size = int(windows[0, 1] - windows[0, 0])
    assert (windows[:, 1] - windows[:, 0] == win_size).all()

    resampled_mask = md.resampled.isin(resampled_steps).values

    scores_all: list[pl.DataFrame] = []

    K = len(permute_seeds)
    candidate_pairs = [
        pp for pp in phoneme_pairs
        if ((md.phoneme_pair == pp).values & resampled_mask).sum() > 0
    ]
    units_per_group = K * n_folds
    total_units = len(candidate_pairs) * units_per_group

    pbar = tqdm(
        total=total_units,
        desc=f"acoustic-null[{subject}] λ={reg_lambda:g} K={K} {target}",
        unit="perm·fold", leave=False,
    )
    try:
        for phoneme_pair in candidate_pairs:
            pbar.set_postfix_str(f"pp={phoneme_pair}")
            selection = (md.phoneme_pair == phoneme_pair).values & resampled_mask

            y = _resolve_target(md, target, phoneme_pair, selection)
            if not _has_enough_per_class(y, n_folds):
                counts = np.bincount(y.astype(np.int64)) if len(np.unique(y)) > 0 else []
                L.warning(
                    f"[acoustic-null][{subject}/{phoneme_pair}] skipping: insufficient "
                    f"class balance for StratifiedKFold(n_splits={n_folds}); "
                    f"class counts = {list(counts)}"
                )
                pbar.update(units_per_group)
                continue

            X_sel = X_full[selection]
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

            scores = _fit_batched_cv_permutations(
                X_batch, y, problem_meta,
                permute_seeds=permute_seeds,
                permutation_chunk_size=permutation_chunk_size,
                reg_lambda=reg_lambda,
                n_folds=n_folds, cv_random_state=cv_random_state,
                device=device, dtype=dtype, tol=tol, max_iter=max_iter,
                pbar=pbar,
            )
            scores_all.append(scores)
    finally:
        pbar.close()

    if not scores_all:
        return pl.DataFrame()

    return pl.concat(scores_all).with_columns(pl.lit(target).alias("target"))


def _run_behavior_core_permutations(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    with_control: bool,
    reg_lambda: float,
    reg_lambda_baseline: Optional[float],
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    n_folds: int,
    cv_random_state: int,
    device: str,
    dtype: torch.dtype,
    tol: float,
    max_iter: int,
) -> pl.DataFrame:
    """
    Permutation-test twin of `_run_behavior_core`.

    For each (phoneme_pair, word_end), for each seed in `permute_seeds`:
      - Shuffle labels with that seed.
      - Refit the full searchlight batch (all electrode × window problems).
      - If `with_control`, also refit the baseline (resampled-only) model.

    Because `_fit_batched_cv_permutations` seeds its RNG deterministically
    from each `permute_seeds[k]`, the full and baseline calls see the same
    shuffled labels per permutation — pairing `full - baseline` works the
    same as in the real run.
    """
    assert epochs.metadata is not None
    md = epochs.metadata
    phoneme_pairs = sorted(md.phoneme_pair.dropna().unique())

    X_full = epochs.get_data(picks=list(electrode_idxs))
    n_electrodes = len(electrode_idxs)
    n_windows = windows.shape[0]
    win_size = int(windows[0, 1] - windows[0, 0])
    assert (windows[:, 1] - windows[:, 0] == win_size).all()

    full_parts: list[pl.DataFrame] = []
    base_parts: list[pl.DataFrame] = []

    groups: list[tuple[str, str]] = []
    for phoneme_pair in phoneme_pairs:
        pp_mask = (md.phoneme_pair == phoneme_pair).values
        if pp_mask.sum() == 0:
            continue
        for word_end in sorted(md.word_end[pp_mask].dropna().unique()):
            groups.append((phoneme_pair, word_end))

    decoder_label = "behavior_full-null" if with_control else "behavior_hga_only-null"
    K = len(permute_seeds)
    n_models_per_group = 2 if with_control else 1
    units_per_group = K * n_folds * n_models_per_group
    total_units = len(groups) * units_per_group

    pbar = tqdm(
        total=total_units,
        desc=f"{decoder_label}[{subject}] λ={reg_lambda:g} K={K}",
        unit="perm·fold", leave=False,
    )
    try:
        for phoneme_pair, word_end in groups:
            pbar.set_postfix_str(f"pp={phoneme_pair} we={word_end} model=full")
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
                pbar.update(units_per_group)
                continue

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
            full_parts.append(_fit_batched_cv_permutations(
                X_full_batch, y, full_meta,
                permute_seeds=permute_seeds,
                permutation_chunk_size=permutation_chunk_size,
                reg_lambda=reg_lambda,
                n_folds=n_folds, cv_random_state=cv_random_state,
                device=device, dtype=dtype, tol=tol, max_iter=max_iter,
                pbar=pbar,
            ))

            if with_control:
                pbar.set_postfix_str(f"pp={phoneme_pair} we={word_end} model=baseline")
                X_base = resampled_feat.reshape(n_trials, 1, 1)
                base_meta = pl.DataFrame({
                    "subject": [subject],
                    "phoneme_pair": [phoneme_pair],
                    "word_end": [word_end],
                    "electrode_idx": [-1],
                    "smin": [-1], "smax": [-1],
                    "model": ["baseline"],
                })
                base_parts.append(_fit_batched_cv_permutations(
                    X_base, y, base_meta,
                    permute_seeds=permute_seeds,
                    permutation_chunk_size=permutation_chunk_size,
                    reg_lambda=(reg_lambda_baseline if reg_lambda_baseline is not None
                                else reg_lambda),
                    n_folds=n_folds, cv_random_state=cv_random_state,
                    device=device, dtype=dtype, tol=tol, max_iter=max_iter,
                    pbar=pbar,
                ))
    finally:
        pbar.close()

    parts = full_parts + base_parts
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts)


def run_behavior_with_control_permutations(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    reg_lambda: float,
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    reg_lambda_baseline: Optional[float] = None,
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> pl.DataFrame:
    """Permutation-test twin of `run_behavior_with_control` (full + baseline)."""
    return _run_behavior_core_permutations(
        epochs, subject, electrode_idxs, windows,
        with_control=True,
        reg_lambda=reg_lambda, reg_lambda_baseline=reg_lambda_baseline,
        permute_seeds=permute_seeds,
        permutation_chunk_size=permutation_chunk_size,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )


def run_behavior_hga_only_permutations(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    reg_lambda: float,
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> pl.DataFrame:
    """Permutation-test twin of `run_behavior_hga_only` (full only)."""
    return _run_behavior_core_permutations(
        epochs, subject, electrode_idxs, windows,
        with_control=False,
        reg_lambda=reg_lambda, reg_lambda_baseline=None,
        permute_seeds=permute_seeds,
        permutation_chunk_size=permutation_chunk_size,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )


# ---------------------------------------------------------------------------
# Ganong decoders (pooled across lexical completions)
# ---------------------------------------------------------------------------
#
# Mirrors the behavior decoders but iterates `phoneme_pair` only — trials from
# both `word_end` groups (e.g. -esolate + -ecessary for `dn`) are pooled into
# a single fit per phoneme_pair. Output schemas drop the `word_end` column.
# The full-vs-baseline contrast (baseline features = [resampled]) exposes the
# HGA contribution to the Ganong boundary shift: the baseline can only see
# the acoustic step, so full − baseline attributes both acoustic-driven and
# lexical-driven perceptual bias to HGA.


def _run_ganong_core(
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
    Shared core for ganong-with-control and ganong-HGA-only decoders.

    Groups by `phoneme_pair` only (no word_end split). For each group:
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

    decoder_label = "ganong_full" if with_control else "ganong_hga_only"
    pbar = tqdm(
        phoneme_pairs,
        desc=f"{decoder_label}[{subject}] λ={reg_lambda:g}",
        unit="pp", leave=False,
    )
    for phoneme_pair in pbar:
        sel = (md.phoneme_pair == phoneme_pair).values
        if sel.sum() == 0:
            continue
        y = _resolve_target(md, "behavior_categorical_forced", phoneme_pair, sel)
        if not _has_enough_per_class(y, n_folds):
            counts = np.bincount(y.astype(np.int64)) if len(np.unique(y)) > 0 else []
            L.warning(
                f"[{decoder_label}][{subject}/{phoneme_pair}] "
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


def run_ganong_with_control(
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
    Ganong decoding with `resampled` control predictor, pooled across
    lexical completions.

    Full model features: [resampled, HGA window]. Baseline features: [resampled].
    Baseline is fit once per (phoneme_pair, fold) and reused across electrodes
    via the `model` key in the output DataFrames. `word_end` is NOT in the
    output schema — trials from both completions enter the same fit.
    """
    return _run_ganong_core(
        epochs, subject, electrode_idxs, windows,
        with_control=True,
        reg_lambda=reg_lambda, reg_lambda_baseline=reg_lambda_baseline,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )


def run_ganong_hga_only(
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
    Ganong decoding without control predictor: single model on HGA window only,
    pooled across lexical completions. Outputs have `model='full'` only.
    """
    return _run_ganong_core(
        epochs, subject, electrode_idxs, windows,
        with_control=False,
        reg_lambda=reg_lambda, reg_lambda_baseline=None,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )


def _run_ganong_core_permutations(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    with_control: bool,
    reg_lambda: float,
    reg_lambda_baseline: Optional[float],
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    n_folds: int,
    cv_random_state: int,
    device: str,
    dtype: torch.dtype,
    tol: float,
    max_iter: int,
) -> pl.DataFrame:
    """Permutation-test twin of `_run_ganong_core`."""
    assert epochs.metadata is not None
    md = epochs.metadata
    phoneme_pairs = sorted(md.phoneme_pair.dropna().unique())

    X_full = epochs.get_data(picks=list(electrode_idxs))
    n_electrodes = len(electrode_idxs)
    n_windows = windows.shape[0]
    win_size = int(windows[0, 1] - windows[0, 0])
    assert (windows[:, 1] - windows[:, 0] == win_size).all()

    full_parts: list[pl.DataFrame] = []
    base_parts: list[pl.DataFrame] = []

    decoder_label = "ganong_full-null" if with_control else "ganong_hga_only-null"
    K = len(permute_seeds)
    n_models_per_group = 2 if with_control else 1
    units_per_group = K * n_folds * n_models_per_group
    candidate_pairs = [
        pp for pp in phoneme_pairs if (md.phoneme_pair == pp).values.sum() > 0
    ]
    total_units = len(candidate_pairs) * units_per_group

    pbar = tqdm(
        total=total_units,
        desc=f"{decoder_label}[{subject}] λ={reg_lambda:g} K={K}",
        unit="perm·fold", leave=False,
    )
    try:
        for phoneme_pair in candidate_pairs:
            pbar.set_postfix_str(f"pp={phoneme_pair} model=full")
            sel = (md.phoneme_pair == phoneme_pair).values
            y = _resolve_target(md, "behavior_categorical_forced", phoneme_pair, sel)
            if not _has_enough_per_class(y, n_folds):
                counts = np.bincount(y.astype(np.int64)) if len(np.unique(y)) > 0 else []
                L.warning(
                    f"[{decoder_label}][{subject}/{phoneme_pair}] "
                    f"skipping: insufficient class balance for "
                    f"StratifiedKFold(n_splits={n_folds}); class counts = {list(counts)}"
                )
                pbar.update(units_per_group)
                continue

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
                "electrode_idx": elec_per,
                "smin": smin_per,
                "smax": smax_per,
                "model": ["full"] * B_full,
            })
            full_parts.append(_fit_batched_cv_permutations(
                X_full_batch, y, full_meta,
                permute_seeds=permute_seeds,
                permutation_chunk_size=permutation_chunk_size,
                reg_lambda=reg_lambda,
                n_folds=n_folds, cv_random_state=cv_random_state,
                device=device, dtype=dtype, tol=tol, max_iter=max_iter,
                pbar=pbar,
            ))

            if with_control:
                pbar.set_postfix_str(f"pp={phoneme_pair} model=baseline")
                X_base = resampled_feat.reshape(n_trials, 1, 1)
                base_meta = pl.DataFrame({
                    "subject": [subject],
                    "phoneme_pair": [phoneme_pair],
                    "electrode_idx": [-1],
                    "smin": [-1], "smax": [-1],
                    "model": ["baseline"],
                })
                base_parts.append(_fit_batched_cv_permutations(
                    X_base, y, base_meta,
                    permute_seeds=permute_seeds,
                    permutation_chunk_size=permutation_chunk_size,
                    reg_lambda=(reg_lambda_baseline if reg_lambda_baseline is not None
                                else reg_lambda),
                    n_folds=n_folds, cv_random_state=cv_random_state,
                    device=device, dtype=dtype, tol=tol, max_iter=max_iter,
                    pbar=pbar,
                ))
    finally:
        pbar.close()

    parts = full_parts + base_parts
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts)


def run_ganong_with_control_permutations(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    reg_lambda: float,
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    reg_lambda_baseline: Optional[float] = None,
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> pl.DataFrame:
    """Permutation-test twin of `run_ganong_with_control` (full + baseline)."""
    return _run_ganong_core_permutations(
        epochs, subject, electrode_idxs, windows,
        with_control=True,
        reg_lambda=reg_lambda, reg_lambda_baseline=reg_lambda_baseline,
        permute_seeds=permute_seeds,
        permutation_chunk_size=permutation_chunk_size,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )


def run_ganong_hga_only_permutations(
    epochs: mne.Epochs,
    subject: str,
    electrode_idxs: Sequence[int],
    windows: np.ndarray,
    *,
    reg_lambda: float,
    permute_seeds: Sequence[int],
    permutation_chunk_size: int,
    n_folds: int = 5,
    cv_random_state: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> pl.DataFrame:
    """Permutation-test twin of `run_ganong_hga_only` (full only)."""
    return _run_ganong_core_permutations(
        epochs, subject, electrode_idxs, windows,
        with_control=False,
        reg_lambda=reg_lambda, reg_lambda_baseline=None,
        permute_seeds=permute_seeds,
        permutation_chunk_size=permutation_chunk_size,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=dtype, tol=tol, max_iter=max_iter,
    )
