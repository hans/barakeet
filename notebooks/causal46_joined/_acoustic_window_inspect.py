"""Reusable helpers for acoustic_decoding_single_electrode_inspect.py.

Window construction/validation and per-fold transfer AUC computation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def resolve_source_and_peak_windows(
    peaks_df,
    subject: str,
    electrode_idx: int,
    phoneme_pair: str,
    source_smin_override=None,
    source_smax_override=None,
):
    """Look up the peak acoustic window for this site and resolve the source window.

    Parameters
    ----------
    peaks_df : polars DataFrame
        Loaded from phon_peaks_all.parquet.
    source_smin_override, source_smax_override : int or None
        If both provided, use as source window instead of the peak window.

    Returns
    -------
    peak_smin, peak_smax : int or None
        Peak window bounds (None if no peak row exists for this site).
    peak_auc : float or None
        Peak-window test AUC from the parquet (display context only).
    source_smin, source_smax : int
        Window to use as the source decoder.
    width : int
        source_smax - source_smin (fixed width for both windows).

    Raises
    ------
    ValueError
        If no peak row and no source override is provided.
    """
    import polars as pl

    row = peaks_df.filter(
        (pl.col("subject") == subject)
        & (pl.col("electrode_idx") == electrode_idx)
        & (pl.col("phoneme_pair") == phoneme_pair)
    )
    has_peak = row.height > 0

    if has_peak:
        peak_smin = int(row["smin"][0])
        peak_smax = int(row["smax"][0])
        peak_auc = float(row["test_roc_auc"][0])
    else:
        peak_smin = peak_smax = peak_auc = None

    if source_smin_override is not None and source_smax_override is not None:
        source_smin = int(source_smin_override)
        source_smax = int(source_smax_override)
    elif has_peak:
        source_smin, source_smax = peak_smin, peak_smax
    else:
        raise ValueError(
            f"No peak row for {subject} e{electrode_idx} {phoneme_pair} and "
            "no source_smin/source_smax override provided."
        )

    width = source_smax - source_smin
    return peak_smin, peak_smax, peak_auc, source_smin, source_smax, width


def build_new_window(
    onset_s,
    onset_sample,
    width: int,
    epoch_tmin: float,
    epoch_sfreq: float,
    min_sample: int,
    n_times: int,
) -> tuple:
    """Resolve new-window (smin, smax) from onset_s (takes priority) or onset_sample.

    Parameters
    ----------
    onset_s : float or None
        Window onset in seconds post word onset.  Takes priority over
        onset_sample when both are provided.
    onset_sample : int or None
        Fallback when onset_s is None.
    width : int
        Window width in samples (must match source window).
    min_sample, n_times : int
        Epoch bounds to validate against.

    Returns
    -------
    (new_smin, new_smax) : (int, int)
    """
    if onset_s is not None:
        smin = int(round((onset_s - epoch_tmin) * epoch_sfreq))
    elif onset_sample is not None:
        smin = int(onset_sample)
    else:
        raise ValueError("Provide new_window_onset_s or new_window_onset_sample.")

    smax = smin + width
    if smin < min_sample:
        raise ValueError(
            f"new_smin={smin} < min_sample={min_sample}. "
            "Increase new_window_onset_s."
        )
    if smax > n_times:
        raise ValueError(
            f"new_smax={smax} > n_times={n_times}. "
            "Decrease new_window_onset_s."
        )
    return smin, smax


def compute_transfer_auc(
    epoch_data: np.ndarray,
    phoneme_pair: str,
    source_smin: int,
    source_smax: int,
    new_smin: int,
    new_smax: int,
    predictions,
    coefficients,
    n_folds: int,
) -> tuple:
    """Per-fold: standardize new-window X with its own scaler, apply source coef.

    Mirrors the cross-scaler normalization in evaluate_phonetic_transfer
    (src/viz_paper.py:2563):
      - X from new window, standardized with new-window (mean, scale) from fold f
      - multiplied by coef from source window, fold f
      - sigmoid → proba; evaluated against new-window held-out labels

    Parameters
    ----------
    epoch_data : ndarray (N_total_epochs, N_times)
        Full epoch data for the one electrode, as returned by
        ``ep.get_data(picks=[electrode_idx]).squeeze(1)``.
        Indexed by epoch_idx values (metadata index labels == positional
        indices assuming a default RangeIndex, as in evaluate_phonetic_transfer).
    phoneme_pair : str
        Filter predictions/coefficients to this phoneme pair.
    source_smin, source_smax : int
        Source-decoder window bounds.
    new_smin, new_smax : int
        New-window bounds (evaluation target).
    predictions, coefficients : polars DataFrame
        Returned by run_acoustic_searchlight, pre-filtered to this electrode.
    n_folds : int

    Returns
    -------
    (mean_auc, fold_aucs) : (float, list[float])
    """
    import polars as pl

    pp = pl.col("phoneme_pair") == phoneme_pair
    src = (pl.col("smin") == source_smin) & (pl.col("smax") == source_smax)
    new = (pl.col("smin") == new_smin) & (pl.col("smax") == new_smax)

    fold_aucs = []
    for fold in range(n_folds):
        f = pl.col("fold") == fold

        # Held-out test epochs for the new window (same split as the refit decoder)
        preds_new = predictions.filter(pp & new & f)
        assert preds_new.height > 0, (
            f"No predictions for new window smin={new_smin} smax={new_smax} fold={fold}"
        )
        epoch_idxs = preds_new["epoch_idx"].to_numpy()
        y_true = preds_new["decoder_target"].to_numpy().astype(np.int32)

        # New-window X for those held-out epochs
        X = epoch_data[epoch_idxs, new_smin:new_smax].astype(np.float64)

        # New-window scaler (standardize target window with its own stats)
        new_row = coefficients.filter(pp & new & f)
        assert new_row.height == 1, (
            f"Expected 1 coef row for new window fold={fold}, got {new_row.height}"
        )
        mean_f = np.array(new_row["mean"][0], dtype=np.float64)
        scale_f = np.array(new_row["scale"][0], dtype=np.float64)

        # Source-window coef (applied to standardized new-window data)
        src_row = coefficients.filter(pp & src & f)
        assert src_row.height == 1, (
            f"Expected 1 coef row for source window fold={fold}, got {src_row.height}"
        )
        coef_f = np.array(src_row["coef"][0], dtype=np.float64)

        X_std = (X - mean_f) / scale_f
        z = X_std @ coef_f
        proba = 1.0 / (1.0 + np.exp(-z))

        fold_aucs.append(float(roc_auc_score(y_true, proba)))

    return float(np.mean(fold_aucs)), fold_aucs
