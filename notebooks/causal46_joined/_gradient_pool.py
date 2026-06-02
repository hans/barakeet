"""Shared electrode-pool builder for the gradient acoustic encoding rules.

Three notebooks in this directory (acoustic_ax_discrimination.py,
acoustic_univariate_gradient.py, multivariate_gradient_perception.py) operate
on the same pool: manifest-curated cells from `filtered_manifest.csv`,
collapsed to (subject, electrode_idx, phoneme_pair), with the peak acoustic
window (`smin`/`smax`) attached from causal6's `phon_peaks.parquet`.

Notebook-local on purpose: src/ stays untouched while the gradient rules are
in flux. Promote to src/ if a caller appears outside this directory.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_acoustic_pool(
    manifest_path: str | Path,
    phon_peaks_path: str | Path,
    subject: str | None = None,
) -> pd.DataFrame:
    """Build the gradient-acoustic-encoding electrode pool.

    Parameters
    ----------
    manifest_path
        Path to `filtered_manifest.csv` (causal46_joined curated cells).
    phon_peaks_path
        Path to a per-subject causal6 `phon_peaks.parquet` (must contain
        ``subject``, ``electrode_idx``, ``phoneme_pair``, ``smin``, ``smax``).
    subject
        If given, restrict the manifest filter to that subject before joining.

    Returns
    -------
    DataFrame keyed on (``subject``, ``electrode_idx``, ``phoneme_pair``) with
    ``smin``/``smax`` columns from the phon_peaks join. One row per site×pair.
    """
    manifest = pd.read_csv(manifest_path)

    # Any cell with an `acoustic tuning` letter (^[a-z]$) qualifies the site.
    mask_tuning = manifest["acoustic tuning"].str.match(r"^[a-z]$", na=False)
    pool = manifest.loc[mask_tuning, ["subject", "electrode_idx", "phoneme_pair"]].copy()

    if subject is not None:
        pool = pool[pool["subject"] == subject].copy()

    # Collapse (site × word_end) cells to (site × pair). Both completions share
    # the same acoustic window — the manifest tags cells, but for pooled-
    # completion gradient analyses we want one row per acoustic site.
    pool = pool.drop_duplicates(
        subset=["subject", "electrode_idx", "phoneme_pair"]
    ).reset_index(drop=True)

    peaks = pd.read_parquet(phon_peaks_path)[
        ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
    ]
    if subject is not None:
        peaks = peaks[peaks["subject"] == subject]

    out = pool.merge(
        peaks, on=["subject", "electrode_idx", "phoneme_pair"], how="inner"
    )
    return out
