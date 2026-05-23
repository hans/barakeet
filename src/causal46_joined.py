"""Helpers for the causal46_joined pipeline.

The joined pipeline restricts behavior and ganong decoders to electrodes that
are acoustic-significant (AS) per the causal6 acoustic peak test. This module
exposes the pure-Python core of the AS-filter checkpoint so it is testable
without ploomber.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd
import polars as pl


def compute_as_filter(
    phon_peaks_df: pl.DataFrame,
    electrode_dfs_by_subject: Mapping[str, pd.DataFrame],
    as_p_threshold: float = 0.05,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Annotate per-subject electrode tables with an ``acoustic_significant`` column.

    Parameters
    ----------
    phon_peaks_df:
        Concatenated peak-test parquet (typically
        ``outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet``).
        Must contain columns ``subject``, ``electrode_idx``, ``phoneme_pair``,
        and ``p_value`` (uncorrected). Other columns are ignored.
    electrode_dfs_by_subject:
        Per-subject electrode tables keyed by subject id. Each frame must have
        an ``electrode_idx`` column and a ``speech_responsive`` boolean
        column. (Other columns are passed through unchanged.)
    as_p_threshold:
        Uncorrected p-value threshold for the AS criterion. An electrode is
        AS if it has at least one row in ``phon_peaks_df`` with
        ``p_value < as_p_threshold`` (OR across ``phoneme_pair``).

    Returns
    -------
    annotated_by_subject:
        Same dict keys as the input; values are copies with a new
        ``acoustic_significant`` boolean column. The column is AND'd with
        ``speech_responsive`` so we never include an electrode that wasn't
        even speech-responsive (defensive).
    subjects_with_as:
        Subjects (sorted by their order in ``electrode_dfs_by_subject``) that
        have at least one ``acoustic_significant`` electrode.
    """
    sig = phon_peaks_df.filter(pl.col("p_value") < as_p_threshold)
    as_per_subj = (
        sig.group_by(["subject", "electrode_idx"])
        .agg(pl.lit(True).alias("acoustic_significant"))
    )

    annotated: dict[str, pd.DataFrame] = {}
    subjects_with_as: list[str] = []
    for subject, sr in electrode_dfs_by_subject.items():
        as_idxs = (
            as_per_subj.filter(pl.col("subject") == subject)["electrode_idx"]
            .to_list()
        )
        sr_out = sr.copy()
        sr_out["acoustic_significant"] = (
            sr_out["electrode_idx"].isin(as_idxs) & sr_out["speech_responsive"]
        )
        annotated[subject] = sr_out
        if sr_out["acoustic_significant"].any():
            subjects_with_as.append(subject)

    return annotated, subjects_with_as
