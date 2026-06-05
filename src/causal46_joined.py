"""Helpers for the causal46_joined pipeline.

The joined pipeline restricts behavior and ganong decoders to electrodes that
are acoustic-significant (AS) per the causal6 acoustic peak test. This module
exposes the pure-Python core of the AS-filter checkpoint so it is testable
without ploomber.

Site-type display constants
---------------------------
``SITE_TYPE_ORDER``, ``SITE_TYPE_LABELS``, ``SITE_TYPE_COLORS``, and
``ETC_SITE_TYPES`` are the canonical definitions shared across all notebooks
and scripts that visualise early-window site classifications.  They match the
Sankey / contrast-plot palette exactly.

``site_type_display_label(raw)`` maps a raw ``site_type`` value (as it appears
in parquet outputs and CSVs) to the human-readable label used in figures.
``site_type_sort_key(raw)`` returns an integer that places types in the
preferred slide/figure order.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Canonical site-type display constants
# ---------------------------------------------------------------------------

# Preferred display order (typed categories first, then catch-all "etc" / Other)
SITE_TYPE_ORDER: list[str] = [
    "type2_early_perceptual",
    "type3_asymmetric",
    "type4_early_perceptual_mirrored",
    "type5_behav_only",
    "type1_acoustic_only",
    "etc",
    # raw values that map into "etc":
    "A_unsigned",
    "problematic",
    "interesting",
    # remaining auto-assigned values kept at end
    "complex",
    "unknown",
    "grab_bag",
]

# Human-readable labels for figures / legends
SITE_TYPE_LABELS: dict[str, str] = {
    "type1_acoustic_only":             "Acoustic only",
    "type2_early_perceptual":          "Acoustic+perceptual",
    "type3_asymmetric":                "Acoustic+perceptual\n(one-sided)",
    "type4_early_perceptual_mirrored": "Acoustic+perceptual\n(mirrored)",
    "type5_behav_only":                "Perceptual only",
    "etc":                             "Other",
    "A_unsigned":                      "Other",
    "problematic":                     "Other",
    "interesting":                     "Other",
    "complex":                         "Complex",
    "unknown":                         "Unknown",
    "grab_bag":                        "Grab-bag",
}

# Colors matching the Sankey / contrast-plot palette
SITE_TYPE_COLORS: dict[str, str] = {
    "type1_acoustic_only":             "#4E79A7",
    "type2_early_perceptual":          "#59A14F",
    "type3_asymmetric":                "#F28E2B",
    "type4_early_perceptual_mirrored": "#B07AA1",
    "type5_behav_only":                "#E15759",
    "etc":                             "#AAAAAA",
    "A_unsigned":                      "#AAAAAA",
    "problematic":                     "#AAAAAA",
    "interesting":                     "#AAAAAA",
    "complex":                         "#762a83",
    "unknown":                         "#d9d9d9",
    "grab_bag":                        "#d73027",
}

# Raw site_type values that are collapsed into "etc" / "Other" in figures
ETC_SITE_TYPES: list[str] = ["A_unsigned", "problematic", "interesting"]

_SORT_KEY: dict[str, int] = {t: i for i, t in enumerate(SITE_TYPE_ORDER)}


def site_type_display_label(raw: str) -> str:
    """Human-readable label for a raw site_type string."""
    return SITE_TYPE_LABELS.get(str(raw), str(raw))


def site_type_sort_key(raw: str) -> int:
    """Integer sort key placing types in the preferred figure order."""
    return _SORT_KEY.get(str(raw), len(SITE_TYPE_ORDER))


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
    as_keys = sig.select(["subject", "electrode_idx"]).unique()

    annotated: dict[str, pd.DataFrame] = {}
    subjects_with_as: list[str] = []
    for subject, sr in electrode_dfs_by_subject.items():
        as_idxs = (
            as_keys.filter(pl.col("subject") == subject)["electrode_idx"]
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
