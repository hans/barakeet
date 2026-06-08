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


def make_title_page(label: str, count: int,
                    page_width: float = 612, page_height: float = 792):
    """Return a pypdf PageObject for a centered title page.

    Uses standard built-in Helvetica-Bold only (no font embedding).
    ``label`` may contain literal newlines to split across lines.
    """
    import io
    from pypdf import PdfReader

    lines = label.split("\n")
    font_size = 48
    line_height = font_size * 1.4
    count_size = 24

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    char_w = font_size * 0.55
    count_char_w = count_size * 0.55
    count_label = f"n = {count}"

    total_h = len(lines) * line_height
    y_top = page_height / 2 + total_h / 2

    parts = ["BT", "/F1 48 Tf"]
    for i, line in enumerate(lines):
        x = (page_width - len(line) * char_w) / 2
        y = y_top - (i + 1) * line_height
        parts.append(f"1 0 0 1 {x:.1f} {y:.1f} Tm")
        parts.append(f"({_esc(line)}) Tj")

    cx = (page_width - len(count_label) * count_char_w) / 2
    cy = y_top - len(lines) * line_height - count_size * 1.2
    parts += ["/F1 24 Tf", f"1 0 0 1 {cx:.1f} {cy:.1f} Tm", f"({_esc(count_label)}) Tj"]
    parts.append("ET")

    content = "\n".join(parts).encode("latin-1")

    buf = io.BytesIO()

    def w(s: str | bytes) -> None:
        buf.write(s.encode("latin-1") if isinstance(s, str) else s)

    offsets: dict[int, int] = {}
    w(b"%PDF-1.4\n")
    offsets[1] = buf.tell()
    w("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n")
    offsets[2] = buf.tell()
    w("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n")
    offsets[3] = buf.tell()
    w(f"3 0 obj\n<< /Type /Page /Parent 2 0 R "
      f"/MediaBox [0 0 {page_width:.0f} {page_height:.0f}] "
      f"/Contents 4 0 R "
      f"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\n")
    offsets[4] = buf.tell()
    w(f"4 0 obj\n<< /Length {len(content)} >>\nstream\n")
    buf.write(content)
    w("\nendstream\nendobj\n\n")
    offsets[5] = buf.tell()
    w("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n\n")
    xref_pos = buf.tell()
    w("xref\n0 6\n")
    w("0000000000 65535 f \n")
    for i in range(1, 6):
        w(f"{offsets[i]:010d} 00000 n \n")
    w(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n")
    buf.seek(0)
    return PdfReader(buf).pages[0]


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
