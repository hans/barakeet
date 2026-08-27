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

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests

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


# ---------------------------------------------------------------------------
# all-speech-responsive perceptual fork (2026-08-27-all-speech-responsive-perceptual)
# ---------------------------------------------------------------------------
#
# The AS-restricted pipeline above (`compute_as_filter`) can only ever find
# perceptual effects at sites that already passed the acoustic-selectivity
# gate — it structurally cannot see a perceptually-selective, non-acoustic
# site. `compute_sr_site_universe` builds the broader (subject, electrode_idx,
# phoneme_pair) universe of ALL speech-responsive sites, annotated (not
# filtered) with acoustic significance, so downstream perceptual testing can
# run unconditioned on acoustic response.
#
# `cell_maxstat_fdr_test` supplies the window-search + multiple-comparisons
# correction that broader universe needs: per-cell best-window CIs are
# self-selected (the window is chosen from the same data being tested) and
# uncorrected across cells. `notebooks/causal46_joined/late_integration_maxstat_significance.py`
# already established the method for exactly this problem on this exact B4
# bootstrap+null structure (max-|z| permutation correction per cell, then
# BH-FDR across cells) — its own note says the self-selected-window count
# "must not be reported as a test". This function MIRRORS that method
# (same statistic, same unbiased-p formula, same BH-FDR step) rather than
# importing from it — `late_integration_maxstat_significance.py` is part of
# the frozen AS-restricted pipeline the fork must not touch, and its search
# range (`smin >= phon_smax`, post-acoustic only) doesn't apply here: non-AS
# cells have no `phon_smax` to anchor to, so this fork searches the full
# `behav_search_range` instead. Two call sites, one method, deliberately not
# one shared import.


def compute_sr_site_universe(
    sr_by_subject: Mapping[str, pd.DataFrame],
    subject_phoneme_pairs: Mapping[str, Sequence[str]],
    phon_peaks_df: pl.DataFrame,
    ac_p_value_threshold: float,
) -> pl.DataFrame:
    """Build the all-speech-responsive (subject, electrode_idx, phoneme_pair)
    universe, annotated with an `acoustic_significant` label (never a filter).

    Parameters
    ----------
    sr_by_subject:
        Per-subject `find_speech_responsive` electrode tables keyed by
        subject id. Each frame must have `electrode_idx` and
        `speech_responsive` columns.
    subject_phoneme_pairs:
        subject -> phoneme pairs that subject's epochs actually contain
        (from epoch metadata — a subject need not have seen every pair).
    phon_peaks_df:
        Concatenated acoustic peak-test parquet (`phon_peaks_all.parquet`).
        Must contain `subject`, `electrode_idx`, `phoneme_pair`, `p_value`,
        `smin`, `smax`, `test_roc_auc`.
    ac_p_value_threshold:
        Uncorrected p-value cutoff defining `acoustic_significant` — must
        match the threshold used by the B4 bootstrap (`t_tests_all_sr`'s
        `ac_p_value_threshold` param) so the two stay consistent by
        construction rather than by convention.

    Returns
    -------
    polars DataFrame keyed (subject, electrode_idx, phoneme_pair), one row
    per SR electrode x phoneme_pair the subject saw, with columns
    `acoustic_significant` (bool), `phon_smin`, `phon_smax`,
    `acoustic_peak_auc` (all three null when not acoustic-significant — a
    site with p_value >= threshold has no row in the filtered join, mirroring
    how `t_tests.py` only carries an acoustic window for AS-qualified cells).
    """
    rows: list[tuple[str, int, str]] = []
    for subject, sr in sr_by_subject.items():
        sr_electrodes = sr.loc[
            sr["speech_responsive"].astype(bool), "electrode_idx"
        ].tolist()
        pairs = list(subject_phoneme_pairs.get(subject, []))
        for electrode_idx in sr_electrodes:
            for phoneme_pair in pairs:
                rows.append((subject, int(electrode_idx), phoneme_pair))

    universe = pl.DataFrame(
        rows,
        schema={"subject": pl.Utf8, "electrode_idx": pl.Int64, "phoneme_pair": pl.Utf8},
        orient="row",
    )

    sig = (
        phon_peaks_df
        .filter(pl.col("p_value") < ac_p_value_threshold)
        .select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "test_roc_auc"])
        .rename({
            "smin": "phon_smin",
            "smax": "phon_smax",
            "test_roc_auc": "acoustic_peak_auc",
        })
    )

    out = (
        universe
        .join(sig, on=["subject", "electrode_idx", "phoneme_pair"], how="left")
        .with_columns(pl.col("phon_smin").is_not_null().alias("acoustic_significant"))
        .sort(["subject", "electrode_idx", "phoneme_pair"])
    )
    return out


def cell_maxstat_fdr_test(
    boot: pl.DataFrame,
    cell_keys: list[str],
    *,
    value_col: str = "mean_diff_raw",
    null_col: str = "mean_diff_raw_null",
    replicate_col: str = "replicate",
    window_cols: tuple[str, str] = ("smin", "smax"),
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Max-|z| permutation correction per cell, then BH-FDR across cells.

    Mirrors `late_integration_maxstat_significance.py`'s method on the same
    B4 bootstrap+null structure:

    1. Per (cell, window): `z_obs = |mean_r(value_col)| / std_r(null_col)` —
       standardized by the null's own SD so the max over windows isn't
       dominated by high-variance windows.
    2. Per cell: `obs_maxz = max over windows of z_obs`.
    3. Per (cell, replicate): `null_maxz = max over windows of
       |null_col| / std_r(null_col)` (same per-window SD as step 1).
    4. Per cell: `p = (#{null_maxz >= obs_maxz} + 1) / (R + 1)` — the
       unbiased permutation p (never exactly 0, unlike a plain fraction).
    5. BH-FDR across ALL cells in `boot` (one family — not split by any
       other grouping; that's a modeling choice, surface it in the caller's
       report rather than deciding it silently here).

    Returns a polars DataFrame keyed by `cell_keys` with `maxstat_obs_maxz`,
    `maxstat_p`, `maxstat_q`, `maxstat_reject` (`q < alpha`), and
    `maxstat_r` (replicate count, for a permutation-floor sanity check:
    `1 / (maxstat_r + 1)`). Cells where every window's null SD is 0 (no
    replicate variance) are dropped — undefined z, same as
    `late_integration_maxstat_significance.py`'s `sd_null > 0` filter.
    Empty DataFrame if `boot` is empty or nothing survives that filter.
    """
    if boot.height == 0:
        return pl.DataFrame()

    smin_col, smax_col = window_cols
    window_keys = cell_keys + [smin_col, smax_col]

    per_window = (
        boot
        .group_by(window_keys)
        .agg(
            pl.col(value_col).mean().abs().alias("__obs_abs"),
            pl.col(null_col).std().alias("__sd_null"),
        )
        .filter(pl.col("__sd_null") > 0)
        .with_columns((pl.col("__obs_abs") / pl.col("__sd_null")).alias("__z_obs"))
    )
    if per_window.height == 0:
        return pl.DataFrame()

    obs = per_window.group_by(cell_keys).agg(pl.col("__z_obs").max().alias("__obs_maxz"))

    null_z = (
        boot
        .join(per_window.select(window_keys + ["__sd_null"]), on=window_keys, how="inner")
        .with_columns((pl.col(null_col).abs() / pl.col("__sd_null")).alias("__z_null"))
    )
    null_max = (
        null_z
        .group_by(cell_keys + [replicate_col])
        .agg(pl.col("__z_null").max().alias("__null_maxz"))
    )

    obs_lookup = {
        tuple(row[k] for k in cell_keys): row["__obs_maxz"]
        for row in obs.iter_rows(named=True)
    }
    rows: list[dict] = []
    for keys_df in null_max.partition_by(cell_keys, maintain_order=True):
        keys = tuple(keys_df[k][0] for k in cell_keys)
        obs_maxz = obs_lookup.get(keys)
        if obs_maxz is None:
            continue
        null_vals = keys_df["__null_maxz"].to_numpy()
        R = len(null_vals)
        p = (int(np.sum(null_vals >= obs_maxz)) + 1) / (R + 1)
        rows.append({
            **dict(zip(cell_keys, keys)),
            "maxstat_obs_maxz": float(obs_maxz),
            "maxstat_p": p,
            "maxstat_r": R,
        })
    if not rows:
        return pl.DataFrame()

    out = pl.DataFrame(rows)
    p_arr = out["maxstat_p"].to_numpy()
    reject, q, _, _ = multipletests(p_arr, method="fdr_bh", alpha=alpha)
    out = out.with_columns([
        pl.Series("maxstat_q", q),
        pl.Series("maxstat_reject", reject),
    ]).sort(cell_keys)
    return out


def maxstat_floor_check(maxstat: pl.DataFrame, *, alpha: float = 0.05) -> dict:
    """Permutation-resolution diagnostic for `cell_maxstat_fdr_test`'s output.

    Mirrors `late_integration_maxstat_significance.py`'s floor check: a
    permutation p can resolve no finer than `1/(R+1)`. If the smallest
    per-cell p sits well above that floor and no cell is pinned there, R is
    sufficient and a null result is genuine, not a resolution artifact.

    This fork needs the check MORE than that notebook did: BH-FDR rejects
    the smallest p_(k) only if `p_(k) <= (k/m) * alpha`, so even a
    maximally-significant cell pinned at the floor (`p = 1/(R+1)`) survives
    rank-1 correction only if `1/(R+1) <= alpha/m`. Solve for m:
    `m <= (R+1)*alpha`. At R=1000, alpha=0.05, that's m <= 50 — a family of
    hundreds of all-SR cells can silently make it IMPOSSIBLE for BH-FDR to
    ever reject anything, no matter how strong the true effect, and a "0 in
    the new cell" partition result would look like confirmation when it's
    actually permutation censoring. `floor_limits_rejection=True` flags
    exactly that condition; the caller should surface it wherever a "0
    survivors" result gets reported as a finding, not bury it.

    Returns a dict: `floor` (best achievable p given R), `min_p` (smallest
    observed per-cell p), `n_at_floor` (cells pinned at the floor), `n_cells`,
    `rank1_bh_threshold` (`alpha / n_cells`, the p a rank-1 cell must clear),
    `floor_limits_rejection` (bool). All fields are `None`/`0`/`nan` as
    appropriate if `maxstat` is empty.
    """
    if maxstat.height == 0:
        return {
            "floor": float("nan"), "min_p": float("nan"), "n_at_floor": 0,
            "n_cells": 0, "rank1_bh_threshold": float("nan"),
            "floor_limits_rejection": None,
        }
    r_vals = maxstat["maxstat_r"].to_numpy().astype(float)
    p_vals = maxstat["maxstat_p"].to_numpy()
    floor_per_cell = 1.0 / (r_vals + 1.0)
    n_at_floor = int(np.sum(np.isclose(p_vals, floor_per_cell)))
    floor = float(np.min(floor_per_cell))
    min_p = float(np.min(p_vals))
    n_cells = int(maxstat.height)
    rank1_bh_threshold = alpha / n_cells
    return {
        "floor": floor,
        "min_p": min_p,
        "n_at_floor": n_at_floor,
        "n_cells": n_cells,
        "rank1_bh_threshold": rank1_bh_threshold,
        "floor_limits_rejection": floor > rank1_bh_threshold,
    }
