"""Tests for the all-speech-responsive perceptual fork's pure-Python core.

`src.causal46_joined.compute_sr_site_universe` and `.cell_maxstat_fdr_test`
back the notebooks in docs/superpowers/plans/2026-08-27-all-speech-responsive-perceptual.md
(`sr_site_universe.py`, `t_tests_all_sr.py`). Synthetic-only, no epoch/pipeline
data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.causal46_joined import cell_maxstat_fdr_test, compute_sr_site_universe, maxstat_floor_check


# ---------------------------------------------------------------------------
# compute_sr_site_universe
# ---------------------------------------------------------------------------


def _make_sr_by_subject() -> dict[str, pd.DataFrame]:
    return {
        "subjA": pd.DataFrame({
            "electrode_idx":     [1, 2, 3, 4],
            "speech_responsive": [True, True, False, True],
        }),
        "subjB": pd.DataFrame({
            "electrode_idx":     [1, 2],
            "speech_responsive": [True, False],
        }),
    }


def _make_phon_peaks() -> pl.DataFrame:
    return pl.DataFrame({
        "subject":       ["subjA", "subjA", "subjA", "subjA", "subjB"],
        "electrode_idx": [1, 1, 2, 4, 1],
        "phoneme_pair":  ["dn", "bm", "dn", "dn", "dn"],
        "p_value":       [0.001, 0.5, 0.5, 0.5, 0.5],
        "smin":          [40, 40, 40, 40, 40],
        "smax":          [60, 60, 60, 60, 60],
        "test_roc_auc":  [0.9, 0.6, 0.55, 0.5, 0.5],
    })


def test_universe_only_includes_speech_responsive_electrodes():
    """Electrode 3 (subjA) is not speech_responsive -> absent from the universe."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"], "subjB": ["dn"]},
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    assert out.filter(pl.col("electrode_idx") == 3).height == 0
    # subjB electrode 2 not speech_responsive -> absent
    assert out.filter((pl.col("subject") == "subjB") & (pl.col("electrode_idx") == 2)).height == 0


def test_universe_crosses_sr_electrodes_with_subject_specific_pairs():
    """Each SR electrode appears once per phoneme_pair the subject saw, no more."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"], "subjB": ["dn"]},
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    # subjA has SR electrodes {1, 2, 4} x pairs {dn, bm} = 6 rows
    assert out.filter(pl.col("subject") == "subjA").height == 6
    # subjB has SR electrode {1} x pairs {dn} = 1 row
    assert out.filter(pl.col("subject") == "subjB").height == 1


def test_annotation_is_a_label_not_a_filter():
    """acoustic_significant is a column, never drops rows: SR non-acoustic
    sites still appear, just flagged False with null phon_smin/phon_smax."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"], "subjB": ["dn"]},
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    row = out.filter(
        (pl.col("subject") == "subjA") & (pl.col("electrode_idx") == 2) & (pl.col("phoneme_pair") == "dn")
    )
    assert row.height == 1
    assert row["acoustic_significant"][0] is False
    assert row["phon_smin"][0] is None
    assert row["phon_smax"][0] is None


def test_acoustic_significant_true_only_below_threshold_for_that_cell():
    """subjA electrode 1, pair dn has p=0.001 < 0.01 -> significant, with
    phon_smin/phon_smax carried through. The SAME electrode's `bm` pair
    (p=0.5) is not -> false, independent of the dn result (no OR-across-pair
    collapse at this granularity, unlike compute_as_filter)."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"], "subjB": ["dn"]},
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    dn_row = out.filter(
        (pl.col("subject") == "subjA") & (pl.col("electrode_idx") == 1) & (pl.col("phoneme_pair") == "dn")
    )
    bm_row = out.filter(
        (pl.col("subject") == "subjA") & (pl.col("electrode_idx") == 1) & (pl.col("phoneme_pair") == "bm")
    )
    assert dn_row["acoustic_significant"][0] is True
    assert dn_row["phon_smin"][0] == 40
    assert dn_row["phon_smax"][0] == 60
    assert bm_row["acoustic_significant"][0] is False


def test_subject_with_no_phoneme_pairs_key_contributes_no_rows():
    """A subject missing from subject_phoneme_pairs cross-joins to nothing
    (defensive: never silently invents a pair)."""
    out = compute_sr_site_universe(
        _make_sr_by_subject(),
        {"subjA": ["dn", "bm"]},  # subjB omitted
        _make_phon_peaks(),
        ac_p_value_threshold=0.01,
    )
    assert out.filter(pl.col("subject") == "subjB").height == 0


def test_threshold_is_strict_inequality():
    out = compute_sr_site_universe(
        {"subjA": pd.DataFrame({"electrode_idx": [1], "speech_responsive": [True]})},
        {"subjA": ["dn"]},
        pl.DataFrame({
            "subject": ["subjA"], "electrode_idx": [1], "phoneme_pair": ["dn"],
            "p_value": [0.01], "smin": [40], "smax": [60], "test_roc_auc": [0.9],
        }),
        ac_p_value_threshold=0.01,
    )
    assert out["acoustic_significant"][0] is False


# ---------------------------------------------------------------------------
# cell_maxstat_fdr_test
#
# Mirrors late_integration_maxstat_significance.py's method: per (cell,
# window) z_obs = |mean_r(value)| / std_r(null); per-cell obs_maxz = max_w
# z_obs; per (cell, replicate) null_maxz = max_w |null| / std_r(null); p =
# (#{null_maxz >= obs_maxz} + 1) / (R + 1); BH-FDR across cells.
# ---------------------------------------------------------------------------

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]


def _cell_row(subject, electrode_idx, replicate, smin, smax, value, null):
    return {
        "subject": subject, "electrode_idx": electrode_idx,
        "phoneme_pair": "dn", "word_end": "necessary",
        "replicate": replicate, "smin": smin, "smax": smax,
        "mean_diff_raw": value, "mean_diff_raw_null": null,
    }


def test_maxstat_empty_boot_returns_empty():
    assert cell_maxstat_fdr_test(pl.DataFrame(), CELL_KEYS).height == 0


def test_maxstat_p_is_never_exactly_zero():
    """The unbiased permutation p = (k+1)/(R+1) has a floor of 1/(R+1) — it
    must never report exactly 0, even when the observed statistic dwarfs
    every null replicate."""
    rng = np.random.default_rng(0)
    n_rep = 40
    rows = [
        _cell_row("S", 1, r, 0, 10, value=100.0 + 0.01 * r, null=float(rng.normal(0, 0.1)))
        for r in range(n_rep)
    ]
    boot = pl.DataFrame(rows)
    out = cell_maxstat_fdr_test(boot, CELL_KEYS)
    assert out.height == 1
    assert out["maxstat_p"][0] == pytest.approx(1.0 / (n_rep + 1))
    assert out["maxstat_p"][0] > 0.0
    assert out["maxstat_reject"][0] is True


def test_maxstat_null_like_effect_does_not_survive():
    """When the observed values ARE the null distribution (same generating
    process, no true effect), obs_maxz should land near the typical null
    replicate's max — p should be large, not tiny, and BH-FDR shouldn't
    reject."""
    rng = np.random.default_rng(1)
    n_rep = 200
    null_vals = rng.normal(0, 1.0, size=n_rep)
    rows = [
        _cell_row("S", 1, r, 0, 10, value=float(null_vals[r]), null=float(null_vals[r]))
        for r in range(n_rep)
    ]
    boot = pl.DataFrame(rows)
    out = cell_maxstat_fdr_test(boot, CELL_KEYS)
    assert out.height == 1
    assert out["maxstat_p"][0] > 0.3
    assert out["maxstat_reject"][0] is False


def test_maxstat_takes_max_over_windows_not_first_window():
    """A weak first window paired with a strong second window: the max over
    windows must pick up the strong one when computing obs_maxz."""
    rng = np.random.default_rng(2)
    n_rep = 40
    rows = []
    for r in range(n_rep):
        rows.append(_cell_row("S", 1, r, 0, 10, value=0.001, null=float(rng.normal(0, 0.05))))
        rows.append(_cell_row("S", 1, r, 10, 20, value=20.0 + 0.01 * r, null=float(rng.normal(0, 0.05))))
    boot = pl.DataFrame(rows)
    out = cell_maxstat_fdr_test(boot, CELL_KEYS)
    assert out.height == 1
    assert out["maxstat_reject"][0] is True


def test_maxstat_computes_independently_per_cell_before_fdr():
    """Two cells, one with a strong effect and one null-like: the strong
    cell's smaller p must not be inflated to match the null one's, and vice
    versa (BH-FDR reorders by rank, it doesn't average across cells)."""
    rng = np.random.default_rng(3)
    n_rep = 60
    rows = []
    for r in range(n_rep):
        rows.append(_cell_row("S", 1, r, 0, 10, value=50.0 + 0.01 * r, null=float(rng.normal(0, 0.1))))
        null_v = float(rng.normal(0, 1.0))
        rows.append(_cell_row("S", 2, r, 0, 10, value=null_v, null=null_v))
    boot = pl.DataFrame(rows)
    out = cell_maxstat_fdr_test(boot, CELL_KEYS).sort("electrode_idx")
    assert out["maxstat_reject"].to_list() == [True, False]
    assert out["maxstat_p"][0] < out["maxstat_p"][1]


def test_maxstat_bh_fdr_is_more_conservative_than_raw_p():
    """With one true effect buried among many null cells, BH-FDR should
    inflate q above the raw per-cell p for at least the weaker cells (the
    multiple-comparisons correction the naive best-window CI lacks)."""
    rng = np.random.default_rng(4)
    # R must be large enough that the true cell's permutation-floor p
    # (1/(R+1)) survives BH-FDR across all cells (needs p <~ alpha/n_cells);
    # too few replicates and even a real effect can't out-resolve the floor.
    n_rep, n_null_cells = 500, 15
    rows = []
    for r in range(n_rep):
        rows.append(_cell_row("S", 0, r, 0, 10, value=30.0 + 0.01 * r, null=float(rng.normal(0, 0.1))))
    for cell in range(1, n_null_cells + 1):
        for r in range(n_rep):
            null_v = float(rng.normal(0, 1.0))
            rows.append(_cell_row("S", cell, r, 0, 10, value=null_v, null=null_v))
    boot = pl.DataFrame(rows)
    out = cell_maxstat_fdr_test(boot, CELL_KEYS)
    assert out.height == n_null_cells + 1
    # BH-FDR must never make q smaller than the raw p it's correcting.
    assert (out["maxstat_q"] >= out["maxstat_p"]).all()
    # Only the true-effect cell (electrode_idx=0) should survive.
    survivors = out.filter(pl.col("maxstat_reject"))["electrode_idx"].to_list()
    assert survivors == [0]


def test_maxstat_drops_cells_with_zero_null_variance():
    """A cell whose null is constant across every replicate at every window
    (std=0) has undefined z — it must be dropped, not divide-by-zero into a
    spurious result (matches late_integration_maxstat_significance.py's
    `sd_null > 0` filter)."""
    rows = [
        _cell_row("S", 1, r, 0, 10, value=5.0, null=0.0)
        for r in range(10)
    ]
    boot = pl.DataFrame(rows)
    out = cell_maxstat_fdr_test(boot, CELL_KEYS)
    assert out.height == 0


# ---------------------------------------------------------------------------
# maxstat_floor_check
#
# The failure mode this guards: BH-FDR rejects a rank-1 p only if
# p <= alpha/n_cells. A permutation p floors at 1/(R+1), so if
# 1/(R+1) > alpha/n_cells, NO cell can ever survive BH-FDR regardless of
# true effect size — a "0 survivors" partition result would look like
# confirmation when it's actually permutation censoring.
# ---------------------------------------------------------------------------


def test_floor_check_empty_input():
    out = maxstat_floor_check(pl.DataFrame())
    assert out["n_cells"] == 0
    assert out["floor_limits_rejection"] is None


def test_floor_check_flags_small_family_where_floor_dominates():
    """R=40 -> floor=1/41≈0.0244. With only 5 cells, alpha/m=0.01 < floor:
    even a floor-pinned cell can't survive BH-FDR. Must be flagged."""
    maxstat = pl.DataFrame({
        "subject": ["S"] * 5, "electrode_idx": list(range(5)),
        "maxstat_p": [1 / 41] * 5,   # every cell pinned at the floor
        "maxstat_r": [40] * 5,
    })
    out = maxstat_floor_check(maxstat, alpha=0.05)
    assert out["n_cells"] == 5
    assert out["n_at_floor"] == 5
    assert out["floor"] == pytest.approx(1 / 41)
    assert out["rank1_bh_threshold"] == pytest.approx(0.05 / 5)
    assert out["floor_limits_rejection"] is True


def test_floor_check_clears_when_family_small_relative_to_R():
    """Same R=40 floor, but only 1 cell in the family: alpha/m=0.05 > floor
    (0.0244) — a floor-pinned cell COULD survive BH-FDR. Not flagged."""
    maxstat = pl.DataFrame({
        "subject": ["S"], "electrode_idx": [1],
        "maxstat_p": [1 / 41],
        "maxstat_r": [40],
    })
    out = maxstat_floor_check(maxstat, alpha=0.05)
    assert out["floor_limits_rejection"] is False


def test_floor_check_at_production_scale_R1000():
    """R=1000 -> floor≈0.001. Matches the advisor's worked example: family
    of ~200 cells (all-SR scale) with alpha=0.05 -> alpha/m=0.00025 < floor
    -> flagged, even though R=1000 is the production default."""
    n_cells = 200
    maxstat = pl.DataFrame({
        "subject": ["S"] * n_cells,
        "electrode_idx": list(range(n_cells)),
        "maxstat_p": [1 / 1001] * n_cells,
        "maxstat_r": [1000] * n_cells,
    })
    out = maxstat_floor_check(maxstat, alpha=0.05)
    assert out["floor"] == pytest.approx(1 / 1001)
    assert out["rank1_bh_threshold"] == pytest.approx(0.05 / 200)
    assert out["floor_limits_rejection"] is True


def test_floor_check_min_p_not_at_floor_when_r_is_ample():
    """A genuinely null family (p values spread well above the floor, none
    pinned) should not be flagged — R is adequate."""
    rng = np.random.default_rng(7)
    p_vals = rng.uniform(0.1, 0.9, size=30)
    maxstat = pl.DataFrame({
        "subject": ["S"] * 30, "electrode_idx": list(range(30)),
        "maxstat_p": p_vals.tolist(),
        "maxstat_r": [1000] * 30,
    })
    out = maxstat_floor_check(maxstat, alpha=0.05)
    assert out["n_at_floor"] == 0
    assert out["min_p"] > out["floor"]
