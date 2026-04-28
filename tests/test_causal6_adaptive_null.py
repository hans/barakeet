"""
Unit tests for src.models.causal6_adaptive_null.

Synthetic-only; no real data, GPU, or torch required. Mirrors the
helper-style of tests/test_significance.py.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.causal6_adaptive_null import (
    filter_null_to_borderline,
    min_pointwise_p_per_site,
    stage1_gate,
)
from src.models.causal6_aggregates import FlavorSpec
from src.models.significance import null_standardized_peak_test


SITE_KEYS = ["subject", "electrode_idx"]
WINDOW_KEYS = ["smin", "smax"]


def _make_agg(
    real: np.ndarray,             # (S, W) real per (site, window)
    null: np.ndarray,             # (S, K, W) null per (site, perm, window)
    *,
    site_ids: list[tuple],        # length S; (subject, electrode_idx)
    stat_col: str = "statistic",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build pre-aggregated (real_agg, null_agg) DataFrames."""
    S, W = real.shape
    assert null.shape[0] == S and null.shape[2] == W
    K = null.shape[1]
    windows = [(w, w + 10) for w in range(W)]

    real_rows = []
    null_rows = []
    for s_i in range(S):
        subj, eidx = site_ids[s_i]
        for w_i in range(W):
            real_rows.append({
                "subject": subj,
                "electrode_idx": eidx,
                "smin": windows[w_i][0],
                "smax": windows[w_i][1],
                stat_col: float(real[s_i, w_i]),
            })
            for k in range(K):
                null_rows.append({
                    "subject": subj,
                    "electrode_idx": eidx,
                    "smin": windows[w_i][0],
                    "smax": windows[w_i][1],
                    "permutation_idx": k,
                    stat_col: float(null[s_i, k, w_i]),
                })
    return pl.DataFrame(real_rows), pl.DataFrame(null_rows)


# =============================================================================
# min_pointwise_p_per_site
# =============================================================================


def test_min_pointwise_p_per_site_basic():
    """3 sites × 5 windows × K=20 nulls. Hand-compute pointwise_p per
    (site, window), take min, compare to function output.
    """
    S, W, K = 3, 5, 20
    real = np.array([
        # site 0: w=2 most extreme — p=1/(K+1) at w=2.
        [0.51, 0.52, 0.99, 0.50, 0.49],
        # site 1: w=0 most extreme.
        [0.95, 0.55, 0.50, 0.45, 0.40],
        # site 2: w=3 dominates.
        [0.40, 0.45, 0.48, 0.95, 0.50],
    ])
    rng = np.random.default_rng(0)
    null = 0.5 + 0.05 * rng.standard_normal((S, K, W))

    real_agg, null_agg = _make_agg(
        real, null,
        site_ids=[("S0", 0), ("S1", 1), ("S2", 2)],
    )
    out = min_pointwise_p_per_site(
        real_agg, null_agg,
        site_keys=SITE_KEYS, window_keys=WINDOW_KEYS,
    )

    assert out.height == S
    out = out.sort("electrode_idx")

    # Expected pointwise_p per (site, window): (#{null >= real} + 1) / (K + 1)
    expected_min = np.empty(S)
    expected_argmin = np.empty(S, dtype=int)
    for s_i in range(S):
        ps = np.empty(W)
        for w_i in range(W):
            ge = int((null[s_i, :, w_i] >= real[s_i, w_i]).sum())
            ps[w_i] = (ge + 1) / (K + 1)
        expected_min[s_i] = ps.min()
        expected_argmin[s_i] = int(np.argmin(ps))

    # min_pointwise_p
    assert np.allclose(out["min_pointwise_p"].to_numpy(), expected_min)
    # argmin_smin = window index since windows = (w, w+10)
    assert out["argmin_smin"].to_list() == expected_argmin.tolist()
    # K_at_argmin = 20 everywhere (no NaN)
    assert (out["K_at_argmin"].to_numpy() == K).all()
    # n_windows = 5
    assert (out["n_windows"].to_numpy() == W).all()


def test_min_pointwise_p_per_site_handles_nan():
    """NaN null entries dropped per-window before ranking; K_w reflects
    only valid entries (matches significance.py:115-117). All-NaN
    windows still appear in the output with a degenerate p=1.
    """
    S, W, K = 1, 3, 10
    # Tuned so window 0 is the argmin AFTER 4 NaNs are dropped:
    #   w0: real=0.7, K=6, ge=0 → p = 1/7 ≈ 0.143
    #   w1: real=0.4, K=10, ge=10 (ties count) → p = 11/11 = 1.0
    #   w2: all-NaN, K=0 → p = 1/1 = 1.0 (degenerate)
    real = np.array([[0.7, 0.4, 0.7]])
    null = np.full((S, K, W), 0.4)
    null[0, :4, 0] = np.nan
    null[0, :, 2] = np.nan

    real_agg, null_agg = _make_agg(real, null, site_ids=[("S0", 0)])
    out = min_pointwise_p_per_site(
        real_agg, null_agg,
        site_keys=SITE_KEYS, window_keys=WINDOW_KEYS,
    )
    row = out.to_dicts()[0]
    assert row["argmin_smin"] == 0
    assert row["K_at_argmin"] == 6
    assert row["min_pointwise_p"] == pytest.approx(1 / 7)
    assert row["n_windows"] == 3


# =============================================================================
# stage1_gate
# =============================================================================


def _make_agg_with_flavor_cols(
    fold_mean_real: np.ndarray,
    fold_mean_null: np.ndarray,
    *,
    site_ids: list[tuple],
    t_stat_real: np.ndarray | None = None,
    t_stat_null: np.ndarray | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build aggregator-output style frames carrying both fold_mean and
    t_stat columns, matching what aggregate_<decoder>(...) returns.
    """
    if t_stat_real is None:
        t_stat_real = fold_mean_real * 10  # arbitrary scaling so they aren't identical
    if t_stat_null is None:
        t_stat_null = fold_mean_null * 10

    S, W = fold_mean_real.shape
    K = fold_mean_null.shape[1]
    windows = [(w, w + 10) for w in range(W)]

    real_rows = []
    null_rows = []
    for s_i in range(S):
        subj, eidx = site_ids[s_i]
        for w_i in range(W):
            real_rows.append({
                "subject": subj,
                "electrode_idx": eidx,
                "smin": windows[w_i][0],
                "smax": windows[w_i][1],
                "fold_mean": float(fold_mean_real[s_i, w_i]),
                "t_stat": float(t_stat_real[s_i, w_i]),
            })
            for k in range(K):
                null_rows.append({
                    "subject": subj,
                    "electrode_idx": eidx,
                    "smin": windows[w_i][0],
                    "smax": windows[w_i][1],
                    "permutation_idx": k,
                    "fold_mean": float(fold_mean_null[s_i, k, w_i]),
                    "t_stat": float(t_stat_null[s_i, k, w_i]),
                })
    return pl.DataFrame(real_rows), pl.DataFrame(null_rows)


def test_stage1_gate_returns_borderline_set():
    """4 sites with min_pointwise_p approximately 0.05 / 0.15 / 0.25 /
    0.50; gate at p_max=0.20 → expect borderline = {site0, site1}.
    Uses a single non-TFCE flavor for simplicity.
    """
    S, W, K = 4, 3, 19  # 1/(K+1) = 0.05; need exact-rank values
    rng = np.random.default_rng(123)
    null = rng.uniform(0.0, 1.0, size=(S, K, W))

    # For each site, control real[w=0] so its rank yields a target p.
    # pointwise_p = (ge_count + 1) / (K + 1), so ge_count = round(p*(K+1))-1.
    # Place real strictly between the (ge)-th and (ge+1)-th largest null
    # values so exactly ge entries are >= real.
    targets = [0.05, 0.15, 0.25, 0.50]
    target_ge_count = [int(round(t * (K + 1))) - 1 for t in targets]
    real = np.zeros((S, W))
    for s_i in range(S):
        sorted_null = np.sort(null[s_i, :, 0])  # ascending
        ge = target_ge_count[s_i]
        if ge == 0:
            real[s_i, 0] = sorted_null[-1] + 1.0
        else:
            # real strictly between (ge+1)-th and ge-th from top → ge_count=ge.
            real[s_i, 0] = (sorted_null[K - ge - 1] + sorted_null[K - ge]) / 2.0
        # Other windows: real=0 in [0, 1] uniform null → ge_count=K, p≈1, so
        # window 0 dominates the per-site min.
        for w_i in range(1, W):
            real[s_i, w_i] = 0.0

    # Make sure no other window has lower p (real=0 → ge_count = K → p=1).
    real_agg, null_agg = _make_agg_with_flavor_cols(
        real, null,
        site_ids=[("S0", 0), ("S1", 1), ("S2", 2), ("S3", 3)],
    )

    flavors = [FlavorSpec("fold_mean", apply_tfce=False)]
    borderline_keys, gate_log = stage1_gate(
        real_agg, null_agg,
        site_keys=SITE_KEYS, flavors=flavors, p_max=0.20,
    )
    gate_log = gate_log.sort("electrode_idx")
    actual_min = gate_log["min_pointwise_p_global"].to_numpy()
    np.testing.assert_allclose(actual_min, targets, atol=0.001)

    assert borderline_keys == {("S0", 0), ("S1", 1)}
    escalated = gate_log["escalated"].to_list()
    assert escalated == [True, True, False, False]


def test_stage1_gate_includes_tfce_flavors():
    """A site with every per-window pointwise_p > p_max but a wide,
    spatially-coherent plateau → TFCE-enhanced flavor must drive
    escalation that the raw flavor misses.

    Construction is hand-crafted to keep the assertion deterministic:
      * Null per (perm, window) alternates ±0.4 (no spatial coherence;
        every null cluster has length 1).
      * Real site 0: full-W plateau at h=0.3. Below the null peak ±0.4
        so individual pointwise_p ≈ 0.5 > 0.20.
      * Real site 1: flat 0 (control — should not escalate).

    Real TFCE on the plateau (extent=W, h=0.3) ≈ W^0.5 * h² / 100 dominates
    the null TFCE, which is bounded by the length-1 cluster contribution
    1^0.5 * 0.4² / 100. Real TFCE > all null TFCE → pointwise_p_TFCE
    hits the floor 1/(K+1).
    """
    S, W = 2, 20
    K = 19

    # Null pattern: alternates ±0.4 by parity of (k + w). No spatial cluster
    # ever exceeds length 1.
    null = np.zeros((S, K, W))
    for s in range(S):
        for k in range(K):
            for w in range(W):
                null[s, k, w] = 0.4 if (k + w) % 2 == 0 else -0.4

    real = np.zeros((S, W))
    real[0, :] = 0.3              # full-W plateau just below null peak
    real[1, :] = 0.0              # control

    real_agg, null_agg = _make_agg_with_flavor_cols(
        real, null, site_ids=[("S0", 0), ("S1", 1)],
    )

    # Raw flavor only: site 0 must NOT escalate (individual p ≈ 0.5).
    raw_only = [FlavorSpec("fold_mean", apply_tfce=False)]
    borderline_raw, gate_raw = stage1_gate(
        real_agg, null_agg,
        site_keys=SITE_KEYS, flavors=raw_only, p_max=0.20,
    )
    s0_min_raw = gate_raw.filter(pl.col("electrode_idx") == 0)[
        "min_pointwise_p_global"
    ][0]
    assert s0_min_raw > 0.20, (
        f"raw min_pointwise_p for site 0 should be > 0.20; got {s0_min_raw}"
    )
    assert ("S0", 0) not in borderline_raw

    # Add TFCE flavor → site 0 must now escalate.
    flavors = [
        FlavorSpec("fold_mean", apply_tfce=False),
        FlavorSpec("fold_mean", apply_tfce=True, tfce_threshold=0.0),
    ]
    borderline, gate = stage1_gate(
        real_agg, null_agg,
        site_keys=SITE_KEYS, flavors=flavors, p_max=0.20,
    )
    s0_row = gate.filter(pl.col("electrode_idx") == 0).to_dicts()[0]
    assert s0_row["min_pointwise_p_fold_mean_tfce"] <= 0.20, (
        f"TFCE-enhanced min_pointwise_p for broad cluster should be ≤ 0.20; "
        f"got {s0_row['min_pointwise_p_fold_mean_tfce']}"
    )
    assert s0_row["argmin_flavor"] == "fold_mean_tfce"
    assert ("S0", 0) in borderline
    # Site 1 (flat control) must NOT escalate even with TFCE.
    assert ("S1", 1) not in borderline


# =============================================================================
# filter_null_to_borderline
# =============================================================================


def test_filter_null_to_borderline_keeps_only_named_site_keys():
    """site_keys includes phoneme_pair; rows with electrode_idx in the
    coarse borderline electrode-set but phoneme_pair NOT in
    borderline_keys must be dropped."""
    site_keys = ["subject", "electrode_idx", "phoneme_pair"]
    rows = []
    for eidx in (0, 1, 2):
        for pp in ("dn", "bm", "pb"):
            for k in range(3):
                rows.append({
                    "subject": "S0", "electrode_idx": eidx,
                    "phoneme_pair": pp, "smin": 0, "smax": 10,
                    "permutation_idx": k, "test_roc_auc": 0.5,
                })
    null_scores = pl.DataFrame(rows)

    # Only flag (S0, 0, dn) and (S0, 1, bm). Note electrodes 0 and 1 each
    # contain non-borderline phoneme_pair rows that must be dropped.
    borderline = {
        ("S0", 0, "dn"),
        ("S0", 1, "bm"),
    }
    out = filter_null_to_borderline(
        null_scores, borderline, site_keys=site_keys,
    )
    kept = set(
        out.select(site_keys).unique().iter_rows()
    )
    assert kept == borderline
    # All 3 perms × 2 keys = 6 rows.
    assert out.height == 6


def test_filter_null_to_borderline_empty_set():
    """Empty borderline_keys → empty output of same schema."""
    null_scores = pl.DataFrame({
        "subject": ["S0"], "electrode_idx": [0], "smin": [0], "smax": [10],
        "permutation_idx": [0], "test_roc_auc": [0.5],
    })
    out = filter_null_to_borderline(
        null_scores, set(), site_keys=["subject", "electrode_idx"],
    )
    assert out.height == 0
    assert out.columns == null_scores.columns


# =============================================================================
# end-to-end: two-stage merge equals flat-K
# =============================================================================


def test_two_stage_p_value_equals_flat_K():
    """Generate K_total perms, run gate at K1 with p_max=1.0 (escalates
    everything), take stage 2 = remaining K2 perms, merge, pass through
    null_standardized_peak_test. Verify p_value bit-identical to
    feeding all K_total perms directly. Proves the merge preserves
    semantics + the gate primitive doesn't drop rows.
    """
    K1, K2 = 30, 70
    K_total = K1 + K2
    S, W = 3, 4
    rng = np.random.default_rng(2026)
    null = rng.standard_normal((S, K_total, W))
    real = rng.standard_normal((S, W))

    site_ids = [("S0", 0), ("S1", 1), ("S2", 2)]
    real_agg, null_agg_full = _make_agg(
        real, null, site_ids=site_ids,
    )

    # Stage-1 slice: perms 0..K1.
    null_stage1 = null_agg_full.filter(pl.col("permutation_idx") < K1)
    flavors = [FlavorSpec("statistic", apply_tfce=False)]
    # Wrap the stat column under both names so stage1_gate sees a
    # FlavorSpec("statistic", ...) consistent with our test input.
    real_for_gate = real_agg.rename({"statistic": "statistic"})
    null_stage1_for_gate = null_stage1.rename({"statistic": "statistic"})
    borderline, gate_log = stage1_gate(
        real_for_gate, null_stage1_for_gate,
        site_keys=SITE_KEYS, flavors=flavors, p_max=1.0,
    )
    # p_max=1.0 → every site escalates.
    assert len(borderline) == S
    assert gate_log["escalated"].all()

    # Stage-2 slice: perms K1..K_total (next non-overlapping seed range).
    null_stage2 = null_agg_full.filter(pl.col("permutation_idx") >= K1)
    null_stage2_filtered = filter_null_to_borderline(
        null_stage2, borderline, site_keys=SITE_KEYS,
    )
    null_merged = pl.concat([null_stage1, null_stage2_filtered])
    assert null_merged.height == null_agg_full.height

    flat_peaks, _ = null_standardized_peak_test(
        real_agg, null_agg_full, site_keys=SITE_KEYS,
    )
    merged_peaks, _ = null_standardized_peak_test(
        real_agg, null_merged, site_keys=SITE_KEYS,
    )

    flat = flat_peaks.sort(SITE_KEYS).to_dicts()
    merged = merged_peaks.sort(SITE_KEYS).to_dicts()
    for f, m in zip(flat, merged):
        assert f["p_value"] == m["p_value"], (
            f"flat={f['p_value']}, merged={m['p_value']} for site {f['subject']}"
        )
        assert f["pointwise_p"] == m["pointwise_p"]
        assert f["peak_smin"] == m["peak_smin"]
        assert f["n_permutations"] == m["n_permutations"] == K_total
