"""
Unit tests for src.models.causal6_adaptive_null.

Synthetic-only; no real data, GPU, or torch required. Mirrors the
helper-style of tests/test_significance.py.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl
import pytest

from src.models.causal6_adaptive_null import (
    filter_null_to_borderline,
    min_pointwise_p_per_site,
    stage1_gate,
)
from src.models.causal6_aggregates import (
    FlavorSpec,
    SITE_KEYS_BEHAVIOR_WITH_CONTROL,
    SITE_KEYS_GANONG_WITH_CONTROL,
    preagg_behavior_with_control_null,
    preagg_ganong_with_control_null,
)
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
    """4 sites; site i has real placed at the (i+1)-th rank of its
    K=19-perm null distribution. With a single window the max-stat
    corrected p reduces to ``(rank + 1) / (K + 1)``: targets become
    0.10 / 0.15 / 0.25 / 0.45. At p_max=0.20 → borderline = {site0, site1}.
    Uses a single non-TFCE flavor.
    """
    S, W, K = 4, 1, 19
    rng = np.random.default_rng(123)
    null = rng.uniform(0.0, 1.0, size=(S, K, W))

    # Place real strictly between the rank-th and (rank+1)-th largest
    # null entries so exactly ``rank`` null values are >= real.
    target_ranks = [0, 1, 3, 7]  # corrected_p = (rank+1+1)/(K+1)
    real = np.zeros((S, W))
    for s_i in range(S):
        sorted_null = np.sort(null[s_i, :, 0])  # ascending
        rank = target_ranks[s_i]
        if rank == 0:
            real[s_i, 0] = sorted_null[-1] + 1.0
        else:
            real[s_i, 0] = (sorted_null[K - rank - 1] + sorted_null[K - rank]) / 2.0

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
    actual = gate_log["min_corrected_p_global"].to_numpy()
    expected = np.array([(r + 2) / (K + 1) for r in target_ranks])
    np.testing.assert_allclose(actual, expected, atol=1e-9)

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
        so individual pointwise_p ≈ 0.5 → raw corrected_p stays high
        because every perm's max-stat sits at the same level.
      * Real site 1: flat 0 (control — should not escalate).

    Real TFCE on the plateau (extent=W, h=0.3) dominates the null TFCE
    (bounded by length-1 cluster contributions). Real TFCE > all null
    TFCE at every window → pointwise_p_real_TFCE = 1/(K+1) at every
    window, T_obs ≫ T_null[k] for all k, corrected p hits 1/(K+1).
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

    # Raw flavor only: site 0 must NOT escalate (corrected p ≫ 0.20).
    raw_only = [FlavorSpec("fold_mean", apply_tfce=False)]
    borderline_raw, gate_raw = stage1_gate(
        real_agg, null_agg,
        site_keys=SITE_KEYS, flavors=raw_only, p_max=0.20,
    )
    s0_corrected_raw = gate_raw.filter(pl.col("electrode_idx") == 0)[
        "min_corrected_p_global"
    ][0]
    assert s0_corrected_raw > 0.20, (
        f"raw corrected_p for site 0 should be > 0.20; got {s0_corrected_raw}"
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
    assert s0_row["corrected_p_fold_mean_tfce"] <= 0.20, (
        f"TFCE-enhanced corrected_p for broad cluster should be ≤ 0.20; "
        f"got {s0_row['corrected_p_fold_mean_tfce']}"
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

    # LazyFrame path must produce the same result. Stage-2 spill mode
    # scans parquet shards and applies this filter lazily, so the
    # collected output has to match the eager path row-for-row.
    lazy_out = filter_null_to_borderline(
        null_scores.lazy(), borderline, site_keys=site_keys,
    )
    assert isinstance(lazy_out, pl.LazyFrame)
    collected = lazy_out.collect()
    assert collected.height == out.height
    lazy_kept = set(collected.select(site_keys).unique().iter_rows())
    assert lazy_kept == borderline


def test_stage1_gate_handles_float32_stat_inputs():
    """Regression: GPU permutation kernels return ``test_roc_auc`` as
    Float32, so ``fold_tstat_aggregate`` produces a Float32 ``fold_mean``
    column while ``t_stat`` and TFCE-enhanced statistics are Float64.
    Without dtype normalization inside ``stage1_gate``, ``pl.concat``
    over per-flavor frames raises ``SchemaError: type Float64 is
    incompatible with expected type Float32``.
    """
    S, W, K = 2, 4, 19
    rng = np.random.default_rng(11)
    fold_mean_real = rng.uniform(0.45, 0.7, size=(S, W)).astype(np.float32)
    fold_mean_null = rng.uniform(0.45, 0.7, size=(S, K, W)).astype(np.float32)
    # t_stat in real pipelines is Float64 (Float32 / Float64 promotion).
    t_stat_real = (fold_mean_real.astype(np.float64) - 0.5) * 10
    t_stat_null = (fold_mean_null.astype(np.float64) - 0.5) * 10

    real_agg, null_agg = _make_agg_with_flavor_cols(
        fold_mean_real, fold_mean_null,
        site_ids=[("S0", 0), ("S1", 1)],
        t_stat_real=t_stat_real, t_stat_null=t_stat_null,
    )
    # Force fold_mean back to Float32 (the helper coerced via float()).
    real_agg = real_agg.with_columns(pl.col("fold_mean").cast(pl.Float32))
    null_agg = null_agg.with_columns(pl.col("fold_mean").cast(pl.Float32))
    assert real_agg.schema["fold_mean"] == pl.Float32
    assert real_agg.schema["t_stat"] == pl.Float64

    # All four BEHAVIOR_HGA_ONLY flavors mix raw + TFCE on both columns —
    # exactly the configuration that tripped the GPU run.
    flavors = [
        FlavorSpec("fold_mean", apply_tfce=False),
        FlavorSpec("t_stat", apply_tfce=False),
        FlavorSpec("fold_mean", apply_tfce=True, tfce_threshold=0.5),
        FlavorSpec("t_stat", apply_tfce=True),
    ]
    borderline, gate_log = stage1_gate(
        real_agg, null_agg,
        site_keys=SITE_KEYS, flavors=flavors, p_max=0.20,
    )
    assert gate_log.height == S
    expected_cols = {
        "corrected_p_fold_mean", "corrected_p_t_stat",
        "corrected_p_fold_mean_tfce", "corrected_p_t_stat_tfce",
        "min_corrected_p_global", "argmin_flavor",
        "peak_smin", "peak_smax",
        "real_at_peak", "n_permutations", "escalated",
    } | set(SITE_KEYS)
    assert expected_cols.issubset(set(gate_log.columns))


# -----------------------------------------------------------------------------
# Paired-decoder baseline-row handling
# -----------------------------------------------------------------------------


def _make_paired_spill(
    *,
    electrode_idxs: Sequence[int] = (0, 1),
    phoneme_pairs: Sequence[str] = ("dn", "bm"),
    word_ends: Sequence[str] = ("necessary",),
    n_perms: int = 2,
    n_folds: int = 2,
    smins: Sequence[int] = (0, 5),
    auc_seed: int = 0,
    include_word_end: bool = True,
) -> pl.DataFrame:
    """Build a synthetic spill-like null frame with model='full' and
    model='baseline' rows, mirroring what
    ``_run_behavior_core_permutations`` writes when ``with_control=True``.

    Baseline rows use the same sentinels production code uses:
    electrode_idx=-1, smin=-1, smax=-1.
    """
    rng = np.random.default_rng(auc_seed)
    rows: list[dict] = []
    for pp in phoneme_pairs:
        for we in word_ends:
            for eidx in electrode_idxs:
                for smin in smins:
                    for perm in range(n_perms):
                        for fold in range(n_folds):
                            row = {
                                "subject": "S0",
                                "electrode_idx": eidx,
                                "phoneme_pair": pp,
                                "smin": smin,
                                "smax": smin + 10,
                                "fold": fold,
                                "permutation_idx": perm,
                                "test_roc_auc": float(rng.uniform(0.4, 0.7)),
                                "model": "full",
                            }
                            if include_word_end:
                                row["word_end"] = we
                            rows.append(row)
            # Baseline shared across electrodes/windows; sentinel rows.
            for perm in range(n_perms):
                for fold in range(n_folds):
                    row = {
                        "subject": "S0",
                        "electrode_idx": -1,
                        "phoneme_pair": pp,
                        "smin": -1,
                        "smax": -1,
                        "fold": fold,
                        "permutation_idx": perm,
                        "test_roc_auc": float(rng.uniform(0.4, 0.6)),
                        "model": "baseline",
                    }
                    if include_word_end:
                        row["word_end"] = we
                    rows.append(row)
    return pl.DataFrame(rows)


def test_filter_null_to_borderline_retains_baseline_rows_behavior_with_control():
    """When ``baseline_site_keys`` is provided, baseline rows (model='baseline',
    sentinel electrode_idx=-1) must be retained for every (subject,
    phoneme_pair, word_end) tuple represented in borderline_keys — not
    dropped by the site_keys semi-join.

    Regression: previously the semi-join keyed on site_keys (which includes
    electrode_idx) silently dropped every baseline row, because baseline
    electrode_idx=-1 is never in any borderline tuple. Downstream
    ``_pair_full_baseline`` left-join then produced NULL baseline_roc_auc
    → NULL diff → NULL fold_mean_diff, and ``null_standardized_peak_test``
    silently filters NULL/NaN rows out — so escalated sites ran at
    K=K1 instead of K1+K2.
    """
    spill = _make_paired_spill()
    site_keys = SITE_KEYS_BEHAVIOR_WITH_CONTROL

    # Borderline: only (S0, 0, dn, necessary). Electrode 1 not escalated.
    # phoneme_pair=bm not escalated for any electrode.
    borderline = {("S0", 0, "dn", "necessary")}

    out = filter_null_to_borderline(
        spill, borderline,
        site_keys=site_keys,
        baseline_site_keys=["subject", "phoneme_pair", "word_end"],
    )

    full_kept = out.filter(pl.col("model") == "full")
    full_sites = set(
        full_kept.select(["electrode_idx", "phoneme_pair", "word_end"])
        .unique().iter_rows()
    )
    assert full_sites == {(0, "dn", "necessary")}, (
        f"full-model rows leaked outside borderline tuples: {full_sites}"
    )

    base_kept = out.filter(pl.col("model") == "baseline")
    assert base_kept.height > 0, "baseline rows must be retained for paired decoders"
    base_sites = set(
        base_kept.select(["phoneme_pair", "word_end"]).unique().iter_rows()
    )
    # phoneme_pair=bm should be dropped (not in any borderline tuple after
    # stripping electrode_idx).
    assert base_sites == {("dn", "necessary")}, (
        f"baseline rows kept for non-borderline (pp, we): {base_sites}"
    )
    assert (base_kept["electrode_idx"] == -1).all()


def test_filter_null_to_borderline_retains_baseline_rows_ganong_with_control():
    """ganong_with_control: site_keys has no word_end, so baseline_site_keys
    is (subject, phoneme_pair). Same retention contract as behavior_with_control.
    """
    # No word_end dimension in ganong site_keys.
    spill = _make_paired_spill(word_ends=(None,), include_word_end=False)
    site_keys = SITE_KEYS_GANONG_WITH_CONTROL

    borderline = {("S0", 0, "dn"), ("S0", 1, "bm")}

    out = filter_null_to_borderline(
        spill, borderline,
        site_keys=site_keys,
        baseline_site_keys=["subject", "phoneme_pair"],
    )

    full_sites = set(
        out.filter(pl.col("model") == "full")
        .select(["electrode_idx", "phoneme_pair"]).unique().iter_rows()
    )
    assert full_sites == {(0, "dn"), (1, "bm")}

    base_sites = set(
        out.filter(pl.col("model") == "baseline")
        .select(["phoneme_pair"]).unique().iter_rows()
    )
    # Both phoneme_pairs appear in borderline (under different electrodes),
    # so both baselines must be kept.
    assert base_sites == {("dn",), ("bm",)}, (
        f"ganong baseline retention wrong: {base_sites}"
    )


def test_filter_null_to_borderline_lazy_path_with_baseline():
    """LazyFrame path must behave identically when baseline_site_keys is set —
    the production notebooks scan_parquet → filter → collect.
    """
    spill = _make_paired_spill()
    site_keys = SITE_KEYS_BEHAVIOR_WITH_CONTROL
    borderline = {("S0", 0, "dn", "necessary")}

    eager = filter_null_to_borderline(
        spill, borderline,
        site_keys=site_keys,
        baseline_site_keys=["subject", "phoneme_pair", "word_end"],
    )
    lazy = filter_null_to_borderline(
        spill.lazy(), borderline,
        site_keys=site_keys,
        baseline_site_keys=["subject", "phoneme_pair", "word_end"],
    )
    assert isinstance(lazy, pl.LazyFrame)
    collected = lazy.collect()
    assert collected.height == eager.height
    # Same model-row counts and same (model, electrode_idx, phoneme_pair, word_end)
    # tuples.
    eager_keys = set(
        eager.select(["model", "electrode_idx", "phoneme_pair", "word_end"])
        .unique().iter_rows()
    )
    lazy_keys = set(
        collected.select(["model", "electrode_idx", "phoneme_pair", "word_end"])
        .unique().iter_rows()
    )
    assert eager_keys == lazy_keys


def test_filter_then_preagg_behavior_with_control_no_null_diff():
    """End-to-end: filtered output fed through
    ``preagg_behavior_with_control_null`` must produce non-NULL
    ``fold_mean_diff`` for every row. This is the actual bug symptom:
    NULL diffs from a left-join on missing baselines propagate through
    to ``null_standardized_peak_test`` which silently drops them.
    """
    spill = _make_paired_spill(n_perms=3, n_folds=3, smins=(0, 5, 10))

    # Build a matching real_scores frame (same schema as spill but no
    # permutation_idx column).
    real_scores = spill.drop("permutation_idx").unique(
        subset=["subject", "electrode_idx", "phoneme_pair", "word_end",
                "smin", "smax", "fold", "model"]
    )

    borderline = {
        ("S0", 0, "dn", "necessary"),
        ("S0", 1, "bm", "necessary"),
    }
    filtered = filter_null_to_borderline(
        spill, borderline,
        site_keys=SITE_KEYS_BEHAVIOR_WITH_CONTROL,
        baseline_site_keys=["subject", "phoneme_pair", "word_end"],
    )
    preagg = preagg_behavior_with_control_null(filtered, real_scores)

    n_null = preagg["fold_mean_diff"].null_count()
    assert n_null == 0, (
        f"preagg produced {n_null}/{preagg.height} NULL fold_mean_diff rows; "
        f"baseline pairing failed downstream of filter_null_to_borderline"
    )
    # And the preagg must actually cover the borderline (site, window) tuples.
    assert preagg.height > 0


def test_filter_then_preagg_ganong_with_control_no_null_diff():
    """Same end-to-end check for ganong_with_control (no word_end in site_keys)."""
    spill = _make_paired_spill(
        word_ends=(None,), include_word_end=False, n_perms=3, n_folds=3, smins=(0, 5, 10),
    )
    real_scores = spill.drop("permutation_idx").unique(
        subset=["subject", "electrode_idx", "phoneme_pair",
                "smin", "smax", "fold", "model"]
    )

    borderline = {("S0", 0, "dn"), ("S0", 1, "bm")}
    filtered = filter_null_to_borderline(
        spill, borderline,
        site_keys=SITE_KEYS_GANONG_WITH_CONTROL,
        baseline_site_keys=["subject", "phoneme_pair"],
    )
    preagg = preagg_ganong_with_control_null(filtered, real_scores)
    n_null = preagg["fold_mean_diff"].null_count()
    assert n_null == 0, (
        f"ganong preagg produced {n_null}/{preagg.height} NULL fold_mean_diff rows"
    )


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
