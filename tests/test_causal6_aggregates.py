"""
Unit tests for preagg helpers and aggregate_* schema compatibility in
src.models.causal6_aggregates.

Synthetic-only; no real data or GPU required.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.causal6_aggregates import (
    SITE_KEYS_BEHAVIOR_HGA_ONLY,
    SITE_KEYS_BEHAVIOR_WITH_CONTROL,
    aggregate_behavior_hga_only,
    aggregate_behavior_with_control,
    preagg_behavior_hga_only_null,
    preagg_behavior_with_control_null,
    _preagg_hga_only_null,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

_SUBJECT = "EC000"
_EPOCH_TMIN = -0.4
_EPOCH_SFREQ = 100.0
_BEHAV_POST_OFFSET_S = 0.2

# Use a short, tight window range inside the behavior peak-search bounds.
# offset_samples for "desolate" = int((0.498 - (-0.4)) * 100 + 0.2 * 100) = 109.
# So smax must be ≤ 109. We use smin=0..4, smax=15..19 (step=1).
_WINDOWS = [(s, s + 15) for s in range(0, 5)]
_WORD_END = "desolate"


def _make_real_hga_only(
    n_electrodes: int = 2,
    n_folds: int = 3,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Per-fold real scores for behavior_hga_only."""
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []
    for ei in range(n_electrodes):
        for pp in ["dn", "bm"]:
            for smin, smax in _WINDOWS:
                for fold in range(n_folds):
                    auc = float(np.clip(rng.normal(0.6 if ei == 0 else 0.5, 0.05), 0.0, 1.0))
                    rows.append({
                        "subject": _SUBJECT,
                        "electrode_idx": ei,
                        "phoneme_pair": pp,
                        "word_end": _WORD_END,
                        "smin": smin,
                        "smax": smax,
                        "fold": fold,
                        "model": "full",
                        "test_roc_auc": auc,
                    })
    return pl.DataFrame(rows)


def _make_null_hga_only(
    real_scores: pl.DataFrame,
    n_perms: int = 10,
    rng: np.random.Generator | None = None,
    signal_offset: float = 0.0,
) -> pl.DataFrame:
    """Per-fold null scores matching real_scores rows, with optional signal_offset."""
    if rng is None:
        rng = np.random.default_rng(1)
    real_for_join = real_scores.select(
        ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax", "fold",
         "test_roc_auc"]
    )
    rows = []
    for row in real_for_join.iter_rows(named=True):
        for perm in range(n_perms):
            auc = float(np.clip(
                row["test_roc_auc"] + signal_offset + rng.normal(0.0, 0.03), 0.0, 1.0
            ))
            rows.append({
                "subject": row["subject"],
                "electrode_idx": row["electrode_idx"],
                "phoneme_pair": row["phoneme_pair"],
                "word_end": row["word_end"],
                "smin": row["smin"],
                "smax": row["smax"],
                "fold": row["fold"],
                "permutation_idx": perm,
                "model": "full",
                "test_roc_auc": auc,
            })
    return pl.DataFrame(rows)


def _make_real_with_control(
    n_electrodes: int = 2,
    n_folds: int = 3,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Per-fold real scores for behavior_with_control (full + baseline rows)."""
    if rng is None:
        rng = np.random.default_rng(2)
    rows = []
    for ei in range(n_electrodes):
        for pp in ["dn"]:
            for smin, smax in _WINDOWS:
                for fold in range(n_folds):
                    full_auc = float(np.clip(rng.normal(0.65 if ei == 0 else 0.52, 0.05), 0, 1))
                    rows.append({
                        "subject": _SUBJECT,
                        "electrode_idx": ei,
                        "phoneme_pair": pp,
                        "word_end": _WORD_END,
                        "smin": smin,
                        "smax": smax,
                        "fold": fold,
                        "model": "full",
                        "test_roc_auc": full_auc,
                    })
    # Baseline: one per (phoneme_pair, word_end, fold), electrode_idx=-1, smin/smax=-1
    for pp in ["dn"]:
        for fold in range(n_folds):
            rows.append({
                "subject": _SUBJECT,
                "electrode_idx": -1,
                "phoneme_pair": pp,
                "word_end": _WORD_END,
                "smin": -1,
                "smax": -1,
                "fold": fold,
                "model": "baseline",
                "test_roc_auc": float(np.clip(rng.normal(0.55, 0.03), 0, 1)),
            })
    return pl.DataFrame(rows)


def _make_null_with_control(
    real_scores: pl.DataFrame,
    n_perms: int = 10,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Per-fold null scores for behavior_with_control."""
    if rng is None:
        rng = np.random.default_rng(3)
    rows = []
    for row in real_scores.iter_rows(named=True):
        for perm in range(n_perms):
            auc = float(np.clip(row["test_roc_auc"] + rng.normal(0.0, 0.03), 0.0, 1.0))
            rows.append({**row, "permutation_idx": perm, "test_roc_auc": auc})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: preagg schema
# ---------------------------------------------------------------------------


def test_preagg_hga_only_schema():
    """Output has correct columns, no fold column, one row per (site×window×perm)."""
    real = _make_real_hga_only(n_electrodes=2, n_folds=3)
    raw_null = _make_null_hga_only(real, n_perms=5)

    site_keys = SITE_KEYS_BEHAVIOR_HGA_ONLY
    preagg = _preagg_hga_only_null(raw_null, real, site_keys=site_keys)

    expected_cols = set(site_keys) | {"smin", "smax", "permutation_idx",
                                       "fold_mean_diff", "fold_std_diff", "n_folds", "t_stat"}
    assert set(preagg.columns) == expected_cols, (
        f"unexpected columns: {set(preagg.columns) ^ expected_cols}"
    )
    assert "fold" not in preagg.columns

    window_keys = site_keys + ["smin", "smax"]
    n_expected = raw_null.select(window_keys + ["permutation_idx"]).n_unique()
    assert preagg.height == n_expected, (
        f"preagg has {preagg.height} rows, expected {n_expected}"
    )

    assert preagg.select(window_keys + ["permutation_idx"]).n_unique() == preagg.height


def test_preagg_n_folds_correct():
    """n_folds in preagg output matches the number of folds in raw_null."""
    real = _make_real_hga_only(n_electrodes=1, n_folds=5)
    raw_null = _make_null_hga_only(real, n_perms=3)

    preagg = preagg_behavior_hga_only_null(raw_null, real)
    assert (preagg["n_folds"] == 5).all(), "n_folds should equal 5 for all rows"


# ---------------------------------------------------------------------------
# Tests: foldmean equivalence
# ---------------------------------------------------------------------------


def test_preagg_foldmean_equivalence():
    """fold_mean_diff ≥ 0 exactly when fold_mean_perm ≥ fold_mean_real.

    Verified row-by-row after joining preagg output with old-style aggregation.
    """
    rng = np.random.default_rng(42)
    real = _make_real_hga_only(n_electrodes=2, n_folds=4, rng=rng)
    raw_null = _make_null_hga_only(real, n_perms=20, rng=rng)

    site_keys = SITE_KEYS_BEHAVIOR_HGA_ONLY
    window_keys = site_keys + ["smin", "smax"]

    # Old-style: fold_mean per (window, perm)
    null_old = (
        raw_null.group_by(window_keys + ["permutation_idx"])
        .agg(pl.col("test_roc_auc").mean().alias("fold_mean_perm"))
    )
    real_mean = (
        real.group_by(window_keys)
        .agg(pl.col("test_roc_auc").mean().alias("fold_mean_real"))
    )
    old_joined = null_old.join(real_mean, on=window_keys, how="left")

    preagg = _preagg_hga_only_null(raw_null, real, site_keys=site_keys)

    # Join preagg with old_joined on (window_keys + permutation_idx)
    check = preagg.join(old_joined, on=window_keys + ["permutation_idx"], how="inner")
    assert check.height == preagg.height, "join lost rows"

    fm_diff_ge0 = (check["fold_mean_diff"] >= 0).to_numpy()
    perm_ge_real = (check["fold_mean_perm"] >= check["fold_mean_real"]).to_numpy()
    assert np.array_equal(fm_diff_ge0, perm_ge_real), (
        f"foldmean equivalence violated on {(fm_diff_ge0 != perm_ge_real).sum()} rows"
    )


# ---------------------------------------------------------------------------
# Tests: aggregate_* compatibility with preagg null
# ---------------------------------------------------------------------------


def test_aggregate_hga_only_with_preagg_matches_schema():
    """aggregate_behavior_hga_only with preagg null returns same schema as raw null path."""
    rng = np.random.default_rng(10)
    real = _make_real_hga_only(n_electrodes=2, n_folds=3, rng=rng)
    raw_null = _make_null_hga_only(real, n_perms=8, rng=rng)

    common_kwargs = dict(
        epoch_tmin=_EPOCH_TMIN,
        epoch_sfreq=_EPOCH_SFREQ,
        behav_peak_post_offset_s=_BEHAV_POST_OFFSET_S,
        peak_search_smin=0,
        peak_search_smax=100,
    )

    preagg_null = preagg_behavior_hga_only_null(raw_null, real)
    real_agg_preagg, null_agg_preagg = aggregate_behavior_hga_only(
        real, preagg_null, **common_kwargs
    )

    real_agg_raw, null_agg_raw = aggregate_behavior_hga_only(
        real, raw_null, **common_kwargs
    )

    # Schemas must match exactly.
    assert set(null_agg_preagg.columns) == set(null_agg_raw.columns), (
        f"column mismatch: preagg={null_agg_preagg.columns}, raw={null_agg_raw.columns}"
    )
    assert set(real_agg_preagg.columns) == set(real_agg_raw.columns)

    # Same number of rows per path (same sites × windows × perms after filtering).
    assert null_agg_preagg.height == null_agg_raw.height, (
        f"null_agg row count: preagg={null_agg_preagg.height}, raw={null_agg_raw.height}"
    )
    assert real_agg_preagg.height == real_agg_raw.height


def test_aggregate_hga_only_preagg_fold_mean_matches():
    """null_agg.fold_mean from preagg path equals fold_mean from raw path (same perm)."""
    rng = np.random.default_rng(20)
    real = _make_real_hga_only(n_electrodes=1, n_folds=3, rng=rng)
    raw_null = _make_null_hga_only(real, n_perms=5, rng=rng)

    common_kwargs = dict(
        epoch_tmin=_EPOCH_TMIN,
        epoch_sfreq=_EPOCH_SFREQ,
        behav_peak_post_offset_s=_BEHAV_POST_OFFSET_S,
        peak_search_smin=0,
        peak_search_smax=100,
    )

    site_keys = SITE_KEYS_BEHAVIOR_HGA_ONLY
    window_keys = site_keys + ["smin", "smax"]

    preagg_null = preagg_behavior_hga_only_null(raw_null, real)
    _, null_agg_preagg = aggregate_behavior_hga_only(real, preagg_null, **common_kwargs)
    _, null_agg_raw = aggregate_behavior_hga_only(real, raw_null, **common_kwargs)

    join_keys = window_keys + ["permutation_idx"]
    check = null_agg_preagg.join(
        null_agg_raw.rename({"fold_mean": "fold_mean_raw"}),
        on=join_keys, how="inner",
    )
    assert check.height == null_agg_preagg.height

    np.testing.assert_allclose(
        check["fold_mean"].to_numpy(),
        check["fold_mean_raw"].to_numpy(),
        atol=1e-5,
        err_msg="preagg fold_mean does not match raw fold_mean",
    )


def test_aggregate_hga_only_preagg_p_values_sane():
    """Preagg path produces valid p-values from null_standardized_peak_test."""
    from src.models.significance import null_standardized_peak_test

    rng = np.random.default_rng(30)
    real = _make_real_hga_only(n_electrodes=2, n_folds=3, rng=rng)
    raw_null = _make_null_hga_only(real, n_perms=15, rng=rng)

    common_kwargs = dict(
        epoch_tmin=_EPOCH_TMIN,
        epoch_sfreq=_EPOCH_SFREQ,
        behav_peak_post_offset_s=_BEHAV_POST_OFFSET_S,
        peak_search_smin=0,
        peak_search_smax=100,
    )
    site_keys = SITE_KEYS_BEHAVIOR_HGA_ONLY

    preagg_null = preagg_behavior_hga_only_null(raw_null, real)
    real_agg, null_agg = aggregate_behavior_hga_only(real, preagg_null, **common_kwargs)

    peaks, _ = null_standardized_peak_test(
        real_agg.rename({"fold_mean": "statistic"}),
        null_agg.rename({"fold_mean": "statistic"}),
        site_keys=site_keys,
        window_keys=["smin", "smax"],
        stat_col="statistic",
    )

    p = peaks["p_value"].to_numpy()
    min_p = 1.0 / (15 + 1)
    assert ((p >= min_p) & (p <= 1.0)).all(), f"p-values out of range: {p}"
    assert peaks.height == real_agg.select(site_keys).n_unique()


def test_aggregate_with_control_preagg_schema():
    """aggregate_behavior_with_control with preagg null returns same schema as raw path."""
    rng = np.random.default_rng(40)
    real = _make_real_with_control(n_electrodes=2, n_folds=3, rng=rng)
    raw_null = _make_null_with_control(real, n_perms=6, rng=rng)

    common_kwargs = dict(
        epoch_tmin=_EPOCH_TMIN,
        epoch_sfreq=_EPOCH_SFREQ,
        behav_peak_post_offset_s=_BEHAV_POST_OFFSET_S,
        peak_search_smin=0,
        peak_search_smax=100,
    )

    preagg_null = preagg_behavior_with_control_null(raw_null, real)

    # preagg null must not have fold or model columns
    assert "fold" not in preagg_null.columns
    assert "model" not in preagg_null.columns
    assert "fold_mean_diff" in preagg_null.columns

    real_agg_preagg, null_agg_preagg = aggregate_behavior_with_control(
        real, preagg_null, **common_kwargs
    )
    real_agg_raw, null_agg_raw = aggregate_behavior_with_control(
        real, raw_null, **common_kwargs
    )

    assert set(null_agg_preagg.columns) == set(null_agg_raw.columns)
    assert null_agg_preagg.height == null_agg_raw.height
    assert set(real_agg_preagg.columns) == set(real_agg_raw.columns)


def test_aggregate_with_control_preagg_fold_mean_matches():
    """null_agg.fold_mean from preagg path equals fold_mean from raw path."""
    rng = np.random.default_rng(50)
    real = _make_real_with_control(n_electrodes=1, n_folds=3, rng=rng)
    raw_null = _make_null_with_control(real, n_perms=5, rng=rng)

    common_kwargs = dict(
        epoch_tmin=_EPOCH_TMIN,
        epoch_sfreq=_EPOCH_SFREQ,
        behav_peak_post_offset_s=_BEHAV_POST_OFFSET_S,
        peak_search_smin=0,
        peak_search_smax=100,
    )

    site_keys = SITE_KEYS_BEHAVIOR_WITH_CONTROL
    window_keys = site_keys + ["smin", "smax"]

    preagg_null = preagg_behavior_with_control_null(raw_null, real)
    _, null_agg_preagg = aggregate_behavior_with_control(real, preagg_null, **common_kwargs)
    _, null_agg_raw = aggregate_behavior_with_control(real, raw_null, **common_kwargs)

    join_keys = window_keys + ["permutation_idx"]
    check = null_agg_preagg.join(
        null_agg_raw.rename({"fold_mean": "fold_mean_raw"}),
        on=join_keys, how="inner",
    )
    assert check.height == null_agg_preagg.height

    np.testing.assert_allclose(
        check["fold_mean"].to_numpy(),
        check["fold_mean_raw"].to_numpy(),
        atol=1e-5,
        err_msg="preagg fold_mean does not match raw fold_mean for with_control",
    )
