"""
Unit tests for src.models.significance.null_standardized_peak_test.

Synthetic-only; no real data or torch required.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.significance import null_standardized_peak_test


SITE_KEYS = ["subject", "electrode_idx"]
WINDOW_KEYS = ["smin", "smax"]


def _make_long(
    *,
    real: np.ndarray,            # (W,) real stat per window
    null: np.ndarray,            # (K, W) null stat per (perm, window)
    site_id: tuple = ("S1", 0),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build (real_stats, null_stats) DataFrames for a single site."""
    W = real.size
    K = null.shape[0]
    windows = [(w, w + 10) for w in range(W)]

    real_df = pl.DataFrame({
        "subject": [site_id[0]] * W,
        "electrode_idx": [site_id[1]] * W,
        "smin": [w[0] for w in windows],
        "smax": [w[1] for w in windows],
        "statistic": real.astype(np.float64),
    })

    rows = []
    for k in range(K):
        for w_i in range(W):
            rows.append({
                "subject": site_id[0],
                "electrode_idx": site_id[1],
                "smin": windows[w_i][0],
                "smax": windows[w_i][1],
                "permutation_idx": k,
                "statistic": float(null[k, w_i]),
            })
    null_df = pl.DataFrame(rows)
    return real_df, null_df


def test_homoscedastic_null_agrees_with_raw_argmax():
    """If null is homoscedastic across windows, null-standardized peak
    equals the raw-argmax peak — monotonic transformation per window.
    """
    rng = np.random.default_rng(0)
    W, K = 5, 200

    # All windows share the same null distribution (Normal(0.5, 0.02)).
    null = 0.5 + 0.02 * rng.standard_normal((K, W))

    # Real has a clear peak in window 2.
    real = np.array([0.52, 0.55, 0.70, 0.50, 0.48])

    real_df, null_df = _make_long(real=real, null=null)
    peak_summary, _ = null_standardized_peak_test(
        real_df, null_df, site_keys=SITE_KEYS,
    )

    assert peak_summary.height == 1
    row = peak_summary.to_dicts()[0]
    assert row["peak_smin"] == 2  # argmax of real (homoscedastic → agrees)
    assert row["pointwise_p"] < 0.01  # clearly extreme
    assert row["p_value"] < 0.05


def test_heteroscedastic_null_prefers_low_null_variance_window():
    """When the raw argmax lands in a high-null-variance window,
    standardization should pull the peak toward a lower-null-variance
    window where the real stat is more extreme vs its own null.
    """
    rng = np.random.default_rng(1)
    W, K = 3, 500

    # Window 0: tight null (SD 0.01), real 0.55 → 5 SD above mean. Null
    #           almost never crosses; empirical p ≈ 1/(K+1).
    # Window 1: wide null (SD 0.20), real 0.80 → 1.5 SD above mean. Null
    #           crosses ~7% of the time; empirical p ≈ 0.07.
    # Window 2: tight null (SD 0.01), real 0.50 → at null mean, p ~ 0.5.
    null = np.stack([
        0.5 + 0.01 * rng.standard_normal(K),   # w=0, tight
        0.5 + 0.20 * rng.standard_normal(K),   # w=1, wide
        0.5 + 0.01 * rng.standard_normal(K),   # w=2, tight
    ], axis=1)

    real = np.array([0.55, 0.80, 0.50])

    # Raw argmax → window 1 (0.80 is the largest real statistic).
    assert int(np.argmax(real)) == 1

    real_df, null_df = _make_long(real=real, null=null)
    peak_summary, _ = null_standardized_peak_test(
        real_df, null_df, site_keys=SITE_KEYS,
    )
    row = peak_summary.to_dicts()[0]
    # Window 0's 0.56 is ~6 SD above its null (never seen); window 1's
    # 0.80 is ~3 SD above its null (rare but occasionally exceeded).
    # Standardization should prefer window 0.
    assert row["peak_smin"] == 0, (
        f"expected peak in low-null-var window 0, got {row['peak_smin']} "
        f"(pointwise_p={row['pointwise_p']:.4g})"
    )


def test_p_value_approximately_uniform_under_null():
    """If the 'real' stat is drawn from the same null distribution, the
    corrected p-value should be approximately uniform on (0, 1]. Check
    that a reasonable fraction fall in [0, 0.5] and [0.5, 1].
    """
    rng = np.random.default_rng(2)
    W, K = 4, 300
    n_sites = 200

    real_rows = []
    null_rows = []
    for s in range(n_sites):
        null = rng.standard_normal((K, W))       # N(0, 1) null
        real = rng.standard_normal(W)            # draw "real" from same null
        for w_i in range(W):
            real_rows.append({
                "subject": f"S{s}", "electrode_idx": 0,
                "smin": w_i, "smax": w_i + 10,
                "statistic": float(real[w_i]),
            })
            for k in range(K):
                null_rows.append({
                    "subject": f"S{s}", "electrode_idx": 0,
                    "smin": w_i, "smax": w_i + 10,
                    "permutation_idx": k,
                    "statistic": float(null[k, w_i]),
                })

    real_df = pl.DataFrame(real_rows)
    null_df = pl.DataFrame(null_rows)
    peak_summary, _ = null_standardized_peak_test(
        real_df, null_df, site_keys=SITE_KEYS,
    )
    p_values = peak_summary["p_value"].to_numpy()
    # Under the null, p should be ~uniform. Loose bounds to avoid flakiness.
    assert 0.3 < (p_values < 0.5).mean() < 0.7, (
        f"fraction below 0.5 = {(p_values < 0.5).mean():.3f}"
    )
    # Not trivially all-significant or all-nonsignificant.
    assert 0.02 < (p_values < 0.1).mean() < 0.2, (
        f"fraction below 0.1 = {(p_values < 0.1).mean():.3f}"
    )


def test_nan_nulls_dropped_per_window_but_site_survives():
    """NaN null entries at a subset of windows get dropped; the site is
    still tested using the remaining valid permutations.
    """
    rng = np.random.default_rng(3)
    W, K = 4, 100
    null = 0.5 + 0.02 * rng.standard_normal((K, W))
    # Punch NaN into 10 permutations at window 1.
    null[:10, 1] = np.nan
    real = np.array([0.52, 0.70, 0.50, 0.48])

    real_df, null_df = _make_long(real=real, null=null)
    peak_summary, window_stats = null_standardized_peak_test(
        real_df, null_df, site_keys=SITE_KEYS,
    )
    row = peak_summary.to_dicts()[0]
    # Site should still rank peak at window 1 (0.70 is the clear real peak)
    # and have a meaningful p-value.
    assert row["peak_smin"] == 1
    # n_permutations counted at the site level = total perms that produced
    # a finite T_null_k; with NaN at 1 of 4 windows, perm's max still
    # defined over the other windows → still valid.
    assert row["n_permutations"] == K


def test_generic_site_keys_schema():
    """Utility accepts arbitrary site-key schemas without code change
    (e.g., acoustic has no word_end; behavior has word_end).
    """
    rng = np.random.default_rng(4)
    W, K = 3, 50
    null = rng.standard_normal((K, W))
    real = rng.standard_normal(W)

    rows_real, rows_null = [], []
    for we in ("desolate", "necessary"):
        for w_i in range(W):
            rows_real.append({
                "subject": "S1", "electrode_idx": 7, "phoneme_pair": "dn",
                "word_end": we,
                "smin": w_i, "smax": w_i + 10,
                "statistic": float(real[w_i]),
            })
            for k in range(K):
                rows_null.append({
                    "subject": "S1", "electrode_idx": 7, "phoneme_pair": "dn",
                    "word_end": we,
                    "smin": w_i, "smax": w_i + 10,
                    "permutation_idx": k,
                    "statistic": float(null[k, w_i]),
                })
    peak_summary, window_stats = null_standardized_peak_test(
        pl.DataFrame(rows_real),
        pl.DataFrame(rows_null),
        site_keys=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    )
    # Two sites (one per word_end)
    assert peak_summary.height == 2
    assert set(peak_summary["word_end"].to_list()) == {"desolate", "necessary"}
    # Schema sanity
    assert set(peak_summary.columns) == {
        "subject", "electrode_idx", "phoneme_pair", "word_end",
        "peak_smin", "peak_smax",
        "real_statistic", "pointwise_p", "T_obs", "p_value",
        "n_permutations", "null_q05", "null_q50", "null_q95", "null_q99",
    }


def test_validation_missing_columns():
    """Raise if required columns are missing from either input."""
    real_df = pl.DataFrame({"subject": ["S1"], "smin": [0], "smax": [10], "statistic": [0.5]})
    null_df = pl.DataFrame({"subject": ["S1"], "smin": [0], "smax": [10], "statistic": [0.5]})
    # null_df missing permutation_idx
    with pytest.raises(ValueError, match="permutation_idx"):
        null_standardized_peak_test(
            real_df, null_df, site_keys=["subject"],
        )
