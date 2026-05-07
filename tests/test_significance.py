"""
Unit tests for src.models.significance.

Synthetic-only; no real data or torch required.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.significance import (
    fold_tstat_aggregate,
    null_standardized_peak_test,
    tfce_1d_per_site,
)


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


# =============================================================================
# fold_tstat_aggregate
# =============================================================================


def test_fold_tstat_aggregate_basic():
    """fold_mean, fold_std, t_stat match hand computation."""
    aucs = np.array([0.60, 0.62, 0.58, 0.63, 0.61])
    scores = pl.DataFrame({
        "subject": ["S1"] * 5,
        "electrode_idx": [0] * 5,
        "fold": list(range(5)),
        "test_roc_auc": aucs,
    })
    out = fold_tstat_aggregate(
        scores, group_keys=["subject", "electrode_idx"], center=0.5, std_floor=0.01,
    )
    assert out.height == 1
    row = out.to_dicts()[0]
    assert row["fold_mean"] == pytest.approx(aucs.mean())
    # polars uses ddof=1 by default for std
    assert row["fold_std"] == pytest.approx(aucs.std(ddof=1))
    assert row["n_folds"] == 5
    expected_t = (aucs.mean() - 0.5) / (max(aucs.std(ddof=1), 0.01) / np.sqrt(5))
    assert row["t_stat"] == pytest.approx(expected_t)


def test_fold_tstat_aggregate_std_floor_clamps():
    """When fold_std < std_floor, the denominator uses std_floor instead."""
    # All folds equal → std = 0 → without floor, t would be +inf
    aucs = np.array([0.70] * 5)
    scores = pl.DataFrame({
        "subject": ["S1"] * 5,
        "electrode_idx": [0] * 5,
        "fold": list(range(5)),
        "test_roc_auc": aucs,
    })
    out = fold_tstat_aggregate(
        scores, group_keys=["subject", "electrode_idx"], center=0.5, std_floor=0.05,
    )
    row = out.to_dicts()[0]
    expected_t = 0.20 / (0.05 / np.sqrt(5))
    assert row["t_stat"] == pytest.approx(expected_t)
    assert np.isfinite(row["t_stat"])


def test_fold_tstat_aggregate_center_param():
    """Changing center shifts t but not the other aggregates."""
    aucs = np.array([0.10, 0.12, 0.08, 0.13, 0.11])  # centered around 0.1 (diff-like)
    scores = pl.DataFrame({
        "subject": ["S1"] * 5,
        "electrode_idx": [0] * 5,
        "fold": list(range(5)),
        "diff": aucs,
    })
    out_0 = fold_tstat_aggregate(
        scores, group_keys=["subject", "electrode_idx"],
        stat_col="diff", center=0.0, std_floor=0.01,
    ).to_dicts()[0]
    out_half = fold_tstat_aggregate(
        scores, group_keys=["subject", "electrode_idx"],
        stat_col="diff", center=0.5, std_floor=0.01,
    ).to_dicts()[0]
    # Means and stds are identical
    assert out_0["fold_mean"] == pytest.approx(out_half["fold_mean"])
    assert out_0["fold_std"] == pytest.approx(out_half["fold_std"])
    # t_stat shifts by (0.0 - 0.5) / sem; center=0 gives the larger (positive) t
    sem = max(aucs.std(ddof=1), 0.01) / np.sqrt(5)
    assert out_0["t_stat"] == pytest.approx(aucs.mean() / sem)
    assert out_half["t_stat"] == pytest.approx((aucs.mean() - 0.5) / sem)


# =============================================================================
# tfce_1d_per_site
# =============================================================================


def _lattice_adj(n: int):
    """Construct a 1D nearest-neighbor adjacency for MNE's TFCE. Required
    because MNE's adjacency=None branch has a long-standing bug for 1D data
    (cluster extents reported as 1 regardless of actual run length)."""
    from scipy import sparse

    rows, cols = [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    return sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n),
    )


def test_tfce_matches_mne_no_threshold():
    """Values match mne.stats.cluster_level._find_clusters exactly for a
    representative smooth signal at threshold=0."""
    from mne.stats.cluster_level import _find_clusters

    rng = np.random.default_rng(42)
    x = rng.standard_normal(25)
    x[5:12] += 1.5  # smooth bump
    x[18:21] += 0.8  # narrower bump

    _, mne_vals = _find_clusters(
        x, threshold={"start": 0.0, "step": 0.05}, tail=1, adjacency=_lattice_adj(len(x)),
    )

    stats_df = pl.DataFrame({
        "subject": ["S1"] * len(x),
        "electrode_idx": [0] * len(x),
        "smin": list(range(len(x))),
        "smax": [s + 10 for s in range(len(x))],
        "statistic": x.astype(np.float64),
    })
    out = tfce_1d_per_site(
        stats_df, site_keys=["subject", "electrode_idx"],
        E=0.5, H=2.0, dh=0.05, threshold=0.0,
    )
    ours = out.sort("smin")["statistic"].to_numpy()
    assert np.allclose(ours, mne_vals, atol=1e-9), (
        f"max |diff| = {np.max(np.abs(ours - mne_vals)):.3e}"
    )


def test_tfce_matches_mne_with_threshold():
    """Same equivalence with a nonzero integration start."""
    from mne.stats.cluster_level import _find_clusters

    rng = np.random.default_rng(7)
    x = 0.5 + 0.1 * rng.standard_normal(30)  # AUC-like values around 0.5
    x[8:18] += 0.15

    _, mne_vals = _find_clusters(
        x, threshold={"start": 0.5, "step": 0.01}, tail=1, adjacency=_lattice_adj(len(x)),
    )
    stats_df = pl.DataFrame({
        "subject": ["S1"] * len(x),
        "electrode_idx": [0] * len(x),
        "smin": list(range(len(x))),
        "smax": [s + 10 for s in range(len(x))],
        "statistic": x.astype(np.float64),
    })
    out = tfce_1d_per_site(
        stats_df, site_keys=["subject", "electrode_idx"],
        E=0.5, H=2.0, dh=0.01, threshold=0.5,
    )
    ours = out.sort("smin")["statistic"].to_numpy()
    assert np.allclose(ours, mne_vals, atol=1e-9)


def test_tfce_broad_cluster_beats_narrow_peak():
    """A broad cluster should receive a larger TFCE value than a narrow
    peak of the same height. This is the whole point of TFCE."""
    n = 30
    x_narrow = np.zeros(n)
    x_narrow[15] = 1.0          # single-window peak

    x_broad = np.zeros(n)
    x_broad[10:20] = 1.0        # 10-window plateau at same height

    def _run(arr):
        df = pl.DataFrame({
            "subject": ["S1"] * n,
            "electrode_idx": [0] * n,
            "smin": list(range(n)),
            "smax": [s + 10 for s in range(n)],
            "statistic": arr.astype(np.float64),
        })
        return tfce_1d_per_site(
            df, site_keys=["subject", "electrode_idx"],
            E=0.5, H=2.0, dh=0.05, threshold=0.0,
        ).sort("smin")["statistic"].to_numpy()

    narrow_vals = _run(x_narrow)
    broad_vals = _run(x_broad)
    # Peak-to-peak comparison: broad's plateau value > narrow's single peak
    assert broad_vals[15] > narrow_vals[15]


def test_tfce_per_site_and_per_perm_isolation():
    """Enhancement applied per (site, perm) group; different groups don't
    interfere."""
    n_windows = 10
    sites = ["A", "B"]
    rows = []
    # Site A: broad cluster in windows 3-6
    # Site B: narrow peak at window 7
    for perm in range(3):
        for site in sites:
            for w in range(n_windows):
                if site == "A":
                    v = 1.0 if 3 <= w <= 6 else 0.0
                else:
                    v = 1.0 if w == 7 else 0.0
                # Add a small per-perm perturbation so groups differ
                v += 0.01 * perm
                rows.append({
                    "subject": site, "electrode_idx": 0,
                    "smin": w, "smax": w + 1,
                    "permutation_idx": perm,
                    "statistic": v,
                })
    df = pl.DataFrame(rows)
    out = tfce_1d_per_site(
        df, site_keys=["subject", "electrode_idx"],
        perm_key="permutation_idx",
        E=0.5, H=2.0, dh=0.05, threshold=0.0,
    )
    # Site A at any window in its cluster should outscore Site B's narrow peak
    a_peak = out.filter(
        (pl.col("subject") == "A") & (pl.col("smin") == 5) & (pl.col("permutation_idx") == 0)
    )["statistic"][0]
    b_peak = out.filter(
        (pl.col("subject") == "B") & (pl.col("smin") == 7) & (pl.col("permutation_idx") == 0)
    )["statistic"][0]
    assert a_peak > b_peak


def test_tfce_rejects_missing_columns():
    df = pl.DataFrame({"subject": ["S1"], "electrode_idx": [0], "smin": [0]})
    with pytest.raises(ValueError, match="smax"):
        tfce_1d_per_site(df, site_keys=["subject", "electrode_idx"])


def _tfce_1d_per_site_old_loop(
    stats: pl.DataFrame,
    *,
    site_keys,
    window_keys=("smin", "smax"),
    perm_key=None,
    stat_col: str = "statistic",
    E: float = 0.5,
    H: float = 2.0,
    dh=None,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Inlined copy of the pre-vectorization tfce_1d_per_site for regression
    testing. Uses the per-row Python boundary scan."""
    from src.models.significance import _tfce_1d

    site_keys = list(site_keys)
    window_keys = list(window_keys)
    group_keys = site_keys + ([perm_key] if perm_key is not None else [])
    order_col = window_keys[0]
    sorted_df = stats.sort(group_keys + [order_col])

    enhanced = np.empty(sorted_df.height, dtype=np.float64)
    stat_np = sorted_df[stat_col].to_numpy()

    if group_keys:
        group_cols = [sorted_df[c].to_numpy() for c in group_keys]
        n = sorted_df.height
        start = 0
        for i in range(1, n + 1):
            at_end = i == n
            if not at_end:
                changed = any(col[i] != col[i - 1] for col in group_cols)
            if at_end or changed:
                enhanced[start:i] = _tfce_1d(
                    stat_np[start:i], E=E, H=H, dh=dh, threshold=threshold,
                )
                start = i
    else:
        enhanced[:] = _tfce_1d(stat_np, E=E, H=H, dh=dh, threshold=threshold)

    return sorted_df.with_columns(pl.Series(stat_col, enhanced))


def test_tfce_vectorized_boundaries_matches_old_loop():
    """Vectorized group-boundary detection produces byte-identical output
    to the original per-row Python scan on a fixture covering the moving
    parts: multiple sites + perms, NaN windows in one site, mixed group
    key dtypes (str + int), and per-group adaptive dh.
    """
    rng = np.random.default_rng(123)
    sites = [("S1", 0), ("S1", 7), ("S42", 3)]
    n_perms = 4
    n_windows = 8

    rows = []
    for site_id, (subj, elec) in enumerate(sites):
        for perm in range(n_perms):
            stats = 0.5 + 0.1 * rng.standard_normal(n_windows)
            # Inject a smooth bump so TFCE has something to enhance.
            stats[2:5] += 0.2 * (1 + 0.05 * perm)
            # NaN-out two windows in site 1 (mixed-NaN code path).
            if site_id == 1:
                stats[6:8] = np.nan
            for w in range(n_windows):
                rows.append({
                    "subject": subj,
                    "electrode_idx": elec,
                    "smin": w * 10,
                    "smax": w * 10 + 10,
                    "permutation_idx": perm,
                    "statistic": float(stats[w]),
                })
    df = pl.DataFrame(rows)

    new = tfce_1d_per_site(
        df, site_keys=["subject", "electrode_idx"],
        perm_key="permutation_idx", E=0.5, H=2.0, dh=None, threshold=0.0,
    )
    old = _tfce_1d_per_site_old_loop(
        df, site_keys=["subject", "electrode_idx"],
        perm_key="permutation_idx", E=0.5, H=2.0, dh=None, threshold=0.0,
    )

    new_arr = new.sort(
        ["subject", "electrode_idx", "permutation_idx", "smin"]
    )["statistic"].to_numpy()
    old_arr = old.sort(
        ["subject", "electrode_idx", "permutation_idx", "smin"]
    )["statistic"].to_numpy()

    assert np.array_equal(new_arr, old_arr), (
        f"max |diff| = {np.max(np.abs(new_arr - old_arr)):.3e}"
    )
