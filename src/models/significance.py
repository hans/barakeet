"""
Null-standardized peak-finding + max-stat permutation test, plus two
helper aggregates used alongside it: a fold-variance-normalized t-stat
and a 1D TFCE enhancement.

The workhorse is ``null_standardized_peak_test``. It takes pre-aggregated
per-(site, window[, permutation]) statistics and produces per-site
peak-windows + p-values by pointwise permutation p + max-stat correction.

The two helpers build different statistics to feed it:

* ``fold_tstat_aggregate`` — collapses fold-wise AUCs to per-(site, window[,
  perm]) with both ``fold_mean`` and ``t_stat = (fold_mean - center) /
  (fold_std / sqrt(n_folds))``. Variance-normalization can improve the
  separation between real and null if real signal is consistent across
  folds while null permutations scatter.
* ``tfce_1d_per_site`` — Threshold-Free Cluster Enhancement along the
  window axis, per site (and per perm). Real signal that spans multiple
  adjacent windows accumulates extent credit; isolated noise peaks do
  not. Use before max-stat so the correction acts on cluster-enhanced
  values rather than on single-window statistics.

Rationale for pointwise standardization: peak-finding on raw fold-mean
AUC implicitly assumes homoscedastic null across windows. In ECoG
searchlight decoding the null varies — low-variance pre-stimulus HGA
gives a tight ~0.5 null while structured task-response HGA gives a wider
null. Raw-AUC peaks therefore systematically favour the noisiest
windows. Pointwise standardization removes that bias, and max-stat
correction keeps family-wise Type I control across windows.

Reference: Westfall & Young (1993) *Resampling-Based Multiple Testing*;
Nichols & Holmes (2002) for fMRI / searchlight applications; Smith &
Nichols (2009) for TFCE.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import polars as pl


def null_standardized_peak_test(
    real_stats: pl.DataFrame,
    null_stats: pl.DataFrame,
    *,
    site_keys: Sequence[str],
    window_keys: Sequence[str] = ("smin", "smax"),
    perm_key: str = "permutation_idx",
    stat_col: str = "statistic",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Per-site peak-finding with pointwise null standardization and max-stat
    permutation test.

    Both inputs must be pre-aggregated to one scalar statistic per
    (site, window[, permutation]); the caller decides what ``stat_col``
    means (fold-mean AUC, fold-mean full−baseline diff, ...).

    Pipeline, per site:

      1. Pointwise p per window (plug-in, inclusive of self for null):
             p_real[w]      = (#{null[:, w] >= real[w]} + 1) / (K_w + 1)
             p_null[k, w]   =  #{null[:, w] >= null[k, w]}  / (K_w + 1)
         where K_w = number of non-NaN null entries at window w. Both
         have min value 1 / (K_w + 1) at the extreme.
      2. Peak window for the site = argmin_w p_real[w].
      3. T_obs      = -log10(p_real[peak_w]).
         T_null_k   = max_w -log10(p_null[k, w]).
      4. Corrected p-value = (#{T_null_k >= T_obs} + 1) / (K + 1).

    Args:
        real_stats: long-format DataFrame. Required columns:
            site_keys + window_keys + [stat_col]. One row per (site, window).
        null_stats: long-format. Required:
            site_keys + window_keys + [perm_key, stat_col]. One row per
            (site, window, permutation).
        site_keys: columns identifying a site (e.g. ``["subject",
            "electrode_idx", "phoneme_pair"]`` or ``[..., "word_end"]``).
        window_keys: columns identifying a window inside a site. Default
            ``("smin", "smax")``.
        perm_key: name of the permutation index column in ``null_stats``.
        stat_col: name of the statistic column in both inputs.

    Returns:
        peak_summary: one row per site, with columns:
            site_keys, peak_{window_keys},
            real_statistic, pointwise_p, T_obs, p_value, n_permutations,
            null_q05, null_q50, null_q95, null_q99 (quantiles of T_null).
        window_stats: one row per (site, window), with columns:
            site_keys, window_keys, real_statistic, pointwise_p,
            null_q05, null_q50, null_q95, null_q99 (quantiles of the raw
            null statistic at that window — useful for diagnostic plots).

    NaN handling: null entries with NaN statistics are dropped per
    (site, window) before ranking; a site with all-NaN at every window
    will appear with ``n_permutations = 0`` and NaN p-value.
    """
    site_keys = list(site_keys)
    window_keys = list(window_keys)
    sw_keys = site_keys + window_keys

    for col in sw_keys + [stat_col]:
        if col not in real_stats.columns:
            raise ValueError(
                f"real_stats missing required column {col!r}; has {real_stats.columns}"
            )
    for col in sw_keys + [perm_key, stat_col]:
        if col not in null_stats.columns:
            raise ValueError(
                f"null_stats missing required column {col!r}; has {null_stats.columns}"
            )

    null_valid = null_stats.filter(
        pl.col(stat_col).is_not_null() & pl.col(stat_col).is_not_nan()
    )

    # --- per (site, window, perm) pointwise p for the null via rank ---
    # rank(method="max", descending=True) gives, for every value x in its
    # (site, window) group, the count of group members with value >= x
    # (ties included). Plug-in p = rank_max / (n + 1) guarantees
    # min(p) = 1 / (n + 1) at a uniquely-largest value.
    null_p_per_sw_perm = null_valid.with_columns(
        pl.col(stat_col)
          .rank(method="max", descending=True)
          .over(sw_keys)
          .alias("_rank"),
        pl.col(stat_col).count().over(sw_keys).alias("_n_valid_w"),
    ).with_columns(
        (pl.col("_rank") / (pl.col("_n_valid_w") + 1)).alias("pointwise_p_null")
    )

    # --- per (site, window) pointwise p for the real stat ---
    real_subset = real_stats.select(sw_keys + [stat_col])
    merged = real_subset.join(
        null_valid.select(sw_keys + [stat_col]).rename({stat_col: "_null_stat"}),
        on=sw_keys,
        how="left",
    )
    # After left-join: (site, window) with no null data → one row with
    # _null_stat = null; these sites get pointwise_p = 1 / 1 = 1 (no
    # information). The is_not_null() guards both sum()s.
    real_p_per_window = (
        merged.group_by(sw_keys)
        .agg(
            pl.col(stat_col).first().alias("real_statistic"),
            (
                (pl.col("_null_stat") >= pl.col(stat_col))
                & pl.col("_null_stat").is_not_null()
            ).sum().alias("_ge_count"),
            pl.col("_null_stat").is_not_null().sum().alias("_n_valid"),
        )
        .with_columns(
            ((pl.col("_ge_count") + 1) / (pl.col("_n_valid") + 1)).alias("pointwise_p"),
        )
        .drop("_ge_count", "_n_valid")
    )

    # --- peak window per site: argmin pointwise_p_real ---
    # Ties on pointwise_p are common at the finite-sample floor (any two
    # windows where real > max(null) both hit 1 / (K + 1)). Break ties by
    # larger real_statistic so the peak falls back to the raw argmax when
    # the plug-in p can't distinguish further.
    peak_per_site = (
        real_p_per_window.sort(["pointwise_p", "real_statistic"], descending=[False, True])
        .group_by(site_keys, maintain_order=True)
        .agg(pl.all().first())
        .with_columns((-pl.col("pointwise_p").log10()).alias("T_obs"))
    )

    # --- T_null_k = max_w -log10(p_null[k, w]) per (site, perm) ---
    t_null_per_site_perm = (
        null_p_per_sw_perm.with_columns(
            (-pl.col("pointwise_p_null").log10()).alias("_neg_log_p")
        )
        .group_by(site_keys + [perm_key])
        .agg(pl.col("_neg_log_p").max().alias("T_null"))
    )

    # --- corrected p-value per site via (#{T_null >= T_obs} + 1) / (K + 1) ---
    joined = t_null_per_site_perm.join(
        peak_per_site.select(site_keys + ["T_obs"]),
        on=site_keys,
        how="inner",
    )
    per_site_pvalue = (
        joined.group_by(site_keys + ["T_obs"])
        .agg(
            pl.len().alias("n_permutations"),
            (pl.col("T_null") >= pl.col("T_obs")).cast(pl.Int64).sum().alias("_ge_count"),
            pl.col("T_null").quantile(0.05).alias("null_q05"),
            pl.col("T_null").quantile(0.50).alias("null_q50"),
            pl.col("T_null").quantile(0.95).alias("null_q95"),
            pl.col("T_null").quantile(0.99).alias("null_q99"),
        )
        .with_columns(
            ((pl.col("_ge_count") + 1) / (pl.col("n_permutations") + 1)).alias("p_value")
        )
        .drop("_ge_count", "T_obs")
    )

    peak_summary = (
        peak_per_site.join(per_site_pvalue, on=site_keys, how="inner")
        .rename({wk: f"peak_{wk}" for wk in window_keys})
        .select([
            *site_keys,
            *[f"peak_{wk}" for wk in window_keys],
            "real_statistic", "pointwise_p", "T_obs", "p_value",
            "n_permutations",
            "null_q05", "null_q50", "null_q95", "null_q99",
        ])
    )

    null_quant_per_window = null_valid.group_by(sw_keys).agg(
        pl.col(stat_col).quantile(0.05).alias("null_q05"),
        pl.col(stat_col).quantile(0.50).alias("null_q50"),
        pl.col(stat_col).quantile(0.95).alias("null_q95"),
        pl.col(stat_col).quantile(0.99).alias("null_q99"),
    )
    window_stats = real_p_per_window.join(
        null_quant_per_window, on=sw_keys, how="left"
    )

    return peak_summary, window_stats


def fold_tstat_aggregate(
    scores: pl.DataFrame,
    *,
    group_keys: Sequence[str],
    stat_col: str = "test_roc_auc",
    center: float = 0.5,
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Aggregate fold-wise scores to per-group (fold_mean, fold_std, t_stat).

    ``t_stat = (fold_mean - center) / (max(fold_std, std_floor) / sqrt(n_folds))``

    The ``std_floor`` prevents the t-stat from blowing up at groups with
    near-zero fold variance (which can happen at degenerate windows);
    0.01 is ~1 pp of AUC and small vs typical fold-SD of ~0.03–0.10.

    The caller supplies the centering value: 0.5 for AUC (chance level),
    0 for signed ``diff`` statistics like ``full - baseline``. Centering
    does not change the rank structure of the permutation test — it
    only changes the sign convention for t — so the pointwise p is
    identical under any monotone recentering that keeps the real and
    null inputs aligned. Provided for interpretability of the raw value.

    Args:
        scores: long-format DataFrame with ``group_keys + ['fold', stat_col]``.
            One row per (group, fold).
        group_keys: columns identifying a group to collapse folds within.
            Typically ``site_keys + window_keys`` for real, or the same
            plus ``perm_key`` for null.
        stat_col: name of the per-fold statistic column.
        center: centering value for the numerator.
        std_floor: minimum effective fold_std used in the t-stat
            denominator. Fold_std below this value gets clamped up to
            std_floor.

    Returns:
        DataFrame with one row per group and columns:
            ``*group_keys, fold_mean, fold_std, n_folds, t_stat``.
    """
    group_keys = list(group_keys)
    if stat_col not in scores.columns:
        raise ValueError(
            f"scores missing required column {stat_col!r}; has {scores.columns}"
        )

    agg = scores.group_by(group_keys).agg(
        pl.col(stat_col).mean().alias("fold_mean"),
        pl.col(stat_col).std().alias("fold_std"),
        pl.col(stat_col).len().alias("n_folds"),
    )
    sem = (
        pl.max_horizontal(pl.col("fold_std"), pl.lit(std_floor))
        / pl.col("n_folds").cast(pl.Float64).sqrt()
    )
    return agg.with_columns(
        ((pl.col("fold_mean") - center) / sem).alias("t_stat")
    )


def _tfce_1d(stat: np.ndarray, *, E: float, H: float, dh: Optional[float], threshold: float) -> np.ndarray:
    """One-tailed 1D TFCE enhancement for a 1D numpy array.

    Matches MNE's convention (see ``mne.stats.cluster_level._find_clusters``
    with ``threshold={'start': threshold, 'step': dh}``, ``tail=1``):

        score[v] = sum over thresholds t of (delta_t)^H * extent(t, v)^E

    where ``delta_t`` is the step between consecutive thresholds (equal
    to ``abs(t_0)`` at the first threshold, ``dh`` for the rest), and
    ``extent(t, v)`` is the length of the contiguous run containing
    voxel ``v`` in ``stat > t`` (or 0 if v is not in any run at that t).

    This differs from the canonical Smith & Nichols (2009) continuous
    form (``integral of h^H * e^E dh``); MNE's form uses ``(dh)^H``
    rather than ``h^H * dh``. Permutation inference is rank-invariant so
    the choice does not affect p-values, but we match MNE's formula so
    the cross-validation test in ``tests/test_significance.py`` can
    assert numerical equivalence.
    """
    out = np.zeros_like(stat, dtype=np.float64)
    finite = stat[np.isfinite(stat)]
    if finite.size == 0:
        return out
    max_stat = float(finite.max())
    if max_stat <= threshold:
        return out
    if dh is None:
        step = max_stat / 100.0
    else:
        step = float(dh)
    if step <= 0:
        raise ValueError(f"dh must be positive, got {step}")

    thresholds = np.arange(threshold, max_stat, step, dtype=np.float64)
    prev_t: Optional[float] = None
    for ti, t in enumerate(thresholds):
        if ti == 0:
            delta = abs(float(t))
        else:
            delta = abs(float(t) - prev_t)  # type: ignore[operator]
        prev_t = float(t)
        delta_H = delta ** H
        if delta_H == 0:
            continue  # degenerate first threshold at h=0 contributes nothing
        above = stat > t  # strict inequality matches MNE
        if not above.any():
            break
        # Contiguous-run extents via edge detection.
        padded = np.concatenate(([False], above, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            extent = e - s
            out[s:e] += (extent ** E) * delta_H
    return out


def tfce_1d_per_site(
    stats: pl.DataFrame,
    *,
    site_keys: Sequence[str],
    window_keys: Sequence[str] = ("smin", "smax"),
    perm_key: Optional[str] = None,
    stat_col: str = "statistic",
    E: float = 0.5,
    H: float = 2.0,
    dh: Optional[float] = None,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Per-site (and per-permutation) 1D TFCE along ``window_keys``.

    Windows are ordered by the first element of ``window_keys`` (default
    ``smin``) within each (site[, perm]) group. The returned DataFrame
    has the same schema as the input but with ``stat_col`` replaced by
    the TFCE-enhanced value.

    Args:
        stats: long-format with ``site_keys + window_keys + [perm_key?,
            stat_col]``. One row per (site, window[, perm]).
        site_keys: columns identifying a site.
        window_keys: columns identifying a window inside a site.
        perm_key: permutation-index column, or ``None`` for a real-only
            (one-enhancement-per-site) run.
        stat_col: name of the statistic to enhance.
        E: extent exponent (TFCE default 0.5).
        H: height exponent (TFCE default 2.0).
        dh: threshold step. ``None`` = ``max(stat_in_group) / 100``
            computed per group (adaptive to each site's dynamic range).
        threshold: one-tailed floor; values at or below are zeroed
            before enhancement. Use 0.5 for AUC to drop below-chance
            windows, or 0 for centered diff-type statistics.

    Returns:
        DataFrame with columns ``site_keys + window_keys + [perm_key?,
        stat_col]``, ``stat_col`` containing TFCE-enhanced values.
    """
    site_keys = list(site_keys)
    window_keys = list(window_keys)
    group_keys = site_keys + ([perm_key] if perm_key is not None else [])
    required = site_keys + window_keys + ([perm_key] if perm_key else []) + [stat_col]
    for col in required:
        if col not in stats.columns:
            raise ValueError(
                f"stats missing required column {col!r}; has {stats.columns}"
            )

    order_col = window_keys[0]
    # Sort so that each group's rows appear contiguously and in window order.
    sorted_df = stats.sort(group_keys + [order_col])

    n = sorted_df.height
    enhanced = np.empty(n, dtype=np.float64)
    stat_np = sorted_df[stat_col].to_numpy()

    if group_keys:
        # Vectorized group-boundary detection. sorted_df is sorted by
        # group_keys + [order_col], so identical group tuples are
        # contiguous; rle_id() yields 0,0,…,1,1,…,2,… and increments
        # exactly where the previous per-row Python comparison fired.
        if n == 0:
            boundaries = np.array([0], dtype=np.int64)
        else:
            group_id = (
                sorted_df.select(pl.struct(group_keys).rle_id().alias("_gid"))[
                    "_gid"
                ].to_numpy()
            )
            boundaries = np.concatenate((
                [0],
                np.flatnonzero(np.diff(group_id)) + 1,
                [n],
            ))
        for k in range(len(boundaries) - 1):
            s, e = int(boundaries[k]), int(boundaries[k + 1])
            enhanced[s:e] = _tfce_1d(
                stat_np[s:e], E=E, H=H, dh=dh, threshold=threshold,
            )
    else:
        enhanced[:] = _tfce_1d(stat_np, E=E, H=H, dh=dh, threshold=threshold)

    return sorted_df.with_columns(
        pl.Series(stat_col, enhanced)
    )
