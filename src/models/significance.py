"""
Null-standardized peak-finding + max-stat permutation test.

Generic utility shared by causal6 peak-finding rules across decoders
(acoustic, behavior-with-control, behavior-HGA-only, and future ganong).
Callers pre-aggregate per-(site, window[, permutation]) statistics; this
module computes pointwise p-values per window against the per-window null
distribution, selects peaks in the standardized space, and applies
max-stat correction across windows.

Rationale: peak-finding on raw fold-mean AUC implicitly assumes
homoscedastic null across windows. In ECoG searchlight decoding the null
varies — low-variance pre-stimulus HGA gives a tight ~0.5 null while
structured task-response HGA gives a wider null. Raw-AUC peaks therefore
systematically favour the noisiest windows. Pointwise standardization
removes that bias, and max-stat correction keeps family-wise Type I
control across windows.

Reference: Westfall & Young (1993) *Resampling-Based Multiple Testing*;
Nichols & Holmes (2002) for fMRI / searchlight applications.
"""

from __future__ import annotations

from typing import Sequence

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
