"""
Two-stage adaptive-K permutation null gate primitives for the causal6
``*_null`` notebooks.

The flow per subject is:

1. Run K1 stage-1 permutations across all speech-responsive electrodes.
2. Aggregate via ``aggregate_<decoder>`` from
   ``src.models.causal6_aggregates`` (single source of truth shared
   with the summarize step) to per-(site, window[, perm]) statistics.
3. ``stage1_gate`` computes per-flavor ``min_pointwise_p`` (min over
   windows of pointwise_p) and a global min over flavors per site.
   Sites whose global min ≤ ``escalate_pointwise_p_max`` are flagged
   as borderline.
4. The notebook runs K2 additional permutations on the borderline
   electrodes (non-overlapping seed range with stage 1 for
   determinism), filters down to the exact borderline site keys via
   ``filter_null_to_borderline``, concatenates with stage 1, and
   writes one merged ``null_scores.parquet``.

Why ``pointwise_p`` (not the corrected p-value) at the gate, and why
TFCE flavors are included:

* For max-stat flavors the bound
  ``corrected_maxstat_p ≥ pointwise_p[peak]`` always holds, so a site
  whose min-over-windows pointwise_p exceeds the threshold cannot
  survive max-stat correction at any K. Rigorous bound, no double-K
  needed at the gate.
* TFCE flavors don't admit that bound (broad sub-threshold evidence
  can be amplified). The gate runs ``tfce_1d_per_site`` on
  ``apply_tfce=True`` flavors before computing pointwise_p so a site
  with a strong cluster but no individually significant window still
  escalates. Cost is pure polars / numpy work — no extra GPU refits.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

from src.models.causal6_aggregates import FlavorSpec
from src.models.significance import tfce_1d_per_site


def _flavor_name(flavor: FlavorSpec) -> str:
    """Stable column-name fragment for a flavor.

    ``fold_mean`` (apply_tfce=False) → ``"fold_mean"``;
    ``fold_mean`` (apply_tfce=True)  → ``"fold_mean_tfce"``.
    """
    return f"{flavor.stat_col}{'_tfce' if flavor.apply_tfce else ''}"


def min_pointwise_p_per_site(
    real_agg: pl.DataFrame,
    null_agg: pl.DataFrame,
    *,
    site_keys: Sequence[str],
    window_keys: Sequence[str] = ("smin", "smax"),
    perm_key: str = "permutation_idx",
    stat_col: str = "statistic",
) -> pl.DataFrame:
    """Per-site min over windows of plug-in pointwise_p.

    For each (site, window): ``pointwise_p[w] = (#{null[:, w] >= real[w]}
    + 1) / (K_w + 1)``, where ``K_w`` is the number of non-NaN null
    entries at window w. Mirrors ``null_standardized_peak_test``'s
    ``real_p_per_window`` (src/models/significance.py:124-158) so the
    gate's pointwise_p is identical to what the production peak-test
    sees at the same K.

    Tie-break on min: prefer the larger ``real_statistic`` (matches the
    peak-test's tie-break in significance.py:165-168).

    Args:
        real_agg: long-format with ``site_keys + window_keys + [stat_col]``.
            One row per (site, window).
        null_agg: long-format with ``site_keys + window_keys + [perm_key,
            stat_col]``. One row per (site, window, permutation).
        site_keys: columns identifying a site.
        window_keys: columns identifying a window inside a site.
        perm_key: permutation-index column in ``null_agg``.
        stat_col: name of the statistic column in both inputs.

    Returns:
        DataFrame with one row per site:
            ``site_keys + ['min_pointwise_p', 'argmin_<wk>'... ,
            'real_at_argmin', 'K_at_argmin', 'n_windows']``.
    """
    site_keys = list(site_keys)
    window_keys = list(window_keys)
    sw_keys = site_keys + window_keys

    null_valid = null_agg.filter(
        pl.col(stat_col).is_not_null() & pl.col(stat_col).is_not_nan()
    )

    real_subset = real_agg.select(sw_keys + [stat_col])
    merged = real_subset.join(
        null_valid.select(sw_keys + [stat_col]).rename({stat_col: "_null_stat"}),
        on=sw_keys,
        how="left",
    )
    # Cast real_statistic to Float64 so the output schema is stable across
    # callers — GPU outputs are Float32, but ``t_stat`` and TFCE-enhanced
    # statistics are computed in Float64; without this cast, stacking
    # per-flavor frames in ``stage1_gate`` raises a polars SchemaError on
    # strict concat.
    pw = (
        merged.group_by(sw_keys)
        .agg(
            pl.col(stat_col).first().cast(pl.Float64).alias("real_statistic"),
            (
                (pl.col("_null_stat") >= pl.col(stat_col))
                & pl.col("_null_stat").is_not_null()
            ).sum().alias("_ge_count"),
            pl.col("_null_stat").is_not_null().sum().alias("K_w"),
        )
        .with_columns(
            ((pl.col("_ge_count") + 1) / (pl.col("K_w") + 1)).alias("pointwise_p"),
        )
        .drop("_ge_count")
    )

    argmin_renames = [pl.col(wk).first().alias(f"argmin_{wk}") for wk in window_keys]
    return (
        pw.sort(["pointwise_p", "real_statistic"], descending=[False, True])
        .group_by(site_keys, maintain_order=True)
        .agg(
            pl.col("pointwise_p").first().alias("min_pointwise_p"),
            *argmin_renames,
            pl.col("real_statistic").first().alias("real_at_argmin"),
            pl.col("K_w").first().alias("K_at_argmin"),
            pl.col("pointwise_p").len().alias("n_windows"),
        )
    )


def stage1_gate(
    real_agg: pl.DataFrame,
    null_agg: pl.DataFrame,
    *,
    site_keys: Sequence[str],
    flavors: Sequence[FlavorSpec],
    p_max: float,
    window_keys: Sequence[str] = ("smin", "smax"),
    perm_key: str = "permutation_idx",
) -> tuple[set[tuple], pl.DataFrame]:
    """Run the stage-1 escalation gate over a list of statistical flavors.

    For each ``FlavorSpec``: optionally TFCE-enhance via
    ``tfce_1d_per_site`` along windows (per (site[, perm]) group), then
    compute ``min_pointwise_p_per_site``. Take the global min over
    flavors per site.

    Args:
        real_agg, null_agg: pre-aggregated tables produced by
            ``aggregate_<decoder>``. Both must contain every column
            named in ``flavor.stat_col`` for each flavor.
        site_keys, window_keys, perm_key: as in
            ``min_pointwise_p_per_site``.
        flavors: per-decoder flavor list (from
            ``src.models.causal6_aggregates``). Each FlavorSpec
            decides which column is read and whether TFCE is applied.
        p_max: a site whose global-min pointwise_p ≤ p_max is flagged
            for escalation.

    Returns:
        borderline_keys: set of site-key tuples whose
            ``min_pointwise_p_global`` ≤ p_max. Tuple ordering matches
            ``site_keys``.
        gate_log: one row per site with columns
            ``site_keys + ['min_pointwise_p_<flavor>'...,
            'min_pointwise_p_global', 'argmin_flavor',
            'argmin_<wk>'..., 'real_at_argmin', 'n_windows', 'escalated']``.
    """
    site_keys = list(site_keys)
    window_keys = list(window_keys)

    long_parts: list[pl.DataFrame] = []
    for flavor in flavors:
        flavor_name = _flavor_name(flavor)
        stat_col = flavor.stat_col

        if flavor.apply_tfce:
            real_in = real_agg.select(site_keys + window_keys + [stat_col]).rename(
                {stat_col: "statistic"}
            )
            null_in = null_agg.select(
                site_keys + window_keys + [perm_key, stat_col]
            ).rename({stat_col: "statistic"})
            real_enh = tfce_1d_per_site(
                real_in,
                site_keys=site_keys, window_keys=window_keys,
                stat_col="statistic", threshold=flavor.tfce_threshold,
            )
            null_enh = tfce_1d_per_site(
                null_in,
                site_keys=site_keys, window_keys=window_keys,
                perm_key=perm_key, stat_col="statistic",
                threshold=flavor.tfce_threshold,
            )
            per_site = min_pointwise_p_per_site(
                real_enh, null_enh,
                site_keys=site_keys, window_keys=window_keys,
                perm_key=perm_key, stat_col="statistic",
            )
        else:
            per_site = min_pointwise_p_per_site(
                real_agg, null_agg,
                site_keys=site_keys, window_keys=window_keys,
                perm_key=perm_key, stat_col=stat_col,
            )

        long_parts.append(
            per_site.with_columns(pl.lit(flavor_name).alias("_flavor"))
        )

    long = pl.concat(long_parts)

    # Pivot min_pointwise_p across flavors → one column per flavor.
    flavor_names = [_flavor_name(f) for f in flavors]
    site_list = real_agg.select(site_keys).unique()
    base = site_list
    for fn in flavor_names:
        sub = long.filter(pl.col("_flavor") == fn).select(
            site_keys
            + [pl.col("min_pointwise_p").alias(f"min_pointwise_p_{fn}")]
        )
        base = base.join(sub, on=site_keys, how="left")

    # Global min + argmin flavor + argmin window per site.
    argmin_window_cols = [pl.col(f"argmin_{wk}").first() for wk in window_keys]
    argmin_per_site = (
        long.sort(
            ["min_pointwise_p", "real_at_argmin"], descending=[False, True],
        )
        .group_by(site_keys, maintain_order=True)
        .agg(
            pl.col("min_pointwise_p").first().alias("min_pointwise_p_global"),
            pl.col("_flavor").first().alias("argmin_flavor"),
            *argmin_window_cols,
            pl.col("real_at_argmin").first(),
            pl.col("n_windows").first(),
        )
    )

    gate_log = base.join(argmin_per_site, on=site_keys, how="inner").with_columns(
        (pl.col("min_pointwise_p_global") <= p_max).alias("escalated")
    )

    borderline_keys = set(
        gate_log.filter(pl.col("escalated"))
        .select(site_keys)
        .iter_rows()
    )
    return borderline_keys, gate_log


def filter_null_to_borderline(
    null_scores: pl.DataFrame,
    borderline_keys: set[tuple],
    *,
    site_keys: Sequence[str],
) -> pl.DataFrame:
    """Filter raw null_scores to rows whose ``site_keys`` tuple is in ``borderline_keys``.

    ``run_<flavor>_permutations`` only takes a coarse ``electrode_idxs``
    subset — when ``site_keys`` includes additional dimensions
    (``phoneme_pair``, ``word_end``), this filters down to exactly the
    site tuples flagged at stage 1 so stage-2 perms don't waste rows on
    sites that were already cleared.
    """
    site_keys = list(site_keys)
    if not borderline_keys:
        return null_scores.head(0)

    keys_data = {
        sk: [k[i] for k in borderline_keys] for i, sk in enumerate(site_keys)
    }
    keys_df = pl.DataFrame(keys_data)
    for sk in site_keys:
        target_dtype = null_scores.schema[sk]
        if keys_df.schema[sk] != target_dtype:
            keys_df = keys_df.with_columns(pl.col(sk).cast(target_dtype))
    return null_scores.join(keys_df, on=site_keys, how="semi")
