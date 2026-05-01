"""
Two-stage adaptive-K permutation null gate primitives for the causal6
``*_null`` notebooks.

The flow per subject is:

1. Run K1 stage-1 permutations across all speech-responsive electrodes.
2. Aggregate via ``aggregate_<decoder>`` from
   ``src.models.causal6_aggregates`` (single source of truth shared
   with the summarize step) to per-(site, window[, perm]) statistics.
3. ``stage1_gate`` computes a per-flavor max-stat (and TFCE) corrected
   p-value at K1 — the same statistic the summarize step computes at
   K1+K2 — and takes the global min over flavors per site. Sites whose
   ``min_corrected_p_global`` ≤ ``escalate_corrected_p_max`` are flagged
   as borderline.
4. The notebook runs K2 additional permutations on the borderline
   electrodes (non-overlapping seed range with stage 1 for
   determinism), filters down to the exact borderline site keys via
   ``filter_null_to_borderline``, concatenates with stage 1, and
   writes one merged ``null_scores.parquet``.

Why corrected p (not min-over-windows pointwise_p) at the gate:

The original gate used ``min over windows of pointwise_p`` against a
``p_max=0.20`` threshold. With W~138 windows in the behavior peak-search
range and K1=1000, the minimum pointwise_p over 138 windows concentrates
near 1/W ≈ 0.007 even under the null, so essentially every site cleared
the gate and stage 2 ran at K2=9000 for ~all electrodes. On
high-electrode subjects (≥80) that blew the RAM budget at the
post-stage-2 ``filter_null_to_borderline().collect()`` and
``pl.concat([null_stage1, null_stage2])``.

Switching to corrected p makes the escalation rate scale with ``p_max``
rather than with the window count, restoring the original two-stage
intent (only refit borderline sites at K2).

TFCE flavors are still gated the same way: TFCE-enhance the per-(site,
perm) statistics first, then run the corrected-p test. Cost is pure
polars / numpy work — no extra GPU refits.
"""

from __future__ import annotations

import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import polars as pl

from src.models.causal6_aggregates import FlavorSpec
from src.models.significance import null_standardized_peak_test, tfce_1d_per_site


@contextmanager
def stage2_spill_dir(parent: Path, name: str = "_stage2_spill") -> Generator[Path]:
    """Scratch directory for stage-2 permutation-null parquet shards.

    Creates ``parent/name`` empty (wiping any leftovers from an aborted
    prior run), yields it for use as the ``spill_dir`` argument to
    ``run_*_permutations``, and removes the directory on exit so shards
    never outlive the rule.

    All five ``*_null`` notebooks share this pattern: stream stage-2
    perm chunks to disk, ``pl.scan_parquet`` + ``filter_null_to_borderline``
    lazily, ``.collect()`` the (much smaller) filtered result. The
    context manager keeps the lifecycle in one place.
    """
    spill = parent / name
    if spill.exists():
        shutil.rmtree(spill)
    spill.mkdir(parents=True)
    try:
        yield spill
    finally:
        shutil.rmtree(spill, ignore_errors=True)


_GATE_TAG = "CAUSAL6-GATE"


def log_stage1_gate(
    subject: str,
    *,
    n_permutations_stage1: int,
    p_max: float,
    gate_log: pl.DataFrame,
    n_borderline: int,
    print_top_k: int = 50,
) -> None:
    """Print a stage-1 escalation summary to stdout immediately after the gate.

    Output is tagged with ``[CAUSAL6-GATE/<subject>]`` so it's grep-able
    in the snakemake log. ``flush=True`` plus the runner's
    ``log_output=True`` ensure the line lands in the rule's log before
    stage 2's GPU work begins (which is the point of the two-stage
    design — surfacing the escalation count ASAP).

    Args:
        subject: subject id, included in the tag.
        n_permutations_stage1: K1 (printed for context).
        p_max: gate threshold (printed for context).
        gate_log: full gate_log returned by ``stage1_gate``; used for
            the corrected-p quantile summary and the top-K escalated
            sites table.
        n_borderline: ``len(borderline_keys)``.
        print_top_k: cap on the escalated-sites table dump (avoid
            multi-thousand-row dumps when the gate escalates broadly).
    """
    n_total = gate_log.height
    print(
        f"[{_GATE_TAG}/{subject}] stage1 K={n_permutations_stage1}: "
        f"{n_borderline}/{n_total} sites escalated "
        f"(min_corrected_p_global ≤ {p_max})",
        flush=True,
    )
    if n_total > 0:
        q = gate_log.select([
            pl.col("min_corrected_p_global").min().alias("min"),
            pl.col("min_corrected_p_global").quantile(0.25).alias("q25"),
            pl.col("min_corrected_p_global").median().alias("median"),
            pl.col("min_corrected_p_global").quantile(0.75).alias("q75"),
            pl.col("min_corrected_p_global").max().alias("max"),
        ]).row(0, named=True)
        print(
            f"[{_GATE_TAG}/{subject}] corrected_p over {n_total} sites: "
            f"min={q['min']:.4g} q25={q['q25']:.4g} median={q['median']:.4g} "
            f"q75={q['q75']:.4g} max={q['max']:.4g}",
            flush=True,
        )
    if n_borderline > 0:
        print(f"[{_GATE_TAG}/{subject}] escalated sites (top {print_top_k}):", flush=True)
        print(
            gate_log.filter(pl.col("escalated"))
            .sort("min_corrected_p_global")
            .head(print_top_k)
            .to_pandas()
            .to_string(index=False),
            flush=True,
        )


def log_stage2_done(
    subject: str,
    *,
    n_permutations_stage2: int,
    n_borderline_electrodes: int,
    n_borderline_sites: int,
    null_height: int,
) -> None:
    """Print a stage-2 completion summary to stdout. See ``log_stage1_gate``."""
    print(
        f"[{_GATE_TAG}/{subject}] stage2 K={n_permutations_stage2} on "
        f"{n_borderline_electrodes} electrodes ({n_borderline_sites} sites "
        f"after site_keys filter); merged null has {null_height} rows",
        flush=True,
    )


def log_stage2_skipped(
    subject: str,
    *,
    n_borderline: int,
    n_permutations_stage2: int,
) -> None:
    """Print a stage-2-skipped notice (no borderline sites or K2=0)."""
    print(
        f"[{_GATE_TAG}/{subject}] stage2 skipped "
        f"(borderline={n_borderline}, K2={n_permutations_stage2})",
        flush=True,
    )


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
    compute the max-stat corrected p-value via
    ``null_standardized_peak_test`` (the same correction the production
    summarize step applies at K1+K2). Take the global min over flavors
    per site.

    Args:
        real_agg, null_agg: pre-aggregated tables produced by
            ``aggregate_<decoder>``. Both must contain every column
            named in ``flavor.stat_col`` for each flavor.
        site_keys, window_keys, perm_key: as in
            ``null_standardized_peak_test``.
        flavors: per-decoder flavor list (from
            ``src.models.causal6_aggregates``). Each FlavorSpec
            decides which column is read and whether TFCE is applied.
        p_max: a site whose ``min_corrected_p_global ≤ p_max`` is
            flagged for escalation. Interpret as a corrected p-value
            (not pointwise) — same scale the summarize step uses.

    Returns:
        borderline_keys: set of site-key tuples whose
            ``min_corrected_p_global`` ≤ p_max. Tuple ordering matches
            ``site_keys``.
        gate_log: one row per site with columns
            ``site_keys + ['corrected_p_<flavor>'...,
            'min_corrected_p_global', 'argmin_flavor',
            'peak_<wk>'..., 'real_at_peak', 'n_permutations',
            'escalated']``.
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
            real_for_test = tfce_1d_per_site(
                real_in,
                site_keys=site_keys, window_keys=window_keys,
                stat_col="statistic", threshold=flavor.tfce_threshold,
            )
            null_for_test = tfce_1d_per_site(
                null_in,
                site_keys=site_keys, window_keys=window_keys,
                perm_key=perm_key, stat_col="statistic",
                threshold=flavor.tfce_threshold,
            )
            test_stat_col = "statistic"
        else:
            # Cast to Float64 so the per-flavor concat below can't trip
            # over fold_mean (Float32 from GPU) vs t_stat (Float64).
            real_for_test = real_agg.select(
                site_keys + window_keys + [stat_col]
            ).with_columns(pl.col(stat_col).cast(pl.Float64))
            null_for_test = null_agg.select(
                site_keys + window_keys + [perm_key, stat_col]
            ).with_columns(pl.col(stat_col).cast(pl.Float64))
            test_stat_col = stat_col

        peak_summary, _ = null_standardized_peak_test(
            real_for_test, null_for_test,
            site_keys=site_keys, window_keys=window_keys,
            perm_key=perm_key, stat_col=test_stat_col,
        )
        flavor_part = peak_summary.select(
            site_keys
            + [pl.col(f"peak_{wk}") for wk in window_keys]
            + [
                pl.col("real_statistic"),
                pl.col("p_value").alias("corrected_p"),
                pl.col("n_permutations"),
            ]
        ).with_columns(pl.lit(flavor_name).alias("_flavor"))
        long_parts.append(flavor_part)

    long = pl.concat(long_parts)

    # Pivot corrected_p across flavors → one column per flavor.
    flavor_names = [_flavor_name(f) for f in flavors]
    site_list = real_agg.select(site_keys).unique()
    base = site_list
    for fn in flavor_names:
        sub = long.filter(pl.col("_flavor") == fn).select(
            site_keys
            + [pl.col("corrected_p").alias(f"corrected_p_{fn}")]
        )
        base = base.join(sub, on=site_keys, how="left")

    # Global min + argmin flavor + peak window per site.
    peak_window_cols = [pl.col(f"peak_{wk}").first() for wk in window_keys]
    argmin_per_site = (
        long.sort(
            ["corrected_p", "real_statistic"], descending=[False, True],
        )
        .group_by(site_keys, maintain_order=True)
        .agg(
            pl.col("corrected_p").first().alias("min_corrected_p_global"),
            pl.col("_flavor").first().alias("argmin_flavor"),
            *peak_window_cols,
            pl.col("real_statistic").first().alias("real_at_peak"),
            pl.col("n_permutations").first(),
        )
    )

    gate_log = base.join(argmin_per_site, on=site_keys, how="inner").with_columns(
        (pl.col("min_corrected_p_global") <= p_max).alias("escalated")
    )

    borderline_keys = set(
        gate_log.filter(pl.col("escalated"))
        .select(site_keys)
        .iter_rows()
    )
    return borderline_keys, gate_log


def filter_null_to_borderline(
    null_scores: pl.DataFrame | pl.LazyFrame,
    borderline_keys: set[tuple],
    *,
    site_keys: Sequence[str],
) -> pl.DataFrame | pl.LazyFrame:
    """Filter raw null_scores to rows whose ``site_keys`` tuple is in ``borderline_keys``.

    ``run_<flavor>_permutations`` only takes a coarse ``electrode_idxs``
    subset — when ``site_keys`` includes additional dimensions
    (``phoneme_pair``, ``word_end``), this filters down to exactly the
    site tuples flagged at stage 1 so stage-2 perms don't waste rows on
    sites that were already cleared.

    Accepts both ``DataFrame`` and ``LazyFrame``; returns the same kind.
    The LazyFrame path lets callers stream raw stage-2 shards from disk,
    apply this filter lazily, and only collect the (much smaller) result —
    avoiding the full-null materialization that triggers OOMs on
    high-electrode-count subjects.
    """
    site_keys = list(site_keys)
    is_lazy = isinstance(null_scores, pl.LazyFrame)
    if not borderline_keys:
        return null_scores.head(0)

    schema = null_scores.collect_schema() if is_lazy else null_scores.schema
    keys_data = {
        sk: [k[i] for k in borderline_keys] for i, sk in enumerate(site_keys)
    }
    keys_df = pl.DataFrame(keys_data)
    for sk in site_keys:
        target_dtype = schema[sk]
        if keys_df.schema[sk] != target_dtype:
            keys_df = keys_df.with_columns(pl.col(sk).cast(target_dtype))
    if isinstance(null_scores, pl.LazyFrame):
        return null_scores.join(keys_df.lazy(), on=site_keys, how="semi")
    return null_scores.join(keys_df, on=site_keys, how="semi")
