"""Acoustic-step contrast (behavior-controlled) bootstrap and shared summary functions.

Provides:
  bootstrap_cell_acoustic  — R replicates of the extreme-step HGA contrast on
                             ambiguous trials, with behavioral report controlled.
  per_window_summary       — aggregates (median, CI, emp_p) over bootstrap rows.
  per_cell_best            — picks the best window per cell from per_window.
  step_tuning_curve        — R replicates of the windowed mean HGA at EVERY
                             qualifying step (not just s_lo/s_hi) — the
                             "per-step profile" facet of the same bootstrap
                             draw, for gradient/tuning inspection.
  step_tuning_summary      — aggregates step_tuning_curve rows to one
                             (mean, CI) row per step.
  exclude_overlapping_windows — drops per-window rows overlapping a per-cell
                             excluded range (e.g. the acoustic-peak window),
                             so per_cell_best can be re-run to find the best
                             window that DOESN'T overlap it (the "late,
                             excl. phon-peak" variant of the tuning curve).
  step_tuning_pass          — runs step_tuning_curve/summary over every row
                             of a per_cell table (needs best_smin/best_smax),
                             tagging output with a window_kind label. Used to
                             produce both the global-best and late-window
                             tuning curves from the same per_cell shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(__file__).parent))
from _within_completion import (  # noqa: E402
    extract_hga,
    n_per_class_from_per_step,
    per_step_class_counts,
    resolve_behavior_col,
    searchlight_mean_diff,
    select_cell_trials_bootstrap_perstep,
)

CI_LOW = 2.5
CI_HIGH = 97.5


def bootstrap_cell_acoustic(
    *,
    md_pp,
    hga: np.ndarray,
    bhv_col: str,
    word_end: str,
    qualifying_steps: list[int],
    acoustic_peak_auc: float,
    search_smin: int,
    search_smax: int,
    window_size: int = 10,
    stride: int = 10,
    R: int = 1000,
    base_seed: int = 0,
) -> tuple[list[dict], int, int] | None:
    """R bootstrap replicates of the extreme-step acoustic contrast on ambiguous trials.

    Contrast: s_hi = max(qualifying_steps) vs s_lo = min(qualifying_steps).
    Behavioral report is balanced within each step (50/50 by construction of
    select_cell_trials_bootstrap_perstep), so the contrast isolates the acoustic
    step effect independent of reported percept.

    The RNG call sequence is IDENTICAL to select_cell_trials_bootstrap under the
    same (per_step, seed): the same rng.choice calls in the same order are made.
    Both contrasts therefore derive from the same bootstrap replicates.

    Null: for each behavior class b, pool d[s_hi][b] + d[s_lo][b] and re-split
    preserving the same step-group sizes (50/50 behavior balance in each pseudo-step).
    Tests whether the observed step difference could arise by chance under the
    same step-size allocation.

    Schema is identical to b4_bootstrap.parquet rows, with two extra columns:
        s_lo : int  — lower extreme step  (= min(qualifying_steps))
        s_hi : int  — upper extreme step  (= max(qualifying_steps))
    mean_diff_aligned == mean_diff_raw (step order fixes polarity; no sign flip).

    Returns (rows, s_lo, s_hi) or None if fewer than 2 qualifying steps or
    either extreme step has insufficient trials.
    """
    if len(qualifying_steps) < 2:
        return None
    s_lo = int(min(qualifying_steps))
    s_hi = int(max(qualifying_steps))

    per_step = per_step_class_counts(
        md_pp, word_end=word_end,
        qualifying_steps=qualifying_steps,
        group_col=bhv_col,
    )

    if s_lo not in per_step or s_hi not in per_step:
        return None

    classes = sorted(per_step[s_lo].keys())
    n_s_lo = min(len(v) for v in per_step[s_lo].values())
    n_s_hi = min(len(v) for v in per_step[s_hi].values())
    if n_s_lo == 0 or n_s_hi == 0:
        return None

    rows: list[dict] = []
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        d = select_cell_trials_bootstrap_perstep(per_step, rng=rng)

        # Acoustic contrast: pos = s_hi pooled over behaviors; neg = s_lo pooled.
        # Each behavior class is 50/50 by construction → behavior balanced.
        pos_parts = [d[s_hi][b] for b in classes if s_hi in d and b in d.get(s_hi, {})]
        neg_parts = [d[s_lo][b] for b in classes if s_lo in d and b in d.get(s_lo, {})]
        if not pos_parts or not neg_parts:
            continue
        pos_idx = np.concatenate(pos_parts)
        neg_idx = np.concatenate(neg_parts)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        windows = searchlight_mean_diff(
            hga, pos_idx, neg_idx,
            search_smin=search_smin, search_smax=search_smax,
            window_size=window_size, stride=stride,
        )

        # Behavior-controlled null: per behavior, pool s_hi+s_lo, re-split by size.
        null_pos_parts: list[np.ndarray] = []
        null_neg_parts: list[np.ndarray] = []
        for b in classes:
            hi_b = d.get(s_hi, {}).get(b, np.array([], dtype=int))
            lo_b = d.get(s_lo, {}).get(b, np.array([], dtype=int))
            n_hi_b, n_lo_b = len(hi_b), len(lo_b)
            if n_hi_b == 0 and n_lo_b == 0:
                continue
            pool = np.concatenate([hi_b, lo_b])
            rng.shuffle(pool)
            null_pos_parts.append(pool[:n_hi_b])
            null_neg_parts.append(pool[n_hi_b:])

        if null_pos_parts and null_neg_parts:
            null_pos = np.concatenate(null_pos_parts)
            null_neg = np.concatenate(null_neg_parts)
            null_windows = searchlight_mean_diff(
                hga, null_pos, null_neg,
                search_smin=search_smin, search_smax=search_smax,
                window_size=window_size, stride=stride,
            )
            null_diff_by_window = {(w.smin, w.smax): w.mean_diff for w in null_windows}
        else:
            null_diff_by_window = {}

        n_per_class = len(pos_idx)
        for w in windows:
            mean_diff_raw = w.mean_diff
            null_raw = null_diff_by_window.get((w.smin, w.smax), float("nan"))
            rows.append({
                "replicate": r,
                "smin": w.smin, "smax": w.smax,
                "tmin": w.smin / epoch_sfreq + epoch_tmin,
                "tmax": w.smax / epoch_sfreq + epoch_tmin,
                "mean_pos_raw": w.mean_pos,
                "mean_neg_raw": w.mean_neg,
                "mean_diff_raw": mean_diff_raw,
                "mean_diff_aligned": mean_diff_raw,  # step order fixes polarity
                "mean_diff_aligned_null": null_raw,
                "n_per_class": n_per_class,
                "acoustic_peak_auc": float(acoustic_peak_auc),
                "s_lo": s_lo,
                "s_hi": s_hi,
            })
    return rows, s_lo, s_hi


def per_window_summary(boot: pl.DataFrame, cell_keys: list[str]) -> pl.DataFrame:
    """Aggregate bootstrap rows into per-(cell × window) summary statistics.

    If `boot` carries per-class raw activation columns `mean_pos`/`mean_neg`
    (e.g. from `bootstrap_A_site`), their per-window medians are also emitted
    as `mean_pos_med`/`mean_neg_med`. Callers whose bootstrap rows lack these
    columns (e.g. `acoustic_on_ambiguous.py`) are unaffected.
    """
    if boot.height == 0:
        return pl.DataFrame()
    class_aggs = []
    if "mean_pos" in boot.columns:
        class_aggs.append(pl.col("mean_pos").median().alias("mean_pos_med"))
    if "mean_neg" in boot.columns:
        class_aggs.append(pl.col("mean_neg").median().alias("mean_neg_med"))
    grouped = (
        boot
        .group_by(cell_keys + ["smin", "smax", "tmin", "tmax"])
        .agg(
            pl.col("mean_diff_raw").median().alias("mean_diff_raw_med"),
            pl.col("mean_diff_raw").quantile(CI_LOW / 100).alias("mean_diff_raw_ci_lo"),
            pl.col("mean_diff_raw").quantile(CI_HIGH / 100).alias("mean_diff_raw_ci_hi"),
            pl.col("mean_diff_raw").std().alias("mean_diff_raw_std"),
            (pl.col("mean_diff_raw") <= 0).cast(pl.Float64).mean().alias("frac_raw_le0"),
            (pl.col("mean_diff_raw") >= 0).cast(pl.Float64).mean().alias("frac_raw_ge0"),

            pl.col("mean_diff_aligned").median().alias("mean_diff_aligned_med"),
            pl.col("mean_diff_aligned").mean().alias("mean_diff_aligned_mean"),
            pl.col("mean_diff_aligned").quantile(CI_LOW / 100).alias("mean_diff_aligned_ci_lo"),
            pl.col("mean_diff_aligned").quantile(CI_HIGH / 100).alias("mean_diff_aligned_ci_hi"),
            pl.col("mean_diff_aligned").std().alias("mean_diff_aligned_std"),
            (pl.col("mean_diff_aligned") <= 0).cast(pl.Float64).mean().alias("frac_aligned_le0"),
            (pl.col("mean_diff_aligned") >= 0).cast(pl.Float64).mean().alias("frac_aligned_ge0"),

            pl.col("n_per_class").first().alias("n_per_class"),
            pl.col("acoustic_peak_auc").first().alias("acoustic_peak_auc"),
            pl.col("replicate").max().alias("R_replicates"),

            *class_aggs,
        )
    )
    grouped = grouped.with_columns([
        pl.min_horizontal(
            2 * pl.min_horizontal("frac_raw_le0", "frac_raw_ge0"),
            pl.lit(1.0),
        ).alias("emp_p_raw"),
        pl.min_horizontal(
            2 * pl.min_horizontal("frac_aligned_le0", "frac_aligned_ge0"),
            pl.lit(1.0),
        ).alias("emp_p_aligned"),
        ((pl.col("mean_diff_raw_ci_lo") > 0) | (pl.col("mean_diff_raw_ci_hi") < 0))
            .alias("ci_raw_excludes_zero"),
        ((pl.col("mean_diff_aligned_ci_lo") > 0) | (pl.col("mean_diff_aligned_ci_hi") < 0))
            .alias("ci_aligned_excludes_zero"),
    ])
    return grouped.sort(cell_keys + ["smin"])


def per_cell_best(per_window: pl.DataFrame, cell_keys: list[str]) -> pl.DataFrame:
    """Pick best window per cell (largest |median aligned diff|, fallback to |raw|)."""
    if per_window.height == 0:
        return pl.DataFrame()
    return (
        per_window
        .with_columns([
            pl.when(pl.col("mean_diff_aligned_med").is_not_null())
                .then(pl.col("mean_diff_aligned_med").abs())
                .otherwise(pl.col("mean_diff_raw_med").abs())
                .alias("__rank"),
        ])
        .sort(cell_keys + ["__rank"], descending=[False] * len(cell_keys) + [True])
        .group_by(cell_keys, maintain_order=True)
        .head(1)
        .drop("__rank")
        .rename({
            "smin": "best_smin", "smax": "best_smax",
            "tmin": "best_tmin", "tmax": "best_tmax",
            "mean_diff_raw_med": "best_mean_diff_raw_med",
            "mean_diff_raw_ci_lo": "best_mean_diff_raw_ci_lo",
            "mean_diff_raw_ci_hi": "best_mean_diff_raw_ci_hi",
            "mean_diff_aligned_med": "best_mean_diff_aligned_med",
            "mean_diff_aligned_ci_lo": "best_mean_diff_aligned_ci_lo",
            "mean_diff_aligned_ci_hi": "best_mean_diff_aligned_ci_hi",
            "emp_p_raw": "best_emp_p_raw",
            "emp_p_aligned": "best_emp_p_aligned",
            "ci_raw_excludes_zero": "best_ci_raw_excludes_zero",
            "ci_aligned_excludes_zero": "best_ci_aligned_excludes_zero",
        })
    )


def step_tuning_curve(
    per_step: dict[int, dict[int, np.ndarray]],
    hga: np.ndarray,
    *,
    window_smin: int,
    window_smax: int,
    R: int = 1000,
    base_seed: int = 0,
) -> list[dict]:
    """R bootstrap replicates of the windowed mean HGA at every qualifying step.

    Third facet of the shared bootstrap draw (see
    select_cell_trials_bootstrap_perstep's docstring: "Per-step profile:
    concat_b d[s][b] for each qualifying step s"). For each qualifying step s,
    pools both behavior classes (50/50 by construction) and takes the mean
    HGA over [window_smin, window_smax) per replicate. Same RNG call sequence
    as bootstrap_cell_acoustic under matching (per_step, seed) — this is an
    orthogonal readout of the same replicates used for the s_lo/s_hi contrast,
    not a new bootstrap draw.

    Unlike bootstrap_cell_acoustic (searchlight over many windows, extreme
    steps only), this fixes ONE window (typically the cell's best_smin/
    best_smax from the s_lo/s_hi contrast) and evaluates ALL qualifying steps,
    to visualize whether the acoustic effect is graded across the continuum
    or concentrated at the extremes.

    Returns one row per (step, replicate): {replicate, step, mean_windowed}.
    Aggregate with step_tuning_summary.
    """
    steps_sorted = sorted(per_step.keys())
    rows: list[dict] = []
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        d = select_cell_trials_bootstrap_perstep(per_step, rng=rng)
        for s in steps_sorted:
            if s not in d:
                continue
            idx = np.concatenate(list(d[s].values()))
            if len(idx) == 0:
                continue
            rows.append({
                "replicate": r,
                "step": s,
                "mean_windowed": float(hga[idx, window_smin:window_smax].mean()),
            })
    return rows


def step_tuning_summary(
    rows: list[dict],
    ci_low: float = CI_LOW,
    ci_high: float = CI_HIGH,
) -> list[dict]:
    """Aggregate step_tuning_curve rows to one (mean, CI) row per step."""
    if not rows:
        return []
    df = pl.DataFrame(rows)
    return (
        df.group_by("step")
        .agg(
            pl.col("mean_windowed").mean().alias("mean"),
            pl.col("mean_windowed").median().alias("median"),
            pl.col("mean_windowed").quantile(ci_low / 100).alias("ci_lo"),
            pl.col("mean_windowed").quantile(ci_high / 100).alias("ci_hi"),
            pl.col("mean_windowed").std().alias("std"),
            pl.len().alias("n_replicates"),
        )
        .sort("step")
        .to_dicts()
    )


def exclude_overlapping_windows(
    per_window: pl.DataFrame,
    excl: pl.DataFrame,
    cell_keys: list[str],
    *,
    excl_smin_col: str = "phon_smin",
    excl_smax_col: str = "phon_smax",
) -> pl.DataFrame:
    """Drop per-window rows whose [smin, smax) overlaps a per-cell excluded range.

    `excl` must carry cell_keys + excl_smin_col + excl_smax_col (one row per
    cell; extra columns are ignored). Intended use: re-run per_cell_best on
    the result to find the best window that does NOT overlap the site's
    acoustic-peak window (phon_smin/phon_smax) — the "late, excludes
    phon-peak" variant of the step-tuning curve, isolating a later effect
    from cells whose global-best window happens to land on/near the
    transient acoustic response.

    Overlap test: [smin, smax) and [excl_smin, excl_smax) overlap iff
    smin < excl_smax and smax > excl_smin. Cells absent from `excl` are kept
    unfiltered (nothing to exclude for them).
    """
    if per_window.height == 0:
        return per_window
    joined = per_window.join(
        excl.select(cell_keys + [excl_smin_col, excl_smax_col]).unique(subset=cell_keys),
        on=cell_keys, how="left",
    )
    overlaps = (
        (pl.col("smin") < pl.col(excl_smax_col))
        & (pl.col("smax") > pl.col(excl_smin_col))
    )
    return (
        joined
        .filter(pl.col(excl_smin_col).is_null() | ~overlaps)
        .drop([excl_smin_col, excl_smax_col])
    )


def step_tuning_pass(
    per_cell: pl.DataFrame,
    epochs_dict: dict,
    *,
    cell_keys: list[str],
    R: int,
    window_kind: str,
    desc: str = "step tuning",
) -> list[dict]:
    """Run step_tuning_curve/summary for every row of `per_cell`.

    `per_cell` needs columns: cell_keys + best_smin/best_smax (the window to
    evaluate) + qualifying_steps (comma-joined string). Rows with a null
    best_smin/best_smax (e.g. no window survived exclusion — see
    exclude_overlapping_windows) are skipped.

    Output rows are tagged with `window_kind` (e.g. "global_best" or
    "late_excl_phon") so multiple passes can be concatenated into one
    parquet and disambiguated downstream.

    Returns a flat list of dicts: cell_keys + window_kind + best_smin +
    best_smax + step_tuning_summary()'s per-step fields (step, mean, median,
    ci_lo, ci_hi, std, n_replicates). Ready for pl.DataFrame(...).
    """
    rows: list[dict] = []
    for row in tqdm(per_cell.iter_rows(named=True), total=per_cell.height, desc=desc):
        subj = row["subject"]
        if subj not in epochs_dict:
            continue
        if row.get("best_smin") is None or row.get("best_smax") is None:
            continue
        ep = epochs_dict[subj]
        md = ep.metadata
        bhv_col = resolve_behavior_col(md)
        pp_mask = (md["phoneme_pair"] == row["phoneme_pair"]).values
        ep_pp = ep[pp_mask]
        md_pp = md[pp_mask].reset_index(drop=True)
        hga = extract_hga(ep_pp, int(row["electrode_idx"]))
        steps = [int(s) for s in row["qualifying_steps"].split(",") if s]
        per_step = per_step_class_counts(
            md_pp, word_end=row["word_end"],
            qualifying_steps=steps, group_col=bhv_col,
        )
        raw = step_tuning_curve(
            per_step, hga,
            window_smin=int(row["best_smin"]), window_smax=int(row["best_smax"]),
            R=R,
        )
        for d in step_tuning_summary(raw):
            rows.append({
                "subject": subj,
                "electrode_idx": int(row["electrode_idx"]),
                "phoneme_pair": row["phoneme_pair"],
                "word_end": row["word_end"],
                "window_kind": window_kind,
                "best_smin": int(row["best_smin"]),
                "best_smax": int(row["best_smax"]),
                **d,
            })
    return rows
