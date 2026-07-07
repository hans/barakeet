"""Acoustic-step contrast (behavior-controlled) bootstrap and shared summary functions.

Provides:
  bootstrap_cell_acoustic  — R replicates of the extreme-step HGA contrast on
                             ambiguous trials, with behavioral report controlled.
  per_window_summary       — aggregates (median, CI, emp_p) over bootstrap rows.
  per_cell_best            — picks the best window per cell from per_window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(__file__).parent))
from _within_completion import (  # noqa: E402
    n_per_class_from_per_step,
    per_step_class_counts,
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
    """Aggregate bootstrap rows into per-(cell × window) summary statistics."""
    if boot.height == 0:
        return pl.DataFrame()
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
