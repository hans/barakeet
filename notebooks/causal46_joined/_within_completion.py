"""Shared trial-selection / HGA-extraction / searchlight-t-test primitives
used by star_plots.py (JON-43), calibration.py and t_tests.py (JON-44).

Notebook-local on purpose: src/ stays untouched while JON-41 Group B is in
flux. Promote to src/ only if a third caller appears outside this directory.

Note on gallery vs. t-test trial subsets (B4 only):
    `star_plots.py::matched_n_star_plot` does pool-then-balance (pool all
    qualifying steps, subsample the larger behavior class to the smaller
    one) — this is what the visual gallery shows. `select_cell_trials` here
    does per-step balanced subsampling at a fixed `n_per_group` then pools
    across steps — this is what the t-tests run on. The two subsets agree
    for B3 (single step, single subsample) but diverge for B4. The
    filtered-gallery hook in `t_tests.py` joins per (site, word_end) key
    only — not by identical trial subset — so B4 gallery PDFs are flagged
    powered/significant based on the t-test's per-step-balanced subsample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


BEHAVIOR_COL_PREFERENCE: tuple[str, ...] = (
    "behavior_dummy_forced",
    "behavior_categorical",
)


def resolve_behavior_col(md: pd.DataFrame) -> str:
    """Pick the first column in BEHAVIOR_COL_PREFERENCE present in `md`.

    Mirrors the fallback in `matched_n_star_plot` so the t-tests don't
    explode on subjects whose epoch metadata predates the
    `behavior_dummy_forced` rename.
    """
    for col in BEHAVIOR_COL_PREFERENCE:
        if col in md.columns:
            return col
    raise KeyError(
        f"none of {BEHAVIOR_COL_PREFERENCE} found in metadata "
        f"(columns: {list(md.columns)})"
    )


def select_cell_trials(
    md_pp: pd.DataFrame,
    *,
    word_end: str | None,
    resampled_steps: Sequence[int],
    group_col: str,
    n_per_group: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[int, np.ndarray]:
    """Return {group_value: trial_indices_into_md_pp} for one B3 or B4 cell.

    md_pp must already be filtered to a single phoneme_pair and have a clean
    integer index (0..len-1). `group_col` partitions trials within each step
    in `resampled_steps`; n_per_group (if set) subsamples per (step × group).

    - For B3: resampled_steps=[step], group_col='behavior_dummy_forced',
      n_per_group=N_cal (subsamples to N_cal per class).
    - For B4: resampled_steps=qualifying_steps, group_col='behavior_dummy_forced',
      n_per_group=N_cal — sampling done per (step × class), pooled across steps.

    For the acoustic-calibration case (resampled=1 vs 6) use the sibling
    helper `select_endpoint_trials` instead — overloading group_col='resampled'
    here would conflate the step axis with the group axis.

    Raises if any (step × group) has fewer than n_per_group trials. Caller is
    responsible for filtering cells where min_class >= n_per_group beforehand.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if word_end is not None:
        we_mask = (md_pp["word_end"] == word_end).values
    else:
        we_mask = np.ones(len(md_pp), dtype=bool)

    groups = sorted(md_pp.loc[we_mask, group_col].dropna().unique())
    out: dict[int, list[np.ndarray]] = {g: [] for g in groups}

    for step in resampled_steps:
        step_mask = we_mask & (md_pp["resampled"] == step).values
        for g in groups:
            cell_mask = step_mask & (md_pp[group_col] == g).values
            idxs = np.where(cell_mask)[0]
            if n_per_group is not None:
                if len(idxs) < n_per_group:
                    raise ValueError(
                        f"step={step} group={g}: only {len(idxs)} trials < "
                        f"n_per_group={n_per_group}"
                    )
                idxs = rng.choice(idxs, size=n_per_group, replace=False)
            out[g].append(idxs)

    return {g: np.concatenate(parts) for g, parts in out.items()}


def select_endpoint_trials(
    md_pp: pd.DataFrame,
    *,
    n_per_group: int,
    rng: np.random.Generator | None = None,
    endpoints: tuple[int, int] = (1, 6),
) -> dict[int, np.ndarray]:
    """Return {endpoint_step: trial_indices} for the acoustic-calibration test.

    Pools across word_ends (the causal6 acoustic peak is word_end-agnostic).
    Raises if either endpoint has fewer than n_per_group trials.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out: dict[int, np.ndarray] = {}
    for step in endpoints:
        idxs = np.where((md_pp["resampled"] == step).values)[0]
        if len(idxs) < n_per_group:
            raise ValueError(
                f"endpoint step={step}: only {len(idxs)} trials < "
                f"n_per_group={n_per_group}"
            )
        out[step] = rng.choice(idxs, size=n_per_group, replace=False)
    return out


def extract_hga(ep, electrode_idx: int) -> np.ndarray:
    """Trials × time HGA for one electrode, baseline-corrected.

    Returned array is indexed by ep.metadata's integer index. Callers slice
    with the indices from select_cell_trials.
    """
    return (
        ep.copy()
        .apply_baseline((None, 0))
        .get_data(picks=[electrode_idx])
        .squeeze(1)
    )


@dataclass
class SearchlightTResult:
    smin: int
    smax: int
    t_stat: float
    df: float
    p_value: float
    hedges_g: float
    n_group1: int
    n_group2: int


def searchlight_ttest(
    hga: np.ndarray,
    g1_idx: np.ndarray,
    g2_idx: np.ndarray,
    *,
    search_smin: int,
    search_smax: int,
    window_size: int = 15,
    stride: int = 15,
) -> list[SearchlightTResult]:
    """Welch's t per window over [search_smin, search_smax) with window_size/stride.

    Returns one result per window start in
    `range(search_smin, search_smax - window_size + 1, stride)`. If that
    range is empty (the search interval is shorter than `window_size`), the
    return list is empty.
    """
    results: list[SearchlightTResult] = []
    n1, n2 = len(g1_idx), len(g2_idx)
    for start in range(search_smin, search_smax - window_size + 1, stride):
        x1 = hga[g1_idx, start : start + window_size].mean(axis=1)
        x2 = hga[g2_idx, start : start + window_size].mean(axis=1)
        t, p = ttest_ind(x1, x2, equal_var=False)
        v1, v2 = float(x1.var(ddof=1)), float(x2.var(ddof=1))
        df_num = (v1 / n1 + v2 / n2) ** 2
        df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else float("nan")
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = (x1.mean() - x2.mean()) / sp if sp > 0 else 0.0
        J = 1 - 3 / (4 * (n1 + n2) - 9)
        g = J * d
        results.append(SearchlightTResult(
            smin=start, smax=start + window_size,
            t_stat=float(t), df=float(df), p_value=float(p),
            hedges_g=float(g), n_group1=n1, n_group2=n2,
        ))
    return results
