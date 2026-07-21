"""Shared primitives for the perceptual-projection analyses (early + late).

The projection statistic is π = ⟨â, p⟩: how much the within-completion percept
contrast on ambiguous trials (`p`) re-expresses the same cell's unambiguous
/d/–/n/ acoustic tuning (`â`), integrated over a window set. Both the early
(`early_perceptual_projection.py`, ADR-0002) and late
(`late_perceptual_projection.py`, ADR-0003) notebooks build `p` and its
permutation null the same way; those window-agnostic primitives live here so the
late notebook does not copy-paste the early one (issue #22).

What is shared vs. what is not
------------------------------
Shared (this module): the qualifying-step rule, the per-step class-balance
selection, the deterministic min-class-weighted percept contrast `p` over an
explicit window list, and the within-step label-permutation null of π.

NOT shared: the **early** notebook pools â across word_ends and π across
completions and draws a single permutation matrix that updates a pooled *and*
per-word_end weight scheme together — a deliberately different reduction with an
RNG stream that cannot be reproduced by the single-weight helper here without
changing its published outputs. Early keeps its own `compute_p_we` /
`compute_permutation_null`; only `get_qualifying_steps` is imported from here.
The late notebook is strict same-word-end (no pooling), so it uses these
single-weight helpers directly.

The canonical B3/B4 subsampling rule these build on is documented in
`_within_completion.py`'s module docstring — read it before touching
trial-selection.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from _within_completion import per_step_class_counts


def get_qualifying_steps(
    md_pp: pd.DataFrame,
    *,
    word_end: str,
    group_col: str,
    ambiguous_threshold: int = 2,
) -> list[int]:
    """Ambiguous steps for one (phoneme_pair, word_end) cell.

    Step s qualifies if: not in endpoints {1,6}, both behavior classes
    present, and minority class count > ambiguous_threshold. Matches
    `src.data.get_ambiguous_resampled_steps` criterion. (Moved verbatim from
    `early_perceptual_projection.py`; imported by both projection notebooks.)
    """
    we_mask = md_pp["word_end"] == word_end
    ambiguous_mask = ~md_pp["resampled"].isin([1, 6])
    steps = sorted(md_pp.loc[we_mask & ambiguous_mask, "resampled"].unique())
    qualifying = []
    for s in steps:
        step_mask = we_mask & ambiguous_mask & (md_pp["resampled"] == s)
        counts = md_pp.loc[step_mask, group_col].value_counts()
        if len(counts) >= 2 and int(counts.min()) > ambiguous_threshold:
            qualifying.append(int(s))
    return qualifying


def select_per_step_balanced(
    md_pp: pd.DataFrame,
    *,
    word_end: str,
    group_col: str,
    K: int,
) -> tuple[Optional[dict], Optional[dict], int]:
    """Per-step class indices + min_class weights + N for one word_end cell.

    Returns (per_step_filtered, min_classes, N_we), or (None, None, 0) if no
    step survives the K-per-class gate.

    - per_step_filtered: {step: {class: trial_indices}} restricted to steps with
      both classes present and min(len(class0), len(class1)) >= K.
    - min_classes: {step: min_class[s]}.
    - N_we: Σ_s min_class[s] (the per-class sample size / weight denominator).

    This is exactly the selection `early_perceptual_projection.compute_p_we`
    performs inline; factored out so `windowed_deterministic_p` and the null can
    share it.
    """
    qualifying = get_qualifying_steps(md_pp, word_end=word_end, group_col=group_col)
    if not qualifying:
        return None, None, 0

    per_step = per_step_class_counts(
        md_pp, word_end=word_end, qualifying_steps=qualifying, group_col=group_col
    )
    per_step_filtered = {
        s: by_cls
        for s, by_cls in per_step.items()
        if 0 in by_cls and 1 in by_cls
        and min(len(by_cls[0]), len(by_cls[1])) >= K
    }
    if not per_step_filtered:
        return None, None, 0

    min_classes = {
        s: min(len(by_cls[0]), len(by_cls[1]))
        for s, by_cls in per_step_filtered.items()
    }
    N_we = int(sum(min_classes.values()))
    return per_step_filtered, min_classes, N_we


def windowed_deterministic_p(
    hga: np.ndarray,
    per_step_filtered: dict,
    min_classes: dict,
    N_we: int,
    windows: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Deterministic B4 min_class-weighted percept contrast over `windows`.

        p(w) = Σ_s [min_class[s] / N_we] * (mean HGA[class1, s, w] - mean HGA[class0, s, w])

    class1 = behavior class 1 = heard phoneme_pair[1] (fixed /n/-side reference).
    `windows` is a list of half-open (smin, smax) sample bounds. All trials in
    each class are used; min_class[s] enters only as the per-step weight. Returns
    an array of length len(windows). Identical arithmetic to early's
    `compute_p_we` inner loop, generalized from a fixed stride grid to an
    explicit window list.
    """
    p = np.zeros(len(windows))
    for s, by_cls in per_step_filtered.items():
        w_s = min_classes[s] / N_we
        idx1 = by_cls[1]
        idx0 = by_cls[0]
        for i, (smin, smax) in enumerate(windows):
            diff = hga[idx1, smin:smax].mean() - hga[idx0, smin:smax].mean()
            p[i] += w_s * diff
    return p


def permutation_null_pi(
    hga: np.ndarray,
    cell_data: Sequence[tuple[np.ndarray, np.ndarray, float]],
    a_hat: np.ndarray,
    windows: Sequence[tuple[int, int]],
    n_perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Within-step label-permutation null of π for a single (single-weight) cell.

    cell_data: list of (idx1, idx0, weight) per qualifying step, where weight =
    min_class[s] / N_we. a_hat: unit acoustic template over `windows` (held
    fixed — derives from unambiguous trials, untouched by the permutation).
    Shuffles the class labels within each step, preserving that step's observed
    (n1, n0) split, and forms π_null = Σ_w a_hat[w] * Σ_step weight * diff_perm(w).
    Same shuffle used for all windows within a step (labels fixed per replicate).

    Returns a (n_perms,) array. Mirrors early's `compute_permutation_null`
    vectorization for a single weight scheme.
    """
    pi_perm = np.zeros(n_perms)
    for idx1, idx0, weight in cell_data:
        n1 = len(idx1)
        all_idx = np.concatenate([idx1, idx0])
        n_total = len(all_idx)

        u = rng.random((n_perms, n_total))
        perm_matrix = np.argsort(u, axis=1)  # uniform random permutations

        for w_idx, (smin, smax) in enumerate(windows):
            X = hga[all_idx, smin:smax].mean(axis=1)  # (n_total,)
            X_perm = X[perm_matrix]                    # (n_perms, n_total)
            diff_perm = (
                X_perm[:, :n1].mean(axis=1) - X_perm[:, n1:].mean(axis=1)
            )
            pi_perm += a_hat[w_idx] * weight * diff_perm
    return pi_perm


def anchored_reliable_run(
    beta_med: np.ndarray,
    reliable: np.ndarray,
) -> Optional[tuple[int, int]]:
    """â-anchored contiguous reliable run (window-rule 1c, ADR-0003 §5).

    Given per-window median β_unamb (`beta_med`) and per-window reliability
    (`reliable`, bool: bootstrap CI excludes 0), both indexed over the ordered
    late window grid:

    1. Anchor at argmax|median β_unamb| over reliable windows.
    2. Return the maximal contiguous run of reliable windows containing the
       anchor, as a half-open index range (lo, hi) into the window list.

    Returns None if no window is reliable (no anchor, no run → π_anchored NaN).
    """
    reliable = np.asarray(reliable, dtype=bool)
    if not reliable.any():
        return None
    abs_beta = np.abs(np.asarray(beta_med, dtype=float))
    # Anchor: strongest tuning locus among reliable windows only.
    masked = np.where(reliable, abs_beta, -np.inf)
    anchor = int(np.argmax(masked))
    # Expand contiguously while reliable.
    lo = anchor
    while lo - 1 >= 0 and reliable[lo - 1]:
        lo -= 1
    hi = anchor
    while hi + 1 < len(reliable) and reliable[hi + 1]:
        hi += 1
    return lo, hi + 1  # half-open
