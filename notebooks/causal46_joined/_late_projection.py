"""Shared projection-gate primitives for the late-perceptual projection analyses.

Lifted verbatim from `late_perceptual_projection.py` (the B4/pooled-step gate),
with one addition: `compute_p` takes an optional `restrict_steps` kwarg so the
single-step notebook (`late_single_step_perceptual_projection.py`) can reuse the
exact same gate logic, restricted to one acoustic step. `restrict_steps=None`
reproduces the original pooled-step behavior exactly.

One non-verbatim change: the original `compute_p` sized its `traces` array off
a notebook-global `ep.times` (only used for the end-of-notebook diagnostic
plots, never serialized to `results.csv`). As a standalone module this
function has no such global, so `traces` is sized from `hga.shape[1]`
instead — equal to `len(ep.times)` by construction (`hga` is extracted from
the same epochs object), so this does not change any computed value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _within_completion import per_step_class_counts  # noqa: E402


def get_qualifying_steps(md_pp, *, word_end, group_col, ambiguous_threshold=2):
    """Ambiguous steps for one (phoneme_pair, word_end) cell.

    Step s qualifies if: not in endpoints {1,6}, both behavior classes
    present, and minority class count > ambiguous_threshold.
    Matches src.data.get_ambiguous_resampled_steps criterion.
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


def compute_a_vector(hga, md_pp, smin, smax,
                     window_size, stride):
    """Acoustic template a(w) = mean HGA[step6] - mean HGA[step1], pooled word_ends.

    Positive direction = phoneme_pair[1] (the 'step6' phoneme). No sign flip.
    """
    assert len(hga) == len(md_pp)

    step1_mask = md_pp["resampled"] == 1
    step6_mask = md_pp["resampled"] == 6

    if step1_mask.sum() == 0 or step6_mask.sum() == 0:
        raise ValueError(f"No trials for one of the endpoints: step 1 = {step1_mask.sum()}, step 6 = {step6_mask.sum()}")

    window_starts = np.arange(smin, smax - window_size + 1, stride)
    a = np.array([
        (hga[step6_mask, s:s + window_size].mean() - hga[step1_mask, s:s + window_size].mean())
        for s in window_starts
    ])
    return a


def compute_a_vector_null(hga, md_pp, smin, smax,
                          window_size, stride,
                          n_perms, rng=None):
    """
    Compute null distribution of acoustic template a(w) under label shuffling.

    Shuffles acoustic labels preserving the observed (n_step1, n_step6) split.

    Returns:
        a_perm: (N_PERMS, N_WINDOWS) null
    """
    assert len(hga) == len(md_pp)
    if rng is None:
        rng = np.random.default_rng()

    step1_mask = md_pp["resampled"] == 1
    step6_mask = md_pp["resampled"] == 6

    if step1_mask.sum() == 0 or step6_mask.sum() == 0:
        raise ValueError(f"No trials for one of the endpoints: step 1 = {step1_mask.sum()}, step 6 = {step6_mask.sum()}")

    window_starts = np.arange(smin, smax - window_size + 1, stride)
    n_windows = len(window_starts)
    a_perm = np.zeros((n_perms, n_windows))

    idx_step1 = np.where(step1_mask)[0]
    idx_step6 = np.where(step6_mask)[0]
    idx_all = np.concatenate([idx_step1, idx_step6])
    n_step1 = len(idx_step1)
    n_total = n_step1 + len(idx_step6)

    # Vectorized permutations: N_PERMS × n_total
    u = rng.random((n_perms, n_total))
    perm_matrix = np.argsort(u, axis=1)  # uniform random permutations

    for w_idx, smin_w in enumerate(window_starts):
        smax_w = smin_w + window_size
        X = hga[idx_all, smin_w:smax_w].mean(axis=1)  # (n_total,)
        X_perm = X[perm_matrix]               # (N_PERMS, n_total)
        diff_perm = (
            X_perm[:, :n_step1].mean(axis=1) - X_perm[:, n_step1:].mean(axis=1)
        )  # (N_PERMS,)
        a_perm[:, w_idx] = diff_perm

    return a_perm


def compute_p(hga, md_pp, word_end, group_col,
              smin, smax,
              window_size, stride, K,
              restrict_steps=None):
    """
    Compute perceptual contrast:

    p(w) = Σ_s [min_class[s]/N_we] * [mean HGA[class1,s,w] - mean HGA[class0,s,w]]

    class1 = behavior_dummy_forced=1 = heard phoneme_pair[1].

    Parameters
    ----------
    restrict_steps : None, int, or sequence of int, optional
        None (default): pool over all qualifying steps (original behavior).
        Otherwise: restrict the weighted sum to these step(s), intersected
        with the qualifying steps, before the min_class_k gate and weight
        (w_s) computation — so weights renormalize automatically (a single
        restricted step gets w_s = 1.0, a plain within-step mean-diff).
        A restriction that leaves no qualifying steps, or whose sole
        surviving step fails `min_class_k`, falls into the same untestable
        `(None, ...)` return as an unrestricted cell with no qualifying
        steps.

    Returns
    -------
    p : np.ndarray
        Perceptual contrast for each window (N_WINDOWS,).
    min_classes : dict
        Dictionary mapping each qualifying step to the minimum class count for that step.
    per_step_filtered : dict
        Dictionary mapping each qualifying step to a dictionary of class counts for that step.
    N : int
        Total number of trials across all qualifying steps.
    """
    assert len(hga) == len(md_pp)

    qualifying = get_qualifying_steps(md_pp, word_end=word_end, group_col=group_col)
    if not qualifying:
        return None, None, None, 0, None

    if restrict_steps is not None:
        if isinstance(restrict_steps, int):
            restrict_steps = [restrict_steps]
        restrict_set = set(restrict_steps)
        qualifying = [s for s in qualifying if s in restrict_set]
        if not qualifying:
            return None, None, None, 0, None

    per_step = per_step_class_counts(
        md_pp, word_end=word_end, qualifying_steps=qualifying, group_col=group_col
    )

    # Apply min_class_k gate
    per_step_filtered = {
        s: by_cls
        for s, by_cls in per_step.items()
        if 0 in by_cls and 1 in by_cls
        and min(len(by_cls[0]), len(by_cls[1])) >= K
    }
    if not per_step_filtered:
        # No qualifying steps after filtering by min_class_k
        return None, None, None, 0, None

    min_classes = {
        s: min(len(by_cls[0]), len(by_cls[1]))
        for s, by_cls in per_step_filtered.items()
    }
    N = int(sum(min_classes.values()))

    window_starts = np.arange(smin, smax - window_size + 1, stride)
    n_windows = len(window_starts)
    p = np.zeros(n_windows)
    traces = np.zeros((2, hga.shape[1]))
    for s, by_cls in per_step_filtered.items():
        w_s = min_classes[s] / N
        idx1 = by_cls[1]
        idx0 = by_cls[0]
        for i, smin in enumerate(window_starts):
            smax_w = smin + window_size
            diff = (
                hga[idx1, smin:smax_w].mean()
                - hga[idx0, smin:smax_w].mean()
            )
            p[i] += w_s * diff

        traces[0] += w_s * hga[idx0].mean(axis=0)
        traces[1] += w_s * hga[idx1].mean(axis=0)

    return p, min_classes, per_step_filtered, N, traces
