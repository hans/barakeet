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

from math import comb
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from _within_completion import (
    beta_summary,
    bootstrap_endpoint_beta,
    per_step_class_counts,
)


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
    mode: str = "reliable_max",
) -> Optional[tuple[int, int]]:
    """â-anchored contiguous reliable run (window-rule 1c, ADR-0003 §5).

    Given per-window median β_unamb (`beta_med`) and per-window reliability
    (`reliable`, bool: bootstrap CI excludes 0), both indexed over the ordered
    late window grid, returns the maximal contiguous run of reliable windows
    containing the anchor, as a half-open index range (lo, hi), or None.

    Two anchor readings — the spec (§5) is internally ambiguous and this is a
    LOCKED-pre-reg ratification point (issue #22), so both are exposed:

    - ``mode="reliable_max"`` (default): anchor = argmax|median β_unamb| **among
      reliable windows only**. Any reliable window ⇒ non-empty run ⇒ non-NaN.
      Matches the spec's population language ("NaN where no reliable window
      exists", §5.5/§8.1: "cells with a non-empty reliable run R") and the stated
      intent ("test the tuning where it reliably lives").
    - ``mode="global_max"``: anchor = argmax|median β_unamb| over **all** windows
      (spec §5 steps 2→3, literal); if that window is not itself reliable, no
      reliable run contains it ⇒ None (cell excluded even if other windows are
      reliable). "The *strongest* tuning window must itself be reliable."

    The two agree exactly when the global-max-|β| window is reliable; they
    diverge only when it isn't.
    """
    reliable = np.asarray(reliable, dtype=bool)
    abs_beta = np.abs(np.asarray(beta_med, dtype=float))
    if mode == "reliable_max":
        if not reliable.any():
            return None
        masked = np.where(reliable, abs_beta, -np.inf)
        anchor = int(np.argmax(masked))
    elif mode == "global_max":
        anchor = int(np.argmax(abs_beta))
        if not reliable[anchor]:
            return None
    else:
        raise ValueError(f"unknown anchor mode={mode!r}")
    lo = anchor
    while lo - 1 >= 0 and reliable[lo - 1]:
        lo -= 1
    hi = anchor
    while hi + 1 < len(reliable) and reliable[hi + 1]:
        hi += 1
    return lo, hi + 1  # half-open


def compute_cell_projection(
    hga: np.ndarray,
    md_pp: pd.DataFrame,
    *,
    word_end: str,
    group_col: str,
    windows: Sequence[tuple[int, int]],
    K: int,
    n_perms: int,
    rng: np.random.Generator,
    r_unamb: int,
    min_endpoint_n: int,
    ci_low: float,
    ci_high: float,
    anchor_mode: str = "reliable_max",
) -> tuple[dict, Optional[np.ndarray], Optional[np.ndarray]]:
    """Full late per-cell projection statistic for one (word_end) cell.

    Encapsulates the claim-critical per-cell path — â over the late grid,
    reliability, the 1c anchored-run window rule, π_anchored + π_peak, and their
    within-step label-permutation nulls (ADR-0003 §§3–6). Factored out of the
    notebook loop so it is unit-testable on synthetic data (the notebook body is
    otherwise unexercised in the dev container, where epochs are absent).

    Returns ``(metrics, null_anchored, null_peak)``:
      - ``metrics``: dict of every π + descriptor column + ``skip_reason``
        (values NaN where undefined). Does not include cell-identity keys — the
        caller merges those.
      - ``null_anchored``: (n_perms,) π_anchored null, or None (not â-reliable).
      - ``null_peak``: (n_perms,) π_peak null, or None (â not estimable / no p).
    """
    metric_keys = [
        "pi_anchored", "pi_peak", "pi_anchored_raw",
        "p_one_tailed", "p_two_tailed",
        "p_one_tailed_peak", "p_two_tailed_peak",
        "null_mean", "null_sd",
        "n_reliable_windows", "run_smin", "run_smax", "run_len",
        "anchor_smin", "anchor_smax",
        "peak_smin", "peak_smax", "peak_beta_unamb_median",
        "peak_beta_unamb_ci_low", "peak_beta_unamb_ci_high",
        "a_raw_norm", "N_ambiguous", "n_qualifying_steps",
        "exhaustive", "perm_space", "min_p",
    ]

    def _nan_metrics(reason, extra=None):
        m = {k: np.nan for k in metric_keys}
        m["skip_reason"] = reason
        if extra:
            m.update(extra)
        return m

    if not windows:
        return _nan_metrics("no_late_windows"), None, None

    # ── â over the late grid (unambiguous endpoints, this word_end) ──────────
    n_w = len(windows)
    beta_med = np.full(n_w, np.nan)
    beta_ci_lo = np.full(n_w, np.nan)
    beta_ci_hi = np.full(n_w, np.nan)
    reliable = np.zeros(n_w, dtype=bool)
    for wi, (smin, smax) in enumerate(windows):
        arr = bootstrap_endpoint_beta(
            hga, md_pp, word_end=word_end, smin=smin, smax=smax,
            R=r_unamb, min_n=min_endpoint_n,
        )
        if arr is None:
            return _nan_metrics("a_not_estimable"), None, None
        bs = beta_summary(arr, ci_low, ci_high)
        beta_med[wi] = bs["med"]
        beta_ci_lo[wi] = bs["ci_low"]
        beta_ci_hi[wi] = bs["ci_high"]
        reliable[wi] = bs["reliable"]

    # ── p over the late grid (ambiguous, this word_end) ─────────────────────
    per_step_filtered, min_classes, N_we = select_per_step_balanced(
        md_pp, word_end=word_end, group_col=group_col, K=K
    )
    if per_step_filtered is None:
        return _nan_metrics("no_ambiguous_cells",
                            extra={"n_reliable_windows": int(reliable.sum())}), None, None

    p_full = windowed_deterministic_p(hga, per_step_filtered, min_classes, N_we, windows)
    cell_data = [
        (by_cls[1], by_cls[0], min_classes[s] / N_we)
        for s, by_cls in per_step_filtered.items()
    ]

    # ── π_peak (diagnostic): single argmax|β_unamb| window, reliability-ignored ─
    peak_i = int(np.argmax(np.abs(beta_med)))
    peak_sign = float(np.sign(beta_med[peak_i]))
    a_unit_peak = np.array([peak_sign])
    pi_peak = float(peak_sign * p_full[peak_i])
    peak_window = windows[peak_i]
    null_peak = permutation_null_pi(hga, cell_data, a_unit_peak, [peak_window], n_perms, rng)
    p_one_peak = float(np.mean(null_peak >= pi_peak))
    p_two_peak = float(np.mean(np.abs(null_peak) >= abs(pi_peak)))

    peak_descr = dict(
        peak_smin=peak_window[0], peak_smax=peak_window[1],
        peak_beta_unamb_median=float(beta_med[peak_i]),
        peak_beta_unamb_ci_low=float(beta_ci_lo[peak_i]),
        peak_beta_unamb_ci_high=float(beta_ci_hi[peak_i]),
        N_ambiguous=int(N_we), n_qualifying_steps=int(len(per_step_filtered)),
        n_reliable_windows=int(reliable.sum()),
        pi_peak=pi_peak, p_one_tailed_peak=p_one_peak, p_two_tailed_peak=p_two_peak,
    )

    # ── π_anchored (claim statistic): â-anchored contiguous reliable run ─────
    run = anchored_reliable_run(beta_med, reliable, mode=anchor_mode)
    if run is None:
        m = {k: np.nan for k in metric_keys}
        m.update(peak_descr)
        m.update(dict(run_len=0, skip_reason="a_not_reliable"))
        return m, None, null_peak

    lo, hi = run
    run_windows = list(windows[lo:hi])
    a_raw_run = beta_med[lo:hi]
    # The tuning locus actually integrated (argmax|β_unamb| within the run) — used
    # by the aggregate's anchor-time-vs-POD diagnostic to check the â-anchor is not
    # pulled into the acoustic-decay tail (contamination check, ADR-0003 §5).
    anchor_win = run_windows[int(np.argmax(np.abs(a_raw_run)))]
    a_raw_norm = float(np.linalg.norm(a_raw_run))
    a_unit = a_raw_run / a_raw_norm
    p_run = p_full[lo:hi]

    pi_anchored = float(np.dot(a_unit, p_run))
    pi_anchored_raw = float(np.dot(a_raw_run, p_run))
    null_pi = permutation_null_pi(hga, cell_data, a_unit, run_windows, n_perms, rng)
    p_one = float(np.mean(null_pi >= pi_anchored))
    p_two = float(np.mean(np.abs(null_pi) >= abs(pi_anchored)))

    _space = 1
    for idx1_c, idx0_c, _w in cell_data:
        _space *= comb(len(idx1_c) + len(idx0_c), len(idx1_c))
        if _space > n_perms * 100:
            _space = n_perms * 100 + 1
            break
    perm_space = int(_space)

    m = dict(peak_descr)
    m.update(dict(
        pi_anchored=pi_anchored, pi_anchored_raw=pi_anchored_raw,
        p_one_tailed=p_one, p_two_tailed=p_two,
        null_mean=float(null_pi.mean()), null_sd=float(null_pi.std()),
        run_smin=int(run_windows[0][0]), run_smax=int(run_windows[-1][1]),
        anchor_smin=int(anchor_win[0]), anchor_smax=int(anchor_win[1]),
        run_len=int(hi - lo), a_raw_norm=a_raw_norm,
        exhaustive=bool(perm_space <= n_perms), perm_space=perm_space,
        min_p=(1.0 / perm_space if perm_space > 0 else np.nan),
        skip_reason="",
    ))
    return m, null_pi, null_peak
