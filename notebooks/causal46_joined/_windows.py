"""Shared window-finding utilities for causal46_joined discriminative-window notebooks.

Extracted from behavioral_discriminative_windows.py so that the family of
window-discovery notebooks (behavioral_discriminative_windows.py,
early_perceptual_windows.py, acoustic_discriminative_windows.py) can import
these helpers without duplication.
"""
from __future__ import annotations

import numpy as np

from src.models.significance import _tfce_1d


def _window_sign(median: float) -> int:
    return 1 if median >= 0 else -1


def _find_maximal_runs(
    sig_windows: list[tuple[int, int]],
    medians: dict[int, float],
) -> list[list[tuple[int, int]]]:
    """Maximal runs of adjacent + significant + same-sign candidate windows."""
    runs: list[list[tuple[int, int]]] = []
    if not sig_windows:
        return runs
    current = [sig_windows[0]]
    for w in sig_windows[1:]:
        prev = current[-1]
        adjacent = (prev[1] == w[0])
        same_sign = (_window_sign(medians[prev[0]]) == _window_sign(medians[w[0]]))
        if adjacent and same_sign:
            current.append(w)
        else:
            runs.append(current)
            current = [w]
    runs.append(current)
    return runs


def _fallback_run(
    cand_windows: list[tuple[int, int]],
    medians: dict[int, float],
) -> list[tuple[int, int]]:
    """Seed at max |median|; grow over adjacent same-sign windows.

    Used when a cell has candidate windows but none reach significance, to still
    emit a single best-guess window. `cand_windows` must be sorted by smin.
    """
    abs_meds = [abs(medians[smin]) for smin, _ in cand_windows]
    seed_idx = int(np.argmax(abs_meds))
    seed_sign = _window_sign(medians[cand_windows[seed_idx][0]])

    # Grow left (toward smaller smin)
    left_indices = [seed_idx]
    for i in range(seed_idx - 1, -1, -1):
        if cand_windows[i][1] != cand_windows[i + 1][0]:  # gap
            break
        if _window_sign(medians[cand_windows[i][0]]) != seed_sign:
            break
        left_indices.insert(0, i)

    # Grow right (toward larger smin)
    right_indices = [seed_idx]
    for i in range(seed_idx + 1, len(cand_windows)):
        if cand_windows[i - 1][1] != cand_windows[i][0]:  # gap
            break
        if _window_sign(medians[cand_windows[i][0]]) != seed_sign:
            break
        right_indices.append(i)

    all_indices = sorted(set(left_indices + right_indices))
    return [cand_windows[i] for i in all_indices]


# =============================================================================
# TFCE gate helpers
#
# Two-tailed wrapper around the one-tailed TFCE engine in
# src/models/significance.py::_tfce_1d (validated there against
# mne.stats.cluster_level._find_clusters; see tests/test_significance.py).
# We enhance |curve| so that both positive (/n/ > /d/) and negative
# (/d/ > /n/) excursions accumulate cluster credit, per plan decision D1/D3
# (no presupposed alignment; report the signed peak).
# See: docs/superpowers/plans/2026-07-20-causal46-late-perceptual-significance.md
# =============================================================================


def tfce_enhance(
    curve: np.ndarray,
    dt: float,
    E: float = 0.5,
    H: float = 2.0,
) -> np.ndarray:
    """Two-tailed 1D TFCE enhancement over the post-acoustic window axis.

    Enhances ``|curve|`` (so both signed excursions accumulate cluster
    credit) via the validated one-tailed engine ``_tfce_1d``, then restores
    the original per-index sign of ``curve`` so the enhanced peak can still
    be reported as signed (plan D3: "enhance |curve|; report the signed
    peak"). Where ``curve[i] == 0`` the enhanced value is also 0 (an exactly
    zero point is never inside a ``stat > 0`` run). A sign flip with no
    zero sample between the two excursions (e.g. ``[1, 2, -2, -1]``) is one
    contiguous run in ``|curve|`` space, so it accumulates extent credit as
    a single cluster; the per-index sign restore still reports each side
    with its own local sign.

    Args:
        curve: 1D array, one value per candidate window, ordered by smin
            (e.g. the per-window median of ``mean_diff_raw`` across
            bootstrap replicates).
        dt: threshold step for the TFCE height integration (same role as
            ``dh`` in ``_tfce_1d``); smaller steps approximate the
            continuous integral more closely at added compute cost.
        E: extent exponent (TFCE 1D default 0.5, pre-registered — D3).
        H: height exponent (TFCE 1D default 2.0, pre-registered — D3).

    Returns:
        1D array, same shape as ``curve``: signed TFCE-enhanced values.
    """
    curve = np.asarray(curve, dtype=np.float64)
    enhanced_abs = _tfce_1d(np.abs(curve), E=E, H=H, dh=dt, threshold=0.0)
    sign = np.sign(curve)
    return enhanced_abs * sign


def max_tfce_null(
    null_curves: np.ndarray,
    dt: float,
    obs_tfce_max_abs: float,
    E: float = 0.5,
    H: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Max-|TFCE| permutation null + two-tailed empirical p for one cell.

    Each row of ``null_curves`` is one coherent across-window null curve
    from a single within-step label permutation (plan D2: replicate r's
    rows in ``b4_bootstrap`` form one coherent curve from one shuffle, not
    ``n_windows`` independent per-window draws). Each replicate's curve is
    TFCE-enhanced independently — windows are never mixed across
    replicates — and reduced to that replicate's max |TFCE|, the standard
    max-statistic construction for family-wise control across windows
    (Westfall & Young 1993; Nichols & Holmes 2002).

    ``null_curves`` must be NaN-free: tied cells (D2's ``preferred is
    None``) have no usable null and must be dropped by the caller before
    this point, not passed in here.

    Args:
        null_curves: shape ``(R, n_windows)``; row r is replicate r's curve
            across all candidate windows, in the same window order as the
            observed curve passed to `tfce_enhance`.
        dt: threshold step, passed through to `tfce_enhance`.
        obs_tfce_max_abs: the observed cell's gate statistic —
            ``np.max(np.abs(tfce_enhance(obs_curve, dt, E=E, H=H)))``.
        E: extent exponent — must match the observed-curve enhancement.
        H: height exponent — must match the observed-curve enhancement.

    Returns:
        ``(null_vec, emp_p)``:
            null_vec: shape ``(R,)``, max |TFCE| per replicate.
            emp_p: two-tailed empirical p on magnitude,
                ``(1 + #{null >= obs}) / (1 + R)``.
    """
    null_curves = np.asarray(null_curves, dtype=np.float64)
    R = null_curves.shape[0]
    null_vec = np.empty(R, dtype=np.float64)
    for r in range(R):
        enhanced = tfce_enhance(null_curves[r], dt, E=E, H=H)
        null_vec[r] = np.max(np.abs(enhanced))
    emp_p = (1 + np.sum(null_vec >= obs_tfce_max_abs)) / (1 + R)
    return null_vec, float(emp_p)


def assert_coherent_null_replicates(
    n_rows: int,
    R: int,
    n_windows: int,
    *,
    context: str = "",
) -> None:
    """Assert a cell's null rows form R coherent, complete across-window curves.

    Mirrors the union-beta row-count assertion in
    ``behavioral_discriminative_windows.py`` (``union_boot.height == R *
    n_comp``): a partial replicate (missing a window) would silently bias
    ``max_tfce_null`` low for that replicate, so we fail loudly instead of
    reshaping around a hole.

    Args:
        n_rows: observed row count for the cell's null data.
        R: number of bootstrap replicates.
        n_windows: number of candidate windows for the cell.
        context: optional string identifying the cell, included in the
            assertion message.

    Raises:
        AssertionError: if ``n_rows != R * n_windows``.
    """
    expected = R * n_windows
    assert n_rows == expected, (
        f"Expected {expected} rows (R={R} × {n_windows} windows)"
        f"{f' for {context}' if context else ''}, got {n_rows}. "
        "Check that every candidate window has a null value for every replicate."
    )


def validate_contiguous_grid(windows: list[tuple[int, int]]) -> int:
    """Assert `windows` form a contiguous, uniform-width grid sorted by smin.

    Shared by the window-discovery notebooks (`behavioral_discriminative_windows.py`,
    `late_perceptual_significance.py`) so "valid grid" (stride == window_size,
    no gaps) is defined once instead of re-derived per notebook.

    Args:
        windows: deduplicated ``(smin, smax)`` pairs, sorted by smin.

    Returns:
        The common window width (``smax - smin``).

    Raises:
        AssertionError: on an empty, non-uniform-width, or non-contiguous grid.
    """
    assert windows, "No windows to validate."
    widths = {smax - smin for smin, smax in windows}
    assert len(widths) == 1, (
        f"Non-uniform grid window widths detected: {widths}. "
        "Grid must have stride == window_size."
    )
    for i in range(len(windows) - 1):
        assert windows[i][1] == windows[i + 1][0], (
            f"Grid gap between {windows[i]} and {windows[i + 1]}. "
            "Grid must be contiguous (smax_i == smin_{i+1})."
        )
    return next(iter(widths))


def late_cell_significance(
    rep_curves: np.ndarray,
    null_curves: np.ndarray,
    *,
    E: float = 0.5,
    H: float = 2.0,
) -> dict:
    """Per-cell late-perceptual significance bundle (plan Step 3).

    Composes the TFCE gate (`tfce_enhance` + `max_tfce_null`) with the
    knob-free integral robustness statistic (D3) and a descriptive
    split-half reliability column (D7), sharing one adaptive ``dt`` (TFCE
    threshold step) across the observed curve and the null so the two are
    enhanced on the same threshold grid.

    Args:
        rep_curves: shape ``(R, n_windows)``, one row per bootstrap
            replicate, the replicate's raw ``mean_diff_raw`` value per
            candidate window (in window order, ordered by smin). The
            observed curve is ``median(rep_curves, axis=0)`` (plan Step 3:
            "median over replicates ... per window").
        null_curves: shape ``(R, n_windows)``, row r the coherent
            within-step label-permutation null curve for replicate r
            (plan D2). NaN-free — tied cells (``preferred is None``) must
            be dropped by the caller before this point.
        E: TFCE extent exponent (pre-registered default 0.5, D3).
        H: TFCE height exponent (pre-registered default 2.0, D3).

    Returns:
        dict with keys:
            tfce_peak: signed TFCE-enhanced value at the observed curve's
                peak |enhancement| (the window driving the gate).
            tfce_max_abs: ``abs(tfce_peak)`` — the gate statistic.
            tfce_emp_p: two-tailed max-TFCE permutation p-value.
            integral_stat: knob-free robustness statistic — mean of the
                observed curve over all candidate windows (D3's rejected-
                as-gate, reported-as-robustness-check integrated-window
                statistic).
            integral_emp_p: two-tailed permutation p for `integral_stat`,
                against the per-replicate null curve means.
            splithalf_sign_agree: descriptive-only reliability column
                (D7), not used for gating. A *replicate*-split proxy for
                the plan's trial-split (this module only sees persisted
                per-replicate curves, not raw trial draws — no epoch
                reload here): replicates are split into first/second
                halves, each half's median curve is evaluated at the
                observed peak window, and the two signs are compared.
                ``None`` when ``R < 2`` (no split is possible) or when
                either half's value at the peak is exactly zero (no sign
                to compare).
    """
    rep_curves = np.asarray(rep_curves, dtype=np.float64)
    null_curves = np.asarray(null_curves, dtype=np.float64)
    R, n_windows = rep_curves.shape
    assert null_curves.shape == (R, n_windows), (
        f"rep_curves and null_curves shape mismatch: {rep_curves.shape} vs "
        f"{null_curves.shape}"
    )
    assert not np.isnan(null_curves).any(), (
        "null_curves contains NaN — tied cells must be dropped by the caller "
        "before calling late_cell_significance."
    )

    obs_curve = np.median(rep_curves, axis=0)

    dt_base = max(float(np.max(np.abs(obs_curve))), float(np.max(np.abs(null_curves))))
    dt = dt_base / 100.0 if dt_base > 0 else 1.0  # inert: both curves all-zero

    enhanced = tfce_enhance(obs_curve, dt, E=E, H=H)
    peak_idx = int(np.argmax(np.abs(enhanced)))
    tfce_peak = float(enhanced[peak_idx])
    tfce_max_abs = float(abs(tfce_peak))

    _, tfce_emp_p = max_tfce_null(null_curves, dt, tfce_max_abs, E=E, H=H)

    integral_stat = float(np.mean(obs_curve))
    null_integral = np.mean(null_curves, axis=1)
    integral_emp_p = float(
        (1 + np.sum(np.abs(null_integral) >= abs(integral_stat))) / (1 + R)
    )

    splithalf_sign_agree: bool | None = None
    if R >= 2:
        half = R // 2
        first_val = float(np.median(rep_curves[:half], axis=0)[peak_idx])
        second_val = float(np.median(rep_curves[half:], axis=0)[peak_idx])
        s1, s2 = np.sign(first_val), np.sign(second_val)
        if s1 != 0 and s2 != 0:
            splithalf_sign_agree = bool(s1 == s2)

    return {
        "tfce_peak": tfce_peak,
        "tfce_max_abs": tfce_max_abs,
        "tfce_emp_p": tfce_emp_p,
        "integral_stat": integral_stat,
        "integral_emp_p": integral_emp_p,
        "splithalf_sign_agree": splithalf_sign_agree,
    }
