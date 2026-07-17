"""Shared window-finding utilities for causal46_joined discriminative-window notebooks.

Extracted from behavioral_discriminative_windows.py so that the family of
window-discovery notebooks (behavioral_discriminative_windows.py,
early_perceptual_windows.py, acoustic_discriminative_windows.py) can import
these helpers without duplication.
"""
from __future__ import annotations

import numpy as np


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
