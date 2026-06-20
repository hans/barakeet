"""Shared window-finding utilities for causal46_joined perceptual window notebooks.

Extracted from behavioral_discriminative_windows.py so that both
behavioral_discriminative_windows.py and early_perceptual_windows.py can import
these helpers without duplication.
"""
from __future__ import annotations


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
