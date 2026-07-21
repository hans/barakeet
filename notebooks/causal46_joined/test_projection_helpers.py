"""Synthetic-data equivalence checks for `_projection.py`.

No epochs required — runs in the dev container. Proves the generic single-weight
helpers reproduce the tested arithmetic of `early_perceptual_projection`'s inline
`compute_p_we` / `compute_permutation_null` (so the late notebook, which uses the
generic helpers, is computing `p` and its null correctly), and that the
`anchored_reliable_run` window rule (ADR-0003 §5) behaves as specified.

Run: `uv run python notebooks/causal46_joined/test_projection_helpers.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _projection import (  # noqa: E402
    anchored_reliable_run,
    permutation_null_pi,
    select_per_step_balanced,
    windowed_deterministic_p,
)


# ── Reference implementations copied from early_perceptual_projection.py ──────
# These are the exact inline computations early performs (pre-extraction), used
# here only as an oracle to check the generic helpers against.

def _ref_compute_p_we(hga, per_step_filtered, min_classes, N_we, window_starts, window_size):
    p_we = np.zeros(len(window_starts))
    for s, by_cls in per_step_filtered.items():
        w_s = min_classes[s] / N_we
        idx1 = by_cls[1]
        idx0 = by_cls[0]
        for i, smin in enumerate(window_starts):
            smax_w = smin + window_size
            diff = hga[idx1, smin:smax_w].mean() - hga[idx0, smin:smax_w].mean()
            p_we[i] += w_s * diff
    return p_we


def _ref_perm_null(hga, cell_data, a_hat, window_starts, window_size, n_perms, rng):
    pi_perm = np.zeros(n_perms)
    for idx1, idx0, weight in cell_data:
        n1 = len(idx1)
        all_idx = np.concatenate([idx1, idx0])
        n_total = len(all_idx)
        u = rng.random((n_perms, n_total))
        perm_matrix = np.argsort(u, axis=1)
        for w_idx, smin in enumerate(window_starts):
            smax_w = smin + window_size
            X = hga[all_idx, smin:smax_w].mean(axis=1)
            X_perm = X[perm_matrix]
            diff_perm = X_perm[:, :n1].mean(axis=1) - X_perm[:, n1:].mean(axis=1)
            pi_perm += a_hat[w_idx] * weight * diff_perm
    return pi_perm


def _make_synth(seed=0, n_trials=120, n_times=150):
    rng = np.random.default_rng(seed)
    hga = rng.standard_normal((n_trials, n_times))
    # word_end/resampled/behavior metadata mimicking one phoneme_pair subset
    word_ends = np.where(np.arange(n_trials) % 2 == 0, "necessary", "desolate")
    resampled = rng.integers(1, 7, size=n_trials)  # steps 1..6
    behavior = rng.integers(0, 2, size=n_trials)
    md = pd.DataFrame({
        "word_end": word_ends,
        "resampled": resampled,
        "behavior_dummy_forced": behavior,
    })
    return hga, md


def test_p_equivalence():
    hga, md = _make_synth(seed=1)
    K = 3
    window_starts = list(range(45, 64, 5))
    window_size = 5
    windows = [(s, s + window_size) for s in window_starts]
    checked = 0
    for we in ("necessary", "desolate"):
        sel = select_per_step_balanced(
            md, word_end=we, group_col="behavior_dummy_forced", K=K
        )
        per_step_filtered, min_classes, N_we = sel
        if per_step_filtered is None:
            continue
        p_generic = windowed_deterministic_p(hga, per_step_filtered, min_classes, N_we, windows)
        p_ref = _ref_compute_p_we(hga, per_step_filtered, min_classes, N_we, window_starts, window_size)
        assert np.allclose(p_generic, p_ref, atol=1e-12), (we, p_generic, p_ref)
        checked += 1
    assert checked > 0, "no qualifying cells in synthetic data — widen the fixture"
    print(f"  p equivalence OK ({checked} word_ends)")


def test_null_equivalence():
    hga, md = _make_synth(seed=2)
    K = 3
    window_starts = list(range(45, 64, 5))
    window_size = 5
    windows = [(s, s + window_size) for s in window_starts]
    sel = select_per_step_balanced(
        md, word_end="necessary", group_col="behavior_dummy_forced", K=K
    )
    per_step_filtered, min_classes, N_we = sel
    assert per_step_filtered is not None
    a_hat = np.linspace(-1, 1, len(windows))
    a_hat = a_hat / np.linalg.norm(a_hat)
    cell_data = [
        (by_cls[1], by_cls[0], min_classes[s] / N_we)
        for s, by_cls in per_step_filtered.items()
    ]
    n_perms = 500
    null_generic = permutation_null_pi(
        hga, cell_data, a_hat, windows, n_perms, np.random.default_rng(7)
    )
    null_ref = _ref_perm_null(
        hga, cell_data, a_hat, window_starts, window_size, n_perms, np.random.default_rng(7)
    )
    assert np.allclose(null_generic, null_ref, atol=1e-12)
    # Null should be centred near zero (labels exchangeable).
    assert abs(null_generic.mean()) < 0.1 * (null_generic.std() + 1e-12) or null_generic.std() == 0
    print(f"  null equivalence OK (n_perms={n_perms}, mean={null_generic.mean():.4g}, sd={null_generic.std():.4g})")


def test_anchored_run():
    # reliable windows: indices 2,3,4 and 7,8; anchor should pick strongest |beta|.
    beta = np.array([0.1, 0.2, 0.5, 0.9, 0.4, 0.1, 0.0, 0.8, 0.3, 0.05])
    rel = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    run = anchored_reliable_run(beta, rel)
    assert run == (2, 5), run  # argmax|beta| among reliable is idx3 (0.9); run 2..4
    # a larger reliable magnitude in the second run flips the anchor
    beta2 = np.array([0.1, 0.2, 0.5, 0.4, 0.4, 0.1, 0.0, 0.99, 0.3, 0.05])
    run2 = anchored_reliable_run(beta2, rel)
    assert run2 == (7, 9), run2  # anchor idx7 (0.99); run 7..8
    # no reliable window → None
    assert anchored_reliable_run(beta, np.zeros_like(rel, dtype=bool)) is None
    # single reliable window → length-1 run
    single = np.array([0, 0, 1, 0, 0], dtype=bool)
    assert anchored_reliable_run(np.array([0, 0, 0.3, 0, 0]), single) == (2, 3)
    print("  anchored_reliable_run OK")


if __name__ == "__main__":
    print("test_projection_helpers:")
    test_p_equivalence()
    test_null_equivalence()
    test_anchored_run()
    print("ALL PASS")
