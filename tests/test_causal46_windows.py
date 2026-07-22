"""
Unit tests for the TFCE gate helpers in notebooks/causal46_joined/_windows.py.

Synthetic-only, pure functions, no pipeline/epoch data required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "causal46_joined"))
from _windows import (  # noqa: E402
    assert_coherent_null_replicates,
    max_tfce_null,
    tfce_enhance,
)

from src.models.significance import _tfce_1d  # noqa: E402


# ---------------------------------------------------------------------------
# tfce_enhance
# ---------------------------------------------------------------------------


def test_tfce_enhance_monotone_bump_ordering_and_sign():
    """Hand-worked monotone bump: enhancement strictly increases toward the
    peak on both flanks, and the (non-negative) input's enhancement is
    non-negative everywhere."""
    curve = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    out = tfce_enhance(curve, dt=0.1, E=0.5, H=2.0)

    assert out[3] > out[2] > out[1] > out[0]
    assert out[3] > out[4] > out[5] > out[6]
    assert np.all(out >= 0)


def test_tfce_enhance_two_tailed_sign_symmetry():
    """Negating the curve negates the enhancement exactly (two-tailed:
    enhancement operates on |curve|, sign is restored from the input)."""
    curve = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
    pos_out = tfce_enhance(curve, dt=0.1, E=0.5, H=2.0)
    neg_out = tfce_enhance(-curve, dt=0.1, E=0.5, H=2.0)

    assert np.allclose(neg_out, -pos_out)
    assert np.all(neg_out <= 0)


def test_tfce_enhance_mixed_sign_bumps_symmetric_magnitude_correct_sign():
    """A negative bump and a positive bump of identical shape/height get
    equal-magnitude, oppositely-signed enhancement at their peaks."""
    curve = np.array([0.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0])
    out = tfce_enhance(curve, dt=0.05, E=0.5, H=2.0)

    assert out[3] < 0
    assert out[7] > 0
    assert np.isclose(abs(out[3]), abs(out[7]))


def test_tfce_enhance_matches_one_tailed_engine_on_positive_curve():
    """On an all-positive curve, the two-tailed wrapper must exactly match
    the validated one-tailed engine (src.models.significance._tfce_1d),
    since |curve| == curve and sign is uniformly +1."""
    rng = np.random.default_rng(0)
    curve = rng.random(20) + 0.01  # strictly positive
    dt = 0.02

    expected = _tfce_1d(curve, E=0.5, H=2.0, dh=dt, threshold=0.0)
    out = tfce_enhance(curve, dt=dt, E=0.5, H=2.0)

    assert np.allclose(out, expected)


def test_tfce_enhance_broad_cluster_beats_narrow_peak():
    """Same point made in test_significance.py: TFCE should favor a broad
    cluster over a narrow peak of equal height."""
    n = 30
    x_narrow = np.zeros(n)
    x_narrow[15] = 1.0

    x_broad = np.zeros(n)
    x_broad[10:20] = 1.0

    narrow_out = tfce_enhance(x_narrow, dt=0.05, E=0.5, H=2.0)
    broad_out = tfce_enhance(x_broad, dt=0.05, E=0.5, H=2.0)

    assert broad_out[15] > narrow_out[15]


def test_tfce_enhance_all_zero_curve_stays_zero():
    curve = np.zeros(10)
    out = tfce_enhance(curve, dt=0.1)
    assert np.all(out == 0)


# ---------------------------------------------------------------------------
# max_tfce_null
# ---------------------------------------------------------------------------


def test_max_tfce_null_monotone_in_amplitude():
    """Scaling a fixed-shape curve up should scale up its max-|TFCE| too —
    used here to build a null vector with a known rank ordering."""
    base = np.array([1.0, 2.0, 1.0])
    scales = np.array([0.0, 1.0, 2.0, 5.0])
    null_curves = scales[:, None] * base[None, :]

    null_vec, _ = max_tfce_null(null_curves, dt=0.1, obs_tfce_max_abs=1e9)

    assert null_vec.shape == (4,)
    assert np.all(np.diff(null_vec) >= 0)  # non-decreasing with amplitude
    assert null_vec[0] == 0.0  # all-zero replicate enhances to zero


def test_max_tfce_null_emp_p_formula():
    """emp_p = (1 + #{null >= obs}) / (1 + R). Use one replicate's own
    max-|TFCE| as the observed statistic, so the expected >= count is known
    by construction (that replicate and every larger-amplitude one)."""
    base = np.array([1.0, 2.0, 1.0])
    scales = np.array([0.0, 1.0, 2.0, 3.0, 10.0])  # R = 5
    null_curves = scales[:, None] * base[None, :]

    null_vec, _ = max_tfce_null(null_curves, dt=0.1, obs_tfce_max_abs=np.inf)
    obs = null_vec[2]
    _, emp_p = max_tfce_null(null_curves, dt=0.1, obs_tfce_max_abs=obs)
    expected_ge_count = int(np.sum(null_vec >= obs))
    expected_p = (1 + expected_ge_count) / (1 + 5)

    assert emp_p == pytest.approx(expected_p)


def test_max_tfce_null_extreme_obs_gives_floor_p():
    """An observed statistic larger than every null replicate hits the
    minimum achievable p, 1 / (R + 1)."""
    rng = np.random.default_rng(1)
    R = 9
    null_curves = rng.random((R, 5))

    null_vec, emp_p = max_tfce_null(null_curves, dt=0.05, obs_tfce_max_abs=1e6)

    assert emp_p == pytest.approx(1.0 / (R + 1))


# ---------------------------------------------------------------------------
# assert_coherent_null_replicates
# ---------------------------------------------------------------------------


def test_assert_coherent_null_replicates_passes_on_expected_shape():
    assert_coherent_null_replicates(n_rows=1000 * 6, R=1000, n_windows=6, context="test cell")


def test_assert_coherent_null_replicates_raises_on_short_rows():
    with pytest.raises(AssertionError, match="Expected 6000"):
        assert_coherent_null_replicates(n_rows=5999, R=1000, n_windows=6, context="test cell")
