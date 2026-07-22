"""
Unit tests for the TFCE gate helpers in notebooks/causal46_joined/_windows.py
(plan Step 2: docs/superpowers/plans/2026-07-20-causal46-late-perceptual-significance.md).

Synthetic-only, pure functions, no pipeline/epoch data required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "causal46_joined"))
from _windows import (  # noqa: E402
    assert_coherent_null_replicates,
    extract_cell_curves,
    late_cell_significance,
    max_tfce_null,
    tfce_enhance,
    validate_contiguous_grid,
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


# ---------------------------------------------------------------------------
# late_cell_significance
# ---------------------------------------------------------------------------


def test_late_cell_significance_signal_cell_gets_small_p():
    """A cell with a consistent bump across replicates, against noise-only
    null curves, should clear the gate with a small two-tailed p on both
    the TFCE and integral statistics."""
    rng = np.random.default_rng(0)
    R, n_windows = 200, 5
    base = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    rep_curves = base[None, :] + rng.normal(0, 0.3, (R, n_windows))
    null_curves = rng.normal(0, 0.3, (R, n_windows))

    out = late_cell_significance(rep_curves, null_curves)

    assert out["tfce_peak"] > 0
    assert out["tfce_max_abs"] == pytest.approx(abs(out["tfce_peak"]))
    assert out["tfce_emp_p"] < 0.05
    assert out["integral_stat"] > 0
    assert out["integral_emp_p"] < 0.05


def test_late_cell_significance_noise_cell_does_not_clear_gate():
    """A cell with no consistent signal (replicates and null drawn from the
    same noise distribution) should not spuriously clear the gate."""
    rng = np.random.default_rng(1)
    R, n_windows = 200, 5
    rep_curves = rng.normal(0, 0.3, (R, n_windows))
    null_curves = rng.normal(0, 0.3, (R, n_windows))

    out = late_cell_significance(rep_curves, null_curves)

    assert out["tfce_emp_p"] > 0.05
    assert out["integral_emp_p"] > 0.05


def test_late_cell_significance_splithalf_agree_true_for_consistent_sign():
    R, n_windows = 10, 3
    rep_curves = np.tile([0.0, 5.0, 0.0], (R, 1))
    null_curves = np.random.default_rng(2).normal(0, 0.1, (R, n_windows))

    out = late_cell_significance(rep_curves, null_curves)

    assert out["splithalf_sign_agree"] is True


def test_late_cell_significance_splithalf_agree_false_for_flipped_half():
    """First half of replicates peaks positive, second half peaks negative
    at the same window — the two halves disagree in sign even though the
    pooled median (and hence the observed curve/peak window) stays positive."""
    R, n_windows = 10, 3
    rep_curves = np.zeros((R, n_windows))
    rep_curves[:5] = [0.0, 5.0, 0.0]
    rep_curves[5:] = [0.0, -1.0, 0.0]
    null_curves = np.random.default_rng(2).normal(0, 0.1, (R, n_windows))

    out = late_cell_significance(rep_curves, null_curves)

    assert out["splithalf_sign_agree"] is False


def test_late_cell_significance_splithalf_none_when_single_replicate():
    rep_curves = np.array([[0.0, 5.0, 0.0]])
    null_curves = np.array([[0.1, -0.1, 0.05]])

    out = late_cell_significance(rep_curves, null_curves)

    assert out["splithalf_sign_agree"] is None


def test_late_cell_significance_all_zero_curves_no_crash():
    R, n_windows = 5, 4
    rep_curves = np.zeros((R, n_windows))
    null_curves = np.zeros((R, n_windows))

    out = late_cell_significance(rep_curves, null_curves)

    assert out["tfce_peak"] == 0.0
    assert out["tfce_max_abs"] == 0.0
    assert out["integral_stat"] == 0.0
    assert 0.0 < out["tfce_emp_p"] <= 1.0


def test_late_cell_significance_raises_on_nan_null():
    rep_curves = np.zeros((3, 2))
    null_curves = np.array([[0.0, np.nan], [0.0, 0.0], [0.0, 0.0]])

    with pytest.raises(AssertionError, match="NaN"):
        late_cell_significance(rep_curves, null_curves)


def test_late_cell_significance_raises_on_shape_mismatch():
    rep_curves = np.zeros((3, 2))
    null_curves = np.zeros((3, 3))

    with pytest.raises(AssertionError, match="shape mismatch"):
        late_cell_significance(rep_curves, null_curves)


# ---------------------------------------------------------------------------
# validate_contiguous_grid
# ---------------------------------------------------------------------------


def test_validate_contiguous_grid_passes_and_returns_width():
    windows = [(0, 10), (10, 20), (20, 30)]
    assert validate_contiguous_grid(windows) == 10


def test_validate_contiguous_grid_raises_on_gap():
    windows = [(0, 10), (20, 30)]
    with pytest.raises(AssertionError, match="Grid gap"):
        validate_contiguous_grid(windows)


def test_validate_contiguous_grid_raises_on_non_uniform_width():
    windows = [(0, 10), (10, 25)]
    with pytest.raises(AssertionError, match="Non-uniform"):
        validate_contiguous_grid(windows)


def test_validate_contiguous_grid_raises_on_empty():
    with pytest.raises(AssertionError, match="No windows"):
        validate_contiguous_grid([])


# ---------------------------------------------------------------------------
# extract_cell_curves
# ---------------------------------------------------------------------------

GRID = [(0, 10), (10, 20), (20, 30)]


def _boot_rows(subject, eidx, pp, we, windows, R, rng, *, tied=False):
    rows = []
    for r in range(R):
        for smin, smax in windows:
            rows.append({
                "subject": subject, "electrode_idx": eidx, "phoneme_pair": pp, "word_end": we,
                "replicate": r, "smin": smin, "smax": smax,
                "mean_diff_raw": float(rng.normal(0, 1)),
                "mean_diff_aligned_null": float("nan") if tied else float(rng.normal(0, 1)),
            })
    return rows


def _per_cell_row(subject, eidx, pp, we, phon_smax):
    return {
        "subject": subject, "electrode_idx": eidx, "phoneme_pair": pp, "word_end": we,
        "phon_smin": 0, "phon_smax": phon_smax,
    }


def test_extract_cell_curves_four_statuses():
    """Four cells covering every branch: ok, no_candidates, missing, tied."""
    rng = np.random.default_rng(0)
    per_cell_rows = [
        _per_cell_row("S1", 1, "dn", "w1", phon_smax=10),   # ok: (10,20),(20,30)
        _per_cell_row("S1", 2, "dn", "w1", phon_smax=25),   # no_candidates: grid maxes at smin=20
        _per_cell_row("S1", 3, "dn", "w1", phon_smax=0),    # missing: no bootstrap rows at all
        _per_cell_row("S1", 4, "dn", "w1", phon_smax=10),   # tied: null all-NaN
    ]
    boot_rows = []
    boot_rows += _boot_rows("S1", 1, "dn", "w1", [(10, 20), (20, 30)], R=4, rng=rng)
    boot_rows += _boot_rows("S1", 2, "dn", "w1", [(0, 10)], R=4, rng=rng)  # present but out of range
    boot_rows += _boot_rows("S1", 4, "dn", "w1", [(10, 20), (20, 30)], R=4, rng=rng, tied=True)

    b4_bootstrap = pl.DataFrame(boot_rows)
    b4_per_cell = pl.DataFrame(per_cell_rows)

    out = extract_cell_curves(b4_bootstrap, b4_per_cell, GRID, we_search_smax={"w1": None})

    ok = out[("S1", 1, "dn", "w1")]
    assert ok["status"] == "ok"
    assert ok["n_windows"] == 2
    assert ok["rep_curves"].shape == (4, 2)
    assert ok["null_curves"].shape == (4, 2)
    assert ok["search_smin"] == 10

    no_cand = out[("S1", 2, "dn", "w1")]
    assert no_cand["status"] == "no_candidates"
    assert no_cand["rep_curves"] is None

    missing = out[("S1", 3, "dn", "w1")]
    assert missing["status"] == "missing"
    assert missing["rep_curves"] is None

    tied = out[("S1", 4, "dn", "w1")]
    assert tied["status"] == "tied"
    assert tied["rep_curves"] is None


def test_extract_cell_curves_respects_we_search_smax_bound():
    """A word-end search bound should exclude windows beyond it (D4)."""
    rng = np.random.default_rng(1)
    per_cell_rows = [_per_cell_row("S1", 1, "dn", "w1", phon_smax=0)]
    boot_rows = _boot_rows("S1", 1, "dn", "w1", GRID, R=3, rng=rng)
    b4_bootstrap = pl.DataFrame(boot_rows)
    b4_per_cell = pl.DataFrame(per_cell_rows)

    out = extract_cell_curves(b4_bootstrap, b4_per_cell, GRID, we_search_smax={"w1": 20})

    cell = out[("S1", 1, "dn", "w1")]
    assert cell["status"] == "ok"
    assert cell["n_windows"] == 2  # (0,10) and (10,20); (20,30) excluded (smax > 20)
    assert cell["search_smax"] == 20


def test_extract_cell_curves_feeds_late_cell_significance_directly():
    """The extracted curves should be usable as-is by late_cell_significance
    (integration between the two shared helpers)."""
    rng = np.random.default_rng(2)
    per_cell_rows = [_per_cell_row("S1", 1, "dn", "w1", phon_smax=10)]
    boot_rows = _boot_rows("S1", 1, "dn", "w1", [(10, 20), (20, 30)], R=50, rng=rng)
    b4_bootstrap = pl.DataFrame(boot_rows)
    b4_per_cell = pl.DataFrame(per_cell_rows)

    out = extract_cell_curves(b4_bootstrap, b4_per_cell, GRID, we_search_smax={"w1": None})
    cell = out[("S1", 1, "dn", "w1")]

    sig = late_cell_significance(cell["rep_curves"], cell["null_curves"])
    assert 0.0 < sig["tfce_emp_p"] <= 1.0
