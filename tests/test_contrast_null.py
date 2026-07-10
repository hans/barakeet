"""Tests for oriented_group_band and the _permute_per_step null machinery.

Four cases from issue #5:
1. Pure-noise cells → observed grand mean inside the null band; null band
   has positive mean inside the orientation window (rectification floor).
2. Signal-injected cells → observed exits the null band in the signal window.
3. Determinism: same seed → identical null_matrix; different seed → different
   null_matrix (but signal conclusions hold either way).
4. Guard for story 3: reusing the fixed observed sign for each null replicate
   (the wrong approach) collapses the null band to near-zero mean, hiding the
   rectification floor; recomputing the sign (the correct approach) gives a
   wider, positive band.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks" / "causal46_joined"))
from _contrast import _permute_per_step, behavioral_bootstrap_meandiff, oriented_group_band

# --------------------------------------------------------------------------- #
# Synthetic-epoch helpers
# --------------------------------------------------------------------------- #
SFREQ = 100.0
TMIN = -0.1
TMAX = 0.5
N_TIMES = int((TMAX - TMIN) * SFREQ) + 1  # 61


def _make_epochs(
    n_per_class_per_step: int = 6,
    seed: int = 0,
    signal_smin: int | None = None,
    signal_smax: int | None = None,
    signal_amp: float = 20.0,
    phoneme_pair: str = "dn",
    word_end: str = "necessary",
    steps: tuple[int, ...] = (2, 3, 4),
) -> mne.EpochsArray:
    """Build a minimal synthetic EpochsArray suitable for behavioral_bootstrap_meandiff.

    Channel 0 carries noise (and optionally a class-0 signal bump).
    Metadata has the columns expected by resolve_behavior_col and
    per_step_class_counts.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for step in steps:
        for cls in [0, 1]:
            for _ in range(n_per_class_per_step):
                rows.append({
                    "phoneme_pair": phoneme_pair,
                    "word_end": word_end,
                    "resampled": step,
                    "behavior_categorical": cls,
                })
    md = pd.DataFrame(rows)
    n_trials = len(md)

    data = rng.standard_normal((n_trials, 1, N_TIMES)) * 1e-6

    if signal_smin is not None and signal_smax is not None:
        for i, cls in enumerate(md["behavior_categorical"]):
            if cls == 0:
                data[i, 0, signal_smin:signal_smax] += signal_amp * 1e-6

    info = mne.create_info(["ch0"], sfreq=SFREQ, ch_types="eeg")
    return mne.EpochsArray(data, info, tmin=TMIN, metadata=md, verbose=False)




# --------------------------------------------------------------------------- #
# _permute_per_step
# --------------------------------------------------------------------------- #
class TestPermutePerStep:
    def test_preserves_class_sizes(self):
        per_step = {
            2: {0: np.array([0, 1, 2]), 1: np.array([3, 4, 5])},
            3: {0: np.array([6, 7]), 1: np.array([8, 9])},
        }
        rng = np.random.default_rng(0)
        perm = _permute_per_step(per_step, rng)
        for s, by_class in per_step.items():
            assert set(perm[s].keys()) == set(by_class.keys())
            for k in by_class:
                assert len(perm[s][k]) == len(by_class[k])

    def test_uses_same_index_pool(self):
        per_step = {
            2: {0: np.array([0, 1, 2]), 1: np.array([3, 4, 5])},
        }
        rng = np.random.default_rng(99)
        perm = _permute_per_step(per_step, rng)
        pool_orig = set(per_step[2][0].tolist() + per_step[2][1].tolist())
        pool_perm = set(perm[2][0].tolist() + perm[2][1].tolist())
        assert pool_orig == pool_perm

    def test_different_from_original_most_of_the_time(self):
        per_step = {
            2: {0: np.arange(20), 1: np.arange(20, 40)},
        }
        n_shuffled = 0
        for seed in range(30):
            rng = np.random.default_rng(seed)
            perm = _permute_per_step(per_step, rng)
            if not np.array_equal(perm[2][0], per_step[2][0]):
                n_shuffled += 1
        assert n_shuffled > 20, "permutation almost never changes indices — seeding bug?"


# --------------------------------------------------------------------------- #
# behavioral_bootstrap_meandiff with perm_rng
# --------------------------------------------------------------------------- #
class TestBehavioralBootstrapPerm:
    def test_null_path_returns_ok_status(self):
        ep = _make_epochs(n_per_class_per_step=5)
        rng = np.random.default_rng(0)
        diff, status = behavioral_bootstrap_meandiff(
            ep, 0, "necessary",
            min_class_k=3, bootstrap_r=10, bootstrap_seed=42,
            perm_rng=rng,
        )
        assert status == "ok"
        assert diff is not None
        assert diff.shape == (N_TIMES,)

    def test_null_and_observed_same_shape(self):
        ep = _make_epochs(n_per_class_per_step=5)
        obs, _ = behavioral_bootstrap_meandiff(
            ep, 0, "necessary",
            min_class_k=3, bootstrap_r=10, bootstrap_seed=42,
        )
        null, _ = behavioral_bootstrap_meandiff(
            ep, 0, "necessary",
            min_class_k=3, bootstrap_r=10, bootstrap_seed=42,
            perm_rng=np.random.default_rng(0),
        )
        assert obs.shape == null.shape


# --------------------------------------------------------------------------- #
# Helpers for independent-cell fixtures
# --------------------------------------------------------------------------- #
N_CELLS_MULTI = 6


def _make_multi(n_cells=N_CELLS_MULTI, smin=20, smax=35, **ep_kwargs):
    """Build n_cells truly independent (subject, epoch) pairs.

    Each subject gets its own EpochsArray from a different seed so that the
    grand-mean trajectory averages genuinely independent draws — necessary for
    the observed to converge toward the null under H0.
    """
    epochs_dict = {
        f"S{i}": _make_epochs(seed=100 + i, **ep_kwargs)
        for i in range(n_cells)
    }
    cells = [
        {
            "subject": f"S{i}",
            "electrode_idx": 0,
            "phoneme_pair": "dn",
            "word_end": "necessary",
            "smin": smin,
            "smax": smax,
        }
        for i in range(n_cells)
    ]
    return cells, epochs_dict


# --------------------------------------------------------------------------- #
# Case 1: pure-noise → null band has positive mean in window (rectification floor)
# --------------------------------------------------------------------------- #
class TestNullBandNoise:
    @pytest.fixture(scope="class")
    def band_result(self):
        cells, epochs_dict = _make_multi(n_cells=N_CELLS_MULTI,
                                         n_per_class_per_step=8)
        obs, null_mat, n_valid = oriented_group_band(
            cells, epochs_dict,
            n_perm=80, seed=0,
            min_class_k=3, bootstrap_r=30, bootstrap_seed=42,
        )
        return obs, null_mat, n_valid

    def test_valid_cells_nonzero(self, band_result):
        _, _, n_valid = band_result
        assert n_valid > 0

    def test_null_matrix_shape(self, band_result):
        _, null_mat, _ = band_result
        assert null_mat.shape == (80, N_TIMES)

    def test_observed_rank_in_null_not_extreme(self, band_result):
        # Under H0, the observed grand-mean's time-average should not be a
        # systematic outlier relative to the null distribution of time-averages.
        # Using the time-mean reduces this to a scalar comparison where the
        # 80 null reps provide a cleaner reference.
        obs, null_mat, _ = band_result
        obs_scalar = obs.mean()
        null_scalars = null_mat.mean(axis=1)
        rank = float((null_scalars < obs_scalar).mean())
        assert 0.03 <= rank <= 0.97, (
            f"observed time-mean ranks at {rank:.2f} in null — "
            "may indicate systematic bias in the implementation"
        )

    def test_null_band_has_positive_mean_in_orientation_window(self, band_result):
        # Rectification floor: the null band should sit above zero in the
        # orientation window (smin:smax = 20:35) because the sign was chosen to
        # make the window average positive — a bias that is equally present in
        # the null trajectory when the sign is recomputed from permuted data.
        _, null_mat, _ = band_result
        smin, smax = 20, 35
        null_window_mean = null_mat[:, smin:smax].mean(axis=1)
        assert null_window_mean.mean() > 0, (
            "null band mean in orientation window should be positive "
            "(rectification floor); got non-positive mean"
        )


# --------------------------------------------------------------------------- #
# Case 2: injected signal → observed exits null band in signal window
# --------------------------------------------------------------------------- #
class TestNullBandSignal:
    @pytest.fixture(scope="class")
    def band_result(self):
        # Signal: class 0 gets a positive bump at samples 30–45.
        # Use independent cells (different seeds per subject).
        cells, epochs_dict = _make_multi(
            n_cells=N_CELLS_MULTI, smin=30, smax=45,
            n_per_class_per_step=8,
            signal_smin=30, signal_smax=45, signal_amp=30.0,
        )
        obs, null_mat, n_valid = oriented_group_band(
            cells, epochs_dict,
            n_perm=60, seed=0,
            min_class_k=3, bootstrap_r=30, bootstrap_seed=42,
        )
        return obs, null_mat, n_valid

    def test_observed_exceeds_null_in_signal_window(self, band_result):
        obs, null_mat, _ = band_result
        null_hi = np.percentile(null_mat, 97.5, axis=0)
        # Observed should exceed the 97.5th percentile of the null in the
        # signal window (samples 30–45)
        exceeds_in_window = (obs[30:45] > null_hi[30:45]).mean()
        assert exceeds_in_window >= 0.5, (
            f"only {exceeds_in_window:.1%} of signal-window timepoints exceed "
            "the null band 97.5th percentile"
        )


# --------------------------------------------------------------------------- #
# Case 3: determinism
# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_same_seed_same_null(self):
        cells, epochs_dict = _make_multi(n_cells=3, n_per_class_per_step=6)
        kw = dict(n_perm=20, seed=7, min_class_k=3, bootstrap_r=15, bootstrap_seed=42)
        _, null1, _ = oriented_group_band(cells, epochs_dict, **kw)
        _, null2, _ = oriented_group_band(cells, epochs_dict, **kw)
        np.testing.assert_array_equal(null1, null2)

    def test_different_seed_different_null(self):
        cells, epochs_dict = _make_multi(n_cells=3, n_per_class_per_step=6)
        kw = dict(n_perm=20, min_class_k=3, bootstrap_r=15, bootstrap_seed=42)
        _, null_a, _ = oriented_group_band(cells, epochs_dict, seed=7, **kw)
        _, null_b, _ = oriented_group_band(cells, epochs_dict, seed=99, **kw)
        assert not np.allclose(null_a, null_b), (
            "null matrices from different seeds should differ"
        )


# --------------------------------------------------------------------------- #
# Case 4: sign-reuse guard
# --------------------------------------------------------------------------- #
def _oriented_group_band_fixed_sign(
    cells: list[dict],
    epochs_dict: dict,
    *,
    n_perm: int,
    seed: int,
    min_class_k: int,
    bootstrap_r: int,
    bootstrap_seed: int,
    candidate_steps=(2, 3, 4, 5),
):
    """Wrong implementation: reuse the observed sign for every null replicate.

    Expected behavior: the null band mean in the orientation window collapses
    toward zero because the fixed sign is uncorrelated with each permuted
    trajectory.
    """
    from _contrast import behavioral_bootstrap_meandiff, _permute_per_step
    kw = dict(min_class_k=min_class_k, bootstrap_r=bootstrap_r,
               bootstrap_seed=bootstrap_seed, candidate_steps=candidate_steps)
    n_times = None
    obs_sum = None
    null_matrix = None
    n_valid = 0
    obs_signs = []

    for cell_idx, cell in enumerate(cells):
        ep = epochs_dict[cell["subject"]]
        ep_pp = ep[ep.metadata["phoneme_pair"].values == cell["phoneme_pair"]]
        smin, smax = int(cell["smin"]), int(cell["smax"])

        obs_diff, status = behavioral_bootstrap_meandiff(ep_pp, 0, cell["word_end"], **kw)
        if status != "ok":
            continue
        if n_times is None:
            n_times = len(obs_diff)
            obs_sum = np.zeros(n_times)
            null_matrix = np.zeros((n_perm, n_times))

        obs_sign = float(np.sign(obs_diff[smin:smax].mean()) or 1.0)
        obs_signs.append((cell_idx, obs_sign, ep_pp, smin, smax, cell["word_end"]))
        obs_sum += obs_sign * obs_diff
        n_valid += 1

    if n_valid == 0:
        return None, None, 0

    for p in range(n_perm):
        for i, (cell_idx, obs_sign, ep_pp, smin, smax, word_end) in enumerate(obs_signs):
            prng = np.random.default_rng([seed, cell_idx, p])
            perm_diff, perm_status = behavioral_bootstrap_meandiff(
                ep_pp, 0, word_end, perm_rng=prng, **kw
            )
            if perm_status != "ok":
                continue
            # WRONG: use the fixed observed sign instead of recomputing
            null_matrix[p] += obs_sign * perm_diff

    return obs_sum / n_valid, null_matrix / n_valid, n_valid


class TestSignReuseGuard:
    """Verify that reusing the fixed observed sign collapses the null band."""

    @pytest.fixture(scope="class")
    def results(self):
        cells, epochs_dict = _make_multi(n_cells=N_CELLS_MULTI,
                                         n_per_class_per_step=8)
        kw = dict(n_perm=60, seed=0, min_class_k=3, bootstrap_r=30, bootstrap_seed=42)
        _, null_correct, _ = oriented_group_band(cells, epochs_dict, **kw)
        _, null_wrong, _ = _oriented_group_band_fixed_sign(cells, epochs_dict, **kw)
        return null_correct, null_wrong

    def test_correct_null_has_positive_floor(self, results):
        null_correct, _ = results
        window_mean = null_correct[:, 20:35].mean()
        assert window_mean > 0, (
            "correct null (recomputed sign) should have positive mean "
            f"in orientation window; got {window_mean:.4f}"
        )

    def test_wrong_null_floor_near_zero(self, results):
        _, null_wrong = results
        window_mean = null_wrong[:, 20:35].mean()
        # The wrong null should have a much smaller mean than the correct one
        # (close to zero because fixed sign is uncorrelated with permuted data)
        correct_mean = results[0][:, 20:35].mean()
        assert window_mean < correct_mean * 0.5, (
            f"wrong null (fixed sign) window mean {window_mean:.4f} should be "
            f"substantially below correct null window mean {correct_mean:.4f}"
        )
