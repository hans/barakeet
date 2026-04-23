"""
Integration-ish tests for causal6 decoder entry points.

Uses a small synthetic `mne.EpochsArray` fixture with a planted signal on
specific electrodes. Verifies that:

  - The searchlight actually detects the planted signal (AUC well above chance
    on the signal electrodes, near chance on pure-noise electrodes).
  - Schema and row counts match expectations.
  - StratifiedKFold CV discipline holds: every trial appears exactly once as a
    held-out prediction per (decoder key).

Synthetic-only by design; any real-data smoke tests should go in a separate
integration suite gated on data availability.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import polars as pl
import torch

from src.models.causal6 import (
    make_windows,
    run_acoustic_searchlight,
    run_behavior_hga_only,
    run_behavior_with_control,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_fake_epochs(
    *,
    n_trials_per_class: int = 120,
    n_electrodes: int = 4,
    n_samples: int = 60,
    sfreq: float = 100.0,
    tmin: float = -0.1,
    signal_electrodes: tuple[int, ...] = (0,),
    signal_window: tuple[int, int] = (10, 25),
    signal_amp: float = 2.0,
    phoneme_pair: str = "dn",
    word_ends: tuple[str, ...] = ("desolate", "necessary"),
    # behavior target: P(behavior=1 | acoustic=+1) vs P(behavior=1 | acoustic=-1)
    behavior_follows_acoustic: bool = True,
    seed: int = 0,
) -> mne.EpochsArray:
    """
    Build a small synthetic epochs object with metadata columns required by the
    causal6 decoders and a planted acoustic signal on `signal_electrodes`
    during `signal_window` (in samples).

    Layout:
      - n_trials = 2 * n_trials_per_class, split 50/50 on categorical_acoustic_cue
      - word_end alternates between the two listed completions (independent of cue)
      - resampled is 1 where cue=-1, 6 where cue=+1 (clean endpoints only)
      - behavior_dummy_forced either follows the cue exactly or is random

    The "signal" is a step-function added to `signal_electrodes` at samples
    `[signal_window[0]:signal_window[1]]`, signed by categorical_acoustic_cue.
    """
    rng = np.random.default_rng(seed)

    n_trials = 2 * n_trials_per_class
    cue = np.concatenate([
        -np.ones(n_trials_per_class, dtype=np.int64),
        +np.ones(n_trials_per_class, dtype=np.int64),
    ])
    # Shuffle so positive and negative cues interleave
    order = rng.permutation(n_trials)
    cue = cue[order]

    data = rng.standard_normal((n_trials, n_electrodes, n_samples))
    smin, smax = signal_window
    for e in signal_electrodes:
        data[:, e, smin:smax] += signal_amp * cue[:, None]

    word_end = np.array([word_ends[i % len(word_ends)] for i in range(n_trials)])
    rng.shuffle(word_end)

    resampled = np.where(cue > 0, 6, 1)
    behavior = cue.copy() if behavior_follows_acoustic else rng.choice([-1, 1], size=n_trials)

    md = pd.DataFrame({
        "phoneme_pair": [phoneme_pair] * n_trials,
        "categorical_acoustic_cue": cue.astype(float),
        "word_end": word_end,
        "resampled": resampled,
        "behavior_categorical_forced": behavior.astype(float),
        "behavior_dummy_forced": (behavior > 0).astype(int),
    })

    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(n_electrodes)],
        sfreq=sfreq,
        ch_types=["seeg"] * n_electrodes,
    )
    epochs = mne.EpochsArray(data, info, tmin=tmin, verbose=False)
    epochs.metadata = md
    return epochs


# ---------------------------------------------------------------------------
# run_acoustic_searchlight
# ---------------------------------------------------------------------------


def test_acoustic_searchlight_detects_planted_signal():
    """A signal electrode's peak-window AUC should be >>0.9; noise electrodes ~0.5."""
    signal_e = 0
    epochs = _make_fake_epochs(
        n_electrodes=5,
        signal_electrodes=(signal_e,),
        signal_window=(15, 30),
        signal_amp=3.0,
        seed=1,
    )
    windows = make_windows(0, epochs.times.shape[0], window_size=10, stride=5)

    scores, preds, coefs = run_acoustic_searchlight(
        epochs, subject="SYN",
        electrode_idxs=list(range(5)),
        windows=windows,
        reg_lambda=1.0,
        target="categorical_acoustic_cue",
        n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )

    # Peak AUC per electrode (averaged over folds, max over windows)
    peaks = (
        scores.group_by(["electrode_idx", "smin", "smax"])
        .agg(pl.col("test_roc_auc").mean())
        .group_by("electrode_idx")
        .agg(pl.col("test_roc_auc").max().alias("peak_auc"))
        .sort("electrode_idx")
    )
    peak_dict = {row["electrode_idx"]: row["peak_auc"] for row in peaks.to_dicts()}

    assert peak_dict[signal_e] > 0.9, (
        f"signal electrode peak AUC {peak_dict[signal_e]:.3f} too low"
    )
    # Noise electrodes: their best-by-chance peak can drift above 0.5 in small n;
    # require a clear gap from the signal electrode.
    for e in range(5):
        if e == signal_e:
            continue
        assert peak_dict[e] < peak_dict[signal_e] - 0.2, (
            f"noise electrode {e} too hot: {peak_dict[e]:.3f}"
        )


def test_acoustic_searchlight_schema_and_shapes():
    """Output DataFrames have expected columns and row counts."""
    epochs = _make_fake_epochs(n_electrodes=3, seed=2)
    windows = make_windows(0, epochs.times.shape[0], window_size=10, stride=5)
    n_windows = windows.shape[0]
    n_folds = 5

    scores, preds, coefs = run_acoustic_searchlight(
        epochs, subject="SYN",
        electrode_idxs=[0, 1, 2],
        windows=windows,
        reg_lambda=1.0, target="categorical_acoustic_cue",
        n_folds=n_folds, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )

    # scores: 3 electrodes × n_windows × n_folds (one phoneme_pair)
    assert scores.height == 3 * n_windows * n_folds
    assert set(scores.columns) >= {
        "subject", "phoneme_pair", "electrode_idx", "smin", "smax",
        "fold", "test_roc_auc", "n_train", "n_test", "n_iter", "converged",
        "target",
    }
    assert scores["target"].unique().to_list() == ["categorical_acoustic_cue"]

    # coefficients: one row per (electrode × window × fold)
    assert coefs.height == 3 * n_windows * n_folds
    # Each coef vector has length d = window_size
    any_coef = coefs["coef"].to_list()[0]
    assert len(any_coef) == 10

    # predictions: every (electrode × window × fold) produces n_test rows,
    # and n_test over all folds sums to n_trials.
    n_trials = epochs.metadata.shape[0]
    total_preds_per_key = (
        preds.group_by(["electrode_idx", "smin", "smax"])
        .agg(pl.len().alias("n_test_rows"))
    )
    assert (total_preds_per_key["n_test_rows"] == n_trials).all()


def test_acoustic_searchlight_cv_partition_is_disjoint():
    """StratifiedKFold: each trial appears as held-out test exactly once per key."""
    epochs = _make_fake_epochs(n_electrodes=2, seed=3)
    windows = make_windows(0, epochs.times.shape[0], window_size=10, stride=5)

    _, preds, _ = run_acoustic_searchlight(
        epochs, subject="SYN",
        electrode_idxs=[0, 1],
        windows=windows,
        reg_lambda=1.0, target="categorical_acoustic_cue",
        n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )

    # For each (electrode_idx, smin, smax), each epoch_idx should appear exactly once.
    per_key_epoch_counts = (
        preds.group_by(["electrode_idx", "smin", "smax", "epoch_idx"])
        .agg(pl.len().alias("n"))
    )
    assert (per_key_epoch_counts["n"] == 1).all()


# ---------------------------------------------------------------------------
# run_behavior_with_control + run_behavior_hga_only
# ---------------------------------------------------------------------------


def test_behavior_with_control_has_both_full_and_baseline_rows():
    """behavior_full produces model='full' per (electrode, window) and model='baseline' once per (phoneme_pair, word_end)."""
    epochs = _make_fake_epochs(n_electrodes=2, seed=4)
    windows = make_windows(0, epochs.times.shape[0], window_size=10, stride=5)
    n_windows = windows.shape[0]
    n_folds = 5

    scores, _, coefs = run_behavior_with_control(
        epochs, subject="SYN",
        electrode_idxs=[0, 1],
        windows=windows,
        reg_lambda=1.0,
        n_folds=n_folds, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )

    models = set(scores["model"].unique().to_list())
    assert models == {"full", "baseline"}

    # Full rows: 2 electrodes × n_windows × 2 word_ends × n_folds
    full_rows = scores.filter(pl.col("model") == "full").height
    assert full_rows == 2 * n_windows * 2 * n_folds

    # Baseline rows: one per (word_end × fold), no electrode/window variation
    base = scores.filter(pl.col("model") == "baseline")
    assert base.height == 2 * n_folds
    assert (base["electrode_idx"] == -1).all()
    assert (base["smin"] == -1).all()


def test_behavior_hga_only_has_no_baseline_rows():
    """behavior_hga_only emits only model='full' rows."""
    epochs = _make_fake_epochs(n_electrodes=2, seed=5)
    windows = make_windows(0, epochs.times.shape[0], window_size=10, stride=5)

    scores, _, _ = run_behavior_hga_only(
        epochs, subject="SYN",
        electrode_idxs=[0, 1],
        windows=windows,
        reg_lambda=1.0,
        n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )

    assert scores["model"].unique().to_list() == ["full"]


def test_behavior_with_control_detects_planted_signal():
    """
    When behavior perfectly follows acoustic cue and a planted signal exists
    on one electrode, behavior_full's AUC on that electrode's peak window
    should clear 0.9.
    """
    signal_e = 0
    epochs = _make_fake_epochs(
        n_electrodes=3,
        signal_electrodes=(signal_e,),
        signal_window=(15, 30),
        signal_amp=3.0,
        behavior_follows_acoustic=True,
        seed=6,
    )
    windows = make_windows(0, epochs.times.shape[0], window_size=10, stride=5)

    scores, _, _ = run_behavior_with_control(
        epochs, subject="SYN",
        electrode_idxs=[0, 1, 2],
        windows=windows,
        reg_lambda=1.0,
        n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )

    peak_full = (
        scores.filter(pl.col("model") == "full")
        .group_by(["electrode_idx", "smin", "smax"])
        .agg(pl.col("test_roc_auc").mean())
        .group_by("electrode_idx")
        .agg(pl.col("test_roc_auc").max().alias("peak_auc"))
    )
    signal_peak = (
        peak_full.filter(pl.col("electrode_idx") == signal_e)["peak_auc"].item()
    )
    assert signal_peak > 0.9, f"signal electrode peak full AUC {signal_peak:.3f} too low"
