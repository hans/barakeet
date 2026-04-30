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
    _fit_batched_cv,
    _fit_batched_cv_permutations,
    audit_class_balance,
    make_windows,
    run_acoustic_searchlight,
    run_acoustic_searchlight_permutations,
    run_behavior_hga_only,
    run_behavior_with_control,
    run_behavior_with_control_permutations,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_fake_epochs(
    *,
    n_trials_per_step: int = 40,
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
    Build a small synthetic epochs object mirroring the real causal5/causal6
    pipeline's metadata contract and a planted acoustic signal on
    `signal_electrodes` during `signal_window`.

    Layout:
      - resampled is uniform over {1, 2, 3, 4, 5, 6} — 6 stimulus steps,
        matching the real 6-step /d/-to-/n/ continuum
      - categorical_acoustic_cue = -1 where resampled <= 3, +1 where resampled >= 4
        (matches src/data.py:add_metadata_features)
      - word_end alternates between the two listed completions
      - behavior_dummy_forced either follows the cue exactly or is random

    The acoustic decoder filters to resampled ∈ {1, 6} by default
    (2 * n_trials_per_step trials after filter), so the signal must survive
    that subset.
    """
    rng = np.random.default_rng(seed)

    steps = np.tile(np.arange(1, 7, dtype=np.int64), n_trials_per_step)
    rng.shuffle(steps)
    n_trials = len(steps)
    cue = np.where(steps > 3, 1, -1).astype(np.int64)

    data = rng.standard_normal((n_trials, n_electrodes, n_samples))
    smin, smax = signal_window
    for e in signal_electrodes:
        data[:, e, smin:smax] += signal_amp * cue[:, None]

    word_end = np.array([word_ends[i % len(word_ends)] for i in range(n_trials)])
    rng.shuffle(word_end)

    behavior = cue.copy() if behavior_follows_acoustic else rng.choice([-1, 1], size=n_trials)

    md = pd.DataFrame({
        "phoneme_pair": [phoneme_pair] * n_trials,
        "categorical_acoustic_cue": cue.astype(float),
        "word_end": word_end,
        "resampled": steps.astype(float),
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
    # and n_test over all folds sums to n_trials_after_acoustic_filter.
    # Acoustic searchlight filters to resampled ∈ {1, 6} by default.
    n_trials_filtered = (
        epochs.metadata["resampled"].isin([1, 6]).sum()
    )
    total_preds_per_key = (
        preds.group_by(["electrode_idx", "smin", "smax"])
        .agg(pl.len().alias("n_test_rows"))
    )
    assert (total_preds_per_key["n_test_rows"] == n_trials_filtered).all()


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


# ---------------------------------------------------------------------------
# Permutation-test helpers: _fit_batched_cv_permutations
# ---------------------------------------------------------------------------


def _small_cv_problem(seed: int = 0, B: int = 3, n: int = 80, d: int = 5):
    """Synthetic (X, y, problem_meta) for _fit_batched_cv_permutations tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, B, d))
    y = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(np.int64)
    rng.shuffle(y)
    problem_meta = pl.DataFrame({
        "subject": ["SYN"] * B,
        "phoneme_pair": ["dn"] * B,
        "electrode_idx": np.arange(B, dtype=np.int64),
    })
    return X, y, problem_meta


def test_fit_batched_cv_permutations_reproducibility():
    """Same seeds → identical scores. Different seeds → different scores."""
    X, y, pm = _small_cv_problem(seed=0)

    kwargs = dict(
        reg_lambda=1.0, n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64, tol=1e-6, max_iter=50,
        permutation_chunk_size=2,
    )

    s0 = _fit_batched_cv_permutations(X, y, pm, permute_seeds=[7, 7, 9], **kwargs)

    # Duplicate seeds → duplicate scores (join on (permutation_idx, electrode_idx, fold))
    join_keys = ["electrode_idx", "fold"]
    s_seed7a = s0.filter(pl.col("permutation_idx") == 0).sort(join_keys)
    s_seed7b = s0.filter(pl.col("permutation_idx") == 1).sort(join_keys)
    s_seed9 = s0.filter(pl.col("permutation_idx") == 2).sort(join_keys)

    np.testing.assert_array_equal(
        s_seed7a["test_roc_auc"].to_numpy(),
        s_seed7b["test_roc_auc"].to_numpy(),
    )
    # Different seed should give different AUCs on at least one row.
    assert not np.array_equal(
        s_seed7a["test_roc_auc"].to_numpy(),
        s_seed9["test_roc_auc"].to_numpy(),
    )


def test_fit_batched_cv_permutations_chunk_invariance():
    """chunk_size doesn't affect the per-(permutation, problem, fold) AUC."""
    X, y, pm = _small_cv_problem(seed=1)

    seeds = [5, 11, 13, 17, 19]
    common = dict(
        permute_seeds=seeds,
        reg_lambda=1.0, n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64, tol=1e-6, max_iter=50,
    )

    s_c1 = _fit_batched_cv_permutations(X, y, pm, permutation_chunk_size=1, **common)
    s_c3 = _fit_batched_cv_permutations(X, y, pm, permutation_chunk_size=3, **common)
    s_call = _fit_batched_cv_permutations(X, y, pm, permutation_chunk_size=len(seeds), **common)

    sort_keys = ["permutation_idx", "electrode_idx", "fold"]
    a1 = s_c1.sort(sort_keys)["test_roc_auc"].to_numpy()
    a3 = s_c3.sort(sort_keys)["test_roc_auc"].to_numpy()
    aall = s_call.sort(sort_keys)["test_roc_auc"].to_numpy()

    np.testing.assert_array_equal(a1, a3)
    np.testing.assert_array_equal(a1, aall)


def test_fit_batched_cv_permutations_matches_manual_reference():
    """
    Reference: split on REAL y, then for each fold manually fit on
    (X, shuffled-y)[train_idx] and score on (X, shuffled-y)[test_idx].
    `_fit_batched_cv_permutations` should match.
    """
    from sklearn.model_selection import StratifiedKFold

    from src.models.decoding_gpu import (
        batched_roc_auc,
        compute_balanced_sample_weight,
        fit_batched_l2_logreg,
        standardise_per_batch,
    )

    X, y, pm = _small_cv_problem(seed=2, B=2, n=60, d=4)
    seed = 42
    y_perm = np.random.default_rng(seed).permutation(y.astype(np.int64))

    # Gold: direct kernel call per fold on shuffled labels with REAL-y splits.
    B = X.shape[1]
    n_tr_trials = X.shape[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    gold_aucs = {}  # (fold, b) -> auc
    X_b = torch.tensor(X.transpose(1, 0, 2).copy(), dtype=torch.float64)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(n_tr_trials), y)):
        X_tr = X_b[:, tr]
        X_te = X_b[:, te]
        mask = torch.ones(B, len(tr), dtype=torch.float64)
        X_tr_std, X_te_std, _, _ = standardise_per_batch(X_tr, mask, X_te)
        y_tr = torch.tensor(
            np.broadcast_to(y_perm[tr], (B, len(tr))).astype(np.float64),
            dtype=torch.float64,
        )
        sw = compute_balanced_sample_weight(y_tr, mask)
        beta, _, _ = fit_batched_l2_logreg(
            X_tr_std, y_tr, mask, sw,
            reg_lambda=1.0, tol=1e-6, max_iter=50,
        )
        z = torch.einsum("bnd,bd->bn", X_te_std, beta)
        proba = torch.sigmoid(z)
        y_te = torch.tensor(
            np.broadcast_to(y_perm[te], (B, len(te))).astype(np.float64),
            dtype=torch.float64,
        )
        aucs = batched_roc_auc(proba, y_te).numpy()
        for b in range(B):
            gold_aucs[(fold, b)] = aucs[b]

    result = _fit_batched_cv_permutations(
        X, y, pm,
        permute_seeds=[seed],
        permutation_chunk_size=1,
        reg_lambda=1.0, n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64, tol=1e-6, max_iter=50,
    )
    assert result["permutation_idx"].unique().to_list() == [0]
    for row in result.iter_rows(named=True):
        expected = gold_aucs[(row["fold"], row["electrode_idx"])]
        got = row["test_roc_auc"]
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            np.testing.assert_allclose(got, expected, atol=1e-10)


def test_fit_batched_cv_permutations_spill_matches_inmem(tmp_path):
    """spill_dir mode produces identical rows to the in-memory return path.

    Stage-2 nulls on high-electrode-count subjects can exceed RAM; spill
    mode streams chunks to parquet and lets the caller scan_parquet +
    filter lazily. The streamed shards must be byte-equivalent to the
    DataFrame we'd otherwise return (modulo row order).
    """
    X, y, pm = _small_cv_problem(seed=3, B=4, n=80, d=5)
    seeds = [2, 4, 6, 8, 10, 12]
    common = dict(
        permute_seeds=seeds,
        permutation_chunk_size=2,
        reg_lambda=1.0, n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64, tol=1e-6, max_iter=50,
    )

    inmem = _fit_batched_cv_permutations(X, y, pm, **common)
    assert inmem is not None

    spill_dir = tmp_path / "shards"
    spill_dir.mkdir()
    out = _fit_batched_cv_permutations(X, y, pm, spill_dir=spill_dir, **common)
    assert out is None

    shards = sorted(spill_dir.glob("*.parquet"))
    assert len(shards) > 0, "spill mode should produce at least one shard"
    streamed = pl.read_parquet(shards)

    # Row counts and schemas match.
    assert streamed.height == inmem.height
    assert streamed.columns == inmem.columns

    # Sort on a key that uniquely identifies each row in both frames, then
    # compare. (problem_meta has no `subject`/`phoneme_pair` cols here, so
    # `electrode_idx` alone identifies the problem.)
    sort_keys = ["permutation_idx", "electrode_idx", "fold"]
    a = inmem.sort(sort_keys)
    b = streamed.sort(sort_keys)
    np.testing.assert_array_equal(
        a["test_roc_auc"].to_numpy(), b["test_roc_auc"].to_numpy()
    )
    for col in ["fold", "permutation_idx", "n_train", "n_test", "electrode_idx"]:
        np.testing.assert_array_equal(
            a[col].to_numpy(), b[col].to_numpy(),
            err_msg=f"mismatch in column {col}",
        )


# ---------------------------------------------------------------------------
# Permutation-test entry points
# ---------------------------------------------------------------------------


def test_run_acoustic_searchlight_permutations_null_is_chance():
    """Null AUC on the signal electrode averages to ~0.5 across permutations."""
    signal_e = 0
    epochs = _make_fake_epochs(
        n_electrodes=3,
        signal_electrodes=(signal_e,),
        signal_window=(15, 30),
        signal_amp=3.0,
        seed=20,
    )
    windows = make_windows(0, epochs.times.shape[0], window_size=10, stride=5)

    real_scores, _, _ = run_acoustic_searchlight(
        epochs, subject="SYN",
        electrode_idxs=[0, 1, 2],
        windows=windows,
        reg_lambda=1.0, target="categorical_acoustic_cue",
        n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )
    real_peak = (
        real_scores.group_by(["electrode_idx", "smin", "smax"])
        .agg(pl.col("test_roc_auc").mean())
        .group_by("electrode_idx")
        .agg(pl.col("test_roc_auc").max().alias("peak_auc"))
    )
    signal_real = real_peak.filter(pl.col("electrode_idx") == signal_e)["peak_auc"].item()
    assert signal_real > 0.9, f"sanity: real signal AUC too low ({signal_real:.3f})"

    null_scores = run_acoustic_searchlight_permutations(
        epochs, subject="SYN",
        electrode_idxs=[0, 1, 2],
        windows=windows,
        reg_lambda=1.0,
        permute_seeds=list(range(30)),
        permutation_chunk_size=5,
        target="categorical_acoustic_cue",
        n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )

    assert null_scores["permutation_idx"].n_unique() == 30

    # Mean fold-mean AUC on the signal electrode across permutations — near 0.5.
    signal_null = (
        null_scores.filter(pl.col("electrode_idx") == signal_e)
        .group_by(["permutation_idx", "smin", "smax"])
        .agg(pl.col("test_roc_auc").mean())
    )
    mean_null = signal_null["test_roc_auc"].mean()
    assert 0.40 < mean_null < 0.60, (
        f"null AUC should center near 0.5, got {mean_null:.3f}"
    )
    assert mean_null < signal_real - 0.3, (
        f"null AUC {mean_null:.3f} too close to real signal AUC {signal_real:.3f}"
    )


def test_run_behavior_with_control_permutations_has_paired_full_and_baseline():
    """behavior-with-control null returns both model='full' and 'baseline' per permutation."""
    epochs = _make_fake_epochs(n_electrodes=2, seed=21)
    windows = make_windows(0, epochs.times.shape[0], window_size=10, stride=5)

    null_scores = run_behavior_with_control_permutations(
        epochs, subject="SYN",
        electrode_idxs=[0, 1],
        windows=windows,
        reg_lambda=1.0,
        permute_seeds=[1, 2, 3],
        permutation_chunk_size=2,
        n_folds=5, cv_random_state=0,
        device="cpu", dtype=torch.float64,
        tol=1e-6, max_iter=50,
    )

    assert set(null_scores["model"].unique().to_list()) == {"full", "baseline"}
    assert set(null_scores["permutation_idx"].unique().to_list()) == {0, 1, 2}

    # Baseline rows: one "electrode" slot (-1) per (phoneme_pair, word_end)
    base = null_scores.filter(pl.col("model") == "baseline")
    assert (base["electrode_idx"] == -1).all()
    # Baseline must be refit per permutation — perms 0 and 2 should give
    # different AUCs on at least one (fold, word_end) row.
    base0 = base.filter(pl.col("permutation_idx") == 0).sort(["word_end", "fold"])
    base2 = base.filter(pl.col("permutation_idx") == 2).sort(["word_end", "fold"])
    assert not np.array_equal(
        base0["test_roc_auc"].to_numpy(),
        base2["test_roc_auc"].to_numpy(),
    ), "baseline rows do not vary across permutations — is baseline actually refitting?"


# ---------------------------------------------------------------------------
# audit_class_balance
# ---------------------------------------------------------------------------


def test_audit_class_balance_flags_low_and_ok():
    """Healthy fixture → all 'ok'; tiny fixture → at least one 'low'/'skipped'.

    Determinism: re-running with the same seed must give bit-identical output
    (guards against accidentally reseeding the splitter from global RNG state).
    """
    epochs = _make_fake_epochs(n_trials_per_step=40)
    audit = audit_class_balance(epochs, subject="EC_fake", n_folds=5)

    assert set(audit["decoder"].unique().to_list()) == {
        "acoustic",
        "behavior_full", "behavior_hga_only",
        "ganong_full", "ganong_hga_only",
    }
    assert (audit["status"] == "ok").all(), audit
    assert audit["min_test_minority_per_fold"].is_not_null().all()
    assert audit["min_train_minority_per_fold"].is_not_null().all()
    # Acoustic and ganong rows pool across completions (word_end null);
    # behavior rows split by word_end.
    pooled = audit.filter(pl.col("decoder").is_in(
        ["acoustic", "ganong_full", "ganong_hga_only"]
    ))
    behavior = audit.filter(pl.col("decoder").is_in(
        ["behavior_full", "behavior_hga_only"]
    ))
    assert pooled["word_end"].is_null().all()
    assert behavior["word_end"].is_not_null().all()

    tiny = _make_fake_epochs(n_trials_per_step=5, seed=11)
    audit_tiny = audit_class_balance(tiny, subject="EC_small", n_folds=5)
    # At n_trials_per_step=5, behavior (pp, word_end) cells have ~5 trials
    # total → min_class is below 2*n_folds, should flag as "low" or "skipped".
    assert (audit_tiny["status"] != "ok").any(), audit_tiny

    audit2 = audit_class_balance(epochs, subject="EC_fake", n_folds=5)
    assert audit.equals(audit2)
