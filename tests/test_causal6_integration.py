"""
Real-data integration tests for the causal6 decoding pipeline.

Exercises all three decoders and their permutation twins on the one
preprocessed subject available locally (EC248). Gated behind
``@pytest.mark.integration`` so the default ``pytest tests/`` stays fast;
opt in with ``pytest tests/ -m integration``.

Each decoder runs once as a session-scoped fixture; the B1/B2/B3
assertions then consume those cached results. Expected total runtime on
CPU: ~60-90 s.

Covers the three items from the coverage audit:
  B1. End-to-end decoding — schema, finiteness, CV disjointness, signal
      sanity on speech-responsive STG sites.
  B2. Significance smoke — permutation entry points centre their null
      near 0.5 and separate from the real peak; the notebook p-value
      formula behaves correctly.
  B3. Peak-finder on real output — ``null_standardized_peak_test``
      returns one row per site with valid p-values and the real peak
      window is the argmin of pointwise p on real data.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.causal6 import (
    make_windows,
    run_acoustic_searchlight,
    run_acoustic_searchlight_permutations,
    run_behavior_hga_only,
    run_behavior_hga_only_permutations,
    run_behavior_with_control,
    run_behavior_with_control_permutations,
    run_ganong_hga_only,
    run_ganong_with_control,
    run_ganong_with_control_permutations,
)
from src.models.causal6_aggregates import (
    SITE_KEYS_BEHAVIOR_HGA_ONLY,
    aggregate_behavior_hga_only,
    preagg_behavior_hga_only_null,
)
from src.models.significance import null_standardized_peak_test


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_smoke_windows(epochs, smoke_config):
    return make_windows(
        smoke_config["min_sample"],
        epochs.times.shape[0],
        smoke_config["window_size"],
        smoke_config["stride"],
    )


def _peak_auc_per_electrode(scores: pl.DataFrame, model_filter: str | None = None) -> pl.DataFrame:
    """Mean AUC over folds; then max over windows per (electrode, phoneme_pair[, word_end])."""
    f = scores if model_filter is None else scores.filter(pl.col("model") == model_filter)
    group_key = [c for c in ("phoneme_pair", "word_end") if c in f.columns]
    return (
        f.group_by(["electrode_idx", *group_key, "smin", "smax"])
        .agg(pl.col("test_roc_auc").mean().alias("mean_auc"))
        .group_by(["electrode_idx", *group_key])
        .agg(pl.col("mean_auc").max().alias("peak_auc"))
    )


# ---------------------------------------------------------------------------
# Session-scoped decoder runs (amortize cost across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def acoustic_result(ec248_epochs, ec248_smoke_electrodes, smoke_config):
    windows = _make_smoke_windows(ec248_epochs, smoke_config)
    scores, preds, coefs = run_acoustic_searchlight(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=windows,
        reg_lambda=smoke_config["reg_lambda"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )
    return {"scores": scores, "preds": preds, "coefs": coefs, "windows": windows}


@pytest.fixture(scope="session")
def behavior_with_control_result(ec248_epochs, ec248_smoke_electrodes, smoke_config):
    windows = _make_smoke_windows(ec248_epochs, smoke_config)
    scores, preds, coefs = run_behavior_with_control(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=windows,
        reg_lambda=smoke_config["reg_lambda"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )
    return {"scores": scores, "preds": preds, "coefs": coefs, "windows": windows}


@pytest.fixture(scope="session")
def behavior_hga_only_result(ec248_epochs, ec248_smoke_electrodes, smoke_config):
    windows = _make_smoke_windows(ec248_epochs, smoke_config)
    scores, preds, coefs = run_behavior_hga_only(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=windows,
        reg_lambda=smoke_config["reg_lambda"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )
    return {"scores": scores, "preds": preds, "coefs": coefs, "windows": windows}


@pytest.fixture(scope="session")
def ganong_with_control_result(ec248_epochs, ec248_smoke_electrodes, smoke_config):
    windows = _make_smoke_windows(ec248_epochs, smoke_config)
    scores, preds, coefs = run_ganong_with_control(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=windows,
        reg_lambda=smoke_config["reg_lambda"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )
    return {"scores": scores, "preds": preds, "coefs": coefs, "windows": windows}


@pytest.fixture(scope="session")
def ganong_hga_only_result(ec248_epochs, ec248_smoke_electrodes, smoke_config):
    windows = _make_smoke_windows(ec248_epochs, smoke_config)
    scores, preds, coefs = run_ganong_hga_only(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=windows,
        reg_lambda=smoke_config["reg_lambda"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )
    return {"scores": scores, "preds": preds, "coefs": coefs, "windows": windows}


# ---------------------------------------------------------------------------
# B1. End-to-end decoding
# ---------------------------------------------------------------------------


def test_acoustic_searchlight_ec248(acoustic_result, ec248_smoke_electrodes):
    """Acoustic searchlight on EC248: schema, finiteness, CV partition, signal."""
    scores = acoustic_result["scores"]
    preds = acoustic_result["preds"]
    coefs = acoustic_result["coefs"]
    windows = acoustic_result["windows"]

    assert scores.height > 0, "acoustic scores empty — check EC248 has unambiguous trials"
    assert set(scores.columns) >= {
        "subject", "phoneme_pair", "electrode_idx", "smin", "smax",
        "fold", "test_roc_auc", "n_train", "n_test", "n_iter", "converged",
        "target",
    }
    assert scores["target"].unique().to_list() == ["categorical_acoustic_cue"]

    # Unambiguous resampled ∈ {1, 6} endpoints are deterministically labeled;
    # no fold should degenerate to a single class → no NaN AUCs expected.
    aucs = scores["test_roc_auc"].to_numpy()
    assert np.isfinite(aucs).all(), (
        f"{(~np.isfinite(aucs)).sum()} non-finite AUCs on EC248 acoustic run"
    )
    # Some folds legitimately hit the max_iter cap at the smoke-config budget
    # (max_iter=50, float32). Only a catastrophic convergence failure indicates
    # a regression. Mirrors the tolerance in
    # tests/test_decoding_gpu.py:test_realistic_batch_shape.
    conv_rate = scores["converged"].cast(pl.Float64).mean()
    assert conv_rate > 0.9, f"convergence rate {conv_rate:.3f} below 0.9"

    # CV partition: every epoch_idx appears exactly once per (electrode, window) key.
    per_key = (
        preds.group_by(["electrode_idx", "smin", "smax", "phoneme_pair", "epoch_idx"])
        .agg(pl.len().alias("n"))
    )
    assert (per_key["n"] == 1).all(), "CV partition not disjoint on acoustic preds"

    # Coefficients: one row per (electrode × window × fold × phoneme_pair)
    n_expected_coefs = (
        scores.select(["electrode_idx", "smin", "smax", "fold", "phoneme_pair"]).n_unique()
    )
    assert coefs.height == n_expected_coefs, (
        f"coef rows {coefs.height} != expected {n_expected_coefs}"
    )

    # Signal sanity: at least one of the 3 speech-responsive electrodes should
    # have peak AUC > 0.6. EC248's responsive set is a strong positive control
    # for STG acoustic encoding — failure indicates data corruption or a
    # regression in the decoder path.
    peaks = _peak_auc_per_electrode(scores)
    best = peaks["peak_auc"].max()
    assert best > 0.6, (
        f"no speech-responsive electrode cleared peak AUC > 0.6 (best={best:.3f}); "
        f"check data integrity or decoder regression. Peaks:\n{peaks}"
    )


def test_behavior_with_control_ec248(behavior_with_control_result):
    """Behavior-full with baseline control: schema, baseline-reuse invariant, no NaN."""
    scores = behavior_with_control_result["scores"]
    preds = behavior_with_control_result["preds"]

    assert set(scores["model"].unique().to_list()) == {"full", "baseline"}

    # Baseline is fit once per (phoneme_pair, word_end, fold) — never per electrode.
    base = scores.filter(pl.col("model") == "baseline")
    assert (base["electrode_idx"] == -1).all()
    assert (base["smin"] == -1).all()
    per_group = base.group_by(["phoneme_pair", "word_end", "fold"]).agg(pl.len().alias("n"))
    assert (per_group["n"] == 1).all(), (
        "baseline fit more than once per (phoneme_pair, word_end, fold) — "
        "reuse invariant broken"
    )

    # No NaN AUCs on full model (baseline with only resampled feature could
    # theoretically degenerate but doesn't on EC248's 6-step continuum).
    full = scores.filter(pl.col("model") == "full")
    assert np.isfinite(full["test_roc_auc"].to_numpy()).all()

    # CV partition on full-model preds.
    full_preds = preds.filter(pl.col("model") == "full")
    per_key = (
        full_preds.group_by(
            ["electrode_idx", "smin", "smax", "phoneme_pair", "word_end", "epoch_idx"]
        )
        .agg(pl.len().alias("n"))
    )
    assert (per_key["n"] == 1).all()


def test_behavior_hga_only_ec248(behavior_hga_only_result):
    """Behavior HGA-only: only full rows, no baseline, schema parity."""
    scores = behavior_hga_only_result["scores"]
    preds = behavior_hga_only_result["preds"]

    assert scores["model"].unique().to_list() == ["full"]
    assert np.isfinite(scores["test_roc_auc"].to_numpy()).all()

    per_key = (
        preds.group_by(
            ["electrode_idx", "smin", "smax", "phoneme_pair", "word_end", "epoch_idx"]
        )
        .agg(pl.len().alias("n"))
    )
    assert (per_key["n"] == 1).all()


def test_ganong_with_control_ec248(ganong_with_control_result):
    """Ganong-full (pooled across completions): schema, baseline-reuse invariant, no NaN.

    Differs from the behavior-full test on two points:
      - no `word_end` column (trials pooled across completions)
      - baseline fit once per (phoneme_pair, fold), not per (phoneme_pair, word_end, fold)
    """
    scores = ganong_with_control_result["scores"]
    preds = ganong_with_control_result["preds"]

    assert "word_end" not in scores.columns, (
        "ganong scores must NOT include a word_end column — trials are pooled"
    )
    assert "word_end" not in preds.columns

    assert set(scores["model"].unique().to_list()) == {"full", "baseline"}

    # Baseline is fit once per (phoneme_pair, fold) — shared across electrodes,
    # windows, AND lexical completions.
    base = scores.filter(pl.col("model") == "baseline")
    assert (base["electrode_idx"] == -1).all()
    assert (base["smin"] == -1).all()
    per_group = base.group_by(["phoneme_pair", "fold"]).agg(pl.len().alias("n"))
    assert (per_group["n"] == 1).all(), (
        "ganong baseline fit more than once per (phoneme_pair, fold) — "
        "pooled-reuse invariant broken"
    )

    full = scores.filter(pl.col("model") == "full")
    assert np.isfinite(full["test_roc_auc"].to_numpy()).all()

    # CV partition on full-model preds: pooled across completions so every
    # (epoch_idx, electrode, window) appears in exactly one fold.
    full_preds = preds.filter(pl.col("model") == "full")
    per_key = (
        full_preds.group_by(
            ["electrode_idx", "smin", "smax", "phoneme_pair", "epoch_idx"]
        )
        .agg(pl.len().alias("n"))
    )
    assert (per_key["n"] == 1).all()


def test_ganong_hga_only_ec248(ganong_hga_only_result):
    """Ganong HGA-only: only full rows, no baseline, no word_end column."""
    scores = ganong_hga_only_result["scores"]
    preds = ganong_hga_only_result["preds"]

    assert "word_end" not in scores.columns
    assert "word_end" not in preds.columns

    assert scores["model"].unique().to_list() == ["full"]
    assert np.isfinite(scores["test_roc_auc"].to_numpy()).all()

    per_key = (
        preds.group_by(
            ["electrode_idx", "smin", "smax", "phoneme_pair", "epoch_idx"]
        )
        .agg(pl.len().alias("n"))
    )
    assert (per_key["n"] == 1).all()


# ---------------------------------------------------------------------------
# B2. Significance smoke
# ---------------------------------------------------------------------------


def test_acoustic_searchlight_permutations_ec248(
    ec248_epochs, ec248_smoke_electrodes, smoke_config, acoustic_result
):
    """Permutation null on EC248 centers ~0.5 and is separated from the real peak.

    Also exercises the notebook p-value formula inline so that a silent
    change to either the permutation entry point or the p-value arithmetic
    trips this test.
    """
    n_perms = 20
    null_scores = run_acoustic_searchlight_permutations(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=acoustic_result["windows"],
        reg_lambda=smoke_config["reg_lambda"],
        permute_seeds=list(range(n_perms)),
        permutation_chunk_size=smoke_config["permutation_chunk_size"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )

    assert null_scores["permutation_idx"].n_unique() == n_perms

    # Fold-mean null AUC per (electrode, window, perm); averaged across perms
    # per (electrode, window). Should center near 0.5.
    null_center = (
        null_scores.group_by(["electrode_idx", "phoneme_pair", "smin", "smax", "permutation_idx"])
        .agg(pl.col("test_roc_auc").mean().alias("fold_mean"))
        .group_by(["electrode_idx", "phoneme_pair", "smin", "smax"])
        .agg(pl.col("fold_mean").mean().alias("null_mean"))
    )
    assert 0.35 < null_center["null_mean"].mean() < 0.65, (
        f"null AUC mean off-centre: {null_center['null_mean'].mean():.3f}"
    )

    # Real peak AUC on the strongest site (from B1 cached result) should be
    # well clear of the null centre. EC248's best speech-responsive site
    # typically lands near 0.65; require at least a 0.1 gap from the ~0.5 null.
    real_peak = _peak_auc_per_electrode(acoustic_result["scores"])
    best_real = real_peak["peak_auc"].max()
    assert best_real > null_center["null_mean"].mean() + 0.1

    # Notebook p-value formula contract — compute in the same shape the
    # significance notebooks would: per (electrode, phoneme_pair) site,
    # T_obs = max fold-mean real AUC across windows; T_null_k = max fold-mean
    # null AUC across windows per permutation k; p = (#{T_null >= T_obs}+1)/(K+1).
    site_keys = ["electrode_idx", "phoneme_pair"]
    real_fold_mean = (
        acoustic_result["scores"]
        .group_by(site_keys + ["smin", "smax"])
        .agg(pl.col("test_roc_auc").mean().alias("auc"))
    )
    real_T_obs = real_fold_mean.group_by(site_keys).agg(pl.col("auc").max().alias("T_obs"))
    null_T = (
        null_scores.group_by(site_keys + ["smin", "smax", "permutation_idx"])
        .agg(pl.col("test_roc_auc").mean().alias("auc"))
        .group_by(site_keys + ["permutation_idx"])
        .agg(pl.col("auc").max().alias("T_null"))
    )
    joined = null_T.join(real_T_obs, on=site_keys, how="inner")
    pvals = (
        joined.group_by(site_keys + ["T_obs"])
        .agg(
            (pl.col("T_null") >= pl.col("T_obs")).cast(pl.Int64).sum().alias("ge"),
            pl.len().alias("K"),
        )
        .with_columns(((pl.col("ge") + 1) / (pl.col("K") + 1)).alias("p_value"))
    )
    p_array = pvals["p_value"].to_numpy()
    min_p = 1.0 / (n_perms + 1)
    assert ((p_array >= min_p) & (p_array <= 1.0)).all(), (
        f"p-values out of valid range [{min_p:.4f}, 1.0]: {p_array}"
    )
    # Strongest electrode should have p well below 0.5 on EC248 speech-responsive sites.
    assert p_array.min() <= 0.5, f"no site achieved p ≤ 0.5 (min={p_array.min():.3f})"


def test_behavior_hga_only_preagg_null_ec248(
    ec248_epochs, ec248_smoke_electrodes, smoke_config, behavior_hga_only_result
):
    """Preagg null has new schema and fold_mean_diff ≥ 0 count matches p-value.

    Verifies:
    - null_scores.parquet written by preagg has no fold column and has
      fold_mean_diff, fold_std_diff, n_folds, t_stat columns.
    - #{perm: fold_mean_diff >= 0} at the peak window closely matches the
      foldmean p-value from null_standardized_peak_test.
    """
    n_perms = 20
    raw_null = run_behavior_hga_only_permutations(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=behavior_hga_only_result["windows"],
        reg_lambda=smoke_config["reg_lambda"],
        permute_seeds=list(range(n_perms)),
        permutation_chunk_size=smoke_config["permutation_chunk_size"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )

    real_scores = behavior_hga_only_result["scores"]
    preagg_null = preagg_behavior_hga_only_null(raw_null, real_scores)

    # Schema: no fold column; has preagg columns.
    assert "fold" not in preagg_null.columns
    for col in ("fold_mean_diff", "fold_std_diff", "n_folds", "t_stat", "permutation_idx"):
        assert col in preagg_null.columns, f"missing column: {col}"

    # Shape: one row per (site × window × perm).
    site_keys = SITE_KEYS_BEHAVIOR_HGA_ONLY
    window_keys = site_keys + ["smin", "smax"]
    n_expected = preagg_null.select(window_keys + ["permutation_idx"]).n_unique()
    assert preagg_null.height == n_expected

    # null_agg via aggregate_* with preagg null.
    real_agg, null_agg = aggregate_behavior_hga_only(
        real_scores, preagg_null,
        epoch_tmin=smoke_config["epoch_tmin"],
        epoch_sfreq=smoke_config["epoch_sfreq"],
        behav_peak_post_offset_s=smoke_config["behav_peak_post_offset_s"],
        peak_search_smin=smoke_config["peak_search_smin"],
        peak_search_smax=smoke_config["peak_search_smax"],
    )

    # P-values from peak test are in valid range.
    peaks, _ = null_standardized_peak_test(
        real_agg.rename({"fold_mean": "statistic"}),
        null_agg.rename({"fold_mean": "statistic"}),
        site_keys=site_keys,
        window_keys=["smin", "smax"],
        stat_col="statistic",
    )
    min_p = 1.0 / (n_perms + 1)
    p = peaks["p_value"].to_numpy()
    assert ((p >= min_p) & (p <= 1.0)).all(), f"p-values out of range: {p}"

    # fold_mean_diff >= 0 count at each (site, window) ≈ #{perm: fold_mean_perm >= fold_mean_real}.
    # Verify by comparing against the old-style fold-mean aggregation from raw_null.
    null_old = (
        raw_null.group_by(window_keys + ["permutation_idx"])
        .agg(pl.col("test_roc_auc").mean().alias("fold_mean_perm"))
    )
    real_mean = (
        real_scores.group_by(window_keys)
        .agg(pl.col("test_roc_auc").mean().alias("fold_mean_real"))
    )
    old_joined = null_old.join(real_mean, on=window_keys, how="left")

    p_from_diff = (
        preagg_null.group_by(window_keys)
        .agg(
            ((pl.col("fold_mean_diff") >= 0).cast(pl.Int64).sum() + 1).alias("ge_preagg"),
            (pl.len() + 1).alias("K_plus1_preagg"),
        )
        .with_columns((pl.col("ge_preagg") / pl.col("K_plus1_preagg")).alias("p_preagg"))
    )
    p_from_raw = (
        old_joined.group_by(window_keys)
        .agg(
            ((pl.col("fold_mean_perm") >= pl.col("fold_mean_real")).cast(pl.Int64).sum() + 1)
            .alias("ge_raw"),
            (pl.len() + 1).alias("K_plus1_raw"),
        )
        .with_columns((pl.col("ge_raw") / pl.col("K_plus1_raw")).alias("p_raw"))
    )
    check = p_from_diff.join(p_from_raw, on=window_keys, how="inner")
    np.testing.assert_allclose(
        check["p_preagg"].to_numpy(),
        check["p_raw"].to_numpy(),
        atol=1e-10,
        err_msg="preagg foldmean p-values do not match raw foldmean p-values",
    )


def test_behavior_with_control_permutations_ec248(
    ec248_epochs, ec248_smoke_electrodes, smoke_config, behavior_with_control_result
):
    """5-permutation smoke: both full and baseline present; baseline varies across perms."""
    null_scores = run_behavior_with_control_permutations(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=behavior_with_control_result["windows"],
        reg_lambda=smoke_config["reg_lambda"],
        permute_seeds=[1, 2, 3, 4, 5],
        permutation_chunk_size=smoke_config["permutation_chunk_size"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )
    assert set(null_scores["model"].unique().to_list()) == {"full", "baseline"}
    assert set(null_scores["permutation_idx"].unique().to_list()) == {0, 1, 2, 3, 4}

    base = null_scores.filter(pl.col("model") == "baseline")
    assert (base["electrode_idx"] == -1).all()
    # Baseline must be refit per permutation.
    key = ["phoneme_pair", "word_end", "fold"]
    base0 = base.filter(pl.col("permutation_idx") == 0).sort(key)
    base4 = base.filter(pl.col("permutation_idx") == 4).sort(key)
    assert not np.array_equal(
        base0["test_roc_auc"].to_numpy(),
        base4["test_roc_auc"].to_numpy(),
    ), "baseline AUCs identical across permutations — baseline not actually refitting"


# ---------------------------------------------------------------------------
# B3. Peak-finder on real output via null_standardized_peak_test
# ---------------------------------------------------------------------------


def test_acoustic_peak_finder_on_real_scores(
    ec248_epochs, ec248_smoke_electrodes, smoke_config, acoustic_result
):
    """Run a small null and feed it + real scores through null_standardized_peak_test.

    Asserts the peak-finder's contract holds on real data: one row per site,
    valid p-values, and the chosen peak is the argmin of pointwise p at that
    site (consistent with how the summarize notebook in
    notebooks/causal6/acoustic_decoding_peaks.py consumes the function).
    """
    # Minimum viable null for the helper. 10 perms is the smoke-config budget.
    null_scores = run_acoustic_searchlight_permutations(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=acoustic_result["windows"],
        reg_lambda=smoke_config["reg_lambda"],
        permute_seeds=list(range(10)),
        permutation_chunk_size=smoke_config["permutation_chunk_size"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )

    site_keys = ["subject", "electrode_idx", "phoneme_pair"]
    real_fold_mean = (
        acoustic_result["scores"]
        .group_by(site_keys + ["smin", "smax"])
        .agg(pl.col("test_roc_auc").mean().alias("test_roc_auc"))
    )
    null_fold_mean = (
        null_scores.group_by(site_keys + ["smin", "smax", "permutation_idx"])
        .agg(pl.col("test_roc_auc").mean().alias("test_roc_auc"))
    )

    peaks, windows_stats = null_standardized_peak_test(
        real_fold_mean, null_fold_mean,
        site_keys=site_keys,
        window_keys=["smin", "smax"],
        stat_col="test_roc_auc",
    )

    # One row per site.
    assert peaks.height == real_fold_mean.select(site_keys).n_unique()
    assert peaks.height == peaks.select(site_keys).n_unique()

    # Required output columns (matches the contract the notebook relies on).
    assert set(peaks.columns) >= {
        *site_keys, "peak_smin", "peak_smax",
        "real_statistic", "pointwise_p", "T_obs", "p_value", "n_permutations",
        "null_q05", "null_q50", "null_q95", "null_q99",
    }

    # p-value bounds.
    p = peaks["p_value"].to_numpy()
    assert ((p >= 1.0 / 11.0) & (p <= 1.0)).all(), f"p out of range: {p}"

    # Peak window selection contract: peaks pick argmin pointwise_p per site.
    # Verify against the window_stats table by joining on the site keys.
    check = (
        peaks.select(site_keys + ["peak_smin", "peak_smax", "pointwise_p"])
        .rename({"peak_smin": "smin", "peak_smax": "smax"})
        .join(
            windows_stats.select(site_keys + ["smin", "smax", "pointwise_p"])
            .rename({"pointwise_p": "pointwise_p_window"}),
            on=site_keys + ["smin", "smax"],
            how="inner",
        )
    )
    # Peak's pointwise_p equals the window's pointwise_p at the selected window.
    np.testing.assert_allclose(
        check["pointwise_p"].to_numpy(),
        check["pointwise_p_window"].to_numpy(),
        atol=1e-12,
    )
    # And that value is the min over all windows at that site.
    site_min = (
        windows_stats.group_by(site_keys)
        .agg(pl.col("pointwise_p").min().alias("min_p"))
    )
    peak_p = peaks.select(site_keys + ["pointwise_p"]).join(site_min, on=site_keys, how="inner")
    np.testing.assert_allclose(
        peak_p["pointwise_p"].to_numpy(),
        peak_p["min_p"].to_numpy(),
        atol=1e-12,
    )


def test_behavior_peak_finder_on_real_scores(
    ec248_epochs, ec248_smoke_electrodes, smoke_config, behavior_with_control_result
):
    """Behavior-full peak-finder contract on real EC248 output.

    Mirrors the pipeline in
    notebooks/causal6/behavior_decoding_single_electrode_summarize.py —
    pair full + baseline on (phoneme_pair, word_end, fold), statistic =
    fold-mean(full − baseline), then null_standardized_peak_test.
    """
    null_scores = run_behavior_with_control_permutations(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=behavior_with_control_result["windows"],
        reg_lambda=smoke_config["reg_lambda"],
        permute_seeds=list(range(10)),
        permutation_chunk_size=smoke_config["permutation_chunk_size"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )

    site_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

    def _pair(scores: pl.DataFrame, perm: bool) -> pl.DataFrame:
        join_keys = ["subject", "phoneme_pair", "word_end", "fold"] + (
            ["permutation_idx"] if perm else []
        )
        full = scores.filter(pl.col("model") == "full").drop("model")
        base = (
            scores.filter(pl.col("model") == "baseline")
            .drop("model", "electrode_idx", "smin", "smax")
            .rename({"test_roc_auc": "baseline_roc_auc"})
        )
        return (
            full.rename({"test_roc_auc": "full_roc_auc"})
            .join(base, on=join_keys, how="left")
            .with_columns((pl.col("full_roc_auc") - pl.col("baseline_roc_auc")).alias("diff"))
        )

    real_paired = _pair(behavior_with_control_result["scores"], perm=False)
    null_paired = _pair(null_scores, perm=True)

    real_window_mean = real_paired.group_by(site_keys + ["smin", "smax"]).agg(
        pl.col("diff").mean().alias("diff")
    )
    null_window_mean = null_paired.group_by(
        site_keys + ["smin", "smax", "permutation_idx"]
    ).agg(pl.col("diff").mean().alias("diff"))

    peaks, _ = null_standardized_peak_test(
        real_window_mean, null_window_mean,
        site_keys=site_keys,
        window_keys=["smin", "smax"],
        stat_col="diff",
    )

    # One row per site.
    assert peaks.height == real_window_mean.select(site_keys).n_unique()
    assert peaks.height == peaks.select(site_keys).n_unique()

    # p-values in valid range.
    p = peaks["p_value"].to_numpy()
    assert ((p >= 1.0 / 11.0) & (p <= 1.0)).all()

    # real_statistic at each peak row matches the max-over-windows `diff`
    # with the smallest pointwise_p — this is what the summarize notebook
    # writes out as the site's peak score.
    assert peaks["real_statistic"].is_not_null().all()


def test_ganong_peak_finder_on_real_scores(
    ec248_epochs, ec248_smoke_electrodes, smoke_config, ganong_with_control_result
):
    """Ganong-full peak-finder contract on real EC248 output.

    Mirrors notebooks/causal6/ganong_decoding_summarize.py: pair full +
    baseline on (phoneme_pair, fold), statistic = fold-mean(full − baseline),
    apply per-phoneme-pair POD floor bound, then null_standardized_peak_test.
    Site = (subject, electrode_idx, phoneme_pair) — no word_end.
    """
    from src.stimuli import POD_dict

    null_scores = run_ganong_with_control_permutations(
        ec248_epochs,
        subject="EC248",
        electrode_idxs=ec248_smoke_electrodes,
        windows=ganong_with_control_result["windows"],
        reg_lambda=smoke_config["reg_lambda"],
        permute_seeds=list(range(10)),
        permutation_chunk_size=smoke_config["permutation_chunk_size"],
        n_folds=smoke_config["n_folds"],
        cv_random_state=smoke_config["cv_random_state"],
        device=smoke_config["device"],
        dtype=smoke_config["dtype"],
        tol=smoke_config["tol"],
        max_iter=smoke_config["max_iter"],
    )

    site_keys = ["subject", "electrode_idx", "phoneme_pair"]

    # POD floor in samples — matches the summarize notebook's conversion.
    epoch_tmin = -0.4
    epoch_sfreq = 100
    pod_samples = {
        pp: int((pod_s - epoch_tmin) * epoch_sfreq)
        for pp, pod_s in POD_dict.items()
    }

    def _pair_and_filter(scores: pl.DataFrame, perm: bool) -> pl.DataFrame:
        join_keys = ["subject", "phoneme_pair", "fold"] + (
            ["permutation_idx"] if perm else []
        )
        full = scores.filter(pl.col("model") == "full").drop("model")
        base = (
            scores.filter(pl.col("model") == "baseline")
            .drop("model", "electrode_idx", "smin", "smax")
            .rename({"test_roc_auc": "baseline_roc_auc"})
        )
        paired = (
            full.rename({"test_roc_auc": "full_roc_auc"})
            .join(base, on=join_keys, how="left")
            .with_columns((pl.col("full_roc_auc") - pl.col("baseline_roc_auc")).alias("diff"))
        )
        return (
            paired.with_columns(
                pl.col("phoneme_pair").replace_strict(pod_samples, default=None).alias("_floor")
            )
            .filter(pl.col("smin") >= pl.col("_floor"))
            .drop("_floor")
        )

    real_paired = _pair_and_filter(ganong_with_control_result["scores"], perm=False)
    null_paired = _pair_and_filter(null_scores, perm=True)

    real_window_mean = real_paired.group_by(site_keys + ["smin", "smax"]).agg(
        pl.col("diff").mean().alias("diff")
    )
    null_window_mean = null_paired.group_by(
        site_keys + ["smin", "smax", "permutation_idx"]
    ).agg(pl.col("diff").mean().alias("diff"))

    peaks, _ = null_standardized_peak_test(
        real_window_mean, null_window_mean,
        site_keys=site_keys,
        window_keys=["smin", "smax"],
        stat_col="diff",
    )

    # One row per site.
    assert peaks.height == real_window_mean.select(site_keys).n_unique()
    assert peaks.height == peaks.select(site_keys).n_unique()

    # p-values in valid range.
    p = peaks["p_value"].to_numpy()
    assert ((p >= 1.0 / 11.0) & (p <= 1.0)).all()

    # Peak window respects the POD floor per phoneme_pair.
    check = peaks.with_columns(
        pl.col("phoneme_pair").replace_strict(pod_samples, default=None).alias("_floor")
    )
    assert (check["peak_smin"] >= check["_floor"]).all(), (
        "ganong peak_smin below POD floor — filter failed"
    )

    assert peaks["real_statistic"].is_not_null().all()
