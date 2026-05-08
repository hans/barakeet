"""
Per-decoder aggregation of (real_scores, null_scores) into per-(site,
window[, perm]) statistics. Single source of truth for site_keys and
flavor lists used by both summarize notebooks (which compute the
production peak-tests + p-values) and adaptive-K null notebooks
(which compute the stage-1 gate before deciding which sites to
escalate to stage 2).

Each ``aggregate_<decoder>(real_scores, null_scores, ...)`` returns
the (real_agg, null_agg) DataFrames the downstream peak-test machinery
expects: one row per ``SITE_KEYS_<DECODER> + ['smin', 'smax']`` for
real, plus ``permutation_idx`` for null. ``fold_tstat_aggregate`` is
applied uniformly so every aggregator produces ``fold_mean``,
``fold_std``, ``n_folds``, ``t_stat`` columns; paired decoders
(behavior_with_control, ganong_with_control) additionally retain
fold-mean ``full_roc_auc`` / ``baseline_roc_auc`` columns on the
real_agg for downstream diagnostics.

The ``apply_tfce`` flag on ``FlavorSpec`` is consumed by
``src.models.causal6_adaptive_null.stage1_gate`` to decide whether to
run ``tfce_1d_per_site`` before computing the gate's pointwise_p; the
summarize notebooks orchestrate their own TFCE calls independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.models.significance import fold_tstat_aggregate


@dataclass(frozen=True)
class FlavorSpec:
    """One statistical flavor produced by a decoder's summarize step.

    Attributes:
        stat_col: Column name in ``real_agg`` / ``null_agg``. Either
            ``"fold_mean"`` or ``"t_stat"``.
        apply_tfce: Whether the gate should TFCE-enhance this column
            before computing pointwise_p. Mirrors whether the
            decoder's summarize step produces a TFCE flavor.
        tfce_threshold: TFCE integration floor when ``apply_tfce=True``.
            Default 0.0 matches centered statistics (``t_stat``,
            with-control ``diff``). Use 0.5 for raw AUC (HGA-only
            fold_mean) so chance-level windows don't contribute. Must
            match what the decoder's summarize step passes to
            ``tfce_1d_per_site`` so the gate's enhanced statistic is
            comparable to the production peak-test's.
    """

    stat_col: str
    apply_tfce: bool
    tfce_threshold: float = 0.0


# Single source of truth for site_keys (imported by both summarize
# notebooks and *_null.py adaptive-K notebooks):
SITE_KEYS_ACOUSTIC: list[str] = ["subject", "electrode_idx", "phoneme_pair"]
SITE_KEYS_BEHAVIOR_WITH_CONTROL: list[str] = [
    "subject", "electrode_idx", "phoneme_pair", "word_end",
]
SITE_KEYS_BEHAVIOR_HGA_ONLY: list[str] = [
    "subject", "electrode_idx", "phoneme_pair", "word_end",
]
SITE_KEYS_GANONG_WITH_CONTROL: list[str] = [
    "subject", "electrode_idx", "phoneme_pair",
]
SITE_KEYS_GANONG_HGA_ONLY: list[str] = [
    "subject", "electrode_idx", "phoneme_pair",
]

# Flavor lists matching what each decoder's summarize step actually
# produces (peak_summary*.parquet variants).
FLAVORS_ACOUSTIC: list[FlavorSpec] = [
    FlavorSpec("fold_mean", apply_tfce=False),
    FlavorSpec("t_stat", apply_tfce=False),
]
# notebooks/causal6/acoustic_decoding_peaks.py:28 documents why TFCE is
# omitted: peak-search window count is too narrow for cluster credit
# to matter.

FLAVORS_BEHAVIOR_WITH_CONTROL: list[FlavorSpec] = [
    FlavorSpec("fold_mean", apply_tfce=False),
    FlavorSpec("t_stat", apply_tfce=False),
    # diff is centered at 0 → tfce_threshold=0.0 (default).
    FlavorSpec("fold_mean", apply_tfce=True),
    FlavorSpec("t_stat", apply_tfce=True),
]
FLAVORS_BEHAVIOR_HGA_ONLY: list[FlavorSpec] = [
    FlavorSpec("fold_mean", apply_tfce=False),
    FlavorSpec("t_stat", apply_tfce=False),
    # fold_mean is raw AUC (centered ~0.5) → integrate above chance.
    FlavorSpec("fold_mean", apply_tfce=True, tfce_threshold=0.5),
    FlavorSpec("t_stat", apply_tfce=True),
]

FLAVORS_GANONG_WITH_CONTROL: list[FlavorSpec] = [
    FlavorSpec("fold_mean", apply_tfce=False),
]
FLAVORS_GANONG_HGA_ONLY: list[FlavorSpec] = [
    FlavorSpec("fold_mean", apply_tfce=False),
]


def _filter_acoustic(
    df: pl.DataFrame,
    *,
    target: str,
    peak_search_smin: int,
    peak_search_smax: int,
) -> pl.DataFrame:
    return df.filter(
        (pl.col("target") == target)
        & (pl.col("smin") >= peak_search_smin)
        & (pl.col("smax") <= peak_search_smax)
    )


def _reconstruct_null_agg(
    preagg_null: pl.DataFrame,
    real_agg: pl.DataFrame,
    window_keys: list[str],
) -> pl.DataFrame:
    """Convert pre-aggregated null (fold_mean_diff schema) to the
    (fold_mean, fold_std, n_folds, t_stat) schema that fold_tstat_aggregate
    produces.

    fold_mean is reconstructed as fold_mean_diff + real_agg.fold_mean,
    recovering the absolute permutation fold-mean. fold_std == fold_std_diff
    (std of paired diffs). t_stat is the pre-computed paired t-stat.
    """
    real_mean = real_agg.select(window_keys + ["fold_mean"])
    return (
        preagg_null.join(real_mean, on=window_keys, how="left")
        .with_columns(
            (pl.col("fold_mean_diff") + pl.col("fold_mean")).alias("_fm")
        )
        .drop("fold_mean_diff", "fold_mean")
        .rename({"_fm": "fold_mean", "fold_std_diff": "fold_std"})
        .select(
            window_keys + ["permutation_idx", "fold_mean", "fold_std", "n_folds", "t_stat"]
        )
    )


def aggregate_acoustic(
    real_scores: pl.DataFrame,
    null_scores: pl.DataFrame,
    *,
    target: str,
    peak_search_smin: int,
    peak_search_smax: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Acoustic decoder aggregation: filter to target + window range,
    then collapse folds into (fold_mean, fold_std, t_stat) per site×window.

    If null_scores is pre-aggregated (fold_mean_diff schema from
    preagg_acoustic_null), the fold-aggregation is skipped and fold_mean is
    reconstructed from the paired differences. The null must be pre-filtered
    by target before preagg — target is not present in the preagg schema.
    """
    site_keys = SITE_KEYS_ACOUSTIC
    window_keys = site_keys + ["smin", "smax"]

    real_agg = fold_tstat_aggregate(
        _filter_acoustic(
            real_scores,
            target=target,
            peak_search_smin=peak_search_smin,
            peak_search_smax=peak_search_smax,
        ),
        group_keys=window_keys,
        stat_col="test_roc_auc",
        center=0.5,
    )
    if "fold_mean_diff" in null_scores.columns:
        null_filtered = null_scores.filter(
            (pl.col("smin") >= peak_search_smin) & (pl.col("smax") <= peak_search_smax)
        )
        null_agg = _reconstruct_null_agg(null_filtered, real_agg, window_keys)
    else:
        null_agg = fold_tstat_aggregate(
            _filter_acoustic(
                null_scores,
                target=target,
                peak_search_smin=peak_search_smin,
                peak_search_smax=peak_search_smax,
            ),
            group_keys=window_keys + ["permutation_idx"],
            stat_col="test_roc_auc",
            center=0.5,
        )
    return real_agg, null_agg


def _pair_full_baseline(
    scores: pl.DataFrame,
    *,
    full_keys: list[str],
) -> pl.DataFrame:
    """Pair model='full' and model='baseline' rows by full_keys and
    derive ``diff = full_roc_auc - baseline_roc_auc``.

    full_keys identifies a (subject, phoneme_pair, [word_end], fold[,
    permutation_idx]) tuple. The baseline rows are joined on those keys
    AFTER dropping electrode_idx / smin / smax, since the baseline model
    has no electrode-window dependence.
    """
    full = scores.filter(pl.col("model") == "full").drop("model")
    base = (
        scores.filter(pl.col("model") == "baseline")
        .drop("model", "electrode_idx", "smin", "smax")
        .rename({"test_roc_auc": "baseline_roc_auc"})
    )
    return (
        full.rename({"test_roc_auc": "full_roc_auc"})
        .join(base, on=full_keys, how="left")
        .with_columns(
            (pl.col("full_roc_auc") - pl.col("baseline_roc_auc")).alias("diff")
        )
    )


def _filter_behavior_window(
    df: pl.DataFrame,
    *,
    offset_samples: dict[str, int],
    peak_search_smin: int,
    peak_search_smax: int,
) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("word_end")
            .replace_strict(offset_samples, default=None)
            .alias("_smax_limit")
        )
        .filter(
            (pl.col("smin") >= peak_search_smin)
            & (pl.col("smax") <= pl.col("_smax_limit"))
            & (pl.col("smax") <= peak_search_smax)
        )
        .drop("_smax_limit")
    )


def _behavior_offset_samples(
    epoch_tmin: float,
    epoch_sfreq: float,
    behav_peak_post_offset_s: float,
) -> dict[str, int]:
    from src.stimuli import OFFSET_DICT

    return {
        we: int(
            (offset_s - epoch_tmin) * epoch_sfreq
            + behav_peak_post_offset_s * epoch_sfreq
        )
        for we, offset_s in OFFSET_DICT.items()
    }


def behavior_null_smax(
    epoch_tmin: float,
    epoch_sfreq: float,
    behav_peak_post_offset_s: float,
    global_peak_search_smax: int,
) -> int:
    """Tightest smax needed for behavior null permutations.

    Equals min(global_peak_search_smax, max per-word-end offset cap),
    matching the filter applied in aggregate_behavior_{hga_only,with_control}.
    """
    offset_samples = _behavior_offset_samples(epoch_tmin, epoch_sfreq, behav_peak_post_offset_s)
    return min(global_peak_search_smax, max(offset_samples.values()))


def aggregate_behavior_with_control(
    real_scores: pl.DataFrame,
    null_scores: pl.DataFrame,
    *,
    epoch_tmin: float,
    epoch_sfreq: float,
    behav_peak_post_offset_s: float,
    peak_search_smin: int,
    peak_search_smax: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Behavior decoder (with baseline control). Pair full + baseline by
    (subject, phoneme_pair, word_end, fold[, perm]) and compute
    ``diff = full - baseline``, then collapse folds into (fold_mean,
    fold_std, t_stat) on diff. real_agg additionally keeps fold-mean
    full_roc_auc / baseline_roc_auc as diagnostic columns.

    If null_scores is pre-aggregated (fold_mean_diff schema from
    preagg_behavior_with_control_null), the pairing and fold-aggregation are
    skipped and fold_mean is reconstructed from the paired differences.
    """
    site_keys = SITE_KEYS_BEHAVIOR_WITH_CONTROL
    window_keys = site_keys + ["smin", "smax"]
    offset_samples = _behavior_offset_samples(
        epoch_tmin, epoch_sfreq, behav_peak_post_offset_s,
    )

    def _pair_and_filter(
        scores: pl.DataFrame, extra_keys: list[str] | None = None,
    ) -> pl.DataFrame:
        full_keys = (
            ["subject", "phoneme_pair", "word_end", "fold"] + (extra_keys or [])
        )
        paired = _pair_full_baseline(scores, full_keys=full_keys)
        return _filter_behavior_window(
            paired,
            offset_samples=offset_samples,
            peak_search_smin=peak_search_smin,
            peak_search_smax=peak_search_smax,
        )

    real_paired = _pair_and_filter(real_scores)
    real_agg_diff = fold_tstat_aggregate(
        real_paired, group_keys=window_keys, stat_col="diff", center=0.0,
    )

    if "fold_mean_diff" in null_scores.columns:
        null_filtered = _filter_behavior_window(
            null_scores,
            offset_samples=offset_samples,
            peak_search_smin=peak_search_smin,
            peak_search_smax=peak_search_smax,
        )
        null_agg_diff = _reconstruct_null_agg(null_filtered, real_agg_diff, window_keys)
    else:
        null_paired = _pair_and_filter(null_scores, extra_keys=["permutation_idx"])
        null_agg_diff = fold_tstat_aggregate(
            null_paired,
            group_keys=window_keys + ["permutation_idx"],
            stat_col="diff",
            center=0.0,
        )

    # Diagnostic full/baseline fold-means joined onto real_agg.
    real_diag = real_paired.group_by(window_keys).agg(
        pl.col("full_roc_auc").mean().alias("full_roc_auc"),
        pl.col("baseline_roc_auc").mean().alias("baseline_roc_auc"),
    )
    real_agg = real_agg_diff.join(real_diag, on=window_keys, how="left")

    return real_agg, null_agg_diff


def aggregate_behavior_hga_only(
    real_scores: pl.DataFrame,
    null_scores: pl.DataFrame,
    *,
    epoch_tmin: float,
    epoch_sfreq: float,
    behav_peak_post_offset_s: float,
    peak_search_smin: int,
    peak_search_smax: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Behavior decoder (HGA-only, no control). Filter windows by
    word_end-relative cap, then collapse folds into (fold_mean, fold_std,
    t_stat) on test_roc_auc (centered at 0.5).

    If null_scores is pre-aggregated (fold_mean_diff schema from
    preagg_behavior_hga_only_null), the fold-aggregation is skipped and
    fold_mean is reconstructed from the paired differences.
    """
    site_keys = SITE_KEYS_BEHAVIOR_HGA_ONLY
    window_keys = site_keys + ["smin", "smax"]
    offset_samples = _behavior_offset_samples(
        epoch_tmin, epoch_sfreq, behav_peak_post_offset_s,
    )

    real_filtered = _filter_behavior_window(
        real_scores,
        offset_samples=offset_samples,
        peak_search_smin=peak_search_smin,
        peak_search_smax=peak_search_smax,
    )
    real_agg = fold_tstat_aggregate(
        real_filtered, group_keys=window_keys,
        stat_col="test_roc_auc", center=0.5,
    )

    null_filtered = _filter_behavior_window(
        null_scores,
        offset_samples=offset_samples,
        peak_search_smin=peak_search_smin,
        peak_search_smax=peak_search_smax,
    )
    if "fold_mean_diff" in null_scores.columns:
        null_agg = _reconstruct_null_agg(null_filtered, real_agg, window_keys)
    else:
        null_agg = fold_tstat_aggregate(
            null_filtered,
            group_keys=window_keys + ["permutation_idx"],
            stat_col="test_roc_auc", center=0.5,
        )
    return real_agg, null_agg


def _ganong_pod_samples(epoch_tmin: float, epoch_sfreq: float) -> dict[str, int]:
    from src.stimuli import POD_dict

    return {
        pp: int((pod_s - epoch_tmin) * epoch_sfreq)
        for pp, pod_s in POD_dict.items()
    }


def _filter_ganong_window(
    df: pl.DataFrame,
    *,
    pod_samples: dict[str, int],
    peak_search_smax: int,
) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("phoneme_pair")
            .replace_strict(pod_samples, default=None)
            .alias("_smin_floor")
        )
        .filter(
            (pl.col("smin") >= pl.col("_smin_floor"))
            & (pl.col("smax") <= peak_search_smax)
        )
        .drop("_smin_floor")
    )


def aggregate_ganong_with_control(
    real_scores: pl.DataFrame,
    null_scores: pl.DataFrame,
    *,
    epoch_tmin: float,
    epoch_sfreq: float,
    peak_search_smax: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ganong decoder (with baseline control). Pair full + baseline by
    (subject, phoneme_pair, fold[, perm]) and compute
    ``diff = full - baseline``, then collapse folds into (fold_mean,
    fold_std, t_stat) on diff. real_agg additionally keeps fold-mean
    full_roc_auc / baseline_roc_auc as diagnostic columns.

    Note ``word_end`` is NOT in site_keys: trials are pooled across
    completions in the ganong refit.

    If null_scores is pre-aggregated (fold_mean_diff schema from
    preagg_ganong_with_control_null), the pairing and fold-aggregation are
    skipped and fold_mean is reconstructed from the paired differences.
    """
    site_keys = SITE_KEYS_GANONG_WITH_CONTROL
    window_keys = site_keys + ["smin", "smax"]
    pod_samples = _ganong_pod_samples(epoch_tmin, epoch_sfreq)

    def _pair_and_filter(
        scores: pl.DataFrame, extra_keys: list[str] | None = None,
    ) -> pl.DataFrame:
        full_keys = ["subject", "phoneme_pair", "fold"] + (extra_keys or [])
        paired = _pair_full_baseline(scores, full_keys=full_keys)
        return _filter_ganong_window(
            paired, pod_samples=pod_samples, peak_search_smax=peak_search_smax,
        )

    real_paired = _pair_and_filter(real_scores)
    real_agg_diff = fold_tstat_aggregate(
        real_paired, group_keys=window_keys, stat_col="diff", center=0.0,
    )

    if "fold_mean_diff" in null_scores.columns:
        null_filtered = _filter_ganong_window(
            null_scores, pod_samples=pod_samples, peak_search_smax=peak_search_smax,
        )
        null_agg_diff = _reconstruct_null_agg(null_filtered, real_agg_diff, window_keys)
    else:
        null_paired = _pair_and_filter(null_scores, extra_keys=["permutation_idx"])
        null_agg_diff = fold_tstat_aggregate(
            null_paired,
            group_keys=window_keys + ["permutation_idx"],
            stat_col="diff",
            center=0.0,
        )

    real_diag = real_paired.group_by(window_keys).agg(
        pl.col("full_roc_auc").mean().alias("full_roc_auc"),
        pl.col("baseline_roc_auc").mean().alias("baseline_roc_auc"),
    )
    real_agg = real_agg_diff.join(real_diag, on=window_keys, how="left")

    return real_agg, null_agg_diff


def aggregate_ganong_hga_only(
    real_scores: pl.DataFrame,
    null_scores: pl.DataFrame,
    *,
    epoch_tmin: float,
    epoch_sfreq: float,
    peak_search_smax: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ganong decoder (HGA-only, no control). Filter windows by
    per-phoneme-pair POD floor + global smax cap, then collapse folds
    into (fold_mean, fold_std, t_stat) on test_roc_auc (centered at 0.5).

    If null_scores is pre-aggregated (fold_mean_diff schema from
    preagg_ganong_hga_only_null), the fold-aggregation is skipped and
    fold_mean is reconstructed from the paired differences.
    """
    site_keys = SITE_KEYS_GANONG_HGA_ONLY
    window_keys = site_keys + ["smin", "smax"]
    pod_samples = _ganong_pod_samples(epoch_tmin, epoch_sfreq)

    real_filtered = _filter_ganong_window(
        real_scores, pod_samples=pod_samples, peak_search_smax=peak_search_smax,
    )
    real_agg = fold_tstat_aggregate(
        real_filtered, group_keys=window_keys,
        stat_col="test_roc_auc", center=0.5,
    )

    null_filtered = _filter_ganong_window(
        null_scores, pod_samples=pod_samples, peak_search_smax=peak_search_smax,
    )
    if "fold_mean_diff" in null_scores.columns:
        null_agg = _reconstruct_null_agg(null_filtered, real_agg, window_keys)
    else:
        null_agg = fold_tstat_aggregate(
            null_filtered,
            group_keys=window_keys + ["permutation_idx"],
            stat_col="test_roc_auc", center=0.5,
        )
    return real_agg, null_agg


# ---------------------------------------------------------------------------
# Preagg helpers: compute paired fold differences at null-generation time
# ---------------------------------------------------------------------------


def _preagg_hga_only_null(
    raw_null: pl.DataFrame,
    real_scores: pl.DataFrame,
    *,
    site_keys: list[str],
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Join on (site_keys + smin + smax + fold), compute diff_k = perm_k −
    real_k per fold, and aggregate over folds.

    Output schema: site_keys + [smin, smax, permutation_idx,
    fold_mean_diff, fold_std_diff, n_folds, t_stat].
    fold_mean_diff ≥ 0 means the permutation outperformed real on that perm.
    """
    window_keys = site_keys + ["smin", "smax"]
    join_keys = window_keys + ["fold"]
    real_fold = (
        real_scores.select(join_keys + ["test_roc_auc"])
        .rename({"test_roc_auc": "_real"})
    )
    with_diff = (
        raw_null.join(real_fold, on=join_keys, how="left")
        .with_columns((pl.col("test_roc_auc") - pl.col("_real")).alias("_diff"))
    )
    agg = (
        with_diff.group_by(window_keys + ["permutation_idx"])
        .agg(
            pl.col("_diff").mean().alias("fold_mean_diff"),
            pl.col("_diff").std().alias("fold_std_diff"),
            pl.col("_diff").len().alias("n_folds"),
        )
    )
    sem = (
        pl.max_horizontal(pl.col("fold_std_diff").fill_nan(std_floor), pl.lit(std_floor))
        / pl.col("n_folds").cast(pl.Float64).sqrt()
    )
    return agg.with_columns((pl.col("fold_mean_diff") / sem).alias("t_stat"))


def _preagg_with_control_null(
    raw_null: pl.DataFrame,
    real_scores: pl.DataFrame,
    *,
    site_keys: list[str],
    pair_base_keys: list[str],
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Pair full−baseline within each (fold, perm), then compute
    paired_diff_k = perm_diff_k − real_diff_k and aggregate over folds.

    pair_base_keys: non-fold, non-perm keys for _pair_full_baseline.
      behavior_with_control → ["subject", "phoneme_pair", "word_end"]
      ganong_with_control   → ["subject", "phoneme_pair"]

    Output schema: site_keys + [smin, smax, permutation_idx,
    fold_mean_diff, fold_std_diff, n_folds, t_stat].
    """
    window_keys = site_keys + ["smin", "smax"]
    real_diff = _pair_full_baseline(
        real_scores, full_keys=pair_base_keys + ["fold"]
    )
    null_diff = _pair_full_baseline(
        raw_null, full_keys=pair_base_keys + ["fold", "permutation_idx"]
    )

    real_for_join = (
        real_diff.select(window_keys + ["fold", "diff"])
        .rename({"diff": "_real_diff"})
    )
    with_paired = (
        null_diff.join(real_for_join, on=window_keys + ["fold"], how="left")
        .with_columns(
            (pl.col("diff") - pl.col("_real_diff")).alias("_paired_diff")
        )
    )
    agg = (
        with_paired.group_by(window_keys + ["permutation_idx"])
        .agg(
            pl.col("_paired_diff").mean().alias("fold_mean_diff"),
            pl.col("_paired_diff").std().alias("fold_std_diff"),
            pl.col("_paired_diff").len().alias("n_folds"),
        )
    )
    sem = (
        pl.max_horizontal(pl.col("fold_std_diff").fill_nan(std_floor), pl.lit(std_floor))
        / pl.col("n_folds").cast(pl.Float64).sqrt()
    )
    return agg.with_columns((pl.col("fold_mean_diff") / sem).alias("t_stat"))


def preagg_acoustic_null(
    raw_null: pl.DataFrame,
    real_scores: pl.DataFrame,
    *,
    site_keys: list[str] = SITE_KEYS_ACOUSTIC,
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Pre-aggregate acoustic null scores into paired fold differences.

    Both raw_null and real_scores must be pre-filtered to the same target
    (target is not in site_keys and would contaminate the join).
    """
    return _preagg_hga_only_null(
        raw_null, real_scores, site_keys=site_keys, std_floor=std_floor,
    )


def preagg_behavior_hga_only_null(
    raw_null: pl.DataFrame,
    real_scores: pl.DataFrame,
    *,
    site_keys: list[str] = SITE_KEYS_BEHAVIOR_HGA_ONLY,
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Pre-aggregate behavior HGA-only null scores into paired fold differences."""
    return _preagg_hga_only_null(
        raw_null, real_scores, site_keys=site_keys, std_floor=std_floor,
    )


def preagg_behavior_with_control_null(
    raw_null: pl.DataFrame,
    real_scores: pl.DataFrame,
    *,
    site_keys: list[str] = SITE_KEYS_BEHAVIOR_WITH_CONTROL,
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Pre-aggregate behavior with-control null scores into paired fold differences."""
    return _preagg_with_control_null(
        raw_null, real_scores,
        site_keys=site_keys,
        pair_base_keys=["subject", "phoneme_pair", "word_end"],
        std_floor=std_floor,
    )


def preagg_ganong_hga_only_null(
    raw_null: pl.DataFrame,
    real_scores: pl.DataFrame,
    *,
    site_keys: list[str] = SITE_KEYS_GANONG_HGA_ONLY,
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Pre-aggregate ganong HGA-only null scores into paired fold differences."""
    return _preagg_hga_only_null(
        raw_null, real_scores, site_keys=site_keys, std_floor=std_floor,
    )


def preagg_ganong_with_control_null(
    raw_null: pl.DataFrame,
    real_scores: pl.DataFrame,
    *,
    site_keys: list[str] = SITE_KEYS_GANONG_WITH_CONTROL,
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Pre-aggregate ganong with-control null scores into paired fold differences."""
    return _preagg_with_control_null(
        raw_null, real_scores,
        site_keys=site_keys,
        pair_base_keys=["subject", "phoneme_pair"],
        std_floor=std_floor,
    )
