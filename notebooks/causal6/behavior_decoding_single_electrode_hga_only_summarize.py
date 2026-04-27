# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# causal6 summarize: behavior decoding HGA-only, null-standardized peak.
#
# Emits four flavors per subject, all with the same peak_summary.parquet
# schema:
#   * foldmean_maxstat — statistic = fold-mean AUC, max-stat correction (v1 contract)
#   * tstat_maxstat    — statistic = (fold_mean - 0.5) / (fold_std / sqrt(n)), max-stat
#   * foldmean_tfce    — fold-mean AUC enhanced by 1D TFCE along windows, max-stat
#   * tstat_tfce       — t-stat enhanced by 1D TFCE, max-stat
#
# Inputs:
#   scores.parquet         — real per-fold AUC (single model, no baseline)
#   predictions.parquet    — real per-trial held-out predictions
#   null_scores.parquet    — permutation null per (site, window, perm)

# %%
import os
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import polars as pl

from src.models.significance import (
    fold_tstat_aggregate,
    null_standardized_peak_test,
    tfce_1d_per_site,
)
from src.stimuli import OFFSET_DICT

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet"
predictions_path = f"outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet"
null_scores_path = f"outputs/causal6/behavior_decoding_single_electrode_hga_only_null/{subject}/null_scores.parquet"
outdir = "."

epoch_tmin = -0.4
epoch_sfreq = 100
behav_peak_post_offset_s = 0.2
peak_search_smin = 0
peak_search_smax = 290

# %%
site_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
window_keys = site_keys + ["smin", "smax"]

offset_samples = {
    we: int((offset_s - epoch_tmin) * epoch_sfreq + behav_peak_post_offset_s * epoch_sfreq)
    for we, offset_s in OFFSET_DICT.items()
}


def _window_filter(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("word_end").replace_strict(offset_samples, default=None).alias("_smax_limit")
        )
        .filter(
            (pl.col("smin") >= peak_search_smin)
            & (pl.col("smax") <= pl.col("_smax_limit"))
            & (pl.col("smax") <= peak_search_smax)
        )
        .drop("_smax_limit")
    )


# %%
real_scores = pl.read_parquet(scores_path).pipe(_window_filter)
predictions = pl.read_parquet(predictions_path)
null_scores = pl.read_parquet(null_scores_path).pipe(_window_filter)

# %% [markdown]
# ## Aggregate folds to (fold_mean, fold_std, t_stat) per (site, window[, perm])
#
# HGA-only uses raw AUC; centering = 0.5 (chance).

# %%
real_agg = fold_tstat_aggregate(
    real_scores, group_keys=window_keys, stat_col="test_roc_auc", center=0.5,
)
null_agg = fold_tstat_aggregate(
    null_scores, group_keys=window_keys + ["permutation_idx"],
    stat_col="test_roc_auc", center=0.5,
)

# v1-compat diagnostic
real_window_mean = real_agg.select(window_keys + ["fold_mean"]).rename(
    {"fold_mean": "test_roc_auc"}
)

# %% [markdown]
# ## Four flavors of null-standardized peak test

# %%
def _run_maxstat(stat_col: str, rename_to: str) -> pl.DataFrame:
    real_in = real_agg.select(window_keys + [stat_col]).rename({stat_col: "statistic"})
    null_in = null_agg.select(window_keys + ["permutation_idx", stat_col]).rename(
        {stat_col: "statistic"}
    )
    peaks, _ = null_standardized_peak_test(
        real_in, null_in,
        site_keys=site_keys, window_keys=["smin", "smax"], stat_col="statistic",
    )
    return peaks.rename({
        "peak_smin": "smin", "peak_smax": "smax", "real_statistic": rename_to,
    })


def _run_tfce_maxstat(stat_col: str, rename_to: str, tfce_threshold: float) -> pl.DataFrame:
    """Threshold is the TFCE integration floor: 0.5 for AUC so chance-level
    windows don't contribute; 0 for centered statistics like t_stat."""
    real_in = real_agg.select(window_keys + [stat_col]).rename({stat_col: "statistic"})
    null_in = null_agg.select(window_keys + ["permutation_idx", stat_col]).rename(
        {stat_col: "statistic"}
    )
    real_enh = tfce_1d_per_site(
        real_in, site_keys=site_keys, window_keys=["smin", "smax"],
        stat_col="statistic", threshold=tfce_threshold,
    )
    null_enh = tfce_1d_per_site(
        null_in, site_keys=site_keys, window_keys=["smin", "smax"],
        perm_key="permutation_idx", stat_col="statistic", threshold=tfce_threshold,
    )
    peaks, _ = null_standardized_peak_test(
        real_enh, null_enh,
        site_keys=site_keys, window_keys=["smin", "smax"], stat_col="statistic",
    )
    return peaks.rename({
        "peak_smin": "smin", "peak_smax": "smax", "real_statistic": rename_to,
    })


peak_summary_foldmean_maxstat = _run_maxstat("fold_mean", rename_to="test_roc_auc")
print(
    f"[foldmean_maxstat] {peak_summary_foldmean_maxstat.height} sites: "
    f"{(peak_summary_foldmean_maxstat['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

peak_summary_tstat_maxstat = _run_maxstat("t_stat", rename_to="t_stat")
print(
    f"[tstat_maxstat]    {peak_summary_tstat_maxstat.height} sites: "
    f"{(peak_summary_tstat_maxstat['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

peak_summary_foldmean_tfce = _run_tfce_maxstat(
    "fold_mean", rename_to="test_roc_auc_tfce", tfce_threshold=0.5,
)
print(
    f"[foldmean_tfce]    {peak_summary_foldmean_tfce.height} sites: "
    f"{(peak_summary_foldmean_tfce['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

peak_summary_tstat_tfce = _run_tfce_maxstat(
    "t_stat", rename_to="t_stat_tfce", tfce_threshold=0.0,
)
print(
    f"[tstat_tfce]       {peak_summary_tstat_tfce.height} sites: "
    f"{(peak_summary_tstat_tfce['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# %%
# peak_predictions derived from the v1 peak windows (foldmean_maxstat).
peak_keys = peak_summary_foldmean_maxstat.select(site_keys + ["smin", "smax"])
peak_predictions = predictions.join(peak_keys, on=site_keys + ["smin", "smax"], how="inner")

# %%
outdir = Path(outdir)
peak_summary_foldmean_maxstat.write_parquet(outdir / "peak_summary.parquet")
peak_summary_tstat_maxstat.write_parquet(outdir / "peak_summary_tstat_maxstat.parquet")
peak_summary_foldmean_tfce.write_parquet(outdir / "peak_summary_foldmean_tfce.parquet")
peak_summary_tstat_tfce.write_parquet(outdir / "peak_summary_tstat_tfce.parquet")
peak_predictions.write_parquet(outdir / "peak_predictions.parquet")
real_window_mean.write_parquet(outdir / "window_mean_scores.parquet")
print(
    f"Wrote peak_summary.parquet ({peak_summary_foldmean_maxstat.height} rows) "
    f"+ 3 extra flavors, peak_predictions.parquet ({peak_predictions.height} rows), "
    f"window_mean_scores.parquet ({real_window_mean.height} rows) to {outdir}"
)
