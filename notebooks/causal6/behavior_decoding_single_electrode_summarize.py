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
# causal6 summarize: behavior decoding with control, null-standardized peak.
#
# Emits four flavors per subject, all with the same peak_summary.parquet
# schema (so downstream consumers can swap which one they read):
#   * foldmean_maxstat — statistic = fold-mean(diff), max-stat correction (v1 contract)
#   * tstat_maxstat    — statistic = t-stat(diff, center=0), max-stat correction
#   * foldmean_tfce    — fold-mean(diff) enhanced by 1D TFCE along windows, max-stat
#   * tstat_tfce       — t-stat(diff) enhanced by 1D TFCE, max-stat
#
# Inputs:
#   scores.parquet         — real per-fold AUC for `model ∈ {full, baseline}`
#   predictions.parquet    — real per-trial held-out predictions
#   null_scores.parquet    — permutation null with both models refit per perm
#
# Outputs:
#   peak_summary.parquet                — foldmean_maxstat (v1 contract)
#   peak_summary_tstat_maxstat.parquet
#   peak_summary_foldmean_tfce.parquet
#   peak_summary_tstat_tfce.parquet
#   peak_predictions.parquet    — trial-level predictions at the foldmean_maxstat
#                                  peak window (v1 contract; other flavors don't
#                                  yet have trial-level consumers).
#   window_mean_scores.parquet  — fold-mean full/baseline/diff per (site, window).

# %%
import os
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import polars as pl

from src.models.causal6_aggregates import (
    SITE_KEYS_BEHAVIOR_WITH_CONTROL as site_keys,
    aggregate_behavior_with_control,
)
from src.models.significance import (
    null_standardized_peak_test,
    tfce_1d_per_site,
)

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet"
predictions_path = f"outputs/causal6/behavior_decoding_single_electrode/{subject}/predictions.parquet"
null_scores_path = f"outputs/causal6/behavior_decoding_single_electrode_null/{subject}/null_scores.parquet"
outdir = "."

epoch_tmin = -0.4
epoch_sfreq = 100
behav_peak_post_offset_s = 0.2
peak_search_smin = 0
peak_search_smax = 290

# %%
window_keys = site_keys + ["smin", "smax"]

# %%
predictions = pl.read_parquet(predictions_path)

real_agg_diff, null_agg_diff = aggregate_behavior_with_control(
    pl.read_parquet(scores_path),
    pl.read_parquet(null_scores_path),
    epoch_tmin=epoch_tmin,
    epoch_sfreq=epoch_sfreq,
    behav_peak_post_offset_s=behav_peak_post_offset_s,
    peak_search_smin=peak_search_smin,
    peak_search_smax=peak_search_smax,
)

# v1 diagnostic: fold-mean full / baseline / diff per (site, window).
real_window_mean = real_agg_diff.select(
    window_keys + ["full_roc_auc", "baseline_roc_auc", "fold_mean"]
).rename({"fold_mean": "diff"})

# %% [markdown]
# ## Four flavors of null-standardized peak test
#
# Each flavor shares the same peak_summary.parquet schema. The only moving
# parts are (a) which column of the aggregate feeds the test
# (``fold_mean`` vs ``t_stat``) and (b) whether TFCE is applied first.

# %%
def _run_maxstat(real_stat: pl.DataFrame, null_stat: pl.DataFrame,
                 stat_col: str, rename_to: str) -> pl.DataFrame:
    real_in = real_stat.select(window_keys + [stat_col]).rename({stat_col: "statistic"})
    null_in = null_stat.select(window_keys + ["permutation_idx", stat_col]).rename(
        {stat_col: "statistic"}
    )
    peaks, _ = null_standardized_peak_test(
        real_in, null_in,
        site_keys=site_keys, window_keys=["smin", "smax"], stat_col="statistic",
    )
    return peaks.rename({
        "peak_smin": "smin", "peak_smax": "smax", "real_statistic": rename_to,
    })


def _run_tfce_maxstat(real_stat: pl.DataFrame, null_stat: pl.DataFrame,
                      stat_col: str, rename_to: str) -> pl.DataFrame:
    """TFCE-enhance per (site[, perm]) along windows, then max-stat."""
    real_in = real_stat.select(window_keys + [stat_col]).rename({stat_col: "statistic"})
    null_in = null_stat.select(window_keys + ["permutation_idx", stat_col]).rename(
        {stat_col: "statistic"}
    )
    real_enh = tfce_1d_per_site(
        real_in, site_keys=site_keys, window_keys=["smin", "smax"], stat_col="statistic",
    )
    null_enh = tfce_1d_per_site(
        null_in, site_keys=site_keys, window_keys=["smin", "smax"],
        perm_key="permutation_idx", stat_col="statistic",
    )
    peaks, _ = null_standardized_peak_test(
        real_enh, null_enh,
        site_keys=site_keys, window_keys=["smin", "smax"], stat_col="statistic",
    )
    return peaks.rename({
        "peak_smin": "smin", "peak_smax": "smax", "real_statistic": rename_to,
    })


# (1) foldmean_maxstat — v1 contract. Re-join full/baseline at the peak window.
peak_summary_foldmean_maxstat = _run_maxstat(
    real_agg_diff, null_agg_diff, stat_col="fold_mean", rename_to="diff",
).join(
    real_window_mean.select(site_keys + ["smin", "smax", "full_roc_auc", "baseline_roc_auc"]),
    on=site_keys + ["smin", "smax"], how="left",
)
print(
    f"[foldmean_maxstat] {peak_summary_foldmean_maxstat.height} sites: "
    f"{(peak_summary_foldmean_maxstat['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# (2) tstat_maxstat
peak_summary_tstat_maxstat = _run_maxstat(
    real_agg_diff, null_agg_diff, stat_col="t_stat", rename_to="t_stat",
)
print(
    f"[tstat_maxstat]    {peak_summary_tstat_maxstat.height} sites: "
    f"{(peak_summary_tstat_maxstat['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# (3) foldmean_tfce
peak_summary_foldmean_tfce = _run_tfce_maxstat(
    real_agg_diff, null_agg_diff, stat_col="fold_mean", rename_to="diff_tfce",
)
print(
    f"[foldmean_tfce]    {peak_summary_foldmean_tfce.height} sites: "
    f"{(peak_summary_foldmean_tfce['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# (4) tstat_tfce
peak_summary_tstat_tfce = _run_tfce_maxstat(
    real_agg_diff, null_agg_diff, stat_col="t_stat", rename_to="t_stat_tfce",
)
print(
    f"[tstat_tfce]       {peak_summary_tstat_tfce.height} sites: "
    f"{(peak_summary_tstat_tfce['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# %%
# peak_predictions is derived from the v1 (foldmean_maxstat) peak windows,
# so existing consumers of trial-level predictions continue to work unchanged.
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
