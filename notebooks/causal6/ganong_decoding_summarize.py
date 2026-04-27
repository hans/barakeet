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
# causal6 summarize: ganong decoding with control, null-standardized peak.
#
# Inputs:
#   scores.parquet         — real per-fold AUC for both `model='full'`
#                            and `model='baseline'`
#   predictions.parquet    — real per-trial held-out predictions
#   null_scores.parquet    — permutation null with both models refit per perm
#
# Paired statistic = fold-mean(full_roc_auc − baseline_roc_auc), computed per
# (site, window[, permutation]). Peak-finding uses null-standardized pointwise
# p (see src/models/significance.py). Site = (subject, electrode, phoneme_pair)
# — `word_end` is NOT a site key here because trials are pooled across
# completions.
#
# Peak-search bounds: per-phoneme-pair POD floor (lower; Ganong shift emerges
# only after the point of disambiguation) + global `peak_search_smax` (upper).
#
# Outputs:
#   peak_summary.parquet        — one row per site: peak window, fold-mean
#                                  full/baseline/diff at that window, +
#                                  pointwise_p / T_obs / p_value / n_permutations
#                                  / null_q{05,50,95,99}.
#   peak_predictions.parquet    — trial-level real predictions filtered to
#                                  the peak window per site.
#   window_mean_scores.parquet  — fold-mean full/baseline/diff per (site,
#                                  window); diagnostic.

# %%
import os
os.environ.setdefault("POLARS_MAX_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import polars as pl

from src.models.causal6_aggregates import (
    SITE_KEYS_GANONG_WITH_CONTROL as site_keys,
    aggregate_ganong_with_control,
)
from src.models.significance import null_standardized_peak_test

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/ganong_decoding_single_electrode/{subject}/scores.parquet"
predictions_path = f"outputs/causal6/ganong_decoding_single_electrode/{subject}/predictions.parquet"
null_scores_path = f"outputs/causal6/ganong_decoding_null/{subject}/null_scores.parquet"
outdir = "."

epoch_tmin = -0.4
epoch_sfreq = 100
peak_search_smax = 290

# %%
window_keys = site_keys + ["smin", "smax"]

# %%
predictions = pl.read_parquet(predictions_path)

real_agg, null_agg = aggregate_ganong_with_control(
    pl.read_parquet(scores_path),
    pl.read_parquet(null_scores_path),
    epoch_tmin=epoch_tmin,
    epoch_sfreq=epoch_sfreq,
    peak_search_smax=peak_search_smax,
)

# v1 diagnostic: fold-mean full / baseline / diff per (site, window).
real_window_mean = real_agg.select(
    window_keys + ["full_roc_auc", "baseline_roc_auc", "fold_mean"]
).rename({"fold_mean": "diff"})

# %% [markdown]
# ## Null-standardized peak test
#
# Statistic = fold-mean `diff` (full − baseline).

# %%
peak_summary_std, _window_stats_std = null_standardized_peak_test(
    real_agg.select(site_keys + ["smin", "smax", "fold_mean"]).rename({"fold_mean": "diff"}),
    null_agg.select(site_keys + ["smin", "smax", "permutation_idx", "fold_mean"]).rename({"fold_mean": "diff"}),
    site_keys=site_keys,
    window_keys=["smin", "smax"],
    stat_col="diff",
)

peak_summary = (
    peak_summary_std
    .rename({"peak_smin": "smin", "peak_smax": "smax", "real_statistic": "diff"})
    .join(
        real_window_mean.select(site_keys + ["smin", "smax", "full_roc_auc", "baseline_roc_auc"]),
        on=site_keys + ["smin", "smax"],
        how="left",
    )
)
print(
    f"{peak_summary.height} sites: "
    f"{(peak_summary['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# %%
peak_keys = peak_summary.select(site_keys + ["smin", "smax"])
peak_predictions = predictions.join(peak_keys, on=site_keys + ["smin", "smax"], how="inner")

# %%
outdir = Path(outdir)
peak_summary.write_parquet(outdir / "peak_summary.parquet")
peak_predictions.write_parquet(outdir / "peak_predictions.parquet")
real_window_mean.write_parquet(outdir / "window_mean_scores.parquet")
print(
    f"Wrote peak_summary.parquet ({peak_summary.height} rows), "
    f"peak_predictions.parquet ({peak_predictions.height} rows), "
    f"window_mean_scores.parquet ({real_window_mean.height} rows) to {outdir}"
)
