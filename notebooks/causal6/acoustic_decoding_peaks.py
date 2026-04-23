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
# causal6: per-subject acoustic-decoding peak-finding with null-standardized
# significance.
#
# Inputs:
#   scores.parquet         — real fold-wise test AUC per (site, window)
#   null_scores.parquet    — permutation-null fold-wise AUC per (site, window, perm)
#
# Picks peaks by argmin of the pointwise permutation p-value (i.e. max
# standardized extremeness), not raw fold-mean AUC — see
# src/models/significance.py for rationale.
#
# Outputs:
#   phon_peaks.parquet                — one row per site with peak window
#                                        + raw + standardized stats + p-value.
#                                        Columns kept backward-compatible with
#                                        prepare_neurometrics / A_neurometrics
#                                        (smin, smax, test_roc_auc preserved),
#                                        new columns (pointwise_p, T_obs,
#                                        p_value, n_permutations, null_q*)
#                                        added alongside.
#   phon_roc_auc_searchlight.parquet  — fold-mean AUC per (site, window) for
#                                        diagnostic plots (unchanged).

# %%
from pathlib import Path

import polars as pl

from src.models.significance import null_standardized_peak_test

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet"
null_scores_path = f"outputs/causal6/acoustic_decoding_null/{subject}/null_scores.parquet"
outdir = "."

target = "categorical_acoustic_cue"
peak_search_smin = 0
peak_search_smax = 290

# %%
site_keys = ["subject", "electrode_idx", "phoneme_pair"]
window_keys = site_keys + ["smin", "smax"]


def _window_filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("smin") >= peak_search_smin) & (pl.col("smax") <= peak_search_smax)
    )


# %%
real_scores = (
    pl.read_parquet(scores_path)
    .filter(pl.col("target") == target)
    .pipe(_window_filter)
)
null_scores = (
    pl.read_parquet(null_scores_path)
    .filter(pl.col("target") == target)
    .pipe(_window_filter)
)

# %% [markdown]
# Aggregate to fold-mean AUC per (site, window) for real, and per
# (site, window, permutation) for null.

# %%
real_fold_mean = real_scores.group_by(window_keys).agg(
    pl.col("test_roc_auc").mean().alias("test_roc_auc")
)

null_fold_mean = null_scores.group_by(window_keys + ["permutation_idx"]).agg(
    pl.col("test_roc_auc").mean().alias("test_roc_auc")
)

# %% [markdown]
# Null-standardized peak test.

# %%
peak_summary_std, window_stats_std = null_standardized_peak_test(
    real_fold_mean, null_fold_mean,
    site_keys=site_keys,
    window_keys=["smin", "smax"],
    stat_col="test_roc_auc",
)

# Rename peak_smin/peak_smax → smin/smax for backward-compatible consumers;
# real_statistic → test_roc_auc (the fold-mean AUC at the peak window).
phon_peaks = (
    peak_summary_std
    .rename({
        "peak_smin": "smin",
        "peak_smax": "smax",
        "real_statistic": "test_roc_auc",
    })
)
print(f"{phon_peaks.height} sites: {(phon_peaks['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)")

# Diagnostic searchlight: per-(site, window) fold-mean AUC, same as the old rule.
phon_roc_auc_searchlight = real_fold_mean

# %%
outdir = Path(outdir)
phon_peaks.write_parquet(outdir / "phon_peaks.parquet")
phon_roc_auc_searchlight.write_parquet(outdir / "phon_roc_auc_searchlight.parquet")
print(
    f"Wrote phon_peaks.parquet ({phon_peaks.height} rows), "
    f"phon_roc_auc_searchlight.parquet ({phon_roc_auc_searchlight.height} rows) to {outdir}"
)
