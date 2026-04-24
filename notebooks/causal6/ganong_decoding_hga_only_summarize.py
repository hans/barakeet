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
# causal6 summarize: ganong decoding HGA-only, null-standardized peak.
#
# Inputs:
#   scores.parquet         — real per-fold AUC (single model, no baseline)
#   predictions.parquet    — real per-trial held-out predictions
#   null_scores.parquet    — permutation null per (site, window, perm)
#
# Peak-finding uses null-standardized pointwise p of fold-mean AUC. Site =
# (subject, electrode, phoneme_pair) — `word_end` is NOT a site key because
# trials are pooled across completions.
#
# Peak-search bounds: per-phoneme-pair POD floor + global `peak_search_smax`.

# %%
from pathlib import Path

import polars as pl

from src.models.significance import null_standardized_peak_test
from src.stimuli import POD_dict

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/scores.parquet"
predictions_path = f"outputs/causal6/ganong_decoding_single_electrode_hga_only/{subject}/predictions.parquet"
null_scores_path = f"outputs/causal6/ganong_decoding_hga_only_null/{subject}/null_scores.parquet"
outdir = "."

epoch_tmin = -0.4
epoch_sfreq = 100
peak_search_smax = 290

# %%
site_keys = ["subject", "electrode_idx", "phoneme_pair"]
window_keys = site_keys + ["smin", "smax"]

pod_samples = {
    pp: int((pod_s - epoch_tmin) * epoch_sfreq)
    for pp, pod_s in POD_dict.items()
}


def _window_filter(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("phoneme_pair").replace_strict(pod_samples, default=None).alias("_smin_floor")
        )
        .filter(
            (pl.col("smin") >= pl.col("_smin_floor"))
            & (pl.col("smax") <= peak_search_smax)
        )
        .drop("_smin_floor")
    )


# %%
real_scores = pl.read_parquet(scores_path).pipe(_window_filter)
predictions = pl.read_parquet(predictions_path)
null_scores = pl.read_parquet(null_scores_path).pipe(_window_filter)

# %% [markdown]
# ## Aggregate to fold-mean per (site, window[, perm])

# %%
real_window_mean = real_scores.group_by(window_keys).agg(
    pl.col("test_roc_auc").mean().alias("test_roc_auc")
)

null_window_mean = null_scores.group_by(window_keys + ["permutation_idx"]).agg(
    pl.col("test_roc_auc").mean().alias("test_roc_auc")
)

# %% [markdown]
# ## Null-standardized peak test

# %%
peak_summary_std, _window_stats_std = null_standardized_peak_test(
    real_window_mean,
    null_window_mean,
    site_keys=site_keys,
    window_keys=["smin", "smax"],
    stat_col="test_roc_auc",
)

peak_summary = peak_summary_std.rename({
    "peak_smin": "smin",
    "peak_smax": "smax",
    "real_statistic": "test_roc_auc",
})
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
