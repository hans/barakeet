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
# Inputs:
#   scores.parquet         — real per-fold AUC for both `model='full'`
#                            and `model='baseline'`
#   predictions.parquet    — real per-trial held-out predictions
#   null_scores.parquet    — permutation null with both models refit per perm
#
# Paired statistic = fold-mean(full_roc_auc − baseline_roc_auc), computed per
# (site, window[, permutation]). Peak-finding uses null-standardized pointwise
# p (see src/models/significance.py).
#
# Outputs:
#   peak_summary.parquet        — one row per site: peak window, fold-mean
#                                  full/baseline/diff at that window, +
#                                  pointwise_p / T_obs / p_value / n_permutations
#                                  / null_q{05,50,95,99}.
#   peak_predictions.parquet    — trial-level real predictions filtered to
#                                  the peak window per site (unchanged contract).
#   window_mean_scores.parquet  — fold-mean full/baseline/diff per (site,
#                                  window); diagnostic, same schema as before.

# %%
from pathlib import Path

import polars as pl

from src.models.significance import null_standardized_peak_test
from src.stimuli import OFFSET_DICT

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
site_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
window_keys = site_keys + ["smin", "smax"]

offset_samples = {
    we: int((offset_s - epoch_tmin) * epoch_sfreq + behav_peak_post_offset_s * epoch_sfreq)
    for we, offset_s in OFFSET_DICT.items()
}


def _pair_and_filter(scores: pl.DataFrame, extra_keys: list[str] | None = None) -> pl.DataFrame:
    """Pair full + baseline on (phoneme_pair, word_end, fold[, perm]) and
    keep only windows inside the peak-search range."""
    extra_keys = extra_keys or []
    full_keys = ["subject", "phoneme_pair", "word_end", "fold"] + extra_keys

    full = scores.filter(pl.col("model") == "full").drop("model")
    base = (
        scores.filter(pl.col("model") == "baseline")
        .drop("model", "electrode_idx", "smin", "smax")
        .rename({"test_roc_auc": "baseline_roc_auc"})
    )
    paired = (
        full.rename({"test_roc_auc": "full_roc_auc"})
        .join(base, on=full_keys, how="left")
        .with_columns(
            (pl.col("full_roc_auc") - pl.col("baseline_roc_auc")).alias("diff")
        )
    )
    return (
        paired.with_columns(
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
real_scores = pl.read_parquet(scores_path)
predictions = pl.read_parquet(predictions_path)
null_scores = pl.read_parquet(null_scores_path)

real_paired = _pair_and_filter(real_scores)
null_paired = _pair_and_filter(null_scores, extra_keys=["permutation_idx"])

# %% [markdown]
# ## Aggregate to fold-mean per (site, window[, perm])

# %%
real_window_mean = real_paired.group_by(window_keys).agg(
    pl.col("full_roc_auc").mean().alias("full_roc_auc"),
    pl.col("baseline_roc_auc").mean().alias("baseline_roc_auc"),
    pl.col("diff").mean().alias("diff"),
)

null_window_mean = null_paired.group_by(window_keys + ["permutation_idx"]).agg(
    pl.col("diff").mean().alias("diff"),
)

# %% [markdown]
# ## Null-standardized peak test
#
# Statistic = fold-mean `diff` (full − baseline).

# %%
peak_summary_std, _window_stats_std = null_standardized_peak_test(
    real_window_mean.select(site_keys + ["smin", "smax", "diff"]),
    null_window_mean,
    site_keys=site_keys,
    window_keys=["smin", "smax"],
    stat_col="diff",
)

# Rename peak_smin/peak_smax → smin/smax and real_statistic → diff, then
# re-join full_roc_auc & baseline_roc_auc at the chosen peak window.
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
