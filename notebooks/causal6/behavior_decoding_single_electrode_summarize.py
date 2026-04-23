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
# causal6 summarize: behavior decoding with control predictor.
#
# Reads the three parquets from `behavior_decoding_single_electrode` and:
#   1. Filters windows to those ending before word_end offset + 200 ms
#   2. For each (subject, electrode, phoneme_pair, word_end), finds the peak window
#      by argmax of mean(full_roc_auc - baseline_roc_auc)
#   3. Writes `peak_summary.parquet`  (one row per site with its peak window)
#   4. Writes `peak_predictions.parquet` (trial-level preds filtered to the peak window)

# %%
from pathlib import Path

import polars as pl

# %%
from src.stimuli import OFFSET_DICT

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet"
predictions_path = f"outputs/causal6/behavior_decoding_single_electrode/{subject}/predictions.parquet"
outdir = "."

epoch_tmin = -0.4
epoch_sfreq = 100
behav_peak_post_offset_s = 0.2
peak_search_smin = 0
peak_search_smax = 290

# %%
scores = pl.read_parquet(scores_path)
predictions = pl.read_parquet(predictions_path)

# %%
# Pivot scores wide (baseline vs full) to compute diff
full_scores = scores.filter(pl.col("model") == "full").drop("model")
base_scores = (
    scores.filter(pl.col("model") == "baseline")
    .drop("model", "electrode_idx", "smin", "smax")
    .rename({"test_roc_auc": "baseline_roc_auc"})
)

paired = full_scores.rename({"test_roc_auc": "full_roc_auc"}).join(
    base_scores, on=["subject", "phoneme_pair", "word_end", "fold"], how="left"
).with_columns(
    (pl.col("full_roc_auc") - pl.col("baseline_roc_auc")).alias("diff")
)

# %%
# Filter windows: in valid range AND ending before word-end offset + post-offset allowance
offset_samples = {
    we: int((offset_s - epoch_tmin) * epoch_sfreq + behav_peak_post_offset_s * epoch_sfreq)
    for we, offset_s in OFFSET_DICT.items()
}
paired = paired.with_columns(
    pl.col("word_end").replace_strict(offset_samples, default=None).alias("_smax_limit")
).filter(
    (pl.col("smin") >= peak_search_smin)
    & (pl.col("smax") <= pl.col("_smax_limit"))
    & (pl.col("smax") <= peak_search_smax)
).drop("_smax_limit")

# %%
# Mean across folds, then argmax diff per site
site_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
window_keys = site_keys + ["smin", "smax"]

window_mean = (
    paired.group_by(window_keys)
    .agg(pl.col("full_roc_auc").mean(), pl.col("baseline_roc_auc").mean(), pl.col("diff").mean())
)

peak_summary = (
    window_mean.sort("diff", descending=True)
    .group_by(site_keys, maintain_order=True)
    .agg(pl.all().first())
)

print(f"{peak_summary.height} sites with peak windows identified")

# %%
# Filter predictions to peak window per site
peak_keys = peak_summary.select(site_keys + ["smin", "smax"])
peak_predictions = predictions.join(
    peak_keys, on=site_keys + ["smin", "smax"], how="inner",
)

# %%
outdir = Path(outdir)
peak_summary.write_parquet(outdir / "peak_summary.parquet")
peak_predictions.write_parquet(outdir / "peak_predictions.parquet")
window_mean.write_parquet(outdir / "window_mean_scores.parquet")
print(f"Wrote peak_summary.parquet ({peak_summary.height} rows), "
      f"peak_predictions.parquet ({peak_predictions.height} rows), "
      f"window_mean_scores.parquet ({window_mean.height} rows) to {outdir}")
