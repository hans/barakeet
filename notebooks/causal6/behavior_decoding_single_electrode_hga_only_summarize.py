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
# causal6 summarize: behavior decoding HGA-only variant.
#
# Peak-finding uses max `test_roc_auc` directly (no baseline comparison).

# %%
from pathlib import Path

import polars as pl

# %%
from src.stimuli import OFFSET_DICT

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/scores.parquet"
predictions_path = f"outputs/causal6/behavior_decoding_single_electrode_hga_only/{subject}/predictions.parquet"
outdir = "."

epoch_tmin = -0.4
epoch_sfreq = 100
behav_peak_post_offset_s = 0.2
min_decoding_sample = 0
max_decoding_sample = 290

# %%
scores = pl.read_parquet(scores_path)
predictions = pl.read_parquet(predictions_path)

# %%
offset_samples = {
    we: int((offset_s - epoch_tmin) * epoch_sfreq + behav_peak_post_offset_s * epoch_sfreq)
    for we, offset_s in OFFSET_DICT.items()
}
scores = scores.with_columns(
    pl.col("word_end").replace_strict(offset_samples, default=None).alias("_smax_limit")
).filter(
    (pl.col("smin") >= min_decoding_sample)
    & (pl.col("smax") <= pl.col("_smax_limit"))
    & (pl.col("smax") <= max_decoding_sample)
).drop("_smax_limit")

# %%
site_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
window_keys = site_keys + ["smin", "smax"]

window_mean = (
    scores.group_by(window_keys)
    .agg(pl.col("test_roc_auc").mean())
)

peak_summary = (
    window_mean.sort("test_roc_auc", descending=True)
    .group_by(site_keys, maintain_order=True)
    .agg(pl.all().first())
)

print(f"{peak_summary.height} sites with peak windows identified")

# %%
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
