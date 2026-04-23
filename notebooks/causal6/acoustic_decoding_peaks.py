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
# causal6: per-subject acoustic-decoding peak finding.
#
# Mirrors causal5's `acoustic_decoding_peaks` rule but reads the causal6 parquet
# outputs. Restricts to `target == "categorical_acoustic_cue"` (matching causal5's
# `measure == "categorical_acoustic_cue"` peak convention) and finds the peak
# window per (electrode, phoneme_pair) by max mean test AUC.

# %%
from pathlib import Path

import polars as pl

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet"
outdir = "."

target = "categorical_acoustic_cue"
peak_search_smin = 0
peak_search_smax = 290  # same as causal5 default

# %%
scores = pl.read_parquet(scores_path).filter(pl.col("target") == target)

scores = scores.filter(
    (pl.col("smin") >= peak_search_smin) & (pl.col("smax") <= peak_search_smax)
)

# %%
site_keys = ["subject", "electrode_idx", "phoneme_pair"]
window_keys = site_keys + ["smin", "smax"]

window_mean = scores.group_by(window_keys).agg(pl.col("test_roc_auc").mean())

peak_summary = (
    window_mean.sort("test_roc_auc", descending=True)
    .group_by(site_keys, maintain_order=True)
    .agg(pl.all().first())
)

# %%
outdir = Path(outdir)
peak_summary.write_parquet(outdir / "phon_peaks.parquet")
window_mean.write_parquet(outdir / "phon_roc_auc_searchlight.parquet")
print(f"Wrote phon_peaks.parquet ({peak_summary.height}) + "
      f"phon_roc_auc_searchlight.parquet ({window_mean.height}) to {outdir}")
