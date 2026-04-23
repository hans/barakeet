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
# causal6: per-subject acoustic decoding significance test.
#
# Proper refit-based permutation test (cf. causal5's score-shuffle shortcut).
# For each of K permutations, shuffles trial labels globally and refits the
# searchlight on the same fold assignments. The test statistic per site is
# the max-across-windows of fold-mean test ROC-AUC; p-value is
# `(#{T_null >= T_obs} + 1) / (K + 1)` (one-tailed, max-stat corrected).
#
# Inputs:
#   epochs_path, electrodes_path — as in the real decoder
#   real_scores_path             — real decoder's scores.parquet (for T_obs)
#   reg_lambda                   — pre-resolved from reg_lambda_winners.json
#
# Outputs:
#   significance.parquet         — one row per site, with p_value + diagnostics
#   null_distribution.parquet    — one row per (site × permutation), T_null

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %%
from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd
import polars as pl
import torch

# %%
from src.data import add_metadata_features
from src.models.causal6 import (
    make_windows,
    run_acoustic_searchlight_permutations,
)

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
real_scores_path = (
    f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet"
)
outdir = "."

min_sample = 1
window_size = 15
stride = 2

target = "categorical_acoustic_cue"
peak_search_smin = 0
peak_search_smax = 290

reg_lambda = 1.0
n_folds = 5
cv_random_state = 42
device = "cuda"
tol = 1e-6
max_iter = 50

n_permutations = 500
permutation_seed = 0
permutation_chunk_size = 6

# %%
subject = re.findall(r"(EC[\d]+)_epo", str(epochs_path))[0]
outdir = Path(outdir)

# %%
electrode_df = pd.read_csv(electrodes_path)
speech_responsive_idxs = sorted(
    electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"].unique().astype(int)
)

epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)

max_sample = epochs.times.shape[0]
windows = make_windows(min_sample, max_sample, window_size, stride)

# %% [markdown]
# ## T_obs per site from the real decoder's scores.parquet
#
# Uses the same peak-window filter as `acoustic_decoding_peaks.py`: keep
# windows with `smin >= peak_search_smin` and `smax <= peak_search_smax`,
# take fold-mean `test_roc_auc` per (site, window), then max over windows.

# %%
site_keys = ["subject", "electrode_idx", "phoneme_pair"]
window_keys = site_keys + ["smin", "smax"]


def _window_filter(scores: pl.DataFrame) -> pl.DataFrame:
    return scores.filter(
        (pl.col("smin") >= peak_search_smin) & (pl.col("smax") <= peak_search_smax)
    )


real_scores = pl.read_parquet(real_scores_path).filter(pl.col("target") == target)
real_scores = _window_filter(real_scores)

real_window_mean = real_scores.group_by(window_keys).agg(
    pl.col("test_roc_auc").mean().alias("fold_mean_auc")
)
real_peaks = (
    real_window_mean.sort("fold_mean_auc", descending=True)
    .group_by(site_keys, maintain_order=True)
    .agg(pl.all().first())
    .rename({"smin": "peak_smin", "smax": "peak_smax", "fold_mean_auc": "T_obs"})
)
print(f"T_obs computed for {real_peaks.height} sites")

# %% [markdown]
# ## Null distribution: refit K permutations per site

# %%
permute_seeds = list(range(permutation_seed, permutation_seed + n_permutations))

null_scores = run_acoustic_searchlight_permutations(
    epochs, subject=subject,
    electrode_idxs=speech_responsive_idxs,
    windows=windows,
    reg_lambda=reg_lambda,
    permute_seeds=permute_seeds,
    permutation_chunk_size=permutation_chunk_size,
    target=target,
    n_folds=n_folds, cv_random_state=cv_random_state,
    device=device, dtype=torch.float32,
    tol=tol, max_iter=max_iter,
)
assert null_scores.height > 0, f"[{subject}] acoustic null run produced no rows"
null_scores = _window_filter(null_scores)

# fold-mean AUC per (site, window, permutation), then max over windows → T_null
null_window_mean = null_scores.group_by(window_keys + ["permutation_idx"]).agg(
    pl.col("test_roc_auc").mean().alias("fold_mean_auc")
)
null_per_site_perm = (
    null_window_mean.group_by(site_keys + ["permutation_idx"])
    .agg(pl.col("fold_mean_auc").max().alias("T_null"))
)

# %% [markdown]
# ## p-values + diagnostics

# %%
joined = (
    null_per_site_perm
    .join(real_peaks, on=site_keys, how="inner")
    .filter(pl.col("T_null").is_not_nan())
)

# Per-site: (#{T_null >= T_obs} + 1) / (K + 1), where K = valid permutation count
per_site_stats = (
    joined.group_by(site_keys + ["peak_smin", "peak_smax", "T_obs"])
    .agg(
        pl.len().alias("n_permutations"),
        (pl.col("T_null") >= pl.col("T_obs")).cast(pl.Int64).sum().alias("_ge_count"),
        pl.col("T_null").quantile(0.05).alias("null_q05"),
        pl.col("T_null").quantile(0.50).alias("null_q50"),
        pl.col("T_null").quantile(0.95).alias("null_q95"),
        pl.col("T_null").quantile(0.99).alias("null_q99"),
    )
    .with_columns(
        ((pl.col("_ge_count") + 1) / (pl.col("n_permutations") + 1)).alias("p_value"),
    )
    .drop("_ge_count")
)

# Compute class-count diagnostics (n_trials, n_pos) per site from the real-run metadata
# by re-deriving via the real scores' per-fold n_train + n_test.
# n_trials = sum(n_test) across folds; n_pos is not stored in scores, so we
# recover it from epochs.metadata directly.
md = epochs.metadata
pp_to_trials: dict[tuple[str], tuple[int, int]] = {}
for pp in sorted(md.phoneme_pair.dropna().unique()):
    mask = (md.phoneme_pair == pp).values & md.resampled.isin([1, 6]).values
    y = (md.categorical_acoustic_cue[mask].values > 0).astype(np.int64)
    pp_to_trials[pp] = (int(y.size), int(y.sum()))
trial_df = pl.DataFrame({
    "phoneme_pair": list(pp_to_trials.keys()),
    "n_trials": [v[0] for v in pp_to_trials.values()],
    "n_pos": [v[1] for v in pp_to_trials.values()],
})

per_site_stats = per_site_stats.join(trial_df, on="phoneme_pair", how="left")

# %%
print(
    f"[{subject}] acoustic: {per_site_stats.height} sites tested, "
    f"{(per_site_stats['p_value'] < 0.05).sum()} with p<0.05 "
    f"(uncorrected; BH-FDR applied in aggregate)"
)

# %%
per_site_stats.write_parquet(outdir / "significance.parquet")
null_per_site_perm.write_parquet(outdir / "null_distribution.parquet")
print(
    f"Wrote significance.parquet ({per_site_stats.height} rows), "
    f"null_distribution.parquet ({null_per_site_perm.height} rows) to {outdir}"
)
