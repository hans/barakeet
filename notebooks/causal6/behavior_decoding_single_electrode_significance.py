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
# causal6: per-subject significance test for behavior decoding with control.
#
# Refit-based permutation test. Statistic per (site, window) is the
# fold-mean of `full_roc_auc - baseline_roc_auc` — the same peak criterion
# used by the summarize rule. Per site T_obs/T_null = max across windows;
# p-value = `(#{T_null >= T_obs} + 1) / (K + 1)`.
#
# Under each permutation we refit BOTH full and baseline models on the same
# shuffled labels (run_behavior_with_control_permutations does this by
# deterministic seeding), keeping the diff statistic properly paired.
#
# Outputs:
#   significance.parquet        — per-site p-values + diagnostics
#   null_distribution.parquet   — per-(site × permutation) T_null

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
    run_behavior_with_control_permutations,
)
from src.stimuli import OFFSET_DICT

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
real_scores_path = (
    f"outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet"
)
outdir = "."

min_sample = 1
window_size = 15
stride = 2

epoch_tmin = -0.4
epoch_sfreq = 100
behav_peak_post_offset_s = 0.2
peak_search_smin = 0
peak_search_smax = 290

reg_lambda = 1.0
reg_lambda_baseline = None
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
# ## Shared window filter + diff aggregation
#
# Pairs full vs baseline on (subject, phoneme_pair, word_end, fold) — same
# key scheme as the summarize rule at
# notebooks/causal6/behavior_decoding_single_electrode_summarize.py.

# %%
offset_samples = {
    we: int((offset_s - epoch_tmin) * epoch_sfreq + behav_peak_post_offset_s * epoch_sfreq)
    for we, offset_s in OFFSET_DICT.items()
}
site_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
window_keys = site_keys + ["smin", "smax"]


def _pair_and_filter(scores: pl.DataFrame, extra_keys: list[str] | None = None) -> pl.DataFrame:
    """Join full + baseline on (phoneme_pair, word_end, fold [, perm]) and
    restrict to valid peak windows.

    `extra_keys` lets callers include `permutation_idx` for null runs.
    """
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


# %% [markdown]
# ## T_obs per site from real scores.parquet

# %%
real_scores = pl.read_parquet(real_scores_path)
real_paired = _pair_and_filter(real_scores)

real_window_mean = real_paired.group_by(window_keys).agg(
    pl.col("diff").mean().alias("fold_mean_diff")
)
real_peaks = (
    real_window_mean.sort("fold_mean_diff", descending=True)
    .group_by(site_keys, maintain_order=True)
    .agg(pl.all().first())
    .rename({"smin": "peak_smin", "smax": "peak_smax", "fold_mean_diff": "T_obs"})
)
print(f"T_obs computed for {real_peaks.height} sites")

# %% [markdown]
# ## Null distribution: K permutation refits (full + baseline together)

# %%
permute_seeds = list(range(permutation_seed, permutation_seed + n_permutations))

null_scores = run_behavior_with_control_permutations(
    epochs, subject=subject,
    electrode_idxs=speech_responsive_idxs,
    windows=windows,
    reg_lambda=reg_lambda,
    reg_lambda_baseline=reg_lambda_baseline,
    permute_seeds=permute_seeds,
    permutation_chunk_size=permutation_chunk_size,
    n_folds=n_folds, cv_random_state=cv_random_state,
    device=device, dtype=torch.float32,
    tol=tol, max_iter=max_iter,
)
assert null_scores.height > 0, (
    f"[{subject}] behavior with-control null run produced no rows"
)
null_paired = _pair_and_filter(null_scores, extra_keys=["permutation_idx"])

null_window_mean = null_paired.group_by(window_keys + ["permutation_idx"]).agg(
    pl.col("diff").mean().alias("fold_mean_diff")
)
null_per_site_perm = (
    null_window_mean.group_by(site_keys + ["permutation_idx"])
    .agg(pl.col("fold_mean_diff").max().alias("T_null"))
)

# %% [markdown]
# ## p-values + diagnostics

# %%
joined = (
    null_per_site_perm
    .join(real_peaks, on=site_keys, how="inner")
    .filter(pl.col("T_null").is_not_nan())
)

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

# Trial-count diagnostics per (phoneme_pair, word_end)
md = epochs.metadata
rows = []
for pp in sorted(md.phoneme_pair.dropna().unique()):
    pp_mask = (md.phoneme_pair == pp).values
    for we in sorted(md.word_end[pp_mask].dropna().unique()):
        sel = pp_mask & (md.word_end == we).values
        y = md.behavior_dummy_forced[sel].values.astype(np.int64)
        rows.append({
            "phoneme_pair": pp, "word_end": we,
            "n_trials": int(y.size), "n_pos": int(y.sum()),
        })
trial_df = pl.DataFrame(rows) if rows else pl.DataFrame({
    "phoneme_pair": [], "word_end": [], "n_trials": [], "n_pos": [],
})

per_site_stats = per_site_stats.join(trial_df, on=["phoneme_pair", "word_end"], how="left")

# %%
print(
    f"[{subject}] behavior_full: {per_site_stats.height} sites tested, "
    f"{(per_site_stats['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# %%
per_site_stats.write_parquet(outdir / "significance.parquet")
null_per_site_perm.write_parquet(outdir / "null_distribution.parquet")
print(
    f"Wrote significance.parquet ({per_site_stats.height} rows), "
    f"null_distribution.parquet ({null_per_site_perm.height} rows) to {outdir}"
)
