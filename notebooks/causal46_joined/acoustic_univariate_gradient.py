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
# # Univariate gradient acoustic encoding — causal46_joined edition
#
# At each manifest-curated acoustic site, extract trial-level mean HGA at the
# peak acoustic window, min-max-normalize so endpoint means map to 0 (low) and
# 1 (high) with a polarity flag, and fit a 2-parameter sigmoid:
#
#   f(x; x0, k) = 1 / (1 + exp(-(x - x0) / k))
#
# Steepness k → 0 = categorical (step function), k → ∞ = graded (linear).
# Adapted from the sigmoid-fit portion of
# notebooks/causal5/acoustic_morphology_on_ambiguous.py.

# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# %%
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _gradient_pool import load_acoustic_pool  # noqa: E402

from src.data import add_metadata_features
from src.models.sigmoid import (
    EFFECTIVELY_LINEAR_K,
    SIGMOID_2P_BOUNDS,
    SIGMOID_2P_P0_LIST,
    fit_model,
    sigmoid_model_2p,
)

# %% tags=["parameters"]
subject = "EC250"

epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/EC250/phon_peaks.parquet"
manifest_path = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv"
outdir = "."

# Drop sites where |mean_step1 − mean_step6| < endpoint_separation_floor *
# pooled within-endpoint std. Inherited from causal5.
endpoint_separation_floor = 0.1

# Require this many distinct morph steps represented in the site's data
# before attempting the sigmoid fit.
min_distinct_steps = 5

# %%
# Winsorize hga_norm at these percentiles per site before sigmoid fit.
# Hard-coded rather than parameterized because papermill flattens tuples
# to strings.
winsor_pct = (2.5, 97.5)

subject = Path(epochs_path).name.split("_")[0]
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Build site pool

# %%
pool = load_acoustic_pool(manifest_path, phon_peaks_path, subject=subject)
print(f"{subject}: {len(pool)} (electrode, phoneme_pair) sites in manifest+phon_peaks pool")

# %% [markdown]
# ## Extract trial-level HGA at each site's peak window

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.apply_baseline()
data_arr = epochs.get_data()  # (n_trials, n_channels, n_samples)
md = add_metadata_features(epochs.metadata).reset_index(drop=True)
md.index.name = "epoch_idx"
md = md.reset_index()

# %%
trial_rows = []

for _, site_row in tqdm(pool.iterrows(), total=len(pool), desc="HGA extraction"):
    ei = int(site_row["electrode_idx"])
    pp = site_row["phoneme_pair"]
    smin_w, smax_w = int(site_row["smin"]), int(site_row["smax"])

    site_md = md[md["phoneme_pair"] == pp]
    if len(site_md) == 0:
        continue

    epoch_idxs = site_md["epoch_idx"].values.astype(int)
    hga_raw = data_arr[epoch_idxs, ei, smin_w:smax_w].mean(axis=1)

    trial_rows.append(pd.DataFrame({
        "subject": subject,
        "electrode_idx": ei,
        "phoneme_pair": pp,
        "smin": smin_w,
        "smax": smax_w,
        "epoch_idx": epoch_idxs,
        "resampled": site_md["resampled"].values,
        "hga_raw": hga_raw,
    }))

_trial_schema_pre = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax",
                     "epoch_idx", "resampled", "hga_raw"]
if not trial_rows:
    trial_df = pd.DataFrame(columns=_trial_schema_pre)
else:
    trial_df = pd.concat(trial_rows, ignore_index=True)
print(f"trial_df pre-norm: {len(trial_df)} rows")

# %% [markdown]
# ## Endpoint min-max normalization with polarity flag
#
# For each (electrode, phoneme_pair) site, normalize HGA so that the endpoint
# with lower mean maps to 0 and the higher endpoint maps to 1. Track
# `hga_polarity` per site: +1 if step 6 was the high end natively (curve rises
# 1→6 unaltered), −1 if step 1 was higher (curve is flipped so normalized
# values still rise 1→6).
#
# Drop sites where the endpoint mean separation is too small (`< floor × std`)
# — normalizing by a tiny denominator amplifies noise into extreme hga_norm.

# %%
_endpoint_stats_cols = ["subject", "electrode_idx", "phoneme_pair",
                        "mean_hga_step1", "mean_hga_step6", "hga_endpoint_std"]

if len(trial_df) > 0:
    endpoint_means = (
        trial_df[trial_df["resampled"].isin([1, 6])]
        .groupby(["subject", "electrode_idx", "phoneme_pair", "resampled"])["hga_raw"]
        .mean()
        .unstack("resampled")
        .rename(columns={1: "mean_hga_step1", 6: "mean_hga_step6"})
    )
    endpoint_std = (
        trial_df[trial_df["resampled"].isin([1, 6])]
        .groupby(["subject", "electrode_idx", "phoneme_pair"])["hga_raw"]
        .std()
        .rename("hga_endpoint_std")
    )
    endpoint_stats = endpoint_means.join(endpoint_std, how="left").reset_index()
    for col in _endpoint_stats_cols:
        if col not in endpoint_stats.columns:
            endpoint_stats[col] = pd.Series(dtype="float64")
else:
    endpoint_stats = pd.DataFrame(columns=_endpoint_stats_cols)

_n_before = len(endpoint_stats)
endpoint_stats = endpoint_stats[
    (endpoint_stats["mean_hga_step1"] - endpoint_stats["mean_hga_step6"]).abs()
    > endpoint_separation_floor * endpoint_stats["hga_endpoint_std"].fillna(0.0)
].copy()
print(f"endpoint filter: {_n_before} → {len(endpoint_stats)} sites "
      f"(dropped {_n_before - len(endpoint_stats)} below separation floor)")

endpoint_stats["hga_polarity"] = np.where(
    endpoint_stats["mean_hga_step6"] > endpoint_stats["mean_hga_step1"], 1, -1
)
endpoint_stats["hga_low"] = endpoint_stats[
    ["mean_hga_step1", "mean_hga_step6"]
].min(axis=1)
endpoint_stats["hga_high"] = endpoint_stats[
    ["mean_hga_step1", "mean_hga_step6"]
].max(axis=1)

trial_df = trial_df.merge(
    endpoint_stats[
        ["subject", "electrode_idx", "phoneme_pair",
         "hga_polarity", "hga_low", "hga_high"]
    ],
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="inner",
)

_raw_norm = (trial_df["hga_raw"] - trial_df["hga_low"]) / (
    trial_df["hga_high"] - trial_df["hga_low"]
)
trial_df["hga_norm"] = np.where(
    trial_df["hga_polarity"] < 0, 1.0 - _raw_norm, _raw_norm
)
trial_df = trial_df.drop(columns=["hga_low", "hga_high"])

trial_df.to_parquet(outdir / "trial_df.parquet", index=False)
print(f"trial_df: {len(trial_df)} rows across "
      f"{trial_df[['electrode_idx', 'phoneme_pair']].drop_duplicates().shape[0]} sites")

# %% [markdown]
# ## Per-site sigmoid fit
#
# Fit to trial-level (resampled, hga_norm) pairs after per-site winsorization.
# Sites failing the fit (or with <5 distinct steps represented) get NaN params.

# %%
steps_all = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
model_rows = []

site_keys = trial_df[
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
].drop_duplicates()

for _, site_row in tqdm(site_keys.iterrows(), total=len(site_keys), desc="Sigmoid fits"):
    ei = int(site_row["electrode_idx"])
    pp = site_row["phoneme_pair"]

    site_trials = trial_df[
        (trial_df["electrode_idx"] == ei)
        & (trial_df["phoneme_pair"] == pp)
    ].dropna(subset=["resampled", "hga_norm"])

    if len(site_trials) == 0:
        continue

    x_all = site_trials["resampled"].values.astype(float)
    y_raw = site_trials["hga_norm"].values
    lo, hi = np.percentile(y_raw, list(winsor_pct))
    y_all = np.clip(y_raw, lo, hi)

    if len(np.unique(x_all)) < min_distinct_steps:
        continue

    popt, _ = fit_model(
        sigmoid_model_2p, x_all, y_all, SIGMOID_2P_P0_LIST, SIGMOID_2P_BOUNDS
    )

    row = {
        "subject": subject,
        "electrode_idx": ei,
        "phoneme_pair": pp,
        "smin": int(site_row["smin"]),
        "smax": int(site_row["smax"]),
        "n_trials_fit": int(len(x_all)),
    }

    if popt is not None:
        y_pred = sigmoid_model_2p(x_all, *popt)
        ss_res = float(np.sum((y_all - y_pred) ** 2))
        ss_tot = float(np.sum((y_all - y_all.mean()) ** 2))
        row["sigmoid_x0"] = float(popt[0])
        row["sigmoid_k"] = float(popt[1])
        row["sigmoid_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        row["sigmoid_effectively_linear"] = float(popt[1]) > EFFECTIVELY_LINEAR_K
    else:
        row.update({
            "sigmoid_x0": np.nan,
            "sigmoid_k": np.nan,
            "sigmoid_r2": np.nan,
            "sigmoid_effectively_linear": np.nan,
        })

    overall_means = (
        site_trials.groupby("resampled")["hga_norm"].mean().reindex(steps_all)
    )
    for s in steps_all:
        val = overall_means.get(s, np.nan)
        row[f"norm_proba_step{int(s)}"] = (
            float(val) if not np.isnan(val) else np.nan
        )

    model_rows.append(row)

_model_schema = (
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax",
     "n_trials_fit", "sigmoid_x0", "sigmoid_k", "sigmoid_r2",
     "sigmoid_effectively_linear"]
    + [f"norm_proba_step{int(s)}" for s in steps_all]
)
model_comparison_df = (
    pd.DataFrame(model_rows) if model_rows
    else pd.DataFrame(columns=_model_schema)
)
model_comparison_df.to_parquet(outdir / "model_comparison_df.parquet", index=False)
print(f"model_comparison_df: {len(model_comparison_df)} sites")
model_comparison_df[["sigmoid_k", "sigmoid_x0", "sigmoid_r2"]].describe()
