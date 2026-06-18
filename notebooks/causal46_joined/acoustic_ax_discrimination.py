# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AX discrimination — causal46_joined edition
#
# Adjacent-step binary decoders (1v2, 2v3, ..., 5v6) at each manifest-curated
# site's peak acoustic window. Hallmark of categorical perception: high AUC
# at the category boundary, low AUC within categories. Adapted from
# notebooks/causal5/acoustic_ax_discrimination.py with two changes:
#   - Electrode pool from filtered_manifest.csv (any annotated cell qualifies
#     the (subject, electrode_idx, phoneme_pair))
#   - Peak window from causal6 phon_peaks.parquet (null-standardized)
#
# Completions are pooled (matches causal5).

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
import pandas as pd
from tqdm.auto import tqdm

from src.data import add_metadata_features
from src.models.decoding import run_ax_discrimination

# %% tags=["parameters"]
subject = "EC250"

epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
phon_peaks_path = f"outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet"
trial_df_path            = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet"
outdir = "."

ac_p_value_threshold = 0.01  # uncorrected; matches t_tests AC_P_VALUE_THRESHOLD
n_jobs = 4

# %%
subject = Path(epochs_path).name.split("_")[0]
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Build site pool

# %%
pool = (
    pd.read_parquet(phon_peaks_path)
    .query("p_value < @ac_p_value_threshold")
    [["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]]
    .drop_duplicates(subset=["subject", "electrode_idx", "phoneme_pair"])
    .reset_index(drop=True)
)
print(f"{subject}: {len(pool)} (electrode, phoneme_pair) sites "
      f"with p_value < {ac_p_value_threshold}")

# %% [markdown]
# ## Load epochs

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.apply_baseline()
data_arr = epochs.get_data()  # (n_trials, n_channels, n_samples)
md = add_metadata_features(epochs.metadata).reset_index(drop=True)
md.index.name = "epoch_idx"
md = md.reset_index()

# %%
trial_df = pd.read_parquet(trial_df_path)

# %% [markdown]
# ## Train adjacent-step decoders

# %%
rows = []

for _, site_row in tqdm(pool.iterrows(), total=len(pool), desc="AX discrimination"):
    ei = int(site_row["electrode_idx"])
    pp = site_row["phoneme_pair"]
    smin_w, smax_w = int(site_row["smin"]), int(site_row["smax"])

    site_md = md[md["phoneme_pair"] == pp].reset_index(drop=True)
    if len(site_md) == 0:
        continue
    epoch_idxs = site_md["epoch_idx"].values.astype(int)
    site_X = data_arr[epoch_idxs, ei, smin_w:smax_w]  # (n_trials, window_size)

    site_rows = run_ax_discrimination(
        metadata=site_md,
        get_X=lambda idx: site_X[idx],
        phoneme_pair=pp,
        fit_kw=dict(n_jobs=n_jobs),
    )
    for row in site_rows:
        row.update(
            subject=subject,
            electrode_idx=ei,
            phoneme_pair=pp,
            smin=smin_w,
            smax=smax_w,
        )
    rows.extend(site_rows)

_ax_schema = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax",
              "step_a", "step_b", "n_a", "n_b", "roc_auc", "roc_auc_std"]
ax_discrimination_df = (
    pd.DataFrame(rows) if rows else pd.DataFrame(columns=_ax_schema)
)
ax_discrimination_df.to_parquet(outdir / "ax_discrimination_df.parquet", index=False)
print(f"ax_discrimination_df: {len(ax_discrimination_df)} rows")
ax_discrimination_df.head()

# %%
ax_discrimination_df

# %%
trial_df

# %%
import numpy as np
import matplotlib.pyplot as plt

steps_all   = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
step_pairs  = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

step_cmap   = plt.cm.RdBu_r
step_colors = {int(s): step_cmap(i / 5) for i, s in enumerate(range(1, 7))}

rng = np.random.default_rng(0)

trial_df["site_label"] = trial_df.subject.str.cat(trial_df.electrode_idx.astype(str), sep="-").str.cat(trial_df.phoneme_pair, sep=": ")
trial_df["hga_dprime"] = trial_df["hga_endpoint_dprime"] * (trial_df["hga_norm"] - 0.5)

pool["site_label"] = pool.subject.str.cat(pool.electrode_idx.astype(str), sep="-").str.cat(pool.phoneme_pair, sep=": ")
n_sample = min(24, len(pool))
n_cols = 4
n_rows = int(np.ceil(n_sample / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
axes_flat = axes.flatten()

for ax_idx, (_, site_row) in enumerate(pool.iterrows()):
    ax = axes_flat[ax_idx]
    site_data = trial_df[trial_df["site_label"] == site_row["site_label"]].dropna(
        subset=["resampled", "hga_dprime"]
    )

    # Faint per-step scatter (background context)
    for step in range(1, 7):
        mask = site_data["resampled"] == step
        xvals = step + rng.uniform(-0.2, 0.2, mask.sum())
        ax.scatter(xvals, site_data.loc[mask, "hga_dprime"],
                   c=[step_colors[step]], alpha=0.12, s=6, linewidths=0)

    # Mean ± SEM neurometric line
    step_stats = (
        site_data.groupby("resampled")["hga_dprime"]
        .agg(["mean", "sem"])
        .reindex(range(1, 7))
        .dropna()
    )
    ax.errorbar(step_stats.index, step_stats["mean"], yerr=step_stats["sem"],
                fmt="k-o", linewidth=1.5, markersize=4, capsize=3, zorder=5)

    # AX discrimination on secondary y-axis
    site_ax = ax_discrimination_df[
        (ax_discrimination_df["subject"] == site_row["subject"])
        & (ax_discrimination_df["electrode_idx"] == site_row["electrode_idx"])
        & (ax_discrimination_df["phoneme_pair"] == site_row["phoneme_pair"])
    ]
    if len(site_ax) > 0:
        ax2 = ax.twinx()
        midpoints = (site_ax["step_a"].values + site_ax["step_b"].values) / 2.0
        ax2.plot(midpoints, site_ax["roc_auc"].values, "D--",
                 color="green", linewidth=1.2, markersize=4, alpha=0.8, zorder=6)
        ax2.set_ylim(0.4, 1.0)
        ax2.tick_params(axis="y", labelcolor="green", labelsize=6)
        if ax_idx % n_cols == n_cols - 1:
            ax2.set_ylabel("AX discrim. AUC", color="green", fontsize=7)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.5, 6.5)
    ax.set_title(f"{site_row['site_label']}\nAUC={site_row['phon_roc_auc']:.2f}", fontsize=7.5)
    ax.set_xlabel("Morph step")
    if ax_idx % n_cols == 0:
        ax.set_ylabel("HGA d-prime (endpoint SDs)")

step_handles = [
    plt.Line2D([0], [0], color=step_colors[s], linewidth=2, label=f"step {s}")
    for s in range(1, 7)
]
step_handles.append(
    plt.Line2D([0], [0], color="green", linewidth=1.2, linestyle="--",
               marker="D", markersize=4, label="AX discrim. AUC")
)
fig.legend(handles=step_handles, loc="lower right", fontsize=8, ncol=4)
fig.suptitle("Neurometric function at individual acoustic sites (sorted by AUC, low→high)\n"
             "colored by morph step (percept label not available)", fontsize=11)
plt.tight_layout()
