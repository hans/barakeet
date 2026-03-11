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
# Inspect Ganong decoding results for a single subject — causal5 pipeline.
#
# Loads `results.joblib` from `ganong_decoding_single_electrode` and shows:
#   1. Distribution of peak full_roc_auc and Δ ROC-AUC (full − baseline) across sites
#   2. Distribution of peak window center time (seconds from epoch onset)
#
# Note: unlike `behavior_decoding_single_electrode`, the Ganong decoder uses
# groupby=None (pooled across completions), so there is no word_end split.
# Peak is defined per (electrode_idx, phoneme_pair): average over folds at each
# window, then take the window maximizing Δ ROC-AUC.

# %%
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# %% tags=["parameters"]
subject = "EC270"

results_path = f"outputs/causal5/ganong_decoding/{subject}/results.joblib"

# Timing parameters (from config.yaml)
sfreq = 100       # Hz
window_size = 15  # samples
epoch_tmin = -0.5  # seconds; epoch start relative to word onset

# %%
data = joblib.load(results_path)

# %% [markdown]
# ## Build results DataFrame
#
# `decoding_results` is a dict mapping `(subject, electrode_idx, phoneme_pair)` →
# DataFrame (one row per fold × window). No `word_end` column (groupby=None).

# %%
results_df = pd.concat(
    data["decoding_results"],
    names=["subject", "population", "phoneme_pair"],
    ignore_index=True,
)
results_df["electrode_idx"] = results_df["population"].astype(int)
results_df["diff"] = results_df["full_roc_auc"] - results_df["baseline_roc_auc"]

print(f"Electrodes: {results_df.electrode_idx.nunique()}, "
      f"phoneme pairs: {results_df.phoneme_pair.nunique()}, "
      f"windows: {results_df[['smin','smax']].drop_duplicates().shape[0]}")
results_df.head()

# %% [markdown]
# ## Find peak window per site
#
# For each (electrode_idx, phoneme_pair): average metrics over folds at each
# window, then take the window with the highest mean Δ ROC-AUC.

# %%
mean_results = (
    results_df
    .groupby(["electrode_idx", "phoneme_pair", "smin", "smax"])[
        ["full_roc_auc", "diff"]
    ]
    .mean()
    .reset_index()
)

peak_idx = mean_results.groupby(["electrode_idx", "phoneme_pair"])["diff"].idxmax()
peak = mean_results.loc[peak_idx].copy()
peak["window_center_s"] = (peak["smin"] + window_size / 2) / sfreq + epoch_tmin

print(f"Sites: {len(peak)}")
peak.describe()

# %% [markdown]
# ## Distributions

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for phoneme_pair, grp in peak.groupby("phoneme_pair"):
    axes[0, 0].hist(grp["full_roc_auc"], bins=15, alpha=0.6, label=phoneme_pair)
axes[0, 0].axvline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[0, 0].set_xlabel("Peak full ROC-AUC")
axes[0, 0].set_ylabel("Sites")
axes[0, 0].set_title(f"{subject} — Ganong peak performance (full model)")
axes[0, 0].legend(title="phoneme_pair")

for phoneme_pair, grp in peak.groupby("phoneme_pair"):
    axes[0, 1].hist(grp["diff"], bins=15, alpha=0.6, label=phoneme_pair)
axes[0, 1].axvline(0, color="k", linestyle="--", linewidth=0.8)
axes[0, 1].set_xlabel("Peak Δ ROC-AUC (full − baseline)")
axes[0, 1].set_ylabel("Sites")
axes[0, 1].set_title(f"{subject} — Ganong peak Δ performance")
axes[0, 1].legend(title="phoneme_pair")

for phoneme_pair, grp in peak.groupby("phoneme_pair"):
    axes[1, 0].hist(grp["window_center_s"], bins=15, alpha=0.6, label=phoneme_pair)
axes[1, 0].set_xlabel("Peak window center (s from onset)")
axes[1, 0].set_ylabel("Sites")
axes[1, 0].set_title(f"{subject} — Ganong peak time")
axes[1, 0].legend(title="phoneme_pair")

# Scatter: peak time vs peak performance
for phoneme_pair, grp in peak.groupby("phoneme_pair"):
    axes[1, 1].scatter(grp["window_center_s"], grp["full_roc_auc"], alpha=0.6, label=phoneme_pair, s=25)
axes[1, 1].axhline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[1, 1].set_xlabel("Peak window center (s from onset)")
axes[1, 1].set_ylabel("Peak full ROC-AUC")
axes[1, 1].set_title(f"{subject} — Peak time vs. performance")
axes[1, 1].legend(title="phoneme_pair")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary table

# %%
peak.groupby("phoneme_pair")[["full_roc_auc", "diff", "window_center_s"]].describe()
