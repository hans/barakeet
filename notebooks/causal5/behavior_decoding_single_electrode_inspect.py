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
# Inspect behavioral decoding results for a single subject — causal5 pipeline.
#
# Loads `A_results.csv` from `behavior_decoding_single_electrode_summarize` and shows:
#   1. Distribution of peak full_roc_auc and Δ ROC-AUC (full − baseline) across sites
#   2. Distribution of peak window center time (seconds from epoch onset)
#
# Peak is defined per (electrode_idx, phoneme_pair, word_end): average over folds at
# each window, then take the window maximizing Δ ROC-AUC (= diff).

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %% tags=["parameters"]
subject = "EC243"

results_path = f"outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A_results.csv"

# Timing parameters (from config.yaml)
sfreq = 100       # Hz
window_size = 15  # samples
epoch_tmin = -0.5  # seconds; epoch start relative to word onset

# %%
results = pd.read_csv(results_path)
results["electrode_idx"] = results["population"].astype(int)
if "diff" not in results.columns:
    results["diff"] = results["full_roc_auc"] - results["baseline_roc_auc"]
results.head()

# %% [markdown]
# ## Find peak window per site
#
# For each (electrode_idx, phoneme_pair, word_end): average metrics over folds at
# each window, then take the window with the highest mean Δ ROC-AUC.

# %%
mean_results = (
    results
    .groupby(["electrode_idx", "phoneme_pair", "word_end", "smin", "smax"])[
        ["full_roc_auc", "diff"]
    ]
    .mean()
    .reset_index()
)

peak_idx = mean_results.groupby(["electrode_idx", "phoneme_pair", "word_end"])["diff"].idxmax()
peak = mean_results.loc[peak_idx].copy()
peak["window_center_s"] = (peak["smin"] + window_size / 2) / sfreq + epoch_tmin

print(f"Sites × word_end: {len(peak)}")
peak.describe()

# %% [markdown]
# ## Distributions

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for word_end, grp in peak.groupby("word_end"):
    axes[0, 0].hist(grp["full_roc_auc"], bins=20, alpha=0.6, label=word_end)
axes[0, 0].axvline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[0, 0].set_xlabel("Peak full ROC-AUC")
axes[0, 0].set_ylabel("Sites")
axes[0, 0].set_title(f"{subject} — Behavioral peak performance (full model)")
axes[0, 0].legend(title="word_end")

for word_end, grp in peak.groupby("word_end"):
    axes[0, 1].hist(grp["diff"], bins=20, alpha=0.6, label=word_end)
axes[0, 1].axvline(0, color="k", linestyle="--", linewidth=0.8)
axes[0, 1].set_xlabel("Peak Δ ROC-AUC (full − baseline)")
axes[0, 1].set_ylabel("Sites")
axes[0, 1].set_title(f"{subject} — Behavioral peak Δ performance")
axes[0, 1].legend(title="word_end")

for word_end, grp in peak.groupby("word_end"):
    axes[1, 0].hist(grp["window_center_s"], bins=20, alpha=0.6, label=word_end)
axes[1, 0].set_xlabel("Peak window center (s from onset)")
axes[1, 0].set_ylabel("Sites")
axes[1, 0].set_title(f"{subject} — Behavioral peak time")
axes[1, 0].legend(title="word_end")

# Scatter: peak time vs peak performance
for word_end, grp in peak.groupby("word_end"):
    axes[1, 1].scatter(grp["window_center_s"], grp["full_roc_auc"], alpha=0.5, label=word_end, s=20)
axes[1, 1].axhline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[1, 1].set_xlabel("Peak window center (s from onset)")
axes[1, 1].set_ylabel("Peak full ROC-AUC")
axes[1, 1].set_title(f"{subject} — Peak time vs. performance")
axes[1, 1].legend(title="word_end")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary table

# %%
peak.groupby("word_end")[["full_roc_auc", "diff", "window_center_s"]].describe()
