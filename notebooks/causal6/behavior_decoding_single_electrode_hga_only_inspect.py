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
# Inspect behavioral decoding (HGA-only, no baseline) for a single subject — causal6 pipeline.
#
# Loads `peak_summary.parquet` + `window_mean_scores.parquet` from
# `behavior_decoding_single_electrode_hga_only_summarize` and shows:
#   1. Distribution of peak ROC-AUC across sites
#   2. Distribution of peak window center time (seconds from epoch onset)
#   3. Median ROC-AUC time-course across electrodes, per (phoneme_pair, word_end)
#
# Peak-finding is already done by the summarize rule (argmax test_roc_auc per
# site × word_end); this notebook just visualizes.

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %% tags=["parameters"]
subject = "EC282"

peak_summary_path = f"outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/peak_summary.parquet"
window_mean_path = f"outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/{subject}/window_mean_scores.parquet"

# Timing parameters (from config.yaml — causal6 uses epoch_tmin = -0.4s)
sfreq = 100       # Hz
window_size = 15  # samples
epoch_tmin = -0.4  # seconds; epoch start relative to word onset

# %%
peaks = pd.read_parquet(peak_summary_path)
window_mean = pd.read_parquet(window_mean_path)

peaks["window_center_s"] = (peaks["smin"] + window_size / 2) / sfreq + epoch_tmin
peaks.head()

# %% [markdown]
# ## Distributions

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for word_end, grp in peaks.groupby("word_end"):
    axes[0].hist(grp["test_roc_auc"], bins=20, alpha=0.6, label=word_end)
axes[0].axvline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[0].set_xlabel("Peak ROC-AUC")
axes[0].set_ylabel("Sites")
axes[0].set_title(f"{subject} — Behavioral HGA-only peak performance")
axes[0].legend(title="word_end")

for word_end, grp in peaks.groupby("word_end"):
    axes[1].hist(grp["window_center_s"], bins=20, alpha=0.6, label=word_end)
axes[1].set_xlabel("Peak window center (s from onset)")
axes[1].set_ylabel("Sites")
axes[1].set_title(f"{subject} — Behavioral HGA-only peak time")
axes[1].legend(title="word_end")

for word_end, grp in peaks.groupby("word_end"):
    axes[2].scatter(grp["window_center_s"], grp["test_roc_auc"], alpha=0.5, label=word_end, s=20)
axes[2].axhline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[2].set_xlabel("Peak window center (s from onset)")
axes[2].set_ylabel("Peak ROC-AUC")
axes[2].set_title(f"{subject} — Peak time vs. performance")
axes[2].legend(title="word_end")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Time-course: median ROC-AUC across electrodes

# %%
fig, ax = plt.subplots(figsize=(9, 4))

for (phoneme_pair, word_end), grp in window_mean.groupby(["phoneme_pair", "word_end"]):
    tc = grp.groupby(["smin", "smax"])["test_roc_auc"].median().reset_index()
    tc["window_center_s"] = (tc["smin"] + window_size / 2) / sfreq + epoch_tmin
    ax.plot(tc["window_center_s"], tc["test_roc_auc"], label=f"{phoneme_pair} / {word_end}")

ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
ax.set_xlabel("Window center (s from onset)")
ax.set_ylabel("Median ROC-AUC across electrodes")
ax.set_title(f"{subject} — Behavioral HGA-only time-course")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary table

# %%
peaks.groupby("word_end")[["test_roc_auc", "window_center_s"]].describe()
