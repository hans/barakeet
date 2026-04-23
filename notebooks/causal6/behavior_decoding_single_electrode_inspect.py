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
# Inspect behavioral decoding (with resampled control) for a single subject — causal6 pipeline.
#
# Loads `peak_summary.parquet` + `window_mean_scores.parquet` from
# `behavior_decoding_single_electrode_summarize` and shows:
#   1. Distribution of peak full_roc_auc and Δ ROC-AUC (full − baseline) across sites
#   2. Distribution of peak window center time (seconds from epoch onset)
#   3. Median Δ ROC-AUC time-course across electrodes, per (phoneme_pair, word_end)
#
# Peak-finding is already done by the summarize rule (argmax diff per
# site × word_end); this notebook just visualizes.

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %% tags=["parameters"]
subject = "EC282"

peak_summary_path = f"outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/peak_summary.parquet"
window_mean_path = f"outputs/causal6/behavior_decoding_single_electrode_summarize/{subject}/window_mean_scores.parquet"

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
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for word_end, grp in peaks.groupby("word_end"):
    axes[0, 0].hist(grp["full_roc_auc"], bins=20, alpha=0.6, label=word_end)
axes[0, 0].axvline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[0, 0].set_xlabel("Peak full ROC-AUC")
axes[0, 0].set_ylabel("Sites")
axes[0, 0].set_title(f"{subject} — Behavioral peak performance (full model)")
axes[0, 0].legend(title="word_end")

for word_end, grp in peaks.groupby("word_end"):
    axes[0, 1].hist(grp["diff"], bins=20, alpha=0.6, label=word_end)
axes[0, 1].axvline(0, color="k", linestyle="--", linewidth=0.8)
axes[0, 1].set_xlabel("Peak Δ ROC-AUC (full − baseline)")
axes[0, 1].set_ylabel("Sites")
axes[0, 1].set_title(f"{subject} — Behavioral peak Δ performance")
axes[0, 1].legend(title="word_end")

for word_end, grp in peaks.groupby("word_end"):
    axes[1, 0].hist(grp["window_center_s"], bins=20, alpha=0.6, label=word_end)
axes[1, 0].set_xlabel("Peak window center (s from onset)")
axes[1, 0].set_ylabel("Sites")
axes[1, 0].set_title(f"{subject} — Behavioral peak time")
axes[1, 0].legend(title="word_end")

for word_end, grp in peaks.groupby("word_end"):
    axes[1, 1].scatter(grp["window_center_s"], grp["diff"], alpha=0.5, label=word_end, s=20)
axes[1, 1].axhline(0, color="k", linestyle="--", linewidth=0.8)
axes[1, 1].set_xlabel("Peak window center (s from onset)")
axes[1, 1].set_ylabel("Peak Δ ROC-AUC (full − baseline)")
axes[1, 1].set_title(f"{subject} — Peak time vs. Δ performance")
axes[1, 1].legend(title="word_end")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Time-course: median Δ ROC-AUC across electrodes

# %%
fig, ax = plt.subplots(figsize=(9, 4))

for (phoneme_pair, word_end), grp in window_mean.groupby(["phoneme_pair", "word_end"]):
    tc = grp.groupby(["smin", "smax"])["diff"].median().reset_index()
    tc["window_center_s"] = (tc["smin"] + window_size / 2) / sfreq + epoch_tmin
    ax.plot(tc["window_center_s"], tc["diff"], label=f"{phoneme_pair} / {word_end}")

ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlabel("Window center (s from onset)")
ax.set_ylabel("Median Δ ROC-AUC across electrodes")
ax.set_title(f"{subject} — Behavioral Δ time-course")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary table

# %%
peaks.groupby("word_end")[["full_roc_auc", "diff", "window_center_s"]].describe()
