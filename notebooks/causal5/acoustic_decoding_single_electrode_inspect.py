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
# Inspect acoustic decoding results for a single subject — causal5 pipeline.
#
# Loads `test_scores.parquet` from `acoustic_decoding_single_electrode` and shows:
#   1. Distribution of peak ROC-AUC across sites (max over windows, avg over folds)
#   2. Distribution of peak window center time (seconds from epoch onset)

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %% tags=["parameters"]
subject = "EC243"

scores_path = f"outputs/causal5/acoustic_decoding_single_electrode/{subject}/test_scores.parquet"

# Timing parameters (from config.yaml)
sfreq = 100       # Hz
window_size = 15  # samples
epoch_tmin = -0.5  # seconds; epoch start relative to word onset

peak_threshold = 0.75

# %%
scores = pd.read_parquet(scores_path)
scores.head()

# %% [markdown]
# ## Find peak window per site
#
# For each (electrode_idx, phoneme_pair): average ROC-AUC over folds at each
# window, then take the window with the highest mean.

# %%
mean_scores = (
    scores
    .groupby(["electrode_idx", "phoneme_pair", "smin", "smax"])["roc_auc"]
    .mean()
)

# drop electrode-phoneme pairs that never exceed the threshold, to focus on the responsive ones
mean_scores = (
    mean_scores
    .groupby(["electrode_idx", "phoneme_pair"])
    .filter(lambda x: (x > peak_threshold).any())
    .reset_index()
)

peak_idx = mean_scores.groupby(["electrode_idx", "phoneme_pair"])["roc_auc"].idxmax()
peak = mean_scores.loc[peak_idx].copy()
peak["window_center_s"] = (peak["smin"] + window_size / 2) / sfreq + epoch_tmin

print(f"Sites: {len(peak)}")
peak.describe()

# %% [markdown]
# ## Distributions

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for phoneme_pair, grp in peak.groupby("phoneme_pair"):
    axes[0].hist(grp["roc_auc"], bins=20, alpha=0.6, label=phoneme_pair)
axes[0].axvline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[0].set_xlabel("Peak ROC-AUC")
axes[0].set_ylabel("Sites")
axes[0].set_title(f"{subject} — Acoustic peak performance")
axes[0].legend()

for phoneme_pair, grp in peak.groupby("phoneme_pair"):
    axes[1].hist(grp["window_center_s"], bins=20, alpha=0.6, label=phoneme_pair)
axes[1].set_xlabel("Peak window center (s from onset)")
axes[1].set_ylabel("Sites")
axes[1].set_title(f"{subject} — Acoustic peak time")
axes[1].legend()

# show mean performance time-course for each phoneme pair
for phoneme_pair, grp in mean_scores.groupby("phoneme_pair"):
    # compute mean across electrodes for each time point
    time_course = grp.groupby(["smin", "smax"])["roc_auc"].median().reset_index()
    time_course["window_center_s"] = (time_course["smin"] + window_size / 2) / sfreq + epoch_tmin
    axes[2].plot(time_course["window_center_s"], time_course["roc_auc"], label=phoneme_pair)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary table

# %%
peak.groupby("phoneme_pair")[["roc_auc", "window_center_s"]].describe()
