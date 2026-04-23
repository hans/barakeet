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
# Inspect acoustic decoding results for a single subject — causal6 pipeline.
#
# Loads `phon_peaks.parquet` + `phon_roc_auc_searchlight.parquet` from
# `acoustic_decoding_peaks` and shows:
#   1. Distribution of peak ROC-AUC across sites (already mean-over-folds, argmax window)
#   2. Distribution of peak window center time (seconds from epoch onset)
#   3. Median time-course over electrodes per phoneme_pair

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %% tags=["parameters"]
subject = "EC243"

peaks_path = f"outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet"
searchlight_path = f"outputs/causal6/acoustic_decoding_peaks/{subject}/phon_roc_auc_searchlight.parquet"

# Timing parameters (from config.yaml — causal6 uses epoch_tmin = -0.4s)
sfreq = 100       # Hz
window_size = 15  # samples
epoch_tmin = -0.4  # seconds; epoch start relative to word onset

peak_threshold = 0.75

# %%
peaks = pd.read_parquet(peaks_path)
searchlight = pd.read_parquet(searchlight_path)
peaks.head()

# %% [markdown]
# ## Filter to responsive sites
#
# Drop (electrode_idx, phoneme_pair) combos where the searchlight never exceeds
# `peak_threshold`. Mirrors the causal5 convention of focusing inspection on
# responsive sites.

# %%
responsive = (
    searchlight
    .groupby(["electrode_idx", "phoneme_pair"])["test_roc_auc"]
    .max()
    .reset_index()
    .query("test_roc_auc > @peak_threshold")[["electrode_idx", "phoneme_pair"]]
)
peaks = peaks.merge(responsive, on=["electrode_idx", "phoneme_pair"], how="inner")
peaks["window_center_s"] = (peaks["smin"] + window_size / 2) / sfreq + epoch_tmin

print(f"Sites: {len(peaks)}")
peaks.describe()

# %% [markdown]
# ## Distributions

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for phoneme_pair, grp in peaks.groupby("phoneme_pair"):
    axes[0].hist(grp["test_roc_auc"], bins=20, alpha=0.6, label=phoneme_pair)
axes[0].axvline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[0].set_xlabel("Peak ROC-AUC")
axes[0].set_ylabel("Sites")
axes[0].set_title(f"{subject} — Acoustic peak performance")
axes[0].legend()

for phoneme_pair, grp in peaks.groupby("phoneme_pair"):
    axes[1].hist(grp["window_center_s"], bins=20, alpha=0.6, label=phoneme_pair)
axes[1].set_xlabel("Peak window center (s from onset)")
axes[1].set_ylabel("Sites")
axes[1].set_title(f"{subject} — Acoustic peak time")
axes[1].legend()

for phoneme_pair, grp in searchlight.groupby("phoneme_pair"):
    time_course = grp.groupby(["smin", "smax"])["test_roc_auc"].median().reset_index()
    time_course["window_center_s"] = (time_course["smin"] + window_size / 2) / sfreq + epoch_tmin
    axes[2].plot(time_course["window_center_s"], time_course["test_roc_auc"], label=phoneme_pair)
axes[2].axhline(0.5, color="k", linestyle="--", linewidth=0.8)
axes[2].set_xlabel("Window center (s from onset)")
axes[2].set_ylabel("Median ROC-AUC across electrodes")
axes[2].set_title(f"{subject} — Acoustic time-course")
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary table

# %%
peaks.groupby("phoneme_pair")[["test_roc_auc", "window_center_s"]].describe()
