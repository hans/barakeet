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

# %%
import re

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from src.data import get_electrode_df, add_metadata_features

# %% tags=["parameters"]
epochs_path = "outputs/epochs_preprocessed/EC260_epo.fif"
outdir = "."

# power threshold relative to pre-speech baseline which defines a "speech responsive" electrode
# if we see absolute value change >= this threshold, call the electrode speech-responsive
speech_responsive_threshold = 0.3

# %%
subject_name = re.findall("(EC[\d]+)_epo", epochs_path)[0]

# %%
epochs_data = mne.read_epochs(epochs_path)

# %%
electrode_df = get_electrode_df(subject_name)
electrode_df["roi"] = electrode_df.roi.astype(str)
electrode_df = electrode_df.droplevel("electrode_name")

# Drop electrodes metadata which don't have corresponding data
electrode_df["keep"] = np.arange(len(electrode_df)) < len(epochs_data.info["ch_names"])
electrode_df = electrode_df[electrode_df["keep"]].drop(columns="keep")

electrode_df

# %%
# demo this
dd = epochs_data.copy().apply_baseline((-0.1, 0)).average().crop(tmin=0, tmax=0.9).get_data()
keep = np.abs(dd).max(axis=1) > speech_responsive_threshold

f, ax = plt.subplots(figsize=(8, 4))
for line, k in zip(dd, keep):
    plt.plot(line, color="r" if k else "k", alpha=0.1)

# %%
epochs_data = epochs_data.copy().apply_baseline((-0.1, 0)).average().crop(tmin=0, tmax=0.9).get_data()
assert epochs_data.ndim == 2
speech_responsive_i = np.abs(epochs_data).max(axis=1) > speech_responsive_threshold

if len(speech_responsive_i) > len(electrode_df):
    speech_responsive_i = speech_responsive_i[:len(electrode_df)]
electrode_df["speech_responsive"] = speech_responsive_i

# %%
electrode_df = electrode_df.astype({"speech_responsive": bool})
electrode_df["subject"] = subject_name

# %%
electrode_df.to_csv(f"{outdir}/{subject_name}_results.csv")
