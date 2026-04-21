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
# Behavioral decoding from single electrodes — causal5 HGA-only variant.
#
# Mirrors `behavior_decoding_single_electrode.py` exactly except that the decoder
# is trained on windowed HGA alone (no `resampled` morph-step control predictor).
# Dropping the stimulus-step prior prevents AUC from saturating at ceiling on
# unambiguous trials, making ambig vs. unambig AUC fairly comparable within an
# electrode in downstream analyses.
#
# For each (electrode × phoneme_pair) site a single decoder is fit:
#   - logistic regression on windowed HGA (no metadata features)
#
# Outputs `results.joblib` with the same structure as the paired comparison
# notebook, but the baseline_* entries are NaN (skipped internally).

# %%
import os

import joblib

os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL (Intel Math Kernel Library)
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limit OpenBLAS
os.environ["NUMEXPR_MAX_THREADS"] = "1"  # Limit NumExpr if installed

# %%
import itertools
import re

import mne
import pandas as pd
from tqdm.auto import tqdm

# %%
from src.data import add_metadata_features
from src.models.decoding import run_decoding_model_comparison_population

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
outdir = "."

min_sample = 1
window_size = 15
stride = 2
n_jobs = 5

# %%
subject = re.findall(r"(EC[\d]+)_epo", str(epochs_path))[0]

# %%
electrode_df = pd.read_csv(electrodes_path)
speech_responsive_idxs = sorted(
    electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"].unique()
)

# %%
epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
max_sample = epochs.times.shape[0]
phoneme_pairs = sorted(epochs.metadata.phoneme_pair.dropna().unique())

# %% [markdown]
# ## Decode from all speech-responsive electrodes (HGA-only)

# %%
A_decoding_results, A_decoders = {}, {}
for electrode_idx, phoneme_pair in tqdm(
    list(itertools.product(speech_responsive_idxs, phoneme_pairs))
):
    key = (subject, electrode_idx, phoneme_pair)
    A_decoding_results[key], A_decoders[key] = run_decoding_model_comparison_population(
        epochs,
        [electrode_idx],
        phoneme_pair=phoneme_pair,
        subject=subject,
        population_name=str(electrode_idx),
        global_min_sample=min_sample,
        global_max_sample=max_sample,
        stride=stride,
        window_size=window_size,
        pca_num_components=[0.1, 0.25, 0.5, 0.75, 0.9],
        target="behavior_categorical_forced",
        baseline_features=[],
        strategy="train-test",
        groupby=["word_end"],
        return_estimators=True,
        n_jobs=n_jobs,
        min_samples_per_class=3,
    )

# %% [markdown]
# ## Save

# %%
joblib.dump(
    {
        "decoding_results": A_decoding_results,
        "decoders": A_decoders,
    },
    f"{outdir}/results.joblib",
)
