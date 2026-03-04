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
# %load_ext autoreload

# %%
import os
os.environ["OMP_NUM_THREADS"] = "5"  # Limit OpenMP
os.environ["MKL_NUM_THREADS"] = "5"  # Limit MKL (Intel Math Kernel Library)
os.environ["OPENBLAS_NUM_THREADS"] = "5"  # Limit OpenBLAS
os.environ["NUMEXPR_MAX_THREADS"] = "5"  # Limit NumExpr if installed

# %%
from collections import defaultdict
import itertools
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.auto import tqdm

# %%
# %autoreload 2
from src.data import get_electrode_df, add_metadata_features
from src.data_cleaning import prepare_ABC_results, compute_stimulus_correlation
from src.models.decoding import run_decoding_population, run_decoding_model_comparison_population

# %%
sns.set(context="paper", font_scale=2)

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"

electrodes_paths = f"outputs/causal4/find_speech_responsive/{subject}_results.csv"

A_result_path = Path("outputs/causal4/unify_As/results.csv")

all_A_result_path = Path(f"outputs/causal4/find_As/{subject}_results.csv")
all_A_decoders_path = Path(f"outputs/causal4/find_As/{subject}_decoders.pt")

B_annotated_path = Path("outputs/causal4/annotated_B_results.csv")
C_annotated_path = Path("outputs/causal4/annotated_C_results.csv")

outdir = "."

min_sample = 1
window_size = 15
stride = 2

# %%
subject = re.findall(r"(EC[\d]+)_epo", str(epochs_path))[0]

# %%
electrode_df = pd.read_csv(electrodes_paths)

# %%
epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
unified_A_results, B_results, C_results = prepare_ABC_results(A_result_path, B_annotated_path, C_annotated_path)

# %%
B_results = B_results[B_results["subject"] == subject]

# %%
C_results = C_results[C_results["subject"] == subject]

# %%
A_decoders = torch.load(all_A_decoders_path)

# %%
A_results = pd.read_csv(all_A_result_path).query("A and subject == @subject")

# %%
A_results["stimulus_correlation"], A_outcomes = compute_stimulus_correlation(
    A_results,
    {subject: A_decoders},
    {subject: epochs},
    return_outcomes=True)

# %% [markdown]
# ## Setup

# %%
max_sample = epochs.times.shape[0]

# %% [markdown]
# ## Decode from A-electrodes

# %%
A_decoding_results, A_decoders = {}, {}
for row in tqdm(A_results.itertuples(), total=len(A_results)):
    key = (row.subject, row.electrode_idx, row.phoneme_pair)
    A_decoding_results[key], A_decoders[key] = run_decoding_model_comparison_population(
        epochs,
        [row.electrode_idx],
        phoneme_pair=row.phoneme_pair,
        subject=row.subject,
        population_name=str(row.electrode_idx),
        global_min_sample=min_sample,
        global_max_sample=max_sample,
        stride=stride,
        window_size=window_size,
        pca_num_components=[0.1, 0.25, 0.5, 0.75, 0.9],
        target="behavior_categorical",
        baseline_features=["resampled"],
        strategy="train-test",
        groupby=["word_end"],
        return_estimators=True,
        n_jobs=5,
    )

# %% [markdown]
# ## Decode from B-electrodes

# %%
# B_decoding_results, B_decoders = {}, {}
# for row in tqdm(B_results.itertuples(), total=len(B_results)):
#     key = (row.subject, row.electrode_idx, row.phoneme_pair)
#     B_decoding_results[key], B_decoders[key] = run_decoding_model_comparison_population(
#         epochs,
#         [row.electrode_idx],
#         phoneme_pair=row.phoneme_pair,
#         subject=row.subject,
#         population_name=str(row.electrode_idx),
#         stride=stride,
#         window_size=window_size,
#         target="behavior_categorical",
#         baseline_features=["resampled"],
#         pca_num_components="auto",
#         strategy="train-test",
#         groupby=["word_end"],
#         return_estimators=True,
#         n_jobs=5,
#     )

# %% [markdown]
# ## Decode from C-electrodes

# %%
# C_decoding_results, C_decoders = {}, {}
# for row in tqdm(C_results.itertuples(), total=len(C_results)):
#     key = (row.subject, row.electrode_idx, row.phoneme_pair)
#     C_decoding_results[key], C_decoders[key] = run_decoding_model_comparison_population(
#         epochs,
#         [row.electrode_idx],
#         phoneme_pair=row.phoneme_pair,
#         subject=row.subject,
#         population_name=str(row.electrode_idx),
#         stride=stride,
#         window_size=window_size,
#         target="behavior_categorical",
#         baseline_features=["resampled"],
#         pca_num_components="auto",
#         strategy="train-test",
#         groupby=["word_end"],
#         return_estimators=True,
#         n_jobs=5,
#     )

# %% [markdown]
# ## Save

# %%
torch.save({"A_decoding_results": A_decoding_results,
            # "B_decoding_results": B_decoding_results,
            "B_decoding_results": {},
            # "C_decoding_results": C_decoding_results,
            "C_decoding_results": {},
            
            "A_decoders": A_decoders,
            # "B_decoders": B_decoders,
            "B_decoders": {},
            # "C_decoders": C_decoders,
            "C_decoders": {},
            },
            f"{outdir}/results.pt")
