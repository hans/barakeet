# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Permutation-based null distribution for behavior_decoding_single_electrode.
#
# For each A-electrode decoder, re-run decoding K times with shuffled behavioral
# labels (`randomize=True`) while keeping the hyperparameters (C, n_components)
# fixed to those found in the true model fit. This avoids re-running the expensive
# inner grid search for every permutation.
#
# Outputs a parquet of all permuted fold-level results (with `permutation_idx`
# column) for downstream use in behavior_decoding_single_electrode_permutation_test.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"
os.environ["NUMEXPR_MAX_THREADS"] = "5"

# %%
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import add_metadata_features
from src.models.decoding import run_decoding_model_comparison_population

# %% tags=["parameters"]
subject = "EC260"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
all_A_result_path = f"outputs/causal4/find_As/{subject}_results.csv"
true_results_path = f"outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt"
n_permutations = 10
outdir = "."
min_sample = 1
window_size = 15
stride = 2

# %%
subject = re.findall(r"(EC[\d]+)_epo", str(epochs_path))[0]
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load data

# %%
epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)

max_sample = epochs.times.shape[0]

A_results = pd.read_csv(all_A_result_path).query("A and subject == @subject")

# Load true decoding results to extract stored best hyperparameters
true_results = torch.load(true_results_path)

# %% [markdown]
# ## Run permutations
#
# For each A-electrode / phoneme-pair combination:
#   - Look up the per-(window, fold) best hyperparameters from the true fit
#   - Run `n_permutations` decoding passes with `randomize=True`, using those
#     fixed hparams instead of re-searching the grid
#
# Each permutation uses a distinct numpy seed so results are reproducible.

# %%
all_perm_dfs = []

for k in tqdm(range(n_permutations), desc="Permutation"):
    np.random.seed(k)

    perm_rows = []
    for row in A_results.itertuples():
        outer_key = (row.subject, row.electrode_idx, row.phoneme_pair)

        # Retrieve the true-fit results DataFrame for this electrode/phoneme_pair.
        # Keys in A_decoding_results are (subject, electrode_idx, phoneme_pair).
        true_df = true_results["A_decoding_results"].get(outer_key)
        if true_df is None or len(true_df) == 0:
            continue

        df = run_decoding_model_comparison_population(
            epochs,
            [row.electrode_idx],
            phoneme_pair=row.phoneme_pair,
            subject=row.subject,
            population_name=str(row.electrode_idx),
            global_min_sample=min_sample,
            global_max_sample=max_sample,
            stride=stride,
            window_size=window_size,

            # Non-None value is required so fit_train_test builds a PCA pipeline step;
            # the actual n_components per fold comes from fixed_hparams_df, not this value.
            pca_num_components=0.5,
            
            target="behavior_categorical",
            baseline_features=["resampled"],
            strategy="train-test",
            groupby=["word_end"],
            return_estimators=False,
            n_jobs=5,
            randomize=True,
            fixed_hparams_df=true_df,
        )
        perm_rows.append(df)

    if not perm_rows:
        continue

    perm_df = pd.concat(perm_rows, ignore_index=True)
    perm_df["permutation_idx"] = k
    all_perm_dfs.append(perm_df)

permutation_results = pd.concat(all_perm_dfs, ignore_index=True)

# %% [markdown]
# ## Save

# %%
permutation_results.to_parquet(outdir / "permutation_results.parquet", index=False)
