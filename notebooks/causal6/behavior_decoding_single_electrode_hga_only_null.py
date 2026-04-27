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
# causal6: behavior-HGA-only permutation-null refits.
#
# Runs K label-shuffled refits of the HGA-only behavior searchlight.
# Writes `null_scores.parquet` with columns:
#   subject, phoneme_pair, word_end, electrode_idx, smin, smax, fold,
#   permutation_idx, model, test_roc_auc, n_train, n_test.
# Only `model='full'` rows (HGA-only has no baseline).

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %%
from pathlib import Path
import re

import mne
import pandas as pd
import torch

# %%
from src.data import add_metadata_features
from src.models.causal6 import (
    make_windows,
    run_behavior_hga_only_permutations,
)

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
outdir = "."

min_sample = 1
window_size = 15
stride = 2

reg_lambda = 1.0
n_folds = 5
cv_random_state = 42
device = "cuda"
tol = 1e-6
max_iter = 15

n_permutations = 500
permutation_seed = 0
permutation_chunk_size = 6

# %%
subject = re.findall(r"(EC[\d]+)_epo", str(epochs_path))[0]
outdir = Path(outdir)

# %%
electrode_df = pd.read_csv(electrodes_path)
speech_responsive_idxs = sorted(
    electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"].unique().astype(int)
)

epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)

max_sample = epochs.times.shape[0]
windows = make_windows(min_sample, max_sample, window_size, stride)

# %%
permute_seeds = list(range(permutation_seed, permutation_seed + n_permutations))

null_scores = run_behavior_hga_only_permutations(
    epochs, subject=subject,
    electrode_idxs=speech_responsive_idxs,
    windows=windows,
    reg_lambda=reg_lambda,
    permute_seeds=permute_seeds,
    permutation_chunk_size=permutation_chunk_size,
    n_folds=n_folds, cv_random_state=cv_random_state,
    device=device, dtype=torch.float32,
    tol=tol, max_iter=max_iter,
)
assert null_scores.height > 0, (
    f"[{subject}] behavior hga_only null run produced no rows"
)

# %%
null_scores.write_parquet(outdir / "null_scores.parquet")
print(f"Wrote null_scores.parquet ({null_scores.height} rows) to {outdir}")
