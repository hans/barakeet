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
# causal6: Behavior decoding from single electrodes — HGA-only variant (GPU-batched).
#
# Replaces causal5's `behavior_decoding_single_electrode_hga_only`. Single model
# per (electrode × phoneme_pair × word_end × window) trained on the windowed HGA
# alone (no `resampled` control predictor). Uses fixed
# `reg_lambda_behavior_hga_only`, tuned independently because the HGA-only
# target has different SNR than the control-augmented version.
#
# Outputs (all parquet, long format with model="full" rows only):
#   scores.parquet, predictions.parquet, coefficients.parquet

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
from src.models.causal6 import make_windows, run_behavior_hga_only

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
outdir = "."

min_sample = 1
window_size = 15
stride = 2

reg_lambda = 1.0  # populated from config.yaml causal6.reg_lambda_behavior_hga_only
n_folds = 5
cv_random_state = 42
device = "cuda"
tol = 1e-6
max_iter = 50

# %%
subject = re.findall(r"(EC[\d]+)_epo", str(epochs_path))[0]

# %%
electrode_df = pd.read_csv(electrodes_path)
speech_responsive_idxs = sorted(
    electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"].unique().astype(int)
)

# %%
epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
max_sample = epochs.times.shape[0]
windows = make_windows(min_sample, max_sample, window_size, stride)

# %%
scores_df, predictions_df, coefficients_df = run_behavior_hga_only(
    epochs, subject=subject,
    electrode_idxs=speech_responsive_idxs,
    windows=windows,
    reg_lambda=reg_lambda,
    n_folds=n_folds,
    cv_random_state=cv_random_state,
    device=device,
    dtype=torch.float32,
    tol=tol, max_iter=max_iter,
)

# %%
outdir = Path(outdir)
scores_df.write_parquet(outdir / "scores.parquet")
predictions_df.write_parquet(outdir / "predictions.parquet")
coefficients_df.write_parquet(outdir / "coefficients.parquet")
print(f"Wrote {scores_df.height} scores, {predictions_df.height} predictions, "
      f"{coefficients_df.height} coefficient rows to {outdir}")
