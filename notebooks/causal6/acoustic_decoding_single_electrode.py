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
# causal6: Acoustic decoding from single electrodes — GPU-batched searchlight.
#
# Replaces causal5's `acoustic_decoding_single_electrode`. Drops PCA and the
# C/PCA grid search in favor of a fixed L2 penalty (`reg_lambda_acoustic`,
# tuned once on the `tuning_subject`) and outer StratifiedKFold CV.
#
# Outputs (all parquet, long format):
#   scores.parquet        — per-fold AUC + metadata per (electrode, phoneme_pair, window, target)
#   predictions.parquet   — held-out predictions, one row per (decoder key × test epoch)
#   coefficients.parquet  — fitted β per (decoder key × fold) in standardized space

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %%
from pathlib import Path
import re

import mne
import pandas as pd
import polars as pl
import torch

# %%
from src.data import add_metadata_features
from src.models.causal6 import make_windows, run_acoustic_searchlight

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
outdir = "."

min_sample = 1
window_size = 15
stride = 2

reg_lambda = 1.0  # populated from config.yaml causal6.reg_lambda_acoustic
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

# %% [markdown]
# The per-target parquets are concatenated; a
# `target` column distinguishes them, matching the causal5 `all_outcomes.parquet`
# `measure` convention.

# %%
targets = ["categorical_acoustic_cue"]

all_scores, all_preds, all_coefs = [], [], []
for target in targets:
    scores, preds, coefs = run_acoustic_searchlight(
        epochs, subject=subject,
        electrode_idxs=speech_responsive_idxs,
        windows=windows,
        reg_lambda=reg_lambda,
        target=target,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
        device=device,
        dtype=torch.float32,
        tol=tol, max_iter=max_iter,
    )

    # Invariant: for categorical_acoustic_cue, every phoneme_pair the subject
    # has trials for should get decoded. The searchlight filters to resampled
    # ∈ {1, 6} and those endpoints are deterministically labeled ∓1 / +1, so
    # imbalance = one endpoint missing from the subject's data. Fail loudly so
    # we notice before the skip propagates to the peaks summary. (Skips are
    # expected for subject_specific_acoustics — behaviorally-set zero-crossings
    # can push all trials into one class — so the check is target-scoped.)
    if target == "categorical_acoustic_cue":
        expected_pairs = set(epochs.metadata.phoneme_pair.dropna().unique())
        fit_pairs = set(scores.get_column("phoneme_pair").unique().to_list())
        missing = expected_pairs - fit_pairs
        assert not missing, (
            f"[{subject}] acoustic({target}) skipped phoneme_pairs {sorted(missing)}; "
            "check preceding class-balance warnings — a resampled endpoint (1 or 6) "
            "is likely missing for those pairs in this subject's data."
        )

    all_scores.append(scores)
    all_preds.append(preds)
    all_coefs.append(coefs)

scores_df = pl.concat(all_scores)
predictions_df = pl.concat(all_preds)
coefficients_df = pl.concat(all_coefs)

# %%
outdir = Path(outdir)
scores_df.write_parquet(outdir / "scores.parquet")
predictions_df.write_parquet(outdir / "predictions.parquet")
coefficients_df.write_parquet(outdir / "coefficients.parquet")
print(f"Wrote {scores_df.height} scores, {predictions_df.height} predictions, "
      f"{coefficients_df.height} coefficient rows to {outdir}")
