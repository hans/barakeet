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
# Phonetic decoding on all speech-responsive electrodes.

# %%
import os

os.environ["OMP_NUM_THREADS"] = "5"  # Limit OpenMP
os.environ["MKL_NUM_THREADS"] = "5"  # Limit MKL (Intel Math Kernel Library)
os.environ["OPENBLAS_NUM_THREADS"] = "5"  # Limit OpenBLAS
os.environ["NUMEXPR_MAX_THREADS"] = "5"  # Limit NumExpr if installed

# %%
from pathlib import Path

import joblib
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

tqdm.pandas()
import mne

# %%
# %load_ext autoreload
# %autoreload 2
# %%
from src.data import add_metadata_features
from src.models.decoding import (
    get_ensemble_predictions,
    run_decoding_searchlight_single_electrode,
)

# %%
sns.set_context("paper", font_scale=2)

# %% tags=["parameters"]
subject = "EC279"

epochs_path = Path(f"outputs/epochs_preprocessed/{subject}_epo.fif")
electrodes_path = Path(f"outputs/causal5/find_speech_responsive/{subject}_results.csv")
outdir = "."

min_sample = 1
window_size = 15
stride = 2

# %%
subject = Path(electrodes_path).name.split("_")[0]

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
electrode_df = pd.read_csv(electrodes_path).set_index("electrode_idx")

# %%
searchlight_electrode_df = electrode_df.query("speech_responsive == True")
searchlight_electrode_df

# %%
max_sample = epochs.times.shape[0]

# %%
train_scores, test_scores, outcomes, models = run_decoding_searchlight_single_electrode(
    epochs={
        subject: epochs[epochs.metadata[epochs.metadata.resampled.isin((1, 6))].index]
    },
    global_min_sample=min_sample,
    global_max_sample=max_sample,
    electrode_df=searchlight_electrode_df,
    target="acoustic",
    smoke_test=True,  # DEV
    strategy="train-test",
    window_size=window_size,
    stride=stride,
)

# %%
scores_df = pd.concat(
    {key: pd.DataFrame(scores_i) for key, scores_i in test_scores.items()},
    names=["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold"],
)
scores_df

# %%
# average over folds/repeats
avg_scores_df = (
    scores_df.groupby(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
    .mean()
    .sort_values("roc_auc", ascending=False)
)
avg_scores_df

# %% [markdown]
# ## Evaluate on all epochs

# %%
all_outcomes = {}
targets = ["categorical_acoustic_cue", "subject_specific_acoustics"]

for target in tqdm(targets):
    for key, models_i in tqdm(models.items(), leave=False):
        # get predictions on all epochs
        epreds = get_ensemble_predictions(
            key, models_i, {subject: epochs}, target=target
        )  # type: ignore

        # Sanity check: for epochs covered in both analyses, we should see the same prediction values
        sanity_check_df = pd.merge(
            epreds,
            outcomes[key].sort_values("epoch_idx"),
            on=["epoch_idx", "fold"],
            how="inner",
        )
        pd.testing.assert_series_equal(
            sanity_check_df.decoder_proba_x,
            sanity_check_df.decoder_proba_y,
            check_names=False,
        )

        output_key = key + (target,)
        all_outcomes[output_key] = epreds

# %%
# model predictions on test folds
pd.concat(
    outcomes, names=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
).droplevel(-1).reset_index().to_parquet(Path(outdir) / "outcomes.parquet")

# model predictions on all relevant epochs for a given decoder (e.g. all p/b epochs for a p/b decoder)
# incorporating multiple measures
pd.concat(
    all_outcomes,
    names=["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "measure"],
).droplevel(-1).reset_index().to_parquet(Path(outdir) / "all_outcomes.parquet")

# %%
pd.concat(
    {k: pd.DataFrame(v) for k, v in train_scores.items()},
    names=["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold"],
).reset_index().to_parquet(Path(outdir) / "train_scores.parquet")

pd.concat(
    {k: pd.DataFrame(v) for k, v in test_scores.items()},
    names=["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold"],
).reset_index().to_parquet(Path(outdir) / "test_scores.parquet")

# %%
avg_scores_df.to_csv(Path(outdir) / "avg_test_scores.csv")

# %%
joblib.dump(models, Path(outdir) / "decoding_models.joblib")
