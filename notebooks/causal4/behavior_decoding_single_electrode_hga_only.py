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
# Simple (HGA-only) behavioral decoder, fit at the peak windows selected by the
# perceptual decoding analysis.
#
# For each (electrode, phoneme_pair, word_end) with a row in `behav_peaks_df`,
# fit a logistic-regression decoder using **only** windowed HGA (no stimulus-step
# control predictor) at the pre-selected (smin, smax). Architecture, CV
# structure, PCA grid, stratification, and target match the existing perceptual
# decoder in every respect except that `baseline_features=[]`.
#
# Emits `predictions.parquet` (per test-fold × trial × site) with a single
# `decoder_proba` column, plus a lightweight `results.csv` with per-fold AUC.

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
import pandas as pd
from tqdm.auto import tqdm

from src.data import add_metadata_features
from src.models.decoding import run_decoding_model_comparison_population

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
behav_peaks_path = "outputs/causal4/prepare_neurometrics/behav_peaks_df.parquet"
outdir = "."

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

# %%
behav_peaks_df = pd.read_parquet(behav_peaks_path).query("subject == @subject")
behav_peaks_df["word_end"] = behav_peaks_df["word_end"].astype(str)
behav_peaks_df["phoneme_pair"] = behav_peaks_df["phoneme_pair"].astype(str)

# %% [markdown]
# ## Fit HGA-only decoder at each peak window
#
# One call per (electrode_idx, phoneme_pair, word_end) — the peak window
# (smin, smax) is site-specific, so we cannot batch word_ends within a single
# call. `filter` restricts epochs to the target word_end; `global_min_sample`,
# `global_max_sample`, `window_size` and `stride` force the searchlight to
# yield exactly one window at (smin, smax). A fresh inner grid search is done
# each time (no `fixed_hparams_df`).

# %%
all_results = []
all_estimators = {}

for row in tqdm(list(behav_peaks_df.itertuples()), desc="sites"):
    # We'll use `run_decoding_model_comparison_population` which
    # is a sliding window analysis. Manufacture the parameters
    # to yield exactly one window
    smin = int(row.smin)
    smax = int(row.smax)
    window_size = smax - smin
    word_end = str(row.word_end)
    phoneme_pair = str(row.phoneme_pair)
    electrode_idx = int(row.electrode_idx)

    df, estimators = run_decoding_model_comparison_population(
        epochs,
        [electrode_idx],
        phoneme_pair=phoneme_pair,
        subject=subject,
        population_name=str(electrode_idx),
        baseline_features=[],
        target="behavior_categorical",
        strategy="train-test",
        groupby=["word_end"],
        filter=f"word_end == '{word_end}'",
        stride=1,
        window_size=window_size,
        global_min_sample=smin,
        global_max_sample=smax,
        pca_num_components=[0.1, 0.25, 0.5, 0.75, 0.9],
        return_estimators=True,
        n_jobs=5,
    )
    if len(df) == 0:
        # All folds failed to fit (e.g. too few trials / class imbalance). Skip.
        continue

    # Sanity: exactly one window should have been estimated per call.
    assert isinstance(df, pd.DataFrame)
    windows_fit = {(r.smin, r.smax) for r in df.itertuples()}
    assert windows_fit == {(smin, smax)}, (
        f"Expected exactly one window ({smin}, {smax}) for "
        f"({electrode_idx}, {phoneme_pair}, {word_end}); got {windows_fit}"
    )

    all_results.append(df)
    all_estimators.update(estimators)

results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# %% [markdown]
# ## Extract per-trial test predictions
#
# Mirror the pattern in `behavior_decoding_single_electrode_summarize.py`:
# each estimator entry carries a `test_predictions` DataFrame keyed by
# (subject, population, phoneme_pair, (word_end,), smin, smax, fold).

# %%
prediction_rows = []
for key, dec_detail in all_estimators.items():
    subject_k, electrode_idx_k, phoneme_pair_k, name_k, smin_k, smax_k, fold_k = key
    word_end_k = name_k[0] if name_k else None
    prediction_rows.append(
        dec_detail["test_predictions"].assign(
            subject=subject_k,
            electrode_idx=int(electrode_idx_k),
            phoneme_pair=phoneme_pair_k,
            word_end=word_end_k,
            smin=smin_k,
            smax=smax_k,
            fold=fold_k,
        )
    )

if prediction_rows:
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    # Drop the (NaN) baseline columns — this decoder has no baseline.
    predictions_df = predictions_df.drop(
        columns=["baseline_decoder_prediction", "baseline_decoder_proba"]
    ).rename(
        columns={
            "full_decoder_prediction": "decoder_prediction",
            "full_decoder_proba": "decoder_proba",
        }
    )
else:
    predictions_df = pd.DataFrame(
        columns=[
            "decoder_target", "decoder_prediction", "decoder_proba",
            "fold", "epoch_idx", "subject", "electrode_idx",
            "phoneme_pair", "word_end", "smin", "smax",
        ]
    )

# %% [markdown]
# ## Save

# %%
predictions_df.to_parquet(outdir / "predictions.parquet", index=False)
results_df.to_csv(outdir / "results.csv", index=False)
