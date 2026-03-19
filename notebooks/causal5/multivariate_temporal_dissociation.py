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
# # Sliding-window multivariate temporal dissociation
#
# Decode acoustic identity and perceptual report using a sliding-window
# multivariate decoder across time. Tests whether the double dissociation
# (peak acoustic decoding early, peak perceptual decoding late) holds at
# the population level.
#
# **NOTE:** Electrode selection uses overall responsiveness collapsed across
# time (double-dip). This inflates performance estimates. TODO: revisit with
# held-out electrode selection in a future iteration.

# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# %%
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
import mne

# %%
# %load_ext autoreload
# %autoreload 2
# %%
from src.data import add_metadata_features
from src.models.decoding import (
    _prepare_decoding_population,
    fit_train_test,
)

# %% tags=["parameters"]
subject = "EC250"

epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
phon_peaks_path = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet"
behav_summary_path = f"outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A_final_summary.csv"
outdir = "."

window_sizes = [10]
stride = 2
pca_num_components = "auto"
n_jobs = 4

phon_response_peak_threshold = 0.65
behav_response_peak_threshold = 0.01

epoch_tmin = -0.4
epoch_sfreq = 100

# %%
subject = Path(epochs_path).name.split("_")[0]

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
phon_peaks_df = pd.read_parquet(phon_peaks_path)

# %%
# Load behavioral summary (from behavior_decoding_single_electrode_summarize)
behav_summary = pd.read_csv(behav_summary_path)
# Normalize column names: 'population' → electrode_idx (stored as str)
if "population" in behav_summary.columns:
    behav_summary["electrode_idx"] = behav_summary["population"].astype(int)

# %% [markdown]
# ## Electrode selection
#
# Acoustic target: sites with significant acoustic selectivity (collapsed across time).
# Perceptual target: sites with significant perceptual selectivity (from behavioral
# decoding summarize, filtering by ROC-AUC improvement over baseline).

# %%
# Acoustic-selective electrodes for this subject
acoustic_sites = phon_peaks_df.query(
    "subject == @subject and phon_roc_auc >= @phon_response_peak_threshold"
)
acoustic_sites_by_pp = acoustic_sites.groupby("phoneme_pair")["electrode_idx"].apply(list).to_dict()
print(f"Acoustic sites per phoneme pair: { {k: len(v) for k, v in acoustic_sites_by_pp.items()} }")

# %%
# Perceptual-selective electrodes for this subject (from A_final_summary)
# Filter by improvement (full_roc_auc - baseline_roc_auc) over threshold,
# then take unique electrodes per phoneme_pair
perceptual_sites = behav_summary.query("diff >= @behav_response_peak_threshold")
perceptual_sites_by_pp = (
    perceptual_sites.groupby("phoneme_pair")["electrode_idx"]
    .apply(lambda x: sorted(x.unique().tolist()))
    .to_dict()
)
print(f"Perceptual sites per phoneme pair: { {k: len(v) for k, v in perceptual_sites_by_pp.items()} }")

# %% [markdown]
# ## Sliding-window decoding

# %%
max_sample = epochs.times.shape[0]

all_acoustic_scores = {}
all_perceptual_scores = {}
all_acoustic_outcomes = {}
all_perceptual_outcomes = {}
all_models = {}


# %%
def _run_sliding_window_population(
    epochs_i, electrode_idxs, phoneme_pair, target, window_size,
    groupby=None, filter_str=None,
):
    """Run population decoding across sliding windows for one (subject, phoneme_pair, target)."""
    total, gen = _prepare_decoding_population(
        epochs_i=epochs_i,
        electrode_idxs=electrode_idxs,
        phoneme_pair=phoneme_pair,
        stride=stride,
        window_size=window_size,
        global_min_sample=0,
        global_max_sample=max_sample,
        target=target,
        groupby=groupby,
        filter=filter_str,
    )

    scores = {}
    outcomes = {}
    models = {}

    for name, smin, smax, selection, X_window, y in tqdm(gen, total=total, desc=f"{phoneme_pair}/{target}/ws={window_size}"):
        num_classes = len(set(y))
        if num_classes < 2:
            continue

        scoring = ["roc_auc", "f1_macro", "accuracy"]

        fitted = fit_train_test(
            X_window,
            y,
            num_classes=num_classes,
            pca_num_components=pca_num_components,
            scoring=scoring,
            num_repeats=5,
            n_jobs=n_jobs,
        )

        if fitted is None:
            continue

        result_key = (subject, phoneme_pair, name, smin, smax)

        scores[result_key] = {k: fitted["test_" + k] for k in scoring}

        # Collect test-fold outcomes
        fold_results = []
        for fold, (test_idxs, estimator) in enumerate(
            zip(fitted["test_idxs"], fitted["estimator"])
        ):
            test_epoch_idxs = epochs_i.metadata.index[selection][test_idxs]
            fold_results.append(
                pd.DataFrame({
                    "decoder_target": y[test_idxs],
                    "decoder_prediction": estimator.predict(X_window[test_idxs]),
                    "decoder_proba": estimator.predict_proba(X_window[test_idxs])[:, 1],
                    "fold": fold,
                    "epoch_idx": test_epoch_idxs,
                })
            )

        outcomes[result_key] = pd.concat(fold_results)
        models[result_key] = fitted["estimator"]

    return scores, outcomes, models


# %%
for window_size in window_sizes:
    for phoneme_pair in sorted(set(acoustic_sites_by_pp.keys()) | set(perceptual_sites_by_pp.keys())):
        # --- Acoustic decoding ---
        acoustic_elecs = acoustic_sites_by_pp.get(phoneme_pair, [])
        if len(acoustic_elecs) >= 2:
            # Filter to endpoint trials (steps 1 and 6)
            acoustic_epochs = epochs[
                epochs.metadata[
                    (epochs.metadata.phoneme_pair == phoneme_pair)
                    & (epochs.metadata.resampled.isin((1, 6)))
                ].index
            ]
            scores, outcomes, models = _run_sliding_window_population(
                acoustic_epochs, acoustic_elecs, phoneme_pair,
                target="acoustic", window_size=window_size,
            )
            for k, v in scores.items():
                all_acoustic_scores[(window_size,) + k] = v
            for k, v in outcomes.items():
                all_acoustic_outcomes[(window_size,) + k] = v
            for k, v in models.items():
                all_models[("acoustic", window_size) + k] = v
        else:
            print(f"Skipping acoustic decoding for {phoneme_pair}: only {len(acoustic_elecs)} electrodes")

        # --- Perceptual decoding ---
        perceptual_elecs = perceptual_sites_by_pp.get(phoneme_pair, [])
        if len(perceptual_elecs) >= 2:
            scores, outcomes, models = _run_sliding_window_population(
                epochs, perceptual_elecs, phoneme_pair,
                target="behavior_categorical_forced",
                window_size=window_size,
                groupby=["word_end"],
                filter_str="ambiguity == 'ambiguous'",
            )
            for k, v in scores.items():
                all_perceptual_scores[(window_size,) + k] = v
            for k, v in outcomes.items():
                all_perceptual_outcomes[(window_size,) + k] = v
            for k, v in models.items():
                all_models[("perceptual", window_size) + k] = v
        else:
            print(f"Skipping perceptual decoding for {phoneme_pair}: only {len(perceptual_elecs)} electrodes")

# %% [markdown]
# ## Save outputs

# %%
outdir = Path(outdir)

# %%
if all_acoustic_scores:
    acoustic_scores_df = pd.concat(
        {k: pd.DataFrame(v) for k, v in all_acoustic_scores.items()},
        names=["window_size", "subject", "phoneme_pair", "groupby_name", "smin", "smax", "fold"],
    ).droplevel(-1).reset_index()
    acoustic_scores_df.to_parquet(outdir / "acoustic_scores.parquet")
    print(f"Acoustic scores: {len(acoustic_scores_df)} rows")
else:
    print("No acoustic scores to save")

# %%
if all_perceptual_scores:
    perceptual_scores_df = pd.concat(
        {k: pd.DataFrame(v) for k, v in all_perceptual_scores.items()},
        names=["window_size", "subject", "phoneme_pair", "groupby_name", "smin", "smax", "fold"],
    ).droplevel(-1).reset_index()
    perceptual_scores_df.to_parquet(outdir / "perceptual_scores.parquet")
    print(f"Perceptual scores: {len(perceptual_scores_df)} rows")
else:
    print("No perceptual scores to save")

# %%
if all_acoustic_outcomes:
    acoustic_outcomes_df = pd.concat(
        all_acoustic_outcomes,
        names=["window_size", "subject", "phoneme_pair", "groupby_name", "smin", "smax"],
    ).droplevel(-1).reset_index()
    acoustic_outcomes_df.to_parquet(outdir / "acoustic_outcomes.parquet")

# %%
if all_perceptual_outcomes:
    perceptual_outcomes_df = pd.concat(
        all_perceptual_outcomes,
        names=["window_size", "subject", "phoneme_pair", "groupby_name", "smin", "smax"],
    ).droplevel(-1).reset_index()
    perceptual_outcomes_df.to_parquet(outdir / "perceptual_outcomes.parquet")

# %%
joblib.dump(all_models, outdir / "models.joblib")
