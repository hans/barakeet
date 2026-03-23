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
# # Population-level gradient perception
#
# Ask whether graded acoustic information is recoverable from the population
# of acoustically selective sites, even though individual electrodes appear
# categorical. Train a logistic regression (with PCA preprocessing) on endpoint
# trials to classify acoustic cue (/d/ vs /n/), then apply to ambiguous trials.
# If the decoder probability tracks morph step continuously rather than snapping
# to 0 or 1, that's evidence for graded acoustic population coding.

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
from IPython.display import display

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
    run_ax_discrimination,
)
from src.stimuli import POD_dict

# %% tags=["parameters"]
epochs_paths = [f"outputs/epochs_preprocessed/{s}_epo.fif" for s in ["EC250"]]
phon_peaks_path = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet"
outdir = "."

pca_num_components = "auto"
n_jobs = 4

phon_response_peak_threshold = 0.65

epoch_tmin = -0.4
epoch_sfreq = 100

# %%
phon_peaks_df = pd.read_parquet(phon_peaks_path)

# %%
all_regression_predictions = []
all_endpoint_predictions = []
all_gradient_stats = []
all_ax_rows = []

# %%
for epochs_path in tqdm(epochs_paths, desc="Subjects"):
    subject = Path(epochs_path).name.split("_")[0]

    epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
    epochs.metadata = add_metadata_features(epochs.metadata)

    # Acoustically selective electrodes for this subject
    subject_peaks = phon_peaks_df.query(
        "subject == @subject and phon_roc_auc >= @phon_response_peak_threshold"
    )
    sites_by_pp = subject_peaks.groupby("phoneme_pair", observed=True)["electrode_idx"].apply(list).to_dict()
    print(f"{subject}: acoustic sites per phoneme pair: { {k: len(v) for k, v in sites_by_pp.items()} }")

    for phoneme_pair, electrode_idxs in sites_by_pp.items():
        if len(electrode_idxs) < 2:
            print(f"  Skipping {phoneme_pair}: only {len(electrode_idxs)} electrode(s)")
            continue

        # Determine acoustic response window: onset to ~300ms
        # Use the union of per-site peak windows, bounded by a reasonable max
        pp_peaks = subject_peaks.query("phoneme_pair == @phoneme_pair")
        # Window in samples: from the earliest smin to the latest smax among peak windows
        acoustic_smin = int(pp_peaks["smin"].min())
        acoustic_smax = int(pp_peaks["smax"].max())
        # Ensure we cover at least onset to 300ms post-onset
        onset_sample = int(round(-epoch_tmin * epoch_sfreq))  # sample index of t=0
        min_smax = onset_sample + int(round(0.3 * epoch_sfreq))  # 300ms post-onset
        acoustic_smax = max(acoustic_smax, min_smax)
        print(f"  {phoneme_pair}: acoustic window samples [{acoustic_smin}, {acoustic_smax}), "
              f"electrodes: {len(electrode_idxs)}")

        # Build feature matrix using a single explicit window
        windows = np.array([[acoustic_smin, acoustic_smax]])

        # --- Endpoint trials: train regression ---
        endpoint_epochs = epochs[
            epochs.metadata[
                (epochs.metadata.phoneme_pair == phoneme_pair)
                & (epochs.metadata.resampled.isin((1, 6)))
            ].index
        ]

        total, gen = _prepare_decoding_population(
            epochs_i=endpoint_epochs,
            electrode_idxs=electrode_idxs,
            phoneme_pair=phoneme_pair,
            target="acoustic",
            windows=windows,
        )

        for name, smin, smax, selection, X_window, y_binary in gen:
            # y_binary: categorical_acoustic_cue mapped to {0, 1}
            # Also track continuous morph step for endpoint predictions
            resampled_endpoint = endpoint_epochs.metadata.resampled[selection].values.astype(float)

            fitted = fit_train_test(
                X_window,
                y_binary,
                num_classes=2,
                scoring=["roc_auc"],
                pca_num_components=pca_num_components,
                num_repeats=5,
                n_jobs=n_jobs,
            )

            if fitted is None:
                print(f"  {phoneme_pair}: classification failed (insufficient data)")
                continue

            # --- Endpoint test-fold predictions (held-out) ---
            endpoint_md_selected = endpoint_epochs.metadata[selection]
            for fold_i, (estimator, test_idx) in enumerate(
                zip(fitted["estimator"], fitted["test_idxs"])
            ):
                X_test = X_window[test_idx]
                proba_endpoint = estimator.predict_proba(X_test)[:, 1]
                all_endpoint_predictions.append(
                    pd.DataFrame({
                        "subject": subject,
                        "phoneme_pair": phoneme_pair,
                        "fold": fold_i,
                        "epoch_idx": endpoint_md_selected.index.values[test_idx],
                        "resampled": resampled_endpoint[test_idx],
                        "decoder_proba": proba_endpoint,
                    })
                )

            # --- Apply to ambiguous trials ---
            ambiguous_epochs = epochs[
                epochs.metadata[
                    (epochs.metadata.phoneme_pair == phoneme_pair)
                    & (~epochs.metadata.resampled.isin((1, 6)))
                ].index
            ]

            if len(ambiguous_epochs) == 0:
                print(f"  {phoneme_pair}: no ambiguous trials")
                continue

            X_ambig = ambiguous_epochs.get_data(picks=electrode_idxs)
            X_ambig = X_ambig[:, :, smin:smax].reshape(X_ambig.shape[0], -1)

            ambig_md = ambiguous_epochs.metadata[
                ambiguous_epochs.metadata.phoneme_pair == phoneme_pair
            ]

            for fold, estimator in enumerate(fitted["estimator"]):
                proba = estimator.predict_proba(X_ambig)[:, 1]
                all_regression_predictions.append(
                    pd.DataFrame({
                        "subject": subject,
                        "phoneme_pair": phoneme_pair,
                        "fold": fold,
                        "epoch_idx": ambig_md.index.values,
                        "resampled": ambig_md.resampled.values,
                        "behavior_categorical_forced": ambig_md.behavior_categorical_forced.values
                        if "behavior_categorical_forced" in ambig_md.columns else np.nan,
                        "word_end": ambig_md.word_end.values,
                        "decoder_proba": proba,
                    })
                )

            all_gradient_stats.append({
                "subject": subject,
                "phoneme_pair": phoneme_pair,
                "n_electrodes": len(electrode_idxs),
                "acoustic_smin": acoustic_smin,
                "acoustic_smax": acoustic_smax,
                "mean_test_roc_auc": fitted["test_roc_auc"].mean(),
                "n_ambiguous_trials": len(ambig_md),
                "n_endpoint_trials": len(endpoint_epochs.metadata.query("phoneme_pair == @phoneme_pair")),
            })

            print(f"  {phoneme_pair}: test ROC-AUC={fitted['test_roc_auc'].mean():.3f}")

            # --- Multivariate AX discrimination: adjacent-step population decoders ---
            # Build full data matrix for this phoneme pair once
            pp_mask = epochs.metadata.phoneme_pair == phoneme_pair
            pp_data = epochs.get_data(picks=electrode_idxs)[pp_mask.values]
            pp_data_windowed = pp_data[:, :, smin:smax].reshape(pp_data.shape[0], -1)

            ax_rows = run_ax_discrimination(
                metadata=epochs.metadata[pp_mask],
                get_X=lambda idx: pp_data_windowed[idx],
                phoneme_pair=phoneme_pair,
                fit_kw=dict(pca_num_components=None, n_jobs=n_jobs),
            )
            for row in ax_rows:
                row.update(subject=subject, phoneme_pair=phoneme_pair,
                           n_electrodes=len(electrode_idxs))
            all_ax_rows.extend(ax_rows)

# %% [markdown]
# ## Save outputs

# %%
outdir = Path(outdir)

# %%
if all_regression_predictions:
    regression_predictions_df = pd.concat(all_regression_predictions, ignore_index=True)
    regression_predictions_df.to_parquet(outdir / "regression_predictions.parquet")
    print(f"Regression predictions: {len(regression_predictions_df)} rows")
else:
    print("No regression predictions to save")

# %%
if all_endpoint_predictions:
    endpoint_predictions_df = pd.concat(all_endpoint_predictions, ignore_index=True)
    endpoint_predictions_df.to_parquet(outdir / "endpoint_predictions.parquet")
    print(f"Endpoint predictions: {len(endpoint_predictions_df)} rows")
else:
    print("No endpoint predictions to save")

# %%
if all_gradient_stats:
    gradient_stats_df = pd.DataFrame(all_gradient_stats)
    gradient_stats_df.to_parquet(outdir / "gradient_stats.parquet")
    print(f"Gradient stats: {len(gradient_stats_df)} rows")
    display(gradient_stats_df)
else:
    print("No gradient stats to save")

# %%
if all_ax_rows:
    ax_discrimination_df = pd.DataFrame(all_ax_rows)
    ax_discrimination_df.to_parquet(outdir / "multivariate_ax_discrimination_df.parquet")
    print(f"Multivariate AX discrimination: {len(ax_discrimination_df)} rows")
    display(ax_discrimination_df)
else:
    print("No multivariate AX discrimination results to save")

