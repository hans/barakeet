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
# # Population-level gradient perception — causal46_joined edition
#
# Per (subject, phoneme_pair), train logistic+PCA on endpoint trials across the
# AS-restricted population, then apply to:
#   - Held-out endpoint test folds → endpoint_predictions
#   - Ambiguous trials (resampled ∉ {1, 6}) → regression_predictions
#   - Each adjacent step pair (multivariate AX) → multivariate_ax_discrimination
#
# If decoder probabilities track morph step continuously, that's evidence for
# graded population coding. Adapted from
# notebooks/causal5/multivariate_gradient_perception.py with three changes:
#   - Electrode pool from filtered_manifest.csv (any annotated cell qualifies)
#   - Peak window source: causal6 phon_peaks.parquet
#   - phon_response_peak_threshold gate dropped (manifest is the gate)
#
# Completions are pooled — the endpoint training set includes both word_ends;
# the ambiguous predictions table preserves `word_end` for downstream filtering.

# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# %%
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.data import add_metadata_features
from src.models.decoding import _prepare_decoding_population, fit_train_test

# %% tags=["parameters"]
subject = "EC250"

epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/EC250/phon_peaks.parquet"
outdir = "."

pca_num_components = "auto"
n_jobs = 4
num_repeats = 5

epoch_tmin = -0.4
epoch_sfreq = 100

# Acoustic union window floor-extended to at least cover this many seconds
# of post-onset response.
min_window_post_onset_s = 0.3

# Skip (subject, pair) populations with fewer than this many AS sites.
min_population_size = 2

ax_min_per_class = 5

# %%
# Adjacent step pairs for the multivariate AX pass. Hard-coded rather than
# parameterized because papermill flattens nested tuples to strings.
ax_step_pairs = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6))

subject = Path(epochs_path).name.split("_")[0]
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Build site pool

# %%
pool = (
    pd.read_parquet(phon_peaks_path)[["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]]
    .drop_duplicates(subset=["subject", "electrode_idx", "phoneme_pair"])
    .reset_index(drop=True)
)
print(f"{subject}: {len(pool)} (electrode, phoneme_pair) sites in phon_peaks pool")

sites_by_pp = (
    pool.groupby("phoneme_pair", observed=True)["electrode_idx"]
    .apply(list)
    .to_dict()
)
print(f"{subject}: sites per phoneme pair: { {k: len(v) for k, v in sites_by_pp.items()} }")

# %% [markdown]
# ## Load epochs

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.metadata = add_metadata_features(epochs.metadata)

# %% [markdown]
# ## Train + apply decoders per (subject, phoneme_pair)

# %%
all_regression_predictions = []
all_endpoint_predictions = []
all_gradient_stats = []
all_ax_rows = []

onset_sample = int(round(-epoch_tmin * epoch_sfreq))
min_window_post_onset_samples = int(round(min_window_post_onset_s * epoch_sfreq))

for phoneme_pair, electrode_idxs in tqdm(sites_by_pp.items(), desc="Phoneme pairs"):
    if len(electrode_idxs) < min_population_size:
        print(f"  {phoneme_pair}: only {len(electrode_idxs)} site(s) — skipped")
        continue

    pp_peaks = pool[pool["phoneme_pair"] == phoneme_pair]
    acoustic_smin = int(pp_peaks["smin"].min())
    acoustic_smax = int(pp_peaks["smax"].max())
    acoustic_smax = max(acoustic_smax, onset_sample + min_window_post_onset_samples)
    print(f"  {phoneme_pair}: window samples [{acoustic_smin}, {acoustic_smax}), "
          f"electrodes: {len(electrode_idxs)}")

    windows = np.array([[acoustic_smin, acoustic_smax]])

    # --- Endpoint trials: train decoder ---
    endpoint_epochs = epochs[
        epochs.metadata[
            (epochs.metadata.phoneme_pair == phoneme_pair)
            & (epochs.metadata.resampled.isin((1, 6)))
        ].index
    ]
    if len(endpoint_epochs) == 0:
        print(f"  {phoneme_pair}: no endpoint trials — skipped")
        continue

    _, gen = _prepare_decoding_population(
        epochs_i=endpoint_epochs,
        electrode_idxs=electrode_idxs,
        phoneme_pair=phoneme_pair,
        target="acoustic",
        windows=windows,
    )

    for _name, smin, smax, selection, X_window, y_binary in gen:
        resampled_endpoint = (
            endpoint_epochs.metadata.resampled[selection].values.astype(float)
        )

        fitted = fit_train_test(
            X_window,
            y_binary,
            num_classes=2,
            scoring=["roc_auc"],
            pca_num_components=pca_num_components,
            num_repeats=num_repeats,
            n_jobs=n_jobs,
        )

        if fitted is None:
            print(f"  {phoneme_pair}: fit_train_test failed (insufficient data)")
            continue

        mean_auc = float(fitted["test_roc_auc"].mean())

        # --- Held-out endpoint predictions ---
        endpoint_md_selected = endpoint_epochs.metadata[selection]
        for fold_i, (estimator, test_idx) in enumerate(
            zip(fitted["estimator"], fitted["test_idxs"])
        ):
            X_test = X_window[test_idx]
            proba_endpoint = estimator.predict_proba(X_test)[:, 1]
            all_endpoint_predictions.append(pd.DataFrame({
                "subject": subject,
                "phoneme_pair": phoneme_pair,
                "fold": fold_i,
                "epoch_idx": endpoint_md_selected.index.values[test_idx],
                "resampled": resampled_endpoint[test_idx],
                "decoder_proba": proba_endpoint,
            }))

        # --- Ambiguous-trial predictions ---
        ambiguous_epochs = epochs[
            epochs.metadata[
                (epochs.metadata.phoneme_pair == phoneme_pair)
                & (~epochs.metadata.resampled.isin((1, 6)))
            ].index
        ]
        n_ambig = 0
        if len(ambiguous_epochs) > 0:
            X_ambig = ambiguous_epochs.get_data(picks=electrode_idxs)
            X_ambig = X_ambig[:, :, smin:smax].reshape(X_ambig.shape[0], -1)
            ambig_md = ambiguous_epochs.metadata[
                ambiguous_epochs.metadata.phoneme_pair == phoneme_pair
            ]
            n_ambig = len(ambig_md)

            for fold_i, estimator in enumerate(fitted["estimator"]):
                proba = estimator.predict_proba(X_ambig)[:, 1]
                all_regression_predictions.append(pd.DataFrame({
                    "subject": subject,
                    "phoneme_pair": phoneme_pair,
                    "fold": fold_i,
                    "epoch_idx": ambig_md.index.values,
                    "resampled": ambig_md.resampled.values,
                    "word_end": ambig_md.word_end.values,
                    "behavior_categorical_forced": (
                        ambig_md.behavior_categorical_forced.values
                        if "behavior_categorical_forced" in ambig_md.columns
                        else np.nan
                    ),
                    "decoder_proba": proba,
                }))
        else:
            print(f"  {phoneme_pair}: no ambiguous trials")

        all_gradient_stats.append({
            "subject": subject,
            "phoneme_pair": phoneme_pair,
            "n_electrodes": len(electrode_idxs),
            "acoustic_smin": acoustic_smin,
            "acoustic_smax": acoustic_smax,
            "mean_test_roc_auc": mean_auc,
            "n_ambiguous_trials": n_ambig,
            "n_endpoint_trials": int(
                (endpoint_epochs.metadata.phoneme_pair == phoneme_pair).sum()
            ),
        })
        print(f"  {phoneme_pair}: test ROC-AUC={mean_auc:.3f}, n_ambig={n_ambig}")

        # --- Multivariate adjacent-step AX ---
        # Apply the endpoint-trained decoder to each adjacent step pair and
        # compute ROC-AUC — tests whether the population already separates
        # adjacent steps without retraining a classifier per pair.
        pp_mask = (epochs.metadata.phoneme_pair == phoneme_pair).values
        pp_md = epochs.metadata[pp_mask]
        pp_X = epochs.get_data(picks=electrode_idxs)[pp_mask]
        pp_X = pp_X[:, :, smin:smax].reshape(pp_X.shape[0], -1)

        for step_a, step_b in ax_step_pairs:
            mask_a = (pp_md.resampled == step_a).values
            mask_b = (pp_md.resampled == step_b).values
            n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
            if n_a < ax_min_per_class or n_b < ax_min_per_class:
                continue

            mask_ab = mask_a | mask_b
            X_ab = pp_X[mask_ab]
            y_ab = (pp_md.resampled[mask_ab] == step_b).astype(int).values

            fold_aucs = [
                roc_auc_score(y_ab, est.predict_proba(X_ab)[:, 1])
                for est in fitted["estimator"]
            ]
            all_ax_rows.append({
                "subject": subject,
                "phoneme_pair": phoneme_pair,
                "step_a": step_a,
                "step_b": step_b,
                "n_a": n_a,
                "n_b": n_b,
                "roc_auc": float(np.mean(fold_aucs)),
                "roc_auc_std": float(np.std(fold_aucs)),
                "n_electrodes": len(electrode_idxs),
            })

# %% [markdown]
# ## Save outputs

# %%
_endpoint_schema = ["subject", "phoneme_pair", "fold", "epoch_idx",
                    "resampled", "decoder_proba"]
_regression_schema = ["subject", "phoneme_pair", "fold", "epoch_idx",
                      "resampled", "word_end", "behavior_categorical_forced",
                      "decoder_proba"]
_stats_schema = ["subject", "phoneme_pair", "n_electrodes", "acoustic_smin",
                 "acoustic_smax", "mean_test_roc_auc", "n_ambiguous_trials",
                 "n_endpoint_trials"]
_ax_schema = ["subject", "phoneme_pair", "step_a", "step_b", "n_a", "n_b",
              "roc_auc", "roc_auc_std", "n_electrodes"]

endpoint_predictions_df = (
    pd.concat(all_endpoint_predictions, ignore_index=True)
    if all_endpoint_predictions
    else pd.DataFrame(columns=_endpoint_schema)
)
regression_predictions_df = (
    pd.concat(all_regression_predictions, ignore_index=True)
    if all_regression_predictions
    else pd.DataFrame(columns=_regression_schema)
)
gradient_stats_df = (
    pd.DataFrame(all_gradient_stats)
    if all_gradient_stats
    else pd.DataFrame(columns=_stats_schema)
)
multivariate_ax_df = (
    pd.DataFrame(all_ax_rows)
    if all_ax_rows
    else pd.DataFrame(columns=_ax_schema)
)

endpoint_predictions_df.to_parquet(outdir / "endpoint_predictions.parquet", index=False)
regression_predictions_df.to_parquet(outdir / "regression_predictions.parquet", index=False)
gradient_stats_df.to_parquet(outdir / "gradient_stats.parquet", index=False)
multivariate_ax_df.to_parquet(outdir / "multivariate_ax_discrimination_df.parquet", index=False)

print(f"endpoint_predictions: {len(endpoint_predictions_df)} rows")
print(f"regression_predictions: {len(regression_predictions_df)} rows")
print(f"gradient_stats: {len(gradient_stats_df)} rows")
print(f"multivariate_ax_discrimination: {len(multivariate_ax_df)} rows")
