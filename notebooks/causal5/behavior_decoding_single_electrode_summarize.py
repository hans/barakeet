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
# Summarize behavior decoding results for a single subject — causal5 simplified pipeline.
#
# Loads `results.joblib` from `behavior_decoding_single_electrode`, partitions the decoding
# results into early (acoustic) and late (perceptual) time windows, finds the peak decoding
# window per site, and saves prediction DataFrames for downstream use by
# `prepare_neurometrics`.
#
# Key outputs (consumed by prepare_neurometrics):
#   - `A-predictions.parquet`       — trial-level predictions at the late/perceptual peak window
#   - `A_early-predictions.parquet` — trial-level predictions at the early/acoustic peak window

# %%
import re
from pathlib import Path

import joblib
import mne
import pandas as pd

# %%
from src.data import add_metadata_features

# %% tags=["parameters"]
subject = "EC243"

result_path = (
    f"outputs/causal5/behavior_decoding_single_electrode/{subject}/results.joblib"
)
groupby = ["word_end"]

electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"

min_decoding_sample = 0
max_decoding_sample = 290  # ~2.5 s post onset

outdir = "."

# %%
subject = re.findall(r"/behavior_decoding_single_electrode/([^/]+)/", result_path)[0]

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
electrode_df = (
    pd.read_csv(electrodes_path)
    .assign(subject=subject)
    .set_index(["subject", "electrode_idx"])
)

# %%
behav_decoder_result = joblib.load(result_path)
A_decoding_results = behav_decoder_result["decoding_results"]
A_decoders = behav_decoder_result["decoders"]

# %% [markdown]
# ## Build results DataFrames

# %%
dec_columns = [
    "subject",
    "population",
    "phoneme_pair",
    "smin",
    "smax",
    "fold",
    "baseline_roc_auc",
    "full_roc_auc",
    "baseline_precision",
    "full_precision",
    "baseline_recall",
    "full_recall",
    "word_end",
    "baseline_clf__C",
    "full_clf__C",
    "full_prep__pca__pca__n_components",
]
if len(A_decoding_results) > 0:
    A_results_df = pd.concat(
        A_decoding_results,
        names=["subject", "population_name", "phoneme_pair"],
        ignore_index=True,
    )
else:
    A_results_df = pd.DataFrame(columns=dec_columns)

# %%
# Sanity: the two windows together cover the full smax range
target_smax = set(
    smax
    for smax in A_results_df.smax.unique()
    if min_decoding_sample <= smax <= max_decoding_sample
)

# %%
A_results_df["diff"] = A_results_df["full_roc_auc"] - A_results_df["baseline_roc_auc"]

# %% [markdown]
# ## Find peak decoding window per site

# %%
A_summary = A_results_df.groupby(
    ["subject", "population", "phoneme_pair", "smin", "smax"] + groupby
)[["baseline_roc_auc", "full_roc_auc", "diff"]].mean()
A_max_points = A_summary.groupby(["subject", "population", "phoneme_pair"] + groupby)[
    "diff"
].idxmax()
A_final_summary = A_summary.loc[A_max_points]
A_final_summary["electrode_idx"] = A_final_summary.index.get_level_values(
    "population"
).astype(int)

# %% [markdown]
# ## Save summary CSVs

# %%
A_results_df.to_csv(Path(outdir) / "A_results.csv")

A_final_summary.to_csv(Path(outdir) / "A_final_summary.csv")

# %% [markdown]
# ## Collect trial-level predictions

# %%
pred_columns = [
    "decoder_target",
    "baseline_decoder_prediction",
    "baseline_decoder_proba",
    "full_decoder_prediction",
    "full_decoder_proba",
    "fold",
    "epoch_idx",
    "subject",
    "electrode_idx",
    "phoneme_pair",
    "word_end",
    "smin",
    "smax",
]

A_decoder_predictions = []
for decs in A_decoders.values():
    for (
        subj,
        electrode_idx,
        phoneme_pair,
        (word_end,),
        smin,
        smax,
        fold,
    ), dec_detail in decs.items():
        if "test_predictions" not in dec_detail:
            raise ValueError("Unexpected decoder format: missing test_predictions")
        A_decoder_predictions.append(
            dec_detail["test_predictions"].assign(
                subject=subj,
                electrode_idx=int(electrode_idx),
                phoneme_pair=phoneme_pair,
                word_end=word_end,
                smin=smin,
                smax=smax,
                fold=fold,
            )
        )

if len(A_decoder_predictions) > 0:
    A_decoder_predictions = pd.concat(A_decoder_predictions, ignore_index=True)

    A_decoder_predictions = A_decoder_predictions[
        (A_decoder_predictions.smin >= min_decoding_sample)
        & (A_decoder_predictions.smin <= max_decoding_sample)
    ]
else:
    A_decoder_predictions = pd.DataFrame(columns=pred_columns)

# %% [markdown]
# ## Save prediction parquets
#
# `A-predictions.parquet` and `A_early-predictions.parquet` are the primary inputs
# consumed by `prepare_neurometrics` via the `A_behav_predictions` and
# `A_early_behav_predictions` parameters.

# %%
A_decoder_predictions.to_parquet(Path(outdir) / "A-predictions.parquet")

# %% [markdown]
# ## Ensembled trial analysis

# %%
decoder_site_key = [
    "subject",
    "electrode_idx",
    "phoneme_pair",
    "word_end",
    "smin",
    "smax",
]


def ensemble_trial_predictions(predictions_df, epochs):
    all_metadata = (
        epochs.metadata.copy()
        .rename_axis("epoch_idx")
        .reset_index()
        .assign(subject=subject)
    )
    ensembled_predictions = predictions_df.groupby(decoder_site_key + ["epoch_idx"])[
        ["full_decoder_proba", "baseline_decoder_proba", "decoder_target"]
    ].mean()
    study_ensemble = pd.merge(
        ensembled_predictions.reset_index(),
        all_metadata[
            ["subject", "epoch_idx", "resampled", "behavior_based_belief_update"]
        ],
        on=["subject", "epoch_idx"],
        how="left",
        validate="m:1",
    )
    study_ensemble["baseline_prediction_new"] = (
        study_ensemble.baseline_decoder_proba
        > study_ensemble.baseline_decoder_proba.mean()
    )
    study_ensemble["full_prediction"] = study_ensemble.full_decoder_proba > 0.5
    study_ensemble["changed"] = (
        study_ensemble.full_prediction != study_ensemble.baseline_prediction_new
    )
    study_ensemble["correct"] = (
        study_ensemble.full_prediction == study_ensemble.decoder_target
    )
    study_ensemble["abs_behavior_based_belief_update"] = (
        study_ensemble.behavior_based_belief_update.abs()
    )
    return study_ensemble


# %%
ensemble_trial_predictions(A_decoder_predictions, epochs).to_csv(
    Path(outdir) / "A-trial_analysis-ensembled.csv"
)
