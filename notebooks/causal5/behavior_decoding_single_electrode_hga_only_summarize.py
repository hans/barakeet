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
# Summarize HGA-only behavior decoding results for a single subject — causal5 pipeline.
#
# Mirrors `behavior_decoding_single_electrode_summarize.py` with three deliberate
# deviations since the HGA-only fit has no baseline to compare against:
#   - peak-finding uses max `full_roc_auc` instead of max `diff = full − baseline`
#   - predictions parquet emits a clean schema (`decoder_proba`, `decoder_prediction`);
#     baseline columns are dropped and `full_*` are renamed
#   - the `ensemble_trial_predictions` block (which depends on baseline-vs-full
#     prediction flips) is removed
#
# Key output (consumed by ganong_decoding_hga_only and downstream analyses):
#   - `A-predictions.parquet` — trial-level predictions at the late/perceptual peak window

# %%
import re
from pathlib import Path

import joblib
import mne
import pandas as pd

# %%
from src.data import add_metadata_features
from src.stimuli import OFFSET_DICT, WORD_END_TO_PHONEME_PAIR

# %% tags=["parameters"]
subject = "EC243"

result_path = (
    f"outputs/causal5/behavior_decoding_single_electrode_hga_only/{subject}/results.joblib"
)
groupby = ["word_end"]

electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"

min_decoding_sample = 0
max_decoding_sample = 290  # ~2.5 s post onset

epoch_tmin = -0.4
epoch_sfreq = 100
behav_peak_post_offset_s = 0.2

outdir = "."

# %%
subject = re.findall(
    r"/behavior_decoding_single_electrode_hga_only/([^/]+)/", result_path
)[0]

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
    "full_roc_auc",
    "full_precision",
    "full_recall",
    "word_end",
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
# Filter windows to those ending before word offset + post-offset allowance
_word_end_offset_samples = {
    we: (offset_s - epoch_tmin) * epoch_sfreq
    for we, offset_s in OFFSET_DICT.items()
}
A_results_df["_smax_limit"] = (
    A_results_df["word_end"].map(_word_end_offset_samples)
    + behav_peak_post_offset_s * epoch_sfreq
)
A_results_df = A_results_df[A_results_df["smax"] <= A_results_df["_smax_limit"]].drop(
    columns=["_smax_limit"]
)

# %% [markdown]
# ## Find peak decoding window per site
#
# Peak criterion: max `full_roc_auc` (averaged over folds). No baseline comparison,
# since the HGA-only decoder has no baseline.

# %%
A_summary = A_results_df.groupby(
    ["subject", "population", "phoneme_pair", "smin", "smax"] + groupby
)[["full_roc_auc"]].mean()
A_max_points = A_summary.groupby(["subject", "population", "phoneme_pair"] + groupby)[
    "full_roc_auc"
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
#
# Clean schema: only `decoder_target`, `decoder_proba`, `decoder_prediction`
# plus site keys. Baseline columns (NaN) are dropped; `full_*` renamed to `decoder_*`.

# %%
pred_columns = [
    "decoder_target",
    "decoder_prediction",
    "decoder_proba",
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

    # Clean schema: drop NaN baseline columns, rename full_* → decoder_*
    A_decoder_predictions = A_decoder_predictions.drop(
        columns=["baseline_decoder_prediction", "baseline_decoder_proba"]
    ).rename(
        columns={
            "full_decoder_prediction": "decoder_prediction",
            "full_decoder_proba": "decoder_proba",
        }
    )
else:
    A_decoder_predictions = pd.DataFrame(columns=pred_columns)

# %% [markdown]
# ## Save prediction parquets

# %%
A_decoder_predictions.to_parquet(Path(outdir) / "A-predictions.parquet")
