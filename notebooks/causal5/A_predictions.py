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
# Collect phonetic-searchlight predictions from `behavior_decoding_single_electrode_acoustic`
# — causal5 simplified pipeline.
#
# The acoustic searchlight fits acoustic-category decoders across all electrodes × time
# windows. Here we extract the predictions on *all* relevant epochs (not just held-out
# test folds) for use in `prepare_neurometrics`.
#
# Output:
#   - `behavior_to_phonetic_decoding.parquet` — phonetic predictions from the acoustic
#     searchlight, used by `prepare_neurometrics` as `phon_predictions_path`.
#
# Note: in causal4 this notebook also produced `phonetic_decoding.parquet` and
# `phonetic_summary.parquet` from the find_As step. Those are no longer needed.

# %%
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

tqdm.pandas()

# %% tags=["parameters"]
# List of paths to `all_outcomes.parquet` files from behavior_decoding_single_electrode_acoustic,
# one per subject.
behav_p_searchlight_paths = list(
    Path("outputs/causal5/behavior_decoding_single_electrode_acoustic").rglob(
        "*/all_outcomes.parquet"
    )
)

outdir = "."

# %% [markdown]
# ## Load acoustic searchlight predictions
#
# `all_outcomes.parquet` contains predictions for *all* relevant epochs under each
# decoder (e.g. all /p/ vs /b/ epochs for a p/b decoder), including multiple measures.
# We keep only the `categorical_acoustic_cue` measure — the fine-grained acoustic
# category target — which is what `prepare_neurometrics` uses as `phon_predictions_path`.

# %%
# all_outcomes.parquet schema:
#   subject, electrode_idx, phoneme_pair, smin, smax, measure,
#   epoch_idx, fold, decoder_target, decoder_proba, decoder_prediction
behav_p_index_names = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "measure"]

behav_p_pred_df = pd.concat(
    [
        pd.read_parquet(path).query("measure == 'categorical_acoustic_cue'")
        for path in tqdm(behav_p_searchlight_paths)
    ],
    ignore_index=True,
).drop(columns=["measure"])

# %%
behav_p_pred_df.to_parquet(Path(outdir) / "behavior_to_phonetic_decoding.parquet")
