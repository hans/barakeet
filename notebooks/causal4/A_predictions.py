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
# Summarize phonetic decoding results from early A search and later searchlight.

# %%
from pathlib import Path
import re

import mne
import pandas as pd
import torch
from tqdm.auto import tqdm
tqdm.pandas()

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from src.data import add_metadata_features

# %% tags=["parameters"]
all_results = list(Path("outputs/causal4/find_As").glob("*_results.csv"))
all_decoders = list(Path("outputs/causal4/find_As").glob("*_decoders.pt"))

# phonetic decoding searchlight on behav electrodes
behav_p_searchlight_paths = list(Path("outputs/causal4/behavior_decoding_single_electrode_acoustic").rglob("*/results.pt"))

outdir = "."

# %%
behav_p_results = {Path(path).parent.name: torch.load(path) for path in tqdm(behav_p_searchlight_paths)}

# %%
# Load phonetic decoding results estimated on behaviorally relevant sites.
# Here the key `all_outcomes` stores predictions for ALL relevant trials (e.g. for a p/b decoder, all p/b trials)
# NB there are multiple predictions for each epoch, because we have multiple decoders that were estimated on
# different folds
behav_p_index_names = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "measure"]
behav_p_pred_df = pd.concat([
    pd.concat(behav_p_results[subject]["all_outcomes"], names=behav_p_index_names)
        .xs("categorical_acoustic_cue", level="measure")
    for subject in tqdm(behav_p_results.keys())
]).droplevel(-1).reset_index()

# %%
phonetic_results = pd.concat([pd.read_csv(path) for path in all_results])

# %%
phonetic_decoders = {re.findall(r"(EC[\d]+)_decoders.pt", str(path))[0]: torch.load(path)
                     for path in all_decoders}

# %%
phon_site_names = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]

# Compute ensembled predictions
phon_pred_df = pd.concat([
    pd.concat([
        pd.concat(dec["held_out_outcomes"], names=phon_site_names).droplevel(-1),
        pd.concat(dec["outcomes"], names=phon_site_names).droplevel(-1)
    ])
    for subject, dec in phonetic_decoders.items()
], ignore_index=False)

# %%
# Sanity check: one prediction per site + epoch
assert set(phon_pred_df.groupby(phon_site_names + ["epoch_idx", "fold"]).size().agg(["min", "max"])) == {1}

# %%
phon_pred_df.to_parquet(Path(outdir) / "phonetic_decoding.parquet")
phonetic_results.to_parquet(Path(outdir) / "phonetic_summary.parquet")
behav_p_pred_df.to_parquet(Path(outdir) / "behavior_to_phonetic_decoding.parquet")
