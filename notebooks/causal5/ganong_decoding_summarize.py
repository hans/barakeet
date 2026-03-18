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
# Summarize Ganong-effect decoding results across all subjects — causal5 pipeline.
#
# Loads `results.joblib` from `ganong_decoding_single_electrode` for every subject,
# finds the peak decoding window per (subject, electrode, phoneme_pair) site,
# and saves combined trial-level predictions and peak-window summaries.
#
# We restrict our peak search to a fixed delta after the latest possible word offset
# for each phoneme pair, configurable with the parameter `behav_peak_post_offset_s`.
#
# Key outputs:
#   - `ganong_predictions.parquet` — trial-level predictions across all windows, all subjects
#   - `ganong_peaks.parquet`       — peak full_roc_auc window per site, all subjects

# %%
from pathlib import Path

import joblib
import pandas as pd
from tqdm.auto import tqdm

from src.stimuli import OFFSET_DICT, WORD_END_TO_PHONEME_PAIR

# %% tags=["parameters"]
# List of paths to `results.joblib` files from ganong_decoding_single_electrode,
# one per subject.
result_paths = list(
    Path("outputs/causal5/ganong_decoding").glob("*/results.joblib")
)

epoch_tmin = -0.4
epoch_sfreq = 100
behav_peak_post_offset_s = 0.2

outdir = "outputs/causal5/ganong_decoding"

# %% [markdown]
# ## Load and combine results across subjects

# %%
all_results_dfs = []
all_decoder_predictions = []

for result_path in tqdm(result_paths):
    ganong_result = joblib.load(result_path)
    A_decoding_results = ganong_result["decoding_results"]
    A_decoders = ganong_result["decoders"]

    # --- Results DataFrame ---
    if len(A_decoding_results) > 0:
        results_df = pd.concat(
            A_decoding_results,
            names=["subject", "population_name", "phoneme_pair"],
            ignore_index=True,
        )
        results_df["diff"] = results_df["full_roc_auc"] - results_df["baseline_roc_auc"]
        all_results_dfs.append(results_df)

    # --- Trial-level predictions ---
    # Key structure: (subject, electrode_idx, phoneme_pair, name, smin, smax, fold)
    # where name = () since groupby=None.
    for (subj, electrode_idx, phoneme_pair), decoders_i in A_decoders.items():
        for (
            _,
            _,
            _,
            name,   # () — empty tuple since groupby=None
            smin,
            smax,
            fold,
        ), dec_detail in decoders_i.items():
            if "test_predictions" not in dec_detail:
                raise ValueError(
                    f"Unexpected decoder format for {subj} electrode {electrode_idx}: "
                    "missing test_predictions"
                )
            all_decoder_predictions.append(
                dec_detail["test_predictions"].assign(
                    subject=subj,
                    electrode_idx=int(electrode_idx),
                    phoneme_pair=phoneme_pair,
                    smin=smin,
                    smax=smax,
                    fold=fold,
                )
            )

# %% [markdown]
# ## Find peak decoding window per site

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
    "smin",
    "smax",
]

if all_results_dfs:
    A_results_df = pd.concat(all_results_dfs, ignore_index=True)

    # Build max word-end offset (in samples) per phoneme_pair.
    # Ganong decoder pools across completions, so use the later offset.
    _pp_max_offset = (
        pd.DataFrame({
            "word_end": list(OFFSET_DICT.keys()),
            "offset_s": list(OFFSET_DICT.values()),
        })
        .assign(phoneme_pair=lambda df: df["word_end"].map(WORD_END_TO_PHONEME_PAIR))
        .groupby("phoneme_pair")["offset_s"].max()
        .rename("max_word_end_offset_s")
    )
    A_results_df = A_results_df.join(_pp_max_offset, on="phoneme_pair")
    smax_limit = (
        (A_results_df["max_word_end_offset_s"] - epoch_tmin) * epoch_sfreq
        + behav_peak_post_offset_s * epoch_sfreq
    )
    A_results_df = A_results_df[A_results_df["smax"] <= smax_limit]

    # Group without word_end — Ganong decoder pools across completions
    A_summary = A_results_df.groupby(
        ["subject", "population", "phoneme_pair", "smin", "smax"]
    )[["baseline_roc_auc", "full_roc_auc", "diff"]].mean()
    A_max_points = A_summary.groupby(
        ["subject", "population", "phoneme_pair"]
    )["diff"].idxmax()
    A_final_summary = A_summary.loc[A_max_points]
    A_final_summary["electrode_idx"] = A_final_summary.index.get_level_values(
        "population"
    ).astype(int)
else:
    A_final_summary = pd.DataFrame()

if all_decoder_predictions:
    ganong_predictions = pd.concat(all_decoder_predictions, ignore_index=True)
else:
    ganong_predictions = pd.DataFrame(columns=pred_columns)

# %% [markdown]
# ## Save outputs

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

A_final_summary.reset_index().to_parquet(outdir / "ganong_peaks.parquet")
ganong_predictions.to_parquet(outdir / "ganong_predictions.parquet")
