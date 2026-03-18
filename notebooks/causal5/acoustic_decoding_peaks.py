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
# Compute phonetic decoder peak windows from the acoustic searchlight.
#
# Extracted from prepare_neurometrics so that downstream analyses requiring
# only peak window information (e.g. acoustic_morphology_on_ambiguous) can
# run immediately after A_predictions without waiting for the full
# prepare_neurometrics pipeline.
#
# Outputs:
#   phon_peaks_df.parquet          — peak acoustic window per (subject, electrode, phoneme_pair)
#   phon_roc_auc_searchlight_df.parquet — fold-level ROC-AUC at every window

# %%
from pathlib import Path

import polars as pl

from src.stimuli import (
    OFFSET_DICT,
    POD_dict,
    WORD_END_TO_PHONEME_PAIR,
)
from src.viz_paper import (
    phoneme_pair_enum,
    pl_roc_auc,
    subject_enum,
    word_end_enum,
)

# %% tags=["parameters"]
all_outcomes_paths = list(
    Path("outputs/causal5/acoustic_decoding_single_electrode").rglob("*/all_outcomes.parquet")
)

epoch_tmin = -0.4
epoch_sfreq = 100

phon_response_tmin_min = 0.0
all_response_tmax_max = 1.3

phon_response_peak_threshold = 0.65

outdir = "outputs/causal5/acoustic_decoding_peaks"

# %%
phon_response_smin_min = (phon_response_tmin_min - epoch_tmin) * epoch_sfreq
all_response_smax_max = int((all_response_tmax_max - epoch_tmin) * epoch_sfreq)

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## word_end_df — timing constants

# %%
import pandas as pd

word_end_df = pl.from_pandas(
    pd.DataFrame.from_dict(OFFSET_DICT, orient="index", columns=["word_end_offset"])
    .rename_axis("word_end")
    .join(
        pd.DataFrame.from_dict(
            WORD_END_TO_PHONEME_PAIR, orient="index", columns=["phoneme_pair"]
        ).rename_axis("word_end"),
        on="word_end",
    )
    .reset_index()
    .join(
        pd.DataFrame.from_dict(POD_dict, orient="index", columns=["pod"]).rename_axis(
            "phoneme_pair"
        ),
        on="phoneme_pair",
    )
    .reset_index()
).with_columns(
    pl.col("word_end").cast(word_end_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    ((pl.col("word_end_offset") - epoch_tmin) * epoch_sfreq).alias(
        "word_end_offset_sample"
    ),
    ((pl.col("pod") - epoch_tmin) * epoch_sfreq).alias("pod_sample"),
)

# %% [markdown]
# ## Phonetic searchlight ROC-AUC

# %%
phon_pred_df = pl.concat(
    [
        pl.read_parquet(p).filter(pl.col("measure") == "categorical_acoustic_cue").drop("measure")
        for p in all_outcomes_paths
    ]
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    (pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"),
)

# %%
group_cols = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold"]
phon_roc_auc_searchlight_df = pl_roc_auc(
    df=phon_pred_df.filter(
        (pl.col("smin") >= phon_response_smin_min)
        & (pl.col("smax") <= all_response_smax_max)
    ),
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=group_cols,
    roc_auc_name="phon_roc_auc",
)

# %%
phon_roc_auc_mean_df = phon_roc_auc_searchlight_df.group_by(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
).agg(pl.col("phon_roc_auc").mean())

# %% [markdown]
# ## Peak window per site
#
# For each (subject, electrode_idx, phoneme_pair), find the window with the highest
# mean phonetic ROC-AUC, constrained to:
#   - smin >= phon_response_smin_min  (response onset, not pre-stimulus)
#   - smax <= word_end_offset_sample  (before word offset)
#   - phon_roc_auc >= phon_response_peak_threshold  (significant selectivity)

# %%
phon_peaks_df = (
    phon_roc_auc_mean_df.join(
        word_end_df.group_by(["phoneme_pair"]).agg(pl.max("word_end_offset_sample")),
        on=["phoneme_pair"],
        how="left",
    )
    .filter(
        pl.col("smin") >= phon_response_smin_min,
        pl.col("smax") <= pl.col("word_end_offset_sample"),
        pl.col("phon_roc_auc") >= phon_response_peak_threshold,
    )
    .sort("phon_roc_auc", descending=True)
    .group_by(["subject", "electrode_idx", "phoneme_pair"])
    .first()
)

mean_count = phon_peaks_df.group_by('subject').agg(pl.len()).with_columns(pl.col('len').mean().alias('mean_count')).select('mean_count').to_numpy()[0][0]
print(f"Acoustic sites: {len(phon_peaks_df)}, per subject mean: {mean_count}")
phon_peaks_df.head()

# %% [markdown]
# ## Save outputs

# %%
phon_peaks_df.write_parquet(outdir / "phon_peaks_df.parquet")
phon_roc_auc_searchlight_df.write_parquet(outdir / "phon_roc_auc_searchlight_df.parquet")

print("Done.")
