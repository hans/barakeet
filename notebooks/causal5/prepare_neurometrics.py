# ---
# jupyter:
#   jupytext:
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
# Prepare the PaperData struct for A_neurometrics visualizations.
# Computes all DataFrames, runs extract_hga_windows_df (the slow step),
# derives early_polarity / late_polarity, and saves everything to parquet.

# %%
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger as L
from tqdm.auto import tqdm

tqdm.pandas()

# %%
from src.data import add_metadata_features
from src.stimuli import (
    OFFSET_DICT,
    WORD_END_TO_PHONEME_PAIR,
    POD_dict,
)
from src.viz_paper import (
    PaperData,
    extract_hga_windows_df,
    phoneme_pair_enum,
    pl_roc_auc,
    subject_enum,
    word_end_enum,
)

# %% tags=["parameters"]
all_epochs = list(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))

A_behav_predictions = list(
    Path("outputs/causal5/behavior_decoding_single_electrode_summarize").glob(
        "*/A-predictions.parquet"
    )
)

phon_predictions_path = Path(
    "outputs/causal5/A_predictions/behavior_to_phonetic_decoding.parquet"
)

phon_peaks_path = Path(
    "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet"
)

phon_roc_auc_searchlight_path = Path(
    "outputs/causal5/acoustic_decoding_peaks/phon_roc_auc_searchlight_df.parquet"
)

electrode_paths = list(Path("outputs/causal5/find_speech_responsive/").glob("*.csv"))

ganong_peaks_path = Path("outputs/causal5/ganong_decoding/ganong_peaks.parquet")
ganong_predictions_path = Path("outputs/causal5/ganong_decoding/ganong_predictions.parquet")

epoch_tmin = -0.4
epoch_sfreq = 100

phon_response_tmin_min = 0.0
all_response_tmax_max = 1.3

phon_response_peak_threshold = 0.6
behav_response_peak_threshold = 0.01
ambiguous_response_threshold = 2

hga_window_source = "decoder"

outdir = "outputs/causal5/prepare_neurometrics"

# %%
phon_response_smin_min = (phon_response_tmin_min - epoch_tmin) * epoch_sfreq
all_response_smax_max = int((all_response_tmax_max - epoch_tmin) * epoch_sfreq)

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load raw data

# %%
electrode_df = pl.concat([pl.read_csv(p) for p in electrode_paths]).with_columns(
    pl.col("subject").cast(subject_enum)
)

# %%
epochs = {}
for path in all_epochs:
    subject = re.findall(r"(EC[\d]+)_epo", str(path))[0]
    ep_i = mne.read_epochs(path, verbose=False)
    assert ep_i.metadata is not None
    ep_i.metadata = add_metadata_features(ep_i.metadata)
    epochs[subject] = ep_i

# %%
all_md = pl.from_pandas(
    pd.concat(
        [
            ep.metadata.rename_axis("epoch_idx").assign(subject=subject).reset_index()
            for subject, ep in epochs.items()
        ],
        ignore_index=True,
    ).drop(columns=["TDT Block"])
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)

# %%
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
# ## Prep phonetic
#
# phon_peaks_df and phon_roc_auc_searchlight_df are pre-computed by the
# acoustic_decoding_peaks rule (notebooks/causal5/acoustic_decoding_peaks.py)
# and loaded here to avoid duplication.

# %%
phon_pred_df = pl.read_parquet(phon_predictions_path).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    (pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"),
)

# %%
phon_peaks_df = pl.read_parquet(phon_peaks_path).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)

# %%
phon_roc_auc_searchlight_df = pl.read_parquet(phon_roc_auc_searchlight_path).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)

# %% [markdown]
# ## Behav prep

# %%
behav_pred_df = pl.concat(
    [pl.read_parquet(f) for f in A_behav_predictions]
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
    (pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"),
)

# %%
behav_baseline_df = pl_roc_auc(
    df=behav_pred_df.unique(
        subset=[
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "word_end",
            "epoch_idx",
            "fold",
        ],
        keep="first",
    ),
    target_col="decoder_target",
    proba_col="baseline_decoder_proba",
    group_cols=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
    roc_auc_name="behav_roc_auc_baseline",
)

# %%
group_cols = [
    "subject",
    "electrode_idx",
    "phoneme_pair",
    "word_end",
    "smin",
    "smax",
    "fold",
]
behav_roc_auc_searchlight = pl_roc_auc(
    df=behav_pred_df.filter(pl.col("smax") <= all_response_smax_max),
    target_col="decoder_target",
    proba_col="full_decoder_proba",
    group_cols=group_cols,
    roc_auc_name="behav_roc_auc",
)

# %%
behav_roc_auc_searchlight_df = behav_roc_auc_searchlight.join(
    behav_baseline_df,
    on=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
    how="inner",
).with_columns(
    (pl.col("behav_roc_auc") - pl.col("behav_roc_auc_baseline")).alias(
        "behav_roc_auc_improvement"
    )
)

# %%
behav_roc_auc_mean_df = behav_roc_auc_searchlight_df.group_by(
    ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
).agg(
    [
        pl.col("behav_roc_auc").mean(),
        pl.col("behav_roc_auc_baseline").mean(),
        pl.col("behav_roc_auc_improvement").mean(),
    ]
)

# %%
behav_peaks_df_unfiltered = (
    behav_roc_auc_mean_df.join(
        phon_peaks_df.select(["subject", "electrode_idx", "phoneme_pair", "smax"]),
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="inner",
        suffix="_phon",
    )
    .join(word_end_df, on=["phoneme_pair", "word_end"], how="left")
    .filter(
        pl.col("smin") > pl.col("smax_phon"),
        pl.col("smax") <= pl.col("word_end_offset_sample") + 20,
    )
    .drop(["smax_phon"])
    .sort("behav_roc_auc_improvement", descending=True)
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .first()
)

# %%
behav_peaks_df = behav_peaks_df_unfiltered.filter(
    pl.col("behav_roc_auc_improvement") > behav_response_peak_threshold
)

# %% [markdown]
# ## Build plot DataFrames

# %%
plot_phon_phon_keys = phon_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
)
plot_phon_phon_df = plot_phon_phon_keys.join(
    phon_pred_df,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="left",
).join(all_md, on=["subject", "epoch_idx", "phoneme_pair"], how="left")

# %%
plot_behav_phon_keys = behav_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "word_end"]
)
plot_behav_phon_df = (
    plot_behav_phon_keys.rename({"word_end": "behav_word_end"})
    .join(
        phon_pred_df,
        on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
        how="left",
    )
    .join(
        all_md,
        on=["subject", "epoch_idx", "phoneme_pair"],
        how="left",
    )
    .filter(pl.col("word_end") == pl.col("behav_word_end"))
    .drop("behav_word_end")
)

# %%
plot_phon_behav_keys = phon_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
)
plot_phon_behav_df = plot_phon_behav_keys.join(
    behav_pred_df,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="left",
)

# %%
plot_behav_behav_keys = behav_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
)
plot_behav_behav_df = plot_behav_behav_keys.join(
    behav_pred_df,
    on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
    how="left",
)

# %% [markdown]
# ## Zoomin keys

# %%
zoomin_keys = phon_peaks_df.join(
    behav_peaks_df, on=["subject", "electrode_idx", "phoneme_pair"]
).select(
    [
        "subject",
        "electrode_idx",
        "phoneme_pair",
        "word_end",
        "phon_roc_auc",
        "behav_roc_auc_improvement",
    ]
)

# %% [markdown]
# ## Bootstrap PaperData → compute HGA windows

# %%
# We need a PaperData to call extract_hga_windows_df, but early_polarity/late_polarity
# depend on hga_df which depends on PaperData — bootstrap with None, then replace.
_bootstrap = PaperData(
    electrode_df=electrode_df,
    plot_phon_phon_df=plot_phon_phon_df,
    plot_behav_phon_df=plot_behav_phon_df,
    plot_behav_behav_df=plot_behav_behav_df,
    plot_phon_behav_df=plot_phon_behav_df,
    behav_roc_auc_searchlight_df=behav_roc_auc_searchlight_df,
    phon_roc_auc_searchlight_df=phon_roc_auc_searchlight_df,
    all_md=all_md,
    word_end_df=word_end_df,
    epochs=epochs,
    phon_peaks_df=phon_peaks_df,
    behav_peaks_df=behav_peaks_df,
    behav_peaks_df_unfiltered=behav_peaks_df_unfiltered,
    behav_baseline_df=behav_baseline_df,
    zoomin_keys=zoomin_keys,
    early_polarity=None,  # type: ignore  — filled in below
    late_polarity=None,  # type: ignore  — filled in below
)

# %%
hga_df = extract_hga_windows_df(
    _bootstrap,
    zoomin_keys=zoomin_keys,
    ambiguous_response_threshold=ambiguous_response_threshold,
    window_source=hga_window_source,
)

# %% [markdown]
# ## Compute polarities and reg_df

# %%
# early_polarity: direction of acoustic category effect in the early window.
# Use only unambiguous acoustic-consistent trials (resampled=1 and 6, follows_acoustics=True),
# consistent with how the early window was found (zoomin_hga / find_site_windows uses
# resampled=1 vs 6 acoustic-consistent trials to select the phoneme window).
hga_df_unambig = hga_df[hga_df.resampled.isin([1.0, 6.0]) & hga_df.follows_acoustics]
early_polarity = (
    hga_df_unambig.groupby(
        ["subject", "electrode_idx", "phoneme_pair", "word_end", "decoder_target"]
    )
    .hga_early.mean()
    .reset_index()
    .set_index("decoder_target")
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .apply(lambda xs: np.sign(xs.loc[1] - xs.loc[0]))  # type: ignore[union-attr]
    .rename(columns={"hga_early": "early_polarity"})
)

# %%
# late_polarity: direction of behavioral choice effect in the late window.
# Use only the ambiguous trials that were used to select the late window
# (behav_steps_chosen), so the polarity is purely behavioral (acoustic content
# is constant within those steps) and consistent with how the window was found.
hga_df_ambig = hga_df[
    hga_df.apply(
        lambda xs: (
            xs.behav_steps_chosen != "None"
            and str(int(xs.resampled)) in xs.behav_steps_chosen
        ),
        axis=1,
    )
]
late_polarity = (
    hga_df_ambig.groupby(
        [
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "word_end",
            "behavior_dummy_forced",
        ]
    )
    .hga_late.mean()
    .reset_index()
    .set_index("behavior_dummy_forced")
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .apply(lambda xs: np.sign(xs.loc[1] - xs.loc[0]))  # type: ignore[union-attr]
    .rename(columns={"hga_late": "late_polarity"})
)

# %%
reg_df = pd.merge(
    hga_df,
    pd.merge(
        early_polarity.reset_index(),
        late_polarity.reset_index(),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    ),
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
)
reg_df["hga_early_signed"] = reg_df["hga_early"] * reg_df["early_polarity"]
reg_df["hga_late_signed"] = reg_df["hga_late"] * reg_df["late_polarity"]
reg_df["is_ambiguous"] = reg_df.apply(
    lambda xs: (
        str(int(xs.resampled)) in xs.behav_steps_chosen
        if xs.behav_steps_chosen is not None
        else np.nan
    ),
    axis=1,
)

# %% [markdown]
# ## Construct final PaperData and save outputs

# %%
paper_data = PaperData(
    electrode_df=electrode_df,
    plot_phon_phon_df=plot_phon_phon_df,
    plot_behav_phon_df=plot_behav_phon_df,
    plot_behav_behav_df=plot_behav_behav_df,
    plot_phon_behav_df=plot_phon_behav_df,
    behav_roc_auc_searchlight_df=behav_roc_auc_searchlight_df,
    phon_roc_auc_searchlight_df=phon_roc_auc_searchlight_df,
    all_md=all_md,
    word_end_df=word_end_df,
    epochs=epochs,
    phon_peaks_df=phon_peaks_df,
    behav_peaks_df=behav_peaks_df,
    behav_peaks_df_unfiltered=behav_peaks_df_unfiltered,
    behav_baseline_df=behav_baseline_df,
    zoomin_keys=zoomin_keys,
    early_polarity=early_polarity,
    late_polarity=late_polarity,
    hga_df=hga_df,
    reg_df=reg_df,
)

# %%
# Save Polars DataFrames as parquet
paper_data.electrode_df.write_parquet(outdir / "electrode_df.parquet")
paper_data.plot_phon_phon_df.write_parquet(outdir / "plot_phon_phon_df.parquet")
paper_data.plot_behav_phon_df.write_parquet(outdir / "plot_behav_phon_df.parquet")
paper_data.plot_behav_behav_df.write_parquet(outdir / "plot_behav_behav_df.parquet")
paper_data.plot_phon_behav_df.write_parquet(outdir / "plot_phon_behav_df.parquet")
paper_data.behav_roc_auc_searchlight_df.write_parquet(
    outdir / "behav_roc_auc_searchlight_df.parquet"
)
paper_data.phon_roc_auc_searchlight_df.write_parquet(
    outdir / "phon_roc_auc_searchlight_df.parquet"
)
paper_data.all_md.write_parquet(outdir / "all_md.parquet")
paper_data.word_end_df.write_parquet(outdir / "word_end_df.parquet")
paper_data.phon_peaks_df.write_parquet(outdir / "phon_peaks_df.parquet")
paper_data.behav_peaks_df.write_parquet(outdir / "behav_peaks_df.parquet")
paper_data.behav_peaks_df_unfiltered.write_parquet(
    outdir / "behav_peaks_df_unfiltered.parquet"
)
paper_data.behav_baseline_df.write_parquet(outdir / "behav_baseline_df.parquet")
paper_data.zoomin_keys.write_parquet(outdir / "zoomin_keys.parquet")

# Save pandas DataFrames (multi-indexed) as parquet via reset_index
paper_data.early_polarity.reset_index().to_parquet(outdir / "early_polarity.parquet")
paper_data.late_polarity.reset_index().to_parquet(outdir / "late_polarity.parquet")
paper_data.hga_df.to_parquet(outdir / "hga_df.parquet")
paper_data.reg_df.to_parquet(outdir / "reg_df.parquet")

# %% [markdown]
# ## Ganong decoding outputs
#
# Copy ganong_peaks.parquet and ganong_predictions.parquet into the prepare_neurometrics
# outdir so that A_neurometrics can find them via neurometrics_dir without needing
# separate path parameters. No computation is done here — the files are produced by
# ganong_decoding_summarize.

# %%
import shutil

shutil.copy(ganong_peaks_path, outdir / "ganong_peaks.parquet")
shutil.copy(ganong_predictions_path, outdir / "ganong_predictions.parquet")

L.success(f"Saved all PaperData fields to {outdir}")
