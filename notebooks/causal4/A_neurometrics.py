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
# Neurometric response functions relating single-neuron activity to both phonetic content and subsequent behavioral choice.
# Compare two neural responses:
# 1. early phonetic response
# 2. late feedback response
# in the same electrodes.

# %%
from pathlib import Path
import re
from typing import Any, TypeAlias, Literal

import joblib
from loguru import logger as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mne
import numpy as np
import pandas as pd
import polars as pl
import polars_ds as pds
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import seaborn as sns
import torch
from tqdm.auto import tqdm
tqdm.pandas()

# %%
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.25,
    "ytick.minor.width": 0.25,
    "lines.linewidth": 1.0,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from src.data import add_metadata_features
from src.stimuli import OFFSET_DICT, POD_dict, WORD_END_TO_PHONEME_PAIR, PHONEME_PAIR_TO_WORD_ENDS
from src.viz import add_textgrid_single
from src.viz_paper import PaperData, subject_enum, phoneme_pair_enum, word_end_enum, \
    zoomin_hga, zoomin_search_hga, find_site_windows, add_textgrid

# %%
sns.set_context("paper", font_scale=1.25)

# %% tags=["parameters"]
all_epochs = list(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))

# Behavioral decoding searchlight
A_behav_predictions = list(Path("outputs/causal4/behavior_decoding_single_electrode_summarize").glob("*/A-predictions.parquet"))
A_early_behav_predictions = list(Path("outputs/causal4/behavior_decoding_single_electrode_summarize").glob("*/A_early-predictions.parquet"))

# phonetic decoding searchlight. NB this is currently only done for behaviorally selective sites
# TODO broaden
phon_predictions_path = Path("outputs/causal4/A_predictions/behavior_to_phonetic_decoding.parquet")

phonetic_searchlight_paths = list(Path(f"outputs/causal4/behavior_decoding_single_electrode_acoustic/").glob("*"))

transfer_results_paths = list(Path("outputs/causal4/behavior_decoding_single_electrode_transfer").glob("*/transfer_results.csv"))

electrode_paths = list(Path("outputs/causal4/find_speech_responsive/").glob("*.csv"))

epoch_tmin = -0.4
epoch_sfreq = 100

# parameters for searching for behavioral peaks
behav_response_tmin_min = 0.3
behav_response_smin_min = (behav_response_tmin_min - epoch_tmin) * epoch_sfreq

# parameters for searching for phonetic peaks
phon_response_tmin_min = 0.0
phon_response_smin_min = (phon_response_tmin_min - epoch_tmin) * epoch_sfreq
# threshold value for significant phonetic decoding peak
phon_response_peak_threshold = 0.64

all_response_tmax_max = 1.3
all_response_smax_max = int((all_response_tmax_max - epoch_tmin) * epoch_sfreq)

relative_performance_twidth = 0.2
relative_performance_swidth = int(relative_performance_twidth * epoch_sfreq)

textgrid_dir = "textgrids"

outdir = "."

max_plot_rows = 15
phoneme_pair_order = ["bm", "dn", "pb"]
source_order = ["phon", "behav"]

# shared palette for categorical variables
categorical_palette = "Set2"

# %%
resampled_palette = sns.color_palette("cool", n_colors=6)

# simplified resampled palette contrasting ambiguous vs unambiguous
resampled_palette_simplified = [resampled_palette[0]] + (4 * [resampled_palette[2]]) + [resampled_palette[5]]

# %% [markdown]
# ## Prepare helpers

# %%
electrode_df = pl.concat([pl.read_csv(p) for p in electrode_paths]) \
    .with_columns(pl.col("subject").cast(subject_enum))

# %%
epochs = {}
for path in all_epochs:
    subject = re.findall(r"(EC[\d]+)_epo", str(path))[0]
    ep_i = mne.read_epochs(path, verbose=False)
    ep_i.metadata = add_metadata_features(ep_i.metadata)
    epochs[subject] = ep_i

# %%
all_md = pl.from_pandas(
    pd.concat([ep.metadata.rename_axis("epoch_idx").assign(subject=subject).reset_index()
               for subject, ep in epochs.items()], ignore_index=True).drop(columns=["TDT Block"])) \
    .with_columns(pl.col("subject").cast(subject_enum),
                  pl.col("phoneme_pair").cast(phoneme_pair_enum),
                  pl.col("word_end").cast(word_end_enum))

# %%
# Load saved phonetic decoders
phonetic_decoder_checkpoints = {
    subject: torch.load(f"outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/results.pt")
    for subject in tqdm(epochs.keys())
}


# %%
# New format, todo

# phonetic_decoder_models = {}
# for dec_dir in tqdm(phonetic_searchlight_paths):
#     subject = dec_dir.name
#     checkpoint_path = dec_dir / "decoding_models.joblib"
#     phonetic_decoder_models[subject] = joblib.load(checkpoint_path)

# # model predictions on test folds
# phonetic_decoder_outcomes = {path.name: pd.read_parquet(path / "outcomes.parquet")
#                              for path in phonetic_searchlight_paths}

# # model predictions on all relevant epochs for a given decoder
# # (e.g. all p/b epochs for a p/b decoder)
# # also incorporates multiple measures
# phonetic_decoder_all_outcomes = {path.name: pd.read_parquet(path / "all_outcomes.parquet")
#                                  for path in phonetic_searchlight_paths}

# %%
def pl_roc_auc(df: pl.DataFrame, target_col: str, proba_col: str, group_cols: list[str],
               roc_auc_name="roc_auc") -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col(proba_col)
            .rank(method="average")
            .over(group_cols)
            .alias("rank")
        )
        .group_by(group_cols) \
        .agg(
            n_pos = pl.col(target_col).sum(),
            n = pl.len(),
            rank_sum_pos = pl.col("rank")
                .filter(pl.col(target_col) == 1)
                .sum(),
        )
        .with_columns(
            n_neg = (pl.col("n") - pl.col("n_pos"))
        )
        .with_columns(
            **{roc_auc_name: pl.when((pl.col("n_pos") > 0) & (pl.col("n_neg") > 0))
                .then(
                    (pl.col("rank_sum_pos") - pl.col("n_pos") * (pl.col("n_pos") + 1) / 2)
                    / (pl.col("n_pos") * pl.col("n_neg"))
                ).otherwise(None)}
        )
        .select(group_cols + [roc_auc_name])
    )


# %%
word_end_df = pl.from_pandas(
    pd.DataFrame.from_dict(OFFSET_DICT, orient="index", columns=["word_end_offset"]).rename_axis("word_end") \
        .join(pd.DataFrame.from_dict(WORD_END_TO_PHONEME_PAIR, orient="index", columns=["phoneme_pair"]).rename_axis("word_end"),
              on="word_end").reset_index() \
        .join(pd.DataFrame.from_dict(POD_dict, orient="index", columns=["pod"]).rename_axis("phoneme_pair"),
              on="phoneme_pair").reset_index()) \
    .with_columns(
        pl.col("word_end").cast(word_end_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
        ((pl.col("word_end_offset") - epoch_tmin) * epoch_sfreq).alias("word_end_offset_sample"),
        ((pl.col("pod") - epoch_tmin) * epoch_sfreq).alias("pod_sample"),
    )

# %% [markdown]
# ## Behav prep

# %%
behav_pred_df = pl.concat([pl.read_parquet(f) for f in A_behav_predictions + A_early_behav_predictions]) \
    .with_columns(pl.col("subject").cast(subject_enum),
                  pl.col("phoneme_pair").cast(phoneme_pair_enum),
                  pl.col("word_end").cast(word_end_enum),
                  (pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"))

# %% [markdown]
# ### Find behav peaks

# %%
# Compute behavioral baselines
behav_baseline_df = pl_roc_auc(
    df=behav_pred_df.unique(subset=["subject", "electrode_idx", "phoneme_pair", "word_end", "epoch_idx", "fold"], keep="first"),
    target_col="decoder_target",
    proba_col="baseline_decoder_proba",
    group_cols=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
    roc_auc_name="behav_roc_auc_baseline"
)

# %%
# Compute per-window, per-fold behavioral prediction performance
# Exclude the late time windows we don't care about
# Include the early time windows for now; these will be reused in later comparison analyses
group_cols = ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax", "fold"]
behav_roc_auc_searchlight = pl_roc_auc(
    df=behav_pred_df.filter(pl.col("smax") <= all_response_smax_max),
    target_col="decoder_target",
    proba_col="full_decoder_proba",
    group_cols=group_cols,
    roc_auc_name="behav_roc_auc")

# %%
behav_roc_auc_searchlight_df = behav_roc_auc_searchlight \
    .join(behav_baseline_df, on=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"], how="inner") \
    .with_columns(
        (pl.col("behav_roc_auc") - pl.col("behav_roc_auc_baseline")).alias("behav_roc_auc_improvement")
    )

# %%
behav_roc_auc_mean_df = behav_roc_auc_searchlight_df \
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]) \
    .agg(
        [pl.col("behav_roc_auc").mean(),
         pl.col("behav_roc_auc_baseline").mean(),
         pl.col("behav_roc_auc_improvement").mean(),
    ])

# %%
behav_peaks_df_unfiltered = (
    behav_roc_auc_mean_df
    .join(word_end_df, on=["phoneme_pair", "word_end"], how="left")
    .filter(pl.col("smax") <= pl.col("word_end_offset_sample") + 20,
            pl.col("smin") >= pl.col("pod_sample"))
    .sort("behav_roc_auc_improvement", descending=True)
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .first()
)

# %%
behav_peaks_df = (
    behav_peaks_df_unfiltered
    .filter(pl.col("behav_roc_auc_improvement") > 0)
)

# %% [markdown]
# ## Prep phonetic

# %%
phon_pred_df = pl.read_parquet(phon_predictions_path) \
    .with_columns(pl.col("subject").cast(subject_enum),
                  pl.col("phoneme_pair").cast(phoneme_pair_enum),
                  (pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"))

# %%
# Compute per-window, per-fold phonetic prediction performance
# Exclude the late time windows we don't care about
# Include the rest of time windows for now; these will be reused in later comparison analyses
group_cols = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold"]
phon_roc_auc_searchlight_df = pl_roc_auc(
    df=phon_pred_df.filter((pl.col("smin") >= phon_response_smin_min) & (pl.col("smax") <= all_response_smax_max)),
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=group_cols,
    roc_auc_name="phon_roc_auc"
)

# %%
phon_roc_auc_mean_df = phon_roc_auc_searchlight_df \
    .group_by(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]) \
    .agg(pl.col("phon_roc_auc").mean())

# %%
phon_peaks_df = (
    phon_roc_auc_mean_df
    .join(word_end_df.group_by(["phoneme_pair"]).agg(pl.max("word_end_offset_sample")), on=["phoneme_pair"], how="left")
    .filter(pl.col("smin") >= phon_response_smin_min,
            pl.col("smax") <= pl.col("word_end_offset_sample"),
            pl.col("phon_roc_auc") >= phon_response_peak_threshold)
    .sort("phon_roc_auc", descending=True)
    .group_by(["subject", "electrode_idx", "phoneme_pair"])
    .first()
)

# %% [markdown]
# ## Electrode distribution

# %%
electrode_distribution_df = (
    phon_peaks_df
    .join(
        (
            behav_peaks_df
            .group_by(["subject", "electrode_idx", "phoneme_pair"])
            .agg(pl.max("behav_roc_auc_improvement"))
        ),
        on=["subject", "electrode_idx", "phoneme_pair"], how="left")
    .group_by(["subject", "electrode_idx"])
    .agg(pl.col("phon_roc_auc").is_not_null().any().alias("phonetic_selective"),
         pl.col("behav_roc_auc_improvement").is_not_null().any().alias("behavior_selective"))
    .group_by("subject")
    .agg(pl.sum("phonetic_selective").alias("phonetic_selective"),
         pl.sum("behavior_selective").alias("behavior_selective"))

    # join in speech responsive facts
    .join(
        electrode_df
        .group_by("subject").agg(
            pl.sum("speech_responsive").alias("speech_responsive"),
            pl.len().alias("total_electrodes")),
        on="subject", how="left")

    .sort("total_electrodes", descending=True)
).to_pandas()
electrode_distribution_df

# %%
fig, ax = plt.subplots(figsize=(3, 0.3 * len(electrode_distribution_df)))

total = electrode_distribution_df["total_electrodes"].values
speech = electrode_distribution_df["speech_responsive"].values
phonetic = electrode_distribution_df["phonetic_selective"].values
behavior = electrode_distribution_df["behavior_selective"].values

plot_palette = sns.color_palette(categorical_palette, 3)

# derive complementary counts
non_speech = total - speech
speech_not_phonetic = speech - phonetic

y = np.arange(len(electrode_distribution_df))
# first (leftmost): phonetic
ax.barh(y, phonetic, color=plot_palette[0],
        label="Phonetically\nselective", alpha=0.9,)

# --- overlay: behaviorally selective (hatched) ---
ax.barh(y, behavior,
    left=0, facecolor="none", edgecolor="k", hatch="///",
    linewidth=0.0, label="Behaviorally\nselective", zorder=5)

# second: speech-responsive but not phonetic
ax.barh(y, speech_not_phonetic, color=plot_palette[1],
        left=phonetic, label="Task-responsive\n(non-phonetic)", alpha=0.7)

# third: non-speech electrodes
ax.barh(y, non_speech, color=plot_palette[2],
        left=phonetic + speech_not_phonetic, label="Other", alpha=0.5,)

ax.set_yticks(y)
ax.set_yticklabels(electrode_distribution_df.subject, rotation=45, ha="right")
ax.set_xlabel("Number of electrodes")
ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.0), fontsize=9)

fig.savefig("figures/electrode_distribution.pdf")

# %%
fig, ax = plt.subplots(figsize=(2, 3))#0.22 * len(electrode_distribution_df)))

electrode_distribution_df_plot = electrode_distribution_df.sort_values("phonetic_selective", ascending=False)
# redo numbering after sorting
electrode_distribution_df_plot.reset_index(drop=True, inplace=True)
# add that as an index column
electrode_distribution_df_plot.reset_index(inplace=True)
electrode_distribution_df_plot["index"] = 10 - electrode_distribution_df_plot["index"]

task_selective = electrode_distribution_df_plot["speech_responsive"].values
phonetic = electrode_distribution_df_plot["phonetic_selective"].values
behavior = electrode_distribution_df_plot["behavior_selective"].values

# calculate proportions relative to task-selective total
phonetic_prop = phonetic / task_selective
behavior_prop = behavior / task_selective
non_phonetic_prop = (task_selective - phonetic) / task_selective

y = np.arange(len(electrode_distribution_df_plot))

plot_palette = sns.color_palette(categorical_palette, 2)

# first (leftmost): phonetic proportion
ax.barh(y, phonetic_prop, color=plot_palette[0],
        label="Phonetically\nselective", 
        alpha=0.9)

# overlay: behaviorally selective proportion (hatched)
ax.barh(y, behavior_prop, 
        left=0, 
        facecolor="none", 
        edgecolor="k", 
        hatch="///", 
        linewidth=0.0,
        label="Behaviorally\nselective", 
        zorder=5)

# second: task-responsive but not phonetic
ax.barh(y, non_phonetic_prop, 
        left=phonetic_prop,
        color="gray",
        label=None,#"Other task-responsive", 
        alpha=0.3)

ax.set_yticks(y)
ax.set_yticklabels(electrode_distribution_df_plot["index"], ha="right")
ax.set_xlabel("% of task-responsive\nelectrodes")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.set_xlim(0, 1)
ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.0), fontsize=10)
sns.despine(ax=ax, top=True, right=True)

fig.savefig("figures/electrode_distribution-task_selective.pdf")

# %%
fig, ax = plt.subplots(figsize=(1.8, 2.2))

electrode_distribution_df_plot = electrode_distribution_df.copy()
electrode_distribution_df_plot["proportion_behavior"] = electrode_distribution_df_plot["behavior_selective"] / electrode_distribution_df_plot["phonetic_selective"]
electrode_distribution_df_plot = electrode_distribution_df_plot.sort_values("proportion_behavior", ascending=False)
electrode_distribution_df_plot.reset_index(drop=True, inplace=True)
electrode_distribution_df_plot.reset_index(inplace=True)
electrode_distribution_df_plot["index"] = 10 - electrode_distribution_df_plot["index"]

phonetic = electrode_distribution_df_plot["phonetic_selective"].values
behavior = electrode_distribution_df_plot["behavior_selective"].values

# proportions relative to phonetically selective
behavior_of_phonetic = behavior / phonetic
non_behavior_of_phonetic = (phonetic - behavior) / phonetic

y = np.arange(len(electrode_distribution_df_plot))

plot_palette = sns.color_palette(categorical_palette, 2)

# base: behaviorally selective portion
ax.barh(y, behavior_of_phonetic,
        color=plot_palette[0],
        label="Behaviorally\nselective",
        alpha=0.9)

# stack on top: phonetically selective but not behaviorally selective
ax.barh(y, non_behavior_of_phonetic,
        left=behavior_of_phonetic,
        color="gray",
        label=None,#"Phonetically\nselective",
        alpha=0.9)


ax.set_yticks(y)
ax.set_yticklabels(electrode_distribution_df_plot["index"], ha="right")
ax.set_xlabel("% of phonetically\nselective electrodes")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.set_xlim(0, 1)
ax.legend(loc="lower left", fontsize=10)
sns.despine(ax=ax, top=True, right=True)

fig.savefig("figures/electrode_distribution-phonetic_selective.pdf")

# %% [markdown]
# ## Stimulus plot

# %%
f, axs = plt.subplots(2, 1, figsize=(8, 1), sharex=True)
md0 = next(iter(epochs.values())).metadata
word_end_time = word_end_df.filter(pl.col("phoneme_pair") == "dn").select(pl.max("word_end_offset")).item()

for i, (ax, word_end) in enumerate(zip(axs, PHONEME_PAIR_TO_WORD_ENDS["dn"])):
    md = md0[md0.word_end == word_end]
    add_textgrid_single(ax, textgrid_dir, md, rotation=0)

    if i < len(axs) - 1:
        ax.set_xticks([])
    else:
        ax.set_xlabel("Time since word onset (s)")
        ax.set_xticks(np.arange(0, word_end_time, 0.1))
    ax.set_yticks([])

# draw where the decoding starts
for ax in axs:
    pod = POD_dict["dn"]
    ax.axvline(pod, color="black", lw=4, alpha=0.5)
    ax.set_xlim((0, word_end_time + 0.01))
    ax.axvspan(pod, ax.get_xlim()[1], color="gray", alpha=0.3)
    

# %% [markdown]
# ## Plot neurometric for phonetic targets

# %%
# --- phon peaks: keys -> join to predictions -> join metadata
plot_phon_phon_keys = phon_peaks_df.select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
plot_phon_phon_df = (
    plot_phon_phon_keys
    .join(phon_pred_df,
          on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
          how="left")
    .join(all_md,
          on=["subject", "epoch_idx", "phoneme_pair"],
          how="left")
)


# %%
# --- behav peaks: keys (+ word_end) -> rename -> join to predictions -> join metadata
plot_behav_phon_keys = behav_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "word_end"]
)

plot_behav_phon_df = (
    plot_behav_phon_keys
    .rename({"word_end": "behav_word_end"})
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
    # retain epochs where metadata word_end matches behav peak word_end
    .filter(pl.col("word_end") == pl.col("behav_word_end"))
    .drop("behav_word_end")
)

# %%
# --- concat with source labels
plot_phon_df = pl.concat(
    [
        plot_phon_phon_df.with_columns(pl.lit("phon").alias("source")),
        plot_behav_phon_df.with_columns(pl.lit("behav").alias("source")),
    ],
    how="vertical",
).with_columns(
    pl.concat_str(
        [pl.col("subject"), pl.col("electrode_idx").cast(pl.Utf8), pl.col("phoneme_pair")],
        separator="_"
    ).alias("site")
)

# --- asserts
assert plot_phon_df.select(pl.col("resampled").is_null().any()).item() is False
assert plot_phon_df.select(pl.col("fold").is_null().any()).item() is False

# %%
# g = sns.catplot(data=plot_phon_df.to_pandas(),
#                 x="resampled", y="decoder_proba",
#                 # hue="lexical_evidence",
#                 col="phoneme_pair", col_order=phoneme_pair_order,
#                 row="source", row_order=["phon", "behav"],
#                 kind="point", units="site",
#                 height=3.5)
# g.set_axis_labels("Stimulus step", "Predicted\nP(second phoneme)")
# g.set_titles(template="Window: {row_name}\nPhoneme pair: {col_name}")

# for ax in g.axes.flat:
#     ax.axhline(0.5, color="red", linestyle="--")

# %%
# compute phonetic accuracy per source and resampled, within fold
phon_acc = (plot_phon_df
 .with_columns(
     ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5))
     .cast(pl.Float32).alias("correct")
 )
 .group_by(["source", "site", "subject", "electrode_idx", "phoneme_pair", "lexical_evidence", "resampled", "fold"])
 .agg(pl.col("correct").mean().alias("accuracy")))

# %%
phon_acc_change = (phon_acc
    .pivot(
        values="accuracy",
        index=["site", "subject", "electrode_idx", "phoneme_pair",
            "resampled", "lexical_evidence", "fold"],
        on="source",
        aggregate_function="first"
    )
    .drop_nulls(["phon", "behav"])
    .with_columns(
        (pl.col("phon") - pl.col("behav")).alias("acc_diff")
    )
)

# %%
# # compute phonetic accuracy per source and resampled, within fold
# phon_acc = plot_phon_df.groupby(["source", "site", "subject", "electrode_idx", "phoneme_pair", "resampled", "lexical_evidence", "smin", "smax", "fold"]) \
#     .progress_apply(lambda xs: accuracy_score(xs.decoder_target == 1, xs.decoder_proba > 0.5)) \
#     .rename("accuracy").reset_index()

# %%
# # Pivot wider and get difference in accuracy depending on phon source vs behav source
# assert phon_acc.groupby(["site", "subject", "electrode_idx", "phoneme_pair", "resampled", "lexical_evidence", "fold"]).size().max() <= 2

# # NB we are doing dropna here -- so those phon sites which are predictive of phon but
# # not predictive of behav aren't included in the comparison
# phon_acc_change = phon_acc.pivot_table(
#     index=["site", "subject", "electrode_idx", "phoneme_pair", "resampled", "lexical_evidence", "fold"],
#     columns="source",
#     values="accuracy").dropna().assign(acc_diff=lambda df: df.phon - df.behav)

# %%
# g = sns.catplot(data=phon_acc_change.to_pandas(),
#                 x="resampled", y="phon", hue="lexical_evidence",
#                 row="phoneme_pair", row_order=phoneme_pair_order,
#                 kind="point", units="site", height=3.5)
# g.set_axis_labels("Stimulus step", "Phonetic\ndecoding accuracy")
# g.set_titles(template="{row_name}")
# for ax in g.axes.flat:
#     ax.axhline(0.5, color="red", linestyle="--")

# %%
# g = sns.catplot(
#     data=phon_acc_change.melt(
#         id_vars=["site", "subject", "electrode_idx", "phoneme_pair", "resampled", "lexical_evidence", "fold"],
#         value_vars=["phon", "behav"],
#         variable_name="source",
#         value_name="accuracy"),
#     x="resampled", y="accuracy", hue="source", hue_order=source_order,
#     row="phoneme_pair", row_order=phoneme_pair_order,
#     kind="point", units="site", height=3.5)
# g.set_axis_labels("Stimulus step", "Phonetic\ndecoding accuracy")
# g.set_titles(template="{row_name}")

# for ax in g.axes.flat:
#     ax.axhline(0.5, color="red", linestyle="--")

# %%
# from scipy.stats import ttest_1samp
# phon_acc_change.groupby(["phoneme_pair", "resampled", "lexical_evidence"]) \
#     .apply(lambda xs: pd.Series(ttest_1samp(xs["phon"], popmean=0.5), index=["t_stat", "p_value"])).sort_values("p_value")

# %%
# g = sns.catplot(data=phon_acc_change.to_pandas(),
#                 x="resampled", y="acc_diff", hue="lexical_evidence",
#                 col="phoneme_pair", col_order=phoneme_pair_order,
#                 kind="point", units="site")
# for ax in g.axes.flat:
#     ax.axhline(0, color="red", linestyle="--")

# %%
# from scipy.stats import ttest_1samp
# phon_acc_change.groupby(["phoneme_pair", "resampled", "lexical_evidence"]) \
#     .apply(lambda xs: pd.Series(ttest_1samp(xs["acc_diff"], popmean=0), index=["t_stat", "p_value"])).sort_values("p_value")

# %% [markdown]
# ## Behav prediction

# %%
plot_phon_behav_keys = phon_peaks_df.select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
plot_phon_behav_df = plot_phon_behav_keys.join(
    behav_pred_df,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="left")
missing_phon_behav = plot_phon_behav_df.filter(pl.col("full_decoder_proba").is_null())
if missing_phon_behav.height > 0:
    L.warning(f"Found {missing_phon_behav.height} phonetic peak sites with no matching behavioral predictions")
    L.warning(missing_phon_behav.select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]))

plot_behav_behav_keys = (
    behav_peaks_df
    .select(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"])
)
plot_behav_behav_keys_unfiltered = (
    behav_peaks_df_unfiltered
    .select(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"])
)
plot_behav_behav_df = plot_behav_behav_keys.join(
    behav_pred_df,
    on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
    how="left"
)
missing_behav_behav = plot_behav_behav_df.filter(pl.col("full_decoder_proba").is_null())
if missing_behav_behav.height > 0:
    L.warning(f"Found {missing_behav_behav.height} behavioral peak sites with no matching behavioral predictions")
    L.warning(missing_behav_behav.select(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]))

# %%
plot_behav_df = pl.concat(
    [
        plot_phon_behav_df.with_columns(pl.lit("phon").alias("source")),
        plot_behav_behav_df.with_columns(pl.lit("behav").alias("source")),
    ],
    how="align",
).with_columns(
    pl.concat_str(
        [pl.col("subject"), pl.col("electrode_idx").cast(pl.Utf8), pl.col("phoneme_pair")],
        separator="_"
    ).alias("site")
).join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"], how="left")

# %%
# g = sns.catplot(data=plot_behav_df.to_pandas(),
#                 x="resampled", y="full_decoder_proba", hue="decoder_target",
#                 col="phoneme_pair", col_order=phoneme_pair_order,
#                 row="source", row_order=["phon", "behav"],
#                 kind="point", units="site",
#                 height=3.5)
# g.set_axis_labels("Stimulus step", "Predicted P(picks\nsecond phoneme)")
# g.set_titles(template="Window: {row_name}\nPhoneme pair: {col_name}")

# for ax in g.axes.flat:
#     ax.axhline(0.5, color="red", linestyle="--")

# %%
behav_roc_auc = pl_roc_auc(
    df=plot_behav_df,
    target_col="decoder_target",
    proba_col="full_decoder_proba",
    group_cols=["source", "site", "subject", "electrode_idx", "phoneme_pair", "word_end", "resampled", "smin", "smax", "fold"],
    roc_auc_name="behav_roc_auc"
).join(
    pl_roc_auc(
        df=plot_behav_df,
        target_col="decoder_target",
        proba_col="baseline_decoder_proba",
        group_cols=["source", "site", "subject", "electrode_idx", "phoneme_pair", "word_end", "resampled", "smin", "smax", "fold"],
        roc_auc_name="behav_roc_auc_baseline"),
    on=["source", "site", "subject", "electrode_idx", "phoneme_pair", "word_end", "resampled", "smin", "smax", "fold"],
    how="inner"
).with_columns(
    (pl.col("behav_roc_auc") - pl.col("behav_roc_auc_baseline")).alias("behav_roc_auc_improvement"),
    pl.concat_str(
        [pl.col("subject"), pl.col("electrode_idx").cast(pl.Utf8), pl.col("phoneme_pair")],
        separator="_"
    ).alias("site")
)

# %%
# g = sns.catplot(data=behav_roc_auc.to_pandas().astype({"resampled": int}),
#                 x="resampled", y="behav_roc_auc",
#                 hue="source", hue_order=source_order,
#                 row="phoneme_pair", row_order=phoneme_pair_order,
#                 kind="point", units="site", height=3.5)
# g.set_axis_labels("Stimulus step", "Behavioral\ndecoding ROC-AUC")
# g.set_titles(template="{row_name}")

# for ax in g.axes.flat:
#     ax.axhline(0.5, color="red", linestyle="--")

# %%
# g = sns.catplot(data=behav_roc_auc.to_pandas(), x="resampled", y="behav_roc_auc_improvement",
#                 hue="source", hue_order=source_order, col="phoneme_pair", col_order=phoneme_pair_order,
#                 kind="point", units="site")
# g.set_axis_labels("Resampled", "ROC AUC Improvement over Baseline")

# for ax in g.axes.flat:
#     ax.axhline(0, color="red", linestyle="--")

# %% [markdown]
# ## Prepare data struct for future plotting

# %%
paper_data = PaperData(
    electrode_df=electrode_df,
    plot_phon_phon_df=plot_phon_phon_df,
    plot_behav_phon_df=plot_behav_phon_df,
    plot_behav_behav_df=plot_behav_behav_df,
    plot_phon_behav_df=plot_phon_behav_df,
    behav_roc_auc_searchlight_df=behav_roc_auc_searchlight_df,
    all_md=all_md,
    word_end_df=word_end_df,
    epochs=epochs
)

# %% [markdown]
# ## Dynamics

# %% [markdown]
# ### Decoding timecourse

# %%
sns.lineplot(
    data=phon_roc_auc_mean_df.to_pandas(),
    x="smax", y="phon_roc_auc", hue="phoneme_pair", hue_order=phoneme_pair_order)

# %%
sns.lineplot(
    data=behav_roc_auc_mean_df.to_pandas().query("word_end == 'necessary'"),
    x="smax", y="behav_roc_auc_improvement", hue="phoneme_pair", hue_order=phoneme_pair_order)

# %%
# above but with median
plot_textgrid = "11_necessary_dn_002.TextGrid"
behav_median_timecourse = behav_roc_auc_mean_df.to_pandas().query("word_end == 'necessary'").groupby(["phoneme_pair", "smin", "smax"]).behav_roc_auc_improvement.median().reset_index()
behav_median_timecourse["tmin"] = behav_median_timecourse["smin"] / epoch_sfreq + epoch_tmin
behav_median_timecourse["tmax"] = behav_median_timecourse["smax"] / epoch_sfreq + epoch_tmin
ax = sns.lineplot(
    data=behav_median_timecourse,
    x="tmax", y="behav_roc_auc_improvement", hue="phoneme_pair", hue_order=phoneme_pair_order)
add_textgrid(ax, textgrid_dir, textgrid_file=plot_textgrid)

# %% [markdown]
# ### Peak timing

# %%
peak_timing_plot = pl.concat([
    phon_peaks_df.with_columns(pl.lit("phon").alias("source")),
    behav_peaks_df.with_columns(pl.lit("behav").alias("source")),
], how="align").with_columns(
    (((pl.col("smin") + pl.col("smax")) / 2) / epoch_sfreq + epoch_tmin).alias("t_center")
)

# %%
g = sns.displot(data=peak_timing_plot.to_pandas(), x="t_center",
                hue="phoneme_pair", hue_order=phoneme_pair_order,
                row="source", row_order=source_order,
                kind="kde", clip=(0, None),
                height=2.5, aspect=2.5)
g.set_axis_labels("Peak decoding time (s)", "Density")

# %%
# g = sns.displot(data=peak_timing_plot.to_pandas(), x="t_center",
#                 hue="source", hue_order=source_order,
#                 row="phoneme_pair", row_order=phoneme_pair_order,
#                 # kind="kde", clip=(0, None),
#                 kind="hist", stat="density", common_norm=False,
#                 height=2.5, aspect=2.5)
# g.set_axis_labels("Peak decoding time (s)", "Density")

# %%
g = sns.displot(data=peak_timing_plot.filter(pl.col("phoneme_pair") == "dn").to_pandas(), x="t_center",
                hue="source", hue_order=source_order, palette=categorical_palette,
                # kind="hist", stat="density", common_norm=False,
                kind="kde", common_norm=False, clip=(0, None),
                height=2.5, aspect=3/2.5)
g.set_axis_labels("Peak decoding time (s)", "Density")

for (row, col, hue), data in g.facet_data():
    ax = g.axes[row][col]
    phoneme_pair = data.phoneme_pair.iloc[0]
    word_stim_info = word_end_df.filter(pl.col("phoneme_pair") == phoneme_pair)
    for word_end in word_stim_info.select("word_end_offset").to_series():
        ax.axvline(word_end, color="red", linestyle="--")
    pod = word_stim_info.select("pod").unique().item()
    ax.axvline(pod, color="blue", linestyle="--")

# %%
plot_word_end = "necessary"
plot_phoneme_pair = "dn"
plot_textgrid = "11_necessary_dn_002.TextGrid"
plot_xlim = (0, 1.2)
vline_extent = 1.1
g = sns.displot(
    data=peak_timing_plot.filter(
        (pl.col("word_end") == plot_word_end) |
        ((pl.col("source") == "phon") & (pl.col("phoneme_pair") == plot_phoneme_pair))).to_pandas()
        .assign(source=lambda df: df.source.map({"phon": "Phonetic", "behav": "Behavioral"})),
        x="t_center",
        hue="source", hue_order={"Phonetic": 0, "Behavioral": 1}, palette=categorical_palette,
        linewidth=2,
        # kind="hist", stat="density", common_norm=False,
        kind="kde", common_norm=False, clip=(0, None),
        # legend=False,
        height=2, aspect=2.75/2)
g.set_axis_labels("Peak decoding time (s)", "Density")
sns.move_legend(g, "upper right", bbox_to_anchor=(0.65, 0.93),
                fontsize=10, frameon=True, title=None)

for (row, col, hue), data in g.facet_data():
    ax = g.axes[row][col]
    ax.set_xlim(plot_xlim)

    phoneme_pair = data.phoneme_pair.iloc[0]
    word_stim_info = word_end_df.filter(pl.col("word_end") == plot_word_end)
    # for word_end in word_stim_info.select("word_end_offset").to_series():
    #     ax.axvline(word_end, color="red", linestyle="--")
    pod = word_stim_info.select("pod").unique().item()
    ax.axvline(pod, ymax=vline_extent, color="red", alpha=0.5, linewidth=2, linestyle="--", clip_on=False)

    add_textgrid(ax, textgrid_dir, textgrid_file=plot_textgrid,
                 include_phonemes=False, fontsize=9, vline_extent=vline_extent)

g.savefig("figures/decoding_timing.pdf")

# %% [markdown]
# ### Peak timing of behavior relative to word offset

# %%
behav_peak_timing_plot = behav_peaks_df.with_columns(
    (((pl.col("smin") + pl.col("smax")) / 2) / epoch_sfreq + epoch_tmin).alias("t_center"),
    pl.col("word_end").replace_strict(OFFSET_DICT).alias("t_offset"),
    pl.col("phoneme_pair").replace_strict(POD_dict).alias("t_pod")
).with_columns(
    (pl.col("t_center") - pl.col("t_offset")).alias("t_from_offset"),
    (pl.col("t_center") - pl.col("t_pod")).alias("t_from_pod")
)

# %%
g = sns.displot(data=behav_peak_timing_plot.to_pandas(), x="t_from_offset",
                hue="phoneme_pair", hue_order=phoneme_pair_order,
                # hue="subject",
                kind="kde",
                height=2, aspect=2.5)
g.set_axis_labels("Time from word offset (s)", "Density")

# %%
g = sns.displot(data=behav_peak_timing_plot.to_pandas(), x="t_from_pod",
                hue="phoneme_pair", hue_order=phoneme_pair_order,
                # hue="subject",
                kind="kde",
                height=2, aspect=2.5)
g.set_axis_labels("Time from\npoint of disambiguation (s)", "Density")

# %%
# (
#     behav_peak_timing_plot
#     .group_by(["subject", "phoneme_pair"])
#     .agg(pl.col("t_from_offset").var(), pl.col("t_from_pod").var())
#     .unpivot(index=["subject", "phoneme_pair"],
#              on=["t_from_offset", "t_from_pod"],
#              variable_name="time_reference",
#              value_name="variance")
# )

# sns.catplot(data=
#     pl.concat([
#         (
#             behav_peak_timing_plot
#             .group_by(["subject", "phoneme_pair"])
#             .agg(pl.col("t_from_offset").var(), pl.col("t_from_pod").var())
#             .unpivot(index=["subject", "phoneme_pair"],
#                     on=["t_from_offset", "t_from_pod"],
#                     variable_name="time_reference",
#                     value_name="variance")
#         ),
#         (
#             peak_timing_plot
#             .filter(pl.col("source") == "behav")
#             .group_by(["subject", "phoneme_pair"])
#             .agg(pl.col("t_center").var().alias("variance"))
#             .with_columns(pl.lit("t_from_onset").alias("time_reference"))
#         )], how="align"),
#         x="phoneme_pair", order=phoneme_pair_order, y="variance",
#         hue="time_reference", kind="box")

# %% [markdown]
# ### Performance relative to peak

# %%
phon_relative_df = (
    phon_peaks_df
    .join(phon_roc_auc_mean_df, on=["subject", "electrode_idx", "phoneme_pair"], how="left")
    .with_columns(
        (pl.col("smin_right") - pl.col("smin")).alias("smin_relative"),
        (pl.col("smax_right") - pl.col("smax")).alias("smax_relative"),
        (pl.col("phon_roc_auc_right") / pl.col("phon_roc_auc")).alias("phon_roc_auc_relative"),
    )
    .with_columns(
        (pl.col("smax_relative") / epoch_sfreq).alias("tmax_relative")
    )
    .filter(
        pl.col("smin_relative") >= -relative_performance_swidth,
        pl.col("smax_relative") <= relative_performance_swidth,
        pl.col("phon_roc_auc_right") > 0.5
    )
)

# %%
# g = sns.relplot(data=phon_relative_df.to_pandas(), x="tmax_relative", y="phon_roc_auc_relative",
#                 hue="phoneme_pair", hue_order=phoneme_pair_order,
#                 kind="line", marker="o")
# g.set_axis_labels("Time from phonetic\ndecoding peak (s)", "Relative phonetic\nROC AUC (%)")
# for ax in g.axes.flat:
#     ax.axvline(0, color="gray", ls="--")
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
#     ax.set_xlim(-relative_performance_twidth, relative_performance_twidth)

# %%
# behav_relative_df = (
#     behav_peaks_df
#     .join(behav_roc_auc_mean_df, on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="left")
#     .with_columns(
#         (pl.col("smin_right") - pl.col("smin")).alias("smin_relative"),
#         (pl.col("smax_right") - pl.col("smax")).alias("smax_relative"),
#         (pl.col("behav_roc_auc_improvement_right") / pl.col("behav_roc_auc_improvement")).alias("behav_roc_auc_relative"),
#     )
#     .with_columns(
#         (pl.col("smax_relative") / epoch_sfreq).alias("tmax_relative")
#     )
#     .filter(
#         pl.col("smin_relative") >= -relative_performance_swidth,
#         pl.col("smax_relative") <= relative_performance_swidth,
#         pl.col("behav_roc_auc_improvement_right") > 0
#     )
# )

# %%
# g = sns.relplot(data=behav_relative_df.to_pandas(), x="tmax_relative", y="behav_roc_auc_relative",
#                 hue="phoneme_pair", hue_order=phoneme_pair_order,
#                 kind="line", marker="o")
# g.set_axis_labels("Time from behavioral\ndecoding peak (s)", "Relative behavioral\nROC AUC improvement (%)")

# for ax in g.axes.flat:
#     ax.axvline(0, color="gray", ls="--")
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
#     ax.set_xlim(-relative_performance_twidth, relative_performance_twidth)

# %% [markdown]
# ## Cross-window analysis

# %% [markdown]
# ### Phon prediction

# %%
phon_roc_auc_comparison_df = (
    pl.concat([
        (plot_phon_phon_keys
        .join(phon_roc_auc_searchlight_df,
            on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
            how="left")
        .with_columns(pl.lit("phon").alias("source"))),
        (plot_behav_phon_keys
        .join(phon_roc_auc_searchlight_df,
            on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
            how="left")
        .with_columns(pl.lit("behav").alias("source"))),
    ], how="align")
    .with_columns(
        pl.concat_str(
            [pl.col("subject"), pl.col("electrode_idx").cast(pl.Utf8), pl.col("phoneme_pair")],
            separator="_"
        ).alias("site")
    )
)

# %%
g = sns.catplot(
    data=(
        phon_roc_auc_comparison_df
        .group_by(["source", "site", "subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
        .agg(pl.mean("phon_roc_auc"))
    ).to_pandas(),
    x="phoneme_pair", order=phoneme_pair_order,
    y="phon_roc_auc",
    hue="source", hue_order=source_order,
    kind="box", units="site")
g.set_axis_labels("Phoneme pair", "Phonetic ROC AUC (%)")
for ax in g.axes.flat:
    ax.axhline(0.5, color="red", linestyle="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# %%
# from src.viz import spaghetti_plot
# spaghetti_plot(phon_roc_auc_overall.assign(word_end="NA"),
#                y="roc_auc", by1="phon", by2="behav", by="source",
#                col="phoneme_pair")

# %%
g = sns.displot(
    data=phon_roc_auc_comparison_df.pivot(
            on="source",
            index=["site", "subject", "electrode_idx", "phoneme_pair"],
            values="phon_roc_auc",
            aggregate_function="mean"
        ).filter(pl.col("phon").is_not_null() & pl.col("behav").is_not_null()
        ).with_columns(
            (pl.col("phon") - pl.col("behav")).alias("roc_auc_diff")
        ).to_pandas(),
    x="roc_auc_diff",
    height=2.5, aspect=2.5)
g.set_axis_labels("Improvement in phonetic prediction\nat early window vs late window\n($\\Delta$ROC-AUC)")

for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

# %% [markdown]
# ### Behav prediction

# %%
behav_roc_auc_comparison_phon = (
    plot_phon_behav_keys
    .join(behav_roc_auc_searchlight_df,
        on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
        how="left")
    .with_columns(pl.lit("phon").alias("source")))

# %%
# NB we are using the unfiltered behav peaks here
# we don't want to double-dip by just looking at improvement in the electrodes
# that were already selected because they show improvement
# we want a description of the trend within all phonetic electrodes
behav_roc_auc_comparison_behav = (
    plot_behav_behav_keys_unfiltered
    .join(behav_roc_auc_searchlight_df,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
        how="left")
    .with_columns(pl.lit("behav").alias("source")))

# %%
behav_roc_auc_comparison_df = (
    pl.concat([
        behav_roc_auc_comparison_phon,
        behav_roc_auc_comparison_behav,
        behav_baseline_df.with_columns(
            pl.lit("baseline").alias("source"),
            pl.col("behav_roc_auc_baseline").alias("behav_roc_auc"))
    ], how="align")
    .with_columns(
        pl.concat_str(
            [pl.col("subject"), pl.col("electrode_idx").cast(pl.Utf8), pl.col("phoneme_pair")],
            separator="_"
        ).alias("site"))
)

# %%
brac_ttest_df = (
        behav_roc_auc_comparison_df
        .group_by(["source", "site", "subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
        .agg(pl.mean("behav_roc_auc"))
    ).to_pandas().set_index("source").dropna(subset=["behav_roc_auc"])

# %%
from scipy.stats import ttest_ind
ttest_ind(brac_ttest_df.loc["phon"].behav_roc_auc, brac_ttest_df.loc["baseline"].behav_roc_auc)

# %%
ttest_ind(brac_ttest_df.loc["behav"].behav_roc_auc, brac_ttest_df.loc["baseline"].behav_roc_auc)

# %%
evaluation_order = ["Early\nwindow", "Baseline", "Late\nwindow"]
g = sns.catplot(
    data=(
        behav_roc_auc_comparison_df
        .group_by(["source", "site", "subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
        .agg(pl.mean("behav_roc_auc"))
    ).to_pandas()
    .assign(source=lambda xs: xs.source.replace({"phon": "Early\nwindow", "behav": "Late\nwindow", "baseline": "Baseline"}))
    .rename(columns={"source": "Evaluation"})
    .query("phoneme_pair == 'dn'"),
    # x="phoneme_pair", order=phoneme_pair_order,
    x="Evaluation", order=evaluation_order,
    y="behav_roc_auc",
    hue="Evaluation", hue_order=evaluation_order,
    showfliers=False,
    kind="box",
    height=3, aspect=2.75/3,
    palette=categorical_palette)
g.set_axis_labels(None, "Behavior prediction\n(ROC-AUC)")
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_ylim(ax.get_ylim()[0], 1.0)

g.savefig("figures/decoding_comparison.pdf")

# %%
# spaghetti_plot(behav_roc_auc_overall.query("source in ['phon', 'behav']"),
#                y="roc_auc", by1="phon", by2="behav", by="source",
#                col="phoneme_pair")

# %%
# within-fold comparison of behavior prediction between two windows and baseline
# then meaned over folds
behav_improvement_df = (
    behav_roc_auc_comparison_df
     .pivot(
         on="source",
         index=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
         values="behav_roc_auc",
         aggregate_function="mean")
     .filter(pl.col("phon").is_not_null() & pl.col("behav").is_not_null() & pl.col("baseline").is_not_null())
     .with_columns((pl.col("behav") - pl.col("baseline")).alias("behav_baseline_diff"),
                   (pl.col("phon") - pl.col("baseline")).alias("phon_baseline_diff"),
                   (pl.col("behav") - pl.col("phon")).alias("behav_phon_diff"))
     .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(pl.mean("behav_baseline_diff").alias("behav_baseline_diff"),
         pl.mean("phon_baseline_diff").alias("phon_baseline_diff"),
         pl.mean("behav_phon_diff").alias("behav_phon_diff")))

# %%
# Spaghetti plot: per-subject mean improvement from baseline
# Left = early/phon, Center = baseline (0), Right = late/behav
from scipy import stats

subject_means = (
    behav_improvement_df
    .group_by("subject")
    .agg(
        pl.mean("phon_baseline_diff"),
        pl.mean("behav_baseline_diff"),
    )
    .to_pandas()
)

# Paired t-tests (one-sample vs 0)
t_early, p_early = stats.ttest_1samp(subject_means["phon_baseline_diff"], 0)
t_late, p_late = stats.ttest_1samp(subject_means["behav_baseline_diff"], 0)
print(f"Early vs baseline: t={t_early:.3f}, p={p_early:.4f}")
print(f"Late vs baseline:  t={t_late:.3f}, p={p_late:.4f}")

def p_to_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

# fig, ax = plt.subplots(figsize=(3.5, 3))
fig, ax = plt.subplots(figsize=(2.75, 2.75))

# Draw individual subject lines (3 points: early, baseline, late)
for _, row in subject_means.iterrows():
    xs = [0, 1, 2]
    ys = [row["phon_baseline_diff"], 0, row["behav_baseline_diff"]]
    ax.plot(xs, ys, color="gray", alpha=0.4, linewidth=1, zorder=1)
    ax.scatter(xs, ys, color="gray", alpha=0.4, s=20, zorder=2)

# Draw grand mean
grand_early = subject_means["phon_baseline_diff"].mean()
grand_late = subject_means["behav_baseline_diff"].mean()
ax.plot([0, 1, 2], [grand_early, 0, grand_late],
        color="black", linewidth=2.5, zorder=3, alpha=0.7)
ax.scatter([0, 1, 2], [grand_early, 0, grand_late],
           color="black", s=60, zorder=4, alpha=0.7)

# Annotate with significance stars
ymax = max(subject_means["phon_baseline_diff"].max(),
           subject_means["behav_baseline_diff"].max())
star_y = ymax * 1.1
ax.annotate(p_to_stars(p_early), xy=(0, star_y), ha="center", va="bottom", fontsize=11)
ax.annotate(p_to_stars(p_late), xy=(2, star_y), ha="center", va="bottom", fontsize=11)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Early\nwindow", "Baseline", "Late\nwindow"])
ax.set_xlabel("Evaluation")
ax.set_ylabel("Behavior prediction\n($\Delta$ROC-AUC)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlim(-0.3, 2.3)
sns.despine()
fig.tight_layout()
plt.show()

fig.savefig("figures/decoding_behavioral_improvement.pdf")

# %%
g = sns.displot(data=behav_improvement_df.to_pandas(),
    x="behav_baseline_diff",
    height=2.5, aspect=2.5)
g.set_axis_labels("Improvement in behavior prediction\nat late window ($\\Delta$ROC-AUC)")

for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))

# %%
g = sns.displot(data=behav_improvement_df.to_pandas(),
    # x="improvement", row="comparison",
    x="behav_phon_diff",
    height=2.5, aspect=2.5)
g.set_axis_labels("Improvement in behavior prediction\nat early window vs late window\n($\\Delta$ROC-AUC)")

for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

# %%
g = sns.displot(data=behav_improvement_df.to_pandas(),
    # x="improvement", row="comparison",
    x="phon_baseline_diff",
    height=2.5, aspect=2.5)
g.set_axis_labels("Improvement in behavior prediction\nat early window ($\\Delta$ROC-AUC)")

for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

# %%

# %% [markdown]
# ### Cross-window transfer, no training

# %% [markdown]
# #### Transfer on phonetic target

# %%
early_to_late_transfer_keys = (
    phon_roc_auc_comparison_df
    .filter(pl.col("source") == "behav")
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"])
    .agg(pl.mean("phon_roc_auc"))
    .sort("phon_roc_auc", descending=True)
    .join(phon_peaks_df, on=["subject", "electrode_idx", "phoneme_pair"],
          how="inner", suffix="_early")
    .filter(pl.col("phon_roc_auc_early") > 0.6)
)
early_to_late_transfer_keys


# %%
def evaluate_phonetic_transfer(t_subject, t_electrode_idx, t_phoneme_pair, t_word_end,
                               t_smin_early, t_smax_early,
                               t_smin_late, t_smax_late,
                               t_num_folds=5, t_measure="categorical_acoustic_cue",
                               t_restrict_to_word_end=True):
    t_early_key = (t_subject, t_electrode_idx, t_phoneme_pair, t_smin_early, t_smax_early)
    t_late_key = (t_subject, t_electrode_idx, t_phoneme_pair, t_smin_late, t_smax_late)

    t_early_models = phonetic_decoder_checkpoints[t_subject]["models"][t_early_key]
    t_early_predictions = phonetic_decoder_checkpoints[t_subject]["outcomes"][t_early_key]
    t_early_all_predictions = phonetic_decoder_checkpoints[t_subject]["all_outcomes"][t_early_key + (t_measure,)]
    assert len(t_early_models) == t_num_folds

    t_late_models = phonetic_decoder_checkpoints[t_subject]["models"][t_late_key]
    t_late_predictions = phonetic_decoder_checkpoints[t_subject]["outcomes"][t_late_key]
    t_late_all_predictions = phonetic_decoder_checkpoints[t_subject]["all_outcomes"][t_late_key + (t_measure,)]
    assert len(t_late_models) == t_num_folds

    t_epochs = epochs[t_subject]
    t_epoch_data = t_epochs.get_data(picks=t_electrode_idx).squeeze(1)

    early_early_outcomes, early_late_outcomes = [], []
    late_late_outcomes, late_early_outcomes = [], []

    def prepare_decoding_data(ep_i, data_i, epoch_idxs, smin, smax, baseline_vars=None):
        X = data_i[epoch_idxs][:, smin:smax]
        # add baseline predictors
        if baseline_vars is not None:
            X_baseline = ep_i.metadata.loc[epoch_idxs][baseline_vars].values
            X = np.concatenate([X_baseline, X], axis=1)
        return X

    def restrict_word_end(outcome_rows, word_end):
        outcome_rows = pd.merge(outcome_rows, t_epochs.metadata, left_on=["epoch_idx"], right_index=True, how="left")
        return outcome_rows.loc[outcome_rows.word_end == word_end][["epoch_idx", "fold", "decoder_target", "decoder_proba"]]

    for fold, fold_rows in t_early_predictions.groupby("fold"):
        early_idxs = fold_rows.epoch_idx

        early_pipe = t_early_models[fold]
        late_pipe = t_late_models[fold]

        X_early = prepare_decoding_data(t_epochs, t_epoch_data, early_idxs, t_smin_early, t_smax_early)
        early_preds = early_pipe.predict_proba(X_early)[:, 1]
        # Verify that we get the same output as stored when testing early->early
        np.testing.assert_allclose(early_preds, fold_rows.decoder_proba.values)

        # Now compute early->early predictions on all epochs
        fold_all_rows = t_early_all_predictions[t_early_all_predictions.fold == fold]
        if t_restrict_to_word_end:
            fold_all_rows = restrict_word_end(fold_all_rows, t_word_end)
        early_all_idxs = fold_all_rows.epoch_idx

        X_early_all = prepare_decoding_data(t_epochs, t_epoch_data, early_all_idxs, t_smin_early, t_smax_early)
        early_all_preds = early_pipe.predict_proba(X_early_all)[:, 1]

        # Now test early->late transfer
        # Use the LATE scaler and the EARLY model
        X_late_all = prepare_decoding_data(t_epochs, t_epoch_data, early_all_idxs, t_smin_late, t_smax_late)
        late_scaler = late_pipe.named_steps["standardscaler"]
        X_late_all = late_scaler.transform(X_late_all)
        late_all_preds = early_pipe.named_steps["logisticregression"].predict_proba(X_late_all)[:, 1]
        
        early_early_outcomes.append(
            fold_all_rows.assign(subject=t_subject,
                                electrode_idx=t_electrode_idx,
                                phoneme_pair=t_phoneme_pair,
                                word_end=t_word_end,
                                smin=t_smin_early, smax=t_smax_early,
                                decoder_target=(fold_all_rows.decoder_target > 0).astype(int),
                                decoder_proba=early_all_preds,)
        )
        early_late_outcomes.append(
            fold_all_rows.assign(subject=t_subject,
                                electrode_idx=t_electrode_idx,
                                phoneme_pair=t_phoneme_pair,
                                word_end=t_word_end,
                                smin=t_smin_late, smax=t_smax_late,
                                decoder_target=(fold_all_rows.decoder_target > 0).astype(int),
                                decoder_proba=late_all_preds,)
        )

    for fold, fold_rows in t_late_predictions.groupby("fold"):
        late_idxs = fold_rows.epoch_idx

        early_pipe = t_early_models[fold]
        late_pipe = t_late_models[fold]

        X_late = prepare_decoding_data(t_epochs, t_epoch_data, late_idxs, t_smin_late, t_smax_late)        
        late_preds = late_pipe.predict_proba(X_late)[:, 1]
        # Verify that we get the same output as stored when testing late->late
        np.testing.assert_allclose(late_preds, fold_rows.decoder_proba.values)

        # Now compute late->late predictions on all epochs
        fold_all_rows = t_late_all_predictions[t_late_all_predictions.fold == fold]
        if t_restrict_to_word_end:
            fold_all_rows = restrict_word_end(fold_all_rows, t_word_end)
        late_all_idxs = fold_all_rows.epoch_idx

        X_late_all = prepare_decoding_data(t_epochs, t_epoch_data, late_all_idxs, t_smin_late, t_smax_late)
        late_all_preds = late_pipe.predict_proba(X_late_all)[:, 1]

        # Now test late->early transfer
        # Use the EARLY scaler and the LATE model
        X_early_all = prepare_decoding_data(t_epochs, t_epoch_data, late_all_idxs, t_smin_early, t_smax_early)
        early_scaler = early_pipe.named_steps["standardscaler"]
        X_early_all = early_scaler.transform(X_early_all)
        early_all_preds = late_pipe.named_steps["logisticregression"].predict_proba(X_early_all)[:, 1]
        
        late_late_outcomes.append(
            fold_all_rows.assign(subject=t_subject,
                                electrode_idx=t_electrode_idx,
                                phoneme_pair=t_phoneme_pair,
                                word_end=t_word_end,
                                smin=t_smin_late, smax=t_smax_late,
                                decoder_target=(fold_all_rows.decoder_target > 0).astype(int),
                                decoder_proba=late_all_preds,)
        )
        late_early_outcomes.append(
            fold_all_rows.assign(subject=t_subject,
                                electrode_idx=t_electrode_idx,
                                phoneme_pair=t_phoneme_pair,
                                word_end=t_word_end,
                                smin=t_smin_early, smax=t_smax_early,
                                decoder_target=(fold_all_rows.decoder_target > 0).astype(int),
                                decoder_proba=early_all_preds,)
        )

    return (pd.concat(early_early_outcomes), pd.concat(early_late_outcomes),
            pd.concat(late_late_outcomes), pd.concat(late_early_outcomes))


# %%
early_early_outcomes, early_late_outcomes = [], []
late_late_outcomes, late_early_outcomes = [], []
for key in tqdm(early_to_late_transfer_keys.iter_rows(named=True), total=early_to_late_transfer_keys.height):
    ee_i, el_i, ll_i, le_i = evaluate_phonetic_transfer(
        t_subject=key["subject"],
        t_electrode_idx=key["electrode_idx"],
        t_phoneme_pair=key["phoneme_pair"],
        t_word_end=key["word_end"],
        t_smin_early=key["smin_early"],
        t_smax_early=key["smax_early"],
        t_smin_late=key["smin"],
        t_smax_late=key["smax"],
    )
    early_early_outcomes.append(ee_i)
    early_late_outcomes.append(el_i)
    late_late_outcomes.append(ll_i)
    late_early_outcomes.append(le_i)

early_early_outcomes_df = (
    pl.from_pandas(pd.concat(early_early_outcomes, ignore_index=True))
    .with_columns(pl.col("subject").cast(subject_enum),
                  pl.col("phoneme_pair").cast(phoneme_pair_enum),
                  pl.col("word_end").cast(word_end_enum))
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"]))
early_late_outcomes_df = (
    pl.from_pandas(pd.concat(early_late_outcomes, ignore_index=True))
    .with_columns(pl.col("subject").cast(subject_enum),
                  pl.col("phoneme_pair").cast(phoneme_pair_enum),
                  pl.col("word_end").cast(word_end_enum))
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"]))
late_late_outcomes_df = (
    pl.from_pandas(pd.concat(late_late_outcomes, ignore_index=True))
    .with_columns(pl.col("subject").cast(subject_enum),
                  pl.col("phoneme_pair").cast(phoneme_pair_enum),
                  pl.col("word_end").cast(word_end_enum))
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"]))
late_early_outcomes_df = (
    pl.from_pandas(pd.concat(late_early_outcomes, ignore_index=True))
    .with_columns(pl.col("subject").cast(subject_enum),
                  pl.col("phoneme_pair").cast(phoneme_pair_enum),
                  pl.col("word_end").cast(word_end_enum))
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"]))

# %%
group_cols = ["subject", "electrode_idx", "phoneme_pair", "fold"]
roc_auc_kwargs = dict(
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=group_cols,
)
phonetic_transfer_results = (
    pl_roc_auc(early_early_outcomes_df, **roc_auc_kwargs, roc_auc_name="early_early_roc_auc")
    .join(
        pl_roc_auc(early_late_outcomes_df, **roc_auc_kwargs, roc_auc_name="early_late_roc_auc"),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner")
    .join(
        pl_roc_auc(late_early_outcomes_df, **roc_auc_kwargs, roc_auc_name="late_early_roc_auc"),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner")
    .join(
        pl_roc_auc(late_late_outcomes_df, **roc_auc_kwargs, roc_auc_name="late_late_roc_auc"),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner")

    # Join information about train early -> test early (should be a sanity check)
    .join(
        phon_roc_auc_searchlight_df.join(early_to_late_transfer_keys.select(["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"]),
                                         left_on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
                                         right_on=["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"],
                                         how="inner").rename({"phon_roc_auc": "early_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner")
    
    # Join information about train late -> test late (should be upper bound?? for the transfer case)
    .join(
        phon_roc_auc_searchlight_df.join(early_to_late_transfer_keys.select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]),
                                         on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
                                         how="inner").rename({"phon_roc_auc": "late_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        suffix="_late",
        how="inner"
    )
    .with_columns(
        # % of late ROC AUC achieved by early->late transfer
        (pl.col("early_late_roc_auc") / pl.col("late_roc_auc")).alias("early_to_late_normalized_roc_auc"),
        # % of early ROC AUC achieved by late->early transfer
        (pl.col("late_early_roc_auc") / pl.col("early_roc_auc")).alias("late_to_early_normalized_roc_auc"),
    )

    # TODO why are there dupes?
    .unique(subset=group_cols)
)

# %%
group_cols = ["subject", "electrode_idx", "phoneme_pair", "fold", "resampled"]

# compute accuracy within resampled step
phonetic_transfer_results_by_resampled = (
    early_early_outcomes_df
        .with_columns(((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias("correct"))
        .group_by(group_cols)
        .agg(pl.mean("correct").alias("early_early_accuracy"))
    .join(
        early_late_outcomes_df
            .with_columns(((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias("correct"))
            .group_by(group_cols)
            .agg(pl.mean("correct").alias("early_late_accuracy")),
        on=group_cols,
        how="inner")
    .join(
        late_early_outcomes_df
            .with_columns(((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias("correct"))
            .group_by(group_cols)
            .agg(pl.mean("correct").alias("late_early_accuracy")),
        on=group_cols,
        how="inner")
    .join(
        late_late_outcomes_df
            .with_columns(((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias("correct"))
            .group_by(group_cols)
            .agg(pl.mean("correct").alias("late_late_accuracy")),
        on=group_cols,
        how="inner")

    # Join information about train early -> test early (should be a sanity check)
    .join(
        phon_roc_auc_searchlight_df.join(early_to_late_transfer_keys.select(["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"]),
                                         left_on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
                                         right_on=["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"],
                                         how="inner").rename({"phon_roc_auc": "early_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner")
    
    # Join information about train late -> test late (should be upper bound?? for the transfer case)
    .join(
        phon_roc_auc_searchlight_df.join(early_to_late_transfer_keys.select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]),
                                         on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
                                         how="inner").rename({"phon_roc_auc": "late_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        suffix="_late",
        how="inner"
    )
)

# %%
g = sns.catplot(
    data=(
        phonetic_transfer_results
        .rename({"early_early_roc_auc": "Early window",
                 "late_late_roc_auc": "Late window",
                 "early_late_roc_auc": "Late window\n(transfer from early)",
                 "late_early_roc_auc": "Early window\n(transfer from late)"})
        .unpivot(
            on=["Early window", "Late window",
                "Late window\n(transfer from early)",
                "Early window\n(transfer from late)"],
            index=["subject", "electrode_idx", "phoneme_pair", "fold"],
            variable_name="Evaluation",
            value_name="roc_auc"
        )).to_pandas().query('phoneme_pair == "dn"'),
    # x="phoneme_pair", order=phoneme_pair_order,
    y="roc_auc",
    hue="Evaluation",
    kind="box")
g.set_axis_labels("", "Phonetic ROC AUC (%)")

for ax in g.axes.flat:
    ax.axhline(0.5, color="red", linestyle="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# %%
# sns.catplot(data=phonetic_transfer_results.unpivot(
#     on=["early_to_late_normalized_roc_auc", "late_to_early_normalized_roc_auc"],
#     index=["subject", "electrode_idx", "phoneme_pair", "fold"],
#     variable_name="transfer_direction",
#     value_name="normalized_roc_auc"
# ).to_pandas(),
#             x="phoneme_pair", order=phoneme_pair_order,
#             y="normalized_roc_auc",
#             hue="transfer_direction",
#             kind="box")

# %%
# g = sns.catplot(data=early_late_outcomes_df.to_pandas().assign(site=lambda xs: xs.subject.astype(str) + "_" + xs.electrode_idx.astype(str) + "_" + xs.phoneme_pair.astype(str)).astype({"resampled": int}),
#             x="resampled", y="decoder_proba", hue="lexical_evidence",
#             col="phoneme_pair", col_order=phoneme_pair_order,
#             kind="point", units="site",
#             height=3.5)
# g.set_axis_labels("Stimulus step", "Predicted\nP(second phoneme)")
# g.set_titles(col_template="Phoneme pair: {col_name}")
# for ax in g.axes.flat:
#     ax.axhline(0.5, color="red", linestyle="--")

# %%
# g = sns.catplot(data=late_early_outcomes_df.to_pandas().assign(site=lambda xs: xs.subject.astype(str) + "_" + xs.electrode_idx.astype(str) + "_" + xs.phoneme_pair.astype(str)),
#             x="resampled", y="decoder_proba", hue="lexical_evidence",
#             col="phoneme_pair", col_order=phoneme_pair_order,
#             kind="point", units="site",
#             height=3.5)
# for ax in g.axes.flat:
#     ax.axhline(0.5, color="red", linestyle="--")

# %%
# sns.catplot(data=phonetic_transfer_results_by_resampled.unpivot(
#     on=["early_early_accuracy", "early_late_accuracy", "late_early_accuracy", "late_late_accuracy"],
#     index=["subject", "electrode_idx", "phoneme_pair", "fold", "resampled"],
#     variable_name="transfer_condition",
#     value_name="accuracy"
# ).to_pandas(),
#             x="resampled", y="accuracy",
#             hue="transfer_condition", col="phoneme_pair", col_order=phoneme_pair_order,
#             kind="point")

# %%
# sns.catplot(data=phonetic_transfer_results_by_resampled.unpivot(
#     on=["early_early_accuracy", "early_late_accuracy", "late_early_accuracy", "late_late_accuracy"],
#     index=["subject", "electrode_idx", "phoneme_pair", "fold", "resampled"],
#     variable_name="transfer_condition",
#     value_name="accuracy"
# ).to_pandas(),
#             x="resampled", y="accuracy",
#             hue="transfer_condition",
#             col="electrode_idx", col_wrap=3, height=2.5, aspect=1.5,
#             kind="point")

# %%
# g = sns.catplot(data=
#     phonetic_transfer_results_by_resampled
#     .with_columns(
#         (pl.col("early_late_accuracy") / pl.col("late_late_accuracy")).alias("early_to_late_normalized_accuracy"),
#         (pl.col("late_early_accuracy") / pl.col("early_early_accuracy")).alias("late_to_early_normalized_accuracy"),
#     ).unpivot(
#         on=["early_to_late_normalized_accuracy", "late_to_early_normalized_accuracy"],
#         index=["subject", "electrode_idx", "phoneme_pair", "fold", "resampled"],
#         variable_name="transfer_direction",
#         value_name="normalized_accuracy"
#     ).to_pandas(),
#             x="resampled", y="normalized_accuracy",
#             hue="transfer_direction",
#             col="phoneme_pair", col_order=phoneme_pair_order,
#             kind="point")

# for ax in g.axes.flat:
#     ax.axhline(1, color="red", linestyle="--")

# %%
# Old method: ROC AUC across stimulus steps

# group_cols = ["subject", "electrode_idx", "phoneme_pair", "fold"]#, "resampled"]

# # compute accuracy within resampled step
# phonetic_transfer_results_controlled = (
#     pl_roc_auc(
#         early_early_outcomes_df
#         .filter(pl.col("follows_acoustics") == False),
#         **roc_auc_kwargs,
#         roc_auc_name="early_early_roc_auc")
#     .join(
#         pl_roc_auc(
#             early_late_outcomes_df
#                 .filter(pl.col("follows_acoustics") == False),
#             **roc_auc_kwargs,
#             roc_auc_name="early_late_roc_auc"),
#         on=group_cols,
#         how="inner")
#     .join(
#         pl_roc_auc(
#             late_early_outcomes_df
#                 .filter(pl.col("follows_acoustics") == False),
#             **roc_auc_kwargs,
#             roc_auc_name="late_early_roc_auc"),
#         on=group_cols,
#         how="inner")
#     .join(
#         pl_roc_auc(
#             late_late_outcomes_df
#                 .filter(pl.col("follows_acoustics") == False),
#             **roc_auc_kwargs,
#             roc_auc_name="late_late_roc_auc"),
#         on=group_cols,
#         how="inner")

#     # Join information about train early -> test early (should be a sanity check)
#     .join(
#         phon_roc_auc_searchlight_df.join(early_to_late_transfer_keys.select(["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"]),
#                                          left_on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
#                                          right_on=["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"],
#                                          how="inner").rename({"phon_roc_auc": "early_roc_auc"}),
#         on=["subject", "electrode_idx", "phoneme_pair", "fold"],
#         how="inner")
    
#     # Join information about train late -> test late (should be upper bound?? for the transfer case)
#     .join(
#         phon_roc_auc_searchlight_df.join(early_to_late_transfer_keys.select(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]),
#                                          on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
#                                          how="inner").rename({"phon_roc_auc": "late_roc_auc",
#                                                               "smin": "smin_late", "smax": "smax_late"}),
#         on=["subject", "electrode_idx", "phoneme_pair", "fold"],
#         how="inner"
#     )
# )

# %%
# group_cols = ["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"]#, "resampled"]
# controlled_resampled_step = 3

# # compute accuracy within resampled step
# phonetic_transfer_results_controlled = (
#     early_early_outcomes_df
#         .filter(pl.col("follows_acoustics") == False,
#                 pl.col("resampled") == controlled_resampled_step)
#         .with_columns(
#             ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias("correct"))
#         .group_by(group_cols)
#         .agg(pl.mean("correct").alias("early_early_accuracy"),
#              pl.count("correct").alias("n_trials"))
        
#         .join(
#             early_late_outcomes_df
#                 .filter(pl.col("follows_acoustics") == False,
#                         pl.col("resampled") == controlled_resampled_step)
#                 .with_columns(
#                     ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias("correct"))
#                 .group_by(group_cols)
#                 .agg(pl.mean("correct").alias("early_late_accuracy")),
#             on=group_cols,
#             how="inner")
#         .join(
#             late_early_outcomes_df
#                 .filter(pl.col("follows_acoustics") == False,
#                         pl.col("resampled") == controlled_resampled_step)
#                 .with_columns(
#                     ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias("correct"))
#                 .group_by(group_cols)
#                 .agg(pl.mean("correct").alias("late_early_accuracy")),
#             on=group_cols,
#             how="inner")
#         .join(
#             late_late_outcomes_df
#                 .filter(pl.col("follows_acoustics") == False,
#                         pl.col("resampled") == controlled_resampled_step)
#                 .with_columns(
#                     ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias("correct"))
#                 .group_by(group_cols)
#                 .agg(pl.mean("correct").alias("late_late_accuracy")),
#             on=group_cols,
#             how="inner")

#         # Join information about train early -> test early (should be a sanity check)
#         .join(
#             phon_roc_auc_searchlight_df.join(early_to_late_transfer_keys.select(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin_early", "smax_early"]),
#                                             left_on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
#                                             right_on=["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"],
#                                             how="inner").rename({"phon_roc_auc": "early_roc_auc"}),
#             on=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
#             how="inner")
        
#         # Join information about train late -> test late (should be upper bound?? for the transfer case)
#         .join(
#             phon_roc_auc_searchlight_df.join(early_to_late_transfer_keys.select(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]),
#                                             on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
#                                             how="inner").rename({"phon_roc_auc": "late_roc_auc",
#                                                                 "smin": "smin_late", "smax": "smax_late"}),
#             on=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
#             how="inner"
#         )
# )

# %%
# transfer_n_trials = phonetic_transfer_results_controlled \
#     .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"]) \
#     .agg(pl.min("n_trials").alias("n_trials")) \
#     .to_pandas().set_index(["subject", "electrode_idx", "phoneme_pair", "word_end"])["n_trials"]
# transfer_n_trials

# %%
# transfer_plot_df = phonetic_transfer_results_controlled.unpivot(
#     on=["early_early_accuracy", "early_late_accuracy", "late_early_accuracy", "late_late_accuracy"],
#     index=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],#, "resampled"],
#     variable_name="transfer_condition",
#     value_name="accuracy"
# ).to_pandas().assign(site=lambda xs: xs.subject.astype(str) + "_" + xs.electrode_idx.astype(str) + "_" + xs.phoneme_pair.astype(str) + "_" + xs.word_end.astype(str))
# transfer_plot_df = transfer_plot_df.merge(transfer_n_trials.reset_index(), on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="left")

# %%
# # Show all transfer mean results as a heatmap, one column per site
# transfer_pixel_df = phonetic_transfer_results_controlled.to_pandas() \
#     .assign(site=lambda xs: xs.subject.astype(str) + "_" + xs.electrode_idx.astype(str) + "_" + xs.phoneme_pair.astype(str) + "_" + xs.word_end.astype(str))
# transfer_pixel_df = transfer_pixel_df[["site", "early_early_accuracy", "early_late_accuracy", "late_early_accuracy", "late_late_accuracy"]] \
#     .groupby("site").mean()

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=4, random_state=0).fit(transfer_pixel_df.fillna(0).values)
# transfer_pixel_df = transfer_pixel_df.assign(cluster=kmeans.labels_)
# transfer_pixel_df = transfer_pixel_df.sort_values("cluster")
# transfer_pixel_df["cluster"] = transfer_pixel_df.cluster / transfer_pixel_df.cluster.max()

# sns.heatmap(transfer_pixel_df)

# %%
# g = sns.catplot(data=transfer_plot_df,
#             y="accuracy",
#             hue="transfer_condition", col="site", col_wrap=3,
#             col_order=transfer_plot_df.set_index("site").sort_values("n_trials", ascending=False).index.drop_duplicates(),
#             kind="box", sharey=False)

# for site, ax in g.axes_dict.items():
#     ax.axhline(0.5, color="red", linestyle="--")
#     data = transfer_plot_df.query("site == @site")
#     n_trials = data["n_trials"].iloc[0]
#     ax.set_title(f"{site}\n(n={n_trials} trials)")

# %% [markdown]
# #### Behavior

# %%
(
    behav_roc_auc_comparison_phon
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"])
    .agg(pl.mean("behav_roc_auc_improvement"))
    .sort("behav_roc_auc_improvement", descending=True)
    .filter(pl.col("behav_roc_auc_improvement") > 0.01)
)

# %%
# TODO transfer analysis on these

# %% [markdown]
# ## Zoomin

# %%
#  phon_peaks_df.join(behav_peaks_df,
#     on=["subject", "electrode_idx", "phoneme_pair"])
#  .sort("phon_roc_auc", descending=True) \
zoomin_keys = phon_peaks_df.join(behav_peaks_df,
                                 on=["subject", "electrode_idx", "phoneme_pair"]) \
    .select(["subject", "electrode_idx", "phoneme_pair", "word_end", "phon_roc_auc", "behav_roc_auc_improvement"])
# zoomin_keys = (
#     behav_peaks_df
#     .select(["subject", "electrode_idx", "phoneme_pair", "word_end", "behav_roc_auc_improvement"])
# )
zoomin_keys


# %%
def zoomin(subject, electrode_idx, phoneme_pair):
    # phonetic predictions
    subplot_phon_phon_df = (
        plot_phon_phon_df.filter(
            (pl.col("electrode_idx") == electrode_idx) &
            (pl.col("subject") == subject) &
            (pl.col("phoneme_pair") == phoneme_pair)
        )
    )
    subplot_behav_phon_df = (
        plot_behav_phon_df.filter(
            (pl.col("electrode_idx") == electrode_idx) &
            (pl.col("subject") == subject) &
            (pl.col("phoneme_pair") == phoneme_pair)
        )
    )

    # behav predictions
    subplot_behav_behav_df = (
        plot_behav_behav_df.filter(
            (pl.col("electrode_idx") == electrode_idx) &
            (pl.col("subject") == subject) &
            (pl.col("phoneme_pair") == phoneme_pair)
        )
    )
    subplot_phon_behav_df = (
        plot_phon_behav_df.filter(
            (pl.col("electrode_idx") == electrode_idx) &
            (pl.col("subject") == subject) &
            (pl.col("phoneme_pair") == phoneme_pair)
        )
    )

    assert subplot_phon_phon_df.select(pl.n_unique("smin")).item() == 1
    assert subplot_behav_phon_df.group_by("word_end").agg(pl.n_unique("smin")).select(pl.max("smin")).item() == 1
    assert subplot_behav_behav_df.group_by("word_end").agg(pl.n_unique("smin")).select(pl.max("smin")).item() == 1
    assert subplot_phon_behav_df.select(pl.n_unique("smin")).item() == 1

    # assert subplot_phon_phon_df.smin.nunique() == 1
    # assert subplot_behav_phon_df.groupby("word_end").smin.nunique().max() == 1
    # assert subplot_behav_behav_df.groupby("word_end").smin.nunique().max() == 1
    # assert subplot_phon_behav_df.smin.nunique() == 1

    subplot_phon_df = pl.concat([
        subplot_phon_phon_df.with_columns(pl.lit("phon").alias("source")),
        subplot_behav_phon_df.with_columns(pl.lit("behav").alias("source")),
    ])

    subplot_behav_df = pl.concat([
        subplot_phon_behav_df.with_columns(pl.lit("phon").alias("source")),
        subplot_behav_behav_df.with_columns(pl.lit("behav").alias("source")),
    ], how="align").join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"], how="left")

    print("Phonetic window:", subplot_phon_phon_df.select(pl.min("smin")).item(), "-", subplot_phon_phon_df.select(pl.max("smax")).item())
    print("Behavioral window:", subplot_behav_behav_df.select(pl.min("smin")).item(), "-", subplot_behav_behav_df.select(pl.max("smax")).item())

    # sub_phon_acc_change = phon_acc_change.query(f"electrode_idx== {electrode_idx} and subject == '{subject}' and phoneme_pair == '{phoneme_pair}'")
    # sub_behav_roc_auc = behav_roc_auc.query(f"electrode_idx== {electrode_idx} and subject == '{subject}' and phoneme_pair == '{phoneme_pair}'")
    # sub_behav_roc_auc_improvement = behav_roc_auc_improvement.query(f"electrode_idx== {electrode_idx} and subject == '{subject}' and phoneme_pair == '{phoneme_pair}'")

    # # Also compute overall phonetic and behavioral ROC-AUC
    # phon_phon_roc_auc = roc_auc_score(subplot_phon_phon_df.decoder_target, subplot_phon_phon_df.decoder_proba)
    # behav_phon_roc_auc = roc_auc_score(subplot_behav_phon_df.decoder_target, subplot_behav_phon_df.decoder_proba)
    # behav_behav_roc_auc = roc_auc_score(subplot_behav_behav_df.decoder_target, subplot_behav_behav_df.full_decoder_proba)
    # phon_behav_roc_auc = roc_auc_score(subplot_phon_behav_df.decoder_target, subplot_phon_behav_df.full_decoder_proba)
    # roc_auc_heatmap_df = pd.DataFrame({
    #     "source": ["phon", "phon", "behav", "behav"],
    #     "target": ["phon", "behav", "phon", "behav"],
    #     "roc_auc": [phon_phon_roc_auc, phon_behav_roc_auc, behav_phon_roc_auc, behav_behav_roc_auc]
    # }).pivot(index="source", columns="target", values="roc_auc")

    # f, ax = plt.subplots(figsize=(4, 4))
    # sns.heatmap(roc_auc_heatmap_df, annot=True, vmin=0.5, vmax=1.0, cmap="Blues", ax=ax)

    g_phon = sns.catplot(
        data=subplot_phon_df.to_pandas(),
        x="resampled", y="decoder_proba", hue="lexical_evidence",
        col="source", col_order=["phon", "behav"],
        kind="point", height=4)
    for ax in g_phon.axes.flat:
        ax.axhline(0.5, color="red", linestyle="--")

    # g_phon_acc_change = sns.catplot(
    #     data=sub_phon_acc_change,
    #     x="resampled", y="acc_diff", hue="lexical_evidence",
    #     kind="point", height=3)
    # g_phon_acc_change.figure.suptitle("Improvement in phonetic decoding\nat phon - behav window")
    # for ax in g_phon_acc_change.axes.flat:
    #     ax.axhline(0, color="red", linestyle="--")

    g_behav = sns.catplot(
        data=subplot_behav_df.to_pandas(),
        x="resampled", y="full_decoder_proba", hue="decoder_target",
        col="source", col_order=["phon", "behav"],
        kind="point", height=4)
    for ax in g_behav.axes.flat:
        ax.axhline(0.5, color="red", linestyle="--")

    # g_behav_roc_auc = sns.catplot(
    #     data=sub_behav_roc_auc,
    #     x="resampled", y="roc_auc",
    #     hue="source", col="word_end",
    #     kind="point", height=3)
    # g_behav_roc_auc.set_axis_labels("Resampled", "ROC AUC")
    # for ax in g_behav_roc_auc.axes.flat:
    #     ax.axhline(0.5, color="red", linestyle="--")

    # g_behav_roc_auc_improvement = sns.catplot(
    #     data=sub_behav_roc_auc_improvement,
    #     x="resampled", y="roc_auc",
    #     hue="source", col="word_end",
    #     kind="point", height=3)
    # g_behav_roc_auc_improvement.set_axis_labels("Resampled", "ROC AUC Improvement\nover Baseline")
    # for ax in g_behav_roc_auc_improvement.axes.flat:
    #     ax.axhline(0, color="red", linestyle="--")

    epochs_i = epochs[subject]
    epoch_data = epochs_i.get_data(picks=electrode_idx).squeeze(1)

    plot_tmin = -0.1
    plot_tmax = word_end_df.filter(pl.col("phoneme_pair") == phoneme_pair).select(pl.max("word_end_offset")).item() + 0.1
    plot_smin = int((plot_tmin - epoch_tmin) * epoch_sfreq)
    plot_smax = int((plot_tmax - epoch_tmin) * epoch_sfreq)    
    epoch_data = epoch_data[:, plot_smin:plot_smax]

    plot_highlight_phon_window = subplot_phon_phon_df.select(["smin", "smax"]).unique().to_numpy().flatten()
    plot_highlight_behav_window = subplot_behav_behav_df.select(["smin", "smax"]).unique().to_numpy().flatten()

    highlight_phon_times = epochs_i.times[[plot_highlight_phon_window[0], plot_highlight_phon_window[1]]]
    highlight_behav_times = epochs_i.times[[plot_highlight_behav_window[0], plot_highlight_behav_window[1]]]

    ### HGA plot by stimulus step

    plot_epoch_keys = subplot_phon_phon_df.select(["epoch_idx", "resampled", "textgrid_path"]).unique()
    palette = sns.color_palette("coolwarm", n_colors=6)
    f_hga_by_phon, ax = plt.subplots(figsize=(5, 3))

    for resampled_value, color in zip(range(1, 7), palette):
        epoch_idxs = plot_epoch_keys.filter(pl.col("resampled") == resampled_value).select("epoch_idx").to_series()
        mean_epoch = epoch_data[epoch_idxs, :].mean(axis=0)
        times = epochs_i.times[plot_smin:plot_smax]
        ax.plot(times, mean_epoch, label=str(resampled_value), color=color)
    ax.legend(title="Stimulus step", loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.axvspan(highlight_phon_times[0], highlight_phon_times[-1], color="gray", alpha=0.3)
    ax.axvspan(highlight_behav_times[0], highlight_behav_times[-1], color="yellow", alpha=0.3)

    ## HGA plot by behavior

    plot_epoch_keys = subplot_behav_df.filter(pl.col("source") == "behav").select(["epoch_idx", "decoder_target", "textgrid_path"]).unique()
    palette = sns.color_palette("Set1", n_colors=2)
    f_hga_by_behav, ax = plt.subplots(figsize=(5, 3))

    for decoder_target_value, color in zip([0, 1], palette):
        epoch_idxs = plot_epoch_keys.filter(pl.col("decoder_target") == decoder_target_value).select("epoch_idx").to_series()
        mean_epoch = epoch_data[epoch_idxs, :].mean(axis=0)
        times = epochs_i.times[plot_smin:plot_smax]
        ax.plot(times, mean_epoch, label=str(decoder_target_value), color=color)
    ax.legend(title="Behavioral choice", loc="upper right", bbox_to_anchor=(1.3, 1))
    ax.axvspan(highlight_phon_times[0], highlight_phon_times[-1], color="gray", alpha=0.3)
    ax.axvspan(highlight_behav_times[0], highlight_behav_times[-1], color="yellow", alpha=0.3)

    add_textgrid_single(ax, textgrid_dir, ep_df=plot_epoch_keys.to_pandas())

    return g_phon, g_behav, f_hga_by_phon, f_hga_by_behav
    # return g_phon, g_phon_acc_change, g_behav, g_behav_roc_auc, g_behav_roc_auc_improvement


# %%
zoomin_search_hga(paper_data, "EC278", 38, "dn", "necessary",
                  controlled_resampled_search_steps=(3,4),
                  textgrid_dir=textgrid_dir)

# %%
# pdfpages render
cols_per_page = 3
max_num_pages = np.inf

from matplotlib.backends.backend_pdf import PdfPages
outf = "hga_zoomin_search.pdf"
hga_zoomin_keys = zoomin_keys.unique(["subject", "electrode_idx", "phoneme_pair", "word_end"]) \
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])

# # # pre-process epoch data
# # epoch_data_preprocessed = {
# #     subject: epochs[subject].copy().apply_baseline().get_data()
# #     for subject in hga_zoomin_keys.select("subject").unique().to_series()
# # }

# with PdfPages(outf) as pdf:
#     row_iter = iter(tqdm(hga_zoomin_keys.iter_rows(named=True), total=hga_zoomin_keys.height))

#     i = 0
#     try:
#         while True:
#             fig, axs = plt.subplots(5, cols_per_page, figsize=(cols_per_page * 3, 5 * 2.5),
#                                     sharex=False)
#             for i in range(cols_per_page):
#                 row = next(row_iter)
#                 subject = row["subject"]
#                 electrode_idx = row["electrode_idx"]
#                 phoneme_pair = row["phoneme_pair"]
#                 word_end = row["word_end"]

#                 print(subject, electrode_idx, phoneme_pair, word_end)
#                 zoomin_search_hga(subject, electrode_idx, phoneme_pair, word_end,
#                                   controlled_resampled_search_steps=[2, 3, 4, 5],
#                                 #   epoch_data=epoch_data_preprocessed[subject],
#                                   axs=axs[:, i])
#                 axs[0, i].set_title(
#                     f"Subject {subject}, Electrode {electrode_idx}, {phoneme_pair}, {word_end}",
#                     pad=20)
                
#             fig.tight_layout()
#             pdf.savefig(fig)
#             plt.close(fig)

#             i += 1
#             if i >= max_num_pages:
#                 print(f"Reached max number of pages ({max_num_pages}), stopping.")
#                 break
#     except StopIteration:
#         pass

#     # if we have a partially filled page, save it as well
#     if (hga_zoomin_keys.height % cols_per_page) != 0:
#         pdf.savefig(fig)
#         plt.close(fig)

# %%
hga_zoomin_keys.to_pandas().to_csv("hga_zoomin_search_keys.csv", index=False)

# %%
# # pdfpages render
# from matplotlib.backends.backend_pdf import PdfPages
# outf = "neurometrics_zoomin.pdf"
# hga_zoomin_keys = zoomin_keys.unique(["subject", "electrode_idx", "phoneme_pair"])
# with PdfPages(outf) as pdf:
#     for row in tqdm(hga_zoomin_keys.iter_rows(named=True), total=hga_zoomin_keys.height):
#         subject = row["subject"]
#         electrode_idx = row["electrode_idx"]
#         phoneme_pair = row["phoneme_pair"]
#         f = zoomin_hga(subject, electrode_idx, phoneme_pair)
#         pdf.savefig(f)
#         plt.close(f)

# %% [markdown]
# ## Star plots

# %%
star_plot_kwargs = dict(
    controlled_resampled_steps=[3, 4],
    figsize=(4, 4),
    include_phonemes=False,
    resampled_palette=resampled_palette_simplified,
    textgrid_dir=textgrid_dir,
)

# %%
fig = zoomin_hga(
    paper_data,
    "EC250", 185, "dn", "desolate",
    hide_bottom=False, legend=False, **star_plot_kwargs)
fig.savefig("figures/zoomin_EC250_185_dn_desolate.pdf")
plt.close(fig)
None

legend_fig = plt.figure(figsize=(2, 2))

legend_handles_labels = fig.axes[0].get_legend_handles_labels()
# reverse sort
legend_handles_labels = (legend_handles_labels[0][::-1], legend_handles_labels[1][::-1])
for handle in legend_handles_labels[0]:
    handle.set_linewidth(3)
    handle.set_color("black")
legend_fig.legend(*legend_handles_labels, loc="center", frameon=True)

legend_fig.savefig("figures/zoomin_legend.pdf")

# %%
zoomin_hga(
    paper_data,
    "EC278", 38, "dn", "necessary",
    hide_bottom=True, **star_plot_kwargs)
plt.gcf().savefig("figures/zoomin_EC278_38_dn_necessary.pdf")
None

# %% [markdown]
# ## Quant HGA search

# %%
find_site_windows(paper_data,
                  subject="EC278", electrode_idx=38, phoneme_pair="dn", word_end="necessary")

# %%
import src.viz_paper

# %%
hga_df = src.viz_paper.extract_hga_windows_df(paper_data, zoomin_keys=zoomin_keys)

# %%
# compute per-site sign relationship between phonetic options in early window
early_polarity = hga_df.groupby(["subject", "electrode_idx", "phoneme_pair", "word_end", "decoder_target"]) \
    .hga_early.mean().reset_index() \
    .set_index("decoder_target") \
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"]) \
    .apply(lambda xs: np.sign(xs.loc[1] - xs.loc[0])) \
    .rename(columns={"hga_early": "early_polarity"})

# compute per-site sign relationship between behavioral options in late window
late_polarity = hga_df.groupby(["subject", "electrode_idx", "phoneme_pair", "word_end", "behavior_dummy_forced"]) \
    .hga_late.mean().reset_index() \
    .set_index("behavior_dummy_forced") \
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"]) \
    .apply(lambda xs: np.sign(xs.loc[1] - xs.loc[0])) \
    .rename(columns={"hga_late": "late_polarity"})

reg_df = pd.merge(
    hga_df,
    pd.merge(
        early_polarity.reset_index(),
        late_polarity.reset_index(),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"]
    ),
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"]
)
reg_df["hga_early_signed"] = reg_df["hga_early"] * reg_df["early_polarity"]
reg_df["hga_late_signed"] = reg_df["hga_late"] * reg_df["late_polarity"]
reg_df["is_ambiguous"] = reg_df.apply(
    lambda xs: str(int(xs.resampled)) in xs.behav_steps_chosen
    if xs.behav_steps_chosen is not None else np.nan, axis=1)

# %%
polarity_contingency = early_polarity.join(late_polarity, on=["subject", "electrode_idx", "phoneme_pair", "word_end"]) \
    .query("phoneme_pair == 'dn'") \
    .value_counts() \
    .unstack("late_polarity")
polarity_contingency

# %%
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(polarity_contingency.fillna(0))
print(f"Chi-squared test: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")

# %%
# sns.displot(data=hga_df, x="phon_tmax")

# %%
reg_df.to_csv("reg_df.csv")

# %%
sns.catplot(data=reg_df, x="phoneme_pair", hue="decoder_target", y="hga_early_signed",
            kind="box", height=3, aspect=1.5)
from scipy.stats import ttest_ind
ttest_ind(reg_df.query("decoder_target == 0")["hga_early_signed"],
          reg_df.query("decoder_target == 1")["hga_early_signed"])

# %%
sns.catplot(data=reg_df, x="early_polarity", hue="decoder_target", y="hga_early",
            kind="box")

# %%
sns.catplot(data=reg_df, hue="behavior_dummy_forced", y="hga_late_signed",
            kind="box")
ttest_ind(reg_df.dropna().query("behavior_dummy_forced == 0")["hga_late_signed"],
          reg_df.dropna().query("behavior_dummy_forced == 1")["hga_late_signed"])

# %%
sns.catplot(data=reg_df, x="late_polarity", hue="behavior_dummy_forced", y="hga_late",
            kind="box")

# %%
# Do some basic normalization to aid a visual t-test
#
# reg_df_norm = pd.merge(reg_df,
#          reg_df.groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"])[["hga_early_signed", "hga_late_signed"]].mean(),
#             on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
#             suffixes=("", "_site_mean"))
# reg_df_norm["hga_early_signed_norm"] = reg_df_norm["hga_early_signed"] - reg_df_norm["hga_early_signed_site_mean"]
# reg_df_norm["hga_late_signed_norm"] = reg_df_norm["hga_late_signed"] - reg_df_norm["hga_late_signed_site_mean"]

# sns.catplot(data=reg_df_norm.melt(id_vars=["subject", "electrode_idx", "phoneme_pair", "word_end", "decoder_target", "behavior_dummy_forced"],
#                             value_vars=["hga_early_signed_norm", "hga_late_signed_norm"],
#                             var_name="window", value_name="hga_signed"),
#             x="window", hue="decoder_target", y="hga_signed",
#             kind="box")

# sns.catplot(data=reg_df_norm.melt(id_vars=["subject", "electrode_idx", "phoneme_pair", "word_end", "decoder_target", "behavior_dummy_forced"],
#                             value_vars=["hga_early_signed_norm", "hga_late_signed_norm"],
#                             var_name="window", value_name="hga_signed"),
#             x="window", hue="behavior_dummy_forced", y="hga_signed",
#             kind="box")

# %% [markdown]
# ### Overall HGA

# %%
pcc_epoch_data_cache = {}

# %%
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPatch

class HandlerRectangle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                    xdescent, ydescent, width, height, fontsize, trans):
        # Use the rectangle's height attribute for scaling
        rect_height = orig_handle.get_height()
        center = 0.5 * height
        p = Rectangle(xy=(-xdescent, center - rect_height * height / 2),
                    width=width, height=rect_height * height,
                    facecolor=orig_handle.get_facecolor(),
                    alpha=orig_handle.get_alpha())
        return [p]

def plot_condition_contrast(plot_df,
                            condition_variable,
                            polarity_correct: Literal[None, "early", "late"] = None,
                            epoch_data_cache=None,
                            ax=None,
                            annotate=True,
                            label=None,
                            textgrid_kwargs=None,
                            vline_extent=1.25,
                            ttest_window_size=8, ttest_window_stride=8,
                            ttest_bar_height_ratio=0.04, ttest_bar_y_ratio=0.95,
                            color=None):
    # compute per-site contrast between phonetic options
    hga_condition_results = []
    if epoch_data_cache is None:
        epoch_data_cache = {}

    for (subject, electrode_idx, phoneme_pair, word_end), rows in plot_df.group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"]):
        epochs_i = epochs[subject]
        if subject not in epoch_data_cache:
            epoch_data_cache[subject] = epochs_i.copy().apply_baseline().get_data()
        epoch_data = epoch_data_cache[subject][:, electrode_idx, :]

        md0 = rows.filter(pl.col(condition_variable) <= 0)
        md1 = rows.filter(pl.col(condition_variable) > 0)

        idxs0 = md0.select(pl.col("epoch_idx")).to_series().unique()
        idxs1 = md1.select(pl.col("epoch_idx")).to_series().unique()

        if len(idxs0) == 0 or len(idxs1) == 0:
            continue

        # compute between-condition difference of means
        epochs_0 = epoch_data[idxs0]
        epochs_1 = epoch_data[idxs1]

        diff_of_means = epochs_1.mean(axis=0) - epochs_0.mean(axis=0)
        hga_condition_results.append({
            "subject": subject,
            "electrode_idx": electrode_idx,
            "phoneme_pair": phoneme_pair,
            "word_end": word_end,
            "times": epochs_i.times,
            "diff_of_means": diff_of_means
        })

    hga_condition_results_df = (
        pl.DataFrame(hga_condition_results)
        .join(pl.from_pandas(early_polarity.reset_index()),
            on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
            how="left")
        .join(pl.from_pandas(late_polarity.reset_index()),
            on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
            how="left"))

    if ax is None:
        f, ax = plt.subplots(figsize=(5, 3))

    plot_rows = hga_condition_results_df
    plot_times = plot_rows.select(pl.col("times"))[0].item()
    plot_diffs = np.stack(plot_rows.select(pl.col("diff_of_means")).to_numpy()[:, 0])

    if polarity_correct is not None:
        plot_signs = plot_rows.select(pl.col(f"{polarity_correct}_polarity")).to_numpy().flatten()
        plot_diffs *= plot_signs[:, np.newaxis]

    plot_diffs = plot_diffs[~np.isnan(plot_diffs).any(axis=1)]

    plot_diff_mean = plot_diffs.mean(axis=0)
    plot_diff_sem = plot_diffs.std(axis=0) / np.sqrt(plot_diffs.shape[0])

    ax.plot(plot_times, plot_diff_mean, label=label,
            linewidth=2, color=color)
    ax.fill_between(plot_times, plot_diff_mean - plot_diff_sem, plot_diff_mean + plot_diff_sem,
                    alpha=0.3, color=color)

    # sliding-window one-sample t-test: diff_of_means vs. zero
    from scipy.stats import ttest_1samp
    n_times = plot_diffs.shape[1]
    ttest_results = []
    for start in range(0, n_times - ttest_window_size + 1, ttest_window_stride):
        window_means = plot_diffs[:, start:start + ttest_window_size].mean(axis=1)
        t_stat, p_value = ttest_1samp(window_means, 0)
        end = min(start + ttest_window_size, n_times - 1)
        ttest_results.append((plot_times[start], plot_times[end], t_stat, p_value))

    # define p-value thresholds and corresponding heights
    p_threshold_height_mults = [1.0, 0.5, 0.25]
    p_thresholds = list(zip([0.00001, 0.0001, 0.001], p_threshold_height_mults))
    p_thresholds = list(zip([0.0001, 0.001, 0.01], p_threshold_height_mults))

    ymin, ymax = ax.get_ylim()
    base_bar_height = (ymax - ymin) * ttest_bar_height_ratio
    bar_y = ymin + (ymax - ymin) * ttest_bar_y_ratio

    for tmin_w, tmax_w, t_stat, p_value in ttest_results:
        # determine height multiplier based on p-value
        height_mult = None
        for p_thresh, mult in p_thresholds:
            if p_value < p_thresh:
                height_mult = mult
                break
        
        if height_mult is not None:
            color = ax.lines[-1].get_color()
            ax.barh(y=bar_y, width=tmax_w - tmin_w, left=tmin_w,
                    height=base_bar_height * height_mult, color=color, alpha=0.5, 
                    edgecolor="none")

    if annotate:
        textgrid_default_kwargs = dict(include_offset=True, vline_extent=vline_extent)
        textgrid_kwargs = {
            **textgrid_default_kwargs,
            **(textgrid_kwargs if textgrid_kwargs is not None else {})}
        add_textgrid(ax, textgrid_dir, plot_df.to_pandas(), **textgrid_kwargs)

        # add legend annotations for p-value thresholds
        p_handles = [
            Rectangle((0, 0), 1, height_mult, facecolor='gray', alpha=0.5, label=f'p < {p_thresh:g}'.replace("-0", "-"))
            for p_thresh, height_mult in p_thresholds
        ]
        p_labels = [h.get_label() for h in p_handles]
        # get existing legend handles/labels if they exist
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + p_handles, 
                labels=labels + p_labels,
                handler_map={Rectangle: HandlerRectangle()},
                loc='best', fontsize=8)

        pod_time = paper_data.word_end_df.filter(pl.col("word_end") == word_end).select(pl.max("pod")).item()
        ax.axvline(pod_time, linestyle="--", linewidth=2,
                alpha=0.5, color="red",
                ymax=vline_extent, clip_on=False)

        ax.set_ylim(-0.1, ax.get_ylim()[1])
        ax.axhline(0, linestyle="--", color="gray", alpha=0.7)
        sns.despine(ax=ax, top=True, right=True)

        return ax, p_handles, p_labels
    
    return ax, None, None


# %%
f, axs = plt.subplots(2, 1, figsize=(5, 2 * 2.5), sharex=True)

plot_word_end = "necessary"
plot_xlim = (0, 1.0)
plot_palette = sns.color_palette(categorical_palette, 2)

plot_condition_contrast(
    (plot_phon_phon_df.filter(pl.col("resampled").is_in([1, 6]),
                              pl.col("word_end") == plot_word_end)),
    "categorical_acoustic_cue",
    polarity_correct="early",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=axs[0], color=plot_palette[0])

ambiguous_keys = pl.DataFrame(
    [(subject, phoneme_pair, word_end, resampled)
     for (subject, phoneme_pair, word_end), resampled_list in paper_data.ambiguous_resampled_steps.items()
     for resampled in resampled_list],
    schema=pl.Schema({
        "subject": subject_enum,
        "phoneme_pair": phoneme_pair_enum,
        "word_end": word_end_enum,
        "resampled": pl.Float32
    }))
plot_behav_rows = (
    ambiguous_keys
    .join(plot_phon_phon_df, on=["subject", "phoneme_pair", "word_end", "resampled"], how="inner")
    .filter(pl.col("word_end") == plot_word_end,
            pl.col("resampled").is_in([3, 4])
            ))
plot_condition_contrast(
    plot_behav_rows,
    "behavior_dummy_forced",
    polarity_correct="late",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=axs[1], color=plot_palette[1],
    vline_extent=1.2,
    textgrid_kwargs=dict(include_phonemes=False))

axs[0].set_xlim(*plot_xlim)
axs[0].set_ylabel("HGA effect size ($z$)")
axs[1].set_xlim(*plot_xlim)
axs[1].set_xlabel("Time from word onset (s)")
axs[1].set_ylabel("HGA effect size ($z$)")


# %%
def plot_condition_contrasts_single_figure(
        plot_word_end, plot_xlim=(0, 1.2)):
    f, ax = plt.subplots(figsize=(2.5, 2))

    plot_palette = sns.color_palette(categorical_palette, 2)

    _, p_handles, p_labels = plot_condition_contrast(
        (plot_phon_phon_df.filter(pl.col("resampled").is_in([1, 6]),
                                pl.col("word_end") == plot_word_end)),
        "categorical_acoustic_cue",
        polarity_correct="early",
        epoch_data_cache=pcc_epoch_data_cache,
        ax=ax, color=plot_palette[0],
        annotate=True, textgrid_kwargs=dict(fontsize=8),
        label="Phonetic\ncontrast",)

    ambiguous_keys = pl.DataFrame(
        [(subject, phoneme_pair, word_end, resampled)
        for (subject, phoneme_pair, word_end), resampled_list in paper_data.ambiguous_resampled_steps.items()
        for resampled in resampled_list],
        schema=pl.Schema({
            "subject": subject_enum,
            "phoneme_pair": phoneme_pair_enum,
            "word_end": word_end_enum,
            "resampled": pl.Float32
        }))
    plot_behav_rows = (
        ambiguous_keys
        .join(plot_phon_phon_df, on=["subject", "phoneme_pair", "word_end", "resampled"], how="inner")
        .filter(pl.col("word_end") == plot_word_end))
    plot_condition_contrast(
        plot_behav_rows,
        "behavior_dummy_forced",
        polarity_correct="late",
        epoch_data_cache=pcc_epoch_data_cache,
        ax=ax, color=plot_palette[1], annotate=False, label="Behavioral\ncontrast",
        vline_extent=1.2,
        textgrid_kwargs=dict(include_phonemes=False),
        ttest_bar_y_ratio=0.87)

    ax.set_xlim(*plot_xlim)
    ax.set_ylabel("HGA effect size ($z$)")
    ax.set_xlabel("Time from word onset (s)")

    handles, labels = ax.get_legend_handles_labels()
    handles += p_handles
    labels += p_labels
    ax.legend(
        handles=handles, labels=labels,
        handler_map={Rectangle: HandlerRectangle()},
        fontsize=10,
        loc="upper right", bbox_to_anchor=(1.6, 1.15))
    
    return f


# %%
f = plot_condition_contrasts_single_figure("necessary")
f.savefig("figures/condition_contrasts.pdf")

# %%
plot_condition_contrasts_single_figure("desolate")
None


# %%
def plot_condition_contrast_peak_aligned(
    plot_df,
    behav_peaks_df,
    condition_variable,
    polarity_correct: Literal[None, "early", "late"] = None,
    epoch_data_cache=None,
    ax=None,
    label=None,
    window_sec=0.3,
    ttest_window_size=4, ttest_window_stride=4,
    ttest_bar_height_ratio=0.04, ttest_bar_y_ratio=0.95,
):
    """
    Like plot_condition_contrast, but aligns each electrode's difference-of-means
    to its behavioral peak (from behav_peaks_df) rather than word onset.

    Parameters
    ----------
    behav_peaks_df : pl.DataFrame
        Must contain columns: subject, electrode_idx, phoneme_pair, word_end, smin, smax.
        The peak center is computed as (smin + smax) / 2.
    window_sec : float
        Half-width of the window around the peak to display, in seconds.
    """
    # Build lookup: (subject, electrode_idx, phoneme_pair, word_end) -> peak center sample
    peak_centers = {}
    for row in behav_peaks_df.iter_rows(named=True):
        key = (row["subject"], row["electrode_idx"], row["phoneme_pair"], row["word_end"])
        peak_centers[key] = (row["smin"] + row["smax"]) / 2

    hga_condition_results = []
    if epoch_data_cache is None:
        epoch_data_cache = {}

    window_samples = int(window_sec * epoch_sfreq)

    for (subject, electrode_idx, phoneme_pair, word_end), rows in plot_df.group_by(
        ["subject", "electrode_idx", "phoneme_pair", "word_end"]
    ):
        key = (subject, electrode_idx, phoneme_pair, word_end)
        if key not in peak_centers:
            continue

        peak_sample = int(round(peak_centers[key]))

        epochs_i = epochs[subject]
        if subject not in epoch_data_cache:
            epoch_data_cache[subject] = epochs_i.copy().apply_baseline().get_data()
        epoch_data = epoch_data_cache[subject][:, electrode_idx, :]

        md0 = rows.filter(pl.col(condition_variable) <= 0)
        md1 = rows.filter(pl.col(condition_variable) > 0)

        idxs0 = md0.select(pl.col("epoch_idx")).to_series()
        idxs1 = md1.select(pl.col("epoch_idx")).to_series()

        if len(idxs0) == 0 or len(idxs1) == 0:
            continue

        epochs_0 = epoch_data[idxs0]
        epochs_1 = epoch_data[idxs1]
        diff_of_means = epochs_1.mean(axis=0) - epochs_0.mean(axis=0)

        # Extract window around behavioral peak
        s_start = peak_sample - window_samples
        s_end = peak_sample + window_samples
        n_total = diff_of_means.shape[0]
        if s_start < 0 or s_end > n_total:
            continue

        diff_aligned = diff_of_means[s_start:s_end]

        hga_condition_results.append({
            "subject": subject,
            "electrode_idx": electrode_idx,
            "phoneme_pair": phoneme_pair,
            "word_end": word_end,
            "diff_aligned": diff_aligned,
        })

    hga_condition_results_df = (
        pl.DataFrame(hga_condition_results)
        .join(pl.from_pandas(early_polarity.reset_index()),
              on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="left")
        .join(pl.from_pandas(late_polarity.reset_index()),
              on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="left")
    )

    if ax is None:
        f, ax = plt.subplots(figsize=(5, 3))

    plot_rows = hga_condition_results_df.filter(pl.col("phoneme_pair") == "dn")
    plot_diffs = np.stack(plot_rows.select(pl.col("diff_aligned")).to_numpy()[:, 0])

    if polarity_correct is not None:
        plot_signs = plot_rows.select(pl.col(f"{polarity_correct}_polarity")).to_numpy().flatten()
        plot_diffs *= plot_signs[:, np.newaxis]

    plot_diffs = plot_diffs[~np.isnan(plot_diffs).any(axis=1)]

    # Relative time axis centered on behavioral peak
    rel_times = np.linspace(-window_sec, window_sec, plot_diffs.shape[1])

    plot_diff_mean = plot_diffs.mean(axis=0)
    plot_diff_sem = plot_diffs.std(axis=0) / np.sqrt(plot_diffs.shape[0])

    ax.plot(rel_times, plot_diff_mean, label=label, linewidth=2)
    ax.fill_between(rel_times, plot_diff_mean - plot_diff_sem, plot_diff_mean + plot_diff_sem, alpha=0.3)

    # Sliding-window one-sample t-test
    from scipy.stats import ttest_1samp
    n_times = plot_diffs.shape[1]
    ttest_results = []
    for start in range(0, n_times - ttest_window_size + 1, ttest_window_stride):
        window_means = plot_diffs[:, start:start + ttest_window_size].mean(axis=1)
        t_stat, p_value = ttest_1samp(window_means, 0)
        end = min(start + ttest_window_size, n_times - 1)
        ttest_results.append((rel_times[start], rel_times[end], t_stat, p_value))

    ymin, ymax = ax.get_ylim()
    bar_height = (ymax - ymin) * ttest_bar_height_ratio
    bar_y = ymin + (ymax - ymin) * ttest_bar_y_ratio
    for tmin_w, tmax_w, t_stat, p_value in ttest_results:
        if p_value < 0.05:
            color = ax.lines[-1].get_color()
            ax.barh(y=bar_y, width=tmax_w - tmin_w, left=tmin_w,
                    height=bar_height, color=color, alpha=0.5, edgecolor="none")

    ax.axvline(0, linestyle="--", linewidth=2, alpha=0.5, color="red")
    ax.axhline(0, linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Time from behavioral peak (s)")
    ax.set_ylabel("HGA effect size ($z$)")
    ax.set_ylim(-0.1, ax.get_ylim()[1])
    sns.despine(ax=ax, top=True, right=True)

    return ax


# %%
f, ax = plt.subplots(figsize=(5, 3))
plot_condition_contrast_peak_aligned(
    plot_behav_rows,
    behav_peaks_df=behav_peaks_df,
    condition_variable="behavior_dummy_forced",
    polarity_correct="late",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=ax, label="Behavioral contrast",
    window_sec=0.3,
    ttest_window_size=4, ttest_window_stride=4,
    ttest_bar_height_ratio=0.04, ttest_bar_y_ratio=0.87
)
ax.set_xlim(-0.3, 0.3)


# %% [markdown]
# ## Behav stackplot

# %%
# show behavior stackplot not grouped by anything
def show_behav_stackplot(subject, phoneme_pair, resampled):
    md_i = all_md.to_pandas().query("subject == @subject and phoneme_pair == @phoneme_pair and resampled == @resampled")
    behav_counts = md_i.groupby("label_behavior").size()
    total = behav_counts.sum()

    # Get unique behaviors for consistent coloring
    all_behaviors = md_i['label_behavior'].unique()
    colors = plt.cm.Set3(range(len(all_behaviors)))
    color_map = {behavior: colors[i] for i, behavior in enumerate(all_behaviors)}
    
    # stacked bar plot, just one bar, showing proportional allocation to each behavior
    # Create the stacked bar
    bottom = 0
    f, axs = plt.subplots(1, 2, figsize=(5, 1.3))
    
    for i, (behavior, count) in enumerate(behav_counts.items()):
        axs[0].barh(0, count, left=bottom, label=behavior, color=color_map[behavior])
        
        # Add label at center of bar
        center = bottom + count / 2
        percentage = (count / total) * 100
        axs[0].text(center, 0, f'Heard /{behavior}/\n({percentage:.0f}%)', 
                ha='center', va='center', fontsize=10)
        
        bottom += count
    
    axs[0].set_yticks([])
    axs[0].set_xticklabels([])
    axs[0].set_xlim(0, bottom)
    axs[0].set_xticks([])

    # now group by lexical evidence
    grouped = md_i.groupby(['word_end', 'label_behavior'], observed=True).size().unstack(fill_value=0)

    # Create figure with multiple rows (one per lexical category)
    n_lexical = len(grouped)
    for idx, (word_end, behav_counts) in enumerate(grouped.iterrows()):
        bottom = 0
        total = behav_counts.sum()
        
        for behavior, count in behav_counts.items():
            if count > 0:  # Only plot non-zero counts
                axs[1].barh(idx, count, left=bottom, color=color_map[behavior])
                
                # Add label at center of bar
                center = bottom + count / 2
                percentage = (count / total) * 100
                label = f"$\it{{{behavior}{word_end[1:]}}}$"
                axs[1].text(center, idx, f'{label}\n({percentage:.0f}%)', 
                        ha='center', va='center', fontsize=10)
                
                bottom += count
        
    axs[1].set_yticks(range(n_lexical))
    axs[1].set_yticklabels(["-" + label[1:] for label in grouped.index])  # Show word end labels
    axs[1].set_xticklabels([])
    axs[1].set_xlim(0, bottom)
    axs[1].set_xticks([])

    sns.despine(ax=axs[0], left=True, bottom=True)
    sns.despine(ax=axs[1], left=True, bottom=True)

    f.tight_layout()
    return f


# %%
# show behavior stackplot not grouped by anything
def show_behav_stackplot2(subject, phoneme_pair, label_word_end=None, resampled_set=(1, 3, 6),
                          filter_word_end=None,
                          figsize=(2.5, 2.8)):
    md_i = all_md.to_pandas().query("subject == @subject and phoneme_pair == @phoneme_pair")
    if filter_word_end is not None:
        md_i = md_i.query("word_end == @filter_word_end")
    
    # # Get unique behaviors for consistent coloring
    all_behaviors = ["d", "n"]
    colors = plt.cm.Set3(range(len(all_behaviors)))
    color_map = {behavior: colors[i] for i, behavior in enumerate(all_behaviors)}

    f, ax = plt.subplots(1, 1, figsize=figsize)
    axs = [ax]
    
    grouped = md_i[md_i.resampled.isin(list(resampled_set))].astype({"resampled": int}).groupby(['resampled', 'label_behavior'], observed=True).size().unstack(fill_value=0)

    # Create figure with multiple rows (one per lexical category)
    n_lexical = len(grouped)
    for idx, (resampled, behav_counts) in enumerate(grouped.iterrows()):
        bottom = 0
        total = behav_counts.sum()
        
        for behavior, count in behav_counts.items():
            if count > 0:  # Only plot non-zero counts
                # y = n_lexical - 1 - idx  # reverse order so that resampled=1 is at the top
                y = idx
                axs[0].barh(y, count, left=bottom, color=color_map[behavior])
                
                # Add label at center of bar
                center = bottom + count / 2
                percentage = (count / total) * 100
                if label_word_end is None:
                    label = f"Heard /{behavior}/"
                else:
                    label_word_end_here = label_word_end
                    if percentage < 25:
                        label_word_end_here = ""
                    elif percentage < 30:
                        label_word_end_here = label_word_end[:2] + "…"
                    label = f"$\\it{{{behavior}{label_word_end_here}}}$"
                axs[0].text(center, y, f'{label}\n({percentage:.0f}%)', 
                        ha='center', va='center', fontsize=9)
                
                bottom += count

    axs[0].set_yticks(range(n_lexical))
    axs[0].set_yticklabels(grouped.index)
    axs[0].set_xticklabels([])
    axs[0].set_xlim(0, bottom)
    axs[0].set_xticks([])
    sns.despine(ax=axs[0], left=True, bottom=True)

    f.tight_layout()
    return f


# %%
# show behavior stackplot grouped by lexical evidence
def show_behav_stackplot_lexical(subject, phoneme_pair, resampled):
    md_i = all_md.to_pandas().query("subject == @subject and phoneme_pair == @phoneme_pair and resampled == @resampled")
    
    # Group by both label_lexical and label_behavior
    grouped = md_i.groupby(['word_end', 'label_behavior'], observed=True).size().unstack(fill_value=0)
    
    # Get unique behaviors for consistent coloring
    all_behaviors = md_i['label_behavior'].unique()
    colors = plt.cm.Set3(range(len(all_behaviors)))
    color_map = {behavior: colors[i] for i, behavior in enumerate(all_behaviors)}
    
    # Create figure with multiple rows (one per lexical category)
    n_lexical = len(grouped)
    f, ax = plt.subplots(figsize=(10, 1 + 0.5 * n_lexical))
        
    for idx, (word_end, behav_counts) in enumerate(grouped.iterrows()):
        bottom = 0
        total = behav_counts.sum()
        
        for behavior, count in behav_counts.items():
            if count > 0:  # Only plot non-zero counts
                ax.barh(idx, count, left=bottom, color=color_map[behavior])
                
                # Add label at center of bar
                center = bottom + count / 2
                percentage = (count / total) * 100
                ax.text(center, idx, f'Heard /{behavior}/\n({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=14, fontweight='bold')
                
                bottom += count
        
    ax.set_yticks(range(n_lexical))
    ax.set_yticklabels(["-" + label[1:] for label in grouped.index])  # Show word end labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(0, bottom)
    
    f.tight_layout()
    return f


# %%
# show_behav_stackplot2("EC250", "dn", label_word_end="esolate", resampled_set=(1,2,3,4,5,6))
# None

# %%
# show_behav_stackplot2("EC250", "dn", label_word_end="esolate", resampled_set=(1,2,3,4,5,6),
#                       filter_word_end="desolate")
# None

# %%
def plot_behav_barplot(plot_subject, plot_phoneme_pair, plot_word_end, plot_resampled_steps,
                       figsize=(2.3, 2.3), resampled_palette=resampled_palette_simplified,
                       legend=True, plot_values: Literal["count", "proportion"] = "proportion",
                       ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)

    behav_barplot_data = (
        all_md.to_pandas()
        .query(f"subject == '{plot_subject}' and phoneme_pair == '{plot_phoneme_pair}' and word_end == '{plot_word_end}' and resampled in {plot_resampled_steps}")
        [["resampled", "label_behavior"]].value_counts()
    )
    max_num_trials = behav_barplot_data.groupby("resampled").sum().max()

    # reindex to get all the levels we'd expect
    full_index = pd.MultiIndex.from_product(
        [plot_resampled_steps, list(plot_phoneme_pair)],
        names=["resampled", "label_behavior"]
    )
    behav_barplot_data = behav_barplot_data.reindex(full_index, fill_value=0).reset_index(name="count")
    behav_barplot_data["resampled_inv"] = 7 - behav_barplot_data["resampled"]
    behav_barplot_data = (
        behav_barplot_data
        .astype({"resampled": int})
        .sort_values(["label_behavior", "resampled"], ascending=False)
    )

    # total count per resampled level
    totals = (
        behav_barplot_data
        .groupby("resampled")["count"]
        .sum()
        .to_dict()
    )

    # proportion column
    behav_barplot_data["prop"] = (
        behav_barplot_data["count"] /
        behav_barplot_data["resampled"].map(totals)
    )

    sns.barplot(
        data=behav_barplot_data,
        y="resampled", x="prop", order=plot_resampled_steps[::-1],
        hue="label_behavior", hue_order=list(plot_phoneme_pair)[::-1],
        palette="viridis",
        orient="h", width=0.8)

    # match solid / dashed linestyle aesthetic with a solid/hatched pattern here
    behavior_styles = {"d": "", "n": "//"}
    linestyles = {"d": "solid", "n": "dashed"}
    for patch, (_, row) in zip(ax.patches, behav_barplot_data.iterrows()):
        patch.set_facecolor(resampled_palette[row["resampled"] - 1])
        patch.set_alpha(1)
        # patch.set_hatch(behavior_styles[row["label_behavior"]])
        # patch.set_linestyle(linestyles[row["label_behavior"]])
        # patch.set_edgecolor("black")
        # patch.set_linewidth(1.5)

        # draw dashed line overlay for 'n' bars
        if row["label_behavior"] == "n":
            x = patch.get_x()
            width = patch.get_width()
            y_center = patch.get_y() + patch.get_height() / 2
            ax.plot(
                [x, x + width],
                [y_center, y_center],
                color="black",
                linestyle="--",
                linewidth=1.5,
                solid_capstyle="butt",
                clip_on=True,
            )

        # --- annotation ---
        width = patch.get_width()
        y_center = patch.get_y() + patch.get_height() / 2

        prop = row["prop"]
        # label = f"{row['label_behavior']}{plot_word_end[1:]}\n{prop:.0%}"
        label = f"{prop:.0%}"

        # ax.text(
        #     width + 0.5,        # horizontal offset (tune)
        #     y_center,
        #     label,
        #     va="center",
        #     ha="left",
        #     fontsize=11
        # )

    max_width = behav_barplot_data["count"].max()

    if plot_values == "count":
        # For count values
        ax.set_xlim(0, max_width * 1.45)
        ax.set_xticks([0, max_num_trials // 2, max_num_trials])
        ax.set_xlabel("# trials")
    elif plot_values == "proportion":
        # For proportion values
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlabel("% trials")
    ax.set_ylabel("step")
    # ax.invert_xaxis()
    ax.tick_params(axis="both", labelsize=12)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            hatch="//",
            label=f"Chose\n$\\it{{{plot_phoneme_pair[1]}{plot_word_end[1:]}}}$"
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            hatch="",
            label=f"Chose\n$\\it{{{plot_phoneme_pair[0]}{plot_word_end[1:]}}}$"
        ),
    ]

    if legend:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=10,
            handlelength=3,
            handleheight=2,
            loc="center right",
            bbox_to_anchor=(1.25, 0.5)
        )
    else:
        ax.get_legend().remove()

    sns.despine(ax=ax, top=True, bottom=False, left=False, right=True)


# %%
plot_behav_barplot("EC250", "dn", "desolate", [1, 3, 4, 6], legend=False)
plt.gcf().savefig("figures/behav_barplot_EC250_dn_desolate.pdf")

# %%
plot_behav_barplot("EC278", "dn", "necessary", [1, 3, 4, 6], legend=False)
plt.gcf().savefig("figures/behav_barplot_EC278_dn_necessary.pdf")
