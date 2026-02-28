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
# Neurometric response functions relating single-neuron activity to both phonetic content and subsequent behavioral choice.
# Compare two neural responses:
# 1. early phonetic response
# 2. late feedback response
# in the same electrodes.

# %%
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mne
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
from scipy.stats import ttest_ind
from tqdm.auto import tqdm

tqdm.pandas()

# %%
matplotlib.rcParams.update(
    {
        "figure.dpi": 300,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.25,
        "ytick.minor.width": 0.25,
        "lines.linewidth": 1.0,
        "font.family": "Helvetica",
        "font.sans-serif": ["Helvetica", "Arial"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    }
)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from src.data import add_metadata_features
from src.stimuli import (
    OFFSET_DICT,
    PHONEME_PAIR_TO_WORD_ENDS,
    POD_dict,
)
from src.viz_paper import (
    PaperData,
    add_textgrid,
    evaluate_phonetic_transfer,
    p_to_stars,
    phoneme_pair_enum,
    pl_roc_auc,
    plot_behav_barplot,
    plot_condition_contrasts_single_figure,
    subject_enum,
    word_end_enum,
    zoomin_hga,
)

# %%
sns.set_context("paper", font_scale=1.25)

# %% tags=["parameters"]
all_epochs = list(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))

phonetic_searchlight_paths = list(
    Path("outputs/causal4/behavior_decoding_single_electrode_acoustic/").glob("*")
)

transfer_results_paths = list(
    Path("outputs/causal4/behavior_decoding_single_electrode_transfer").glob(
        "*/transfer_results.csv"
    )
)

neurometrics_dir = "outputs/causal4/prepare_neurometrics"

epoch_tmin = -0.4
epoch_sfreq = 100

relative_performance_twidth = 0.2
relative_performance_swidth = int(relative_performance_twidth * epoch_sfreq)

textgrid_dir = "textgrids"

outdir = "outputs/causal4/A_neurometrics"

max_plot_rows = 15
phoneme_pair_order = ["bm", "dn", "pb"]
source_order = ["phon", "behav"]

# shared palette for categorical variables
categorical_palette = "Set2"

# %%
resampled_palette = sns.color_palette("cool", n_colors=6)

# simplified resampled palette contrasting ambiguous vs unambiguous
resampled_palette_simplified = (
    [resampled_palette[0]] + (4 * [resampled_palette[2]]) + [resampled_palette[5]]
)

# %% [markdown]
# ## Prepare helpers

# %%
epochs = {}
for path in all_epochs:
    subject = re.findall(r"(EC[\d]+)_epo", str(path))[0]
    ep_i = mne.read_epochs(path, verbose=False)
    ep_i.metadata = add_metadata_features(ep_i.metadata)
    epochs[subject] = ep_i

# %%
# Load saved phonetic decoders
phonetic_decoder_checkpoints = {
    subject: torch.load(
        f"outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/results.pt"
    )
    for subject in tqdm(epochs.keys())
}

# %%
# Load precomputed neurometrics data
neurometrics_path = Path(neurometrics_dir)

electrode_df = pl.read_parquet(neurometrics_path / "electrode_df.parquet").with_columns(
    pl.col("subject").cast(subject_enum)
)
plot_phon_phon_df = pl.read_parquet(
    neurometrics_path / "plot_phon_phon_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
plot_behav_phon_df = pl.read_parquet(
    neurometrics_path / "plot_behav_phon_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
plot_behav_behav_df = pl.read_parquet(
    neurometrics_path / "plot_behav_behav_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
plot_phon_behav_df = pl.read_parquet(
    neurometrics_path / "plot_phon_behav_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
behav_roc_auc_searchlight_df = pl.read_parquet(
    neurometrics_path / "behav_roc_auc_searchlight_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
phon_roc_auc_searchlight_df = pl.read_parquet(
    neurometrics_path / "phon_roc_auc_searchlight_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)
all_md = pl.read_parquet(neurometrics_path / "all_md.parquet").with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
word_end_df = pl.read_parquet(neurometrics_path / "word_end_df.parquet").with_columns(
    pl.col("word_end").cast(word_end_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)
phon_peaks_df = pl.read_parquet(
    neurometrics_path / "phon_peaks_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)
behav_peaks_df = pl.read_parquet(
    neurometrics_path / "behav_peaks_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
behav_peaks_df_unfiltered = pl.read_parquet(
    neurometrics_path / "behav_peaks_df_unfiltered.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
behav_baseline_df = pl.read_parquet(
    neurometrics_path / "behav_baseline_df.parquet"
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
zoomin_keys = pl.read_parquet(neurometrics_path / "zoomin_keys.parquet").with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
early_polarity = pd.read_parquet(
    neurometrics_path / "early_polarity.parquet"
).set_index(["subject", "electrode_idx", "phoneme_pair", "word_end"])
late_polarity = pd.read_parquet(neurometrics_path / "late_polarity.parquet").set_index(
    ["subject", "electrode_idx", "phoneme_pair", "word_end"]
)
hga_df = pd.read_parquet(neurometrics_path / "hga_df.parquet")
reg_df = pd.read_parquet(neurometrics_path / "reg_df.parquet")

# DEV filter phon peaks
# print(phon_peaks_df.height)
# phon_peaks_df = phon_peaks_df.filter(pl.col("phon_roc_auc") > 0.64)
# print("-> filtered phon peaks:", phon_peaks_df.height)

# DEV filter behav peaks
print(behav_peaks_df.height)
behav_peaks_df = behav_peaks_df.filter(pl.col("behav_roc_auc_improvement") > 0.01)
print("-> filtered behav peaks:", behav_peaks_df.height)

# %%
# Check: are behav peaks a subset of phon peaks?
(
    pd.merge(
        behav_peaks_df.to_pandas().drop_duplicates(
            ["subject", "electrode_idx", "phoneme_pair"]
        ),
        phon_peaks_df.to_pandas().drop_duplicates(
            ["subject", "electrode_idx", "phoneme_pair"]
        ),
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="outer",
        indicator=True,
    ).drop_duplicates(["subject", "electrode_idx"])
)._merge.value_counts()

# %% [markdown]
# ### Follow through on peak filtering consequences

# %%
plot_phon_phon_keys = phon_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
)

# merge in case necessary
plot_phon_phon_df = plot_phon_phon_df.join(
    plot_phon_phon_keys,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="inner",
)

plot_behav_phon_keys = behav_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "word_end"]
)

# merge in case necessary
plot_behav_phon_df = plot_behav_phon_df.join(
    plot_behav_phon_keys,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "word_end"],
    how="inner",
)

# %%
plot_phon_behav_keys = phon_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
)
# merge in case necessary
plot_phon_behav_df = plot_phon_behav_df.join(
    plot_phon_behav_keys,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="inner",
)

plot_behav_behav_keys = behav_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
)
plot_behav_behav_keys_unfiltered = behav_peaks_df_unfiltered.select(
    ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
)
plot_behav_behav_df = plot_behav_behav_df.join(
    plot_behav_behav_keys,
    on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
    how="inner",
)

# %%
# filter based on possibly filtered peaks
join_keys = ["subject", "electrode_idx", "phoneme_pair"]
early_polarity = early_polarity.reset_index().merge(
    phon_peaks_df.to_pandas()[join_keys], on=join_keys, how="inner"
)

join_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
late_polarity = late_polarity.merge(
    behav_peaks_df.to_pandas()[join_keys], on=join_keys, how="inner"
)

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
# Derived; cheap to recompute from precomputed searchlight data
phon_roc_auc_mean_df = phon_roc_auc_searchlight_df.group_by(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
).agg(pl.col("phon_roc_auc").mean())

behav_roc_auc_mean_df = behav_roc_auc_searchlight_df.group_by(
    ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
).agg(
    [
        pl.col("behav_roc_auc").mean(),
        pl.col("behav_roc_auc_baseline").mean(),
        pl.col("behav_roc_auc_improvement").mean(),
    ]
)

# %% [markdown]
# ## Electrode distribution

# %%
electrode_distribution_df = (
    phon_peaks_df.join(
        (
            behav_peaks_df.group_by(["subject", "electrode_idx", "phoneme_pair"]).agg(
                pl.max("behav_roc_auc_improvement")
            )
        ),
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="left",
    )
    .group_by(["subject", "electrode_idx"])
    .agg(
        pl.col("phon_roc_auc").is_not_null().any().alias("phonetic_selective"),
        pl.col("behav_roc_auc_improvement")
        .is_not_null()
        .any()
        .alias("behavior_selective"),
    )
    .group_by("subject")
    .agg(
        pl.sum("phonetic_selective").alias("phonetic_selective"),
        pl.sum("behavior_selective").alias("behavior_selective"),
    )
    # join in speech responsive facts
    .join(
        electrode_df.group_by("subject").agg(
            pl.sum("speech_responsive").alias("speech_responsive"),
            pl.len().alias("total_electrodes"),
        ),
        on="subject",
        how="left",
    )
    .sort("total_electrodes", descending=True)
).to_pandas()
electrode_distribution_df

# %%
# fig, ax = plt.subplots(figsize=(3, 0.3 * len(electrode_distribution_df)))

# total = electrode_distribution_df["total_electrodes"].values
# speech = electrode_distribution_df["speech_responsive"].values
# phonetic = electrode_distribution_df["phonetic_selective"].values
# behavior = electrode_distribution_df["behavior_selective"].values

# plot_palette = sns.color_palette(categorical_palette, 3)

# # derive complementary counts
# non_speech = total - speech
# speech_not_phonetic = speech - phonetic

# y = np.arange(len(electrode_distribution_df))
# # first (leftmost): phonetic
# ax.barh(
#     y,
#     phonetic,
#     color=plot_palette[0],
#     label="Phonetically\nselective",
#     alpha=0.9,
# )

# # --- overlay: behaviorally selective (hatched) ---
# ax.barh(
#     y,
#     behavior,
#     left=0,
#     facecolor="none",
#     edgecolor="k",
#     hatch="///",
#     linewidth=0.0,
#     label="Behaviorally\nselective",
#     zorder=5,
# )

# # second: speech-responsive but not phonetic
# ax.barh(
#     y,
#     speech_not_phonetic,
#     color=plot_palette[1],
#     left=phonetic,
#     label="Task-responsive\n(non-phonetic)",
#     alpha=0.7,
# )

# # third: non-speech electrodes
# ax.barh(
#     y,
#     non_speech,
#     color=plot_palette[2],
#     left=phonetic + speech_not_phonetic,
#     label="Other",
#     alpha=0.5,
# )

# ax.set_yticks(y)
# ax.set_yticklabels(electrode_distribution_df.subject, rotation=45, ha="right")
# ax.set_xlabel("Number of electrodes")
# ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.0), fontsize=9)

# fig.savefig("figures/electrode_distribution.pdf")

# %%
fig, ax = plt.subplots(figsize=(1.8, 2.2))

electrode_distribution_df_plot = electrode_distribution_df.copy()
electrode_distribution_df_plot["proportion_behavior"] = (
    electrode_distribution_df_plot["behavior_selective"]
    / electrode_distribution_df_plot["phonetic_selective"]
)
electrode_distribution_df_plot = electrode_distribution_df_plot.sort_values(
    "proportion_behavior", ascending=False
)
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
ax.barh(
    y,
    behavior_of_phonetic,
    color=plot_palette[0],
    label="Behaviorally\nselective",
    alpha=0.9,
)

# stack on top: phonetically selective but not behaviorally selective
ax.barh(
    y,
    non_behavior_of_phonetic,
    left=behavior_of_phonetic,
    color="gray",
    label=None,  # "Phonetically\nselective",
    alpha=0.9,
)


ax.set_yticks(y)
ax.set_yticklabels(electrode_distribution_df_plot["index"], ha="right")
ax.set_xlabel("% of phonetically\nselective electrodes")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.set_xlim(0, 1)
ax.legend(loc="lower left", fontsize=10)
sns.despine(ax=ax, top=True, right=True)

fig.savefig("figures/electrode_distribution-phonetic_selective.pdf")

# %% [markdown]
# ## Plot neurometric for phonetic targets

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
        [
            pl.col("subject"),
            pl.col("electrode_idx").cast(pl.Utf8),
            pl.col("phoneme_pair"),
        ],
        separator="_",
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
phon_acc = (
    plot_phon_df.with_columns(
        ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5))
        .cast(pl.Float32)
        .alias("correct")
    )
    .group_by(
        [
            "source",
            "site",
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "lexical_evidence",
            "resampled",
            "fold",
        ]
    )
    .agg(pl.col("correct").mean().alias("accuracy"))
)

# %%
phon_acc_change = (
    phon_acc.pivot(
        values="accuracy",
        index=[
            "site",
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "resampled",
            "lexical_evidence",
            "fold",
        ],
        on="source",
        aggregate_function="first",
    )
    .drop_nulls(["phon", "behav"])
    .with_columns((pl.col("phon") - pl.col("behav")).alias("acc_diff"))
)

# %% [markdown]
# ## Behav prediction

# %%
# summarize: per electrode, selective for one completion or both?
(
    behav_peaks_df.with_columns(
        (
            pl.col("word_end").cast(pl.String).str.slice(0, 1)
            == pl.col("phoneme_pair").cast(pl.String).str.slice(0, 1)
        )
        .cast(pl.UInt8)
        .alias("lexical_evidence")
    )
    .pivot(
        index=["subject", "electrode_idx", "phoneme_pair"],
        on="lexical_evidence",
        values="behav_roc_auc_improvement",
        aggregate_function="len",
    )
    .with_columns(
        (pl.col("0").is_not_null() & pl.col("1").is_not_null()).alias(
            "both_completions"
        ),
        (pl.col("0").is_not_null() & pl.col("1").is_null()).alias(
            "first_completion_only"
        ),
        (pl.col("0").is_null() & pl.col("1").is_not_null()).alias(
            "second_completion_only"
        ),
    )
    .group_by("phoneme_pair")
    .agg(
        pl.sum("both_completions").alias("both_completions"),
        pl.sum("first_completion_only").alias("first_completion_only"),
        pl.sum("second_completion_only").alias("second_completion_only"),
    )
)

# %%
plot_behav_df = (
    pl.concat(
        [
            plot_phon_behav_df.with_columns(pl.lit("phon").alias("source")),
            plot_behav_behav_df.with_columns(pl.lit("behav").alias("source")),
        ],
        how="align",
    )
    .with_columns(
        pl.concat_str(
            [
                pl.col("subject"),
                pl.col("electrode_idx").cast(pl.Utf8),
                pl.col("phoneme_pair"),
            ],
            separator="_",
        ).alias("site")
    )
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"], how="left")
)

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
behav_roc_auc = (
    pl_roc_auc(
        df=plot_behav_df,
        target_col="decoder_target",
        proba_col="full_decoder_proba",
        group_cols=[
            "source",
            "site",
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "word_end",
            "resampled",
            "smin",
            "smax",
            "fold",
        ],
        roc_auc_name="behav_roc_auc",
    )
    .join(
        pl_roc_auc(
            df=plot_behav_df,
            target_col="decoder_target",
            proba_col="baseline_decoder_proba",
            group_cols=[
                "source",
                "site",
                "subject",
                "electrode_idx",
                "phoneme_pair",
                "word_end",
                "resampled",
                "smin",
                "smax",
                "fold",
            ],
            roc_auc_name="behav_roc_auc_baseline",
        ),
        on=[
            "source",
            "site",
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "word_end",
            "resampled",
            "smin",
            "smax",
            "fold",
        ],
        how="inner",
    )
    .with_columns(
        (pl.col("behav_roc_auc") - pl.col("behav_roc_auc_baseline")).alias(
            "behav_roc_auc_improvement"
        ),
        pl.concat_str(
            [
                pl.col("subject"),
                pl.col("electrode_idx").cast(pl.Utf8),
                pl.col("phoneme_pair"),
            ],
            separator="_",
        ).alias("site"),
    )
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
# ## Dynamics

# %% [markdown]
# ### Decoding timecourse

# %%
# sns.lineplot(
#     data=phon_roc_auc_mean_df.to_pandas(),
#     x="smax",
#     y="phon_roc_auc",
#     hue="phoneme_pair",
#     hue_order=phoneme_pair_order,
# )

# %%
# sns.lineplot(
#     data=(
#         behav_roc_auc_mean_df
#         .join(plot_behav_behav_keys, on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="inner")
#     ).to_pandas(),
#     x="smax",
#     y="behav_roc_auc_improvement",
#     hue="phoneme_pair",
#     hue_order=phoneme_pair_order,
# )

# %%
# # above but with median
# plot_textgrid = "11_necessary_dn_002.TextGrid"
# behav_median_timecourse = (
#     behav_roc_auc_mean_df.to_pandas()
#     .query("word_end == 'necessary'")
#     .groupby(["phoneme_pair", "smin", "smax"])
#     .behav_roc_auc_improvement.median()
#     .reset_index()
# )
# behav_median_timecourse["tmin"] = (
#     behav_median_timecourse["smin"] / epoch_sfreq + epoch_tmin
# )
# behav_median_timecourse["tmax"] = (
#     behav_median_timecourse["smax"] / epoch_sfreq + epoch_tmin
# )
# ax = sns.lineplot(
#     data=behav_median_timecourse,
#     x="tmax",
#     y="behav_roc_auc_improvement",
#     hue="phoneme_pair",
#     hue_order=phoneme_pair_order,
# )
# add_textgrid(ax, textgrid_dir, textgrid_file=plot_textgrid)

# %% [markdown]
# ### Peak timing

# %%
peak_timing_plot = pl.concat(
    [
        phon_peaks_df.with_columns(pl.lit("phon").alias("source")),
        behav_peaks_df.with_columns(pl.lit("behav").alias("source")),
    ],
    how="align",
).with_columns(
    (((pl.col("smin") + pl.col("smax")) / 2) / epoch_sfreq + epoch_tmin).alias(
        "t_center"
    )
)

# %%
g = sns.displot(
    data=peak_timing_plot.to_pandas(),
    x="t_center",
    hue="phoneme_pair",
    hue_order=phoneme_pair_order,
    row="source",
    row_order=source_order,
    kind="kde",
    clip=(0, None),
    height=2.5,
    aspect=2.5,
)
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
plot_word_end = "necessary"
plot_phoneme_pair = "dn"
plot_textgrid = "11_necessary_dn_002.TextGrid"
plot_xlim = (0, 1.2)
vline_extent = 1.1
g = sns.displot(
    data=peak_timing_plot.filter(
        (pl.col("word_end") == plot_word_end)
        | ((pl.col("source") == "phon") & (pl.col("phoneme_pair") == plot_phoneme_pair))
    )
    .to_pandas()
    .assign(
        source=lambda df: df.source.map({"phon": "Acoustic", "behav": "Perceptual"})
    ),
    x="t_center",
    hue="source",
    hue_order={"Acoustic": 0, "Perceptual": 1},
    palette=categorical_palette,
    linewidth=2,
    # kind="hist", stat="density", common_norm=False,
    kind="kde",
    common_norm=False,
    clip=(0, None),
    # legend=False,
    height=2,
    aspect=2.75 / 2,
)
g.set_axis_labels("Peak decoding time (s)", "Density")
sns.move_legend(
    g, "upper right", bbox_to_anchor=(0.65, 0.93), fontsize=10, frameon=True, title=None
)

for (row, col, hue), data in g.facet_data():
    ax = g.axes[row][col]
    ax.set_xlim(plot_xlim)

    phoneme_pair = data.phoneme_pair.iloc[0]
    word_stim_info = word_end_df.filter(pl.col("word_end") == plot_word_end)
    # for word_end in word_stim_info.select("word_end_offset").to_series():
    #     ax.axvline(word_end, color="red", linestyle="--")
    pod = word_stim_info.select("pod").unique().item()
    ax.axvline(
        pod,
        ymax=vline_extent,
        color="red",
        alpha=0.5,
        linewidth=2,
        linestyle="--",
        clip_on=False,
    )

    add_textgrid(
        ax,
        textgrid_dir,
        textgrid_file=plot_textgrid,
        include_phonemes=False,
        fontsize=9,
        vline_extent=vline_extent,
    )

g.savefig("figures/decoding_timing.pdf")

# %%
plot_word_end = "desolate"
plot_phoneme_pair = "dn"
plot_textgrid = "11_necessary_dn_002.TextGrid"
plot_xlim = (0, 0.7)
vline_extent = 1.1
g = sns.displot(
    data=peak_timing_plot.filter(
        (pl.col("word_end") == plot_word_end)
        | ((pl.col("source") == "phon") & (pl.col("phoneme_pair") == plot_phoneme_pair))
    )
    .to_pandas()
    .assign(
        source=lambda df: df.source.map({"phon": "Acoustic", "behav": "Perceptual"})
    ),
    x="t_center",
    hue="source",
    hue_order={"Acoustic": 0, "Perceptual": 1},
    palette=categorical_palette,
    linewidth=2,
    # kind="hist", stat="density", common_norm=False,
    kind="kde",
    common_norm=False,
    clip=(0, None),
    # legend=False,
    height=2,
    aspect=2.75 / 2,
)
g.set_axis_labels("Peak decoding time (s)", "Density")
sns.move_legend(
    g, "upper right", bbox_to_anchor=(0.65, 0.93), fontsize=10, frameon=True, title=None
)

for (row, col, hue), data in g.facet_data():
    ax = g.axes[row][col]
    ax.set_xlim(plot_xlim)

    phoneme_pair = data.phoneme_pair.iloc[0]
    word_stim_info = word_end_df.filter(pl.col("word_end") == plot_word_end)
    # for word_end in word_stim_info.select("word_end_offset").to_series():
    #     ax.axvline(word_end, color="red", linestyle="--")
    pod = word_stim_info.select("pod").unique().item()
    ax.axvline(
        pod,
        ymax=vline_extent,
        color="red",
        alpha=0.5,
        linewidth=2,
        linestyle="--",
        clip_on=False,
    )

    add_textgrid(
        ax,
        textgrid_dir,
        textgrid_file=plot_textgrid,
        include_phonemes=False,
        fontsize=9,
        vline_extent=vline_extent,
    )

g.savefig("figures/decoding_timing-desolate.pdf")

# %%
plot_word_ends = ["desolate", "necessary"]
plot_phoneme_pair = "dn"
plot_textgrid = "11_necessary_dn_002.TextGrid"
plot_xlim = (0, 1.2)
vline_extent = 1.1
g = sns.displot(
    data=peak_timing_plot.filter(
        (pl.col("word_end").is_in(plot_word_ends))
        | ((pl.col("source") == "phon") & (pl.col("phoneme_pair") == plot_phoneme_pair))
    )
    .to_pandas()
    .assign(
        source=lambda df: df.source.map({"phon": "Acoustic", "behav": "Perceptual"})
    ),
    x="t_center",
    hue="source",
    hue_order={"Acoustic": 0, "Perceptual": 1},
    palette=categorical_palette,
    linewidth=2,
    # kind="hist", stat="density", common_norm=False,
    kind="kde",
    common_norm=False,
    clip=(0, None),
    # legend=False,
    height=2,
    aspect=2.75 / 2,
)
g.set_axis_labels("Peak decoding time (s)", "Density")
sns.move_legend(
    g, "upper right", bbox_to_anchor=(0.65, 0.93), fontsize=10, frameon=True, title=None
)

for (row, col, hue), data in g.facet_data():
    ax = g.axes[row][col]
    ax.set_xlim(plot_xlim)

    phoneme_pair = data.phoneme_pair.iloc[0]
    word_stim_info = word_end_df.filter(pl.col("word_end").is_in(plot_word_ends))
    # for word_end in word_stim_info.select("word_end_offset").to_series():
    #     ax.axvline(word_end, color="red", linestyle="--")
    pod = word_stim_info.select("pod").unique().item()
    ax.axvline(
        pod,
        ymax=vline_extent,
        color="red",
        alpha=0.5,
        linewidth=2,
        linestyle="--",
        clip_on=False,
    )

    add_textgrid(
        ax,
        textgrid_dir,
        textgrid_file=plot_textgrid,
        include_phonemes=False,
        fontsize=9,
        vline_extent=vline_extent,
    )

g.savefig("figures/decoding_timing-both.pdf")

# %% [markdown]
# ### Peak timing of behavior relative to word offset

# %%
behav_peak_timing_plot = behav_peaks_df.with_columns(
    (((pl.col("smin") + pl.col("smax")) / 2) / epoch_sfreq + epoch_tmin).alias(
        "t_center"
    ),
    pl.col("word_end").replace_strict(OFFSET_DICT).alias("t_offset"),
    pl.col("phoneme_pair").replace_strict(POD_dict).alias("t_pod"),
).with_columns(
    (pl.col("t_center") - pl.col("t_offset")).alias("t_from_offset"),
    (pl.col("t_center") - pl.col("t_pod")).alias("t_from_pod"),
)

# %%
# g = sns.displot(
#     data=behav_peak_timing_plot.to_pandas(),
#     x="t_from_offset",
#     hue="phoneme_pair",
#     hue_order=phoneme_pair_order,
#     # hue="subject",
#     kind="kde",
#     height=2,
#     aspect=2.5,
# )
# g.set_axis_labels("Time from word offset (s)", "Density")

# %%
# g = sns.displot(
#     data=behav_peak_timing_plot.to_pandas(),
#     x="t_from_pod",
#     hue="phoneme_pair",
#     hue_order=phoneme_pair_order,
#     # hue="subject",
#     kind="kde",
#     height=2,
#     aspect=2.5,
# )
# g.set_axis_labels("Time from\npoint of disambiguation (s)", "Density")

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
    phon_peaks_df.join(
        phon_roc_auc_mean_df,
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="left",
    )
    .with_columns(
        (pl.col("smin_right") - pl.col("smin")).alias("smin_relative"),
        (pl.col("smax_right") - pl.col("smax")).alias("smax_relative"),
        (pl.col("phon_roc_auc_right") / pl.col("phon_roc_auc")).alias(
            "phon_roc_auc_relative"
        ),
    )
    .with_columns((pl.col("smax_relative") / epoch_sfreq).alias("tmax_relative"))
    .filter(
        pl.col("smin_relative") >= -relative_performance_swidth,
        pl.col("smax_relative") <= relative_performance_swidth,
        pl.col("phon_roc_auc_right") > 0.5,
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
phon_roc_auc_comparison_df = pl.concat(
    [
        (
            plot_phon_phon_keys.join(
                phon_roc_auc_searchlight_df,
                on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
                how="left",
            ).with_columns(pl.lit("phon").alias("source"))
        ),
        (
            plot_behav_phon_keys.join(
                phon_roc_auc_searchlight_df,
                on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
                how="left",
            ).with_columns(pl.lit("behav").alias("source"))
        ),
    ],
    how="align",
).with_columns(
    pl.concat_str(
        [
            pl.col("subject"),
            pl.col("electrode_idx").cast(pl.Utf8),
            pl.col("phoneme_pair"),
        ],
        separator="_",
    ).alias("site")
)

# %%
phon_roc_auc_pivot_df = (
    phon_roc_auc_comparison_df.pivot(
        on="source",
        index=["subject", "electrode_idx", "phoneme_pair", "fold"],
        values="phon_roc_auc",
        aggregate_function="mean",
    )
    .drop_nulls()
    .group_by(["subject", "electrode_idx", "phoneme_pair"])
    .agg(
        pl.col("phon").mean().alias("roc_auc_from_phon"),
        pl.col("behav").mean().alias("roc_auc_from_behav"),
    )
)

subject_means = (
    phon_roc_auc_pivot_df.group_by("subject")
    .agg(
        pl.col("roc_auc_from_phon").mean().alias("phon_mean"),
        pl.col("roc_auc_from_behav").mean().alias("behav_mean"),
    )
    .to_pandas()
)
from scipy import stats

ttest_t, ttest_p = stats.ttest_rel(
    subject_means["phon_mean"], subject_means["behav_mean"]
)
print(
    f"Paired t-test comparing phonetic decoding from phonetic vs. perceptual peaks: "
    f"t={ttest_t:.3f}, p={ttest_p:g}"
)

fig, ax = plt.subplots(figsize=(2.75, 2.75))

# Draw individual subject lines (3 points: early, baseline, late)
for _, row in subject_means.iterrows():
    xs = [0, 1]
    ys = [row["phon_mean"], row["behav_mean"]]
    ax.plot(xs, ys, color="gray", alpha=0.4, linewidth=1, zorder=1)
    ax.scatter(xs, ys, color="gray", alpha=0.4, s=20, zorder=2)

# Draw grand mean
grand_early = subject_means["phon_mean"].mean()
grand_late = subject_means["behav_mean"].mean()
ax.plot(
    [0, 1],
    [grand_early, grand_late],
    color="black",
    linewidth=2.5,
    zorder=3,
    alpha=0.7,
)
ax.scatter([0, 1], [grand_early, grand_late], color="black", s=60, zorder=4, alpha=0.7)

# Annotate with significance stars
ymax = max(subject_means["phon_mean"].max(), subject_means["behav_mean"].max())
ax.text(
    0.5,
    0.85,
    p_to_stars(ttest_p),
    ha="center",
    va="bottom",
    fontsize=11,
    transform=ax.transAxes,
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Acoustic\nwindow", "Perceptual\nwindow"])
ax.set_xlabel("Evaluation")
ax.set_ylabel("Acoustic prediction\n(ROC-AUC)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
ax.axhline(0.5, color="k", linestyle="--", alpha=0.3)
ax.set_xlim(-0.3, 1.3)
sns.despine()
fig.tight_layout()
plt.show()

fig.savefig("figures/decoding_phonetic.pdf")

# %%
g = sns.catplot(
    data=(
        phon_roc_auc_comparison_df.group_by(
            [
                "source",
                "site",
                "subject",
                "electrode_idx",
                "phoneme_pair",
                "smin",
                "smax",
            ]
        ).agg(pl.mean("phon_roc_auc"))
    ).to_pandas(),
    x="phoneme_pair",
    order=phoneme_pair_order,
    y="phon_roc_auc",
    hue="source",
    hue_order=source_order,
    kind="box",
    units="site",
)
g.set_axis_labels("Phoneme pair", "Phonetic ROC AUC (%)")
for ax in g.axes.flat:
    ax.axhline(0.5, color="red", linestyle="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

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
        aggregate_function="mean",
    )
    .filter(pl.col("phon").is_not_null() & pl.col("behav").is_not_null())
    .with_columns((pl.col("phon") - pl.col("behav")).alias("roc_auc_diff"))
    .to_pandas(),
    x="roc_auc_diff",
    height=2.5,
    aspect=2.5,
)
g.set_axis_labels(
    "Improvement in phonetic prediction\nat early window vs late window\n($\\Delta$ROC-AUC)"
)

for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0%}".format(x)))

# %% [markdown]
# ### Behav prediction

# %%
behav_roc_auc_comparison_phon = plot_phon_behav_keys.join(
    behav_roc_auc_searchlight_df,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="left",
).with_columns(pl.lit("phon").alias("source"))

# %%
# NB we are using the unfiltered behav peaks here
# we don't want to double-dip by just looking at improvement in the electrodes
# that were already selected because they show improvement
# we want a description of the trend within all phonetic electrodes
behav_roc_auc_comparison_behav = plot_behav_behav_keys_unfiltered.join(
    behav_roc_auc_searchlight_df,
    on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
    how="left",
).with_columns(pl.lit("behav").alias("source"))

# %%
behav_roc_auc_comparison_df = pl.concat(
    [
        behav_roc_auc_comparison_phon,
        behav_roc_auc_comparison_behav,
        behav_baseline_df.with_columns(
            pl.lit("baseline").alias("source"),
            pl.col("behav_roc_auc_baseline").alias("behav_roc_auc"),
        ),
    ],
    how="align",
).with_columns(
    pl.concat_str(
        [
            pl.col("subject"),
            pl.col("electrode_idx").cast(pl.Utf8),
            pl.col("phoneme_pair"),
        ],
        separator="_",
    ).alias("site")
)

# %%
brac_ttest_df = (
    (
        behav_roc_auc_comparison_df.group_by(
            [
                "source",
                "site",
                "subject",
                "electrode_idx",
                "phoneme_pair",
                "smin",
                "smax",
            ]
        ).agg(pl.mean("behav_roc_auc"))
    )
    .to_pandas()
    .set_index("source")
    .dropna(subset=["behav_roc_auc"])
)

# %%

ttest_ind(
    brac_ttest_df.loc["phon"].behav_roc_auc, brac_ttest_df.loc["baseline"].behav_roc_auc
)

# %%
ttest_ind(
    brac_ttest_df.loc["behav"].behav_roc_auc,
    brac_ttest_df.loc["baseline"].behav_roc_auc,
)

# %%
evaluation_order = ["Acoustic\nwindow", "Baseline", "Perceptual\nwindow"]
g = sns.catplot(
    data=(
        behav_roc_auc_comparison_df.group_by(
            [
                "source",
                "site",
                "subject",
                "electrode_idx",
                "phoneme_pair",
                "smin",
                "smax",
            ]
        ).agg(pl.mean("behav_roc_auc"))
    )
    .to_pandas()
    .assign(
        source=lambda xs: xs.source.replace(
            {
                "phon": "Acoustic\nwindow",
                "behav": "Perceptual\nwindow",
                "baseline": "Baseline",
            }
        )
    )
    .rename(columns={"source": "Evaluation"})
    .query("phoneme_pair == 'dn'"),
    # x="phoneme_pair", order=phoneme_pair_order,
    x="Evaluation",
    order=evaluation_order,
    y="behav_roc_auc",
    hue="Evaluation",
    hue_order=evaluation_order,
    showfliers=False,
    kind="box",
    height=3,
    aspect=2.75 / 3,
    palette=categorical_palette,
)
g.set_axis_labels(None, "Perceptual prediction\n(ROC-AUC)")
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
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
    behav_roc_auc_comparison_df.pivot(
        on="source",
        index=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
        values="behav_roc_auc",
        aggregate_function="mean",
    )
    .filter(
        pl.col("phon").is_not_null()
        & pl.col("behav").is_not_null()
        & pl.col("baseline").is_not_null()
    )
    .with_columns(
        (pl.col("behav") - pl.col("baseline")).alias("behav_baseline_diff"),
        (pl.col("phon") - pl.col("baseline")).alias("phon_baseline_diff"),
        (pl.col("behav") - pl.col("phon")).alias("behav_phon_diff"),
    )
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.mean("behav_baseline_diff").alias("behav_baseline_diff"),
        pl.mean("phon_baseline_diff").alias("phon_baseline_diff"),
        pl.mean("behav_phon_diff").alias("behav_phon_diff"),
    )
)

# %%
# Spaghetti plot: per-subject mean improvement from baseline
# Left = early/phon, Center = baseline (0), Right = late/behav
from scipy import stats

subject_means = (
    behav_improvement_df.group_by("subject")
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


# fig, ax = plt.subplots(figsize=(3.5, 3))
fig, ax = plt.subplots(figsize=(2.75, 2.75))

# Draw individual subject lines (3 points: early, baseline, late)
for _, row in subject_means.iterrows():
    xs = [0, 1, 2]
    ys = [row["phon_baseline_diff"], 0, row["behav_baseline_diff"]]
    ax.plot(xs, ys, color="gray", alpha=0.4, linewidth=1, zorder=1)

    # only plot dots on left and right
    ax.scatter([0, 2], [ys[0], ys[2]], color="gray", alpha=0.4, s=20, zorder=2)

# Draw grand mean
grand_early = subject_means["phon_baseline_diff"].mean()
grand_late = subject_means["behav_baseline_diff"].mean()
ax.plot(
    [0, 1, 2],
    [grand_early, 0, grand_late],
    color="black",
    linewidth=2.5,
    zorder=3,
    alpha=0.7,
)
# annotate the left and right but not the center
for x, y in zip([0, 2], [grand_early, grand_late]):
    ax.scatter(x, y, color="black", s=60, zorder=4, alpha=0.7)

# Annotate with significance stars
ymax = max(
    subject_means["phon_baseline_diff"].max(),
    subject_means["behav_baseline_diff"].max(),
)
star_y = ymax * 1.1
ax.annotate(p_to_stars(p_early), xy=(0, star_y), ha="center", va="bottom", fontsize=11)
ax.annotate(p_to_stars(p_late), xy=(2, star_y), ha="center", va="bottom", fontsize=11)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Acoustic\nwindow", "Baseline", "Perceptual\nwindow"])
ax.set_xlabel("Evaluation")
ax.set_ylabel("Perceptual prediction\n($\Delta$ROC-AUC)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlim(-0.3, 2.3)
sns.despine()
fig.tight_layout()
plt.show()

fig.savefig("figures/decoding_behavioral_improvement.pdf")

# %%
# Spaghetti plot: per-subject mean improvement from baseline
# Left = early/phon, Center = baseline (0), Right = late/behav
from scipy import stats

subject_means = (
    behav_improvement_df.group_by("subject")
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
t_early_vs_late, p_early_vs_late = stats.ttest_rel(
    subject_means["behav_baseline_diff"], subject_means["phon_baseline_diff"]
)
print(f"Late vs early: t={t_early_vs_late:.3f}, p={p_early_vs_late:g}")


# fig, ax = plt.subplots(figsize=(3.5, 3))
fig, ax = plt.subplots(figsize=(2.75, 2.75))

# Draw individual subject lines (3 points: early, baseline, late)
for _, row in subject_means.iterrows():
    xs = [0, 1]
    ys = [row["phon_baseline_diff"], row["behav_baseline_diff"]]
    ax.plot(xs, ys, color="gray", alpha=0.4, linewidth=1, zorder=1)
    ax.scatter(xs, ys, color="gray", alpha=0.4, s=20, zorder=2)

# Draw grand mean
grand_early = subject_means["phon_baseline_diff"].mean()
grand_late = subject_means["behav_baseline_diff"].mean()
ax.plot(
    [0, 1],
    [grand_early, grand_late],
    color="black",
    linewidth=2.5,
    zorder=3,
    alpha=0.7,
)
# annotate the left and right but not the center
for x, y in zip([0, 1], [grand_early, grand_late]):
    ax.scatter(x, y, color="black", s=60, zorder=4, alpha=0.7)

# annotate with significance stars
ymax = max(
    subject_means["phon_baseline_diff"].max(),
    subject_means["behav_baseline_diff"].max(),
)
ax.text(
    0.5,
    0.85,
    p_to_stars(p_early_vs_late),
    ha="center",
    va="bottom",
    fontsize=11,
    transform=ax.transAxes,
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Acoustic\nwindow", "Perceptual\nwindow"])
ax.set_xlabel("Evaluation")
ax.set_ylabel("Perceptual prediction\n($\Delta$ROC-AUC)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
ax.axhline(0, color="k", linestyle="--", alpha=0.3)
ax.set_xlim(-0.3, 1.3)
sns.despine()
fig.tight_layout()
plt.show()

fig.savefig("figures/decoding_behavioral_improvement-no_baseline.pdf")

# %%
g = sns.displot(
    data=behav_improvement_df.to_pandas(),
    x="behav_baseline_diff",
    height=2.5,
    aspect=2.5,
)
g.set_axis_labels(
    "Improvement in perceptual prediction\nat late window ($\\Delta$ROC-AUC)"
)

for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.1%}".format(x)))

# %%
g = sns.displot(
    data=behav_improvement_df.to_pandas(),
    # x="improvement", row="comparison",
    x="behav_phon_diff",
    height=2.5,
    aspect=2.5,
)
g.set_axis_labels(
    "Improvement in perceptual prediction\nat early window vs late window\n($\\Delta$ROC-AUC)"
)

for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0%}".format(x)))

# %%
g = sns.displot(
    data=behav_improvement_df.to_pandas(),
    # x="improvement", row="comparison",
    x="phon_baseline_diff",
    height=2.5,
    aspect=2.5,
)
g.set_axis_labels(
    "Improvement in perceptual prediction\nat early window ($\\Delta$ROC-AUC)"
)

for ax in g.axes.flat:
    ax.axvline(0, color="gray", linestyle="--")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0%}".format(x)))

# %%

# %% [markdown]
# ### Cross-window transfer, no training

# %% [markdown]
# #### Transfer on phonetic target

# %%
early_to_late_transfer_keys = (
    phon_roc_auc_comparison_df.filter(pl.col("source") == "behav")
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"])
    .agg(pl.mean("phon_roc_auc"))
    .sort("phon_roc_auc", descending=True)
    .join(
        phon_peaks_df,
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="inner",
        suffix="_early",
    )
    .filter(pl.col("phon_roc_auc_early") > 0.6)
)
early_to_late_transfer_keys

# %%
early_early_outcomes, early_late_outcomes = [], []
late_late_outcomes, late_early_outcomes = [], []
for key in tqdm(
    early_to_late_transfer_keys.iter_rows(named=True),
    total=early_to_late_transfer_keys.height,
):
    ee_i, el_i, ll_i, le_i = evaluate_phonetic_transfer(
        data=paper_data,
        phonetic_decoder_checkpoints=phonetic_decoder_checkpoints,
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
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
        pl.col("word_end").cast(word_end_enum),
    )
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"])
)
early_late_outcomes_df = (
    pl.from_pandas(pd.concat(early_late_outcomes, ignore_index=True))
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
        pl.col("word_end").cast(word_end_enum),
    )
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"])
)
late_late_outcomes_df = (
    pl.from_pandas(pd.concat(late_late_outcomes, ignore_index=True))
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
        pl.col("word_end").cast(word_end_enum),
    )
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"])
)
late_early_outcomes_df = (
    pl.from_pandas(pd.concat(late_early_outcomes, ignore_index=True))
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
        pl.col("word_end").cast(word_end_enum),
    )
    .join(all_md, on=["subject", "epoch_idx", "phoneme_pair", "word_end"])
)

# %%
group_cols = ["subject", "electrode_idx", "phoneme_pair", "fold"]
roc_auc_kwargs = dict(
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=group_cols,
)
phonetic_transfer_results = (
    pl_roc_auc(
        early_early_outcomes_df, **roc_auc_kwargs, roc_auc_name="early_early_roc_auc"
    )
    .join(
        pl_roc_auc(
            early_late_outcomes_df, **roc_auc_kwargs, roc_auc_name="early_late_roc_auc"
        ),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    .join(
        pl_roc_auc(
            late_early_outcomes_df, **roc_auc_kwargs, roc_auc_name="late_early_roc_auc"
        ),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    .join(
        pl_roc_auc(
            late_late_outcomes_df, **roc_auc_kwargs, roc_auc_name="late_late_roc_auc"
        ),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    # Join information about train early -> test early (should be a sanity check)
    .join(
        phon_roc_auc_searchlight_df.join(
            early_to_late_transfer_keys.select(
                ["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"]
            ),
            left_on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
            right_on=[
                "subject",
                "electrode_idx",
                "phoneme_pair",
                "smin_early",
                "smax_early",
            ],
            how="inner",
        ).rename({"phon_roc_auc": "early_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    # Join information about train late -> test late (should be upper bound?? for the transfer case)
    .join(
        phon_roc_auc_searchlight_df.join(
            early_to_late_transfer_keys.select(
                ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
            ),
            on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
            how="inner",
        ).rename({"phon_roc_auc": "late_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        suffix="_late",
        how="inner",
    )
    .with_columns(
        # % of late ROC AUC achieved by early->late transfer
        (pl.col("early_late_roc_auc") / pl.col("late_roc_auc")).alias(
            "early_to_late_normalized_roc_auc"
        ),
        # % of early ROC AUC achieved by late->early transfer
        (pl.col("late_early_roc_auc") / pl.col("early_roc_auc")).alias(
            "late_to_early_normalized_roc_auc"
        ),
        # difference between early early ROC AUC and late->early transfer
        (pl.col("early_early_roc_auc") - pl.col("late_early_roc_auc")).alias(
            "early_transfer_effect"
        ),
    )
    # TODO why are there dupes?
    .unique(subset=group_cols)
)

# %%
group_cols = ["subject", "electrode_idx", "phoneme_pair", "fold"]
roc_auc_kwargs = dict(
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=group_cols,
)

# Compute ROC-AUC on just the extreme acoustic steps
phonetic_transfer_extreme_results = (
    pl_roc_auc(
        early_early_outcomes_df.filter(pl.col("resampled").is_in([1, 6])),
        **roc_auc_kwargs,
        roc_auc_name="early_early_roc_auc",
    )
    .join(
        pl_roc_auc(
            early_late_outcomes_df.filter(pl.col("resampled").is_in([1, 6])),
            **roc_auc_kwargs,
            roc_auc_name="early_late_roc_auc",
        ),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    .join(
        pl_roc_auc(
            late_early_outcomes_df.filter(pl.col("resampled").is_in([1, 6])),
            **roc_auc_kwargs,
            roc_auc_name="late_early_roc_auc",
        ),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    .join(
        pl_roc_auc(
            late_late_outcomes_df.filter(pl.col("resampled").is_in([1, 6])),
            **roc_auc_kwargs,
            roc_auc_name="late_late_roc_auc",
        ),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    # Join information about train early -> test early (should be a sanity check)
    .join(
        phon_roc_auc_searchlight_df.join(
            early_to_late_transfer_keys.select(
                ["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"]
            ),
            left_on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
            right_on=[
                "subject",
                "electrode_idx",
                "phoneme_pair",
                "smin_early",
                "smax_early",
            ],
            how="inner",
        ).rename({"phon_roc_auc": "early_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    # Join information about train late -> test late (should be upper bound?? for the transfer case)
    .join(
        phon_roc_auc_searchlight_df.join(
            early_to_late_transfer_keys.select(
                ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
            ),
            on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
            how="inner",
        ).rename({"phon_roc_auc": "late_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        suffix="_late",
        how="inner",
    )
    .with_columns(
        # % of late ROC AUC achieved by early->late transfer
        (pl.col("early_late_roc_auc") / pl.col("late_roc_auc")).alias(
            "early_to_late_normalized_roc_auc"
        ),
        # % of early ROC AUC achieved by late->early transfer
        (pl.col("late_early_roc_auc") / pl.col("early_roc_auc")).alias(
            "late_to_early_normalized_roc_auc"
        ),
        # difference between early early ROC AUC and late->early transfer
        (pl.col("early_early_roc_auc") - pl.col("late_early_roc_auc")).alias(
            "early_transfer_effect"
        ),
    )
    # TODO why are there dupes?
    .unique(subset=group_cols)
)

# %%
phonetic_transfer_extreme_results

# %%
group_cols = ["subject", "electrode_idx", "phoneme_pair", "fold", "resampled"]

# compute ROC-AUC on just the extremes
phonetic_transfer_results_by_resampled = (
    early_early_outcomes_df.with_columns(
        ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias(
            "correct"
        )
    )
    .group_by(group_cols)
    .agg(pl.mean("correct").alias("early_early_accuracy"))
    .join(
        early_late_outcomes_df.with_columns(
            ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias(
                "correct"
            )
        )
        .group_by(group_cols)
        .agg(pl.mean("correct").alias("early_late_accuracy")),
        on=group_cols,
        how="inner",
    )
    .join(
        late_early_outcomes_df.with_columns(
            ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias(
                "correct"
            )
        )
        .group_by(group_cols)
        .agg(pl.mean("correct").alias("late_early_accuracy")),
        on=group_cols,
        how="inner",
    )
    .join(
        late_late_outcomes_df.with_columns(
            ((pl.col("decoder_target") == 1) == (pl.col("decoder_proba") >= 0.5)).alias(
                "correct"
            )
        )
        .group_by(group_cols)
        .agg(pl.mean("correct").alias("late_late_accuracy")),
        on=group_cols,
        how="inner",
    )
    # Join information about train early -> test early (should be a sanity check)
    .join(
        phon_roc_auc_searchlight_df.join(
            early_to_late_transfer_keys.select(
                ["subject", "electrode_idx", "phoneme_pair", "smin_early", "smax_early"]
            ),
            left_on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
            right_on=[
                "subject",
                "electrode_idx",
                "phoneme_pair",
                "smin_early",
                "smax_early",
            ],
            how="inner",
        ).rename({"phon_roc_auc": "early_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="inner",
    )
    # Join information about train late -> test late (should be upper bound?? for the transfer case)
    .join(
        phon_roc_auc_searchlight_df.join(
            early_to_late_transfer_keys.select(
                ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
            ),
            on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
            how="inner",
        ).rename({"phon_roc_auc": "late_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        suffix="_late",
        how="inner",
    )
    .with_columns(
        (pl.col("early_early_accuracy") - pl.col("late_early_accuracy")).alias(
            "early_transfer_effect"
        ),
    )
)

# %%
# mean over folds
phonetic_transfer_results_mean = phonetic_transfer_results.group_by(
    ["subject", "electrode_idx", "phoneme_pair"]
).agg(
    pl.mean("early_early_roc_auc"),
    pl.mean("early_late_roc_auc"),
    pl.mean("late_early_roc_auc"),
    pl.mean("late_late_roc_auc"),
    pl.mean("early_to_late_normalized_roc_auc"),
    pl.mean("late_to_early_normalized_roc_auc"),
    pl.mean("early_transfer_effect"),
)

phonetic_transfer_extreme_results_mean = phonetic_transfer_extreme_results.group_by(
    ["subject", "electrode_idx", "phoneme_pair"]
).agg(
    pl.mean("early_early_roc_auc"),
    pl.mean("early_late_roc_auc"),
    pl.mean("late_early_roc_auc"),
    pl.mean("late_late_roc_auc"),
    pl.mean("early_to_late_normalized_roc_auc"),
    pl.mean("late_to_early_normalized_roc_auc"),
    pl.mean("early_transfer_effect"),
)

phonetic_transfer_results_by_resampled_mean = (
    phonetic_transfer_results_by_resampled.group_by(
        ["subject", "electrode_idx", "phoneme_pair", "resampled"]
    ).agg(
        pl.mean("early_early_accuracy"),
        pl.mean("early_late_accuracy"),
        pl.mean("late_early_accuracy"),
        pl.mean("late_late_accuracy"),
        pl.mean("early_transfer_effect"),
    )
)

# %%
phonetic_transfer_extreme_results_mean.to_pandas().to_csv(
    Path(outdir) / "phonetic_transfer_extreme_results_mean.csv", index=False
)

# %%
g = sns.catplot(
    data=(
        phonetic_transfer_results_by_resampled_mean.filter(
            pl.col("resampled").is_in([1, 6])
        )
        .rename(
            {
                "early_early_accuracy": "Early window",
                "late_late_accuracy": "Late window",
                "early_late_accuracy": "Late window\n(transfer from early)",
                "late_early_accuracy": "Early window\n(transfer from late)",
            }
        )
        .unpivot(
            on=[
                "Early window",
                "Late window",
                "Late window\n(transfer from early)",
                "Early window\n(transfer from late)",
            ],
            index=["subject", "electrode_idx", "phoneme_pair"],
            variable_name="Evaluation",
            value_name="roc_auc",
        )
    )
    .to_pandas()
    .query('phoneme_pair == "dn"'),
    # x="phoneme_pair", order=phoneme_pair_order,
    y="roc_auc",
    hue="Evaluation",
    kind="box",
)
g.set_axis_labels("", "Phonetic ROC AUC (%)")

for ax in g.axes.flat:
    ax.axhline(0.5, color="red", linestyle="--")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

# %%
sns.displot(
    data=phonetic_transfer_results_by_resampled_mean.filter(
        pl.col("resampled").is_in([1, 6])
    ).to_pandas(),
    x="early_transfer_effect",
    height=2.5,
    aspect=2.5,
)

# %%
(
    phonetic_transfer_results_by_resampled_mean.filter(
        pl.col("resampled").is_in([1, 6])
    )
    .group_by(["subject", "electrode_idx", "phoneme_pair"])
    .agg(
        pl.mean("early_early_accuracy").alias("early_early_accuracy"),
        pl.mean("late_early_accuracy").alias("late_early_accuracy"),
    )
    .rename(
        {
            "early_early_accuracy": "Early window",
            "late_early_accuracy": "Early window\n(transfer from late)",
        }
    )
).sort("Early window\n(transfer from late)")

# %%
# spaghetti plot
early_col = "Acoustic window"
transfer_col = "Acoustic window\n(transfer from\nperceptual window)"
hue_order = [early_col, transfer_col]

df_wide = (
    phonetic_transfer_results_by_resampled_mean.filter(
        pl.col("resampled").is_in([1, 6])
    )
    .group_by(["subject", "electrode_idx", "phoneme_pair"])
    .agg(
        pl.mean("early_early_accuracy").alias("early_early_accuracy"),
        pl.mean("late_early_accuracy").alias("late_early_accuracy"),
    )
    .rename(
        {
            "early_early_accuracy": "Acoustic window",
            "late_early_accuracy": "Acoustic window\n(transfer from\nperceptual window)",
        }
    )
    .to_pandas()
    .query('phoneme_pair == "dn"')[
        [*["subject", "electrode_idx", "phoneme_pair"], early_col, transfer_col]
    ]
    .dropna(subset=[early_col, transfer_col])
)

fig, ax = plt.subplots(figsize=(2, 2.5))

colors = sns.color_palette(categorical_palette, 2)
x0, x1 = 0, 1

# spaghetti lines
for _, row in df_wide.iterrows():
    ax.plot(
        [x0, x1],
        [row[early_col], row[transfer_col]],
        color="gray",
        alpha=0.2,
        linewidth=0.8,
        zorder=0,
    )

# points at each end
ax.scatter(
    [x0] * len(df_wide),
    df_wide[early_col],
    color=colors[0],
    s=12,
    alpha=0.4,
    zorder=2,
    label=early_col,
)
ax.scatter(
    [x1] * len(df_wide),
    df_wide[transfer_col],
    color=colors[1],
    s=12,
    alpha=0.4,
    zorder=2,
    label=transfer_col,
)

ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
ax.set_xticks([x0, x1])
ax.set_xticklabels(hue_order)
ax.set_ylabel("Acoustic prediction\n(accuracy)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
# ax.legend(loc="lower right", fontsize=8)
sns.despine(ax=ax)

fig.savefig("figures/decoding_acoustic_transfer.pdf")

# %%
# spaghetti plot
early_col = "Acoustic window"
transfer_col = "Acoustic window\n(transfer from\nperceptual window)"
hue_order = [early_col, transfer_col]

df_wide = (
    phonetic_transfer_extreme_results_mean.group_by(
        ["subject", "electrode_idx", "phoneme_pair"]
    )
    .agg(
        pl.mean("early_early_roc_auc").alias("early_early_roc_auc"),
        pl.mean("late_early_roc_auc").alias("late_early_roc_auc"),
    )
    .rename(
        {
            "early_early_roc_auc": "Acoustic window",
            "late_early_roc_auc": "Acoustic window\n(transfer from\nperceptual window)",
        }
    )
    .to_pandas()
    .query('phoneme_pair == "dn"')[
        [*["subject", "electrode_idx", "phoneme_pair"], early_col, transfer_col]
    ]
    .dropna(subset=[early_col, transfer_col])
)

fig, ax = plt.subplots(figsize=(2, 2.5))

colors = sns.color_palette(categorical_palette, 2)
x0, x1 = 0, 1

# spaghetti lines
for _, row in df_wide.iterrows():
    ax.plot(
        [x0, x1],
        [row[early_col], row[transfer_col]],
        color="gray",
        alpha=0.2,
        linewidth=0.8,
        zorder=0,
    )

# points at each end
ax.scatter(
    [x0] * len(df_wide),
    df_wide[early_col],
    color=colors[0],
    s=12,
    alpha=0.4,
    zorder=2,
    label=early_col,
)
ax.scatter(
    [x1] * len(df_wide),
    df_wide[transfer_col],
    color=colors[1],
    s=12,
    alpha=0.4,
    zorder=2,
    label=transfer_col,
)

ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
ax.set_xticks([x0, x1])
ax.set_xticklabels(hue_order)
ax.set_ylabel("Acoustic prediction\n(ROC AUC)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
# ax.legend(loc="lower right", fontsize=8)
sns.despine(ax=ax)

fig.savefig("figures/decoding_acoustic_transfer-roc_auc.pdf")

# %%
phonetic_transfer_extreme_results_mean.select(pl.min("early_transfer_effect"))

# %%
g = sns.displot(
    data=phonetic_transfer_extreme_results_mean.to_pandas().assign(
        early_transfer_effect=lambda df: -df["early_transfer_effect"]
    ),
    x="early_transfer_effect",
    height=2.5,
    aspect=2,
    bins=15,
)

g.set_axis_labels("Early transfer effect\n($\Delta$ROC-AUC)")
g.axes[0, 0].axvline(0, color="gray", linestyle="--")

# %% [markdown]
# #### Behavior

# %%
(
    behav_roc_auc_comparison_phon.group_by(
        ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
    )
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
# zoomin_keys = (
#     behav_peaks_df
#     .select(["subject", "electrode_idx", "phoneme_pair", "word_end", "behav_roc_auc_improvement"])
# )
zoomin_keys

# %%
# pdfpages render
cols_per_page = 3
max_num_pages = np.inf

outf = "hga_zoomin_search.pdf"
hga_zoomin_keys = zoomin_keys.unique(
    ["subject", "electrode_idx", "phoneme_pair", "word_end"]
).sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])

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
    "EC250",
    185,
    "dn",
    "desolate",
    hide_bottom=False,
    legend=False,
    **star_plot_kwargs,
)
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
    paper_data, "EC278", 38, "dn", "necessary", hide_bottom=True, **star_plot_kwargs
)
plt.gcf().savefig("figures/zoomin_EC278_38_dn_necessary.pdf")
None

# %% [markdown]
# ## Quant HGA search

# %%
hga_df = paper_data.hga_df
early_polarity = paper_data.early_polarity
late_polarity = paper_data.late_polarity
reg_df = paper_data.reg_df

# %%
# sns.catplot(
#     data=reg_df,
#     x="phoneme_pair",
#     hue="decoder_target",
#     y="hga_early_signed",
#     kind="box",
#     height=3,
#     aspect=1.5,
# )
# from scipy.stats import ttest_ind

# ttest_ind(
#     reg_df.query("decoder_target == 0")["hga_early_signed"],
#     reg_df.query("decoder_target == 1")["hga_early_signed"],
# )

# %%
# sns.catplot(
#     data=reg_df, x="early_polarity", hue="decoder_target", y="hga_early", kind="box"
# )

# %%
# sns.catplot(data=reg_df, hue="behavior_dummy_forced", y="hga_late_signed", kind="box")
# ttest_ind(
#     reg_df.dropna().query("behavior_dummy_forced == 0")["hga_late_signed"],
#     reg_df.dropna().query("behavior_dummy_forced == 1")["hga_late_signed"],
# )

# %%
# sns.catplot(
#     data=reg_df,
#     x="late_polarity",
#     hue="behavior_dummy_forced",
#     y="hga_late",
#     kind="box",
# )

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
f = plot_condition_contrasts_single_figure(
    paper_data,
    textgrid_dir,
    epoch_data_cache=pcc_epoch_data_cache,
    plot_word_ends=["necessary"],
)
f.savefig("figures/condition_contrasts.pdf")

# %%
plot_condition_contrasts_single_figure(
    paper_data,
    textgrid_dir,
    epoch_data_cache=pcc_epoch_data_cache,
    plot_word_ends=["desolate"],
)
plt.gca().set_xlim(0, 0.7)
plt.gcf().savefig("figures/condition_contrasts-desolate.pdf")


# %%
plot_condition_contrasts_single_figure(
    paper_data,
    textgrid_dir,
    epoch_data_cache=pcc_epoch_data_cache,
    plot_word_ends=["necessary", "desolate"],
    vline_extent=1.0,
    textgrid_kwargs=dict(
        include_offset=False, include_phonemes=False, vline_extent=1.0
    ),
)
plt.gcf().savefig("figures/condition_contrasts-both.pdf")
None

# %%
# f, ax = plt.subplots(figsize=(5, 3))
# plot_condition_contrast_peak_aligned(
#     plot_behav_rows,
#     behav_peaks_df=behav_peaks_df,
#     data=paper_data,
#     condition_variable="behavior_dummy_forced",
#     polarity_correct="late",
#     epoch_data_cache=pcc_epoch_data_cache,
#     ax=ax,
#     label="Behavioral contrast",
#     window_sec=0.3,
#     ttest_window_size=4,
#     ttest_window_stride=4,
#     ttest_bar_height_ratio=0.04,
#     ttest_bar_y_ratio=0.87,
# )
# ax.set_xlim(-0.3, 0.3)


# %% [markdown]
# ## Behav stackplot


# %%
plot_behav_barplot(
    all_md,
    "EC250",
    "dn",
    "desolate",
    [1, 3, 4, 6],
    legend=False,
    resampled_palette=resampled_palette_simplified,
)
plt.gcf().savefig("figures/behav_barplot_EC250_dn_desolate.pdf")

# %%
plot_behav_barplot(
    all_md,
    "EC278",
    "dn",
    "necessary",
    [1, 3, 4, 6],
    legend=False,
    resampled_palette=resampled_palette_simplified,
)
plt.gcf().savefig("figures/behav_barplot_EC278_dn_necessary.pdf")

# %% [markdown]
# ## Exploratory: polarity relationships

# %%
early_polarity_strict = early_polarity.copy()

# %%
late_polarity_strict = late_polarity.dropna()
late_polarity_strict["lexical_evidence"] = (
    late_polarity_strict.word_end.str[0] == late_polarity_strict.phoneme_pair.str[1]
).astype(int)


# %%
def plot_summary_acoustic_vs_presence_of_response(phoneme_pair: str):
    # First: summarize relationship between early polarity
    # and on which words the subsequent effect appears
    f, ax = plt.subplots(figsize=(1.1, 1.5))

    word_end_1, word_end_2 = PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]
    completion_1 = "-" + word_end_1[1:]
    completion_2 = "-" + word_end_2[1:]

    sns.heatmap(
        late_polarity_strict.query("phoneme_pair == @phoneme_pair")
        .pivot_table(
            index=["subject", "electrode_idx", "phoneme_pair"],
            columns="lexical_evidence",
            values="late_polarity",
            aggfunc="count",
        )
        .fillna(0)
        .astype({0: bool, 1: bool})
        .merge(
            (
                early_polarity_strict.query("phoneme_pair == @phoneme_pair")
                .groupby(["subject", "electrode_idx", "phoneme_pair"])
                .filter(lambda xs: xs.early_polarity.nunique() == 1)
                .drop(columns=["word_end"])
                .drop_duplicates()
                .set_index(["subject", "electrode_idx", "phoneme_pair"])
            ),
            left_index=True,
            right_index=True,
            how="inner",
        )
        .groupby("early_polarity")
        .value_counts(sort=False)
        .reset_index()
        .pipe(
            lambda df: df.assign(
                early_polarity=df.early_polarity.map(
                    {-1: phoneme_pair[0], 1: phoneme_pair[1]}
                )
            )
        )
        .rename(columns={0: completion_1, 1: completion_2})
        .set_index([completion_1, completion_2, "early_polarity"])["count"]
        .unstack()
        .fillna(0)
        .sort_index(ascending=False),
        annot=True,
        ax=ax,
        cbar=False,
    )

    ax.set_xlabel("Acoustic\nselectivity")
    ax.set_ylabel(None)
    yticklabel_map = {
        "False-True": f"{completion_2}\nresponse",
        "True-False": f"{completion_1}\nresponse",
        "True-True": "Both",
        "False-False": "Neither",
    }
    ax.set_yticklabels(
        [yticklabel_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()],
        rotation=0,
    )

    return f


# %%
f = plot_summary_acoustic_vs_presence_of_response("dn")
f.savefig("figures/early_polarity-late_response.pdf")

# %%
f = plot_summary_acoustic_vs_presence_of_response("bm")
f.savefig("figures/early_polarity-late_response-bm.pdf")

# %%
f = plot_summary_acoustic_vs_presence_of_response("pb")
f.savefig("figures/early_polarity-late_response-pb.pdf")


# %%
def plot_summary_acoustic_vs_presence_of_response_stackplot(
    phoneme_pair=None, ax=None, palette="Set2"
):
    if phoneme_pair is None:
        # use these labels w.l.o.g.
        phoneme_pair_label = "dn"
        completion_1, completion_2 = "-esolate", "-ecessary"
        early_polarity_strict__ = early_polarity_strict
        late_polarity_strict__ = late_polarity_strict
    else:
        phoneme_pair_label = phoneme_pair
        completion_1, completion_2 = [
            f"-{w[1:]}" for w in PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]
        ]
        early_polarity_strict__ = early_polarity_strict.query(
            "phoneme_pair == @phoneme_pair"
        )
        late_polarity_strict__ = late_polarity_strict.query(
            "phoneme_pair == @phoneme_pair"
        )

    df = (
        late_polarity_strict__.pivot_table(
            index=["subject", "electrode_idx", "phoneme_pair"],
            columns="lexical_evidence",
            values="late_polarity",
            aggfunc="count",
        )
        .fillna(0)
        .astype({0: bool, 1: bool})
        .merge(
            (
                early_polarity_strict__.groupby(
                    ["subject", "electrode_idx", "phoneme_pair"]
                )
                .filter(lambda xs: xs.early_polarity.nunique() == 1)
                .drop(columns=["word_end"])
                .drop_duplicates()
                .set_index(["subject", "electrode_idx", "phoneme_pair"])
            ),
            left_index=True,
            right_index=True,
            how="inner",
        )
        .groupby("early_polarity")
        .value_counts(sort=False)
        .reset_index()
        .pipe(
            lambda df: df.assign(
                early_polarity=df.early_polarity.map(
                    {-1: phoneme_pair_label[0], 1: phoneme_pair_label[1]}
                )
            )
        )
        .rename(columns={0: completion_1, 1: completion_2})
        .set_index([completion_1, completion_2, "early_polarity"])["count"]
        .unstack()
        .fillna(0)
        .sort_index(ascending=False)
        .T
    )

    # map to readable columns
    column_order = [completion_1, completion_2, "Both"]
    df.columns = df.columns.map(
        dict(zip([(True, False), (False, True), (True, True)], column_order))
    )
    df = df[column_order]

    # # Chi-square test on one-sided responses only
    # # Create 2x2 contingency table: acoustic (rows) x perceptual completion (columns)
    # chi2_table = df[[completion_1, completion_2]].values
    # from scipy.stats import chi2_contingency
    # chi2, p_value, dof, expected = chi2_contingency(chi2_table)
    # print(chi2, p_value)
    # sig_stars = p_to_stars(p_value)

    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    if ax is None:
        f, ax = plt.subplots(figsize=(2.25, 2))
    else:
        f = ax.get_figure()

    df_pct.plot(
        kind="bar",
        stacked=True,
        color=sns.color_palette(palette, n_colors=3),
        ax=ax,
        width=0.7,
    )
    sns.despine(ax=ax)
    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(1.75, 1), title="Perceptual\nresponse"
    )
    plt.setp(legend.get_title(), multialignment="center")
    ax.set_xlabel("Acoustic preference")
    ax.set_ylabel(None)
    ax.set_xticks(range(len(df_pct.index)))
    ax.set_xticklabels(df_pct.index, rotation=0)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # # Add significance annotation
    # ax.text(0.5, 0.9, sig_stars, ha='center', va='bottom',
    #         fontsize=14, transform=ax.transAxes)

    return f


# %%
f = plot_summary_acoustic_vs_presence_of_response_stackplot()
f.savefig("figures/early_polarity-late_response_stackplot.pdf")

# %%
plot_summary_acoustic_vs_presence_of_response_stackplot("dn")
None

# %%
# Matches in late polarity for electrodes showing responses at both completion
late_polarity_strict.set_index(
    ["subject", "electrode_idx", "phoneme_pair", "lexical_evidence"]
).late_polarity.unstack().dropna().groupby("phoneme_pair").value_counts()

# %%

# %%
# Next: summarize relationship between early polarity and late polarity
# NB in this setup, we are separately counting responses to -ecessary and -esolate
# at each electrode, since the polarities of either the early or late response
# might differ

f, ax = plt.subplots(figsize=(1, 1))
sns.heatmap(
    (
        late_polarity_strict.drop_duplicates(
            ["subject", "electrode_idx", "phoneme_pair", "word_end"]
        )
        .drop(columns=["lexical_evidence"])
        .merge(
            early_polarity_strict.drop_duplicates(),
            on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        )[["early_polarity", "late_polarity"]]
        .value_counts(sort=False)
        .reset_index()
        .pipe(
            lambda df: df.assign(
                early_selectivity=df.early_polarity.map({-1: "d", 1: "n"}),
                late_selectivity=df.late_polarity.map({-1: "d", 1: "n"}),
            )
        )
        .set_index(["early_selectivity", "late_selectivity"])["count"]
        .unstack()
    ),
    annot=True,
    vmin=0,
    cbar=False,
)

ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_ylabel("Acoustic\npreference", rotation=0, va="center", labelpad=34)
ax.set_xlabel("Perceptual\npreference")

f.savefig("figures/early_polarity-late_polarity.pdf")

# %%
preference_relationship_df = (
    late_polarity_strict.drop_duplicates(
        ["subject", "electrode_idx", "phoneme_pair", "word_end"]
    )
    .drop(columns=["lexical_evidence"])
    .merge(
        early_polarity_strict.drop_duplicates(),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    )[["early_polarity", "late_polarity"]]
    .value_counts(sort=False)
    .reset_index()
    .pipe(
        lambda df: df.assign(
            early_selectivity=df.early_polarity.map({-1: "d", 1: "n"}),
            late_selectivity=df.late_polarity.map({-1: "d", 1: "n"}),
        )
    )
    .set_index(["early_selectivity", "late_selectivity"])["count"]
    .unstack()
)
preference_relationship_df

# %%
(
    preference_relationship_df.sum(axis=0).loc["d"],
    preference_relationship_df.sum().sum(),
    preference_relationship_df.sum(axis=0) / preference_relationship_df.sum().sum(),
)

# %%
from scipy.stats import chi2_contingency

chi2_contingency(preference_relationship_df)

# %%
preference_relationship_pct_df = (
    preference_relationship_df.div(preference_relationship_df.sum(axis=1), axis=0) * 100
)
preference_relationship_pct_df

# %%
# Stacked bar chart form
f, ax = plt.subplots(figsize=(2.25, 2))
(
    preference_relationship_pct_df.plot(
        kind="bar",
        stacked=True,
        width=0.7,
        color=sns.color_palette("Set2", n_colors=2),
        ax=ax,
    )
)

# Add 50% chance line
plt.axhline(y=50, color="black", linestyle="--", linewidth=1, alpha=0.5)

plt.xlabel("Acoustic preference", fontsize=12)
plt.ylabel(None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(title="Perceptual\npreference", loc="upper right", bbox_to_anchor=(1.6, 1))
sns.despine(ax=ax)

f.savefig("figures/early_polarity-late_polarity_stackbar.pdf")

# %%
# Relationship between polarity of -ecessary and -esolate responses at the same electrode
f, ax = plt.subplots(figsize=(2.5, 1.5))
sns.heatmap(
    (
        late_polarity_strict.set_index(
            ["subject", "electrode_idx", "phoneme_pair", "lexical_evidence"]
        )
        .late_polarity.unstack("lexical_evidence")
        .fillna(0)
        .astype(int)
        .value_counts(sort=False)
        .unstack(1)
        # update labels
        .rename(
            index={-1: "/d/-selective\nin -esolate", 1: "/n/-selective\nin -esolate"}
        )
        .rename(
            columns={
                -1: "/d/-selective\nin -ecessary",
                1: "/n/-selective\nin -ecessary",
            }
        )
    ),
    annot=True,
    cbar=False,
)

ax.set_xlabel(None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_ylabel(None)

f.savefig("figures/late_polarity_relationship.pdf")

# %% [markdown]
# ## Late response on unambiguous vs. ambiguous trials
#
# For each site with a valid late window (hga_late not NaN), test whether the
# late-window signed HGA (hga_late_signed) also differentiates behavioral choice
# on *unambiguous* trials (resampled 1 and 6), vs. only on the ambiguous trials
# that were used to find the window.
#
# At resampled=1, behavioral choice is almost always 0; at resampled=6 almost
# always 1. So we compare hga_late_signed between behavior_dummy_forced=0 and =1
# within unambiguous trials per site.  We use acoustic-consistent trials only
# (follows_acoustics=True) to keep the same inclusion criteria used in zoomin_hga.

# %%

site_cols = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

# Only sites with a valid late window
reg_df_valid = reg_df.dropna(subset=["hga_late_signed"])

unambig_late_results = []
for site_key, site_data in reg_df_valid.groupby(site_cols):
    # unambiguous acoustic-consistent trials only
    unambig = site_data[
        site_data.resampled.isin([1.0, 6.0]) & site_data.follows_acoustics
    ]
    grp0 = unambig[unambig.behavior_dummy_forced == 0]["hga_late_signed"].dropna()
    grp1 = unambig[unambig.behavior_dummy_forced == 1]["hga_late_signed"].dropna()

    if len(grp0) >= 3 and len(grp1) >= 3:
        t, p = ttest_ind(grp1, grp0, equal_var=False)
        has_unambig_response = p < 0.05 and t > 0  # expect higher for behavior=1
    else:
        p, has_unambig_response = np.nan, False

    unambig_late_results.append(
        {
            **dict(zip(site_cols, site_key)),
            "n_unambig": len(unambig),
            "n_grp0": len(grp0),
            "n_grp1": len(grp1),
            "t": t if not np.isnan(p) else np.nan,
            "p_unambig": p,
            "late_on_unambig": has_unambig_response,
        }
    )

unambig_late_df = pd.DataFrame(unambig_late_results)

n_total = len(unambig_late_df)
n_both = unambig_late_df["late_on_unambig"].sum()
n_ambig_only = n_total - n_both

print(
    f"Sites with valid late window: {n_total}\n"
    f"  Late response on BOTH unambiguous and ambiguous trials: {n_both} "
    f"({100 * n_both / n_total:.0f}%)\n"
    f"  Late response on ambiguous trials ONLY:                 {n_ambig_only} "
    f"({100 * n_ambig_only / n_total:.0f}%)"
)
unambig_late_df.to_csv(Path(outdir) / "unambig_late_df.csv", index=False)
unambig_late_df

# %%
(
    behav_peaks_df[["subject", "electrode_idx", "phoneme_pair", "word_end"]]
    .to_pandas()
    .merge(
        unambig_late_df,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="left",
    )
    .late_on_unambig.value_counts()
)

# %%
(
    behav_peaks_df[["subject", "electrode_idx", "phoneme_pair", "word_end"]]
    .to_pandas()
    .merge(
        unambig_late_df,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="left",
    )
    .late_on_unambig.value_counts(normalize=True)
)

# %% [markdown]
# ## Perceptual contrast on unambiguous trials, split by late-response generalization
#
# Compare sites that show a late behavioral response on *unambiguous* trials
# (`late_on_unambig=True`) vs. those that only show it on ambiguous trials.
# Both groups are evaluated here on unambiguous trials (resampled 1 and 6),
# using `behavior_dummy_forced` as the condition variable and `late_polarity`
# sign correction — i.e. the same signed HGA contrast used to classify sites.

# %%
from matplotlib.patches import Rectangle

from src.viz_paper import HandlerRectangle, plot_condition_contrast

unambig_late_pl = (
    pl.from_pandas(unambig_late_df[site_cols + ["late_on_unambig"]])
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
        pl.col("word_end").cast(word_end_enum),
    )
    .join(
        behav_peaks_df,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="inner",
    )
)

# %%
# phonetic response on electrodes that don't show a behavioral response
ax, _, _ = plot_condition_contrast(
    (
        paper_data.plot_phon_phon_df.join(
            paper_data.plot_behav_behav_df,
            on=["subject", "electrode_idx", "phoneme_pair"],
            how="inner",
        ).filter(pl.col("phoneme_pair") == "dn")
    ),
    "categorical_acoustic_cue",
    data=paper_data,
    textgrid_dir=textgrid_dir,
    polarity_correct="early",
    epoch_data_cache=pcc_epoch_data_cache,
    pval_thresholds=(0.0000001, 0.000001, 0.00001),
)
ax.set_xlim(0, 2.0)

# %%
for plot_word_end, plot_xlim in zip(["necessary", "desolate"], [(0, 1.2), (0, 0.7)]):
    plot_unambig = paper_data.plot_phon_phon_df.filter(
        pl.col("resampled").is_in([1.0, 6.0]),
        pl.col("word_end").is_in([plot_word_end]),
    )
    plot_generalize = plot_unambig.join(
        unambig_late_pl.filter(pl.col("late_on_unambig")), on=site_cols, how="inner"
    )
    plot_specific = plot_unambig.join(
        unambig_late_pl.filter(~pl.col("late_on_unambig")), on=site_cols, how="inner"
    )

    n_generalize = plot_generalize.select(site_cols).unique().height
    n_specific = plot_specific.select(site_cols).unique().height

    f, ax = plt.subplots(figsize=(3, 2))
    plot_palette = sns.color_palette("Set1", 2)

    _, p_handles, p_labels = plot_condition_contrast(
        plot_generalize,
        "behavior_dummy_forced",
        data=paper_data,
        textgrid_dir=textgrid_dir,
        polarity_correct="late",
        epoch_data_cache=pcc_epoch_data_cache,
        ax=ax,
        color=plot_palette[0],
        annotate=True,
        textgrid_kwargs={"fontsize": 8},
        label=f"Generalizes (n={n_generalize})",
        pval_thresholds=(0.01, 0.05),
    )
    plot_condition_contrast(
        plot_specific,
        "behavior_dummy_forced",
        data=paper_data,
        textgrid_dir=textgrid_dir,
        polarity_correct="late",
        epoch_data_cache=pcc_epoch_data_cache,
        ax=ax,
        color=plot_palette[1],
        annotate=False,
        label=f"Ambig-only (n={n_specific})",
        ttest_bar_y_ratio=0.87,
        pval_thresholds=(0.01, 0.05),
    )

    ax.set_xlim(*plot_xlim)
    ax.set_ylabel("HGA effect size ($z$)")
    ax.set_xlabel("Time from word onset (s)")

    if p_handles is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles += p_handles
        labels += p_labels
        ax.legend(
            handles=handles,
            labels=labels,
            handler_map={Rectangle: HandlerRectangle()},
            fontsize=8,
            loc="best",
        )

    f.savefig(f"figures/perceptual_contrast_unambig_split-{plot_word_end}.pdf")
    plt.show()

# %%
# Behavioral contrast on electrodes with vs without late unambiguous response,
# plotted on the UNambiguous trials

plot_word_ends = ["necessary", "desolate"]
plot_xlim = (0, 1.2)

plot_unambig = paper_data.plot_phon_phon_df.filter(
    pl.col("resampled").is_in([1.0, 6.0]),
    pl.col("word_end").is_in(plot_word_ends),
    pl.col("follows_acoustics") == True,
)
plot_generalize = plot_unambig.join(
    unambig_late_pl.filter(pl.col("late_on_unambig")), on=site_cols, how="inner"
)
plot_specific = plot_unambig.join(
    unambig_late_pl.filter(~pl.col("late_on_unambig")), on=site_cols, how="inner"
)

n_generalize = plot_generalize.select(site_cols).unique().height
n_specific = plot_specific.select(site_cols).unique().height

f, ax = plt.subplots(figsize=(3, 2))
plot_palette = sns.color_palette("Set1", 2)

_, p_handles, p_labels = plot_condition_contrast(
    plot_generalize,
    "behavior_dummy_forced",
    data=paper_data,
    textgrid_dir=textgrid_dir,
    polarity_correct="late",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=ax,
    color=plot_palette[0],
    annotate=True,
    textgrid_kwargs=dict(include_phonemes=False, include_offset=False),
    label=f"Generalizes (n={n_generalize})",
    pval_thresholds=(0.001, 0.01, 0.05),
)
plot_condition_contrast(
    plot_specific,
    "behavior_dummy_forced",
    data=paper_data,
    textgrid_dir=textgrid_dir,
    polarity_correct="late",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=ax,
    color=plot_palette[1],
    annotate=False,
    label=f"Ambig-only (n={n_specific})",
    ttest_bar_y_ratio=0.87,
    pval_thresholds=(0.001, 0.01, 0.05),
)

ax.set_xlim(*plot_xlim)
ax.set_ylabel("HGA effect size ($z$)")
ax.set_xlabel("Time from word onset (s)")

if p_handles is not None:
    handles, labels = ax.get_legend_handles_labels()
    handles += p_handles
    labels += p_labels
    ax.legend(
        handles=handles,
        labels=labels,
        handler_map={Rectangle: HandlerRectangle()},
        fontsize=8,
        loc="best",
    )

f.savefig("figures/perceptual_contrast_unambig_split-both.pdf")
plt.show()

# %%
# Behavioral contrast on electrodes with vs without late unambiguous response,
# plotted on the ambiguous trials

plot_word_ends = ["necessary", "desolate"]
plot_xlim = (0, 1.2)

plot_unambig = paper_data.plot_phon_phon_df.filter(
    ~pl.col("resampled").is_in([1.0, 6.0]),
    pl.col("word_end").is_in(plot_word_ends),
)
plot_generalize = plot_unambig.join(
    unambig_late_pl.filter(pl.col("late_on_unambig")), on=site_cols, how="inner"
)
plot_specific = plot_unambig.join(
    unambig_late_pl.filter(~pl.col("late_on_unambig")), on=site_cols, how="inner"
)

n_generalize = plot_generalize.select(site_cols).unique().height
n_specific = plot_specific.select(site_cols).unique().height

f, ax = plt.subplots(figsize=(3, 2))
plot_palette = sns.color_palette("Set1", 2)

_, p_handles, p_labels = plot_condition_contrast(
    plot_generalize,
    "behavior_dummy_forced",
    data=paper_data,
    textgrid_dir=textgrid_dir,
    polarity_correct="late",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=ax,
    color=plot_palette[0],
    annotate=True,
    textgrid_kwargs=dict(include_phonemes=False, include_offset=False),
    label=f"Generalizes (n={n_generalize})",
    pval_thresholds=(0.001, 0.01, 0.05),
)
plot_condition_contrast(
    plot_specific,
    "behavior_dummy_forced",
    data=paper_data,
    textgrid_dir=textgrid_dir,
    polarity_correct="late",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=ax,
    color=plot_palette[1],
    annotate=False,
    label=f"Ambig-only (n={n_specific})",
    ttest_bar_y_ratio=0.87,
    pval_thresholds=(0.001, 0.01, 0.05),
)

ax.set_xlim(*plot_xlim)
ax.set_ylabel("HGA effect size ($z$)")
ax.set_xlabel("Time from word onset (s)")

if p_handles is not None:
    handles, labels = ax.get_legend_handles_labels()
    handles += p_handles
    labels += p_labels
    ax.legend(
        handles=handles,
        labels=labels,
        handler_map={Rectangle: HandlerRectangle()},
        fontsize=8,
        loc="best",
    )

# f.savefig(f"figures/perceptual_contrast_unambig_split-both.pdf")
plt.show()

# %%
