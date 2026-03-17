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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy import stats
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
    PHONEME_PAIR_TO_WORD_ENDS,
    WORD_END_TO_PHONEME_PAIR,
)
from src.figure_builder import FigureBuilder
from src.viz_paper import (
    HandlerRectangle,
    PaperData,
    add_textgrid,
    evaluate_behav_decoder_on_phon_window,
    evaluate_phon_decoder_on_behav_window,
    evaluate_phonetic_transfer,
    p_to_stars,
    phoneme_pair_enum,
    pl_roc_auc,
    plot_behav_barplot,
    plot_condition_contrast,
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

neurometrics_dir = "outputs/causal4/prepare_neurometrics/p65_b5_a3"

epoch_tmin = -0.4
epoch_sfreq = 100

ambiguous_response_threshold = 2

textgrid_dir = "textgrids"

outdir = "outputs/causal4/A_neurometrics"
outdir_talk = Path(outdir) / "talk"
outdir_talk.mkdir(parents=True, exist_ok=True)

# %%
max_plot_rows = 15
phoneme_pair_order = ["bm", "dn", "pb"]
source_order = ["phon", "behav"]

# shared palette for categorical variables
categorical_palette = "Set2"

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

# %%
# TODO why do we have nulls here?
plot_phon_behav_df = plot_phon_behav_df.filter(pl.col("decoder_target").is_not_null())

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

plot_phon_phon_keys = phon_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
)
plot_behav_phon_keys = behav_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "word_end"]
)
plot_phon_behav_keys = phon_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
)
plot_behav_behav_keys = behav_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
)
plot_behav_behav_keys_unfiltered = behav_peaks_df_unfiltered.select(
    ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
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
summary_distribution = electrode_distribution_df[["phonetic_selective", "behavior_selective"]].sum()

non_behavior_of_phonetic = (summary_distribution["phonetic_selective"] - summary_distribution["behavior_selective"]).sum() / summary_distribution["phonetic_selective"].sum()
behavior_of_phonetic = summary_distribution["behavior_selective"].sum() / summary_distribution["phonetic_selective"].sum()

# stackplot
f, ax = plt.subplots(figsize=(2, 0.3))
plot_palette = sns.color_palette(categorical_palette, 2)
ax.barh(0, behavior_of_phonetic, left=0,
        color=plot_palette[1], label="Behavior-selective")
ax.barh(0, non_behavior_of_phonetic, left=behavior_of_phonetic,
        color=plot_palette[0], label="Phonetic-selective\nonly")
sns.despine(ax=ax, left=True, bottom=True)
ax.set_yticks([])
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

f.savefig(Path(outdir) / "phonetic_behavior_selective_stackbar.pdf", transparent=True)

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

fig.savefig(Path(outdir) / "electrode_distribution-phonetic_selective.pdf")

# %% [markdown]
# ## Prepare prediction summaries

# %% [markdown]
# ### Phonetic

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
# ### Behavior

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
group_cols = [
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
]
behav_roc_auc = (
    pl_roc_auc(
        df=plot_behav_df,
        target_col="decoder_target",
        proba_col="full_decoder_proba",
        group_cols=group_cols,
        roc_auc_name="behav_roc_auc",
    )
    .join(
        pl_roc_auc(
            df=plot_behav_df,
            target_col="decoder_target",
            proba_col="baseline_decoder_proba",
            group_cols=group_cols,
            roc_auc_name="behav_roc_auc_baseline",
        ),
        on=group_cols,
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

# %% [markdown]
# ## Dynamics

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
def plot_peak_timing(
    plot_phoneme_pair, plot_word_ends, plot_xlim=None, vline_extent=1.1
):
    plot_textgrid = "11_necessary_dn_002.TextGrid"
    plot_data = (
        peak_timing_plot.filter(
            (pl.col("word_end").is_in(plot_word_ends))
            | (
                (pl.col("source") == "phon")
                & (pl.col("phoneme_pair") == plot_phoneme_pair)
            )
        )
        .to_pandas()
        .assign(
            source=lambda df: df.source.map({"phon": "Acoustic", "behav": "Perceptual"})
        )
    )

    fb = FigureBuilder(figsize=(2.75, 2))
    ax = fb.ax
    legend_bbox_to_anchor = (0.85, 0.93)

    ax.set_xlabel("Peak decoding time (s)")
    ax.set_ylabel("Density")
    ax.set_xlim(plot_xlim)

    word_stim_info = word_end_df.filter(pl.col("word_end").is_in(plot_word_ends))
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

    palette = sns.color_palette(categorical_palette, n_colors=2)
    dummy_handles = [
        Line2D([0], [0], color=palette[0], linewidth=2, label="Acoustic"),
        Line2D([0], [0], color=palette[1], linewidth=2, label="Perceptual"),
    ]
    ax.legend(
        handles=dummy_handles,
        loc="upper right",
        bbox_to_anchor=legend_bbox_to_anchor,
        fontsize=10,
        frameon=True,
    )
    sns.despine(ax=ax, top=True, right=True)
    fb.stage("skeleton")

    source_order = ["Acoustic", "Perceptual"]
    for i, source in enumerate(source_order):
        subset = plot_data[plot_data["source"] == source]

        sns.kdeplot(
            data=subset,
            x="t_center",
            hue="source",
            hue_order=["Acoustic", "Perceptual"],
            palette=categorical_palette,
            linewidth=2,
            common_norm=False,
            clip=(0, None),
            ax=ax,
            legend=False,
        )
        sns.move_legend(
            ax,
            "upper right",
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=10,
            frameon=True,
            title=None,
        )
        fb.stage(f"data-{i}")

    return fb


# %%
fb = plot_peak_timing(
    plot_phoneme_pair="dn",
    plot_word_ends=["necessary"],
    plot_xlim=(0, 1.2),
    vline_extent=1.1,
)
fb.fig.savefig(Path(outdir) / "decoding_timing-necessary.pdf")
fb.render(outdir_talk / "decoding_timing-necessary", fmt="pdf")

# %%
fb = plot_peak_timing(
    plot_phoneme_pair="dn",
    plot_word_ends=["desolate"],
    plot_xlim=(0, 0.7),
    vline_extent=1.1,
)
fb.fig.savefig(Path(outdir) / "decoding_timing-desolate.pdf")
fb.render(outdir_talk / "decoding_timing-desolate", fmt="pdf")

# %%
fb = plot_peak_timing(
    plot_phoneme_pair="dn",
    plot_word_ends=["desolate", "necessary"],
    plot_xlim=(0, 1.2),
    vline_extent=1.1,
)
fb.fig.savefig(Path(outdir) / "decoding_timing-both.pdf")
fb.render(outdir_talk / "decoding_timing-both", fmt="pdf")
fb.fig

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
        (pl.col("roc_auc_from_phon").std() / pl.col("roc_auc_from_phon").count().cast(pl.Float64).sqrt()).alias("phon_sem"),
        (pl.col("roc_auc_from_behav").std() / pl.col("roc_auc_from_behav").count().cast(pl.Float64).sqrt()).alias("behav_sem"),
    )
    .to_pandas()
)

ttest_t, ttest_p = stats.ttest_rel(
    subject_means["phon_mean"], subject_means["behav_mean"]
)
print(
    f"Paired t-test comparing phonetic decoding from phonetic vs. perceptual peaks: "
    f"t={ttest_t:.3f}, p={ttest_p:g}"
)

def _plot_decoding_phonetic():
    fb = FigureBuilder(figsize=(3, 2.75))
    ax = fb.ax

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Acoustic\nwindow", "Perceptual\nwindow"])
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Acoustic\nprediction\n(ROC-AUC)", rotation=0, labelpad=40, va="top")
    ax.yaxis.label.set_position((1, 1.0))
    ax.annotate(
        "/n/ vs. /d/",
        xy=(0.5, 0.0),
        xycoords=ax.yaxis.label,
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        color="#666666",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.3, 1.3)
    sns.despine(ax=ax)

    fb.stage("skeleton")

    # Draw individual subject lines
    for _, row in subject_means.iterrows():
        xs = [0, 1]
        ys = [row["phon_mean"], row["behav_mean"]]
        errs = [row["phon_sem"], row["behav_sem"]]
        ax.plot(xs, ys, color="gray", alpha=0.4, linewidth=1, zorder=1)
        ax.errorbar(xs, ys, yerr=errs, color="gray", alpha=0.4, fmt="o", ms=3, zorder=2, capsize=1.5)

    # Draw grand mean
    grand_early = subject_means["phon_mean"].mean()
    grand_late = subject_means["behav_mean"].mean()
    ax.plot([0, 1], [grand_early, grand_late], color="black", linewidth=2.5, zorder=3, alpha=0.7)
    grand_phon_sem = subject_means["phon_mean"].sem()
    grand_behav_sem = subject_means["behav_mean"].sem()
    ax.errorbar(
        [0, 1], [grand_early, grand_late],
        yerr=[grand_phon_sem, grand_behav_sem],
        color="black", fmt="o", ms=7, zorder=4, alpha=0.7, capsize=3,
    )

    # Significance bracket
    bracket_y, tick_h = 1.10, 0.03
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        [0, 0, 1, 1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
        color="black", linewidth=1.0, transform=trans, clip_on=False,
    )
    ax.text(0.5, bracket_y + 0.01, p_to_stars(ttest_p), ha="center", va="bottom", fontsize=11, transform=trans)

    fb.stage("data")
    return fb

fb = _plot_decoding_phonetic()
fb.fig.savefig(Path(outdir) / "decoding_phonetic.pdf")
fb.render(outdir_talk / "decoding_phonetic", fmt="pdf")
fb.fig

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

g.savefig(Path(outdir) / "decoding_comparison.pdf")

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

fig.savefig(Path(outdir) / "decoding_behavioral_improvement.pdf")

# %%
# Spaghetti plot: per-subject mean improvement from baseline
subject_means = (
    behav_improvement_df.group_by("subject")
    .agg(
        pl.mean("phon_baseline_diff"),
        pl.mean("behav_baseline_diff"),
        (pl.col("phon_baseline_diff").std() / pl.col("phon_baseline_diff").count().cast(pl.Float64).sqrt()).alias("phon_baseline_diff_sem"),
        (pl.col("behav_baseline_diff").std() / pl.col("behav_baseline_diff").count().cast(pl.Float64).sqrt()).alias("behav_baseline_diff_sem"),
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


def _plot_decoding_behavioral_improvement():
    fb = FigureBuilder(figsize=(3, 2.75))
    ax = fb.ax

    ymax = 1.1 * max(
        subject_means["phon_baseline_diff"].max() + subject_means["phon_baseline_diff_sem"].max(),
        subject_means["behav_baseline_diff"].max() + subject_means["behav_baseline_diff_sem"].max(),
    )
    ymin = 1.1 * min(
        subject_means["phon_baseline_diff"].min() - subject_means["phon_baseline_diff_sem"].max(),
        subject_means["behav_baseline_diff"].min() - subject_means["behav_baseline_diff_sem"].max(),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Acoustic\nwindow", "Perceptual\nwindow"])
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Perceptual\nprediction\n($\Delta$ROC-AUC)", rotation=0, labelpad=40)
    ax.yaxis.label.set_position((1, 1.0))
    ax.yaxis.set_label_position("right")
    ax.annotate(
        "heard /n/\nvs.\nheard /d/",
        xy=(0.5, 0.0),
        xycoords=ax.yaxis.label,
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        color="#666666",
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(ymin, ymax)
    sns.despine(ax=ax, left=True, top=True, right=False)

    fb.stage("skeleton")

    # Draw individual subject lines
    for _, row in subject_means.iterrows():
        xs = [0, 1]
        ys = [row["phon_baseline_diff"], row["behav_baseline_diff"]]
        errs = [row["phon_baseline_diff_sem"], row["behav_baseline_diff_sem"]]
        ax.plot(xs, ys, color="gray", alpha=0.4, linewidth=1, zorder=1)
        ax.errorbar(xs, ys, yerr=errs, color="gray", alpha=0.4, fmt="o", ms=3, zorder=2, capsize=1.5)

    # Draw grand mean
    grand_early = subject_means["phon_baseline_diff"].mean()
    grand_late = subject_means["behav_baseline_diff"].mean()
    ax.plot([0, 1], [grand_early, grand_late], color="black", linewidth=2.5, zorder=3, alpha=0.7)
    grand_phon_sem = subject_means["phon_baseline_diff"].sem()
    grand_behav_sem = subject_means["behav_baseline_diff"].sem()
    ax.errorbar(
        [0, 1], [grand_early, grand_late],
        yerr=[grand_phon_sem, grand_behav_sem],
        color="black", fmt="o", ms=7, zorder=4, alpha=0.7, capsize=3,
    )

    # Significance annotations
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x, p in zip([0, 1], [p_early, p_late]):
        ax.text(x, 0.94, p_to_stars(p), ha="center", va="bottom", fontsize=11, transform=trans)

    bracket_y, tick_h = 1.10, 0.03
    ax.plot(
        [0, 0, 1, 1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
        color="black", linewidth=1.0, transform=trans, clip_on=False,
    )
    ax.text(0.5, bracket_y + 0.01, p_to_stars(p_early_vs_late), ha="center", va="bottom", fontsize=11, transform=trans)

    ax.set_ylim(ymin, ymax)

    fb.stage("data")
    return fb

fb = _plot_decoding_behavioral_improvement()
fb.fig.savefig(Path(outdir) / "decoding_behavioral_improvement-no_baseline.pdf")
fb.render(outdir_talk / "decoding_behavioral_improvement-no_baseline", fmt="pdf")
fb.fig

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

# %% [markdown]
# ### Cross-window phonetic transfer, no training

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

# %%
# Load saved behavioral decoders
behavioral_decoder_checkpoints = {
    subject: torch.load(
        f"outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt"
    )
    for subject in tqdm(epochs.keys())
}

# %%
# Keys: electrodes with both a phonetic peak and a behavioral peak
behav_transfer_keys = phon_peaks_df.join(
    behav_peaks_df.rename({"smin": "smin_behav", "smax": "smax_behav"}),
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="inner",
)
behav_transfer_keys

# %%
# Apply the behavioral decoder (trained on behavioral target at the perceptual peak)
# to acoustic-window neural data, evaluated against the acoustic target.
# Transfer from perceptual window decoder to phonetic decoding task in phonetic window.
behav_on_phon_outcomes = []
group_cols_we = [
    "subject",
    "electrode_idx",
    "phoneme_pair",
    "word_end",
    "fold",
    "smin_behav",
    "smax_behav",
    "smin_phon",
    "smax_phon",
]
for key in tqdm(
    behav_transfer_keys.iter_rows(named=True), total=behav_transfer_keys.height
):
    outcomes_i = evaluate_behav_decoder_on_phon_window(
        data=paper_data,
        behavioral_decoder_checkpoints=behavioral_decoder_checkpoints,
        phonetic_decoder_checkpoints=phonetic_decoder_checkpoints,
        t_subject=key["subject"],
        t_electrode_idx=key["electrode_idx"],
        t_phoneme_pair=key["phoneme_pair"],
        t_word_end=key["word_end"],
        t_smin_phon=key["smin"],
        t_smax_phon=key["smax"],
        t_smin_behav=key["smin_behav"],
        t_smax_behav=key["smax_behav"],
    )
    behav_on_phon_outcomes.append(outcomes_i)

behav_on_phon_df = pl.from_pandas(
    pd.concat(behav_on_phon_outcomes, ignore_index=True)
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)

# %%
# ROC-AUC of behavioral decoder predicting acoustic target at acoustic window
behav_on_phon_roc_auc = pl_roc_auc(
    behav_on_phon_df.filter(pl.col("resampled").is_in([1, 6])),
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=group_cols_we,
    roc_auc_name="behav_decoder_phon_roc_auc",
)

# Compare to in-window acoustic decoding (already computed)
behav_on_phon_comparison_df = (
    behav_on_phon_roc_auc.join(
        phon_roc_auc_searchlight_df.rename({"phon_roc_auc": "in_window_phon_roc_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair", "fold"],
        how="left",
    )
    .filter(
        pl.col("smin_phon")
        == pl.col("smin"),  # ensure we're comparing predictions onto the same window
        pl.col("smax_phon") == pl.col("smax"),
    )
    .drop(["smin", "smax"])
)

# %%
# Mean over folds
behav_on_phon_mean = behav_on_phon_comparison_df.group_by(
    ["subject", "electrode_idx", "phoneme_pair", "word_end"]
).agg(
    pl.mean("behav_decoder_phon_roc_auc"),
    pl.mean("in_window_phon_roc_auc"),
).with_columns(
    (pl.col("behav_decoder_phon_roc_auc") - pl.col("in_window_phon_roc_auc")).alias("roc_auc_diff")
)

subject_means_phon_transfer = (
    behav_on_phon_mean.group_by("subject")
    .agg(
        pl.mean("in_window_phon_roc_auc").alias("in_window"),
        pl.mean("behav_decoder_phon_roc_auc").alias("transfer"),
    )
    .to_pandas()
)

t_phon, p_phon = stats.ttest_rel(
    subject_means_phon_transfer["in_window"], subject_means_phon_transfer["transfer"]
)
print(
    f"Phonetic transfer — acoustic decoding: in-window vs perceptual decoder transfer: "
    f"t={t_phon:.3f}, p={p_phon:g}"
)
behav_on_phon_mean.write_csv(Path(outdir) / "transfer-behavioral_decoder_on_phonetic_window.csv")

# %%
# spaghetti plot: acoustic in-window vs perceptual (behavioral) decoder on acoustic window
colors = sns.color_palette(categorical_palette, 2)
x0, x1 = 0, 1

in_col = "Acoustic\nwindow"
transfer_col = "Acoustic window\n(perceptual decoder)"

df_wide = (
    behav_on_phon_mean.rename(
        {
            "in_window_phon_roc_auc": in_col,
            "behav_decoder_phon_roc_auc": transfer_col,
        }
    )
    .to_pandas()
    .query('phoneme_pair == "dn"')
    .dropna(subset=[in_col, transfer_col])
)
subject_means_plot = (
    df_wide.groupby("subject")[[in_col, transfer_col]].mean().reset_index()
)


def _plot_decoding_acoustic_transfer():
    fb = FigureBuilder(figsize=(3, 2.5))
    ax = fb.ax

    ax.set_ylim(0.025, 1)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels([in_col, transfer_col])
    ax.set_ylabel("Acoustic prediction\n(ROC AUC)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    sns.despine(ax=ax)

    fb.stage("skeleton")

    for _, row in df_wide.iterrows():
        ax.plot([x0, x1], [row[in_col], row[transfer_col]], color="gray", alpha=0.2, linewidth=0.8, zorder=0)
    ax.scatter([x0] * len(df_wide), df_wide[in_col], color=colors[0], s=12, alpha=0.4, zorder=2)
    ax.scatter([x1] * len(df_wide), df_wide[transfer_col], color=colors[1], s=12, alpha=0.4, zorder=2)

    grand_0 = subject_means_plot[in_col].mean()
    grand_1 = subject_means_plot[transfer_col].mean()
    ax.plot([x0, x1], [grand_0, grand_1], color="black", linewidth=2.5, zorder=4, alpha=0.7)
    ax.scatter([x0, x1], [grand_0, grand_1], color="black", s=60, zorder=5, alpha=0.7)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    bracket_y, tick_h = 1.10, 0.03
    ax.plot(
        [x0, x0, x1, x1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
        color="black", linewidth=1.0, transform=trans, clip_on=False,
    )
    ax.text(0.5, bracket_y + 0.01, p_to_stars(p_phon), ha="center", va="bottom", fontsize=11, transform=trans)

    fb.stage("data")
    return fb

fb = _plot_decoding_acoustic_transfer()
fb.fig.savefig(Path(outdir) / "decoding_acoustic_transfer-roc_auc.pdf")
fb.render(outdir_talk / "decoding_acoustic_transfer-roc_auc", fmt="pdf")
fb.fig

# %%
g = sns.displot(
    data=behav_on_phon_mean.to_pandas(), x="roc_auc_diff", height=1.5, aspect=1.5,
    bins=10, color="gray", edgecolor="black", alpha=0.3
)
g.set_axis_labels("$\Delta$ROC-AUC", "")
g.axes[0, 0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
g.axes[0, 0].axvline(0, color="red", linestyle="--")
g.savefig(Path(outdir) / "decoding_acoustic_transfer-roc_auc_diff_dist.pdf",
          transparent=True)

# %% [markdown]
# #### Exploratory analyses relating phonetic transfer to other properties

# %%
behav_on_phon_mean_explo = (
    behav_on_phon_mean
    .join(behav_peaks_df, on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
           how="inner")
    .join(phon_peaks_df.rename({"smin": "smin_phon", "smax": "smax_phon"}),
            on=["subject", "electrode_idx", "phoneme_pair"])
)
behav_on_phon_mean_explo

# %%
(
    behav_on_phon_mean_explo.to_pandas()
    [["roc_auc_diff", "smax", "smax_phon",
      "phon_roc_auc", "behav_roc_auc_improvement"]].corr()
)

# %% [markdown]
# ### Cross-window behavioral transfer, no training

# %% [markdown]
# Apply the acoustic decoder (trained on acoustic target at the acoustic peak)
# to behavioral-window neural data, evaluated against the behavioral target.
# Ask: does the learned acoustic representation in the early window generalize to the perceptual representation in the later window?

# %%
phon_on_behav_outcomes = []
for key in tqdm(
    behav_transfer_keys.iter_rows(named=True), total=behav_transfer_keys.height
):
    outcomes_i = evaluate_phon_decoder_on_behav_window(
        data=paper_data,
        phonetic_decoder_checkpoints=phonetic_decoder_checkpoints,
        behavioral_decoder_checkpoints=behavioral_decoder_checkpoints,
        t_subject=key["subject"],
        t_electrode_idx=key["electrode_idx"],
        t_phoneme_pair=key["phoneme_pair"],
        t_word_end=key["word_end"],
        t_smin_phon=key["smin"],
        t_smax_phon=key["smax"],
        t_smin_behav=key["smin_behav"],
        t_smax_behav=key["smax_behav"],
    )
    phon_on_behav_outcomes.append(outcomes_i)

phon_on_behav_df = pl.from_pandas(
    pd.concat(phon_on_behav_outcomes, ignore_index=True)
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)

# %%
# ROC-AUC of acoustic decoder predicting behavioral target at behavioral window
phon_on_behav_roc_auc = pl_roc_auc(
    phon_on_behav_df,
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=group_cols_we,
    roc_auc_name="phon_decoder_behav_roc_auc",
)

# Compare to in-window behavioral decoding (behav_roc_auc_searchlight_df)
phon_on_behav_comparison_df = (
    phon_on_behav_roc_auc.join(
        behav_roc_auc_searchlight_df.rename(
            {
                "behav_roc_auc": "in_window_behav_roc_auc",
            }
        ),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
        how="left",
    )
    .filter(
        pl.col("smin_behav")
        == pl.col("smin"),  # ensure we're comparing predictions onto the same window
        pl.col("smax_behav") == pl.col("smax"),
    )
    .drop(["smin", "smax"])
)

# %%
# Mean over folds
phon_on_behav_mean = (
    phon_on_behav_comparison_df.with_columns(
        (
            (
                pl.col("phon_decoder_behav_roc_auc") - pl.col("behav_roc_auc_baseline")
            ).alias("phon_decoder_behav_roc_auc_improvement")
        ),
        (
            (
                pl.col("in_window_behav_roc_auc") - pl.col("behav_roc_auc_baseline")
            ).alias("in_window_behav_roc_auc_improvement")
        ),
    )
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.mean("phon_decoder_behav_roc_auc"),
        pl.mean("in_window_behav_roc_auc"),
        pl.mean("phon_decoder_behav_roc_auc_improvement"),
        pl.mean("in_window_behav_roc_auc_improvement"),
    )
    .with_columns(
        (pl.col("phon_decoder_behav_roc_auc") - pl.col("in_window_behav_roc_auc")).alias("roc_auc_diff")
    )
)
phon_on_behav_mean.write_csv(Path(outdir) / "transfer-phonetic_decoder_on_behav_window.csv")

subject_means_2 = (
    phon_on_behav_mean.group_by("subject")
    .agg(
        pl.mean("in_window_behav_roc_auc_improvement").alias("in_window"),
        pl.mean("phon_decoder_behav_roc_auc_improvement").alias("transfer"),
    )
    .to_pandas()
)

t_2, p_2 = stats.ttest_rel(subject_means_2["in_window"], subject_means_2["transfer"])
print(
    f"Analysis 2 — behavioral decoding: in-window vs acoustic decoder transfer: "
    f"t={t_2:.3f}, p={p_2:g}"
)

colors = sns.color_palette(categorical_palette, 2)
x0, x1 = 0, 1

in_col_2, transfer_col_2 = "Perceptual\nwindow", "Perceptual window\n(acoustic decoder)"

df_wide_2 = (
    phon_on_behav_mean.rename(
        {
            "in_window_behav_roc_auc": in_col_2,
            "phon_decoder_behav_roc_auc": transfer_col_2,
        }
    )
    .to_pandas()
    .query('phoneme_pair == "dn"')
    .dropna(subset=[in_col_2, transfer_col_2])
)
subject_means_2_plot = (
    df_wide_2.groupby("subject")[[in_col_2, transfer_col_2]].mean().reset_index()
)


def _plot_decoding_phon_on_behav():
    fb = FigureBuilder(figsize=(3, 2.5))
    ax = fb.ax

    ax.set_ylim(0.025, 1)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels([in_col_2, transfer_col_2])
    ax.set_ylabel("Perceptual prediction\n(ROC AUC)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    sns.despine(ax=ax, left=True, top=True, right=False)

    fb.stage("skeleton")

    for _, row in df_wide_2.iterrows():
        ax.plot([x0, x1], [row[in_col_2], row[transfer_col_2]], color="gray", alpha=0.2, linewidth=0.8, zorder=0)
    ax.scatter([x0] * len(df_wide_2), df_wide_2[in_col_2], color=colors[0], s=12, alpha=0.4, zorder=2)
    ax.scatter([x1] * len(df_wide_2), df_wide_2[transfer_col_2], color=colors[1], s=12, alpha=0.4, zorder=2)

    grand_0_2 = subject_means_2_plot[in_col_2].mean()
    grand_1_2 = subject_means_2_plot[transfer_col_2].mean()
    ax.plot([x0, x1], [grand_0_2, grand_1_2], color="black", linewidth=2.5, zorder=4, alpha=0.7)
    ax.scatter([x0, x1], [grand_0_2, grand_1_2], color="black", s=60, zorder=5, alpha=0.7)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    bracket_y, tick_h = 1.10, 0.03
    ax.plot(
        [x0, x0, x1, x1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
        color="black", linewidth=1.0, transform=trans, clip_on=False,
    )
    ax.text(0.5, bracket_y + 0.01, p_to_stars(p_2), ha="center", va="bottom", fontsize=11, transform=trans)

    fb.stage("data")
    return fb

fb = _plot_decoding_phon_on_behav()
fb.fig.savefig(Path(outdir) / "decoding_phon_decoder_on_behav_window.pdf")
fb.render(outdir_talk / "decoding_phon_decoder_on_behav_window", fmt="pdf")

# %%
g = sns.displot(
    data=phon_on_behav_mean.to_pandas(), x="roc_auc_diff", height=1.5, aspect=1.5,
    bins=10, color="gray", alpha=0.3
)
g.set_axis_labels("$\Delta$ROC-AUC", "")
g.axes[0, 0].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
g.axes[0, 0].axvline(0, color="red", linestyle="--")
g.savefig(Path(outdir) / "decoding_phon_on_behav-roc_auc_diff_dist.pdf",
          transparent=True)

# %% [markdown]
# ## Zoomin

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
hga_zoomin_keys.to_pandas().to_csv(
    Path(outdir) / "hga_zoomin_search_keys.csv", index=False
)

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
# this may fail depending on whether the specific electrode is a peak in this
# analysis run
try:
    fb = zoomin_hga(
        paper_data,
        "EC250",
        185,
        "dn",
        "desolate",
        hide_bottom=False,
        legend=False,
        **star_plot_kwargs,
    )
    fb.fig.savefig(Path(outdir) / "zoomin_EC250_185_dn_desolate.pdf")
    plt.close(fb.fig)
    None

    legend_fig = plt.figure(figsize=(2, 2))

    legend_handles_labels = fb.fig.axes[0].get_legend_handles_labels()
    # reverse sort
    legend_handles_labels = (
        legend_handles_labels[0][::-1],
        legend_handles_labels[1][::-1],
    )
    for handle in legend_handles_labels[0]:
        handle.set_linewidth(3)
        handle.set_color("black")
    legend_fig.legend(*legend_handles_labels, loc="center", frameon=True)

    legend_fig.savefig(Path(outdir) / "zoomin_legend.pdf")
except:
    pass

# %%
# this may fail depending on whether the specific electrode is a peak in this analysis run
try:
    fb = zoomin_hga(
        paper_data, "EC278", 38, "dn", "necessary", hide_bottom=True, **star_plot_kwargs
    )
    fb.fig.savefig(Path(outdir) / "zoomin_EC278_38_dn_necessary.pdf")
    fb.render(outdir_talk / "zoomin_EC278_38_dn_necessary", fmt="pdf")
except:
    pass

# %% [markdown]
# ## Quant HGA search

# %%
hga_df = paper_data.hga_df
early_polarity = paper_data.early_polarity
late_polarity = paper_data.late_polarity
reg_df = paper_data.reg_df

# %% [markdown]
# ### Overall HGA

# %%
pcc_epoch_data_cache = {}

# %%
fb = plot_condition_contrasts_single_figure(
    paper_data,
    textgrid_dir,
    epoch_data_cache=pcc_epoch_data_cache,
    ambiguous_response_threshold=ambiguous_response_threshold,
    plot_word_ends=["necessary"],
)
fb.fig.savefig(Path(outdir) / "condition_contrasts-necessary.pdf")
plt.close(fb.fig)

# %%
fb = plot_condition_contrasts_single_figure(
    paper_data,
    textgrid_dir,
    epoch_data_cache=pcc_epoch_data_cache,
    ambiguous_response_threshold=ambiguous_response_threshold,
    plot_word_ends=["desolate"],
)
fb.ax.set_xlim(0, 0.7)
fb.fig.savefig(Path(outdir) / "condition_contrasts-desolate.pdf")
plt.close(fb.fig)


# %%
fb = plot_condition_contrasts_single_figure(
    paper_data,
    textgrid_dir,
    epoch_data_cache=pcc_epoch_data_cache,
    ambiguous_response_threshold=ambiguous_response_threshold,
    plot_word_ends=["necessary", "desolate"],
    vline_extent=1.0,
    textgrid_kwargs=dict(
        include_offset=False, include_phonemes=False, vline_extent=1.0
    ),
    pval_thresholds=(0.00001,),
    plot_ylim=(-0.1, 0.7),
)
fb.fig.savefig(Path(outdir) / "condition_contrasts-both.pdf")
fb.render(outdir_talk / "condition_contrasts-both", fmt="pdf")
plt.close(fb.fig)

# %% [markdown]
# ## Behav stackplot


# %%
fb = plot_behav_barplot(
    all_md,
    "EC250",
    "dn",
    "desolate",
    [1, 3, 4, 6],
    legend=False,
    resampled_palette=resampled_palette_simplified,
)
fb.fig.savefig(Path(outdir) / "behav_barplot_EC250_dn_desolate.pdf")
plt.close(fb.fig)

# %%
fb = plot_behav_barplot(
    all_md,
    "EC250",
    "dn",
    "desolate",
    [1, 2, 3, 4, 5, 6],
    legend=False,
    resampled_palette=resampled_palette_simplified,
)
fb.fig.savefig(Path(outdir) / "behav_barplot_EC250_dn_desolate-all_steps.pdf")
plt.close(fb.fig)

# %%
fb = plot_behav_barplot(
    all_md,
    "EC278",
    "dn",
    "necessary",
    [1, 3, 4, 6],
    legend=False,
    resampled_palette=resampled_palette_simplified,
)
fb.fig.savefig(Path(outdir) / "behav_barplot_EC278_dn_necessary.pdf")
fb.render(outdir_talk / "behav_barplot_EC278_dn_necessary", fmt="pdf")

# %%
fb = plot_behav_barplot(
    all_md,
    "EC278",
    "dn",
    "necessary",
    [1, 2, 3, 4, 5, 6],
    legend=False,
    resampled_palette=resampled_palette_simplified,
)
fb.fig.savefig(Path(outdir) / "behav_barplot_EC278_dn_necessary-all_steps.pdf")
plt.close(fb.fig)

# %% [markdown]
# ## Polarity relationships

# %%
early_polarity_strict = early_polarity.reset_index()

# %%
late_polarity_strict = late_polarity.dropna().reset_index()
late_polarity_strict["lexical_evidence"] = (
    late_polarity_strict.word_end.str[0] == late_polarity_strict.phoneme_pair.str[1]
).astype(int)


# %% [markdown]
# ### Acoustic tuning vs. presence of perceptual response

# %%
def plot_summary_acoustic_vs_presence_of_response(phoneme_pair: str):
    # First: summarize relationship between early polarity
    # and on which words the subsequent effect appears
    f, ax = plt.subplots(figsize=(1.1, 1.5))

    word_end_1, word_end_2 = PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]
    completion_1 = "-" + word_end_1[1:]
    completion_2 = "-" + word_end_2[1:]

    sns.heatmap(
        (
            late_polarity_strict.query("phoneme_pair == @phoneme_pair")
            .pivot_table(
                index=["subject", "electrode_idx", "phoneme_pair"],
                columns="lexical_evidence",
                values="late_polarity",
                aggfunc="count",
            )
            .fillna(0)
            .astype(bool)
            .rename(columns={0: completion_1, 1: completion_2})
            # make sure we have both possible lexical evidences for this word pair
            .assign(
                **{
                    completion_1: lambda df: df.get(completion_1, False),
                    completion_2: lambda df: df.get(completion_2, False),
                }
            )
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
            .set_index([completion_1, completion_2, "early_polarity"])["count"]
            .unstack()
            .fillna(0)
            .sort_index(ascending=False)
        ),
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
f.savefig(Path(outdir) / "early_polarity-late_response.pdf")

# %%
f = plot_summary_acoustic_vs_presence_of_response("bm")
f.savefig(Path(outdir) / "early_polarity-late_response-bm.pdf")

# %%
f = plot_summary_acoustic_vs_presence_of_response("pb")
f.savefig(Path(outdir) / "early_polarity-late_response-pb.pdf")


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

    # Chi-square test on one-sided responses only
    # Create 2x2 contingency table: acoustic (rows) x perceptual completion (columns)
    chi2_table = df.values#[[completion_1, completion_2]].values
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(chi2_table)
    print(chi2, p_value)
    sig_stars = p_to_stars(p_value)

    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    if ax is None:
        fb = FigureBuilder(figsize=(1.3, 2))
        ax = fb.ax
    else:
        f = ax.get_figure()

    df_pct.plot(
        kind="bar",
        stacked=True,
        color=sns.color_palette(palette, n_colors=3),
        ax=ax,
        width=0.5,
    )
    sns.despine(ax=ax)
    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(2.3, 1), title="Perceptual\nresponse"
    )
    plt.setp(legend.get_title(), multialignment="center")
    ax.set_xlabel("Acoustic tuning")
    ax.set_ylabel(None)
    ax.set_xticks(range(len(df_pct.index)))
    ax.set_xticklabels(df_pct.index, rotation=0)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    bracket_y, tick_h = 1.10, 0.03
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        [0, 0, 1, 1],
        [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
        color="black",
        linewidth=1.0,
        transform=trans,
        clip_on=False,
    )
    ax.text(
        0.5,
        bracket_y + 0.01,
        sig_stars,
        ha="center",
        va="bottom",
        fontsize=11,
        transform=trans,
    )

    fb.stage("data")

    return fb


# %%
fb = plot_summary_acoustic_vs_presence_of_response_stackplot()
fb.fig.savefig(Path(outdir) / "early_polarity-late_response_stackplot.pdf")
fb.render(outdir_talk / "early_polarity-late_response_stackplot", fmt="pdf")

# %%
plot_summary_acoustic_vs_presence_of_response_stackplot("dn")
None

# %% [markdown]
# ### Presence of preceptual response vs. perceptual tuning

# %%
late_polarity_strict.groupby("phoneme_pair")[["lexical_evidence", "late_polarity"]].value_counts().unstack("late_polarity").fillna(0).astype(int)


# %%
def plot_summary_perceptual_vs_presence_of_response_stackplot(
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
        late_polarity_strict__
        [["lexical_evidence", "late_polarity"]]
        .value_counts().unstack("late_polarity")
        .fillna(0).astype(int)
    )

    df.index = df.index.map({0: completion_1, 1: completion_2})
    df.columns = df.columns.map({-1: phoneme_pair_label[0], 1: phoneme_pair_label[1]})
    df = df[list(phoneme_pair_label)]
    # # map to readable columns
    # column_order = [completion_1, completion_2, "Both"]
    # df.columns = df.columns.map(
    #     dict(zip([(True, False), (False, True), (True, True)], column_order))
    # )
    # df = df[column_order]

    # Chi-square test on one-sided responses only
    # Create 2x2 contingency table: acoustic (rows) x perceptual completion (columns)
    chi2_table = df.values
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(chi2_table)
    print(chi2, p_value)
    sig_stars = p_to_stars(p_value)

    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    if ax is None:
        f, ax = plt.subplots(figsize=(1.3, 2))
    else:
        f = ax.get_figure()

    df_pct.plot(
        kind="bar",
        stacked=True,
        color=sns.color_palette(palette, n_colors=3),
        ax=ax,
        width=0.5,
    )
    sns.despine(ax=ax)
    legend = ax.legend(
        loc="upper right", bbox_to_anchor=(2.1, 1), title="Perceptual\ntuning"
    )
    plt.setp(legend.get_title(), multialignment="center")
    ax.set_xlabel("Perceptual response")
    ax.set_ylabel(None)
    ax.set_xticks(range(len(df_pct.index)))
    ax.set_xticklabels(df_pct.index, rotation=0)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.axhline(50, color="red", linestyle="--")

    # bracket_y, tick_h = 1.10, 0.03
    # trans = blended_transform_factory(ax.transData, ax.transAxes)
    # ax.plot(
    #     [0, 0, 1, 1],
    #     [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
    #     color="black",
    #     linewidth=1.0,
    #     transform=trans,
    #     clip_on=False,
    # )
    # ax.text(
    #     0.5,
    #     bracket_y + 0.01,
    #     sig_stars,
    #     ha="center",
    #     va="bottom",
    #     fontsize=11,
    #     transform=trans,
    # )

    return f


# %%
plot_summary_perceptual_vs_presence_of_response_stackplot()
None

# %% [markdown]
# ### Acoustic tuning vs perceptual tuning

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
    .fillna(0)
    .astype(int)
)
preference_relationship_df

# %%
# Next: summarize relationship between early polarity and late polarity
# NB in this setup, we are separately counting responses to -ecessary and -esolate
# at each electrode, since the polarities of either the early or late response
# might differ

f, ax = plt.subplots(figsize=(1, 1))
sns.heatmap(
    preference_relationship_df,
    annot=True,
    vmin=0,
    cbar=False,
)

ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_ylabel("Acoustic\npreference", rotation=0, va="center", labelpad=34)
ax.set_xlabel("Perceptual\npreference")

f.savefig(Path(outdir) / "early_polarity-late_polarity.pdf")

# %% [markdown]
# #### Congruency analysis

# %%
# get diagonal and off-diagonal sums
congruent_responses = np.diag(preference_relationship_df).sum()
incongruent_responses = (
    preference_relationship_df.values.sum() - np.diag(preference_relationship_df).sum()
)

test = stats.binomtest(
    congruent_responses,
    congruent_responses + incongruent_responses,
    p=0.5,
    alternative="greater",
)

# write this result to a file that can be human readable but also aggregated into a meta-analysis later
with open(Path(outdir) / "early_late_preference_relationship.txt", "w") as f:
    f.write(
        f"Congruent responses: {congruent_responses}\n"
        f"Incongruent responses: {incongruent_responses}\n"
        f"Binomial test p-value: {test.pvalue:.4e}\n"
        f"Significant at alpha=0.05: {'Yes' if test.pvalue < 0.05 else 'No'}\n"
    )

print(f"Congruent responses: {congruent_responses}")
print(f"Incongruent responses: {incongruent_responses}")
print(f"Binomial test p-value: {test.pvalue:.4e}")
print(f"Significant at alpha=0.05: {'Yes' if test.pvalue < 0.05 else 'No'}")

# %% [markdown]
# #### More in-depth selectivity relationship

# %%
(
    preference_relationship_df.sum(axis=0).loc["d"],
    preference_relationship_df.sum().sum(),
    preference_relationship_df.sum(axis=0) / preference_relationship_df.sum().sum(),
)

# %%
from scipy.stats import chi2_contingency



# %%
preference_relationship_pct_df = (
    preference_relationship_df.div(preference_relationship_df.sum(axis=1), axis=0) * 100
)
preference_relationship_pct_df

# %%
# chi2 test
chi2, p_value, dof, expected = chi2_contingency(preference_relationship_df.values)
print(f"Chi-square test: chi2={chi2:.3f}, p={p_value:.4e}")
sig_stars = p_to_stars(p_value)


def _plot_polarity_stackbar():
    fb = FigureBuilder(figsize=(1.3, 2))
    ax = fb.ax

    preference_relationship_pct_df.plot(
        kind="bar", stacked=True, width=0.5,
        color=sns.color_palette("Set2", n_colors=2), ax=ax,
    )

    # Set bars invisible for skeleton
    for patch in ax.patches:
        patch.set_alpha(0)
        patch.set_edgecolor("none")

    ax.set_xlabel("Acoustic tuning", fontsize=12)
    ax.set_ylabel(None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(title="Perceptual\ntuning", loc="upper right", bbox_to_anchor=(2.1, 1))
    sns.despine(ax=ax)

    fb.stage("skeleton")

    # Restore bars
    for patch in ax.patches:
        patch.set_alpha(1)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    ax.axhline(y=50, color="black", linestyle="--", linewidth=1, alpha=0.5)

    bracket_y, tick_h = 1.10, 0.03
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        [0, 0, 1, 1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
        color="black", linewidth=1.0, transform=trans, clip_on=False,
    )
    ax.text(0.5, bracket_y + 0.01, sig_stars, ha="center", va="bottom", fontsize=11, transform=trans)

    fb.stage("data")
    return fb

fb = _plot_polarity_stackbar()
fb.fig.savefig(Path(outdir) / "early_polarity-late_polarity_stackbar.pdf")
fb.render(outdir_talk / "early_polarity-late_polarity_stackbar", fmt="pdf")

# %% [markdown]
# ### Compare congruency analysis vs. decoder analysis
#
# Consistent results would make sense here -- sites with congruent acoustic and perceptual responses should be more likely to have a shared representation that supports cross-window transfer. Let's check that.

# %%
transfer_congruency_df = (
    late_polarity_strict.drop_duplicates(
        ["subject", "electrode_idx", "phoneme_pair", "word_end"]
    )
    .drop(columns=["lexical_evidence"])
    .merge(
        early_polarity_strict.drop_duplicates(),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    )
    .merge(
        phon_on_behav_mean.to_pandas(),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="left",
    )
    .merge(
        behav_on_phon_mean.to_pandas(),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="left",
    )

    # merge in facts about decoder timing as well
    .merge(
        behav_peaks_df.to_pandas()[["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax", "behav_roc_auc_improvement"]]
        .rename(columns={"smin": "smin_behav", "smax": "smax_behav"}),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    ).merge(
        phon_peaks_df.to_pandas()[["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "phon_roc_auc"]]
        .rename(columns={"smin": "smin_phon", "smax": "smax_phon"}),
        on=["subject", "electrode_idx", "phoneme_pair"],
    )
)
transfer_congruency_df["behav_transfer_effect"] = transfer_congruency_df["behav_decoder_phon_roc_auc"] - transfer_congruency_df["in_window_phon_roc_auc"]
transfer_congruency_df["phon_transfer_effect"] = transfer_congruency_df["phon_decoder_behav_roc_auc"] - transfer_congruency_df["in_window_behav_roc_auc"]
transfer_congruency_df["congruent"] = transfer_congruency_df.early_polarity == transfer_congruency_df.late_polarity

transfer_congruency_df["tmin_phon"] = transfer_congruency_df.smin_phon / epoch_sfreq + epoch_tmin
transfer_congruency_df["tmax_phon"] = transfer_congruency_df.smax_phon / epoch_sfreq + epoch_tmin
transfer_congruency_df["tmin_behav"] = transfer_congruency_df.smin_behav / epoch_sfreq + epoch_tmin
transfer_congruency_df["tmax_behav"] = transfer_congruency_df.smax_behav / epoch_sfreq + epoch_tmin

# %%
sns.catplot(data=transfer_congruency_df, x="congruent", y="phon_transfer_effect", kind="box")

# %%
transfer_congruency_df.query("phoneme_pair == 'dn'").sort_values("behav_transfer_effect")

# %%
ambig_steps = paper_data.get_ambiguous_resampled_steps(3)


# %%
def plot_for_congruency(transfer_row):
    plot_subject = transfer_row.subject
    plot_electrode_idx = transfer_row.electrode_idx
    plot_phoneme_pair = transfer_row.phoneme_pair
    plot_word_end = transfer_row.word_end

    vline_extent = 1.0
    fb = zoomin_hga(
        paper_data,
        plot_subject,
        plot_electrode_idx,
        plot_phoneme_pair,
        plot_word_end,
        controlled_resampled_steps=ambig_steps[plot_subject, plot_phoneme_pair, plot_word_end],
        textgrid_dir=textgrid_dir,
        vline_extent=vline_extent,
    )

    # highlight behav and acoustic windows
    fb.fig.axes[0].axvspan(
        transfer_row.tmin_phon,
        transfer_row.tmax_phon,
        ymax=vline_extent,
        color="blue",
        alpha=0.3,)

    fb.fig.axes[1].axvspan(
        transfer_row.tmin_behav,
        transfer_row.tmax_behav,
        ymax=vline_extent,
        color="orange",
        alpha=0.3,)

    return fb.fig


# %%
explo_plots = transfer_congruency_df.query("phoneme_pair == 'dn'").sort_values("behav_transfer_effect")

# write to pdfpages
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages(Path(outdir) / "congruency_exploration.pdf")

def add_title_page(pdf, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=24)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

with pdf:
    for congruent, rows in explo_plots.groupby("congruent"):
        # write title page
        add_title_page(pdf, f"Congruent: {congruent}, worst transfer")
        for _, row in rows.head(5).iterrows():
            print(row[["subject", "electrode_idx", "phoneme_pair", "word_end", "behav_transfer_effect", "phon_transfer_effect"]])
            f = plot_for_congruency(row)
            f.suptitle(
                f"Subject {row.subject}, Electrode {row.electrode_idx}, "
                f"{row.phoneme_pair}, {row.word_end}\n"
                f"Behav transfer effect: {row.behav_transfer_effect:.3f}, "
                f"Phon transfer effect: {row.phon_transfer_effect:.3f},\n"
                f"Behav ROC AUC improvement: {row.behav_roc_auc_improvement:.3f}, "
                f"Phon ROC AUC: {row.phon_roc_auc:.3f}",
                fontsize=10,
            )
            pdf.savefig(f)
            plt.close(f)

        add_title_page(pdf, f"Congruent: {congruent}, best transfer")
        for _, row in rows.tail(5).iterrows():
            print(row[["subject", "electrode_idx", "phoneme_pair", "word_end", "behav_transfer_effect", "phon_transfer_effect"]])
            f = plot_for_congruency(row)
            f.suptitle(
                f"Subject {row.subject}, Electrode {row.electrode_idx}, "
                f"{row.phoneme_pair}, {row.word_end}\n"
                f"Behav transfer effect: {row.behav_transfer_effect:.3f}, "
                f"Phon transfer effect: {row.phon_transfer_effect:.3f},\n"
                f"Behav ROC AUC improvement: {row.behav_roc_auc_improvement:.3f}, "
                f"Phon ROC AUC: {row.phon_roc_auc:.3f}",
                fontsize=10,
            )
            pdf.savefig(f)
            plt.close(f)

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
extra_cols = ["behav_smin", "behav_smax", "phon_smin", "phon_smax"]

# Only sites with a valid late window
reg_df_valid = reg_df.dropna(subset=["hga_late_signed"])

unambig_late_results = []
for site_key, site_data in reg_df_valid.groupby(site_cols + extra_cols):
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
            **dict(zip(site_cols + extra_cols, site_key)),
            "n_unambig": len(unambig),
            "n_grp0": len(grp0),
            "n_grp1": len(grp1),
            "t": t if not np.isnan(p) else np.nan,
            "p_unambig": p,
            "late_on_unambig": has_unambig_response,
        }
    )

unambig_late_df = pd.DataFrame(unambig_late_results).dropna()

# Merge in behav and phon peaks
unambig_late_df = unambig_late_df.merge(
    behav_peaks_df.to_pandas()[
        ["subject", "electrode_idx", "phoneme_pair", "word_end", "behav_roc_auc_improvement"]
    ],
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    how="inner",
).merge(
    phon_peaks_df.to_pandas()[["subject", "electrode_idx", "phoneme_pair", "phon_roc_auc"]],
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="inner",
)

n_total = len(unambig_late_df)
n_both = unambig_late_df["late_on_unambig"].sum()
n_ambig_only = n_total - n_both

# write this result to a file that can be human readable but also aggregated into a meta-analysis later
with open(Path(outdir) / "late_response_summary.txt", "w") as f:
    f.write(
        f"Sites with valid late window: {n_total}\n"
        f"  Late response on BOTH unambiguous and ambiguous trials: {n_both} ({100 * n_both / n_total:.0f}%)\n"
        f"  Late response on ambiguous trials ONLY: {n_ambig_only} ({100 * n_ambig_only / n_total:.0f}%)\n"
    )
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
print(stats.ttest_ind(
    unambig_late_df.query("late_on_unambig")["behav_smax"],
    unambig_late_df.query("~late_on_unambig")["behav_smax"],
))
sns.catplot(data=unambig_late_df, x="late_on_unambig", y="behav_smax", kind="box")

# %%
unambig_late_df["phon_behav_smax_latency"] = unambig_late_df.behav_smax - unambig_late_df.phon_smax
print(stats.ttest_ind(
    unambig_late_df.query("late_on_unambig")["phon_behav_smax_latency"],
    unambig_late_df.query("~late_on_unambig")["phon_behav_smax_latency"],
))

# %%
unambig_late_df["behav_tmax"] = unambig_late_df.behav_smax / epoch_sfreq + epoch_tmin
sns.displot(
    data=(
        late_polarity_strict.drop_duplicates(
            ["subject", "electrode_idx", "phoneme_pair", "word_end"]
        )
        .drop(columns=["lexical_evidence"])
        .merge(
            early_polarity_strict.drop_duplicates(),
            on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        )
        .merge(unambig_late_df, on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="left")
        .assign(congruent=lambda xs: xs.early_polarity == xs.late_polarity)
    ),
    x="behav_tmax", hue="late_on_unambig",
    # kind="kde", cut=0,
    row="phoneme_pair",
    height=2, aspect=2
)

# %%
# In congruent late_on_unambig windows, does cross-decoding with retraining work?
# Show that the signals are still distinct
sns.catplot(
    data=(
        pd.merge(unambig_late_df,
                phon_roc_auc_pivot_df.to_pandas(),
                on=["subject", "electrode_idx", "phoneme_pair"])
        .assign(roc_auc_diff=lambda xs: xs.roc_auc_from_phon - xs.roc_auc_from_behav)
    ),
    x="late_on_unambig",
    y="roc_auc_diff"
)

# %%
# In congruent late_on_unambig windows, does cross-decoding with retraining work?
# Show that the signals are still distinct
sns.catplot(
    data=(
        pd.merge(unambig_late_df,
                behav_improvement_df.to_pandas(),
                on=["subject", "electrode_idx", "phoneme_pair", "word_end"])
    ),
    x="late_on_unambig",
    y="behav_phon_diff"
)

# %%
print(stats.ttest_ind(
    unambig_late_df.query("late_on_unambig")["behav_roc_auc_improvement"],
    unambig_late_df.query("~late_on_unambig")["behav_roc_auc_improvement"],
))
sns.catplot(data=unambig_late_df, x="late_on_unambig", y="behav_roc_auc_improvement",
            hue="phoneme_pair", kind="box")

# %%
print(stats.ttest_ind(
    unambig_late_df.query("late_on_unambig")["phon_roc_auc"],
    unambig_late_df.query("~late_on_unambig")["phon_roc_auc"],
))
sns.catplot(data=unambig_late_df, x="late_on_unambig", y="phon_roc_auc",
            hue="phoneme_pair", kind="box")

# %%
# Transfer results in sites with unambiguous late responses vs. not
sns.catplot(
    data=pd.merge(
        unambig_late_df,
        behav_on_phon_mean.to_pandas()[["subject", "electrode_idx", "phoneme_pair", "word_end",
                                        "in_window_phon_roc_auc", "behav_decoder_phon_roc_auc",
                                        "roc_auc_diff"]],
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],     
    ),
    x="late_on_unambig",
    y="roc_auc_diff",
    kind="box",
    height=3
)

# %%
(
    late_polarity_strict.drop_duplicates(
        ["subject", "electrode_idx", "phoneme_pair", "word_end"]
    )
    .drop(columns=["lexical_evidence"])
    .merge(
        early_polarity_strict.drop_duplicates(),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    )
    .merge(unambig_late_df, on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="left")
    .assign(congruent=lambda xs: xs.early_polarity == xs.late_polarity,
            t_abs=lambda xs: xs.t.abs())
    [["late_on_unambig", "congruent"]].value_counts()
    .unstack("congruent")
)

# %%
# Polarity vs. late unambig or late all

polarity_vs_unambig_late_df = pd.merge(
    early_polarity_strict,
    unambig_late_df[["subject", "electrode_idx", "phoneme_pair", "word_end", "late_on_unambig"]],
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    how="inner",
).pipe(
    lambda df: df.assign(
        early_selectivity=df.early_polarity.map({-1: "d", 1: "n"}),
    )
).groupby("early_selectivity")["late_on_unambig"].value_counts().unstack().fillna(0)
polarity_vs_unambig_late_df.columns = polarity_vs_unambig_late_df.columns.map(
    {False: "Ambig.\ntrials", True: "Ambig.\nand unambig.\ntrials"}
)

# chi2 test
chi2, p_value, dof, expected = chi2_contingency(polarity_vs_unambig_late_df.values)
print(f"Chi-square test: chi2={chi2:.3f}, p={p_value:.4e}")
chi2_sig_stars = p_to_stars(p_value)

# binomial test on each row
binom_results = []
for early_sel, row in polarity_vs_unambig_late_df.iterrows():
    n_ambig_only = row["Ambig.\ntrials"]
    n_total = row.sum()
    test = stats.binomtest(n_ambig_only, n_total, p=0.5, alternative="greater")
    binom_results.append(
        {
            "early_selectivity": early_sel,
            "n_ambig_only": n_ambig_only,
            "n_total": n_total,
            "p_value": test.pvalue,
            "significant": test.pvalue < 0.05,
        }
    )
binom_results_df = pd.DataFrame(binom_results)

polarity_vs_unambig_late_df = polarity_vs_unambig_late_df.div(polarity_vs_unambig_late_df.sum(axis=1), axis=0) * 100

def _plot_unambig_stackbar():
    fb = FigureBuilder(figsize=(1.3, 2))
    ax = fb.ax

    polarity_vs_unambig_late_df.plot(
        kind="bar", stacked=True, width=0.5,
        color=sns.color_palette("Set2", n_colors=2), ax=ax,
    )

    # Set bars invisible for skeleton, but keep old legend handles
    old_handles, old_labels = ax.get_legend_handles_labels()
    from copy import deepcopy
    old_handles = deepcopy(old_handles)
    for patch in ax.patches:
        patch.set_alpha(0)
        patch.set_edgecolor("none")

    ax.set_xlabel("Acoustic tuning", fontsize=12)
    ax.set_ylabel(None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.legend(handles=old_handles, labels=old_labels,
              title="Perceptual\nresponse", loc="upper right",
              bbox_to_anchor=(2.6, 1))

    sns.despine(ax=ax)

    fb.stage("skeleton")

    # Restore bars
    for patch in ax.patches:
        patch.set_alpha(1)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)

    # add stars for binomial test
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, row in binom_results_df.iterrows():
        early_sel = row.early_selectivity
        p_value = row.p_value
        sig_stars = p_to_stars(p_value)
        if sig_stars:
            tick_y = 1.01
            ax.text(i, tick_y, sig_stars, ha="center", va="bottom", fontsize=11,
                    transform=trans)

    # add stars for chi2 test
    bracket_y, tick_h = 1.19, 0.03
    ax.plot(
        [0, 0, 1, 1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
        color="black", linewidth=1.0, transform=trans, clip_on=False,
    )
    ax.text(0.5, bracket_y + 0.01, chi2_sig_stars, ha="center", va="bottom", fontsize=11, transform=trans)

    fb.stage("data")
    return fb

fb = _plot_unambig_stackbar()
fb.fig.savefig(Path(outdir) / "early_polarity-late_unambig_response_stackbar.pdf")
fb.render(outdir_talk / "early_polarity-late_unambig_response_stackbar", fmt="pdf")
fb.fig

# %%
# Late polarity vs. late unambig or late all

late_polarity_vs_unambig_late_df = pd.merge(
    late_polarity_strict,
    unambig_late_df[["subject", "electrode_idx", "phoneme_pair", "word_end", "late_on_unambig"]],
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    how="inner",
).pipe(
    lambda df: df.assign(
        late_selectivity=df.late_polarity.map({-1: "d", 1: "n"}),
    )
).groupby("late_selectivity")["late_on_unambig"].value_counts().unstack().fillna(0)
late_polarity_vs_unambig_late_df.columns = late_polarity_vs_unambig_late_df.columns.map(
    {False: "Unambig.\nsounds", True: "Unambig.\nand ambig.\nsounds"}
)
late_polarity_vs_unambig_late_df = late_polarity_vs_unambig_late_df.div(late_polarity_vs_unambig_late_df.sum(axis=1), axis=0) * 100

f, ax = plt.subplots(figsize=(1.3, 2))
late_polarity_vs_unambig_late_df.plot(
    kind="bar",
    stacked=True,
    width=0.5,
    color=sns.color_palette("Set2", n_colors=2),
    ax=ax,
)

plt.xlabel("Perceptual tuning", fontsize=12)
plt.ylabel(None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(title="Perceptual\nresponse", loc="upper right", bbox_to_anchor=(2.45, 1))
sns.despine(ax=ax)

f.savefig(Path(outdir) / "late_polarity-late_unambig_response_stackbar.pdf")

# %% [markdown]
# ## Perceptual contrast on unambiguous trials, split by late-response generalization
#
# Compare sites that show a late behavioral response on *unambiguous* trials
# (`late_on_unambig=True`) vs. those that only show it on ambiguous trials.
# Both groups are evaluated here on unambiguous trials (resampled 1 and 6),
# using `behavior_dummy_forced` as the condition variable and `late_polarity`
# sign correction — i.e. the same signed HGA contrast used to classify sites.

# %%
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
# # phonetic response on electrodes that don't show a behavioral response
# ax, _, _ = plot_condition_contrast(
#     (
#         paper_data.plot_phon_phon_df.join(
#             paper_data.plot_behav_behav_df,
#             on=["subject", "electrode_idx", "phoneme_pair"],
#             how="inner",
#         ).filter(pl.col("phoneme_pair") == "dn")
#     ),
#     "categorical_acoustic_cue",
#     data=paper_data,
#     textgrid_dir=textgrid_dir,
#     polarity_correct="early",
#     epoch_data_cache=pcc_epoch_data_cache,
#     pval_thresholds=(0.0000001, 0.000001, 0.00001),
# )
# ax.set_xlim(0, 2.0)

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

    f.savefig(Path(outdir) / "perceptual_contrast_unambig_split-{plot_word_end}.pdf")
    plt.close()

None

# %%
# Behavioral contrast on electrodes with vs without late unambiguous response,
# plotted on the UNambiguous trials

plot_word_ends = ["necessary", "desolate"]
plot_xlim = (0, 1.2)

plot_unambig = paper_data.plot_phon_phon_df.filter(
    pl.col("resampled").is_in([1.0, 6.0]),
    # pl.col("word_end").is_in(plot_word_ends),
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

plot_palette = sns.color_palette("Set1", 2)
pval_thresholds = (0.001, 0.01)


def _plot_perceptual_contrast_both():
    fb = FigureBuilder(figsize=(3, 2))
    ax = fb.ax

    ax.set_xlim(*plot_xlim)
    ax.set_ylabel("HGA effect size ($z$)")
    ax.set_xlabel("Time from word onset (s)")

    fb.stage("skeleton")

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
        # label=f"Ambig. and unambig. (n={n_generalize})",
        label=f"Ambig.\nand unambig.",
        pval_thresholds=pval_thresholds,
    )

    fb.stage("generalize")

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
        # label=f"Ambig-only (n={n_specific})",
        label=f"Ambig-only",
        ttest_bar_y_ratio=0.87,
        pval_thresholds=pval_thresholds,
    )

    if p_handles is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles += p_handles
        labels += p_labels
        ax.legend(
            handles=handles, labels=labels,
            handler_map={Rectangle: HandlerRectangle()},
            fontsize=8, loc="upper right", bbox_to_anchor=(1.5, 1.0),
        )

    fb.stage("specific")
    return fb

fb = _plot_perceptual_contrast_both()
fb.fig.savefig(Path(outdir) / "perceptual_contrast_unambig_split-both.pdf")
fb.render(outdir_talk / "perceptual_contrast_unambig_split-both", fmt="pdf")
fb.fig

# %%
# Behavioral contrast on electrodes with vs without late unambiguous response,
# plotted on the UNambiguous trials, with early polarity correction (so you can see the phon effect)

plot_word_ends = ["necessary", "desolate"]
plot_xlim = (0, 1.2)

plot_unambig = paper_data.plot_phon_phon_df.filter(
    pl.col("resampled").is_in([1.0, 6.0]),
    # pl.col("word_end").is_in(plot_word_ends),
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
pval_thresholds = (0.001, 0.01)

_, p_handles, p_labels = plot_condition_contrast(
    plot_generalize,
    "behavior_dummy_forced",
    data=paper_data,
    textgrid_dir=textgrid_dir,
    polarity_correct="early",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=ax,
    color=plot_palette[0],
    annotate=True,
    textgrid_kwargs=dict(include_phonemes=False, include_offset=False),
    label=f"Generalizes (n={n_generalize})",
    pval_thresholds=pval_thresholds,
)
plot_condition_contrast(
    plot_specific,
    "behavior_dummy_forced",
    data=paper_data,
    textgrid_dir=textgrid_dir,
    polarity_correct="early",
    epoch_data_cache=pcc_epoch_data_cache,
    ax=ax,
    color=plot_palette[1],
    annotate=False,
    label=f"Ambig-only (n={n_specific})",
    ttest_bar_y_ratio=0.87,
    pval_thresholds=pval_thresholds,
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
        loc="upper right",
        bbox_to_anchor=(1.3, 1.0),
    )

f.savefig(Path(outdir) / "perceptual_contrast_unambig_split-both-early_polarity_correction.pdf")

# %%
# Behavioral contrast on electrodes with vs without late unambiguous response,
# plotted on the ambiguous trials

plot_word_ends = ["necessary", "desolate"]
plot_xlim = (0, 1.2)

plot_unambig = paper_data.plot_phon_phon_df.filter(
    ~pl.col("resampled").is_in([1.0, 6.0]),
    # pl.col("word_end").is_in(plot_word_ends),
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
pval_thresholds = (0.001, 0.01)

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
    label=f"Ambig.\nand unambig. (n={n_generalize})",
    pval_thresholds=pval_thresholds,
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
    pval_thresholds=pval_thresholds,
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
        loc="upper right",
        bbox_to_anchor=(1.3, 1)
    )

f.savefig(Path(outdir) / "perceptual_contrast_ambig_split-both.pdf")

# %%
