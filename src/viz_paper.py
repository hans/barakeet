"""
Final visualization functions for paper figures.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.typing import ColorType
import mne
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import textgrid
from loguru import logger as L
from matplotlib import transforms
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import ttest_1samp, ttest_ind
from tqdm.auto import tqdm

from src.data import get_ambiguous_resampled_steps as _get_ambiguous_resampled_steps
from src.figure_builder import FigureBuilder
from src.stimuli import OFFSET_DICT

subject_enum = pl.Enum(
    [
        "EC237",
        "EC243",
        "EC248",
        "EC250",
        "EC253",
        "EC260",
        "EC270",
        "EC278",
        "EC279",
        "EC282",
        "EC287",
    ]
)
phoneme_pair_enum = pl.Enum(["bm", "dn", "pb"])
word_end_enum = pl.Enum(list(OFFSET_DICT.keys()))

# Helpers for typing and readability
Subject: TypeAlias = str
PhonemePair: TypeAlias = str
WordEnd: TypeAlias = str


epoch_tmin = -0.4
epoch_sfreq = 100

# resampled_palette = sns.color_palette("cool", n_colors=6)
resampled_cmap = {
    1: "#b9529f",
    2: "#9956a2",
    3: "#7292cb",
    4: "#7192cb",
    5: "#49c8f3",
    6: "#6fccdd",
}

@dataclass
class Factor:
    _levels: tuple[str, ...]
    _colors: tuple[ColorType, ...] | None = None
    _labels: tuple[str, ...] | None = None

    @classmethod
    def from_tuples(cls, *level_specs):
        levels = [x[0] for x in level_specs]
        colors = tuple(x[1] for x in level_specs)

        labels = None
        if len(level_specs[0]) > 2:
            labels = tuple(x[2] for x in level_specs)

        return cls(tuple(levels), colors, labels)

    def __post_init__(self):
        if self._colors is not None:
            assert len(self._levels) == len(self._colors)
        if self._labels is not None:
            assert len(self._levels) == len(self._labels)

    @property
    def order(self):
        return self._levels
    @property
    def label_order(self):
        return self._labels
    @property
    def label_dict(self):
        return dict(zip(self._levels, self._labels))
    @property
    def colors(self):
        if self._colors is None:
            return None
        return {v: self._colors[k] for k, v in enumerate(self._levels)}
    @property
    def label_colors(self):
        if self._labels is None or self._colors is None:
            return None
        return {v: self._colors[k] for k, v in enumerate(self._labels)}


F_PHONEME_PAIR = Factor(("dn", "bm", "pb"))
F_EARLY = Factor.from_tuples(
    ("perceptual", "#4e79a7", "Perceptual"),
    ("acoustic", "#f27200", "Acoustic"),
)
F_LATE = Factor(("absent", "one-sided", "two-sided"),
                sns.color_palette("Set3", n_colors=3),
                ["Absent", "One-sided", "Two-sided"])
F_ACOUSTIC_TYPES = Factor(["early", "late"],
                          sns.color_palette("tab10", n_colors=2),
                          ["Early only", "Early + late"])

CONJUNCTION_CATEGORIES = {
    "Acoustic": {
        "early_category": "acoustic",
        "late_category":  "absent",
    },
    "Acoustic + integration": {
        "early_category": "acoustic",
        "late_category": ["two-sided", "one-sided"],
    },
    "Perceptual": {
        "early_category": "perceptual",
        "late_category":  "absent",
    },
    "Perceptual + integration": {
        "early_category": "perceptual",
        "late_category":  ["two-sided", "one-sided"],
    }
}

# Highlight color for early and late perceptual effects
EPP_COLOR = "#1b7837"
LPP_COLOR = "#762a83"

# purple for light accent elements
ACCENT_COLOR = "#c2a5cf"        # light purple
ACCENT_COLOR_STRONG = "#9970ab" # if it needs to survive a thin stroke
ACCENT_COLOR_FILL = "#e7d4e8"   # span/shaded-region fills behind data


@dataclass
class PaperData:
    electrode_df: pl.DataFrame
    plot_phon_phon_df: pl.DataFrame
    """
    One row per (site × fold × trial), at the site's phoneme-peak window.
    Built from phon_pred_df joined to phon_peaks and all_md.

    Key columns:
      subject: Enum           – participant ID
      electrode_idx: Int64
      phoneme_pair: Enum       – 'bm' | 'dn' | 'pb'
      word_end: Enum           – which word context
      smin, smax: Int64        – peak window in samples (sfreq=100, tmin=-0.4 s)
      epoch_idx: Int64         – index into epochs[subject]
      fold: Int64              – CV fold
      resampled: Float64       – continuum step 1–6
      decoder_target: Int8     – 0 or 1 (phoneme identity)
      decoder_proba: Float64   – phoneme decoder P(target=1)
      decoder_prediction: Int64
      behavior_dummy_forced: Int64   – 0 or 1 (forced-choice behavioral response)
      label_behavior_forced: String  – e.g. 'b' or 'm'
      label_acoustic: String         – acoustic ground truth label
      follows_acoustics: Boolean
      textgrid_path: String
    """
    plot_behav_phon_df: pl.DataFrame
    """
    Same as plot_phon_phon_df but the window is chosen from the behavioral peak
    (behav_peaks_df), not the phoneme peak. Otherwise identical schema.
    """
    plot_behav_behav_df: pl.DataFrame
    """
    One row per (site × fold × trial), at the site's behavioral-peak window.
    Built from behav_pred_df joined to behav_peaks and all_md.
    Same identifiers as plot_phon_phon_df, but uses the behavioral decoder:

      full_decoder_proba: Float64  – behavioral decoder P(target=1)
      decoder_target: Int8         – behavioral target (= behavior_dummy_forced)
    """
    plot_phon_behav_df: pl.DataFrame
    """
    Behavioral decoder predictions at the phoneme-peak window (cross-decoder).
    Same schema as plot_behav_behav_df (full_decoder_proba), at phon-peak smin/smax.
    """
    behav_roc_auc_searchlight_df: pl.DataFrame
    """Fold-level behavioral ROC-AUC at the behavioral-peak window, with improvement over baseline."""
    phon_roc_auc_searchlight_df: pl.DataFrame
    """Fold-level phonetic ROC-AUC across all time windows (filtered to pre-word-end)."""
    all_md: pl.DataFrame
    word_end_df: pl.DataFrame
    epochs: dict[str, mne.Epochs]
    phon_peaks_df: pl.DataFrame
    """Best phonetic-peak window per site (highest phon_roc_auc within constraints)."""
    behav_peaks_df: pl.DataFrame
    """Best behavioral-peak window per site (highest behav_roc_auc_improvement > 0)."""
    behav_peaks_df_unfiltered: pl.DataFrame
    """Like behav_peaks_df but includes sites with behav_roc_auc_improvement ≤ 0."""
    behav_baseline_df: pl.DataFrame
    """Per-site per-fold baseline behavioral ROC-AUC (decoded from shuffled/baseline predictor)."""
    zoomin_keys: pl.DataFrame
    """Sites that have both a phonetic and a behavioral peak; used for HGA window search."""
    early_polarity: pd.DataFrame
    """
    Per-site sign of the phonetic HGA difference (category 1 minus 0) in the early window.
    Index: (subject, electrode_idx, phoneme_pair, word_end). Column: early_polarity (+1/-1).
    """
    late_polarity: pd.DataFrame
    """
    Per-site sign of the behavioral HGA difference (behavior=1 minus behavior=0) in the late window.
    Index: (subject, electrode_idx, phoneme_pair, word_end). Column: late_polarity (+1/-1).
    """
    hga_df: pd.DataFrame | None = None
    """
    Per-site × per-trial mean HGA in the early and late windows.
    Output of extract_hga_windows_df. None if not yet computed.
    Shape: ~40k rows × 21 columns.

    Site identifiers:
      subject: str                – participant ID
      electrode_idx: int64
      phoneme_pair: str           – 'bm' | 'dn' | 'pb'
      word_end: str               – word context

    Trial identifiers:
      epoch_idx: int64            – index into epochs[subject]
      resampled: float64          – acoustic continuum step 1.0–6.0
      decoder_target: int64       – acoustic category 0/1
      behavior_dummy_forced: int64 – forced-choice behavioral response 0/1
      follows_acoustics: bool
      mismatch: int64             – lexical context (-1/0/1)

    HGA values (NaN when no valid late window found):
      hga_early: float64          – mean HGA in phoneme-separability window
      hga_late: float64           – mean HGA in behavior window (best variant by |t-stat|)

    Window metadata (per site, same for all trials of that site):
      phon_tmin/tmax: float64     – phoneme window in seconds
      phon_smin/smax: int64       – phoneme window in samples
      behav_tmin/tmax: float64    – behavior window in seconds (NaN if no window)
      behav_smin/smax: float64    – behavior window in samples (NaN if no window)
      behav_steps_chosen: str     – repr of resampled steps used to find late window,
                                    e.g. '(3,)' | '(3, 4)' | 'None'
    """
    reg_df: pd.DataFrame | None = None
    """
    hga_df merged with early_polarity and late_polarity; adds signed HGA columns
    and is_ambiguous flag. None if not yet computed.
    Same shape as hga_df (~40k rows), adds these columns:

      early_polarity: float64     – sign(mean HGA at decoder_target=1 − 0) in early window;
                                    per-site constant (+1 or -1)
      late_polarity: float64      – sign(mean HGA at behavior=1 − 0) in late window;
                                    per-site constant (+1 or -1)
      hga_early_signed: float64   – hga_early * early_polarity (positive = acoustic cat 1)
      hga_late_signed: float64    – hga_late * late_polarity   (positive = behavior choice 1)
      is_ambiguous: bool          – True if this trial's resampled step is one of the
                                    behav_steps_chosen (i.e. the step used to define the
                                    late window); False for resampled=1, 6 and any step
                                    not in the chosen set
    """

    def get_ambiguous_resampled_steps(
        self,
        ambiguous_response_threshold: int = 2,
    ) -> dict[tuple[Subject, PhonemePair, WordEnd], set[int]]:
        """
        Thin wrapper around `src.data.get_ambiguous_resampled_steps`; see that function
        for the full docstring.
        """
        return _get_ambiguous_resampled_steps(
            self.all_md,
            ambiguous_response_threshold=ambiguous_response_threshold,
        )


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def add_textgrid(
    ax,
    textgrid_dir,
    ep_df=None,
    textgrid_file=None,
    rotation=0,
    include_phonemes=True,
    fontsize=10,
    include_first_phoneme_offset=True,
    include_offset=True,
    vline_extent=1.25,
    tmax=None,
):
    if ep_df is None and textgrid_file is None:
        raise ValueError("Either ep_df or textgrid_file must be provided")
    if textgrid_file is not None and ep_df is not None:
        raise ValueError("Only one of ep_df or textgrid_file should be provided")

    if ep_df is not None:
        textgrid_file = ep_df.sort_values("textgrid_path").textgrid_path.iloc[0]
    tg = textgrid.TextGrid.fromFile(Path(textgrid_dir) / textgrid_file)
    assert tg.getNames() == ["phonemes"]

    plot_intervals = [
        interval
        for interval in tg.tiers[0].intervals
        if interval.mark is not None and interval.mark.strip()
    ]
    for i, interval in enumerate(plot_intervals):
        if include_phonemes:
            plotx = interval.minTime + 0.035
            if tmax is not None and plotx > tmax:
                continue
            ax.text(
                plotx,
                1.025,
                interval.mark.strip(),
                rotation=rotation,
                ha="right",
                va="bottom",
                fontsize=fontsize,
                transform=transforms.blended_transform_factory(
                    ax.transData, ax.transAxes
                ),
            )

        # add offset of first phoneme as vertical line
        if include_first_phoneme_offset and i == 0:
            ax.axvline(
                interval.maxTime,
                ymax=vline_extent,
                linestyle="--",
                alpha=0.5,
                color="black",
                clip_on=False,
            )

        # word offset line
        if include_offset and i == len(plot_intervals) - 1:
            if tmax is None or interval.maxTime <= tmax:
                ax.axvline(
                    interval.maxTime,
                    ymax=vline_extent,
                    linestyle="--",
                    alpha=0.5,
                    color="blue",
                    clip_on=False,
                )


def _plot_phon_controlled(
    plot_phon_keys,
    ax,
    epoch_data,
    epochs_i,
    plot_phon_smin,
    plot_phon_smax,
    controlled_resampled_steps,
    color_strategy="resampled",
):
    # helper for zoomin_hga

    if color_strategy == "extreme_vs_controlled":
        palette = {
            False: "#9E9E9E",  # extreme values, 1 or 6
            True: "#D62728",  # controlled values
        }
    elif color_strategy == "resampled":
        palette = resampled_cmap
    else:
        raise ValueError(f"Unknown color strategy: {color_strategy}")

    linestyles = {0: "solid", 1: "dashed"}

    all_epoch_data = {}

    i = 0
    plot_phon_keys = plot_phon_keys.to_pandas()
    # exclude cases of extreme stimuli with non-acoustic-following behavior
    plot_phon_keys = plot_phon_keys[
        ~((plot_phon_keys.resampled == 1) & (plot_phon_keys.behavior_dummy_forced == 1))
        & ~(
            (plot_phon_keys.resampled == 6)
            & (plot_phon_keys.behavior_dummy_forced == 0)
        )
    ]
    plot_phon_keys["controlled"] = plot_phon_keys.resampled.isin(
        controlled_resampled_steps
    )

    word_ends = plot_phon_keys.word_end
    assert word_ends.nunique() == 1
    word_end = word_ends.iloc[0]

    for (
        controlled,
        decoder_target,
        label_behavior,
    ), rows in plot_phon_keys.groupby(
        ["controlled", "behavior_dummy_forced", "label_behavior_forced"]
    ):
        epoch_i = epoch_data[rows.epoch_idx, :]
        all_epoch_data[controlled, decoder_target] = epoch_i
        epoch_i = epoch_i[:, plot_phon_smin:plot_phon_smax]
        epoch_mean = epoch_i.mean(axis=0)
        epoch_sem = epoch_i.std(axis=0) / np.sqrt(epoch_i.shape[0])
        times = epochs_i.times[plot_phon_smin:plot_phon_smax]

        if color_strategy == "extreme_vs_controlled":
            color = palette[controlled]
        elif color_strategy == "resampled":
            if rows.resampled.nunique() > 1:
                L.warning(
                    f"Multiple resampled values for controlled={controlled}, decoder_target={decoder_target}, label_behavior={label_behavior}: {rows.resampled.unique()}. Using color for most common resampled value."
                )
            most_common_resampled = int(rows.resampled.mode().iloc[0])
            color = palette[most_common_resampled - 1]

        linewidth = 3

        # label = f"Chose $\\it{{{label_behavior}{word_end[1:]}}}$"
        label = f"Chose /{label_behavior}/"
        ax.plot(
            times,
            epoch_mean,
            label=label,
            color=color,
            ls=linestyles[decoder_target],
            linewidth=linewidth,
        )
        ax.fill_between(
            times,
            epoch_mean - epoch_sem,
            epoch_mean + epoch_sem,
            color=color,
            alpha=0.3,
            rasterized=True,
        )
        ax.set_xlim(times[0], times[-1])

        i += 1

    return all_epoch_data


def _plot_windowed_ttest(
    ax,
    all_epoch_data,
    group1,
    group2,
    test_smin,
    test_smax,
    test_window_size=4,
    test_window_stride=4,
    color="black",
    alpha=0.5,
    bar_height_ratio=0.01,
    bar_y_ratio=0.95,
):
    # Windowed t-test. helper for zoomin_hga

    # Get windowed means for each condition
    test_window_starts = np.arange(
        test_smin, test_smax - test_window_size + 1, test_window_stride
    )
    test_results = []
    for start in test_window_starts:
        group1_data = all_epoch_data.get(group1, np.empty((0, test_window_size)))[
            :, start : start + test_window_size
        ]
        group2_data = all_epoch_data.get(group2, np.empty((0, test_window_size)))[
            :, start : start + test_window_size
        ]
        # average over time within the window
        group1_data = group1_data.mean(axis=1)
        group2_data = group2_data.mean(axis=1)
        if len(group1_data) > 0 and len(group2_data) > 0:
            t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
        else:
            t_stat, p_value = np.nan, np.nan
        test_results.append((start, start + test_window_size, t_stat, p_value))
    test_results_df = pd.DataFrame(
        test_results, columns=["start_sample", "end_sample", "t_stat", "p_value"]
    )
    test_results_df["tmin"] = test_results_df["start_sample"] / epoch_sfreq + epoch_tmin
    test_results_df["tmax"] = test_results_df["end_sample"] / epoch_sfreq + epoch_tmin
    # print(test_results_df)

    ymin, ymax = ax.get_ylim()
    bar_height = (ymax - ymin) * bar_height_ratio
    bar_y = ymin + (ymax - ymin) * bar_y_ratio
    for row in test_results_df.itertuples():
        if row.p_value < 0.05:
            ax.barh(
                y=bar_y,
                width=row.tmax - row.tmin,
                left=row.tmin,
                height=bar_height,
                color=color,
                alpha=alpha,
                edgecolor="none",
            )


def zoomin_hga(
    data: PaperData,
    subject,
    electrode_idx,
    phoneme_pair,
    word_end,
    textgrid_dir,
    controlled=True,
    controlled_resampled_steps=(3,),
    resampled_palette=resampled_cmap,
    include_phonemes=True,
    include_offset=False,
    hide_bottom=False,
    legend=False,
    figsize=(5.25, 4),
    vline_extent=1.25,
    title=False,
    highlight_windows=True,
):
    """
    hide_bottom: skip xticks and xlabel; because it's going to be stacked into a vertical figure
    highlight_windows: shade the phonetic decoding window on the top (unambiguous)
        panel and the behavioral decoding window on the bottom (ambiguous) panel
    """
    assert set(controlled_resampled_steps).isdisjoint({1, 6}), (
        "Violated plotting assumption"
    )

    subplot_phon_phon_df = data.plot_phon_phon_df.filter(
        (pl.col("electrode_idx") == electrode_idx),
        (pl.col("subject") == subject),
        (pl.col("phoneme_pair") == phoneme_pair),
        (pl.col("word_end") == word_end),
    )
    subplot_behav_phon_df = data.plot_behav_phon_df.filter(
        (pl.col("electrode_idx") == electrode_idx),
        (pl.col("subject") == subject),
        (pl.col("phoneme_pair") == phoneme_pair),
        pl.col("word_end") == word_end,
    )

    # behav predictions
    subplot_behav_behav_df = data.plot_behav_behav_df.filter(
        (pl.col("electrode_idx") == electrode_idx),
        (pl.col("subject") == subject),
        (pl.col("phoneme_pair") == phoneme_pair),
        (pl.col("word_end") == word_end),
    )
    subplot_phon_behav_df = data.plot_phon_behav_df.filter(
        (pl.col("electrode_idx") == electrode_idx),
        (pl.col("subject") == subject),
        (pl.col("phoneme_pair") == phoneme_pair),
        (pl.col("word_end") == word_end),
    )

    assert subplot_phon_phon_df.select(pl.n_unique("smin")).item() == 1
    assert (
        subplot_behav_phon_df.group_by("word_end")
        .agg(pl.n_unique("smin"))
        .select(pl.max("smin"))
        .item()
        == 1
    )
    assert (
        subplot_behav_behav_df.group_by("word_end")
        .agg(pl.n_unique("smin"))
        .select(pl.max("smin"))
        .item()
        == 1
    )
    assert subplot_phon_behav_df.select(pl.n_unique("smin")).item() == 1

    ###

    epochs_i = data.epochs[subject]
    epoch_data = (
        epochs_i.copy().apply_baseline().get_data(picks=electrode_idx).squeeze(1)
    )

    plot_tmin = 0
    plot_tmax = (
        data.word_end_df.filter(
            pl.col("phoneme_pair") == phoneme_pair, (pl.col("word_end") == word_end)
        )
        .select(pl.max("word_end_offset"))
        .item()
        + 0.1
    )
    plot_smin = int((plot_tmin - epoch_tmin) * epoch_sfreq)
    plot_smax = int((plot_tmax - epoch_tmin) * epoch_sfreq)

    plot_highlight_phon_window = (
        subplot_phon_phon_df.select(["smin", "smax"]).unique().to_numpy().flatten()
    )
    plot_highlight_behav_window = (
        subplot_behav_behav_df.select(["smin", "smax"]).unique().to_numpy().flatten()
    )

    highlight_phon_times = epochs_i.times[
        [plot_highlight_phon_window[0], plot_highlight_phon_window[1]]
    ]
    highlight_behav_times = epochs_i.times[
        [plot_highlight_behav_window[0], plot_highlight_behav_window[1]]
    ]

    fb = FigureBuilder(figsize=figsize, nrows=2, ncols=1, sharex=True)
    axs = fb.axes

    if title:
        fb.fig.suptitle(
            f"Subject {subject}, Electrode {electrode_idx}, {phoneme_pair}, {word_end}"
        )

    ### HGA plot by stimulus step

    plot_phon_smin = plot_smin
    plot_phon_smax = plot_highlight_behav_window[1] + 10

    # --- Compute plot_epoch_keys first (needed for decorations) ---
    plot_epoch_keys = subplot_phon_phon_df.select(
        ["epoch_idx", "resampled", "textgrid_path"]
    ).unique()

    if controlled:
        plot_epoch_keys = (
            subplot_phon_phon_df.filter(
                pl.col("resampled").is_in([1, 6] + list(controlled_resampled_steps))
            )
            .select(
                [
                    "epoch_idx",
                    "resampled",
                    "behavior_dummy_forced",
                    "label_behavior_forced",
                    "textgrid_path",
                    "word_end",
                ]
            )
            .unique()
        )
    else:
        raise NotImplementedError()

    # --- Axis decorations (skeleton: axes/labels/vlines/textgrid in place, no data) ---
    pod_time = (
        data.word_end_df.filter(pl.col("word_end") == word_end)
        .select(pl.max("pod"))
        .item()
    )
    ep_df = plot_epoch_keys.to_pandas()
    for i, ax in enumerate(axs):
        ax.axvline(
            pod_time,
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            color="red",
            ymax=vline_extent if i == 0 else 1,
            clip_on=False,
        )
        ax.set_ylabel("HGA ($z$)")
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        sns.despine(ax=ax, top=True, right=True)

        include_phonemes_i = include_phonemes and i == 0
        vline_extent_i = vline_extent if i == 0 else 1
        add_textgrid(
            ax,
            textgrid_dir,
            ep_df=ep_df,
            include_phonemes=include_phonemes_i,
            vline_extent=vline_extent_i,
            include_offset=include_offset,
        )

    if hide_bottom:
        axs[-1].tick_params(axis="x", which="both", labelbottom=False)
    else:
        axs[-1].set_xlabel("Time from word onset (s)")

    if legend:
        # Placeholder legend with dummy handles; replaced after data is plotted
        phon_labels = list(phoneme_pair)[::-1]
        dummy_handles = [
            Line2D([0], [0], color="black", label=f"Chose /{l}/") for l in phon_labels
        ]
        axs[0].legend(
            handles=dummy_handles,
            title=None,
            fontsize=10,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.05, 1.575),
        )

    fb.stage("skeleton")

    # --- Plot top panel: unambiguous/extreme trials ---
    all_epoch_data_extreme = _plot_phon_controlled(
        plot_epoch_keys.filter(pl.col("resampled").is_in([1, 6])),
        axs[0],
        epoch_data,
        epochs_i,
        plot_phon_smin,
        plot_phon_smax,
        controlled_resampled_steps,
    )
    # add windowed t-test results for phonetic contrast
    _plot_windowed_ttest(
        axs[0],
        all_epoch_data_extreme,
        group1=(False, 0),
        group2=(False, 1),
        color="black",
        alpha=0.5,
        bar_height_ratio=0.04,
        test_smin=plot_phon_smin,
        test_smax=plot_phon_smax,
    )

    if legend:
        legend_handles_labels = axs[0].get_legend_handles_labels()
        # reverse sort
        legend_handles_labels = (
            legend_handles_labels[0][::-1],
            legend_handles_labels[1][::-1],
        )
        leg = axs[0].legend(
            *legend_handles_labels,
            title=None,
            fontsize=10,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.05, 1.575),
        )
        # make the lines black to make clear that this is not specific to the top plot
        for line in leg.get_lines():
            line.set_color("black")

    fb.stage("unambiguous")

    # --- Plot bottom panel: controlled/ambiguous trials ---
    all_epoch_data_controlled = _plot_phon_controlled(
        plot_epoch_keys.filter(pl.col("resampled").is_in(controlled_resampled_steps)),
        axs[1],
        epoch_data,
        epochs_i,
        plot_phon_smin,
        plot_phon_smax,
        controlled_resampled_steps,
    )
    # add windowed t-test results for behavior at controlled resampled step
    _plot_windowed_ttest(
        axs[1],
        all_epoch_data_controlled,
        group1=(True, 0),
        group2=(True, 1),
        color="black",
        alpha=0.5,
        bar_height_ratio=0.04,
        test_smin=plot_phon_smin,
        test_smax=plot_phon_smax,
    )
    if highlight_windows:
        # phonetic decoding window on the unambiguous (top) panel,
        # behavioral decoding window on the ambiguous (bottom) panel.
        axs[0].axvspan(
            highlight_phon_times[0], highlight_phon_times[-1],
            color="blue", alpha=0.15,
        )
        axs[1].axvspan(
            highlight_behav_times[0], highlight_behav_times[-1],
            color="orange", alpha=0.15,
        )

    fb.stage("controlled")

    return fb


def plot_congruency_compressed(
    data: PaperData,
    subject,
    electrode_idx,
    phoneme_pair,
    word_end,
    controlled_resampled_steps=None,
    behav_tmin=None,
    behav_tmax=None,
    perceptual_pad=0.1,
    highlight_phon_window=False,
    highlight_behav_window=False,
    phon_tmin=None,
    phon_tmax=None,
    resampled_palette=resampled_cmap,
    figsize=(8, 2.5),
    break_mark_size=0.015,
):
    """
    Compressed single-row plot with broken axis.

    Left segment: acoustic response on unambiguous trials (word onset to POD).
    Right segment: perceptual response on ambiguous trials (behavioral window ± pad).

    Parameters
    ----------
    behav_tmin, behav_tmax : float or None
        Behavioral window bounds in seconds. If None, extracted from
        data.behav_peaks_df.
    perceptual_pad : float
        Seconds of padding on each side of the perceptual window.
    highlight_phon_window : bool
        If True, draw blue axvspan on acoustic window.
    highlight_behav_window : bool
        If True, draw orange axvspan on perceptual window.

    Returns
    -------
    fig, (ax_left, ax_right)
    """
    from matplotlib.gridspec import GridSpec

    if controlled_resampled_steps is None:
        ambig_steps = data.get_ambiguous_resampled_steps()
        controlled_resampled_steps = ambig_steps.get(
            (subject, phoneme_pair, word_end), (3,)
        )

    assert set(controlled_resampled_steps).isdisjoint({1, 6})

    # --- Filter site data ---
    site_filter = (
        (pl.col("electrode_idx") == electrode_idx)
        & (pl.col("subject") == subject)
        & (pl.col("phoneme_pair") == phoneme_pair)
        & (pl.col("word_end") == word_end)
    )
    subplot_df = data.plot_phon_phon_df.filter(site_filter)

    # --- Extract epoch data ---
    epochs_i = data.epochs[subject]
    epoch_data = (
        epochs_i.copy().apply_baseline().get_data(picks=electrode_idx).squeeze(1)
    )

    # --- POD time ---
    pod_time = (
        data.word_end_df.filter(pl.col("word_end") == word_end)
        .select(pl.max("pod"))
        .item()
    )

    # --- Resolve acoustic and behavioral window ---
    if phon_tmin is None or phon_tmax is None:
        phon_row = data.phon_peaks_df.filter(
            pl.col("electrode_idx") == electrode_idx,
            pl.col("subject") == subject,
            pl.col("phoneme_pair") == phoneme_pair,
        )
        phon_tmin = int(phon_row["smin"][0]) / epoch_sfreq + epoch_tmin
        phon_tmax = int(phon_row["smax"][0]) / epoch_sfreq + epoch_tmin
    if behav_tmin is None or behav_tmax is None:
        behav_row = data.behav_peaks_df.filter(site_filter)
        behav_tmin = int(behav_row["smin"][0]) / epoch_sfreq + epoch_tmin
        behav_tmax = int(behav_row["smax"][0]) / epoch_sfreq + epoch_tmin

    # --- Time ranges ---
    left_tmin, left_tmax = 0.0, pod_time
    right_tmin = behav_tmin - perceptual_pad
    right_tmax = behav_tmax + perceptual_pad

    left_smin = int((left_tmin - epoch_tmin) * epoch_sfreq)
    left_smax = int((left_tmax - epoch_tmin) * epoch_sfreq)
    right_smin = int((right_tmin - epoch_tmin) * epoch_sfreq)
    right_smax = int((right_tmax - epoch_tmin) * epoch_sfreq)

    left_duration = left_tmax - left_tmin
    right_duration = right_tmax - right_tmin

    # --- Create figure with broken axis ---
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1, 2, width_ratios=[left_duration, right_duration], wspace=0.05, figure=fig
    )
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

    # --- Prepare trial metadata ---
    trial_keys = (
        subplot_df.select([
            "epoch_idx", "resampled", "behavior_dummy_forced",
            "label_behavior_forced", "word_end",
        ])
        .unique()
        .to_pandas()
    )
    # Exclude mismatches on unambiguous steps
    # trial_keys = trial_keys[
    #     ~((trial_keys.resampled == 1) & (trial_keys.behavior_dummy_forced == 1))
    #     & ~((trial_keys.resampled == 6) & (trial_keys.behavior_dummy_forced == 0))
    # ]

    linestyles = {0: "solid", 1: "dashed"}

    # --- Helper to plot traces on an axis ---
    def _plot_segment(ax, keys, smin, smax):
        times = epochs_i.times[smin:smax]
        for (behav, label_behav), rows in keys.groupby(
            ["behavior_dummy_forced", "label_behavior_forced"]
        ):
            epoch_i = epoch_data[rows.epoch_idx.values, smin:smax]
            mean = epoch_i.mean(axis=0)
            sem = epoch_i.std(axis=0) / np.sqrt(epoch_i.shape[0])
            most_common_resampled = int(rows.resampled.mode().iloc[0])
            color = resampled_cmap[most_common_resampled]
            label = f"Chose /{label_behav}/"
            ax.plot(times, mean, label=label, color=color,
                    ls=linestyles[behav], linewidth=3)
            ax.fill_between(times, mean - sem, mean + sem,
                            color=color, alpha=0.3, rasterized=True)

    # --- Plot left segment (acoustic: steps 1 & 6) ---
    left_keys = trial_keys[trial_keys.resampled.isin([1, 6])]
    _plot_segment(ax_left, left_keys, left_smin, left_smax)

    # --- Plot right segment (perceptual: ambiguous steps) ---
    right_keys = trial_keys[
        trial_keys.resampled.isin(list(controlled_resampled_steps))
    ]
    _plot_segment(ax_right, right_keys, right_smin, right_smax)

    # --- POD line on left segment ---
    ax_left.axvline(pod_time, linestyle="--", linewidth=2, alpha=0.5, color="red")

    # --- Optional window highlighting ---
    if highlight_phon_window and phon_tmin is not None and phon_tmax is not None:
        ax_left.axvspan(phon_tmin, phon_tmax, color="blue", alpha=0.3)
    if highlight_behav_window and behav_tmin is not None and behav_tmax is not None:
        ax_right.axvspan(behav_tmin, behav_tmax, color="orange", alpha=0.3)

    # --- Broken-axis styling ---
    sns.despine(ax=ax_left, top=True, right=True)
    sns.despine(ax=ax_right, top=True, left=True, right=True)
    ax_right.tick_params(labelleft=False, left=False)

    # Diagonal break marks
    d = break_mark_size
    kwargs = dict(color="k", clip_on=False, linewidth=1)
    kwargs_l = dict(transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs_l)
    # ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_l)
    kwargs_r = dict(transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-d, +d), (-d, +d), **kwargs_r)
    # ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_r)

    # --- Labels ---
    ax_left.set_ylabel("HGA ($z$)")

    # make a supxlabel that is the same fontsize as the ylabel
    fig.supxlabel("Time from word onset (s)", y=-0.2,
                  fontsize=ax_left.yaxis.label.get_size())

    return fig, (ax_left, ax_right)


def zoomin_search_hga(
    data: PaperData,
    subject,
    electrode_idx,
    phoneme_pair,
    word_end,
    textgrid_dir,
    controlled_resampled_search_steps=(3,),
    epoch_data=None,
    axs=None,
    figsize=(4, 7.5),
    title=False,
):
    """
    Like zoomin but we'll search over possible resampled steps, plot all of them
    along with a picture of the behavioral variability supporting this
    """
    assert set(controlled_resampled_search_steps).isdisjoint({1, 6}), (
        "Violated plotting assumption"
    )
    if axs is not None:
        assert len(axs) == 1 + len(controlled_resampled_search_steps), (
            "Number of axes must match number of conditions to plot"
        )

    subplot_phon_phon_df = data.plot_phon_phon_df.filter(
        (pl.col("electrode_idx") == electrode_idx),
        (pl.col("subject") == subject),
        (pl.col("phoneme_pair") == phoneme_pair),
        (pl.col("word_end") == word_end),
    )
    subplot_behav_phon_df = data.plot_behav_phon_df.filter(
        (pl.col("electrode_idx") == electrode_idx),
        (pl.col("subject") == subject),
        (pl.col("phoneme_pair") == phoneme_pair),
        (pl.col("word_end") == word_end),
    )

    # behav predictions
    subplot_behav_behav_df = data.plot_behav_behav_df.filter(
        (pl.col("electrode_idx") == electrode_idx),
        (pl.col("subject") == subject),
        (pl.col("phoneme_pair") == phoneme_pair),
        (pl.col("word_end") == word_end),
    )
    subplot_phon_behav_df = data.plot_phon_behav_df.filter(
        (pl.col("electrode_idx") == electrode_idx),
        (pl.col("subject") == subject),
        (pl.col("phoneme_pair") == phoneme_pair),
        (pl.col("word_end") == word_end),
    )

    assert subplot_phon_phon_df.select(pl.n_unique("smin")).item() == 1
    assert (
        subplot_behav_phon_df.group_by("word_end")
        .agg(pl.n_unique("smin"))
        .select(pl.max("smin"))
        .item()
        == 1
    )
    assert (
        subplot_behav_behav_df.group_by("word_end")
        .agg(pl.n_unique("smin"))
        .select(pl.max("smin"))
        .item()
        == 1
    )
    assert subplot_phon_behav_df.select(pl.n_unique("smin")).item() == 1

    ###

    epochs_i = data.epochs[subject]
    if epoch_data is None:
        epoch_data = epochs_i.copy().apply_baseline().get_data()
    epoch_data = epoch_data[:, electrode_idx, :]

    plot_tmin = 0
    plot_tmax = (
        data.word_end_df.filter(
            pl.col("phoneme_pair") == phoneme_pair, (pl.col("word_end") == word_end)
        )
        .select(pl.max("word_end_offset"))
        .item()
        + 0.1
    )
    plot_smin = int((plot_tmin - epoch_tmin) * epoch_sfreq)
    plot_smax = int((plot_tmax - epoch_tmin) * epoch_sfreq)

    plot_highlight_phon_window = (
        subplot_phon_phon_df.select(["smin", "smax"]).unique().to_numpy().flatten()
    )
    plot_highlight_behav_window = (
        subplot_behav_behav_df.select(["smin", "smax"]).unique().to_numpy().flatten()
    )

    highlight_phon_times = epochs_i.times[
        [plot_highlight_phon_window[0], plot_highlight_phon_window[1]]
    ]
    highlight_behav_times = epochs_i.times[
        [plot_highlight_behav_window[0], plot_highlight_behav_window[1]]
    ]

    ### HGA plot by stimulus step

    plot_phon_smin = plot_smin
    plot_phon_smax = plot_highlight_behav_window[1] + 10

    def plot_phon_controlled(plot_phon_keys, ax, color="#9E9E9E"):
        linestyles = {0: "solid", 1: "dashed"}

        all_epoch_data = {}

        i = 0
        plot_phon_keys = plot_phon_keys.to_pandas()
        # exclude cases of extreme stimuli with non-acoustic-following behavior
        plot_phon_keys = plot_phon_keys[
            ~(
                (plot_phon_keys.resampled == 1)
                & (plot_phon_keys.behavior_dummy_forced == 1)
            )
            & ~(
                (plot_phon_keys.resampled == 6)
                & (plot_phon_keys.behavior_dummy_forced == 0)
            )
        ]
        for (decoder_target, label_behavior), rows in plot_phon_keys.groupby(
            ["behavior_dummy_forced", "label_behavior_forced"]
        ):
            epoch_i = epoch_data[rows.epoch_idx, :]
            all_epoch_data[decoder_target] = epoch_i
            epoch_i = epoch_i[:, plot_phon_smin:plot_phon_smax]
            epoch_mean = epoch_i.mean(axis=0)
            epoch_sem = epoch_i.std(axis=0) / np.sqrt(epoch_i.shape[0])
            times = epochs_i.times[plot_phon_smin:plot_phon_smax]

            linewidth = 2
            ax.plot(
                times,
                epoch_mean,
                label=f"Chose /{label_behavior}/",
                color=color,
                ls=linestyles[decoder_target],
                linewidth=linewidth,
            )
            ax.fill_between(
                times,
                epoch_mean - epoch_sem,
                epoch_mean + epoch_sem,
                color=color,
                alpha=0.3,
            )

            i += 1

        return all_epoch_data

    def plot_windowed_ttest(
        ax,
        all_epoch_data,
        group1,
        group2,
        test_smin=plot_phon_smin,
        test_smax=plot_phon_smax,
        test_window_size=4,
        test_window_stride=4,
        color="black",
        alpha=0.5,
        bar_height_ratio=0.01,
        bar_y_ratio=0.95,
    ):
        # Windowed t-test
        from scipy.stats import ttest_ind

        # Get windowed means for each condition
        test_window_starts = np.arange(
            test_smin, test_smax - test_window_size + 1, test_window_stride
        )
        test_results = []
        for start in test_window_starts:
            group1_data = all_epoch_data.get(group1, np.empty((0, test_window_size)))[
                :, start : start + test_window_size
            ]
            group2_data = all_epoch_data.get(group2, np.empty((0, test_window_size)))[
                :, start : start + test_window_size
            ]
            if len(group1_data) > 0 and len(group2_data) > 0:
                # average over time within the window
                group1_data = group1_data.mean(axis=1)
                group2_data = group2_data.mean(axis=1)
                t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
            else:
                t_stat, p_value = np.nan, np.nan
            test_results.append((start, start + test_window_size, t_stat, p_value))
        test_results_df = pd.DataFrame(
            test_results, columns=["start_sample", "end_sample", "t_stat", "p_value"]
        )
        test_results_df["tmin"] = (
            test_results_df["start_sample"] / epoch_sfreq + epoch_tmin
        )
        test_results_df["tmax"] = (
            test_results_df["end_sample"] / epoch_sfreq + epoch_tmin
        )
        # print(test_results_df)

        ymin, ymax = ax.get_ylim()
        bar_height = (ymax - ymin) * bar_height_ratio
        bar_y = ymin + (ymax - ymin) * bar_y_ratio
        for row in test_results_df.itertuples():
            if row.p_value < 0.05:
                ax.barh(
                    y=bar_y,
                    width=row.tmax - row.tmin,
                    left=row.tmin,
                    height=bar_height,
                    color=color,
                    alpha=alpha,
                    edgecolor="none",
                )

    vline_extent = 1.15

    def add_textgrid(ax, textgrid_dir, ep_df, rotation=0, include_phonemes=True):
        textgrid_file = ep_df.textgrid_path.iloc[0]
        tg = textgrid.TextGrid.fromFile(Path(textgrid_dir) / textgrid_file)
        assert tg.getNames() == ["phonemes"]

        plot_intervals = [
            interval
            for interval in tg.tiers[0].intervals
            if interval.mark is not None and interval.mark.strip()
        ]
        for i, interval in enumerate(plot_intervals):
            if include_phonemes:
                ax.text(
                    interval.minTime + 0.01,
                    1.025,
                    interval.mark.strip(),
                    rotation=rotation,
                    ha="right",
                    va="bottom",
                    fontsize=10,
                    transform=transforms.blended_transform_factory(
                        ax.transData, ax.transAxes
                    ),
                )

            if i == len(plot_intervals) - 1:
                # also add end time
                ax.axvline(
                    interval.maxTime,
                    ymax=vline_extent if include_phonemes else 1,
                    linestyle="--",
                    alpha=0.5,
                    color="blue",
                    clip_on=False,
                )

    def add_trial_count_inset(ax, plot_keys_i):
        # Create inset barplot instead of legend
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        ax_inset = inset_axes(
            ax,
            width="25%",
            height="25%",
            loc="upper right",
            bbox_to_anchor=(0.05, 0.05, 1, 1),
            bbox_transform=ax.transAxes,
        )

        # Get trial counts for each behavior condition
        trial_counts = []
        labels = []
        colors_inset = []
        linestyles_inset = []

        plot_keys_i_pd = plot_keys_i.to_pandas()
        # Exclude extreme stimuli with non-acoustic-following behavior (same as in plot_phon_controlled)
        plot_keys_i_pd = plot_keys_i_pd[
            ~(
                (plot_keys_i_pd.resampled == 1)
                & (plot_keys_i_pd.behavior_dummy_forced == 1)
            )
            & ~(
                (plot_keys_i_pd.resampled == 6)
                & (plot_keys_i_pd.behavior_dummy_forced == 0)
            )
        ]

        for (decoder_target, label_behavior), rows in plot_keys_i_pd.groupby(
            ["behavior_dummy_forced", "label_behavior_forced"]
        ):
            trial_counts.append(len(rows))
            labels.append(f"/{label_behavior}/")
            colors_inset.append(palette[True])
            linestyles_inset.append("solid" if decoder_target == 0 else "dashed")

        # Plot bars
        x_pos = np.arange(len(trial_counts))
        bars = ax_inset.bar(
            x_pos,
            trial_counts,
            color=colors_inset,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add linestyle indicators on top of bars
        for j, (bar, ls) in enumerate(zip(bars, linestyles_inset)):
            if ls == "dashed":
                ax_inset.plot(
                    [bar.get_x(), bar.get_x() + bar.get_width()],
                    [bar.get_height(), bar.get_height()],
                    "k--",
                    linewidth=2,
                )

        ax_inset.set_xticks(x_pos)
        ax_inset.set_xticklabels(labels, fontsize=8)
        ax_inset.set_ylabel("# trials", fontsize=8)
        ax_inset.tick_params(axis="both", labelsize=7)
        ax_inset.spines["top"].set_visible(False)
        ax_inset.spines["right"].set_visible(False)

    if axs is None:
        f, axs = plt.subplots(
            1 + len(controlled_resampled_search_steps), 1, figsize=figsize, sharex=True
        )
        if title:
            f.suptitle(
                f"Subject {subject}, Electrode {electrode_idx}, {phoneme_pair}, {word_end}"
            )

    plot_epoch_keys = subplot_phon_phon_df.select(
        [
            "epoch_idx",
            "resampled",
            "textgrid_path",
            "behavior_dummy_forced",
            "label_behavior_forced",
        ]
    ).unique()

    palette = {
        False: "#9E9E9E",  # extreme values, 1 or 6
        True: "#D62728",  # controlled values
    }

    # First axis: plot extremes
    ax_e = axs[0]
    all_epoch_data_extreme = plot_phon_controlled(
        plot_epoch_keys.filter(pl.col("resampled").is_in([1, 6])),
        ax_e,
        color=palette[False],
    )
    plot_windowed_ttest(
        ax_e,
        all_epoch_data_extreme,
        group1=0,
        group2=1,
        color="gray",
        alpha=0.5,
        bar_height_ratio=0.04,
    )

    for i, controlled_resampled_step in enumerate(controlled_resampled_search_steps):
        ax_c = axs[i + 1]

        plot_keys_i = plot_epoch_keys.filter(
            pl.col("resampled") == controlled_resampled_step
        )
        all_epoch_data_controlled_i = plot_phon_controlled(
            plot_keys_i, ax_c, color=palette[True]
        )
        plot_windowed_ttest(
            ax_c,
            all_epoch_data_controlled_i,
            group1=0,
            group2=1,
            color="red",
            alpha=0.5,
            bar_height_ratio=0.04,
        )

        ax_c.set_title(f"Resampled step {controlled_resampled_step}")  # , pad=20)
        # ax_c.legend(title="Behavior", loc="upper right", bbox_to_anchor=(1.2, 1))

        add_trial_count_inset(ax_c, plot_keys_i)

    pod_time = (
        data.word_end_df.filter(pl.col("word_end") == word_end)
        .select(pl.max("pod"))
        .item()
    )
    for i, ax in enumerate(axs):
        ax.axvline(
            pod_time,
            linestyle="--",
            alpha=0.5,
            color="red",
            ymax=vline_extent if i == 0 else 1,
            clip_on=False,
        )
        ax.set_ylabel("HGA ($z$)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        add_textgrid(
            ax,
            textgrid_dir,
            ep_df=plot_epoch_keys.to_pandas(),
            include_phonemes=(i == 0),
        )

    axs[-1].set_xlabel("Time from word onset (s)")

    # f.subplots_adjust(hspace=0.3)


@dataclass
class WindowSearchResult:
    """Best window found by sliding t-test on raw HGA."""

    smin: int  # absolute sample index (epoch_tmin=-0.4 s, sfreq=100 Hz)
    smax: int
    tmin: float
    tmax: float
    t_stat: float
    p_value: float
    n_group1: int
    n_group2: int


def find_site_windows(
    data: PaperData,
    subject: str,
    electrode_idx: int,
    phoneme_pair: str,
    word_end: str,
    window_size: int = 15,
    window_stride: int = 1,
    search_smin: int | None = None,
    search_smax: int | None = None,
    behavior_resampled_steps: tuple[tuple[int, ...], ...] = (
        (3,),
        (4,),
        (3, 4),
        (2, 3, 4, 5),
    ),
) -> dict:
    """
    For a single site, find the time window of maximal HGA separability for two contrasts:

      "phon":  resampled=1 vs resampled=6 (acoustic-consistent trials only —
               same exclusion as zoomin_hga: drops resampled=1 trials where
               behavior followed the non-acoustic side, and vice versa for =6)

      "behav": behavior=0 vs behavior=1, searched independently for each entry
               in behavior_resampled_steps.  The latent design choice is which
               resampled steps to pool: ambiguous-only (3), (4), (3,4), or a
               wider range (2,3,4,5).  All variants are returned so you can
               compare.

    Best window is chosen by max |t-stat| (Welch's t-test on per-trial window
    means) over the search range.

    Returns:
        {
            "phon":  WindowSearchResult,
            "behav": {(3,): WindowSearchResult, (4,): WindowSearchResult, ...}
        }
    """
    site_df = data.plot_phon_phon_df.filter(
        pl.col("subject") == subject,
        pl.col("electrode_idx") == electrode_idx,
        pl.col("phoneme_pair") == phoneme_pair,
        pl.col("word_end") == word_end,
    ).to_pandas()

    epochs_i = data.epochs[subject]
    epoch_data = (
        epochs_i.copy().apply_baseline().get_data(picks=electrode_idx).squeeze(1)
    )

    # default search range: word onset → word offset + 200 ms
    if search_smin is None:
        search_smin = int((0 - epoch_tmin) * epoch_sfreq)  # = 40
    if search_smax is None:
        word_offset_sample = (
            data.word_end_df.filter(
                pl.col("phoneme_pair") == phoneme_pair, pl.col("word_end") == word_end
            )
            .select("word_end_offset_sample")
            .item()
        )
        search_smax = int(word_offset_sample) + 20

    def _sliding_best(g1_idx, g2_idx):
        g1 = epoch_data[g1_idx, :]
        g2 = epoch_data[g2_idx, :]
        best_t, best_p, best_start = 0.0, 1.0, search_smin
        for start in range(search_smin, search_smax - window_size + 1, window_stride):
            g1_mean = g1[:, start : start + window_size].mean(axis=1)
            g2_mean = g2[:, start : start + window_size].mean(axis=1)
            t, p = ttest_ind(g1_mean, g2_mean, equal_var=False)
            if not np.isnan(t) and abs(t) > abs(best_t):
                best_t, best_p, best_start = t, p, start
        return WindowSearchResult(
            smin=best_start,
            smax=best_start + window_size,
            tmin=best_start / epoch_sfreq + epoch_tmin,
            tmax=(best_start + window_size) / epoch_sfreq + epoch_tmin,
            t_stat=best_t,
            p_value=best_p,
            n_group1=len(g1_idx),
            n_group2=len(g2_idx),
        )

    # --- phoneme window: resampled=1 vs 6, acoustic-consistent only
    phon_df = site_df[site_df.resampled.isin([1, 6])].copy()
    phon_df = phon_df[
        ~((phon_df.resampled == 1) & (phon_df.behavior_dummy_forced == 1))
        & ~((phon_df.resampled == 6) & (phon_df.behavior_dummy_forced == 0))
    ]
    g1 = (
        phon_df[phon_df.decoder_target == 0]
        .drop_duplicates("epoch_idx")
        .epoch_idx.values
    )
    g2 = (
        phon_df[phon_df.decoder_target == 1]
        .drop_duplicates("epoch_idx")
        .epoch_idx.values
    )
    phon_result = _sliding_best(g1, g2)

    # --- behavior windows: behavior=0 vs 1, varying which resampled steps to pool
    behav_results = {}
    for steps in behavior_resampled_steps:
        key = tuple(steps)
        sub = site_df[site_df.resampled.isin(key)]
        g1 = (
            sub[sub.behavior_dummy_forced == 0]
            .drop_duplicates("epoch_idx")
            .epoch_idx.values
        )
        g2 = (
            sub[sub.behavior_dummy_forced == 1]
            .drop_duplicates("epoch_idx")
            .epoch_idx.values
        )
        if len(g1) == 0 or len(g2) == 0:
            L.warning(
                f"find_site_windows: empty behav group steps={key} at "
                f"{subject}/{electrode_idx}/{phoneme_pair}/{word_end}"
            )
            behav_results[key] = None
        else:
            behav_results[key] = _sliding_best(g1, g2)

    return {"phon": phon_result, "behav": behav_results}


def _steps_tag(steps: tuple[int, ...]) -> str:
    """Short column-name tag for a set of resampled steps, e.g. (3,4) → 's34'."""
    return "s" + "".join(str(s) for s in steps)


def extract_hga_windows_df(
    data: PaperData,
    zoomin_keys: pl.DataFrame,
    window_size: int = 15,
    window_stride: int = 15,
    ambiguous_response_threshold: int = 2,
    window_source: Literal["ttest_searchlight", "decoder"] = "ttest_searchlight",
) -> pd.DataFrame:
    """
    For each site in zoomin_keys, find optimal early (phoneme) and late (behavior)
    HGA windows, then extract per-trial mean HGA in each window.

    Returns a wide DataFrame (one row per site × trial) with:

      Site:   subject, electrode_idx, phoneme_pair, word_end
      Trial:  epoch_idx

      Predictors for lmer:
        resampled               – acoustic continuum step 1–6
        decoder_target          – acoustic category 0/1
        behavior_dummy_forced   – behavioral response 0/1
        follows_acoustics       – bool
        mismatch                – lexical context (-1/0/1)

      HGA outcomes:
        hga_early               – mean HGA in phoneme-separability window
        hga_late                – mean HGA in behavior window

      Window choice metadata (per site):
        behav_steps_chosen      – ambiguous resampled steps for this site, e.g. '(3, 4)'
        phon_tmin, phon_tmax, phon_smin, phon_smax
        behav_tmin, behav_tmax, behav_smin, behav_smax

    Parameters
    ----------
    window_source : {"ttest_searchlight", "decoder"}
        How windows are selected per site:
        - "ttest_searchlight" (default): run find_site_windows() — a sliding Welch's
          t-test searchlight — independently for each site. This is the original behaviour.
        - "decoder": look up windows from data.phon_peaks_df (acoustic window) and
          data.behav_peaks_df (behavioral window), aligning selectivity/congruency
          with the decoder-located peaks used by the transfer analysis.

    To feed into lmer, pivot to long format:
        df_long = df.melt(
            id_vars=[...metadata...],
            value_vars=["hga_early", "hga_late"],
            var_name="window", value_name="hga"
        )
    Then: HGA ~ window * decoder_target + window * behavior_dummy_forced + (1 | subject)
    """
    # Cache baseline-corrected epoch data per subject across sites
    epoch_cache: dict[tuple, np.ndarray] = {}

    sites = zoomin_keys.unique(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    rows = []

    # Ambiguous resampled steps (steps where behavior varies) — used for behav_steps_chosen
    # and, in ttest_searchlight mode, to drive the behavioral window search.
    ambiguous_resampled_steps = data.get_ambiguous_resampled_steps(
        ambiguous_response_threshold=ambiguous_response_threshold
    )

    for site_row in tqdm(
        sites.iter_rows(named=True), total=sites.height, desc="Extracting HGA windows"
    ):
        subject = site_row["subject"]
        electrode_idx = site_row["electrode_idx"]
        phoneme_pair = site_row["phoneme_pair"]
        word_end = site_row["word_end"]

        cache_key = subject
        if cache_key not in epoch_cache:
            epoch_cache[cache_key] = (
                data.epochs[subject].copy().apply_baseline().get_data()
            )
        epoch_data = epoch_cache[cache_key][:, electrode_idx, :]

        ambig_steps = ambiguous_resampled_steps.get(
            (subject, phoneme_pair, word_end), None
        )

        if window_source == "ttest_searchlight":
            # Original behaviour: run a sliding Welch's t-test searchlight per site.
            behavior_resampled_steps_i = [ambig_steps if ambig_steps else ()]
            windows = find_site_windows(
                data,
                subject,
                electrode_idx,
                phoneme_pair,
                word_end,
                window_size=window_size,
                window_stride=window_stride,
                behavior_resampled_steps=behavior_resampled_steps_i,
            )
            phon_win = windows["phon"]
            valid_behav = {k: v for k, v in windows["behav"].items() if v is not None}
            if valid_behav:
                best_steps = max(valid_behav, key=lambda k: abs(valid_behav[k].t_stat))
                best_bwin = valid_behav[best_steps]
            else:
                best_steps, best_bwin = None, None

            phon_smin_i = phon_win.smin
            phon_smax_i = phon_win.smax
            behav_smin_i = best_bwin.smin if best_bwin else None
            behav_smax_i = best_bwin.smax if best_bwin else None

        else:  # "decoder"
            # Use decoder-located windows from PaperData, matching the transfer analysis.
            phon_row = data.phon_peaks_df.filter(
                (pl.col("subject") == subject)
                & (pl.col("electrode_idx") == electrode_idx)
                & (pl.col("phoneme_pair") == phoneme_pair)
            )
            behav_row = data.behav_peaks_df.filter(
                (pl.col("subject") == subject)
                & (pl.col("electrode_idx") == electrode_idx)
                & (pl.col("phoneme_pair") == phoneme_pair)
                & (pl.col("word_end") == word_end)
            )
            phon_smin_i = int(phon_row["smin"][0])
            phon_smax_i = int(phon_row["smax"][0])
            behav_smin_i = (
                int(behav_row["smin"][0]) if not behav_row.is_empty() else None
            )
            behav_smax_i = (
                int(behav_row["smax"][0]) if not behav_row.is_empty() else None
            )

        # Window timing metadata (same for every trial at this site)
        ep = data.epochs[subject]
        _t0, _sf = ep.tmin, ep.info["sfreq"]
        timing = {
            "phon_tmin": _t0 + phon_smin_i / _sf,
            "phon_tmax": _t0 + phon_smax_i / _sf,
            "phon_smin": phon_smin_i,
            "phon_smax": phon_smax_i,
            "behav_tmin": _t0 + behav_smin_i / _sf
            if behav_smin_i is not None
            else np.nan,
            "behav_tmax": _t0 + behav_smax_i / _sf
            if behav_smin_i is not None
            else np.nan,
            "behav_smin": behav_smin_i if behav_smin_i is not None else np.nan,
            "behav_smax": behav_smax_i if behav_smin_i is not None else np.nan,
            "behav_steps_chosen": str(ambig_steps),
        }

        # One row per trial (deduplicated across CV folds)
        site_df = (
            data.plot_phon_phon_df.filter(
                pl.col("subject") == subject,
                pl.col("electrode_idx") == electrode_idx,
                pl.col("phoneme_pair") == phoneme_pair,
                pl.col("word_end") == word_end,
            )
            .unique(subset=["epoch_idx"], keep="first")
            .to_pandas()
        )

        for _, trial in site_df.iterrows():
            epoch_idx = int(trial.epoch_idx)
            trace = epoch_data[epoch_idx, :]

            rows.append(
                {
                    "subject": subject,
                    "electrode_idx": electrode_idx,
                    "phoneme_pair": phoneme_pair,
                    "word_end": word_end,
                    "epoch_idx": epoch_idx,
                    "resampled": trial.resampled,
                    "decoder_target": trial.decoder_target,
                    "behavior_dummy_forced": trial.behavior_dummy_forced,
                    "follows_acoustics": trial.follows_acoustics,
                    "mismatch": trial.mismatch,
                    "hga_early": trace[phon_smin_i:phon_smax_i].mean(),
                    "hga_late": (
                        trace[behav_smin_i:behav_smax_i].mean()
                        if behav_smin_i is not None
                        else np.nan
                    ),
                    **timing,
                }
            )

    return pd.DataFrame(rows)


def pl_roc_auc(
    df: pl.DataFrame,
    target_col: str,
    proba_col: str,
    group_cols: list[str],
    roc_auc_name="roc_auc",
) -> pl.DataFrame:
    assert set(df.select(pl.col(target_col).unique()).to_series()) <= {0, 1}, (
        "Target column must be binary (0/1) for ROC AUC calculation."
    )

    return (
        df.with_columns(
            pl.col(proba_col).rank(method="average").over(group_cols).alias("rank")
        )
        .group_by(group_cols)
        .agg(
            n_pos=pl.col(target_col).sum(),
            n=pl.len(),
            rank_sum_pos=pl.col("rank").filter(pl.col(target_col) == 1).sum(),
        )
        .with_columns(n_neg=(pl.col("n") - pl.col("n_pos")))
        .with_columns(
            **{
                roc_auc_name: pl.when((pl.col("n_pos") > 0) & (pl.col("n_neg") > 0))
                .then(
                    (
                        pl.col("rank_sum_pos")
                        - pl.col("n_pos") * (pl.col("n_pos") + 1) / 2
                    )
                    / (pl.col("n_pos") * pl.col("n_neg"))
                )
                .otherwise(None)
            }
        )
        .select(group_cols + [roc_auc_name])
    )


class HandlerRectangle(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        rect_height = orig_handle.get_height()
        center = 0.5 * height
        p = Rectangle(
            xy=(-xdescent, center - rect_height * height / 2),
            width=width,
            height=rect_height * height,
            facecolor=orig_handle.get_facecolor(),
            alpha=orig_handle.get_alpha(),
        )
        return [p]


def get_condition_contrast(
    plot_df,
    condition_variable,
    data: PaperData,
    epoch_data_cache=None,
) -> pl.DataFrame:
    hga_condition_results = []
    if epoch_data_cache is None:
        epoch_data_cache = {}

    for (subject, electrode_idx, phoneme_pair, word_end), rows in plot_df.group_by(
        ["subject", "electrode_idx", "phoneme_pair", "word_end"]
    ):
        epochs_i = data.epochs[subject]
        if subject not in epoch_data_cache:
            epoch_data_cache[subject] = epochs_i.copy().apply_baseline().get_data()
        epoch_data = epoch_data_cache[subject][:, electrode_idx, :]

        md0 = rows.filter(pl.col(condition_variable) <= 0)
        md1 = rows.filter(pl.col(condition_variable) > 0)

        idxs0 = md0.select(pl.col("epoch_idx")).to_series().unique()
        idxs1 = md1.select(pl.col("epoch_idx")).to_series().unique()

        if len(idxs0) == 0 or len(idxs1) == 0:
            continue

        epochs_0 = epoch_data[idxs0]
        epochs_1 = epoch_data[idxs1]

        diff_of_means = epochs_1.mean(axis=0) - epochs_0.mean(axis=0)
        hga_condition_results.append(
            {
                "subject": subject,
                "electrode_idx": electrode_idx,
                "phoneme_pair": phoneme_pair,
                "word_end": word_end,
                "times": epochs_i.times,
                "diff_of_means": diff_of_means,
            }
        )

    return (
        pl.DataFrame(hga_condition_results)
        .join(
            pl.from_pandas(data.early_polarity.reset_index()),
            on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
            how="left",
        )
        .join(
            pl.from_pandas(data.late_polarity.reset_index()),
            on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
            how="left",
        )
    )


def plot_condition_contrast(
    plot_df,
    condition_variable,
    data: PaperData,
    textgrid_dir,
    polarity_correct: Literal[None, "early", "late"] = None,
    epoch_data_cache=None,
    ax=None,
    annotate=True,
    label=None,
    textgrid_kwargs=None,
    pval_thresholds=(0.00001, 0.0001, 0.001),
    vline_extent=1.25,
    ttest_window_size=15,
    ttest_window_stride=15,
    ttest_bar_height_ratio=0.04,
    ttest_bar_y_ratio=0.95,
    color=None,
):
    hga_condition_results_df = get_condition_contrast(
        plot_df, condition_variable, data, epoch_data_cache=epoch_data_cache
    )

    if ax is None:
        f, ax = plt.subplots(figsize=(5, 3))

    plot_rows = hga_condition_results_df
    plot_times = plot_rows.select(pl.col("times"))[0].item()
    plot_diffs = np.stack(plot_rows.select(pl.col("diff_of_means")).to_numpy()[:, 0])

    if polarity_correct is not None:
        plot_signs = (
            plot_rows.select(pl.col(f"{polarity_correct}_polarity"))
            .to_numpy()
            .flatten()
        )
        plot_diffs *= plot_signs[:, np.newaxis]

    plot_diffs = plot_diffs[~np.isnan(plot_diffs).any(axis=1)]

    plot_diff_mean = plot_diffs.mean(axis=0)
    plot_diff_sem = plot_diffs.std(axis=0) / np.sqrt(plot_diffs.shape[0])

    ax.plot(plot_times, plot_diff_mean, label=label, linewidth=2, color=color)
    ax.fill_between(
        plot_times,
        plot_diff_mean - plot_diff_sem,
        plot_diff_mean + plot_diff_sem,
        alpha=0.3,
        color=color,
    )

    n_times = plot_diffs.shape[1]
    ttest_results = []
    for start in range(0, n_times - ttest_window_size + 1, ttest_window_stride):
        window_means = plot_diffs[:, start : start + ttest_window_size].mean(axis=1)
        t_stat, p_value = ttest_1samp(window_means, 0)
        end = min(start + ttest_window_size, n_times - 1)
        ttest_results.append((plot_times[start], plot_times[end], t_stat, p_value))

    p_threshold_height_mults = [1.0, 0.5, 0.25]
    # p_thresholds = list(zip([0.0001, 0.001, 0.01], p_threshold_height_mults))
    p_thresholds = list(zip(pval_thresholds, p_threshold_height_mults))
    # p_thresholds = list(zip([0.001, 0.01, 0.05], p_threshold_height_mults))

    ymin, ymax = ax.get_ylim()
    base_bar_height = (ymax - ymin) * ttest_bar_height_ratio
    bar_y = ymin + (ymax - ymin) * ttest_bar_y_ratio

    for tmin_w, tmax_w, t_stat, p_value in ttest_results:
        height_mult = None
        for p_thresh, mult in p_thresholds:
            if p_value < p_thresh:
                height_mult = mult
                break

        if height_mult is not None:
            color = ax.lines[-1].get_color()
            ax.barh(
                y=bar_y,
                width=tmax_w - tmin_w,
                left=tmin_w,
                height=base_bar_height * height_mult,
                color=color,
                alpha=0.5,
                edgecolor="none",
            )

    if annotate:
        textgrid_default_kwargs = dict(include_offset=True, vline_extent=vline_extent)
        textgrid_kwargs = {
            **textgrid_default_kwargs,
            **(textgrid_kwargs if textgrid_kwargs is not None else {}),
        }
        add_textgrid(ax, textgrid_dir, plot_df.to_pandas(), **textgrid_kwargs)

        p_handles = [
            Rectangle(
                (0, 0),
                1,
                height_mult,
                facecolor="gray",
                alpha=0.5,
                label=f"p < {p_thresh:g}".replace("-0", "-"),
            )
            for p_thresh, height_mult in p_thresholds
        ]
        p_labels = [h.get_label() for h in p_handles]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles + p_handles,
            labels=labels + p_labels,
            handler_map={Rectangle: HandlerRectangle()},
            loc="best",
            fontsize=8,
        )

        word_end = plot_rows["word_end"][0]
        pod_time = (
            data.word_end_df.filter(pl.col("word_end") == word_end)
            .select(pl.max("pod"))
            .item()
        )
        ax.axvline(
            pod_time,
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            color="red",
            ymax=vline_extent,
            clip_on=False,
        )

        ax.set_ylim(-0.1, ax.get_ylim()[1])
        ax.axhline(0, linestyle="--", color="gray", alpha=0.7)
        sns.despine(ax=ax, top=True, right=True)

        return ax, p_handles, p_labels

    return ax, None, None


def plot_condition_contrasts_single_figure(
    data: PaperData,
    textgrid_dir,
    epoch_data_cache=None,
    textgrid_kwargs=None,
    vline_extent=1.2,
    plot_word_ends: list[str] = ("necessary",),
    plot_xlim=(0, 1.2),
    plot_ylim=None,
    pval_thresholds=(0.00001, 0.0001, 0.001),
    ambiguous_response_threshold: int = 2,
):
    fb = FigureBuilder(figsize=(2.5, 2))
    ax = fb.ax

    plot_palette = sns.color_palette("Set2", 2)

    # Skeleton: axis labels/limits in place, no data lines yet
    ax.set_xlim(*plot_xlim)
    ax.set_ylabel("HGA effect size ($z$)")
    ax.set_xlabel("Time from word onset (s)")

    # Make dummy handles for the two lines and
    # pval threshold bars
    dummy_handles = [Line2D([0], [0], color=color, lw=2) for color in plot_palette]
    dummy_labels = ["Acoustic\ncontrast", "Perceptual\ncontrast"]

    for pval_threshold, height_mult in zip(pval_thresholds, [1.0, 0.5, 0.25]):
        dummy_handles.append(
            Rectangle(
                (0, 0),
                1,
                height_mult,
                facecolor="gray",
                alpha=0.5,
                label=f"p < {pval_threshold:g}".replace("-0", "-"),
            )
        )
        dummy_labels.append(f"p < {pval_threshold:g}".replace("-0", "-"))
    legend_bbox_to_anchor = (1.65, 1)

    ax.legend(
        handles=dummy_handles,
        labels=dummy_labels,
        handler_map={Rectangle: HandlerRectangle()},
        fontsize=10,
        loc="upper right",
        bbox_to_anchor=legend_bbox_to_anchor,
    )
    sns.despine(ax=ax, top=True, right=True)
    if plot_ylim is not None:
        ax.set_ylim(*plot_ylim)

    fb.stage("skeleton")

    _, p_handles, p_labels = plot_condition_contrast(
        (
            data.plot_phon_phon_df.filter(
                pl.col("resampled").is_in([1, 6]),
                pl.col("word_end").is_in(plot_word_ends),
            )
        ),
        "categorical_acoustic_cue",
        data=data,
        textgrid_dir=textgrid_dir,
        polarity_correct="early",
        epoch_data_cache=epoch_data_cache,
        ax=ax,
        color=plot_palette[0],
        annotate=True,
        vline_extent=vline_extent,
        pval_thresholds=pval_thresholds,
        textgrid_kwargs={
            "fontsize": 8,
            "vline_extent": vline_extent,
            **(textgrid_kwargs or {}),
        },
        label="Acoustic\ncontrast",
    )

    ax.legend(
        handles=dummy_handles,
        labels=dummy_labels,
        handler_map={Rectangle: HandlerRectangle()},
        fontsize=10,
        loc="upper right",
        bbox_to_anchor=legend_bbox_to_anchor,
    )

    fb.stage("acoustic")

    ambiguous_keys = pl.DataFrame(
        [
            (subject, phoneme_pair, word_end, resampled)
            for (
                subject,
                phoneme_pair,
                word_end,
            ), resampled_list in data.get_ambiguous_resampled_steps(
                ambiguous_response_threshold=ambiguous_response_threshold
            ).items()
            for resampled in resampled_list
        ],
        schema=pl.Schema(
            {
                "subject": subject_enum,
                "phoneme_pair": phoneme_pair_enum,
                "word_end": word_end_enum,
                "resampled": pl.Float32,
            }
        ),
    )
    plot_behav_rows = ambiguous_keys.join(
        data.plot_phon_phon_df,
        on=["subject", "phoneme_pair", "word_end", "resampled"],
        how="inner",
    ).filter(pl.col("word_end").is_in(plot_word_ends))
    plot_condition_contrast(
        plot_behav_rows,
        "behavior_dummy_forced",
        data=data,
        textgrid_dir=textgrid_dir,
        polarity_correct="late",
        pval_thresholds=pval_thresholds,
        epoch_data_cache=epoch_data_cache,
        ax=ax,
        color=plot_palette[1],
        annotate=False,
        label="Perceptual\ncontrast",
        vline_extent=vline_extent,
        textgrid_kwargs={
            "fontsize": 8,
            "vline_extent": vline_extent,
            "include_phonemes": False,
            **(textgrid_kwargs or {}),
        },
        ttest_bar_y_ratio=0.87,
    )

    ax.legend(
        handles=dummy_handles,
        labels=dummy_labels,
        handler_map={Rectangle: HandlerRectangle()},
        fontsize=10,
        loc="upper right",
        bbox_to_anchor=legend_bbox_to_anchor,
    )

    fb.stage("behavioral")

    return fb


def plot_condition_contrast_peak_aligned(
    plot_df,
    behav_peaks_df,
    data: PaperData,
    condition_variable,
    polarity_correct: Literal[None, "early", "late"] = None,
    epoch_data_cache=None,
    ax=None,
    label=None,
    window_sec=0.3,
    ttest_window_size=4,
    ttest_window_stride=4,
    ttest_bar_height_ratio=0.04,
    ttest_bar_y_ratio=0.95,
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
    peak_centers = {}
    for row in behav_peaks_df.iter_rows(named=True):
        key = (
            row["subject"],
            row["electrode_idx"],
            row["phoneme_pair"],
            row["word_end"],
        )
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

        epochs_i = data.epochs[subject]
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

        s_start = peak_sample - window_samples
        s_end = peak_sample + window_samples
        n_total = diff_of_means.shape[0]
        if s_start < 0 or s_end > n_total:
            continue

        diff_aligned = diff_of_means[s_start:s_end]

        hga_condition_results.append(
            {
                "subject": subject,
                "electrode_idx": electrode_idx,
                "phoneme_pair": phoneme_pair,
                "word_end": word_end,
                "diff_aligned": diff_aligned,
            }
        )

    hga_condition_results_df = (
        pl.DataFrame(hga_condition_results)
        .join(
            pl.from_pandas(data.early_polarity.reset_index()),
            on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
            how="left",
        )
        .join(
            pl.from_pandas(data.late_polarity.reset_index()),
            on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
            how="left",
        )
    )

    if ax is None:
        f, ax = plt.subplots(figsize=(5, 3))

    plot_rows = hga_condition_results_df.filter(pl.col("phoneme_pair") == "dn")
    plot_diffs = np.stack(plot_rows.select(pl.col("diff_aligned")).to_numpy()[:, 0])

    if polarity_correct is not None:
        plot_signs = (
            plot_rows.select(pl.col(f"{polarity_correct}_polarity"))
            .to_numpy()
            .flatten()
        )
        plot_diffs *= plot_signs[:, np.newaxis]

    plot_diffs = plot_diffs[~np.isnan(plot_diffs).any(axis=1)]

    rel_times = np.linspace(-window_sec, window_sec, plot_diffs.shape[1])

    plot_diff_mean = plot_diffs.mean(axis=0)
    plot_diff_sem = plot_diffs.std(axis=0) / np.sqrt(plot_diffs.shape[0])

    ax.plot(rel_times, plot_diff_mean, label=label, linewidth=2)
    ax.fill_between(
        rel_times,
        plot_diff_mean - plot_diff_sem,
        plot_diff_mean + plot_diff_sem,
        alpha=0.3,
    )

    n_times = plot_diffs.shape[1]
    ttest_results = []
    for start in range(0, n_times - ttest_window_size + 1, ttest_window_stride):
        window_means = plot_diffs[:, start : start + ttest_window_size].mean(axis=1)
        t_stat, p_value = ttest_1samp(window_means, 0)
        end = min(start + ttest_window_size, n_times - 1)
        ttest_results.append((rel_times[start], rel_times[end], t_stat, p_value))

    ymin, ymax = ax.get_ylim()
    bar_height = (ymax - ymin) * ttest_bar_height_ratio
    bar_y = ymin + (ymax - ymin) * ttest_bar_y_ratio
    for tmin_w, tmax_w, t_stat, p_value in ttest_results:
        if p_value < 0.05:
            color = ax.lines[-1].get_color()
            ax.barh(
                y=bar_y,
                width=tmax_w - tmin_w,
                left=tmin_w,
                height=bar_height,
                color=color,
                alpha=0.5,
                edgecolor="none",
            )

    ax.axvline(0, linestyle="--", linewidth=2, alpha=0.5, color="red")
    ax.axhline(0, linestyle="--", color="gray", alpha=0.7)
    ax.set_xlabel("Time from behavioral peak (s)")
    ax.set_ylabel("HGA effect size ($z$)")
    ax.set_ylim(-0.1, ax.get_ylim()[1])
    sns.despine(ax=ax, top=True, right=True)

    return ax


def show_behav_stackplot(all_md: pl.DataFrame, subject, phoneme_pair, resampled):
    md_i = all_md.to_pandas().query(
        "subject == @subject and phoneme_pair == @phoneme_pair and resampled == @resampled"
    )
    behav_counts = md_i.groupby("label_behavior").size()
    total = behav_counts.sum()

    all_behaviors = md_i["label_behavior"].unique()
    colors = plt.cm.Set3(range(len(all_behaviors)))
    color_map = {behavior: colors[i] for i, behavior in enumerate(all_behaviors)}

    bottom = 0
    f, axs = plt.subplots(1, 2, figsize=(5, 1.3))

    for i, (behavior, count) in enumerate(behav_counts.items()):
        axs[0].barh(0, count, left=bottom, label=behavior, color=color_map[behavior])

        center = bottom + count / 2
        percentage = (count / total) * 100
        axs[0].text(
            center,
            0,
            f"Heard /{behavior}/\n({percentage:.0f}%)",
            ha="center",
            va="center",
            fontsize=10,
        )

        bottom += count

    axs[0].set_yticks([])
    axs[0].set_xticklabels([])
    axs[0].set_xlim(0, bottom)
    axs[0].set_xticks([])

    grouped = (
        md_i.groupby(["word_end", "label_behavior"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    n_lexical = len(grouped)
    for idx, (word_end, behav_counts) in enumerate(grouped.iterrows()):
        bottom = 0
        total = behav_counts.sum()

        for behavior, count in behav_counts.items():
            if count > 0:
                axs[1].barh(idx, count, left=bottom, color=color_map[behavior])

                center = bottom + count / 2
                percentage = (count / total) * 100
                label = f"$\\it{{{behavior}{word_end[1:]}}}$"
                axs[1].text(
                    center,
                    idx,
                    f"{label}\n({percentage:.0f}%)",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

                bottom += count

    axs[1].set_yticks(range(n_lexical))
    axs[1].set_yticklabels(["-" + label[1:] for label in grouped.index])
    axs[1].set_xticklabels([])
    axs[1].set_xlim(0, bottom)
    axs[1].set_xticks([])

    sns.despine(ax=axs[0], left=True, bottom=True)
    sns.despine(ax=axs[1], left=True, bottom=True)

    f.tight_layout()
    return f


def show_behav_stackplot2(
    all_md: pl.DataFrame,
    subject,
    phoneme_pair,
    label_word_end=None,
    resampled_set=(1, 3, 6),
    filter_word_end=None,
    figsize=(2.5, 2.8),
):
    md_i = all_md.to_pandas().query(
        "subject == @subject and phoneme_pair == @phoneme_pair"
    )
    if filter_word_end is not None:
        md_i = md_i.query("word_end == @filter_word_end")

    all_behaviors = list(phoneme_pair)
    colors = plt.cm.Set3(range(len(all_behaviors)))
    color_map = {behavior: colors[i] for i, behavior in enumerate(all_behaviors)}

    f, ax = plt.subplots(1, 1, figsize=figsize)
    axs = [ax]

    grouped = (
        md_i[md_i.resampled.isin(list(resampled_set))]
        .astype({"resampled": int})
        .groupby(["resampled", "label_behavior"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    n_lexical = len(grouped)
    for idx, (resampled, behav_counts) in enumerate(grouped.iterrows()):
        bottom = 0
        total = behav_counts.sum()

        for behavior, count in behav_counts.items():
            if count > 0:
                y = idx
                axs[0].barh(y, count, left=bottom, color=color_map[behavior])

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
                axs[0].text(
                    center,
                    y,
                    f"{label}\n({percentage:.0f}%)",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

                bottom += count

    axs[0].set_yticks(range(n_lexical))
    axs[0].set_yticklabels(grouped.index)
    axs[0].set_xticklabels([])
    axs[0].set_xlim(0, bottom)
    axs[0].set_xticks([])
    sns.despine(ax=axs[0], left=True, bottom=True)

    f.tight_layout()
    return f


def show_behav_stackplot_lexical(
    all_md: pl.DataFrame, subject, phoneme_pair, resampled
):
    md_i = all_md.to_pandas().query(
        "subject == @subject and phoneme_pair == @phoneme_pair and resampled == @resampled"
    )

    grouped = (
        md_i.groupby(["word_end", "label_behavior"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    all_behaviors = md_i["label_behavior"].unique()
    colors = plt.cm.Set3(range(len(all_behaviors)))
    color_map = {behavior: colors[i] for i, behavior in enumerate(all_behaviors)}

    n_lexical = len(grouped)
    f, ax = plt.subplots(figsize=(10, 1 + 0.5 * n_lexical))

    for idx, (word_end, behav_counts) in enumerate(grouped.iterrows()):
        bottom = 0
        total = behav_counts.sum()

        for behavior, count in behav_counts.items():
            if count > 0:
                ax.barh(idx, count, left=bottom, color=color_map[behavior])

                center = bottom + count / 2
                percentage = (count / total) * 100
                ax.text(
                    center,
                    idx,
                    f"Heard /{behavior}/\n({percentage:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

                bottom += count

    ax.set_yticks(range(n_lexical))
    ax.set_yticklabels(["-" + label[1:] for label in grouped.index])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(0, bottom)

    f.tight_layout()
    return f


def plot_behav_barplot(
    all_md: pl.DataFrame,
    plot_subject,
    plot_phoneme_pair,
    plot_word_end,
    plot_resampled_steps,
    figsize=(2.3, 2.3),
    resampled_palette=resampled_cmap,
    legend=True,
    legend_bbox_to_anchor=(1.75, 0.45),
    plot_values: Literal["count", "proportion"] = "proportion",
    collapse_zero_bars: bool = False,
    ax=None,
):
    if isinstance(resampled_palette, dict):
        assert set(resampled_palette.keys()) == set(range(1, 7))
        resampled_palette = [resampled_palette[i] for i in range(1, 7)]

    fb = None
    if ax is None:
        fb = FigureBuilder(figsize=figsize)
        ax = fb.ax

    behav_barplot_data = (
        all_md.to_pandas()
        .query(
            f"subject == '{plot_subject}' and phoneme_pair == '{plot_phoneme_pair}' and word_end == '{plot_word_end}' and resampled in {plot_resampled_steps}"
        )[["resampled", "label_behavior"]]
        .value_counts()
    )
    max_num_trials = behav_barplot_data.groupby("resampled").sum().max()

    full_index = pd.MultiIndex.from_product(
        [plot_resampled_steps, list(plot_phoneme_pair)],
        names=["resampled", "label_behavior"],
    )
    behav_barplot_data = behav_barplot_data.reindex(
        full_index, fill_value=0
    ).reset_index(name="count")
    behav_barplot_data["resampled_inv"] = 7 - behav_barplot_data["resampled"]
    behav_barplot_data = behav_barplot_data.astype({"resampled": int}).sort_values(
        ["label_behavior", "resampled"], ascending=False
    )

    totals = behav_barplot_data.groupby("resampled")["count"].sum().to_dict()

    behav_barplot_data["prop"] = behav_barplot_data["count"] / behav_barplot_data[
        "resampled"
    ].map(totals)

    sns.barplot(
        data=behav_barplot_data,
        y="resampled",
        x="prop" if plot_values == "proportion" else "count",
        order=plot_resampled_steps[::-1],
        hue="label_behavior",
        hue_order=list(plot_phoneme_pair)[::-1],
        palette="viridis",
        orient="h",
        width=0.8,
        ax=ax,
    )

    behavior_styles = dict(zip(list(plot_phoneme_pair), ["", "//"]))
    linestyles = dict(zip(list(plot_phoneme_pair), ["solid", "dashed"]))

    max_width = behav_barplot_data["count"].max()

    if plot_values == "count":
        ax.set_xlim(0, max_width * 1.45)
        ax.set_xticks([0, max_num_trials // 2, max_num_trials])
        ax.set_xlabel("# trials")
    elif plot_values == "proportion":
        ax.set_xlim(0, 1.01)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xlabel("% trials")
    ax.set_ylabel("step")

    legend_handles = [
        Line2D(
            [0], [0],
            color="k",
            linewidth=1.5,
            linestyle="--",
            label=f"Chose\n$\\it{{{plot_phoneme_pair[1]}{plot_word_end[1:]}}}$",
        ),
        Line2D(
            [0], [0],
            color="k",
            linewidth=1.5,
            linestyle="-",
            label=f"Chose\n$\\it{{{plot_phoneme_pair[0]}{plot_word_end[1:]}}}$",
        ),
    ]

    if legend:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=10,
            handlelength=3,
            handleheight=3,
            loc="center right",
            bbox_to_anchor=legend_bbox_to_anchor,
        )
    else:
        ax.get_legend().remove()

    sns.despine(ax=ax, top=True, bottom=False, left=False, right=True)

    # Skeleton: bars invisible, all labels/legend in place
    if fb is not None:
        for patch in ax.patches:
            patch.set_alpha(0)
            patch.set_edgecolor("none")
        fb.stage("skeleton")

    # Data: color and style bars, add line overlays
    patch_rows = list(zip(ax.patches, behav_barplot_data.itertuples(index=False)))

    if collapse_zero_bars:
        groups = defaultdict(list)
        for patch, row in patch_rows:
            groups[row.resampled].append((patch, row))

        for resampled_step, group in groups.items():
            nonzero = [pr for pr in group if pr[1].count > 0]
            zero = [pr for pr in group if pr[1].count == 0]
            if zero and len(nonzero) == 1:
                surviving_patch, _ = nonzero[0]
                empty_patch, _ = zero[0]
                # merge the two dodge slots into one full-height bar
                full_y0 = min(surviving_patch.get_y(), empty_patch.get_y())
                full_height = surviving_patch.get_height() + empty_patch.get_height()
                surviving_patch.set_y(full_y0)
                surviving_patch.set_height(full_height)
                empty_patch.set_visible(False)

    for patch, row in patch_rows:
        if collapse_zero_bars and row.count == 0:
            continue  # already hidden above, and no line overlay needed

        patch_color = resampled_palette[row.resampled - 1]
        patch.set_facecolor(patch_color)
        patch.set_alpha(0.3)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
        patch.set_rasterized(True)

        x = patch.get_x()
        width = patch.get_width()
        y_center = patch.get_y() + patch.get_height() / 2
        linestyle = linestyles[row.label_behavior]
        ax.plot(
            [x, x + width],
            [y_center, y_center],
            color=patch_color,
            linestyle=linestyle,
            linewidth=1.5,
            solid_capstyle="butt",
            clip_on=True,
        )

    if fb is not None:
        fb.stage("data")

    return fb


def evaluate_phonetic_transfer(
    data: PaperData,
    phonetic_decoder_checkpoints: dict,
    t_subject,
    t_electrode_idx,
    t_phoneme_pair,
    t_word_end,
    t_smin_early,
    t_smax_early,
    t_smin_late,
    t_smax_late,
    t_num_folds=5,
    t_measure="categorical_acoustic_cue",
    t_restrict_to_word_end=True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t_early_key = (
        t_subject,
        t_electrode_idx,
        t_phoneme_pair,
        t_smin_early,
        t_smax_early,
    )
    t_late_key = (t_subject, t_electrode_idx, t_phoneme_pair, t_smin_late, t_smax_late)

    t_early_models = phonetic_decoder_checkpoints[t_subject]["models"][t_early_key]
    t_early_predictions = phonetic_decoder_checkpoints[t_subject]["outcomes"][
        t_early_key
    ]
    t_early_all_predictions = phonetic_decoder_checkpoints[t_subject]["all_outcomes"][
        t_early_key + (t_measure,)
    ]
    assert len(t_early_models) == t_num_folds

    t_late_models = phonetic_decoder_checkpoints[t_subject]["models"][t_late_key]
    t_late_predictions = phonetic_decoder_checkpoints[t_subject]["outcomes"][t_late_key]
    t_late_all_predictions = phonetic_decoder_checkpoints[t_subject]["all_outcomes"][
        t_late_key + (t_measure,)
    ]
    assert len(t_late_models) == t_num_folds

    t_epochs = data.epochs[t_subject]
    t_epoch_data = t_epochs.get_data(picks=t_electrode_idx).squeeze(1)

    early_early_outcomes, early_late_outcomes = [], []
    late_late_outcomes, late_early_outcomes = [], []

    def prepare_decoding_data(ep_i, data_i, epoch_idxs, smin, smax, baseline_vars=None):
        X = data_i[epoch_idxs][:, smin:smax]
        if baseline_vars is not None:
            X_baseline = ep_i.metadata.loc[epoch_idxs][baseline_vars].values
            X = np.concatenate([X_baseline, X], axis=1)
        return X

    def restrict_word_end(outcome_rows, word_end):
        outcome_rows = pd.merge(
            outcome_rows,
            t_epochs.metadata,
            left_on=["epoch_idx"],
            right_index=True,
            how="left",
        )
        return outcome_rows.loc[outcome_rows.word_end == word_end][
            ["epoch_idx", "fold", "decoder_target", "decoder_proba"]
        ]

    for fold, fold_rows in t_early_predictions.groupby("fold"):
        early_idxs = fold_rows.epoch_idx

        early_pipe = t_early_models[fold]
        late_pipe = t_late_models[fold]

        X_early = prepare_decoding_data(
            t_epochs, t_epoch_data, early_idxs, t_smin_early, t_smax_early
        )
        early_preds = early_pipe.predict_proba(X_early)[:, 1]
        np.testing.assert_allclose(early_preds, fold_rows.decoder_proba.values)

        fold_all_rows = t_early_all_predictions[t_early_all_predictions.fold == fold]
        if t_restrict_to_word_end:
            fold_all_rows = restrict_word_end(fold_all_rows, t_word_end)
        early_all_idxs = fold_all_rows.epoch_idx

        X_early_all = prepare_decoding_data(
            t_epochs, t_epoch_data, early_all_idxs, t_smin_early, t_smax_early
        )
        early_all_preds = early_pipe.predict_proba(X_early_all)[:, 1]

        X_late_all = prepare_decoding_data(
            t_epochs, t_epoch_data, early_all_idxs, t_smin_late, t_smax_late
        )
        late_scaler = late_pipe.named_steps["standardscaler"]
        X_late_all = late_scaler.transform(X_late_all)
        late_all_preds = early_pipe.named_steps["logisticregression"].predict_proba(
            X_late_all
        )[:, 1]

        early_early_outcomes.append(
            fold_all_rows.assign(
                subject=t_subject,
                electrode_idx=t_electrode_idx,
                phoneme_pair=t_phoneme_pair,
                word_end=t_word_end,
                smin=t_smin_early,
                smax=t_smax_early,
                decoder_target=(fold_all_rows.decoder_target > 0).astype(int),
                decoder_proba=early_all_preds,
            )
        )
        early_late_outcomes.append(
            fold_all_rows.assign(
                subject=t_subject,
                electrode_idx=t_electrode_idx,
                phoneme_pair=t_phoneme_pair,
                word_end=t_word_end,
                smin=t_smin_late,
                smax=t_smax_late,
                decoder_target=(fold_all_rows.decoder_target > 0).astype(int),
                decoder_proba=late_all_preds,
            )
        )

    for fold, fold_rows in t_late_predictions.groupby("fold"):
        late_idxs = fold_rows.epoch_idx

        early_pipe = t_early_models[fold]
        late_pipe = t_late_models[fold]

        X_late = prepare_decoding_data(
            t_epochs, t_epoch_data, late_idxs, t_smin_late, t_smax_late
        )
        late_preds = late_pipe.predict_proba(X_late)[:, 1]
        np.testing.assert_allclose(late_preds, fold_rows.decoder_proba.values)

        fold_all_rows = t_late_all_predictions[t_late_all_predictions.fold == fold]
        if t_restrict_to_word_end:
            fold_all_rows = restrict_word_end(fold_all_rows, t_word_end)
        late_all_idxs = fold_all_rows.epoch_idx

        X_late_all = prepare_decoding_data(
            t_epochs, t_epoch_data, late_all_idxs, t_smin_late, t_smax_late
        )
        late_all_preds = late_pipe.predict_proba(X_late_all)[:, 1]

        X_early_all = prepare_decoding_data(
            t_epochs, t_epoch_data, late_all_idxs, t_smin_early, t_smax_early
        )
        early_scaler = early_pipe.named_steps["standardscaler"]
        X_early_all = early_scaler.transform(X_early_all)
        early_all_preds = late_pipe.named_steps["logisticregression"].predict_proba(
            X_early_all
        )[:, 1]

        late_late_outcomes.append(
            fold_all_rows.assign(
                subject=t_subject,
                electrode_idx=t_electrode_idx,
                phoneme_pair=t_phoneme_pair,
                word_end=t_word_end,
                smin=t_smin_late,
                smax=t_smax_late,
                decoder_target=(fold_all_rows.decoder_target > 0).astype(int),
                decoder_proba=late_all_preds,
            )
        )
        late_early_outcomes.append(
            fold_all_rows.assign(
                subject=t_subject,
                electrode_idx=t_electrode_idx,
                phoneme_pair=t_phoneme_pair,
                word_end=t_word_end,
                smin=t_smin_early,
                smax=t_smax_early,
                decoder_target=(fold_all_rows.decoder_target > 0).astype(int),
                decoder_proba=early_all_preds,
            )
        )

    return (
        pd.concat(early_early_outcomes),
        pd.concat(early_late_outcomes),
        pd.concat(late_late_outcomes),
        pd.concat(late_early_outcomes),
    )


def evaluate_behav_decoder_on_phon_window(
    data: PaperData,
    behavioral_decoder_checkpoints: dict,
    phonetic_decoder_checkpoints: dict,
    t_subject,
    t_electrode_idx,
    t_phoneme_pair,
    t_word_end,
    t_smin_phon,
    t_smax_phon,
    t_smin_behav,
    t_smax_behav,
    t_num_folds=5,
) -> pd.DataFrame:
    """Apply a behavioral decoder (trained at the behavioral/perceptual window) to
    acoustic-window neural data, evaluated against the ACOUSTIC target.

    This asks: does the representation learned for behavioral prediction in the late
    window also encode acoustic content in the early window?

    Cross-scaler normalization (mirroring evaluate_phonetic_transfer):
      - Acoustic window data is normalized with the acoustic decoder's own StandardScaler
        (fit on acoustic-window training data), not the behavioral decoder's scaler.
      - After normalization, the behavioral decoder's PCA projects into its learned space.
      - The resampled baseline feature is zeroed out (contributes nothing to the LR).
      - Only the behavioral decoder's LogisticRegression is applied to the result.

    Checkpoints:
      behavioral_decoder_checkpoints: from behavior_decoding_single_electrode/{subject}/results.pt
      phonetic_decoder_checkpoints:   from behavior_decoding_single_electrode_acoustic/{subject}/
    """
    outer_key = (t_subject, t_electrode_idx, t_phoneme_pair)
    inner_dict = behavioral_decoder_checkpoints[t_subject]["A_decoders"][outer_key]
    phon_key = (t_subject, t_electrode_idx, t_phoneme_pair, t_smin_phon, t_smax_phon)
    phon_models = phonetic_decoder_checkpoints[t_subject]["models"][phon_key]

    t_epochs = data.epochs[t_subject]
    md = t_epochs.metadata
    t_epoch_data = t_epochs.get_data(picks=t_electrode_idx).squeeze(1)

    outcomes = []
    for fold in range(t_num_folds):
        inner_key = (
            t_subject,
            str(t_electrode_idx),
            t_phoneme_pair,
            (t_word_end,),
            t_smin_behav,
            t_smax_behav,
            fold,
        )
        if inner_key not in inner_dict:
            continue

        cp = inner_dict[inner_key]
        estimator = cp["estimator"]
        test_preds = cp["test_predictions"]
        epoch_idxs = test_preds["epoch_idx"].values

        # Decompose behavioral decoder pipeline components
        behav_est = estimator.best_estimator_
        behav_pca = (
            behav_est.named_steps["prep"].named_transformers_["pca"].named_steps["pca"]
        )
        behav_lr = behav_est.named_steps["clf"]

        # Cross-scaler: normalize acoustic window data with the acoustic decoder's own scaler
        # (fit on acoustic-window training data), so features are on the right scale.
        phon_scaler = phon_models[fold].named_steps["standardscaler"]
        X_phon_neural = t_epoch_data[epoch_idxs][:, t_smin_phon:t_smax_phon]
        X_phon_scaled = phon_scaler.transform(X_phon_neural)

        # Project through behavioral PCA, null the resampled column, apply behavioral LR
        X_phon_pca = behav_pca.transform(X_phon_scaled)
        X_for_lr = np.concatenate([np.zeros((len(epoch_idxs), 1)), X_phon_pca], axis=1)
        proba = behav_lr.predict_proba(X_for_lr)[:, 1]

        # Evaluate against acoustic target (not behavioral)
        acoustic_target = md.loc[epoch_idxs]["categorical_acoustic_cue"].values

        outcomes.append(
            pd.DataFrame(
                {
                    "epoch_idx": epoch_idxs,
                    "fold": fold,
                    "resampled": md.loc[epoch_idxs]["resampled"].values.astype(int),
                    "decoder_target": (acoustic_target > 0).astype(int),
                    "decoder_proba": proba,
                    "subject": t_subject,
                    "electrode_idx": t_electrode_idx,
                    "phoneme_pair": t_phoneme_pair,
                    "word_end": t_word_end,
                    "smin_phon": t_smin_phon,
                    "smax_phon": t_smax_phon,
                    "smin_behav": t_smin_behav,
                    "smax_behav": t_smax_behav,
                }
            )
        )

    return pd.concat(outcomes)


def evaluate_phon_decoder_on_behav_window(
    data: PaperData,
    phonetic_decoder_checkpoints: dict,
    behavioral_decoder_checkpoints: dict,
    t_subject,
    t_electrode_idx,
    t_phoneme_pair,
    t_word_end,
    t_smin_phon,
    t_smax_phon,
    t_smin_behav,
    t_smax_behav,
) -> pd.DataFrame:
    """Apply an acoustic decoder (trained at the acoustic/phonetic window) to
    behavioral-window neural data, evaluated against the BEHAVIORAL target.

    This asks: does the representation learned for acoustic decoding in the early
    window also predict behavioral choices at the late window?

    Cross-scaler normalization (mirroring evaluate_phonetic_transfer):
      - Behavioral window data is normalized with the behavioral decoder's own neural
        StandardScaler (the one inside the pca branch of its ColumnTransformer, fit on
        behavioral-window training data), not the acoustic decoder's scaler.
      - Only the acoustic decoder's LogisticRegression is then applied to the result.

    Checkpoints:
      phonetic_decoder_checkpoints:   from behavior_decoding_single_electrode_acoustic/{subject}/
      behavioral_decoder_checkpoints: from behavior_decoding_single_electrode/{subject}/results.pt
    """
    phon_key = (t_subject, t_electrode_idx, t_phoneme_pair, t_smin_phon, t_smax_phon)
    acoustic_models = phonetic_decoder_checkpoints[t_subject]["models"][phon_key]
    acoustic_outcomes = phonetic_decoder_checkpoints[t_subject]["outcomes"][phon_key]

    behav_outer_key = (t_subject, t_electrode_idx, t_phoneme_pair)
    behav_inner_dict = behavioral_decoder_checkpoints[t_subject]["A_decoders"][
        behav_outer_key
    ]

    t_epochs = data.epochs[t_subject]
    md = t_epochs.metadata
    t_epoch_data = t_epochs.get_data(picks=t_electrode_idx).squeeze(1)

    outcomes = []
    for fold, fold_rows in acoustic_outcomes.groupby("fold"):
        epoch_idxs_all = fold_rows["epoch_idx"].values
        # Filter to the word_end of interest
        we_mask = md.loc[epoch_idxs_all]["word_end"] == t_word_end
        epoch_idxs = epoch_idxs_all[we_mask.values]
        if len(epoch_idxs) == 0:
            continue

        phon_clf = acoustic_models[fold].named_steps["logisticregression"]

        # Cross-scaler: normalize behavioral window data with the behavioral decoder's own
        # neural StandardScaler (fit on behavioral-window training data).
        behav_inner_key = (
            t_subject,
            str(t_electrode_idx),
            t_phoneme_pair,
            (t_word_end,),
            t_smin_behav,
            t_smax_behav,
            fold,
        )
        behav_cp = behav_inner_dict.get(behav_inner_key)
        if behav_cp is None:
            continue
        behav_neural_scaler = (
            behav_cp["estimator"]
            .best_estimator_.named_steps["prep"]
            .named_transformers_["pca"]
            .named_steps["standardscaler"]
        )

        X_behav_neural = t_epoch_data[epoch_idxs][:, t_smin_behav:t_smax_behav]
        X_behav_scaled = behav_neural_scaler.transform(X_behav_neural)
        proba = phon_clf.predict_proba(X_behav_scaled)[:, 1]

        # Evaluate against behavioral target (not acoustic)
        behav_target = md.loc[epoch_idxs]["behavior_dummy_forced"].values

        outcomes.append(
            pd.DataFrame(
                {
                    "epoch_idx": epoch_idxs,
                    "fold": fold,
                    "decoder_target": (behav_target > 0).astype(int),
                    "decoder_proba": proba,
                    "subject": t_subject,
                    "electrode_idx": t_electrode_idx,
                    "phoneme_pair": t_phoneme_pair,
                    "word_end": t_word_end,
                    "smin_behav": t_smin_behav,
                    "smax_behav": t_smax_behav,
                    "smin_phon": t_smin_phon,
                    "smax_phon": t_smax_phon,
                }
            )
        )

    return pd.concat(outcomes)
