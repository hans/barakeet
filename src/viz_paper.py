"""
Final visualization functions for paper figures.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger as L
from matplotlib import transforms
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ttest_ind
import seaborn as sns
import textgrid


@dataclass
class PaperData:
    electrode_df: pl.DataFrame
    plot_phon_phon_df: pl.DataFrame
    plot_behav_phon_df: pl.DataFrame
    plot_behav_behav_df: pl.DataFrame
    plot_phon_behav_df: pl.DataFrame
    behav_roc_auc_searchlight_df: pl.DataFrame
    all_md: pl.DataFrame
    word_end_df: pl.DataFrame
    epochs: dict[str, mne.Epochs]


epoch_tmin = -0.4
epoch_sfreq = 100

resampled_palette = sns.color_palette("cool", n_colors=6)


def zoomin_hga(data: PaperData,
               subject, electrode_idx, phoneme_pair, word_end,
               textgrid_dir,
               controlled=True, controlled_resampled_steps=(3,),
               resampled_palette=resampled_palette,
               include_phonemes=True,
               hide_bottom=False,
               figsize=(5.25, 4), title=False):
    """
    hide_bottom: skip xticks and xlabel; because it's going to be stacked into a vertical figure
    """
    assert set(controlled_resampled_steps).isdisjoint({1, 6}), "Violated plotting assumption"

    subplot_phon_phon_df = (
        data.plot_phon_phon_df.filter(
            (pl.col("electrode_idx") == electrode_idx),
            (pl.col("subject") == subject),
            (pl.col("phoneme_pair") == phoneme_pair),
            (pl.col("word_end") == word_end)
        )
    )
    subplot_behav_phon_df = (
        data.plot_behav_phon_df.filter(
            (pl.col("electrode_idx") == electrode_idx),
            (pl.col("subject") == subject),
            (pl.col("phoneme_pair") == phoneme_pair),
            pl.col("word_end") == word_end
        )
    )

    # behav predictions
    subplot_behav_behav_df = (
        data.plot_behav_behav_df.filter(
            (pl.col("electrode_idx") == electrode_idx),
            (pl.col("subject") == subject),
            (pl.col("phoneme_pair") == phoneme_pair),
            (pl.col("word_end") == word_end)
        )
    )
    subplot_phon_behav_df = (
        data.plot_phon_behav_df.filter(
            (pl.col("electrode_idx") == electrode_idx),
            (pl.col("subject") == subject),
            (pl.col("phoneme_pair") == phoneme_pair),
            (pl.col("word_end") == word_end)
        )
    )

    assert subplot_phon_phon_df.select(pl.n_unique("smin")).item() == 1
    assert subplot_behav_phon_df.group_by("word_end").agg(pl.n_unique("smin")).select(pl.max("smin")).item() == 1
    assert subplot_behav_behav_df.group_by("word_end").agg(pl.n_unique("smin")).select(pl.max("smin")).item() == 1
    assert subplot_phon_behav_df.select(pl.n_unique("smin")).item() == 1

    ###

    epochs_i = data.epochs[subject]
    epoch_data = epochs_i.copy().apply_baseline().get_data(picks=electrode_idx).squeeze(1)

    plot_tmin = 0
    plot_tmax = data.word_end_df.filter(pl.col("phoneme_pair") == phoneme_pair,
                                   (pl.col("word_end") == word_end)).select(pl.max("word_end_offset")).item() + 0.1
    plot_smin = int((plot_tmin - epoch_tmin) * epoch_sfreq)
    plot_smax = int((plot_tmax - epoch_tmin) * epoch_sfreq)    

    plot_highlight_phon_window = subplot_phon_phon_df.select(["smin", "smax"]).unique().to_numpy().flatten()
    plot_highlight_behav_window = subplot_behav_behav_df.select(["smin", "smax"]).unique().to_numpy().flatten()

    highlight_phon_times = epochs_i.times[[plot_highlight_phon_window[0], plot_highlight_phon_window[1]]]
    highlight_behav_times = epochs_i.times[[plot_highlight_behav_window[0], plot_highlight_behav_window[1]]]

    f, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if title:
        f.suptitle(f"Subject {subject}, Electrode {electrode_idx}, {phoneme_pair}, {word_end}")

    ### HGA plot by stimulus step

    plot_phon_smin = plot_smin
    plot_phon_smax = plot_highlight_behav_window[1] + 10

    def plot_phon_controlled(plot_phon_keys, ax, color_strategy="resampled"):
        if color_strategy == "extreme_vs_controlled":
            palette = {
                False: "#9E9E9E", # extreme values, 1 or 6
                True: "#D62728",  # controlled values
            }
        elif color_strategy == "resampled":
            palette = resampled_palette
        else:
            raise ValueError(f"Unknown color strategy: {color_strategy}")

        linestyles = {0: "solid", 1: "dashed"}

        all_epoch_data = {}

        i = 0
        plot_phon_keys = plot_phon_keys.to_pandas()
        # exclude cases of extreme stimuli with non-acoustic-following behavior
        plot_phon_keys = plot_phon_keys[~((plot_phon_keys.resampled == 1) & (plot_phon_keys.behavior_dummy_forced == 1)) &
                                        ~((plot_phon_keys.resampled == 6) & (plot_phon_keys.behavior_dummy_forced == 0))]
        plot_phon_keys["controlled"] = plot_phon_keys.resampled.isin(controlled_resampled_steps)

        word_ends = plot_phon_keys.word_end
        assert word_ends.nunique() == 1
        word_end = word_ends.iloc[0]

        for ((controlled, decoder_target, label_behavior), rows) in plot_phon_keys.groupby(["controlled", "behavior_dummy_forced", "label_behavior_forced"]):
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
                    L.warning(f"Multiple resampled values for controlled={controlled}, decoder_target={decoder_target}, label_behavior={label_behavior}: {rows.resampled.unique()}. Using color for most common resampled value.")
                most_common_resampled = int(rows.resampled.mode().iloc[0])
                color = palette[most_common_resampled - 1]

            linewidth = 3

            if controlled:
                acoustic_label = "Ambig"
            else:
                resampled = 1 if decoder_target == 0 else 6
                acoustic_label = str(resampled)
            ax.plot(times, epoch_mean, label=f"Chose $\\it{{{label_behavior}{word_end[1:]}}}$",
                    color=color, ls=linestyles[decoder_target], linewidth=linewidth)
            ax.fill_between(times, epoch_mean - epoch_sem, epoch_mean + epoch_sem,
                            color=color, alpha=0.3, rasterized=True)
            ax.set_xlim(times[0], times[-1])

            i += 1

        return all_epoch_data

    def plot_windowed_ttest(ax, all_epoch_data, group1, group2,
                            test_smin=plot_phon_smin, test_smax=plot_phon_smax,
                            test_window_size=4, test_window_stride=4,
                            color="black", alpha=0.5,
                            bar_height_ratio=0.01, bar_y_ratio=0.95):
        # Windowed t-test

        # Get windowed means for each condition
        test_window_starts = np.arange(test_smin, test_smax - test_window_size + 1, test_window_stride)
        test_results = []
        for start in test_window_starts:
            group1_data = all_epoch_data.get(group1, np.empty((0, test_window_size)))[:, start:start+test_window_size]
            group2_data = all_epoch_data.get(group2, np.empty((0, test_window_size)))[:, start:start+test_window_size]
            # average over time within the window
            group1_data = group1_data.mean(axis=1)
            group2_data = group2_data.mean(axis=1)
            if len(group1_data) > 0 and len(group2_data) > 0:
                t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)
            else:
                t_stat, p_value = np.nan, np.nan
            test_results.append((start, start + test_window_size, t_stat, p_value))
        test_results_df = pd.DataFrame(test_results, columns=["start_sample", "end_sample", "t_stat", "p_value"])
        test_results_df["tmin"] = test_results_df["start_sample"] / epoch_sfreq + epoch_tmin
        test_results_df["tmax"] = test_results_df["end_sample"] / epoch_sfreq + epoch_tmin
        # print(test_results_df)

        ymin, ymax = ax.get_ylim()
        bar_height = (ymax - ymin) * bar_height_ratio
        bar_y = ymin + (ymax - ymin) * bar_y_ratio
        for row in test_results_df.itertuples():
            if row.p_value < 0.05:
                ax.barh(y=bar_y,
                        width=row.tmax - row.tmin,
                        left=row.tmin,
                        height=bar_height,
                        color=color, alpha=alpha, edgecolor="none")

    plot_epoch_keys = subplot_phon_phon_df.select(["epoch_idx", "resampled", "textgrid_path"]).unique()

    if controlled:
        plot_epoch_keys = subplot_phon_phon_df.filter(
            pl.col("resampled").is_in([1, 6] + list(controlled_resampled_steps))
        ).select(["epoch_idx", "resampled", "behavior_dummy_forced", "label_behavior_forced", "textgrid_path", "word_end"]).unique()

        all_epoch_data_extreme = plot_phon_controlled(plot_epoch_keys.filter(pl.col("resampled").is_in([1, 6])), axs[0])
        all_epoch_data_controlled = plot_phon_controlled(plot_epoch_keys.filter(pl.col("resampled").is_in(controlled_resampled_steps)), axs[1])

        # add windowed t-test results for behavior at controlled resampled step
        plot_windowed_ttest(axs[1], all_epoch_data_controlled,
                            group1=(True, 0),
                            group2=(True, 1),
                            color="black", alpha=0.5,
                            bar_height_ratio=0.04)
        # add windowed t-test results for phonetic contrast
        plot_windowed_ttest(axs[0], all_epoch_data_extreme,
                            group1=(False, 0), group2=(False, 1),
                            color="black", alpha=0.5,
                            bar_height_ratio=0.04)
        # axs[0].axvspan(highlight_phon_times[0], highlight_phon_times[-1], color="gray", alpha=0.3)
        # axs[0].axvspan(highlight_behav_times[0], highlight_behav_times[-1], color="yellow", alpha=0.3)
    else:
        raise NotImplementedError()

    def add_textgrid(ax, textgrid_dir, ep_df, rotation=0, include_phonemes=True,
                     vline_extent=1.25):
        textgrid_file = ep_df.textgrid_path.iloc[0]
        tg = textgrid.TextGrid.fromFile(Path(textgrid_dir) / textgrid_file)
        assert tg.getNames() == ["phonemes"]

        plot_intervals = [interval for interval in tg.tiers[0].intervals
                        if interval.mark is not None and interval.mark.strip()]
        for i, interval in enumerate(plot_intervals):
            if include_phonemes:
                ax.text(interval.minTime + 0.02, 1.025, interval.mark.strip(), rotation=rotation,
                        ha="right", va="bottom", fontsize=10,
                        transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))

            # add offset of first phoneme as vertical line
            if i == 0:
                ax.axvline(interval.maxTime, ymax=vline_extent,
                           linestyle="--", alpha=0.5, color="black", clip_on=False)

            # # word offset line
            # if i == len(plot_intervals) - 1:
            #     ax.axvline(interval.maxTime, ymax=vline_extent,
            #                linestyle="--", alpha=0.5, color="blue", clip_on=False)

    vline_extent = 1.25
    pod_time = data.word_end_df.filter(pl.col("word_end") == word_end).select(pl.max("pod")).item()
    for i, ax in enumerate(axs):
        ax.axvline(pod_time, linestyle="--", linewidth=2,
                   alpha=0.5, color="red",
                   ymax=vline_extent if i == 0 else 1, clip_on=False)
        ax.set_ylabel("HGA ($z$)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        sns.despine(ax=ax, top=True, right=True)

        include_phonemes_i = include_phonemes and i == 0
        vline_extent = 1.25 if i == 0 else 1
        add_textgrid(ax, textgrid_dir, ep_df=plot_epoch_keys.to_pandas(),
                     include_phonemes=include_phonemes_i, vline_extent=vline_extent)

    # # annotate POD on the upper axis
    # axs[0].annotate("POD", xy=(pod_time, 1), xytext=(pod_time, 1.2),
    #                 arrowprops=dict(arrowstyle="->", color="red"), ha="center", va="bottom", fontsize=11,
    #                 xycoords=transforms.blended_transform_factory(axs[0].transData, axs[0].transAxes))
    
    if hide_bottom:
        axs[-1].tick_params(axis="x", which="both", labelbottom=False)
    else:
        axs[-1].set_xlabel("Time from word onset (s)")

    # axs[0].set_title("Unambiguous", pad=18, loc="left", va="bottom")

    # legend_handles_labels = axs[0].get_legend_handles_labels()
    # # reverse sort
    # legend_handles_labels = (legend_handles_labels[0][::-1], legend_handles_labels[1][::-1])
    # legend = axs[0].legend(*legend_handles_labels, title=None,#"Behavior",
    #                        fontsize=10, frameon=False,
    #                     #    loc="upper right", bbox_to_anchor=(1.175, 0.9))
    #                        loc="upper right", bbox_to_anchor=(1.05, 1.575))
    # # make the lines black to make clear that this is not specific to the top plot
    # for line in legend.get_lines():
    #     line.set_color("black")

    return f