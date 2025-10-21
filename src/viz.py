import logging
from pathlib import Path
from typing import Literal, cast, Optional, Any

import h5py
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import pandas as pd
import seaborn as sns
import textgrid
from tqdm.auto import tqdm

from src.stimuli import POD_dict


L = logging.getLogger(__name__)


def _check_grouper(df, grouper, col,
                   order: Optional[list[Any] | dict[str, Any]] = None,
                   bins=None,
                   share_groupers=True,
                   grouper_type="hue",
                   palette="tab10"
                   ) -> tuple[list[Any], list[Any]] | \
                        tuple[dict[str, list[Any]], dict[Any, list[Any]]]:
    if grouper is None:
        return [], []

    col_values = cast(list[Any], df[col].unique())
    # if we weren't given an order and grouper is continuous, bin first
    if order is None and df[grouper].dtype == float:
        if bins is not None:
            df[grouper] = pd.cut(df[grouper], bins)
            if df[grouper].isna().any():
                raise ValueError(f"NaNs in {grouper} after binning")
        else:
            df[grouper] = pd.cut(df[grouper], 4)

    _style_list = ["-", "--", ":", "-."]
    if share_groupers:
        if order is None:
            order = sorted(df[grouper].unique())
        elif not isinstance(order, list):
            raise ValueError("If share_groupers, order must be a list")
        
        if grouper_type == "hue":
            styles = sns.color_palette(palette, len(order))
        elif grouper_type == "style":
            assert len(_style_list) >= len(order)
            styles = _style_list[:len(order)]

        return order, styles
    else:
        if order is None:
            order = {col_: sorted(df[df[col] == col_][grouper].unique())
                     for col_ in col_values}
        else:
            if isinstance(order, dict):
                pass
            elif isinstance(order, list):
                order = {col_: order for col_ in col_values}
            else:
                raise ValueError("If not share_groupers, order must be a dict or a list")

            if set(order.keys()) != set(df[col].unique()):
                raise ValueError("order keys must match unique values of col")
        
        if grouper_type == "hue":
            styles = {col_: list(sns.color_palette(palette, len(order[col_]))) for col_ in order}
        elif grouper_type == "style":
            assert all(len(order[col_]) <= len(_style_list) for col_ in order)
            styles = {col_: _style_list[:len(order[col_])] for col_ in order}

        return order, styles


def plot_epochs(epochs: dict[Any, np.ndarray],
                epochs_df: pd.DataFrame,
                hue=None, style=None,
                hue_bins=None, hue_order=None, style_order=None,
                palette="tab10",
                row="facet_label",
                col="phoneme_pair",
                errorbar="se",
                height=4, aspect=2,
                epoch_times=None,
                close=False,
                share_groupers=True,
                onset_vline=False,
                sharex=True,
                sharey=True,
                fix_ylim: Literal[None, "percentile"] = None,
                drop_minority_traces: Optional[int] = None,
                smoke_test=False):
    """
    Args:
        share_groupers: If True, all facets share hues/styles and legends are shown
            once per row. In this case, `hue_order` and `style_order` should be lists.
            If False, each column has its own hue/style and legend is shown on each axis.
            In this case, `hue_order` and `style_order` should be dicts mapping from column
            variable to list of levels.
        fix_ylim: If "percentile", fix ylims of each epoch chart to 2.5th and 97.5th percentiles
            of all data. Otherwise, ylims are determined by matplotlib.
        drop_minority_traces: If not None, drop traces with fewer than this many epochs.
    """
    # the below modify epochs_df inplace, so we'll copy here
    epochs_df = epochs_df.copy()
    hue_order, cmap = _check_grouper(epochs_df, hue, col, hue_order, hue_bins,
                                     share_groupers=share_groupers, grouper_type="hue",
                                     palette=palette)
    style_order, style_mapper = _check_grouper(epochs_df, style, col, style_order,
                                               share_groupers=share_groupers, grouper_type="style")
    
    if errorbar not in ["se", None]:
        raise ValueError("Only 'se' and None are supported for errorbar")

    if epoch_times is None:
        epoch_times = np.arange(next(iter(epochs.values())).shape[1])

    if smoke_test:
        epochs_df["site"] = epochs_df.index.get_level_values("subject").str.cat(
            (epochs_df.index.get_level_values("channel") + 1).astype(str), sep="_")
        plot_sites = epochs_df.site.unique()[:2]
        epochs_df = epochs_df[epochs_df.site.isin(plot_sites)]

    if fix_ylim == "percentile":
        # compute ylims across all data
        sites = set((subject, channel) for subject, channel, _ in epochs_df.index)
        all_data = np.concatenate([epochs[subject, channel].flatten()
                                   for subject, channel in sites])
        ylim = tuple(np.percentile(all_data, [1, 99]))
    else:
        ylim = None

    if drop_minority_traces is not None:
        grouper_factors = [row, col]
        if hue is not None:
            grouper_factors.append(hue)
        if style is not None:
            grouper_factors.append(style)
        epochs_df = epochs_df.groupby(grouper_factors) \
            .filter(lambda x: len(x) >= drop_minority_traces)

    col_order = sorted(epochs_df[col].unique())
    g = sns.FacetGrid(data=epochs_df.reset_index(["subject", "channel"]),
                      row=row,
                      col=col, col_order=col_order,
                      height=height, aspect=aspect,
                      sharex=sharex, sharey=sharey,
                      gridspec_kws={"hspace": 0.55})

    def f(data, **f_kwargs):
        ax = plt.gca()

        subject = data.subject.iloc[0]
        channel = data.channel.iloc[0]
        phoneme_pair = data.phoneme_pair.iloc[0]
        col_ = data[col].iloc[0]
        
        ax.set_title(f"{subject}_{channel + 1} {col_}")

        if onset_vline:
            ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

        if ylim is not None:
            ax.set_ylim(ylim)

        if hue is None:
            hue_order_ = [-1]
            cmap_ = None
        elif share_groupers:
            hue_order_ = hue_order
            cmap_ = cmap
        else:
            hue_order_ = hue_order[col_]
            cmap_ = cmap[col_]

        if style is None:
            style_order_ = [-1]
            style_mapper_ = None
        elif share_groupers:
            style_order_ = style_order
            style_mapper_ = style_mapper
        else:
            style_order_ = style_order[col_]
            style_mapper_ = style_mapper[col_]
        
        seen_hues = set()
        seen_styles = set()
        for hue_level in hue_order_:
            for style_level in style_order_:
                md_ij = data
                if hue_level != -1:
                    md_ij = md_ij[md_ij[hue] == hue_level]
                if style_level != -1:
                    md_ij = md_ij[md_ij[style] == style_level]
                eps_ij_idxs = md_ij.index

                color = cmap_[hue_order_.index(hue_level)] if cmap_ is not None else None
                linestyle = style_mapper_[style_order_.index(style_level)] if style_mapper_ is not None else None
                label = f"{hue_level} {style_level}" if style_level != -1 else hue_level

                # eps_ij_idxs = eps_ij_idxs_[eps_ij_idxs_ < all_plot_epochs[subject, channel].shape[0]]
                if len(eps_ij_idxs) == 0:
                    print(f"yerp, no epochs for {subject} {channel} {col_} {hue_level} {style_level}")

                    # still add a legend handle
                    ax.plot([], [], label=label, color=color, linestyle=linestyle)
                else:
                    eps_ij = epochs[subject, channel][eps_ij_idxs]
                    ax.plot(epoch_times, eps_ij.mean(axis=0),
                            label=label,
                            color=color, linestyle=linestyle)
                    seen_hues.add(hue_level)

                    if errorbar == "se":
                        # fillbetween with sem
                        sem = eps_ij.std(axis=0) / np.sqrt(eps_ij.shape[0])
                        ax.fill_between(epoch_times, eps_ij.mean(axis=0) - sem, eps_ij.mean(axis=0) + sem, alpha=0.3,
                                        label=None, color=color)

        # annotate point of disambiguation
        pod = POD_dict[phoneme_pair]
        ax.axvline(pod, color="black", linestyle="dotted")

    g.map_dataframe(f)

    for row in g.axes:
        if share_groupers:
            legend_title = []
            if hue is not None:
                legend_title.append(hue)
            if style is not None:
                legend_title.append(style)
            legend_title = " ".join(legend_title)

            # add legend on final axis outside of data, left-aligned to the axis edge
            row[-1].legend(loc="center left", bbox_to_anchor=(1.1, 0.75), title=legend_title)
        else:
            for ax in row:
                ax.legend(loc="center right", bbox_to_anchor=(1.15, 0.5))

    if close:
        plt.close(g.figure)

    return g, epochs, epochs_df


def add_pod_line(g):
    for row in g.axes:
        for ax, col_name in zip(row, g.col_names):
            phoneme_pair = col_name.split()[0].strip()
            pod = POD_dict[phoneme_pair]
            ax.axvline(pod, color="black", linestyle="dotted")

    return g


def add_uv_annotation(g, feature_block, eoi_df):
    for row, name in zip(g.axes, g.row_names):
        subject, channel_name = name.split("_")
        channel_uv = eoi_df.loc[(subject, int(channel_name) - 1, feature_block)].unique_variance
        row[-1].text(1.2, 0.5, f"UV={channel_uv:.4f}", transform=row[-1].transAxes, ha="left", va="center")

    return g


def add_textgrid_single(ax, textgrid_dir, ep_df, rotation=90):
    textgrid_file = ep_df.textgrid_path.iloc[0]
    tg = textgrid.TextGrid.fromFile(Path(textgrid_dir) / textgrid_file)
    assert tg.getNames() == ["phonemes"]

    plot_intervals = [interval for interval in tg.tiers[0].intervals
                      if interval.mark is not None and interval.mark.strip()]
    for i, interval in enumerate(plot_intervals):
        ax.axvline(interval.minTime, linestyle="--", alpha=0.5, color="salmon")
        ax.text(interval.minTime, 0.025, interval.mark.strip(), rotation=rotation,
                ha="right", va="bottom",
                transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))

        if i == len(plot_intervals) - 1:
            # also add end time
            ax.axvline(interval.maxTime, linestyle="--", alpha=0.5, color="blue")


def add_textgrid(g, textgrid_path, ep_df):
    for row, name in zip(g.axes, g.row_names):
        for ax, col_name in zip(row, g.col_names):
            phoneme_pair = col_name.split()[0].strip()
            subject, _ = name.split("_")

            ep_df_i = ep_df.query("subject == @subject and phoneme_pair == @phoneme_pair")
            textgrid_path_i = ep_df_i.textgrid_path.iloc[0]

            tg = textgrid.TextGrid.fromFile(Path(textgrid_path) / textgrid_path_i)
            assert tg.getNames() == ["phonemes"]
            
            for interval in tg.tiers[0].intervals:
                if interval.mark is None or not interval.mark.strip():
                    continue
                ax.axvline(interval.minTime, linestyle="--", alpha=0.5, color="salmon")
                ax.text(interval.minTime, 0.025, interval.mark.strip(), rotation=90,
                        ha="right", va="bottom",
                        transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))

    return g


def add_behavior_insets(g, ep_df, plot_only_once: Optional[Literal["row", "global"]] = "row"):
    seen_plots = set()
    for row, name in zip(g.axes, g.row_names):
        subject, _ = name.split("_")
        if plot_only_once == "row":
            # refresh seen_plots for each row
            seen_plots = set()
        
        inset_width, inset_height = 0.25, 0.3
        inset_wspace = 0.025
        inset_anchor_x = 0.025
        inset_anchor_y = 1.3

        for ax, col_name in zip(row, g.col_names):
            phoneme_pair = col_name.split()[0].strip()

            plot_id = (subject, phoneme_pair)
            if plot_only_once and plot_id in seen_plots:
                continue
            seen_plots.add(plot_id)

            ep_df_i = ep_df.query("subject == @subject and phoneme_pair == @phoneme_pair")

            ax_inset = ax.inset_axes([inset_anchor_x,
                                      inset_anchor_y - inset_height,
                                      inset_width, inset_height])
            
            md = ep_df_i.groupby("label").label_behavior.value_counts(normalize=True) \
                .sort_index(key=lambda s: s.str[0].map({**{phon: idx for idx, phon in enumerate(phoneme_pair)},
                                                        **{"~": 2}})) \
                .unstack()
            
            # reorder
            md = md.reindex(columns=[*phoneme_pair, "~"], fill_value=0)
            md.plot(kind="barh", stacked=True, color=sns.color_palette()[:3], width=0.95, ax=ax_inset)
            # remove border
            ax_inset.spines[:].set_visible(False)
            ax_inset.set_xlabel(None)
            ax_inset.set_ylabel(None)
            ax_inset.set_xticks(np.linspace(0, 1, 5))
            ax_inset.set_xticklabels([])
            # put xticks on top
            ax_inset.xaxis.set_ticks_position('top')
            # add gridlines
            ax_inset.grid(axis="x", linestyle="--", alpha=0.8)
            ax_inset.legend(loc="upper right", bbox_to_anchor=(1.6, 1.15),
                            labelspacing=0.15, fontsize=10, title_fontsize=10, title="behavior")
            
    return g


def precompute_timit_bounds(epoch_sources, subjects,
                            percentile_bounds=(10, 90)):
    """
    pre-compute plot bounds for TIMIT plots on the given subjects
    """

    results = {}
    for subject in tqdm(subjects):
        for epoch_source in epoch_sources.values():
            with h5py.File(epoch_source, "r") as f:
                if subject not in f:
                    L.warning(f"{subject} not found in {epoch_source}")
                    continue
                phoneme_epochs = f[subject]["epochs"]
                
                # n_channels * 2
                results[subject, epoch_source] = np.percentile(phoneme_epochs[()], percentile_bounds, axis=(0, 2)).T

    return results


def timit_subplots(subject, channel, plot_phonemes, epoch_sources,
                   timit_bounds_dict: Optional[dict] = None,
                   n_cols=2, cell_height=4, cell_aspect=2):
    n_cols = 2
    n_rows = len(epoch_sources) // n_cols + (len(epoch_sources) % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(cell_height * cell_aspect * n_cols, cell_height * n_rows),
                            sharex=True)

    for i, (ax, (epoch_source_name, epoch_source)) in enumerate(zip(axs.flat, epoch_sources.items())):
        try:
            epoch_df = pd.read_hdf(epoch_source, f"{subject}/epoch_df")
        except KeyError:
            L.warning(f"{subject} not found in {epoch_source}")
            continue

        plot_epoch_dfs = [
            epoch_df[epoch_df.epoch_label == phoneme]
            for phoneme in plot_phonemes
        ]

        with h5py.File(epoch_source, "r") as f:
            epoch_tmin = cast(int, f[subject].attrs["epoch_tmin"])
            epoch_tmax = cast(int, f[subject].attrs["epoch_tmax"])
            epoch_sfreq = cast(int, f[subject].attrs["sfreq"])
            plot_epochs_ij = cast(list[np.ndarray], [
                f[subject]["epochs"][plot_epoch_df.index, channel, :]
                for phoneme, plot_epoch_df in zip(plot_phonemes, plot_epoch_dfs)
            ])

        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        for phoneme, ph_df, ph_epochs in zip(plot_phonemes, plot_epoch_dfs, plot_epochs_ij):
            times = np.arange(ph_epochs.shape[1]) / epoch_sfreq + epoch_tmin
            ax.plot(times, ph_epochs.mean(axis=0), label=phoneme)
            sem = ph_epochs.std(axis=0) / np.sqrt(ph_epochs.shape[0])
            ax.fill_between(times, ph_epochs.mean(axis=0) - sem, ph_epochs.mean(axis=0) + sem, alpha=0.3)

        ax.set_title(epoch_source_name, fontsize="small")
        # ax.set_xlim(epoch_tmin, epoch_tmax)
        ax.set_xlim(times[0], times[-1])

        if timit_bounds_dict is not None and (subject, epoch_source) in timit_bounds_dict:
            # use pre-computed bounds
            subject_bounds = timit_bounds_dict[subject, epoch_source]
            ymin, ymax = subject_bounds[channel]
            ax.set_ylim(ymin, ymax)

        # if we're on the top right cell, add a legend
        if i == min(len(epoch_sources), n_cols) - 1:
            ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.))

    return fig


def add_timit_insets(g, epoch_sources):
    for row, name in zip(g.axes, g.row_names):
        subject, channel_name = name.split("_")
        channel = int(channel_name) - 1

        num_insets = len(epoch_sources)
        inset_width, inset_height = 0.25, 0.2
        inset_wspace = 0.025
        inset_anchor_x = 1 - num_insets * (inset_width + inset_wspace)
        inset_anchor_y = 1.3
        assert inset_anchor_x >= 0

        # compute ymin and ymax across sources for epoched phoneme response
        ys = []
        skip = False
        for epoch_source in epoch_sources.values():
            with h5py.File(epoch_source, "r") as f:
                if subject not in f:
                    L.warning(f"{subject} not found in {epoch_source}")
                    skip = True
                    continue
                phoneme_epochs = f[subject]["epochs"]
                if channel >= phoneme_epochs.shape[1]:
                    L.warning(f"{subject} {channel} not found in {epoch_source}")
                    skip = True
                    continue

                phoneme_epochs = phoneme_epochs[:, channel, :]
                ys.append(phoneme_epochs.flatten())

        if skip:
            continue
        ys = np.concatenate(ys)
        ymin = np.percentile(ys, 10)
        ymax = np.percentile(ys, 90)
        del ys

        for i, (epoch_source_name, epoch_source) in enumerate(epoch_sources.items()):
            epoch_df = pd.read_hdf(epoch_source, f"{subject}/epoch_df")

            for ax, col_name in zip(row, g.col_names):
                phoneme_pair = col_name.split()[0].strip()

                plot_phonemes = list(phoneme_pair.upper())
                plot_epoch_dfs = [
                    epoch_df[epoch_df.epoch_label == phoneme]
                    for phoneme in plot_phonemes
                ]

                with h5py.File(epoch_source, "r") as f:
                    epoch_tmin = cast(int, f[subject].attrs["epoch_tmin"])
                    epoch_tmax = cast(int, f[subject].attrs["epoch_tmax"])
                    epoch_sfreq = cast(int, f[subject].attrs["sfreq"])
                    plot_epochs_ij = cast(list[np.ndarray], [
                        f[subject]["epochs"][plot_epoch_df.index, channel, :]
                        for phoneme, plot_epoch_df in zip(plot_phonemes, plot_epoch_dfs)
                    ])

                ax_inset = ax.inset_axes([inset_anchor_x + inset_wspace + i * (inset_width + inset_wspace),
                                          inset_anchor_y - inset_height,
                                          inset_width, inset_height])
                ax_inset.axvline(0, color="gray", linestyle="--", alpha=0.5)
                ax_inset.axhline(0, color="gray", linestyle="--", alpha=0.5)
                for phoneme, ph_df, ph_epochs in zip(plot_phonemes, plot_epoch_dfs, plot_epochs_ij):
                    times = np.arange(ph_epochs.shape[1]) / epoch_sfreq + epoch_tmin
                    ax_inset.plot(times, ph_epochs.mean(axis=0), label=phoneme)
                    sem = ph_epochs.std(axis=0) / np.sqrt(ph_epochs.shape[0])
                    ax_inset.fill_between(times, ph_epochs.mean(axis=0) - sem, ph_epochs.mean(axis=0) + sem, alpha=0.3)

                ax_inset.set_title(epoch_source_name, fontsize="small")
                ax_inset.set_xlim(epoch_tmin, epoch_tmax)
                ax_inset.set_ylim(ymin, ymax)

                # y tick labels on first axis only
                if i > 0:
                    ax_inset.tick_params(axis="y", labelleft=False)
                # legend and x tick labels on last axis only
                if i == num_insets - 1:
                    ax_inset.legend(loc="upper right", bbox_to_anchor=(1.5, 1.2))
                else:
                    ax_inset.tick_params(axis="x", labelbottom=False)

                # move title to left edge so insets have room
                ax.title.set_position((0.2, 1))

    return g


def plot_timit_epochs_faceted(plot_subject, plot_electrode_idx,
                              phoneme_order,
                              timit_epoch_sources,
                              timit_bounds: Optional[dict] = None,
                              epoch_source: str = "All",
                              show_traces=False, baseline=True,
                              facetgrid_kwargs: Optional[dict] = None):
    """Plot epoched phoneme responses."""
    epoch_source_path = timit_epoch_sources[epoch_source]
    epoch_df = pd.read_hdf(epoch_source_path, f"{plot_subject}/epoch_df")

    facetgrid_kwargs = {
        "height": 1.5,
        "sharey": True,
        "col_wrap": 8,
        **(facetgrid_kwargs or {})
    }
    g = sns.FacetGrid(data=epoch_df, col="epoch_label",
                      col_order=phoneme_order,
                      **facetgrid_kwargs)

    def plot_phoneme_epochs(data, **kwargs):
        if data.empty:
            return
        phoneme = data.epoch_label.iloc[0]
        
        with h5py.File(epoch_source_path, "r") as f:
            tmin = cast(int, f[plot_subject].attrs["epoch_tmin"])
            tmax = cast(int, f[plot_subject].attrs["epoch_tmax"])
            sfreq = cast(int, f[plot_subject].attrs["sfreq"])

            plot_epochs = cast(np.ndarray, f[plot_subject]["epochs"][data.epoch_idx, plot_electrode_idx, :])

        ax = plt.gca()
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

        if baseline:
            # baseline by pre-zero region
            assert tmin < 0
            baseline_start_idx = 0
            baseline_end_idx = int(0 - tmin * sfreq)
            baseline_data = plot_epochs[:, baseline_start_idx:baseline_end_idx]
            baseline_mean = baseline_data.mean(axis=1, keepdims=True)
            plot_epochs = plot_epochs - baseline_mean

        times = np.arange(plot_epochs.shape[1]) / sfreq + tmin
        ax.plot(times, plot_epochs.mean(axis=0), label=phoneme)
        sem = plot_epochs.std(axis=0) / np.sqrt(plot_epochs.shape[0])

        if show_traces:
            for epoch in plot_epochs:
                ax.plot(times, epoch, color="gray", alpha=0.05)
        else:
            ax.fill_between(times, plot_epochs.mean(axis=0) - sem, plot_epochs.mean(axis=0) + sem, alpha=0.3)

        if timit_bounds is not None and (plot_subject, epoch_source_path) in timit_bounds:
            # use pre-computed bounds
            subject_bounds = timit_bounds[plot_subject, epoch_source_path]
            ymin, ymax = subject_bounds[plot_electrode_idx]
            ax.set_ylim(ymin, ymax)

    for ax, (label, data) in zip(tqdm(g.axes.flat), g.facet_data()):
        plt.sca(ax)
        plot_phoneme_epochs(data)

    g.set_titles(col_template="{col_name}")
    g.tight_layout()
    g.fig.suptitle(f"Subject {plot_subject}, electrode {plot_electrode_idx + 1}\nTIMIT tuning", y=1.03)
    return g


def plot_timit_epochs_faceted_by_feature(
        plot_subject, plot_electrode_idx,
        feature_order, feature_map: dict[str, str],
        timit_epoch_sources,
        timit_bounds: Optional[dict] = None,
        epoch_source: str = "All",
        show_traces=False, baseline=True,
        facetgrid_kwargs: Optional[dict] = None):
    """
    Plot epoched phonetic feature responses.
    
    Args:
        feature_map: Maps phoneme to to list of features.
    """
    epoch_source_path = timit_epoch_sources[epoch_source]
    epoch_df = pd.read_hdf(epoch_source_path, f"{plot_subject}/epoch_df")

    facetgrid_kwargs = {
        "height": 1.5,
        "sharey": True,
        "col_wrap": 4,
        **(facetgrid_kwargs or {})
    }
    
    feature_df = pd.DataFrame.from_dict(feature_map, orient="index") \
        .rename_axis("phoneme").reset_index().melt(id_vars=["phoneme"], value_name="feature") \
        .drop(columns=["variable"]).dropna()
    epoch_df = pd.merge(epoch_df, feature_df, left_on="epoch_label", right_on="phoneme", how="left")
    g = sns.FacetGrid(data=epoch_df, col="feature",
                      col_order=feature_order,
                      **facetgrid_kwargs)

    def plot_feature_epochs(data, **kwargs):
        if data.empty:
            return
        feature = data.feature.iloc[0]
        phonemes = sorted(data.epoch_label.unique())
        
        with h5py.File(epoch_source_path, "r") as f:
            tmin = cast(int, f[plot_subject].attrs["epoch_tmin"])
            tmax = cast(int, f[plot_subject].attrs["epoch_tmax"])
            sfreq = cast(int, f[plot_subject].attrs["sfreq"])

            plot_epochs = np.concatenate([
                cast(np.ndarray, f[plot_subject]["epochs"][ph_data.epoch_idx, plot_electrode_idx, :])
                for _, ph_data in data.groupby("epoch_label")
            ])

        ax = plt.gca()
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

        if baseline:
            # baseline by pre-zero region
            assert tmin < 0
            baseline_start_idx = 0
            baseline_end_idx = int(0 - tmin * sfreq)
            baseline_data = plot_epochs[:, baseline_start_idx:baseline_end_idx]
            baseline_mean = baseline_data.mean(axis=1, keepdims=True)
            plot_epochs = plot_epochs - baseline_mean

        times = np.arange(plot_epochs.shape[1]) / sfreq + tmin
        ax.plot(times, plot_epochs.mean(axis=0), label=feature)
        sem = plot_epochs.std(axis=0) / np.sqrt(plot_epochs.shape[0])

        if show_traces:
            for epoch in plot_epochs:
                ax.plot(times, epoch, color="gray", alpha=0.05)
        else:
            ax.fill_between(times, plot_epochs.mean(axis=0) - sem, plot_epochs.mean(axis=0) + sem, alpha=0.3)

        if timit_bounds is not None and (plot_subject, epoch_source_path) in timit_bounds:
            # use pre-computed bounds
            subject_bounds = timit_bounds[plot_subject, epoch_source_path]
            ymin, ymax = subject_bounds[plot_electrode_idx]
            ax.set_ylim(ymin, ymax)

    for ax, (label, data) in zip(tqdm(g.axes.flat), g.facet_data()):
        plt.sca(ax)
        plot_feature_epochs(data)

    g.set_titles(col_template="{col_name}")
    g.fig.suptitle(f"Subject {plot_subject}, electrode {plot_electrode_idx + 1}\nTIMIT tuning", y=1.03)
    g.tight_layout()
    return g