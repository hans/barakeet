from typing import cast, Optional, Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.stimuli import POD_dict


def _check_grouper(df, grouper, col,
                   order: Optional[list[Any] | dict[str, Any]] = None,
                   bins=None,
                   share_groupers=True,
                   grouper_type="hue", palette="tab10"
                   ) -> tuple[list[Any], list[Any]] | \
                        tuple[dict[str, list[Any]], dict[Any, list[Any]]]:
    if grouper is None:
        return [], []

    col_values = cast(list[Any], df[col].unique())
    # if grouper is continuous, bin first
    if df[grouper].dtype == float:
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
            if not isinstance(order, dict):
                raise ValueError("If not share_groupers, order must be a dict")
            if set(order.keys()) != set(df[col].unique()):
                raise ValueError("order keys must match unique values of col")
        
        if grouper_type == "hue":
            styles = {col_: list(sns.color_palette(palette, len(order[col_]))) for col_ in order}
        elif grouper_type == "style":
            styles = {col_: _style_list[:len(order[col_])] for col_ in order}

        return order, styles


def plot_epochs(epochs: dict[Any, np.ndarray],
                epochs_df: pd.DataFrame,
                hue=None, style=None,
                hue_bins=None, hue_order=None, style_order=None,
                col="phoneme_pair",
                epoch_times=None,
                close=False,
                share_groupers=True,
                smoke_test=False):
    """
    Args:
        share_groupers: If True, all facets share hues/styles and legends are shown
            once per row. In this case, `hue_order` and `style_order` should be lists.
            If False, each column has its own hue/style and legend is shown on each axis.
            In this case, `hue_order` and `style_order` should be dicts mapping from column
            variable to list of levels.
    """
    hue_order, cmap = _check_grouper(epochs_df, hue, col, hue_order, hue_bins,
                                     share_groupers=share_groupers, grouper_type="hue")
    style_order, style_mapper = _check_grouper(epochs_df, style, col, style_order,
                                               share_groupers=share_groupers, grouper_type="style")
    
    if epoch_times is None:
        epoch_times = np.arange(next(iter(epochs.values())).shape[1])

    if smoke_test:
        epochs_df["site"] = epochs_df.index.get_level_values("subject").str.cat(
            (epochs_df.index.get_level_values("channel") + 1).astype(str), sep="_")
        plot_sites = epochs_df.site.unique()[:2]
        epochs_df = epochs_df[epochs_df.site.isin(plot_sites)]

    col_order = sorted(epochs_df[col].unique())
    g = sns.FacetGrid(data=epochs_df.reset_index(["subject", "channel"]),
                      row="facet_label",
                      col=col, col_order=col_order,
                      height=4, aspect=2,
                      gridspec_kws={"hspace": 0.55})

    def f(data, **f_kwargs):
        ax = plt.gca()

        subject = data.subject.iloc[0]
        channel = data.channel.iloc[0]
        phoneme_pair = data.phoneme_pair.iloc[0]
        col_ = data[col].iloc[0]
        
        ax.set_title(f"{subject}_{channel + 1} {col}")

        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

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
                    print(f"yerp, no epochs for {subject} {channel} {col} {hue_level} {style_level}")

                    # still add a legend handle
                    ax.plot([], [], label=label, color=color, linestyle=linestyle)
                else:
                    eps_ij = epochs[subject, channel][eps_ij_idxs]
                    ax.plot(epoch_times, eps_ij.mean(axis=0),
                            label=label,
                            color=color, linestyle=linestyle)
                    seen_hues.add(hue_level)

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
            # add legend on final axis outside of data, left-aligned to the axis edge
            row[-1].legend(loc="center left", bbox_to_anchor=(1.1, 0.75), title=hue)
        else:
            for ax in row:
                ax.legend(loc="center right", bbox_to_anchor=(1.15, 0.5))

    if close:
        plt.close(g.figure)

    return g, epochs, epochs_df


def add_pod_line(g):
    for row in g.axes:
        for ax, phoneme_pair in zip(row, g.col_names):
            pod = POD_dict[phoneme_pair]
            ax.axvline(pod, color="black", linestyle="dotted")

    return g


def add_uv_annotation(g, feature_block, eoi_df):
    for row, name in zip(g.axes, g.row_names):
        subject, channel_name = name.split("_")
        channel_uv = eoi_df.loc[(subject, int(channel_name) - 1, feature_block)].unique_variance
        row[-1].text(1.2, 0.5, f"UV={channel_uv:.4f}", transform=row[-1].transAxes, ha="left", va="center")

    return g


def add_timit_insets(g, epoch_sources):
    for row, name in zip(g.axes, g.row_names):
        subject, channel_name = name.split("_")
        channel = int(channel_name) - 1

        num_insets = len(epoch_sources)
        inset_width, inset_height = 0.25, 0.2
        inset_wspace = 0.025
        inset_anchor_x = 1 - num_insets * (inset_width + inset_wspace)
        inset_anchor_y = 1.2
        assert inset_anchor_x >= 0

        # compute ymin and ymax across sources for epoched phoneme response
        ys = []
        for epoch_source in epoch_sources.values():
            with h5py.File(epoch_source, "r") as f:
                phoneme_epochs = f[subject]["epochs"][:, channel, :]
                ys.append(phoneme_epochs.flatten())
        ys = np.concatenate(ys)
        ymin = np.percentile(ys, 10)
        ymax = np.percentile(ys, 90)
        del ys

        for i, (epoch_source_name, epoch_source) in enumerate(epoch_sources.items()):
            epoch_df = pd.read_hdf(epoch_source, f"{subject}/epoch_df")

            for ax, phoneme_pair in zip(row, g.col_names):
                plot_phonemes = sorted(list(phoneme_pair.upper()))
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
                ax.title.set_position((0.1, 1))

    return g