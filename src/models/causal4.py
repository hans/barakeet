from dataclasses import dataclass
from typing import Literal

from matplotlib import transforms
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import textgrid
from src.models.causal import run_phase1


@dataclass
class Causal4AnalysisResult:
    subject: str
    population_A: list[int]
    population_A_window: tuple[float, float]

    epochs: mne.Epochs
    test_trial_metadata: pd.DataFrame
    """
    Describes the indices into `epochs` and associated metadata
    for the trials that were used to compute the test outputs.
    """

    p_gt_phoneme: np.ndarray
    """
    Test output: probability of the ground-truth phoneme
    computed from population A.
    """

    phase1_val_score: float
    """Score on the data that will be passed to phase2"""

    phase1_test_scores: np.ndarray

    # post-init validation checks
    def __post_init__(self):
        assert len(self.p_gt_phoneme) == len(self.test_trial_metadata)


def run_causal4_analysis(epochs, subject, phoneme_pair,
                         population_A, population_A_window,
                         split_strategy="stratify",
                         pca_num_components=32):
    # phase1: train a GT phoneme decoder on population A
    fitted, metadata, idxs_val, y_proba_val = run_phase1(
        epochs, subject, phoneme_pair,
        population_A, population_A_window,
        split_strategy=split_strategy,
    )
    
    # estimate ROC-AUC on validation set
    roc_auc = roc_auc_score(
        metadata.iloc[idxs_val].categorical_acoustic_cue.combine(0, max),
        y_proba_val[:, 1],
    )

    # get p(gt phoneme) on the validation set predictions. class 0 of the
    # decoder corresponds to the left phoneme (just when `lexical_evidence == 0`)
    # and class 1 corresponds to the right phoneme (just when `lexical_evidence == 1`)
    # so we can use the `lexical_evidence` column to index into the probabilities
    # predicted by the phoneme decoder
    p_gt_phoneme = y_proba_val[np.arange(len(idxs_val)),
                               metadata.iloc[idxs_val].lexical_evidence]

    return Causal4AnalysisResult(
        subject=subject,
        population_A=population_A,
        population_A_window=population_A_window,

        epochs=epochs[subject],
        test_trial_metadata=metadata.iloc[idxs_val],

        p_gt_phoneme=p_gt_phoneme,
        phase1_val_score=roc_auc,
        phase1_test_scores=fitted['test_score'],
    )


def evaluate_counterfactual_correlation(
        A_source, B_source, B_electrode_idx, B_window_start, B_window_end,
        searchlight_decoder_outputs,
        searchlight_electrode_activations,
        sanity_check=False,
        left=True):
    """
    Evaluate a cross-subject correlation between a source of decoder outputs from A
    and a source of electrode activations from B.
    """

    A_subject, A_phoneme_pair, A_population_A = A_source
    B_subject, B_phoneme_pair, B_population_A = B_source

    assert A_phoneme_pair == B_phoneme_pair, "Phoneme pairs must match for counterfactual evaluation."

    A_activations = searchlight_electrode_activations[A_subject, A_phoneme_pair, A_population_A, B_window_start, B_window_end]
    B_activations = searchlight_electrode_activations[B_subject, B_phoneme_pair, B_population_A, B_window_start, B_window_end]
    # print("--", B_subject, B_phoneme_pair, B_population_A, B_window_start, B_window_end)

    A_metadata = A_activations["metadata"].copy()
    B_metadata = B_activations["metadata"]
    A_mask_left = A_activations["mask_left"]
    B_mask_left = B_activations["mask_left"]

    # print("here", A_source, B_source, B_window_start, B_window_end, B_electrode_idx, B_activations["window_data_left_mean"].shape, B_activations["window_data_right_mean"].shape)
    if left:
        A_metadata = A_metadata[A_mask_left]
        B_metadata = B_metadata[B_mask_left]

        A_p_gt_phoneme = searchlight_decoder_outputs[A_subject, A_phoneme_pair, A_population_A]["p_gt_phoneme_mean"][A_mask_left]
        B_response = B_activations["window_data_left_mean"][:, B_electrode_idx]
    else:
        A_metadata = A_metadata[~A_mask_left]
        B_metadata = B_metadata[~B_mask_left]

        A_p_gt_phoneme = searchlight_decoder_outputs[A_subject, A_phoneme_pair, A_population_A]["p_gt_phoneme_mean"][~A_mask_left]
        B_response = B_activations["window_data_right_mean"][:, B_electrode_idx]

    # resample and resort data.
    # resample: make sure that A and B have the same number of trials
    # resort: make sure that A and B are aligned by stimulus properties. beyond that,
    # we want to randomize the order within e.g. each stimulus condition
    A_metadata["rand"] = np.random.rand(len(A_metadata))

    if len(A_metadata) != len(B_metadata):
        # resample A to have the same number of trials as B
        B_grouped = B_metadata.groupby(["resampled", "lexical_evidence"])
        draw_A_idxs = A_metadata.groupby(["resampled", "lexical_evidence"], as_index=False).apply(
                lambda xs: xs.iloc[np.random.choice(len(xs),
                                                    size=len(B_grouped.groups[xs.name]),
                                                    replace=len(B_grouped.groups[xs.name]) > len(xs))]) \
            .reset_index(level=0, drop=True).sort_values(["resampled", "lexical_evidence", "rand"]).index
        A_metadata["local_idx"] = np.arange(len(A_metadata))
        A_metadata = A_metadata.loc[draw_A_idxs]
        A_p_gt_phoneme = A_p_gt_phoneme[A_metadata.local_idx]
    else:
        # resort A
        # if we are sanity-checking, then don't randomize within stimulus condition
        sort_by = ["resampled", "lexical_evidence", "rand"] if not sanity_check else ["resampled", "lexical_evidence"]
        A_metadata = A_metadata.reset_index().sort_values(sort_by)
        A_p_gt_phoneme = A_p_gt_phoneme[A_metadata.index]
    
    # align
    B_resorted = B_metadata.reset_index().sort_values(["resampled", "lexical_evidence"]).index
    B_response = B_response[B_resorted]

    return spearmanr(A_p_gt_phoneme, B_response)


def counterfactual_baseline(subject, phoneme_pair, population_A, electrode_idx, window_start, window_end,
                            searchlight_decoder_outputs, searchlight_electrode_activations,
                            left=True,
                            sanity_check=False):
    """
    Evaluate a cross-subject null-hypothesis correlation between the decoder outputs from a
    counterfactual subject population A and the electrode activations from a site B.
    """
    if sanity_check:
        # DEV sanity check: only use the same subject and phoneme pair
        control_alternative_keys = [(subject_alt, phoneme_pair_alt, population_A_alt) for subject_alt, phoneme_pair_alt, population_A_alt in searchlight_decoder_outputs.keys()
                                    if subject_alt == subject and phoneme_pair_alt == phoneme_pair]
    else:
        control_alternative_keys = [(subject_alt, phoneme_pair_alt, population_A_alt) for subject_alt, phoneme_pair_alt, population_A_alt in searchlight_decoder_outputs.keys()
                                    if (subject_alt != subject and phoneme_pair_alt == phoneme_pair)]

    n_trials = searchlight_electrode_activations[(subject, phoneme_pair, population_A, window_start, window_end)]["metadata"].shape[0]

    ret = []
    for key in control_alternative_keys:
        subject_alt, phoneme_pair_alt, population_A_alt = key
        if (subject_alt, phoneme_pair_alt, population_A_alt, window_start, window_end) not in searchlight_electrode_activations:
            # Activations for the counterfactual key are not available at this time window.
            # It's likely that the counterfactual key couldn't draw on this time region because
            # its corresponding population A was overlapping.
            continue

        # # DEV
        # if subject_alt != "EC282" or phoneme_pair_alt != "dn" or population_A_alt != "5":
        #     continue

        if sanity_check:
            n_runs = 1
        else:
            n_runs = 5
            n_counterfactual_trials = searchlight_electrode_activations[*key, window_start, window_end]["metadata"].shape[0]
            if n_counterfactual_trials != n_trials:
                # We will be stochastically resampling the trials in order to align
                # counterfactual and real target trials. Do this multiple times so
                # we can get a more stable estimate.
                n_runs *= 10

        for _ in range(n_runs):
            try:
                val = evaluate_counterfactual_correlation(
                    key,
                    (subject, phoneme_pair, population_A),
                    electrode_idx,
                    window_start,
                    window_end,
                    searchlight_decoder_outputs,
                    searchlight_electrode_activations,
                    left=left,
                    sanity_check=sanity_check,
                )

                if val is not None:
                    ret.append((key, val))
            except Exception as e:
                raise
                # print(f"Error evaluating counterfactual for {key}: {e}")
                # continue
    return ret


def realign_epochs_by_behavior(epochs, new_anchor_idx=50):
    target_idxs = epochs.time_as_index(epochs.metadata["slider.rt"])
    data = epochs.get_data() # n_epochs, n_channels, n_times
    n_epochs, n_channels, n_times = data.shape

    new_data = np.zeros_like(data) * np.nan
    mask = np.zeros(data.shape[0], dtype=bool)
    for i in range(data.shape[0]):
        target_idx = target_idxs[i]
        if target_idx >= n_times:
            # impossible; behavior starts after epoch window
            continue

        shift = new_anchor_idx - target_idx
        if shift > 0:
            # push data to right; pad with zeros
            new_data[i, :, shift:] = data[i, :, :n_times - shift]
        elif shift < 0:
            # push data to left; pad with zeros
            new_data[i, :, :n_times + shift] = data[i, :, -shift:]
        else:
            new_data[i] = data[i]
        mask[i] = True

    new_data = new_data[mask]
    max_length = (~np.isnan(new_data)).max(axis=1).argmin(1).max()
    new_data = new_data[:, :, :max_length]

    # create EpochsArray, copying over metadata and indexing appropriately
    new_epochs = mne.EpochsArray(
        new_data, epochs.info,
        tmin=-new_anchor_idx / epochs.info["sfreq"],
        metadata=epochs.metadata[mask],
        verbose=False
    )
    return new_epochs


## visualization functions

def plot_causal4_scatter(plot_meta, plot_meta_df, subject, population_B_window):
    """
    Plot scatter plots relating P(gt phoneme) from population A
    to the HGA from population B.
    """
    g_scatter = sns.FacetGrid(
        plot_meta_df,
        row="electrode_idx",
        col="lexical_evidence",
        aspect=1, height=3, sharey="row")
    scatter_legend_data = {}
    
    def plot_facet_scatter(data, color, **kwargs):
        data = data.copy() # avoid SettingWithCopyWarning

        electrode_idx = data.electrode_idx.iloc[0]
        epoch_idxs = data.epoch_idx

        word_end = data.word_end.iloc[0]
        tg_path = data.textgrid_path.iloc[0]

        is_left = data.lexical_evidence.iloc[0] == 0

        ax = plt.gca()
        ax.set_xlabel("Estimated phoneme probability")
        ax.set_ylabel("HGA")
        ax.set_title(f"{subject} {electrode_idx + 1}, {word_end}")

        # plot epoched response at this electrode
        plot_epochs = plot_meta.epochs[epoch_idxs]
        plot_epoch_data = plot_epochs.copy().pick(electrode_idx).get_data()
        
        window_start_samp, window_end_samp = population_B_window
        plot_epoch_data = plot_epoch_data.squeeze(1)[:, window_start_samp:window_end_samp]
        plot_epoch_data = plot_epoch_data.mean(1)
        assert plot_epoch_data.ndim == 1  # n_trials

        data["HGA"] = plot_epoch_data
        data["resampled_str"] = data.resampled.astype(int).astype(str)

        scatter_cmap = {str(int(resampled)): color
                        for resampled, color in zip(
                            sorted(data.resampled.unique()),
                            sns.color_palette("plasma", n_colors=data.resampled.nunique())
                        )}
        scatter_colors = data.resampled_str.map(scatter_cmap)

        # store for legend
        for resampled_str in data.resampled_str.unique():
            scatter_legend_data[resampled_str] = \
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=scatter_cmap[resampled_str], markersize=8)

        sns.regplot(
            data=data,
            x="p_gt_phoneme",
            y="HGA",
            scatter_kws={
                "s": 20,
                "c": scatter_colors,
                "color": None, # block `color` which would be set by regplot
            },
            ax=ax,
        )

        corr, corr_p = stats.pearsonr(data.p_gt_phoneme, plot_epoch_data)
        control_corr, control_corr_p = stats.pearsonr(data.resampled, plot_epoch_data)
        ax.text(0.05, 0.95, f"$r$ = {corr:.2f} ($p$ = {corr_p:.2g})\ncontrol $r$ = {control_corr:.2f} ($p$ = {control_corr_p:.2g})",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

        return ax

    g_scatter.map_dataframe(plot_facet_scatter)
    g_scatter.add_legend(scatter_legend_data,
                         label_order=sorted(scatter_legend_data.keys()),
                         title="Stimulus step")
    
    return g_scatter


def plot_causal4_evoked(plot_meta, plot_meta_df, subject, population_B_window,
                        hue: Literal["resampled", "p_gt_phoneme_bin_center"] = "p_gt_phoneme_bin_center"):
    """
    Plot evoked responses for each electrode in population B,
    grouped by P(gt phoneme) from population A or by stimulus step.
    """
    g = sns.FacetGrid(
        plot_meta_df,
        row="electrode_idx",
        hue=hue, palette="plasma",
        col="lexical_evidence",
        aspect=3, height=3, sharey="row")

    def plot_facet_evoked(data, color, **kwargs):
        electrode_idx = data.electrode_idx.iloc[0]
        epoch_idxs = data.epoch_idx

        word_end = data.word_end.iloc[0]
        tg_path = data.textgrid_path.iloc[0]
        tg = textgrid.TextGrid.fromFile(str(tg_path))

        ax = plt.gca()
        ax.set_xlabel("Time since word onset (sec)")
        ax.set_ylabel("HGA")
        ax.set_title(f"{subject} {electrode_idx + 1}, {word_end}")

        # plot epoched response at this electrode
        plot_epochs = plot_meta.epochs[epoch_idxs]
        plot_epoch_data = plot_epochs.copy().pick(electrode_idx).get_data().squeeze(1)
        assert plot_epoch_data.ndim == 2  # n_trials * n_times

        ax.axvspan(*plot_epochs.times[list(population_B_window)], color="gray", alpha=0.2)

        plot_times = plot_epochs.times
        plot_epoch_data_mean = plot_epoch_data.mean(0)
        plot_epoch_data_sem = plot_epoch_data.std(0) / np.sqrt(plot_epoch_data.shape[0])
        ax.plot(plot_times, plot_epoch_data_mean, color=color, alpha=0.5, **kwargs)
        ax.fill_between(plot_times, plot_epoch_data_mean - plot_epoch_data_sem,
                        plot_epoch_data_mean + plot_epoch_data_sem, color=color, alpha=0.2)
        
        ax.set_xlim(plot_epochs.times[0], plot_epochs.times[-1])
        
        # ylim = ax.get_ylim()
        # for i in range(len(plot_epoch_data)):
        #     ax.plot(plot_times, plot_epoch_data[i], color=color, alpha=0.1)
        # ax.set_ylim(ylim)
        
        intervals = [interval for interval in tg.tiers[0].intervals
                    if interval.mark is not None and interval.mark.strip()]
        for i, interval in enumerate(intervals):
            if interval.mark is None or not interval.mark.strip():
                    continue
            ax.axvline(interval.minTime, linestyle="--", alpha=0.5, color="salmon")
            ax.text(interval.minTime, 0.025, interval.mark.strip(), rotation=90,
                    ha="right", va="bottom",
                    transform=transforms.blended_transform_factory(ax.transData, ax.transAxes))
            
            if i == len(intervals) - 1:
                # plot offset as well.
                ax.axvline(interval.maxTime, linestyle="--", alpha=0.5, color="blue")

        return ax

    g.map_dataframe(plot_facet_evoked)

    def add_evoked_behavioral_spans(g):
        # Add vertical lines for behavioral RT distributions
        for i, row in enumerate(g.row_names):
            for j, col in enumerate(g.col_names):
                ax = g.axes[i, j]
                subplot_df = plot_meta_df[(plot_meta_df.electrode_idx == int(row)) & (plot_meta_df.lexical_evidence == col)]

                value = subplot_df["slider.rt"]
                ax.axvline(value.median(), color="forestgreen", linestyle="--", alpha=0.5)
                # plot IQR
                ax.axvspan(
                    value.quantile(0.25),
                    value.quantile(0.75),
                    color="forestgreen", alpha=0.2
                )
    add_evoked_behavioral_spans(g)

    hue_label = "P(gt phoneme)" if hue == "p_gt_phoneme_bin_center" else "Stimulus step"
    g.add_legend(title=hue_label)
    g.fig.suptitle(f"Evoked responses by {hue_label}")

    return g


def plot_causal4_raster(plot_meta, plot_meta_df, subject, population_B_window,
                        sort_by: Literal["p_gt_phoneme", "resampled"] = "p_gt_phoneme",
                        rasterized=True,
                        clip_zscore=(-3, 3),
                        parameter_cache=None):
    g_raster = sns.FacetGrid(
        plot_meta_df,
        row="electrode_idx",
        col="lexical_evidence",
        aspect=3, height=3)
    
    parameter_cache = parameter_cache or {}
    
    def plot_facet_raster(data, color, sort_by="p_gt_phoneme",
                          clip_zscore=(-3, 3),
                          rasterized=True, **kwargs):
        if sort_by not in ["p_gt_phoneme", "resampled"]:
            raise NotImplementedError(f"Sorting by {sort_by} is not implemented. "
                                      "Use 'p_gt_phoneme' or 'resampled'.")
        # re-sort according to sort_by
        data = data.sort_values(sort_by)

        electrode_idx = data.electrode_idx.iloc[0]
        epoch_idxs = data.epoch_idx

        word_end = data.word_end.iloc[0]
        tg_path = data.textgrid_path.iloc[0]
        tg = textgrid.TextGrid.fromFile(str(tg_path))

        # grab the existing Axes
        ax = plt.gca()
        # create a divider for the existing axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        # append an axes on the left, 15% width of the heatmap ax
        ax_line = divider.append_axes("left", size="15%", pad=0.05)

        # PLOT THE LINE (stacked left)
        # y = trial index (0..n-1), x = sort_by value
        ax_line.plot(data[sort_by], np.arange(len(data)), linestyle='-')
        ax_line.invert_yaxis()            # so trial 0 is at top (same as heatmap)
        ax_line.xaxis.set_label_position("top")
        ax_line.xaxis.tick_top()
        ax_line.yaxis.set_visible(False)
        ax_line.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax_line.spines['right'].set_visible(True)
        ax_line.spines['top'].set_visible(False)
        ax_line.spines['left'].set_visible(False)
        ax_line.spines['bottom'].set_visible(False)

        assert data.label_lexical.nunique() == 1
        if sort_by == "p_gt_phoneme":
            ax_line.set_xlabel(f"P(/{data.label_lexical.iloc[0]}/)")
            ax_line.set_xlim(0, 1)
            ax_line.set_xticks(np.linspace(0, 1, 5))
        elif sort_by == "resampled":
            ax_line.set_xlabel("Stimulus step")
            ax_line.set_xlim(data.resampled.min() - 1, data.resampled.max() + 1)
        ax_line.set_xticklabels([])

        ax.set_xlabel("Time since word onset (sec)")
        ax.set_ylabel("Trial")
        ax.set_title(f"{subject} {electrode_idx + 1}, {word_end}")

        # plot epoched response at this electrode
        plot_epochs = plot_meta.epochs[epoch_idxs]
        plot_epoch_data = plot_epochs.copy().pick(electrode_idx).get_data().squeeze(1)
        assert plot_epoch_data.ndim == 2

        # z-score across trials and time points
        if (subject, electrode_idx) not in parameter_cache:
            # cache mean and std for z-scoring
            all_epoch_data = plot_meta.epochs.copy().pick(electrode_idx).get_data().squeeze(1)
            hga_mean = all_epoch_data.flatten().mean()
            hga_std = all_epoch_data.flatten().std()
            parameter_cache[subject, electrode_idx] = (hga_mean, hga_std)
        else:
            hga_mean, hga_std = parameter_cache[subject, electrode_idx]
        plot_epoch_data = (plot_epoch_data - hga_mean) / hga_std

        # clip extreme values to avoid outliers
        plot_epoch_data = np.clip(plot_epoch_data, clip_zscore[0], clip_zscore[1])

        ax = sns.heatmap(
            plot_epoch_data,
            cmap="plasma", cbar=False, ax=ax,
            yticklabels=False,
            rasterized=rasterized,
        )

        # Plot a subsample of time points
        zero_idx = plot_epochs.time_as_index(0)[0]
        xticks = np.linspace(zero_idx, plot_epoch_data.shape[1] - 1,
                             num=10, dtype=int)
        # but make sure we include the start and end of the population B window
        xticks = np.unique(np.concatenate([
            xticks,
            [0], # label first time point
            [population_B_window[0]],
            [population_B_window[1]],
        ]))
        ax.set_xticks(xticks)
        ax.set_xticklabels(plot_epochs.times[xticks].round(2), rotation=45)

        ax.axvline(population_B_window[0], linestyle="--", color="red", linewidth=2)
        ax.axvline(population_B_window[1], linestyle="--", color="red", linewidth=2)

        # ax.axvspan(*plot_epochs.times[list(population_B_window)], color="gray", alpha=0.2)

    g_raster.map_dataframe(plot_facet_raster, sort_by=sort_by, clip_zscore=clip_zscore,
                           rasterized=rasterized)
    
    sort_label = "P(gt phoneme)" if sort_by == "p_gt_phoneme" else "Stimulus step"
    g_raster.fig.suptitle(f"Rasters sorted by {sort_label}")

    return g_raster