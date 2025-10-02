from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from matplotlib import transforms
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import textgrid

from src.models.causal import run_phase1
import src.viz as viz


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

    p_left_phoneme: np.ndarray
    """
    Test output: probability of the left phoneme
    computed from population A."""

    phase1_val_score: float
    """Score on the data that will be passed to phase2"""

    phase1_test_scores: np.ndarray

    # post-init validation checks
    def __post_init__(self):
        assert len(self.p_gt_phoneme) == len(self.test_trial_metadata)


@dataclass
class SearchlightSpec:
    align_to: Literal["onset", "word_offset"]
    window_size: float
    window_start: float
    window_end: float


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
        p_left_phoneme=y_proba_val[:, 0],
        phase1_val_score=roc_auc,
        phase1_test_scores=fitted['test_score'],
    )


def prepare_searchlight_windows(searchlight_spec: SearchlightSpec,
                                epochs: mne.Epochs,
                                textgrid_dir=None,
                                global_tmin=None):
    """
    Prepare searchlight windows for the given epochs and the given searchlight
    spec.

    Arguments:
        searchlight_spec: A dictionary containing the searchlight parameters.   
        epochs: MNE Epochs object containing the epochs data.
        global_tmin: The global minimum time point for any searchlight window.

    Returns:
        A list of tuples, each containing

            (absolute_window_start, absolute_window_end,
             relative_window_start, relative_window_end)

        for each searchlight window, each in units of time samples of `epochs`.
        Absolute samples are given relative to the start of the epoch;
        relative samples are given relative to the searchlight's `align_to`.
    """

    window_size = searchlight_spec.window_size
    window_start = searchlight_spec.window_start

    # searchlight window should end at latest at the 80% percentile of the behavioral onset
    behavior_onsets = epochs.metadata["slider.rt"].values
    behavior_cutoff = np.percentile(behavior_onsets, 80)
    window_end = min(searchlight_spec.window_end, behavior_cutoff)

    # convert to samples
    sfreq = epochs.info["sfreq"]
    window_size_samp = int(window_size * sfreq)

    if searchlight_spec.align_to == "onset":
        alignment_point = 0
    elif searchlight_spec.align_to == "word_offset":
        assert textgrid_dir is not None, "textgrid_dir must be provided for word_offset alignment"
        tg = textgrid.TextGrid.fromFile(str(Path(textgrid_dir) / epochs.metadata.textgrid_path.iloc[0]))
        phoneme_intervals = [interval for interval in tg.tiers[0].intervals
                             if interval.mark is not None and interval.mark.strip()]
        alignment_point = phoneme_intervals[-1].maxTime
        # TODO ensure same phase?
    alignment_sample = epochs.time_as_index(alignment_point)[0]
    # print(f"Alignment point: {alignment_point} s, sample: {alignment_sample}", epochs.metadata.word_end.iloc[0])

    window_start += alignment_point
    window_end += alignment_point

    if global_tmin is not None:
        window_start = max(window_start, global_tmin)

    # TODO not sure if phase alignment is necessary
    # # if we shifted, make sure the window is still aligned to the requested start + n * window_size
    # if (window_start - searchlight_spec.window_start) % window_size != 0:
    #     window_start += (window_size - (window_start - searchlight_spec.window_start) % window_size)

    window_start_samp = epochs.time_as_index(window_start)[0]
    window_end_samp = epochs.time_as_index(window_end)[0]

    window_starts = np.arange(window_start_samp,
                              window_end_samp - window_size_samp + 1,
                              window_size_samp)
    window_ends = window_starts + window_size_samp

    relative_window_starts = window_starts - alignment_sample
    relative_window_ends = window_ends - alignment_sample

    return list(zip(window_starts, window_ends, relative_window_starts, relative_window_ends))


def decode_behavior(subject, phoneme_pair, population_A, electrode_idx, window_start, window_end,
                    searchlight_electrode_activations, left=True, n_splits=3):
    if np.isnan(window_start) or np.isnan(window_end):
        return np.nan
    # Attempt to decode behavior
    side = "left" if left else "right"
    behav_activations = searchlight_electrode_activations[subject, phoneme_pair, population_A,
                                                          window_start, window_end]
    behav_mask = behav_activations["mask_left"] if side == "left" else ~behav_activations["mask_left"]
    behav_y = behav_activations["metadata"][behav_mask].behavior_dummy_forced
    behav_X = behav_activations[f"window_data_{side}_mean"][:, [electrode_idx]]
    assert behav_y.shape[0] == behav_X.shape[0]

    if behav_y.nunique() < 2:
        # not enough variability in the behavior to decode
        return np.nan
    if behav_y.value_counts().min() < n_splits:
        n_splits = max(2, behav_y.value_counts().min() // 2)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    scores = cross_val_score(clf, behav_X, behav_y, cv=cv, scoring="roc_auc")
    
    return scores.mean()


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

    activation_key = "left" if left else "right"
    A_decoder_outputs = searchlight_decoder_outputs[A_subject, A_phoneme_pair, A_population_A]
    B_activations = searchlight_electrode_activations[B_subject, B_phoneme_pair, B_population_A, B_window_start, B_window_end]
    # print("--", B_subject, B_phoneme_pair, B_population_A, B_window_start, B_window_end)

    A_metadata = A_decoder_outputs["test_trial_metadata"].copy()
    B_metadata = B_activations["metadata"]
    A_mask_left = A_decoder_outputs["mask_left"]
    B_mask_left = B_activations["mask_left"]

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


class Causal4Plotter:

    def __init__(self, epochs: dict[str, mne.Epochs | mne.epochs.EpochsFIF],
                 A_results: pd.DataFrame, B_results: pd.DataFrame,
                 A_decoders,
                 electrode_df: pd.DataFrame,
                 textgrid_dir: str,
                 timit_epoch_sources: dict[str, str]):
        """
        Args:
            epochs: MNE Epochs
            A_results: 
            B_results:
            A_decoders:
            electrode_df:
            textgrid_dir:
            timit_epoch_sources: A dictionary mapping TIMIT epoch category names to pre-computed paths
        """

        self.epochs = epochs
        self.A_results = A_results
        self.B_results = B_results

        self.A_decoders = A_decoders

        self.electrode_df = electrode_df
        self.textgrid_dir = textgrid_dir

        self.timit_epoch_sources = timit_epoch_sources
        # pre-compute TIMIT bounds
        self._timit_bounds = viz.precompute_timit_bounds(
            timit_epoch_sources, subjects=self.epochs.keys()
        )

        self._parameter_cache = {}

    def _get_A_result(self, subject, phoneme_pair, population_name):
        A_row = self.A_results[(self.A_results.subject == subject) & (self.A_results.phoneme_pair == phoneme_pair) & (self.A_results.population_name == population_A)]
        assert len(A_row) == 1, f"Expected one row for {subject} {phoneme_pair} {population_A}, got {len(A_row)}"
        A_row = A_row.iloc[0]
        return A_row

    def _prepare_plot_meta_df(self, row):
        subject = row.subject
        phoneme_pair = row.phoneme_pair

        population_A = row.population_name if hasattr(row, "population_name") else None
        population_B = [row.electrode_idx]
        left = row.left_is_best

        plot_key = (subject, phoneme_pair, population_A)
        plot_num_quantiles = 4

        epochs_i = self.epochs[subject]

        if population_A is not None:
            A_outcomes = self.A_decoders["held_out_outcomes"][subject, population_A, phoneme_pair]

            # sanity check: test trials are the same across repeats
            test_trial_indices = A_outcomes.groupby("fold").apply(lambda xs: xs.epoch_idx.unique()).values
            for i in range(1, len(test_trial_indices)):
                np.testing.assert_array_equal(
                    test_trial_indices[i],
                    test_trial_indices[0],
                )

            # merge estimated probabilities from repeats
            plot_meta_df = pd.merge(
                A_outcomes.groupby("epoch_idx").decoder_proba.mean().reset_index(),
                epochs_i.metadata,
                how="left", left_on="epoch_idx", right_index=True
            )
        else:
            plot_meta_df = epochs_i.metadata.query("phoneme_pair == @phoneme_pair").rename_axis("epoch_idx").reset_index()
            plot_meta_df["decoder_proba"] = np.nan

        # p(gt phoneme) is decoder probability if we are looking at the right phoneme,
        # or 1 - decoder probability if we are looking at the left phoneme
        plot_meta_df["p_gt_phoneme"] = plot_meta_df.groupby("label_lexical").decoder_proba.transform(
            lambda xs: 1 - xs if xs.name == phoneme_pair[0] else xs)
        plot_meta_df["p_gt_phoneme_binned"] = pd.qcut(
            plot_meta_df.p_gt_phoneme, plot_num_quantiles,
            duplicates="drop",
            # labels=[f"Q{i+1}" for i in range(plot_num_quantiles)]
        )
        plot_meta_df["p_gt_phoneme_bin_center"] = plot_meta_df.p_gt_phoneme_binned.apply(
            lambda x: x.mid
        ).astype(float).round(3)

        # TODO concat outcomes from extremes

        def get_textgrid_path(row):
            return Path(self.textgrid_dir) / (Path(row.wav_file).with_suffix(".TextGrid").name)
        plot_meta_df["textgrid_path"] = plot_meta_df.apply(get_textgrid_path, axis=1)

        # cross by electrodes
        plot_meta_df = pd.merge(
            plot_meta_df,
            self.electrode_df.loc[subject].loc[population_B].reset_index(),
            how="cross")

        return plot_meta_df

    def __call__(self, row, smoke_test=False):
        subject = row.subject
        phoneme_pair = row.phoneme_pair
        left_phoneme, right_phoneme = phoneme_pair

        population_A = row.population_name
        population_B = [row.electrode_idx]
        left = row.left_is_best
        population_B_window = (int(row.window_start_samp), int(row.window_end_samp))

        A_row = self._get_A_result(subject, phoneme_pair, population_A)
        # convert to samples
        population_A_window = (
            epochs_i.time_as_index(A_row.smin)[0],
            epochs_i.time_as_index(A_row.smax)[0],
        )

        plot_key = (subject, phoneme_pair, population_A)
        plot_num_quantiles = 4

        epochs_i = self.epochs[subject]

        plot_meta_df = self._prepare_plot_meta_df(row)

        ####

        # Plot: A-population output vs. stimulus step
        # (Underlying Q: How closely tied is this A-population to stimulus vs. internal state?)

        # Concatenate extreme data
        A_extreme_outcomes = self.A_decoders["outcomes"][subject, population_A, phoneme_pair]
        plot_extreme_meta_df = pd.merge(
            A_extreme_outcomes.groupby("epoch_idx").decoder_proba.mean().reset_index(),
            epochs_i.metadata,
            how="left", left_on="epoch_idx", right_index=True
        )
        assert len(set(plot_extreme_meta_df.epoch_idx) & set(plot_meta_df.epoch_idx)) == 0, \
            "Expected no overlap between extreme and regular outcomes"
        plot_extreme_meta_df["p_gt_phoneme"] = plot_extreme_meta_df.groupby("label_lexical").decoder_proba.transform(
            lambda xs: 1 - xs if xs.name == phoneme_pair[0] else xs)

        all_A_df = pd.concat([plot_meta_df, plot_extreme_meta_df], ignore_index=True) \
            .astype({"resampled": int})

        # Plot A relationship between stimulus step and P(phoneme)
        g_A = sns.catplot(data=all_A_df, x="resampled", y="decoder_proba",
                          hue="label_lexical", hue_order=list(phoneme_pair),
                          kind="strip", height=3, aspect=1.25)
        g_A.set_axis_labels("Stimulus step", f"P(/{right_phoneme}/)")
        g_A.ax.set_title(f"{subject} {population_A} {phoneme_pair}")
        g_A.legend.set_title("Lexical\nevidence")
        g_A.ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)

        # Add correlation information
        corr, p_val = spearmanr(all_A_df["resampled"], all_A_df["decoder_proba"])
        g_A.ax.text(0.05, 0.95, f"$r$ = {corr:.2f} ($p$ = {p_val:.2g})",
                    transform=g_A.ax.transAxes)

        ####

        # Plot: B-population output in A-window vs. stimulus step
        # (Underlying Q: Is this B-population also an A?)

        s_epoch_idxs = plot_meta_df.epoch_idx
        s_epochs = epochs_i[s_epoch_idxs].copy().pick(population_B).get_data().squeeze()
        s_epoch_data = s_epochs[:, A_row.smin:A_row.smax]

        plot_meta_df["A_window_B_activation"] = s_epoch_data.mean(axis=1)

        g_B_at_A = sns.catplot(
            data=plot_meta_df, x="resampled", y="A_window_B_activation",
            hue="label_lexical", hue_order=list(phoneme_pair),
            kind="strip", height=3, aspect=1.25)

        # Add correlation information
        corr, p_val = spearmanr(plot_meta_df["resampled"], plot_meta_df["A_window_B_activation"])
        g_B_at_A.ax.text(0.05, 0.95, f"$r$ = {corr:.2f} ($p$ = {p_val:.2g})",
                         transform=g_B_at_A.ax.transAxes)

        ####
        
        g_scatter = plot_causal4_scatter(
            epochs_i, plot_meta_df, subject, population_B_window)

        ####

        # # displot showing spearmanr results
        # spearmanr_results = np.array(row.counterfactual_spearmanr_list)
        # g_displot = sns.displot(
        #     spearmanr_results,
        #     kind="kde", fill=True, color="blue",
        #     height=2.5, aspect=3,
        # )
        # spearmanr_obs = row.corr_left if row.p_val_left < row.p_val_right else row.corr_right
        # g_displot.ax.axvline(spearmanr_obs, color="red", linestyle="--", label="Observed", linewidth=2)
        # g_displot.ax.set_xlabel("Spearman correlation")
        # g_displot.ax.set_title(f"Permutation baseline results\n(z={row.counterfactual_test_z:.2f}, p={row.counterfactual_test_p:.2g})")
        g_displot = None

        ####

        g = plot_causal4_evoked(
            epochs_i, plot_meta_df, subject, population_A_window, population_B_window,
            hue="p_gt_phoneme_bin_center", smoke_test=smoke_test
        )

        ####

        # # plot re-aligned to behavior
        # def plot_facet_evoked_align_behavior(data, color, **kwargs):
        #     electrode_idx = data.electrode_idx.iloc[0]
        #     epoch_idxs = data.epoch_idx

        #     word_end = data.word_end.iloc[0]
        #     tg_path = data.textgrid_path.iloc[0]
        #     tg = textgrid.TextGrid.fromFile(str(tg_path))

        #     ax = plt.gca()
        #     ax.set_xlabel("Time relative to behavior onset (sec)")
        #     ax.set_ylabel("HGA")
        #     ax.set_title(f"{subject} {electrode_idx + 1}, {word_end}")

        #     # plot epoched response at this electrode
        #     plot_epochs = epochs[subject][epoch_idxs]
        #     plot_epochs = causal4.realign_epochs_by_behavior(plot_epochs)

        #     plot_epoch_data = plot_epochs.copy().pick(electrode_idx).get_data().squeeze(1)
        #     assert plot_epoch_data.ndim == 2  # n_trials * n_times

        #     plot_times = plot_epochs.times
        #     plot_epoch_data_mean = np.nanmean(plot_epoch_data, 0)
        #     plot_epoch_data_sem = np.nanstd(plot_epoch_data, 0) / np.sqrt((~np.isnan(plot_epoch_data)).sum(0))
        #     ax.plot(plot_times, plot_epoch_data_mean, color=color, alpha=0.5, **kwargs)
        #     ax.fill_between(plot_times, plot_epoch_data_mean - plot_epoch_data_sem,
        #                     plot_epoch_data_mean + plot_epoch_data_sem, color=color, alpha=0.2)
            
        #     ax.set_xlim(plot_epochs.times[0], plot_epochs.times[-1])

        #     return ax
        
        # g_evoked_by_behavior = sns.FacetGrid(
        #     plot_meta_df,
        #     row="electrode_idx",
        #     hue="p_gt_phoneme_bin_center", palette="plasma",
        #     col="lexical_evidence",
        #     aspect=3, height=3, sharey="row"
        # ).map_dataframe(plot_facet_evoked_align_behavior).add_legend()
        # g_evoked_by_behavior.fig.suptitle("Evoked responses by P(gt phoneme), aligned to behavior onset")
        g_evoked_by_behavior = None

        ####

        g_evoked_resampled = plot_causal4_evoked(
            epochs_i, plot_meta_df, subject, population_A_window, population_B_window,
            hue="resampled", smoke_test=smoke_test
        )

        ####

        g_raster = plot_causal4_raster(
            epochs_i, plot_meta_df, subject, population_B_window,
            plot_extremes=True,
            sort_by="p_gt_phoneme", parameter_cache=self._parameter_cache,
            cbar=False)
        
        g_raster_resampled = plot_causal4_raster(
            epochs_i, plot_meta_df, subject, population_B_window,
            sort_by="resampled", parameter_cache=self._parameter_cache,
            cbar=False)

        g_raster_behavior = plot_causal4_raster(
            epochs_i, plot_meta_df, subject, population_B_window,
            sort_by="behavior_linear", parameter_cache=self._parameter_cache,
            cbar=False)

        ####

        timit_fig = viz.timit_subplots(
            subject, population_B[0],
            plot_phonemes=[ph.upper() for ph in phoneme_pair],
            cell_aspect=1.5,
            epoch_sources=self.timit_epoch_sources,
            timit_bounds_dict=self._timit_bounds)
        timit_fig.suptitle(f"TIMIT responses for {subject} {population_B[0] + 1}")

        return (g_A, g_B_at_A, g_scatter, g_displot,
                g, g_evoked_by_behavior, g_evoked_resampled,
                *g_raster, *g_raster_resampled, *g_raster_behavior,
                timit_fig)


def plot_causal4_scatter(epochs, plot_meta_df, subject, population_B_window,
                         statistic: Literal["spearmanr", "pearsonr"] = "spearmanr",
                         height=3, aspect=1):
    """
    Plot scatter plots relating P(gt phoneme) from population A
    to the HGA from population B.
    """
    g_scatter = sns.FacetGrid(
        plot_meta_df,
        row="electrode_idx",
        col="lexical_evidence",
        aspect=aspect, height=height, sharey="row")
    scatter_legend_data = {}
    
    def plot_facet_scatter(data, color, **kwargs):
        data = data.copy() # avoid SettingWithCopyWarning

        electrode_idx = data.electrode_idx.iloc[0]
        epoch_idxs = data.epoch_idx

        word_end = data.word_end.iloc[0]
        tg_path = data.textgrid_path.iloc[0]

        is_left = data.lexical_evidence.iloc[0] == 0

        # plot epoched response at this electrode
        plot_epochs = epochs[epoch_idxs]
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

        ax = plt.gca()
        sns.regplot(
            data=data,
            x="p_gt_phoneme",
            y="HGA",
            scatter_kws={
                "s": 60,
                "c": scatter_colors,
                "color": None, # block `color` which would be set by regplot
            },
            ax=ax,
        )
        ax.set_xlabel("Estimated P(gt phoneme)")
        ax.set_ylabel("HGA")
        ax.set_title(f"{subject} {electrode_idx + 1}, {word_end}")

        if statistic == "spearmanr":
            corr, corr_p = spearmanr(data.p_gt_phoneme, plot_epoch_data)
            control_corr, control_corr_p = spearmanr(data.resampled, plot_epoch_data)
        elif statistic == "pearsonr":
            corr, corr_p = stats.pearsonr(data.p_gt_phoneme, plot_epoch_data)
            control_corr, control_corr_p = stats.pearsonr(data.resampled, plot_epoch_data)

        ax.text(0.05, 0.95, f"$r$ = {corr:.2f} ($p$ = {corr_p:.2g})\ncontrol $r$ = {control_corr:.2f} ($p$ = {control_corr_p:.2g})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

        return ax

    g_scatter.map_dataframe(plot_facet_scatter)
    g_scatter.add_legend(scatter_legend_data,
                         label_order=sorted(scatter_legend_data.keys()),
                         title="Stimulus step")
    
    return g_scatter


def plot_causal4_evoked(epochs: mne.Epochs,
                        plot_meta_df: pd.DataFrame,
                        subject: str,
                        population_A_window: tuple[int, int],
                        population_B_window: tuple[int, int],
                        height=3, aspect=3,
                        highlight_A_span=False,
                        highlight_B_span=True,
                        hue: Literal["resampled", "p_gt_phoneme_bin_center"] = "p_gt_phoneme_bin_center",
                        smoke_test=False):
    """
    Plot evoked responses for each electrode in population B,
    grouped by P(gt phoneme) from population A or by stimulus step.
    """
    g = sns.FacetGrid(
        plot_meta_df,
        row="electrode_idx",
        hue=hue, palette="plasma",
        col="lexical_evidence",
        aspect=aspect, height=height, sharey="row")

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
        plot_epochs = epochs[epoch_idxs]
        plot_epoch_data = plot_epochs.copy().pick(electrode_idx).get_data().squeeze(1)
        assert plot_epoch_data.ndim == 2  # n_trials * n_times

        if highlight_A_span:
            ax.axvspan(*plot_epochs.times[list(population_A_window)], color="red", alpha=0.05)
        if highlight_B_span:
            ax.axvspan(*plot_epochs.times[list(population_B_window)], color="gray", alpha=0.05)

        plot_times = plot_epochs.times
        plot_epoch_data_mean = plot_epoch_data.mean(0)
        plot_epoch_data_sem = plot_epoch_data.std(0) / np.sqrt(plot_epoch_data.shape[0])
        ax.plot(plot_times, plot_epoch_data_mean, color=color, alpha=0.5, **kwargs)

        if not smoke_test:
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
    g.fig.suptitle(f"Evoked responses by {hue_label}", y=1.1)

    return g


def plot_causal4_raster(epochs, plot_meta_df, subject, population_B_window,
                        sort_by: Literal["p_gt_phoneme", "resampled", "behavior_linear"] = "p_gt_phoneme",
                        rasterized=True,
                        height=3, aspect=3,
                        clip_zscore=(-2, 2),
                        plot_extremes=False,
                        parameter_cache=None,
                        cbar=True,
                        smoke_test=False):
    g_raster = sns.FacetGrid(
        plot_meta_df,
        row="electrode_idx",
        col="lexical_evidence",
        aspect=aspect, height=height)

    sort_ascending = False if sort_by == "p_gt_phoneme" else True

    parameter_cache = parameter_cache or {}
    
    def plot_facet_raster(data, color, sort_by="p_gt_phoneme",
                          clip_zscore=(-3, 3),
                          rasterized=True, **kwargs):
        if sort_by not in ["p_gt_phoneme", "resampled", "behavior_linear"]:
            raise NotImplementedError(f"Sorting by {sort_by} is not implemented. "
                                      "Use 'p_gt_phoneme', 'resampled', or 'behavior_linear'.")
        # re-sort according to sort_by
        data = data.sort_values(sort_by, ascending=sort_ascending)

        if smoke_test:
            data = data.sample(10)

        electrode_idx = data.electrode_idx.iloc[0]
        epoch_idxs = data.epoch_idx

        word_end = data.word_end.iloc[0]
        phoneme_pair = data.phoneme_pair.iloc[0]
        gt_phoneme = data.label_lexical.iloc[0]
        tg_path = data.textgrid_path.iloc[0]
        tg = textgrid.TextGrid.fromFile(str(tg_path))

        ## prepare epoch data

        # plot epoched response at this electrode
        plot_epochs = epochs[epoch_idxs]
        plot_epoch_data = plot_epochs.copy().pick(electrode_idx).get_data().squeeze(1)
        assert plot_epoch_data.ndim == 2

        # get extreme-left options
        plot_extreme_left_idxs = epochs.metadata.query("phoneme_pair == @phoneme_pair and resampled == 1") \
            .sort_values("lexical_evidence").index
        plot_extreme_left_epochs = epochs[plot_extreme_left_idxs]
        # get extreme-right options
        plot_extreme_right_idxs = epochs.metadata.query("phoneme_pair == @phoneme_pair and resampled == 6") \
            .sort_values("lexical_evidence").index
        plot_extreme_right_epochs = epochs[plot_extreme_right_idxs]
    
        plot_extreme_left_data = plot_extreme_left_epochs.copy().pick(electrode_idx).get_data().squeeze(1)
        plot_extreme_right_data = plot_extreme_right_epochs.copy().pick(electrode_idx).get_data().squeeze(1)

        # z-score across trials and time points
        if (subject, electrode_idx) not in parameter_cache:
            # cache mean and std for z-scoring
            all_epoch_data = epochs.copy().pick(electrode_idx).get_data().squeeze(1)
            hga_mean = all_epoch_data.flatten().mean()
            hga_std = all_epoch_data.flatten().std()
            parameter_cache[subject, electrode_idx] = (hga_mean, hga_std)
        else:
            hga_mean, hga_std = parameter_cache[subject, electrode_idx]

        plot_epoch_data = (plot_epoch_data - hga_mean) / hga_std
        plot_extreme_left_data = (plot_extreme_left_data - hga_mean) / hga_std
        plot_extreme_right_data = (plot_extreme_right_data - hga_mean) / hga_std

        # clip extreme values to avoid outliers
        plot_epoch_data = np.clip(plot_epoch_data, clip_zscore[0], clip_zscore[1])
        plot_extreme_left_data = np.clip(plot_extreme_left_data, clip_zscore[0], clip_zscore[1])
        plot_extreme_right_data = np.clip(plot_extreme_right_data, clip_zscore[0], clip_zscore[1])

        # compute x-positions for extreme top, bottom values
        if sort_by == "p_gt_phoneme":
            extreme_keys = [0, 1]
        elif sort_by == "resampled":
            extreme_keys = [1, 6]
        elif sort_by == "behavior_linear":
            extreme_keys = [-1, 1]
        extreme_bottom, extreme_top = extreme_keys

        # now figure out how to order the extremes
        if sort_by == "p_gt_phoneme":
            gt_is_extreme_left = gt_phoneme == phoneme_pair[0]
            if gt_is_extreme_left:
                top_data, bottom_data = plot_extreme_left_data, plot_extreme_right_data
            else:
                top_data, bottom_data = plot_extreme_right_data, plot_extreme_left_data
        else:
            # doesn't change as a function of gt phoneme
            top_data, bottom_data = plot_extreme_left_data, plot_extreme_right_data

        # grab the existing Axes
        ax = plt.gca()
        # create a divider for the existing axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        # append an axes on the left, 15% width of the heatmap ax
        ax_line = divider.append_axes("left", size="15%", pad=0.05)

        # PLOT THE LINE (stacked left)
        # y = trial index (0..n-1), x = sort_by value
        offset = 0

        if plot_extremes:
            # plot top extreme
            ax_line.plot(np.repeat(extreme_top, len(top_data)),
                         np.arange(offset, offset + len(top_data)),
                         linestyle="-", linewidth=6)
            offset += len(top_data)
            ax_line.axhline(offset, color="gray", linestyle="--", linewidth=2)
            offset += 1  # leave a gap for the line

        # plot unseen data
        ax_line.plot(data[sort_by], np.arange(offset, offset + len(data)), linestyle='-', linewidth=4)
        offset += len(data)

        if plot_extremes:
            # plot bottom extreme
            ax_line.axhline(offset, color="gray", linestyle="--", linewidth=2)
            # leave a gap for the line
            offset += 1
            ax_line.plot(np.repeat(extreme_bottom, len(bottom_data)),
                         np.arange(offset, offset + len(bottom_data)),
                         linestyle="-", linewidth=6)

        ax_line.invert_yaxis()            # so trial 0 is at top (same as heatmap)
        ax_line.xaxis.set_label_position("top")
        ax_line.xaxis.tick_top()
        ax_line.yaxis.set_visible(False)
        ax_line.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax_line.spines['right'].set_visible(True)
        ax_line.spines['top'].set_visible(False)
        ax_line.spines['left'].set_visible(False)
        ax_line.spines['bottom'].set_visible(False)
        ax_line.set_xticklabels([])
        ax_line.margins(y=0)

        assert data.label_lexical.nunique() == 1
        if sort_by == "p_gt_phoneme":
            ax_line.set_xlabel(f"P(/{gt_phoneme}/)")
            ax_line.set_xlim(0, 1)
            ax_line.set_xticks(np.linspace(0, 1, 5))
            xticklabels = ["" for _ in range(5)]
            xticklabels[0] = "0"
            xticklabels[-1] = "1"
            ax_line.set_xticklabels(xticklabels)
        elif sort_by == "resampled":
            ax_line.set_xlabel("Stimulus step")
            ax_line.set_xlim(data.resampled.min() - 1, data.resampled.max() + 1)
        elif sort_by == "behavior_linear":
            ax_line.set_xlabel("Slider position")
            ax_line.set_xlim(-1, 1)

        ax.set_xlabel("Time since word onset (sec)")
        ax.set_ylabel("Trial")
        ax.set_title(f"{subject} {electrode_idx + 1}, {word_end}")

        if plot_extremes:
            plot_all_data = np.concatenate([
                top_data,
                # leave a gap for the line
                np.full((1, plot_epoch_data.shape[1]), np.nan),
                plot_epoch_data,
                # leave a gap for the line
                np.full((1, plot_epoch_data.shape[1]), np.nan),
                bottom_data,
            ])
        else:
            plot_all_data = plot_epoch_data

        ax = sns.heatmap(
            plot_all_data,
            cmap="plasma", cbar=False, ax=ax,
            yticklabels=False,
            rasterized=rasterized,
        )

        if plot_extremes:
            # add hlines separating extreme data
            ax.axhline(len(top_data), color="gray", linestyle="--", linewidth=2)
            ax.axhline(len(top_data) + len(plot_epoch_data), color="gray", linestyle="--", linewidth=2)

        # Plot a subsample of time points
        zero_idx = plot_epochs.time_as_index(0)[0]
        xticks = np.linspace(zero_idx, plot_all_data.shape[1] - 1,
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
    
    sort_labels = {
        "p_gt_phoneme": "P(gt phoneme)",
        "resampled": "Stimulus step",
        "behavior_linear": "Slider position"
    }
    sort_label = sort_labels.get(sort_by, "???")
    g_raster.fig.suptitle(f"Rasters sorted by {sort_label}", y=1.1)

    cax_f = None
    if cbar:
        # add colorbar
        norm = plt.Normalize(clip_zscore[0], clip_zscore[1])
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        cax_f, cax = plt.subplots(figsize=(2, 6))
        cax_f.colorbar(sm, cax=cax, label="HGA (z)",
                            ticks=np.linspace(clip_zscore[0], clip_zscore[1], 5))

    return g_raster, cax_f