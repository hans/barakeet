from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
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
        left=True):
    A_subject, A_phoneme_pair, A_population_A = A_source
    B_subject, B_phoneme_pair, B_population_A = B_source

    assert A_phoneme_pair == B_phoneme_pair, "Phoneme pairs must match for counterfactual evaluation."

    A_activations = searchlight_electrode_activations[A_subject, A_phoneme_pair, A_population_A, B_window_start, B_window_end]
    B_activations = searchlight_electrode_activations[B_subject, B_phoneme_pair, B_population_A, B_window_start, B_window_end]

    A_metadata = A_activations["metadata"].copy()
    B_metadata = B_activations["metadata"]
    A_mask_left = A_activations["mask_left"]
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

    if len(A_metadata) != len(B_metadata):
        # resample A to have the same number of trials as B
        B_grouped = B_metadata.groupby(["resampled", "lexical_evidence"])
        draw_A_idxs = A_metadata.groupby(["resampled", "lexical_evidence"], as_index=False).apply(
                lambda xs: xs.iloc[np.random.choice(len(xs),
                                                    size=len(B_grouped.groups[xs.name]),
                                                    replace=len(B_grouped.groups[xs.name]) > len(xs))]) \
            .reset_index(level=0, drop=True).sort_values(["resampled", "lexical_evidence"]).index
        A_metadata["local_idx"] = np.arange(len(A_metadata))
        A_metadata = A_metadata.loc[draw_A_idxs]
        A_p_gt_phoneme = A_p_gt_phoneme[A_metadata.local_idx]
    else:
        # resort A
        A_metadata = A_metadata.reset_index().sort_values(["resampled", "lexical_evidence"])
        A_p_gt_phoneme = A_p_gt_phoneme[A_metadata.index]
    
    # align
    B_resorted = B_metadata.reset_index().sort_values(["resampled", "lexical_evidence"]).index
    B_response = B_response[B_resorted]

    return spearmanr(A_p_gt_phoneme, B_response)


def counterfactual_baseline(subject, phoneme_pair, population_A, electrode_idx, window_start, window_end,
                            searchlight_decoder_outputs, searchlight_electrode_activations,
                            left=True,
                            sanity_check=False):
    if sanity_check:
        # DEV sanity check: only use the same subject and phoneme pair
        control_alternative_keys = [(subject_alt, phoneme_pair_alt, population_A_alt) for subject_alt, phoneme_pair_alt, population_A_alt in searchlight_decoder_outputs.keys()
                                    if subject_alt == subject and phoneme_pair_alt == phoneme_pair]
    else:
        control_alternative_keys = [(subject_alt, phoneme_pair_alt, population_A_alt) for subject_alt, phoneme_pair_alt, population_A_alt in searchlight_decoder_outputs.keys()
                                    if subject_alt != subject and phoneme_pair_alt == phoneme_pair]

    ret = []
    for key in control_alternative_keys:
        try:
            val = evaluate_counterfactual_correlation(
                (subject, phoneme_pair, population_A),
                key,
                electrode_idx,
                window_start,
                window_end,
                searchlight_decoder_outputs,
                searchlight_electrode_activations,
                left=left
            )
            ret.append((key, val))
        except Exception as e:
            raise
            # print(f"Error evaluating counterfactual for {key}: {e}")
            # continue
    return ret