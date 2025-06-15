from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
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