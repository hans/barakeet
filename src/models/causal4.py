from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from src.models.causal import run_phase1


@dataclass
class Causal4AnalysisResult:
    subject: str
    population_A: list[int]
    population_A_window: tuple[float, float]
    population_B: list[int]
    population_B_window: tuple[float, float]

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
    p_gt_lexical_evidence: np.ndarray
    """
    Test output: probability of the ground-truth lexical evidence
    computed from population B.
    """

    phase1_val_score: float
    """Score on the data that will be passed to phase2"""

    phase1_test_scores: np.ndarray
    phase2_test_scores: np.ndarray

    # post-init validation checks
    def __post_init__(self):
        assert len(self.p_gt_phoneme) == len(self.test_trial_metadata)
        assert len(self.p_gt_lexical_evidence) == len(self.test_trial_metadata)


def run_phase2(epochs, subject, phoneme_pair,
                population_B, population_B_window,
                metadata,
                idxs_val, y_proba_val):
    """
    Run the second phase of the causal analysis.
    """
    # Train a lexical evidence decoder on the same set
    idxs_train = np.array(list(set(np.arange(len(metadata))) - set(idxs_val)))
    X = epochs[subject][f"phoneme_pair == '{phoneme_pair}'"] \
        .pick(population_B) \
        .copy().crop(*population_B_window) \
        .get_data()
    X = X.reshape(len(X), -1)
    X_train = X[idxs_train]
    y_train = metadata.iloc[idxs_train].lexical_evidence.values
    X_val = X[idxs_val]
    y_val = metadata.iloc[idxs_val].lexical_evidence.values

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.9),
        LogisticRegressionCV(Cs=5, max_iter=1000, cv=StratifiedKFold(2, shuffle=True),
                             class_weight="balanced"),
    )
    test_scores = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(2, shuffle=True))
    model = pipeline.fit(X_train, y_train)
    assert list(model.classes_) == [0, 1]

    # # Retain only trials with mismatch
    # keep_idxs = [idx for idx, idx_is_mismatch in zip(idxs_val, metadata.iloc[idxs_val].mismatch)
    #              if idx_is_mismatch == 1]
    keep_idxs = np.arange(len(idxs_val))

    p_gt_phoneme = y_proba_val[np.arange(len(keep_idxs)),
                               metadata.iloc[keep_idxs].lexical_evidence]
    p_gt_lexical_evidence = model.predict_proba(X_val) \
        [np.arange(len(keep_idxs)),
         metadata.iloc[keep_idxs].lexical_evidence]

    return test_scores, p_gt_phoneme, p_gt_lexical_evidence


def run_causal4_analysis(epochs, subject, phoneme_pair,
                            population_A, population_A_window,
                            population_B, population_B_window,
                            split_strategy="stratify",
                            pca_num_components=32):
    # phase1
    fitted, metadata, idxs_val, y_proba_val = run_phase1(
        epochs, subject, phoneme_pair,
        population_A, population_A_window,
        split_strategy=split_strategy,
    )
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(
        metadata.iloc[idxs_val].categorical_acoustic_cue.combine(0, max),
        y_proba_val[:, 1],
    )

    test_scores, p_gt_phoneme, p_gt_lexical_evidence = run_phase2(
        epochs, subject, phoneme_pair,
        population_B, population_B_window,
        metadata,
        idxs_val, y_proba_val
    )

    return Causal4AnalysisResult(
        subject=subject,
        population_A=population_A,
        population_A_window=population_A_window,
        population_B=population_B,
        population_B_window=population_B_window,

        epochs=epochs[subject],
        test_trial_metadata=metadata.iloc[idxs_val],

        p_gt_phoneme=p_gt_phoneme,
        p_gt_lexical_evidence=p_gt_lexical_evidence,
        phase1_val_score=roc_auc,
        phase1_test_scores=fitted['test_score'],
        phase2_test_scores=test_scores,
    )