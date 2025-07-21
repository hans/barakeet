from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from src.models.causal import run_phase1


@dataclass
class Causal3AnalysisResult:
    p_gt_phoneme: np.ndarray
    p_mismatch_to_gt: np.ndarray
    correlation_p: float

    phase2_test_scores: np.ndarray

    phase1_val_score: float
    """Score on the data that will be passed to phase2"""

    phase1_test_scores: np.ndarray


def stratified_nested_cv(X, Y1, Y2, metadata, idxs_val):
    # This section is more complicated because we need to do nested CV with an auxiliary
    # stratification variable (not the regressor but a metadata attribute).
    # This has to be done manually.

    # Define outer and inner cross-validation
    cv_outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Define C values for logistic regression
    Cs = np.logspace(0, 3, 6)

    # Storage for results
    outer_scores_Y1 = []
    outer_models_Y1 = []
    outer_scores_Y2 = []
    outer_models_Y2 = []

    def build_pipeline(C):
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=0.9),
            LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=42,
            ),
        )

    for train_idx, test_idx in cv_outer.split(X, metadata.iloc[idxs_val].resampled.values):
        # Split data for outer CV
        X_train, X_test = X[train_idx], X[test_idx]
        Y1_train, Y1_test = Y1[train_idx], Y1[test_idx]
        Y2_train, Y2_test = Y2[train_idx], Y2[test_idx]

        # Inner CV for hyperparameter tuning on 1
        best_alpha1 = None
        best_score1 = -np.inf
        for train_inner_idx, val_inner_idx in cv_inner.split(X_train, metadata.iloc[idxs_val[train_idx]].resampled.values):
            X_train_inner, X_val_inner = X_train[train_inner_idx], X_train[val_inner_idx]
            Y1_train_inner, Y_val_inner = Y1_train[train_inner_idx], Y1_train[val_inner_idx]

            for C in Cs:
                # Train logistic regression on inner loop
                model1_inner = build_pipeline(C)
                model1_inner.fit(X_train_inner, Y1_train_inner)
                
                # Evaluate performance on validation set
                score1 = model1_inner.score(X_val_inner, Y_val_inner)
                
                # Store the best performing alpha
                if score1 > best_score1:
                    best_score1 = score1
                    best_alpha1 = C

        # Train final model on full training set with best alpha
        model1_final = build_pipeline(best_alpha1)
        model1_final.fit(X_train, Y1_train)
        score_final = model1_final.score(X_test, Y1_test)

        ## 

        # Inner CV for hyperparameter tuning on 2
        best_alpha2 = None
        best_score2 = -np.inf
        for train_inner_idx, val_inner_idx in cv_inner.split(X_train, metadata.iloc[idxs_val[train_idx]].resampled.values):
            X_train_inner, X_val_inner = X_train[train_inner_idx], X_train[val_inner_idx]
            Y2_train_inner, Y_val_inner = Y2_train[train_inner_idx], Y2_train[val_inner_idx]

            for C in Cs:
                # Train logistic regression on inner loop
                model2_inner = build_pipeline(C)
                model2_inner.fit(X_train_inner, Y2_train_inner)
                
                # Evaluate performance on validation set
                score2 = model2_inner.score(X_val_inner, Y_val_inner)
                
                # Store the best performing alpha
                if score2 > best_score2:
                    best_score2 = score2
                    best_alpha2 = C

        # Train final model on full training set with best alpha
        model2_final = build_pipeline(best_alpha2)
        model2_final.fit(X_train, Y2_train)
        score_final2 = model2_final.score(X_test, Y2_test)

        # Store results
        outer_scores_Y1.append(score_final)
        outer_models_Y1.append(model1_final)
        outer_scores_Y2.append(score_final2)
        outer_models_Y2.append(model2_final)

    # Convert results to match cross_validate output format
    fitted2_1 = {
        'test_score': np.array(outer_scores_Y1),
        'estimator': np.array(outer_models_Y1),
        'num_outputs': Y1.shape[1] if Y1.ndim > 1 else 1,
    }

    fitted2_2 = {
        'test_score': np.array(outer_scores_Y2),
        'estimator': np.array(outer_models_Y2),
        'num_outputs': Y2.shape[1] if Y2.ndim > 1 else 1,
    }

    return fitted2_1, fitted2_2


def run_phase2(epochs, subject, phoneme_pair,
                population_B, population_B_window,
                metadata,
                idxs_val, y_proba_val):
    """
    Run the second phase of the causal analysis.
    """
    # Train a mismatch decoder on the same set
    idxs_train = np.array(list(set(np.arange(len(metadata))) - set(idxs_val)))
    # TODO might want to try mismatch direction
    X = epochs[subject][f"phoneme_pair == '{phoneme_pair}'"] \
        .pick(population_B) \
        .copy().crop(*population_B_window) \
        .get_data()
    X = X.reshape(len(X), -1)
    X_train = X[idxs_train]
    y_train = metadata.iloc[idxs_train].mismatch_left_right.values
    X_val = X[idxs_val]
    y_val = metadata.iloc[idxs_val].mismatch_left_right.values

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
    assert list(model.classes_) == [-1, 0, 1]

    # Retain only trials with mismatch
    keep_idxs = [idx for idx, idx_is_mismatch in zip(idxs_val, metadata.iloc[idxs_val].mismatch)
                 if idx_is_mismatch == 1]

    p_gt_phoneme = y_proba_val[np.arange(len(keep_idxs)),
                               metadata.iloc[keep_idxs].lexical_evidence]
    p_mismatch_to_gt = model.predict_proba(X_val) \
        [np.arange(len(keep_idxs)),
         metadata.iloc[keep_idxs].mismatch_left_right + 1]
    
    return test_scores, p_gt_phoneme, p_mismatch_to_gt, \
        np.corrcoef(p_gt_phoneme, p_mismatch_to_gt)[0, 1]


def run_causal3_analysis(epochs, subject, phoneme_pair,
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

    test_scores, p_gt_phoneme, p_mismatch_to_gt, correlation_p = run_phase2(
        epochs, subject, phoneme_pair,
        population_B, population_B_window,
        metadata,
        idxs_val, y_proba_val
    )


    return Causal3AnalysisResult(
        p_gt_phoneme=p_gt_phoneme,
        p_mismatch_to_gt=p_mismatch_to_gt,
        correlation_p=correlation_p,
        phase2_test_scores=test_scores,
        phase1_val_score=roc_auc,
        phase1_test_scores=fitted['test_score'],
    )