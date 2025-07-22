
import itertools
from typing import Literal, Optional, cast, TypeAlias

import mne
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import check_scoring, make_scorer
from tqdm.auto import tqdm


DecoderFitKey: TypeAlias = tuple[str, int, str, int, int]  # (subject, electrode_idx, phoneme_pair, smin, smax)


def run_decoding_analysis_single_electrode(
        epochs, electrode_df,
        stride: int, window_size: int,
        global_min_sample: int = 0,
        global_max_sample: Optional[int] = None,
        target: Literal["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right"] = "lexical_evidence",
        strategy: Literal["nested-cv", "train-test"] = "nested-cv",
        filter_speech_responsive=True,
        return_outcomes=True,
        include_only_full_windows=True,
        smoke_test=False,
        randomize=False
        ) -> tuple[dict[DecoderFitKey, dict[str, float]],
                   dict[DecoderFitKey, dict[str, float]],
                   dict[DecoderFitKey, pd.DataFrame],
                   dict[DecoderFitKey, list[BaseEstimator]]]:
    """
    stride: in samples
    window_size: in samples
    """

    if target not in ["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right"]:
        raise ValueError(f"Invalid target {target}")
    if strategy not in ["nested-cv", "train-test"]:
        raise ValueError(f"Invalid strategy {strategy}")
    
    if global_max_sample is not None:
        assert global_max_sample > global_min_sample, \
            f"global_max_sample ({global_max_sample}) must be greater than global_min_sample ({global_min_sample})"

    data_max_sample = min([epoch.get_data().shape[2] for epoch in epochs.values()])
    if global_max_sample is None:
        global_max_sample = cast(int, data_max_sample)
    else:
        global_max_sample = min(global_max_sample, data_max_sample)

    if global_max_sample - global_min_sample < window_size:
        raise ValueError(f"Window size ({window_size}) is larger than the available data range "
                         f"({global_max_sample - global_min_sample}). Please adjust the parameters.")

    windows_left = np.arange(global_min_sample, global_max_sample, stride)
    windows_right = windows_left + window_size
    windows = np.array(list(zip(windows_left, windows_right)))
    if include_only_full_windows:
        windows = windows[windows[:, 1] <= global_max_sample]

    # `outcomes` stores prediction outcomes for each epoch under the optimal model
    outcomes = {}
    # `test_scores` stores cross-validated estimates of held-out generalization
    train_scores, test_scores = {}, {}
    # `models` stores the fitted models
    models = {}

    phoneme_pairs = next(iter(epochs.values())).metadata.phoneme_pair.unique()

    if filter_speech_responsive:
        electrodes = electrode_df.query("speech_responsive").reset_index()
    else:
        # include all electrodes
        electrodes = electrode_df.reset_index()

    if smoke_test:
        electrodes = electrodes.iloc[:5]

    for _, row in tqdm(electrodes.iterrows(), total=len(electrodes)):
        for smin, smax in windows:
            for phoneme_pair in phoneme_pairs:
                epochs_ij = epochs[row.subject]
                # manual filtering for performance
                selection = epochs_ij.metadata.phoneme_pair == phoneme_pair

                if selection.sum() == 0:
                    continue

                # num_trials * num_times
                X = epochs_ij.get_data(picks=[row.electrode_idx])[selection][:, 0, smin:smax]

                if target == "acoustic":
                    y = epochs_ij.metadata.categorical_acoustic_cue[selection].values
                elif target == "lexical_evidence":
                    y = (epochs_ij.metadata.word_end.str[0] == phoneme_pair[0])[selection].values
                elif target == "mismatch":
                    y = epochs_ij.metadata.mismatch[selection].values
                elif target == "mismatch_left_right":
                    y = epochs_ij.metadata.mismatch_left_right[selection].values

                    # Subset data to only include mismatch trials
                    X = X[y != 0]
                    y = y[y != 0]

                num_classes = len(set(y))
                # stratify_class = epochs_ij.metadata.stratify_class[selection].values

                if randomize:
                    # Randomize the labels
                    y = np.random.permutation(y)

                ####

                scoring = ["roc_auc", "f1_macro", "accuracy"] if num_classes == 2 else ["f1_macro", "accuracy"]

                if strategy == "nested-cv":
                    fitted = fit_nested_cv(X, y, num_classes=num_classes, scoring=scoring)
                elif strategy == "train-test":
                    fitted = fit_train_test(X, y, num_classes=num_classes, scoring=scoring,
                                            num_repeats=5)

                result_key = (row.subject, row.electrode_idx, phoneme_pair, smin, smax)

                if isinstance(scoring, list):
                    train_scores[result_key] = {k: fitted["train_" + k] for k in scoring}
                    test_scores[result_key] = {k: fitted["test_" + k] for k in scoring}
                else:
                    train_scores[result_key] = fitted["train_score"]
                    test_scores[result_key] = fitted["test_score"]

                if return_outcomes:
                    # only store outcomes on test folds
                    fold_results = []
                    for fold, (test_idxs, estimator) in enumerate(zip(fitted["test_idxs"], fitted["estimator"])):
                        # test_idxs are indices into X, y, which themselves are indices into epochs_ij[selection]
                        test_epoch_idxs = epochs_ij.metadata.index[selection][test_idxs]
                        fold_results.append(pd.DataFrame({
                            "decoder_target": y[test_idxs],
                            "decoder_prediction": estimator.predict(X[test_idxs]),
                            "decoder_proba": estimator.predict_proba(X[test_idxs])[:, 1],
                            "fold": fold,
                            "epoch_idx": test_epoch_idxs,
                        }))

                    outcomes[result_key] = pd.concat(fold_results)

                models[result_key] = fitted["estimator"]

    return train_scores, test_scores, outcomes, models


def fit_nested_cv(X, y, num_classes: int, scoring: list[str],
                  num_outer_folds=2, num_inner_folds=2, random_state=42):
    """
    Fit a nested cross-validation model with logistic regression.
    Returns a fitted model and cross-validation results.
    """
    cv_inner = StratifiedKFold(num_inner_folds, shuffle=True, random_state=random_state)
    cv_outer = StratifiedKFold(num_outer_folds, shuffle=True, random_state=random_state)

    Cs = np.logspace(-3, 2, 6)

    pipeline: list[BaseEstimator] = [StandardScaler()]

    solver = "liblinear" if num_classes == 2 else "saga"
    pipeline.append(LogisticRegressionCV(
        Cs=Cs, cv=cv_inner, max_iter=100000,
        class_weight="balanced", fit_intercept=False,
        solver=solver))
    model = make_pipeline(*pipeline)
    fitted = cross_validate(model, X, y, cv=cv_outer, scoring=scoring,
                            return_estimator=True,
                            return_train_score=True)
    
    # add information about the item idxs in each train/test fold
    fitted["train_idxs"] = []
    fitted["test_idxs"] = []
    for train_idxs, test_idxs in cv_outer.split(X, y):
        fitted["train_idxs"].append(train_idxs)
        fitted["test_idxs"].append(test_idxs)

    return fitted


def fit_train_test(X, y, num_classes: int, scoring: list[str],
                   test_fraction=0.2, num_folds=3,
                   num_repeats=1, random_state=42):
    seeds = np.random.RandomState(random_state).randint(0, 10000, num_repeats)

    results = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, idxs_train, idxs_test = \
            train_test_split(X, y, np.arange(len(X)),
                            test_size=test_fraction, stratify=y, random_state=seed)

        Cs = np.logspace(-3, 2, 6)

        pipeline: list[BaseEstimator] = [StandardScaler()]
        solver = "liblinear" if num_classes == 2 else "saga"
        pipeline.append(LogisticRegressionCV(
            Cs=Cs, cv=StratifiedKFold(num_folds, shuffle=True, random_state=seed),
            max_iter=100000, class_weight="balanced", fit_intercept=False,
            solver=solver))
        model = make_pipeline(*pipeline)
        model.fit(X_train, y_train)

        # Get optimal C
        if num_classes == 2:
            best_C = model[-1].C_[0]
        else:
            best_C = model[-1].C_

        # re-fit on whole training set
        refit_model = make_pipeline(*pipeline[:-1], LogisticRegression(
            C=best_C, class_weight="balanced", fit_intercept=False,
            solver=solver))
        refit_model.fit(X_train, y_train)

        scorer = check_scoring(refit_model, scoring=scoring)

        train_scores = scorer(refit_model, X_train, y_train)
        test_scores = scorer(refit_model, X_test, y_test)
        results.append({
            **{f"train_{k}": np.array([v]) for k, v in train_scores.items()},
            **{f"test_{k}": np.array([v]) for k, v in test_scores.items()},
            "train_idxs": [idxs_train],
            "test_idxs": [idxs_test],
            "estimator": [refit_model]
        })

    # Concatenate results from all repeats
    fitted = {k: np.concatenate([r[k] for r in results]) if isinstance(results[0][k], np.ndarray)
              else list(itertools.chain.from_iterable(r[k] for r in results))
              for k in results[0].keys()}
    return fitted


def get_ensemble_predictions(model_key: DecoderFitKey,
                             models: list[BaseEstimator],
                             epochs: dict[str, mne.Epochs]) -> pd.DataFrame:
    """
    Get predictions from an ensemble of fit models on held-out epochs,
    subsetting appropriately to match the properties of the data the model
    was fit on.

    Args:
        model_key: tuple of `(subject, electrode_idx, phoneme_pair, smin, smax)`;
            i.e. the keys of the dictionary returned by `run_decoding_analysis_single_electrode`
        models: list of fitted models
        epochs: mne.Epochs object containing the held-out epochs
    """
    subject, electrode_idx, phoneme_pair, smin, smax = model_key

    epochs_ij = epochs[subject]
    selection = epochs_ij.metadata.phoneme_pair == phoneme_pair
    if selection.sum() == 0:
        raise ValueError(f"No epochs found for subject {subject}, "
                         f"phoneme pair {phoneme_pair} in the given epochs.")
    
    X = epochs_ij.get_data(picks=[electrode_idx])[selection][:, 0, smin:smax]

    outcomes = []
    for i, model in enumerate(models):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        outcomes.append(pd.DataFrame({
            "epoch_idx": epochs_ij.metadata.index[selection].values,
            "decoder_target": epochs_ij.metadata.categorical_acoustic_cue[selection].values,
            "decoder_prediction": y_pred,
            "decoder_proba": y_proba,
            "fold": i
        }))

    return pd.concat(outcomes)