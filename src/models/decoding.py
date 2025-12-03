
import itertools
from typing import Literal, Optional, cast, TypeAlias, Protocol

from loguru import logger as L
import mne
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import check_scoring, make_scorer, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics._scorer import _MultimetricScorer, _check_multimetric_scoring
from tqdm.auto import tqdm
from sklearn.decomposition import PCA


DecoderFitKey: TypeAlias = tuple[str, int, str, int, int]  # (subject, electrode_idx, phoneme_pair, smin, smax)
"""Result of a single electrode decoder analysis."""

PopulationDecoderFitKey: TypeAlias = tuple[str, str, str, int, int]  # (subject, population_name, phoneme_pair, smin, smax)
"""Result of a population decoder analysis"""

Epochs: TypeAlias = mne.Epochs | mne.epochs.EpochsFIF


class ClassifierLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def _prepare_decoding_population(
        epochs_i: Epochs,
        electrode_idxs: list[int],
        phoneme_pair: str,
        stride: int, window_size: int,
        global_min_sample: int = 0,
        global_max_sample: Optional[int] = None,
        target: Literal["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right", "behavior_categorical"] = "lexical_evidence",
        groupby: Optional[list[str]] = None,
        filter: Optional[str] = None,
        include_only_full_windows=True,
        randomize=False):
    """
    Prepare windowed decoding inputs/targets for the given parameters.

    Args:
        groupby: Yield separate samples for each combination of these grouping variables.
        filter: Optional filter string on epoch metadata, passed to pd.DataFrame.query.

    Yields tuples for each window of form:
        - name: tuple of grouping values, same length as `groupby`.
            Empty tuple if `groupby` is None.
        - smin: start sample of window
        - smax: end sample of window
        - selection: boolean array indicating selected epochs in `epochs_i`
        - X_window: windowed input data
        - y: target labels
    """

    if target not in ["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right", "behavior_categorical"]:
        raise ValueError(f"Invalid target {target}")
    assert epochs_i.metadata is not None
    
    if global_max_sample is not None:
        assert global_max_sample > global_min_sample, \
            f"global_max_sample ({global_max_sample}) must be greater than global_min_sample ({global_min_sample})"

    data_max_sample = epochs_i.get_data().shape[2]
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

    selection = epochs_i.metadata.phoneme_pair == phoneme_pair
    if selection.sum() == 0:
        raise ValueError(f"No epochs found for phoneme pair {phoneme_pair} in the given epochs.")
    
    X = epochs_i.get_data(picks=electrode_idxs)

    md = epochs_i.metadata
    if filter is not None:
        selection = selection & md.eval(filter)

    grouper = md.groupby(groupby) if groupby is not None else [((), md)]

    for name, metadata_subset in grouper:
        selection_i = selection & md.index.isin(metadata_subset.index)
        if selection_i.sum() == 0:
            continue

        for smin, smax in windows:
            # num_trials * num_electrodes * num_times
            X_window = X[selection_i][:, :, smin:smax]
            # flatten space * time
            X_window = X_window.reshape(X_window.shape[0], -1)

            if target == "acoustic":
                y = md.categorical_acoustic_cue[selection_i].values
            elif target == "lexical_evidence":
                y = (md.word_end.str[0] == phoneme_pair[0])[selection_i].values
            elif target == "mismatch":
                y = md.mismatch[selection_i].values
            elif target == "mismatch_left_right":
                y = md.mismatch_left_right[selection_i].values

                # Subset data to only include mismatch trials
                X_window = X_window[y != 0]
                y = y[y != 0]
            elif target == "behavior_categorical":
                y = md.behavior_dummy_forced[selection_i].values

            
            # stratify_class = epochs_ij.metadata.stratify_class[selection_i].values

            if randomize:
                # Randomize the labels
                y = np.random.permutation(y)

            yield name, smin, smax, selection_i, X_window, y



def run_decoding_population(
        epochs_i: Epochs,
        electrode_idxs: list[int],
        phoneme_pair: str,
        subject: str,
        population_name: str,
        stride: int, window_size: int,
        global_min_sample: int = 0,
        global_max_sample: Optional[int] = None,
        target: Literal["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right", "behavior_categorical"] = "lexical_evidence",
        strategy: Literal["nested-cv", "train-test"] = "nested-cv",
        groupby: Optional[list[str]] = None,
        pca_num_components: Optional[float] = None,
        return_outcomes=True,
        include_only_full_windows=True,
        smoke_test=False,
        randomize=False):

    assert epochs_i.metadata is not None
    _gen = _prepare_decoding_population(
        epochs_i=epochs_i,
        electrode_idxs=electrode_idxs,
        phoneme_pair=phoneme_pair,
        stride=stride,
        window_size=window_size,
        global_min_sample=global_min_sample,
        global_max_sample=global_max_sample,
        target=target,
        groupby=groupby,
        include_only_full_windows=include_only_full_windows,
        randomize=randomize
    )

    # `outcomes` stores prediction outcomes for each epoch under the optimal model
    outcomes = {}
    # `test_scores` stores cross-validated estimates of held-out generalization
    train_scores, test_scores = {}, {}
    # `models` stores the fitted models
    models = {}

    for name, smin, smax, selection, X_window, y in _gen:

        ####

        num_classes = len(set(y))
        scoring = ["roc_auc", "f1_macro", "accuracy"] if num_classes == 2 else ["f1_macro", "accuracy"]

        if strategy == "nested-cv":
            fitted = fit_nested_cv(X_window, y, num_classes=num_classes,
                                   pca_num_components=pca_num_components,
                                   scoring=scoring)
        elif strategy == "train-test":
            fitted = fit_train_test(X_window, y, num_classes=num_classes,
                                    pca_num_components=pca_num_components,
                                    scoring=scoring,
                                    num_repeats=5)

        result_key = (subject, population_name, phoneme_pair, name, smin, smax)

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
                test_epoch_idxs = epochs_i.metadata.index[selection][test_idxs]
                fold_results.append(pd.DataFrame({
                    "decoder_target": y[test_idxs],
                    "decoder_prediction": estimator.predict(X_window[test_idxs]),
                    "decoder_proba": estimator.predict_proba(X_window[test_idxs])[:, 1],
                    "fold": fold,
                    "epoch_idx": test_epoch_idxs,
                }))

            outcomes[result_key] = pd.concat(fold_results)

        models[result_key] = fitted["estimator"]

    return train_scores, test_scores, outcomes, models


def run_decoding_model_comparison_population(
        epochs_i: Epochs,
        electrode_idxs: list[int],
        phoneme_pair: str,
        subject: str,
        population_name: str,
        stride: int, window_size: int,
        baseline_features: list[str],
        global_min_sample: int = 0,
        global_max_sample: Optional[int] = None,
        target: Literal["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right", "behavior_categorical"] = "lexical_evidence",
        strategy: Literal["nested-cv", "train-test"] = "nested-cv",
        groupby: Optional[list[str]] = None,
        filter: Optional[str] = None,
        stratify: tuple[str, ...] = ("resampled", "lexical_evidence"),
        pca_num_components: Optional[float | Literal["auto"]] = None,
        include_only_full_windows=True,
        return_estimators=False,
        n_jobs=None,
        smoke_test=False,
        randomize=False):
    """
    Run a model comparison evaluating target prediction using either
    `baseline_features` (indexing into the epoch metadata) or the
    combination of `baseline_features` plus ECoG data.
    """

    assert epochs_i.metadata is not None
    _gen = _prepare_decoding_population(
        epochs_i=epochs_i,
        electrode_idxs=electrode_idxs,
        phoneme_pair=phoneme_pair,
        stride=stride,
        window_size=window_size,
        global_min_sample=global_min_sample,
        global_max_sample=global_max_sample,
        target=target,
        groupby=groupby,
        filter=filter,
        include_only_full_windows=include_only_full_windows,
        randomize=randomize
    )

    seed = 42
    results = []

    def _fit(X, y, num_classes, stratify, random_state,
             reg_range, reg_grid_size,
             pca_dimensions=None):
        if strategy == "nested-cv":
            return fit_nested_cv(X, y, num_classes=num_classes,
                                 stratify=stratify,
                                 reg_range=reg_range,
                                 reg_grid_size=reg_grid_size,
                                 pca_num_components=pca_num_components,
                                 pca_dimensions=pca_dimensions,
                                 scoring=["roc_auc"], random_state=random_state)
        elif strategy == "train-test":
            return fit_train_test(X, y, num_classes=num_classes,
                                  stratify=stratify,
                                  reg_range=reg_range,
                                  reg_grid_size=reg_grid_size,
                                  pca_num_components=pca_num_components,
                                  pca_dimensions=pca_dimensions,
                                  scoring=["roc_auc"],
                                  n_jobs=n_jobs,
                                  num_repeats=5, random_state=random_state)
        else:
            raise ValueError("Unknown strategy: {}".format(strategy))

    all_estimators = {}

    for name, smin, smax, selection, X_window, y in _gen:
        num_classes = len(set(y))
        if num_classes != 2:
            L.warning(f"Skipping model comparison for {subject}, {population_name}, {phoneme_pair}, {name}, {smin}-{smax} because num_classes={num_classes} != 2")
            continue

        md = epochs_i.metadata[selection]

        # Prepare baseline features
        X_baseline = md[baseline_features].values
        X_full = np.concatenate([X_baseline, X_window], axis=1)

        # Prepare codes for stratified sampling of data
        stratify_codes = None
        if stratify is not None:
            stratify_codes = pd.factorize(md[list(stratify)].apply(tuple, axis=1))[0]

        # Fit N baseline models
        baseline_results = _fit(X_baseline, y, num_classes,
                                stratify=stratify_codes,
                                reg_range=(-1, 1),
                                reg_grid_size=2,
                                random_state=seed)
        # Fit N full models
        full_results = _fit(X_full, y, num_classes,
                            stratify=stratify_codes,
                            reg_range=(-8, 3),
                            reg_grid_size=10,
                            pca_dimensions=np.arange(X_baseline.shape[1], X_full.shape[1]),
                            random_state=seed)

        if baseline_results is None or full_results is None:
            L.warning(f"Skipping model comparison for {subject}, {population_name}, {phoneme_pair}, {name}, {smin}-{smax} because fitting failed.")
            continue

        # Because we matched seeds above, the fold of the ith baseline model
        # has the same samples as the fold of the ith full model.
        # So we will now do a paired comparison of ROC-AUC outcomes
        for fold, (baseline_test_idxs, baseline_estimator) in enumerate(zip(baseline_results["test_idxs"], baseline_results["estimator"])):
            full_test_idxs = full_results["test_idxs"][fold]
            full_estimator = full_results["estimator"][fold]

            # Just validate the assumption first
            assert full_test_idxs.tolist() == baseline_test_idxs.tolist()
            test_idxs = full_test_idxs

            baseline_proba = baseline_estimator.predict_proba(X_baseline[test_idxs])[:, 1]
            full_proba = full_estimator.predict_proba(X_full[test_idxs])[:, 1]

            baseline_prediction = baseline_estimator.predict(X_baseline[test_idxs])
            full_prediction = full_estimator.predict(X_full[test_idxs])

            # compute precision/recall
            baseline_precision, baseline_recall, _, _ = \
                precision_recall_fscore_support(y[test_idxs], baseline_prediction, average="binary")
            full_precision, full_recall, _, _ = \
                precision_recall_fscore_support(y[test_idxs], full_prediction, average="binary")

            result_i = {
                "subject": subject,
                "population": population_name,
                "phoneme_pair": phoneme_pair,
                "smin": smin,
                "smax": smax,
                "fold": fold,

                "baseline_roc_auc": roc_auc_score(y[test_idxs], baseline_proba),
                "full_roc_auc": roc_auc_score(y[test_idxs], full_proba),

                "baseline_precision": baseline_precision,
                "full_precision": full_precision,

                "baseline_recall": baseline_recall,
                "full_recall": full_recall,
            }
            for groupby_variable, value in zip(groupby or [], name):
                result_i[groupby_variable] = value

            # add hparam fits
            for param, val in baseline_estimator.best_params_.items():
                result_i["baseline_" + param] = val
            for param, val in full_estimator.best_params_.items():
                result_i["full_" + param] = val

            results.append(result_i)

            if return_estimators:
                key = (
                    subject, population_name, phoneme_pair,
                    name, smin, smax, fold
                )
                all_estimators[key] = {
                    "electrode_idxs": electrode_idxs,
                    "estimator": full_estimator,

                    "test_predictions": pd.DataFrame({
                        "decoder_target": y[test_idxs],
                        "baseline_decoder_prediction": baseline_prediction,
                        "baseline_decoder_proba": baseline_proba,
                        "full_decoder_prediction": full_prediction,
                        "full_decoder_proba": full_proba,
                        "fold": fold,
                        "epoch_idx": md.iloc[test_idxs].index.values,
                    }),
                }

    if return_estimators:
        return pd.DataFrame(results), all_estimators
    else:
        return pd.DataFrame(results)


def run_decoding_searchlight_single_electrode(
        epochs, electrode_df,
        stride: int, window_size: int,
        global_min_sample: int = 0,
        global_max_sample: Optional[int] = None,
        target: Literal["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right", "behavior_categorical"] = "lexical_evidence",
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

    if target not in ["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right", "behavior_categorical"]:
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
                elif target == "behavior_categorical":
                    y = epochs_ij.metadata.behavior_dummy_forced[selection].values

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
                    fitted = fit_train_test_old(X, y, num_classes=num_classes, scoring=scoring,
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
                  stratify: Optional[np.ndarray] = None,
                  num_outer_folds=2, num_inner_folds=2,
                  pca_num_components: Optional[float] = None,
                  random_state=42):
    """
    Fit a nested cross-validation model with logistic regression.
    Returns a fitted model and cross-validation results.
    """
    if stratify is not None:
        raise NotImplementedError("Custom stratified nested CV not implemented yet.")

    cv_inner = StratifiedKFold(num_inner_folds, shuffle=True, random_state=random_state)
    cv_outer = StratifiedKFold(num_outer_folds, shuffle=True, random_state=random_state)

    Cs = np.logspace(-3, 2, 6).tolist()

    pipeline: list[BaseEstimator] = [StandardScaler()]
    if pca_num_components is not None:
        pipeline.append(PCA(n_components=pca_num_components))

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
                   stratify: Optional[np.ndarray] = None,
                   test_fraction=0.2, num_folds=3,
                   reg_range: tuple[float, float] = (-8, 3),
                   reg_grid_size: int = 10,
                   pca_num_components: Optional[float | Literal["auto"]] = None,
                   pca_dimensions: Optional[np.ndarray] = None,
                   num_repeats=1, n_jobs=None, random_state=42):
    """
    Args:
        reg_range: tuple of (min_exp, max_exp) for log10 regularization strength
            grid search
    """
    seeds = np.random.RandomState(random_state).randint(0, 10000, num_repeats)

    # Find if we are going to be able to have both positive and negative classes in each
    # inner fold. If not, scale down the number of folds.
    if stratify is not None:
        min_class_count = test_fraction * np.min(np.bincount(stratify, minlength=num_classes))
        if min_class_count < num_folds:
            num_folds = max(1, int(np.floor(min_class_count)))
            L.warning(f"Reducing num_folds to {num_folds} due to limited class samples per fold.")

    results = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, idxs_train, idxs_test = \
            train_test_split(X, y, np.arange(len(X)),
                             test_size=test_fraction,
                             stratify=stratify,
                             shuffle=True,
                             random_state=seed)

        num_folds_i = num_folds
        class_counts_train = np.bincount(y_train, minlength=num_classes)
        class_counts_test = np.bincount(y_test, minlength=num_classes)
        if np.any(class_counts_train < num_folds) or np.any(class_counts_test < num_folds):
            if min(np.min(class_counts_train), np.min(class_counts_test)) >= 2:
                num_folds_i = np.min(class_counts_test)
                L.warning(f"Reducing num_folds to {num_folds_i} due to limited class samples "
                          f"in train/test split: train counts {class_counts_train}, "
                          f"test counts {class_counts_test}.")
            else:
                L.warning(f"Skipping repeat with seed {seed} due to insufficient class samples "
                        f"in train/test split: train counts {class_counts_train}, "
                        f"test counts {class_counts_test}.")
                continue

        # Prepare stratified CV inner splits
        if stratify is None:
            cv_inner = KFold(num_folds_i, shuffle=True, random_state=seed)
            splits = list(cv_inner.split(X_train))
        else:
            cv_inner = StratifiedKFold(num_folds_i, shuffle=True, random_state=seed)
            splits = list(cv_inner.split(X_train, stratify[idxs_train]))

        Cs = np.logspace(*reg_range, reg_grid_size).tolist()
        pca_num_components = [0.25, 0.5, 0.9] if pca_num_components == "auto" \
            else pca_num_components
        

        pipeline = []
        if pca_num_components is not None and X.shape[1] > 1:
            pca_m = PCA(n_components=pca_num_components)
            if pca_dimensions is not None:
                non_pca_dimensions = np.setdiff1d(np.arange(X.shape[1]), pca_dimensions)
                pipeline.append(("prep", ColumnTransformer([
                    ("baseline", StandardScaler(), non_pca_dimensions),
                    ("pca", make_pipeline(StandardScaler(), pca_m), pca_dimensions)
                ])))
            else:
                pipeline.append(("prep", make_pipeline(StandardScaler(), pca_m)))

        solver = "liblinear" if num_classes == 2 else "saga"
        logreg_kwargs = dict(max_iter=100000, class_weight="balanced",
                             fit_intercept=False, solver=solver)
        pipeline.append(("clf", LogisticRegression(**logreg_kwargs)))
        model = Pipeline(pipeline)

        param_grid = {
            "clf__C": Cs
        }
        if pca_num_components is not None:
            if "pca" in model.named_steps:
                param_grid["pca__n_components"] = pca_num_components
            elif "prep" in model.named_steps:
                param_grid["prep__pca__pca__n_components"] = pca_num_components

        gs = GridSearchCV(model, param_grid, cv=splits,
                          scoring="roc_auc" if num_classes == 2 else "accuracy",
                          refit=True, n_jobs=n_jobs)
        gs.fit(X_train, y_train)

        if callable(scoring):
            scorers = scoring
        elif scoring is None or isinstance(scoring, str):
            scorers = check_scoring(gs, scoring)
        else:
            scorers = _check_multimetric_scoring(gs, scoring)
            # _check_refit_for_multimetric(scorers)
            scorers = _MultimetricScorer(scorers=scorers)

        train_scores = scorers(gs, X_train, y_train)
        test_scores = scorers(gs, X_test, y_test)
        results.append({
            **{f"train_{k}": np.array([v]) for k, v in train_scores.items()},
            **{f"test_{k}": np.array([v]) for k, v in test_scores.items()},
            "train_idxs": [idxs_train],
            "test_idxs": [idxs_test],
            "estimator": [gs],
        })

    # Concatenate results from all repeats
    if len(results) == 0:
        return None

    fitted = {k: np.concatenate([r[k] for r in results]) if isinstance(results[0][k], np.ndarray)
              else list(itertools.chain.from_iterable(r[k] for r in results))
              for k in results[0].keys()}
    return fitted


def fit_train_test_old(X, y, num_classes: int, scoring: list[str],
                       test_fraction=0.2, num_folds=3,
                       pca_num_components: Optional[float] = None,
                       num_repeats=1, random_state=42):
    seeds = np.random.RandomState(random_state).randint(0, 10000, num_repeats)

    results = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, idxs_train, idxs_test = \
            train_test_split(X, y, np.arange(len(X)),
                            test_size=test_fraction, stratify=y, random_state=seed)

        Cs = np.logspace(-3, 2, 6)

        pipeline: list[BaseEstimator] = [StandardScaler()]
        if pca_num_components is not None:
            pipeline.append(PCA(n_components=pca_num_components))

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

        if callable(scoring):
            scorers = scoring
        elif scoring is None or isinstance(scoring, str):
            scorers = check_scoring(refit_model, scoring)
        else:
            scorers = _check_multimetric_scoring(refit_model, scoring)
            # _check_refit_for_multimetric(scorers)
            scorers = _MultimetricScorer(scorers=scorers)

        train_scores = scorers(refit_model, X_train, y_train)
        test_scores = scorers(refit_model, X_test, y_test)
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
                             models: list[ClassifierLike],
                             epochs: dict[str, mne.Epochs]) -> pd.DataFrame:
    """
    Get predictions from an ensemble of fit single-electrode models on held-out epochs,
    subsetting appropriately to match the properties of the data the model
    was fit on.

    Args:
        model_key: tuple of `(subject, electrode_idx, phoneme_pair, smin, smax)`;
            i.e. the keys of the dictionary returned by `run_decoding_analysis_single_electrode`
        models: list of fitted models
        epochs: dict mapping from subject to mne.Epochs containing the held-out epochs
    """
    subject, electrode_idx, phoneme_pair, smin, smax = model_key

    epochs_ij = epochs[subject]
    assert epochs_ij.metadata is not None
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


def get_population_ensemble_predictions(model_key: PopulationDecoderFitKey,
                                        models: list[ClassifierLike],
                                        electrode_idxs: list[int],
                                        epochs: mne.Epochs) -> pd.DataFrame:
    """
    Get predictions from an ensemble of fit single-electrode models on held-out epochs,
    subsetting appropriately to match the properties of the data the model
    was fit on.

    Args:
        model_key: tuple of `(subject, population_name, phoneme_pair, smin, smax)`;
            i.e. the keys of the dictionary returned by `run_decoding_population`
        models: list of fitted models
        epochs: mne.Epochs object containing the held-out epochs
    """
    subject, population_name, phoneme_pair, smin, smax = model_key

    assert epochs.metadata is not None
    selection = epochs.metadata.phoneme_pair == phoneme_pair
    if selection.sum() == 0:
        raise ValueError(f"No epochs found for subject {subject}, "
                         f"phoneme pair {phoneme_pair} in the given epochs.")

    X = epochs.get_data(picks=electrode_idxs)[selection][:, :, smin:smax]
    X = X.reshape(X.shape[0], -1)  # flatten space * time

    outcomes = []
    for i, model in enumerate(models):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        outcomes.append(pd.DataFrame({
            "epoch_idx": epochs.metadata.index[selection].values,
            "decoder_target": epochs.metadata.categorical_acoustic_cue[selection].values,
            "decoder_prediction": y_pred,
            "decoder_proba": y_proba,
            "fold": i
        }))

    return pd.concat(outcomes)