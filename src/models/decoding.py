"""
Sliding-window decoding models for ECoG data.

===========================================================================
CHECKPOINT FILE FORMATS
===========================================================================

------------------------------------------------------------------------
outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt
------------------------------------------------------------------------
Written by: notebooks/causal4/behavior_decoding_single_electrode.ipynb
Producer:   run_decoding_model_comparison_population (this module)

Load:
    data = torch.load(path)

Top-level keys:
    "A_decoding_results", "B_decoding_results", "C_decoding_results"
    "A_decoders",         "B_decoders",         "C_decoders"

Scores (e.g. data["A_decoding_results"]):
    dict[outer_key, pd.DataFrame]
    outer_key = (subject: str, electrode_idx: int, phoneme_pair: str)
    DataFrame columns (one row per fold × time-window × groupby-group):
        subject, population (=str(electrode_idx)), phoneme_pair
        smin, smax          – sample window bounds
        fold                – repeat/fold index
        word_end            – groupby value (e.g. "pb")
        baseline_roc_auc, full_roc_auc
        baseline_precision, full_precision
        baseline_recall,    full_recall
        baseline_log_loss,  full_log_loss
        baseline_clf__C, full_clf__C           – best hparams
        full_prep__pca__pca__n_components      – best PCA hparam (if PCA used)

Estimators (e.g. data["A_decoders"]):
    dict[outer_key, inner_dict]
    outer_key = (subject, electrode_idx, phoneme_pair)   # same as above
    inner_dict keys: (subject, population_name, phoneme_pair, name, smin, smax, fold)
        name   – groupby tuple value, e.g. ("pb",)  (from groupby=["word_end"])
        smin, smax – sample window
        fold   – repeat index
    inner_dict values: {
        "electrode_idxs":   list[int],
        "estimator":        fitted GridSearchCV (or _FixedHParamEstimator),
        "test_predictions": pd.DataFrame with columns
                                decoder_target, baseline_decoder_prediction,
                                baseline_decoder_proba, full_decoder_prediction,
                                full_decoder_proba, fold, epoch_idx
    }

Quick-access recipe:
    key = (subject, electrode_idx, phoneme_pair)
    # best window by full ROC-AUC:
    df = data["A_decoding_results"][key]
    best = df.groupby(["smin","smax","word_end"])["full_roc_auc"].mean().idxmax()
    smin, smax, word_end = best
    fold = 0
    est_key = (subject, str(electrode_idx), phoneme_pair, (word_end,), smin, smax, fold)
    estimator = data["A_decoders"][key][est_key]["estimator"]

------------------------------------------------------------------------
outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/
------------------------------------------------------------------------
Written by: notebooks/causal4/behavior_decoding_single_electrode_acoustic.ipynb
Producer:   run_decoding_searchlight_single_electrode (this module)

Files saved per subject (some notebooks use .joblib, others .pt — same structure):
    decoding_models.joblib  OR  results.pt   – fitted models (see below)
    outcomes.parquet         – test-fold predictions
    all_outcomes.parquet     – predictions on all relevant epochs (multiple targets)
    train_scores.parquet, test_scores.parquet
    avg_test_scores.csv

decoding_models.joblib / results.pt (models dict):
    import joblib; models = joblib.load(path)        # .joblib variant
    import torch;  models = torch.load(path)         # .pt variant
    # dict[DecoderFitKey, list[estimator]]
    # DecoderFitKey = (subject: str, electrode_idx: int, phoneme_pair: str,
    #                  smin: int, smax: int)
    # list has one entry per repeat/fold (default num_repeats=5)

outcomes.parquet / all_outcomes.parquet:
    pd.read_parquet(path)
    columns: subject, electrode_idx, phoneme_pair, smin, smax,
             epoch_idx, fold, decoder_target, decoder_prediction, decoder_proba
    (all_outcomes also has a "measure" level for categorical_acoustic_cue /
     subject_specific_acoustics variants)

Quick-access recipe:
    import joblib  # or torch
    models = joblib.load(".../decoding_models.joblib")
    key = (subject, electrode_idx, phoneme_pair, smin, smax)
    estimators = models[key]   # list of fitted models, one per repeat
    proba = estimators[0].predict_proba(X)[:, 1]

------------------------------------------------------------------------
outputs/causal5/behavior_decoding_single_electrode/{subject}/results.joblib
------------------------------------------------------------------------
Written by: notebooks/causal5/behavior_decoding_single_electrode.py
Producer:   run_decoding_model_comparison_population (this module)

Load:
    import joblib
    data = joblib.load(path)

Top-level keys (simplified vs causal4 — no B/C variants):
    "decoding_results"   (was "A_decoding_results" in causal4)
    "decoders"           (was "A_decoders" in causal4)

Same inner structure as causal4 A_decoding_results / A_decoders (see above).

Note on viz_paper.py compatibility:
    evaluate_behav_decoder_on_phon_window expects key name "A_decoders".
    The causal5 A_neurometrics normalizes on load:
        {"A_decoders": data["decoders"]}

------------------------------------------------------------------------
outputs/causal5/behavior_decoding_single_electrode_acoustic/{subject}/
------------------------------------------------------------------------
Same format as causal4 acoustic outputs. Files:
    decoding_models.joblib  – fitted models (joblib, same structure as causal4)
    outcomes.parquet         – test-fold predictions
    all_outcomes.parquet     – predictions on all relevant epochs (multiple targets)
    train_scores.parquet, test_scores.parquet, avg_test_scores.csv

Loading for A_neurometrics (phonetic_decoder_checkpoints structure):
    The causal5 A_neurometrics builds the structure expected by viz_paper.py from
    the three separate files:
        checkpoints[subject] = {
            "models":       joblib.load("decoding_models.joblib"),
            "outcomes":     dict from outcomes.parquet grouped by (subject, electrode_idx,
                                phoneme_pair, smin, smax),
            "all_outcomes": dict from all_outcomes.parquet grouped by (..., measure),
        }

------------------------------------------------------------------------
outputs/causal5/ganong_decoding/{subject}/results.joblib
------------------------------------------------------------------------
Written by: notebooks/causal5/ganong_decoding_single_electrode.py
Producer:   run_decoding_model_comparison_population (this module)

Load / structure: identical to causal5 behavior_decoding_single_electrode (above).
    import joblib
    data = joblib.load(path)
    # top-level keys: "decoding_results", "decoders"

Key difference from behavior_decoding_single_electrode:
    - groupby=None  (pooled across both lexical completions — no word_end split)
    - stratify=("resampled", "lexical_evidence")  (balance steps AND completions per fold)

Because groupby=None, the `name` tuple in inner keys is always an empty tuple ():
    outer_key = (subject, electrode_idx, phoneme_pair)
    inner_key = (subject, str(electrode_idx), phoneme_pair, (), smin, smax, fold)

decoding_results DataFrames have the same columns as behavior_decoding_single_electrode,
EXCEPT there is no `word_end` column (groupby was not applied).

Electrode set: restricted to sites where behavior_decoding_single_electrode_summarize
reports full_roc_auc >= behav_peak_threshold (default 0.6) at their peak window
(i.e. only sites with significant behavioral decoding are decoded here).
===========================================================================

===========================================================================
SKLEARN PIPELINE STRUCTURES
===========================================================================

------------------------------------------------------------------------
Behavioral decoder pipeline  (behavior_decoding_single_electrode)
Producer: fit_train_test (this module)
------------------------------------------------------------------------

The value stored at data["A_decoders"][outer_key][inner_key]["estimator"]
is a fitted sklearn GridSearchCV whose best_estimator_ is:

    Pipeline([
        ("prep", ColumnTransformer([
            ("baseline", StandardScaler(),  [0]),        # col 0: resampled (continuous morph step)
            ("pca",      Pipeline([                      # cols 1+: neural window (15 samples)
                ("standardscaler", StandardScaler()),
                ("pca",            PCA(n_components=k)),
            ]),                             [1:]),
        ])),
        ("clf", LogisticRegression(...)),
    ])

Input to the pipeline:  X of shape (n_epochs, 1 + n_neural_features)
    col 0   – resampled (the continuous acoustic morph step, 1–6)
    cols 1+ – HGA amplitude at each sample in the time window

Accessor shortcuts (given est = checkpoint["estimator"].best_estimator_):
    neural StandardScaler : est.named_steps["prep"]
                               .named_transformers_["pca"]
                               .named_steps["standardscaler"]
    PCA                   : est.named_steps["prep"]
                               .named_transformers_["pca"]
                               .named_steps["pca"]
    LogisticRegression    : est.named_steps["clf"]

Cross-window transfer note:
    When applying this decoder to a *different* time window, bypass the
    ColumnTransformer entirely to avoid cross-scaler contamination:
      1. Normalise new-window neural data with that window's own scaler.
      2. Project through est's PCA.
      3. Prepend a zeros column (null resampled feature).
      4. Apply est's LogisticRegression directly.

------------------------------------------------------------------------
Acoustic decoder pipeline  (behavior_decoding_single_electrode_acoustic)
Producer: fit_train_test / run_decoding_searchlight_single_electrode
------------------------------------------------------------------------

Each entry in models[DecoderFitKey] (one per repeat/fold) is a plain
sklearn Pipeline built with make_pipeline:

    Pipeline([
        ("standardscaler",    StandardScaler()),
        ("logisticregression", LogisticRegression(...)),
    ])

Input to the pipeline:  X of shape (n_epochs, n_neural_features)
    n_neural_features = smax - smin  (always 15 samples for causal4 runs)
    There is no resampled feature column; neural data only.

Accessor shortcuts (given model = models[key][fold]):
    StandardScaler        : model.named_steps["standardscaler"]
    LogisticRegression    : model.named_steps["logisticregression"]

Cross-window transfer note:
    When applying this decoder to a *different* time window, bypass the
    pipeline's own scaler to avoid cross-scaler contamination:
      1. Normalise new-window data with that window's own scaler.
      2. Apply model's LogisticRegression directly (predict_proba on scaled X).
===========================================================================
"""
import itertools
from typing import Literal, Optional, Protocol, TypeAlias, cast

import mne
import numpy as np
import pandas as pd
from loguru import logger as L
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    check_scoring,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from tqdm.auto import tqdm

DecoderFitKey: TypeAlias = tuple[
    str, int, str, int, int
]  # (subject, electrode_idx, phoneme_pair, smin, smax)
"""Result of a single electrode decoder analysis."""

PopulationDecoderFitKey: TypeAlias = tuple[
    str, str, str, int, int
]  # (subject, population_name, phoneme_pair, smin, smax)
"""Result of a population decoder analysis"""

Epochs: TypeAlias = mne.Epochs | mne.epochs.EpochsFIF


class ClassifierLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class _FixedHParamEstimator:
    """Wraps a fitted sklearn Pipeline to mimic the GridSearchCV interface.

    Used in permutation testing where hyperparameters are fixed from the true
    model fit rather than re-searched, so we avoid the cost of inner-CV.
    """

    _estimator_type = "classifier"  # required for sklearn scoring utilities

    def __init__(self, pipeline: Pipeline, params: dict):
        self.best_params_ = params
        self._pipeline = pipeline

    @property
    def classes_(self) -> np.ndarray:
        return self._pipeline.classes_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FixedHParamEstimator":
        self._pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)


def _prepare_decoding_population(
    epochs_i: Epochs,
    electrode_idxs: list[int],
    phoneme_pair: str,
    stride: Optional[int] = None,
    window_size: Optional[int] = None,
    global_min_sample: int = 0,
    global_max_sample: Optional[int] = None,
    target: Literal[
        "acoustic",
        "lexical_evidence",
        "mismatch",
        "mismatch_left_right",
        "behavior_categorical",
        "behavior_categorical_forced",
    ] = "lexical_evidence",
    groupby: Optional[list[str]] = None,
    filter: Optional[str] = None,
    include_only_full_windows=True,
    randomize=False,
    windows: Optional[np.ndarray] = None,
):
    """
    Prepare windowed decoding inputs/targets for the given parameters.

    Args:
        groupby: Yield separate samples for each combination of these grouping variables.
        filter: Optional filter string on epoch metadata, passed to pd.DataFrame.query.
        windows: Optional explicit (N, 2) array of (smin, smax) pairs. When provided,
            stride/window_size/global_min_sample/global_max_sample are ignored.

    Yields tuples for each window of form:
        - name: tuple of grouping values, same length as `groupby`.
            Empty tuple if `groupby` is None.
        - smin: start sample of window
        - smax: end sample of window
        - selection: boolean array indicating selected epochs in `epochs_i`
        - X_window: windowed input data
        - y: target labels
    """

    if target not in [
        "acoustic",
        "lexical_evidence",
        "mismatch",
        "mismatch_left_right",
        "behavior_categorical",
        "behavior_categorical_forced",
    ]:
        raise ValueError(f"Invalid target {target}")
    assert epochs_i.metadata is not None

    X = epochs_i.get_data(picks=electrode_idxs)

    if windows is None:
        assert stride is not None and window_size is not None, (
            "stride and window_size must be provided when windows is not given explicitly"
        )
        if global_max_sample is not None:
            assert global_max_sample > global_min_sample, (
                f"global_max_sample ({global_max_sample}) must be greater than global_min_sample ({global_min_sample})"
            )

        data_max_sample = X.shape[2]
        if global_max_sample is None:
            global_max_sample = cast(int, data_max_sample)
        else:
            global_max_sample = min(global_max_sample, data_max_sample)

        if global_max_sample - global_min_sample < window_size:
            raise ValueError(
                f"Window size ({window_size}) is larger than the available data range "
                f"({global_max_sample - global_min_sample}). Please adjust the parameters."
            )

        windows_left = np.arange(global_min_sample, global_max_sample, stride)
        windows_right = windows_left + window_size
        windows = np.array(list(zip(windows_left, windows_right)))
        if include_only_full_windows:
            windows = windows[windows[:, 1] <= global_max_sample]

    selection = epochs_i.metadata.phoneme_pair == phoneme_pair
    if selection.sum() == 0:
        raise ValueError(
            f"No epochs found for phoneme pair {phoneme_pair} in the given epochs."
        )

    md = epochs_i.metadata
    if filter is not None:
        selection = selection & md.eval(filter)

    _make_grouper = lambda: md.groupby(groupby) if groupby is not None else [((), md)]

    # Count non-empty groups upfront so callers can pass total= to tqdm.
    n_nonempty_groups = sum(
        1
        for _, md_sub in _make_grouper()
        if (selection & md.index.isin(md_sub.index)).sum() > 0
    )
    total = n_nonempty_groups * len(windows)

    def _generate():
        for name, metadata_subset in _make_grouper():
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
                elif target in ("behavior_categorical", "behavior_categorical_forced"):
                    y = md.behavior_dummy_forced[selection_i].values

                # stratify_class = epochs_ij.metadata.stratify_class[selection_i].values

                if randomize:
                    # Randomize the labels
                    y = np.random.permutation(y)

                yield name, smin, smax, selection_i, X_window, y

    return total, _generate()


def run_decoding_population(
    epochs_i: Epochs,
    electrode_idxs: list[int],
    phoneme_pair: str,
    subject: str,
    population_name: str,
    stride: int,
    window_size: int,
    global_min_sample: int = 0,
    global_max_sample: Optional[int] = None,
    target: Literal[
        "acoustic",
        "lexical_evidence",
        "mismatch",
        "mismatch_left_right",
        "behavior_categorical",
    ] = "lexical_evidence",
    strategy: Literal["nested-cv", "train-test"] = "nested-cv",
    groupby: Optional[list[str]] = None,
    pca_num_components: Optional[float] = None,
    return_outcomes=True,
    include_only_full_windows=True,
    smoke_test=False,
    randomize=False,
):

    assert epochs_i.metadata is not None
    _, _gen = _prepare_decoding_population(
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
        randomize=randomize,
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
        scoring = (
            ["roc_auc", "f1_macro", "accuracy"]
            if num_classes == 2
            else ["f1_macro", "accuracy"]
        )

        if strategy == "nested-cv":
            fitted = fit_nested_cv(
                X_window,
                y,
                num_classes=num_classes,
                pca_num_components=pca_num_components,
                scoring=scoring,
            )
        elif strategy == "train-test":
            fitted = fit_train_test(
                X_window,
                y,
                num_classes=num_classes,
                pca_num_components=pca_num_components,
                scoring=scoring,
                num_repeats=5,
            )

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
            for fold, (test_idxs, estimator) in enumerate(
                zip(fitted["test_idxs"], fitted["estimator"])
            ):
                # test_idxs are indices into X, y, which themselves are indices into epochs_ij[selection]
                test_epoch_idxs = epochs_i.metadata.index[selection][test_idxs]
                fold_results.append(
                    pd.DataFrame(
                        {
                            "decoder_target": y[test_idxs],
                            "decoder_prediction": estimator.predict(
                                X_window[test_idxs]
                            ),
                            "decoder_proba": estimator.predict_proba(
                                X_window[test_idxs]
                            )[:, 1],
                            "fold": fold,
                            "epoch_idx": test_epoch_idxs,
                        }
                    )
                )

            outcomes[result_key] = pd.concat(fold_results)

        models[result_key] = fitted["estimator"]

    return train_scores, test_scores, outcomes, models


def _fit_comparison_window(
    name,
    smin: int,
    smax: int,
    selection,
    X_window: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    baseline_features: list,
    stratify_cols,
    groupby,
    electrode_idxs: list,
    pca_num_components,
    strategy: str,
    fixed_hparams_df,
    baseline_hparam_cols,
    full_hparam_cols,
    return_estimators: bool,
    subject: str,
    population_name: str,
    phoneme_pair: str,
    seed: int,
) -> tuple[list[dict], dict]:
    """Fit baseline + full models for one window/group. Called by Parallel."""
    num_classes = len(set(y))
    if num_classes != 2:
        L.warning(
            f"Skipping model comparison for {subject}, {population_name}, "
            f"{phoneme_pair}, {name}, {smin}-{smax} because num_classes={num_classes} != 2"
        )
        return [], {}

    md = metadata[selection]
    X_baseline = md[baseline_features].values
    X_full = np.concatenate([X_baseline, X_window], axis=1)

    stratify_codes = None
    if stratify_cols is not None:
        stratify_codes = pd.factorize(md[list(stratify_cols)].apply(tuple, axis=1))[0]

    baseline_fixed_params = None
    full_fixed_params = None
    if fixed_hparams_df is not None:
        window_mask = (fixed_hparams_df["smin"] == smin) & (
            fixed_hparams_df["smax"] == smax
        )
        for gb_var, gb_val in zip(groupby or [], name):
            window_mask &= fixed_hparams_df[gb_var] == gb_val
        window_rows = (
            fixed_hparams_df[window_mask].sort_values("fold").to_dict("records")
        )
        if not window_rows:
            return [], {}
        baseline_fixed_params = [
            {c.removeprefix("baseline_"): row[c] for c in baseline_hparam_cols}
            for row in window_rows
        ]
        full_fixed_params = [
            {c.removeprefix("full_"): row[c] for c in full_hparam_cols}
            for row in window_rows
        ]

    def _fit(
        X,
        y,
        num_classes,
        stratify,
        random_state,
        reg_range,
        reg_grid_size,
        baseline_results=None,
        pca_dimensions=None,
        fixed_params_list=None,
    ):
        if strategy == "nested-cv":
            return fit_nested_cv(
                X,
                y,
                num_classes=num_classes,
                stratify=stratify,
                reg_range=reg_range,
                reg_grid_size=reg_grid_size,
                pca_num_components=pca_num_components,
                pca_dimensions=pca_dimensions,
                scoring=["roc_auc"],
                random_state=random_state,
            )
        elif strategy == "train-test":
            return fit_train_test(
                X,
                y,
                num_classes=num_classes,
                stratify=stratify,
                reg_range=reg_range,
                reg_grid_size=reg_grid_size,
                pca_num_components=pca_num_components,
                pca_dimensions=pca_dimensions,
                baseline_results=baseline_results,
                fixed_params_list=fixed_params_list,
                scoring=["roc_auc"],
                n_jobs=1,
                num_repeats=len(fixed_params_list) if fixed_params_list is not None else 5,
                random_state=random_state,
            )
        else:
            raise ValueError("Unknown strategy: {}".format(strategy))

    baseline_results = _fit(
        X_baseline,
        y,
        num_classes,
        stratify=stratify_codes,
        reg_range=(-1, 1),
        reg_grid_size=2,
        random_state=seed,
        fixed_params_list=baseline_fixed_params,
    )
    full_results = _fit(
        X_full,
        y,
        num_classes,
        stratify=stratify_codes,
        reg_range=(-8, 3),
        reg_grid_size=10,
        pca_dimensions=np.arange(X_baseline.shape[1], X_full.shape[1]),
        random_state=seed,
        fixed_params_list=full_fixed_params,
    )

    if baseline_results is None or full_results is None:
        L.warning(
            f"Skipping model comparison for {subject}, {population_name}, "
            f"{phoneme_pair}, {name}, {smin}-{smax} because fitting failed."
        )
        return [], {}

    results_list = []
    estimators_dict = {}

    for fold, (baseline_test_idxs, baseline_estimator) in enumerate(
        zip(baseline_results["test_idxs"], baseline_results["estimator"])
    ):
        full_test_idxs = full_results["test_idxs"][fold]
        full_estimator = full_results["estimator"][fold]

        assert full_test_idxs.tolist() == baseline_test_idxs.tolist()
        test_idxs = full_test_idxs

        baseline_proba = baseline_estimator.predict_proba(X_baseline[test_idxs])[:, 1]
        full_proba = full_estimator.predict_proba(X_full[test_idxs])[:, 1]

        baseline_prediction = baseline_estimator.predict(X_baseline[test_idxs])
        full_prediction = full_estimator.predict(X_full[test_idxs])

        baseline_precision, baseline_recall, _, _ = precision_recall_fscore_support(
            y[test_idxs], baseline_prediction, average="binary", zero_division=0.0
        )
        full_precision, full_recall, _, _ = precision_recall_fscore_support(
            y[test_idxs], full_prediction, average="binary", zero_division=0.0
        )

        baseline_log_loss = log_loss(y[test_idxs], baseline_proba)
        full_log_loss = log_loss(y[test_idxs], full_proba)

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
            "baseline_log_loss": baseline_log_loss,
            "full_log_loss": full_log_loss,
        }
        for groupby_variable, value in zip(groupby or [], name):
            result_i[groupby_variable] = value

        for param, val in baseline_estimator.best_params_.items():
            result_i["baseline_" + param] = val
        for param, val in full_estimator.best_params_.items():
            result_i["full_" + param] = val

        results_list.append(result_i)

        if return_estimators:
            key = (subject, population_name, phoneme_pair, name, smin, smax, fold)
            estimators_dict[key] = {
                "electrode_idxs": electrode_idxs,
                "estimator": full_estimator,
                "test_predictions": pd.DataFrame(
                    {
                        "decoder_target": y[test_idxs],
                        "baseline_decoder_prediction": baseline_prediction,
                        "baseline_decoder_proba": baseline_proba,
                        "full_decoder_prediction": full_prediction,
                        "full_decoder_proba": full_proba,
                        "fold": fold,
                        "epoch_idx": md.iloc[test_idxs].index.values,
                    }
                ),
            }

    return results_list, estimators_dict


def run_decoding_model_comparison_population(
    epochs_i: Epochs,
    electrode_idxs: list[int],
    phoneme_pair: str,
    subject: str,
    population_name: str,
    baseline_features: list[str],
    stride: Optional[int] = None,
    window_size: Optional[int] = None,
    global_min_sample: int = 0,
    global_max_sample: Optional[int] = None,
    target: Literal[
        "acoustic",
        "lexical_evidence",
        "mismatch",
        "mismatch_left_right",
        "behavior_categorical",
        "behavior_categorical_forced",
    ] = "lexical_evidence",
    strategy: Literal["nested-cv", "train-test"] = "nested-cv",
    groupby: Optional[list[str]] = None,
    filter: Optional[str] = None,
    stratify: tuple[str, ...] = ("resampled", "lexical_evidence"),
    pca_num_components: Optional[float | Literal["auto"]] = None,
    include_only_full_windows=True,
    return_estimators=False,
    n_jobs=None,
    smoke_test=False,
    randomize=False,
    fixed_hparams_df: Optional[pd.DataFrame] = None,
):
    """
    Run a model comparison evaluating target prediction using either
    `baseline_features` (indexing into the epoch metadata) or the
    combination of `baseline_features` plus ECoG data.

    For each sliding window over the trial time-axis (defined by `stride` and
    `window_size`) and each group of trials produced by `groupby`, two
    LogisticRegression models are fit and compared on held-out test folds
    (strategy="train-test", `num_repeats` outer splits):

      baseline model  — trained on acoustic/metadata features only
                        (`baseline_features`, e.g. ["resampled"])
      full model      — trained on baseline features + windowed ECoG activity
                        (flattened to electrodes × time)

    Both models undergo inner-CV grid search (GridSearchCV) over:
      - regularisation strength C  (baseline: 2 values; full: 10 values)
      - PCA variance ratio         (if `pca_num_components` is not None)

    The per-fold test metric is ROC-AUC; downstream analysis typically focuses
    on Δ ROC-AUC = full_roc_auc − baseline_roc_auc as the effect of neural
    activity beyond the acoustic cue.

    Args:
        stride: Stride between windows in samples. Required when fixed_hparams_df
            is None; ignored when fixed_hparams_df is provided (windows are derived
            from it instead).
        window_size: Window size in samples. Same optionality as stride.
        fixed_hparams_df: When provided, skips the inner grid search entirely
            and reuses hyperparameters from a previous fit stored in this
            DataFrame (typically the true-model results from a prior call).
            The set of (smin, smax) windows to run is derived from the unique
            values in this DataFrame — stride/window_size/global_min_sample/
            global_max_sample are then unused. Any (group, window) combination
            absent from fixed_hparams_df is silently skipped.
            The DataFrame must contain columns of the form ``"baseline_<param>"``
            and ``"full_<param>"`` (e.g. ``"baseline_clf__C"``, ``"full_clf__C"``,
            ``"full_prep__pca__pca__n_components"``). Any column whose name
            contains ``"__"`` is treated as a hyperparameter column.

            Combine with ``randomize=True`` for permutation testing without
            paying the cost of K grid searches.
    """

    assert epochs_i.metadata is not None

    # When fixed_hparams_df is provided, derive the windows to iterate from it
    # rather than from stride/window_size. Any (group, window) not present in
    # fixed_hparams_df is skipped inside the loop below.
    _explicit_windows: Optional[np.ndarray] = None
    if fixed_hparams_df is not None:
        _explicit_windows = (
            fixed_hparams_df[["smin", "smax"]].drop_duplicates().to_numpy()
        )
    else:
        assert stride is not None and window_size is not None, (
            "stride and window_size are required when fixed_hparams_df is not provided"
        )

    n_windows, _gen = _prepare_decoding_population(
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
        randomize=randomize,
        windows=_explicit_windows,
    )

    seed = 42

    # Pre-compute hparam column names (None when fixed_hparams_df not provided).
    _baseline_hparam_cols = None
    _full_hparam_cols = None
    if fixed_hparams_df is not None:
        _baseline_hparam_cols = [
            c
            for c in fixed_hparams_df.columns
            if c.startswith("baseline_") and "__" in c
        ]
        _full_hparam_cols = [
            c for c in fixed_hparams_df.columns if c.startswith("full_") and "__" in c
        ]

    metadata = epochs_i.metadata

    per_window = Parallel(n_jobs=n_jobs)(
        delayed(_fit_comparison_window)(
            name, smin, smax, selection, X_window, y,
            metadata, baseline_features, stratify, groupby,
            electrode_idxs, pca_num_components, strategy,
            fixed_hparams_df, _baseline_hparam_cols, _full_hparam_cols,
            return_estimators, subject, population_name, phoneme_pair, seed,
        )
        for name, smin, smax, selection, X_window, y in tqdm(
            _gen, total=n_windows, unit="window", leave=False,
        )
    )

    results = []
    all_estimators = {}
    for result_list, est_dict in per_window:
        results.extend(result_list)
        all_estimators.update(est_dict)

    if return_estimators:
        return pd.DataFrame(results), all_estimators
    else:
        return pd.DataFrame(results)


def _fit_searchlight_electrode(
    row: pd.Series,
    windows: np.ndarray,
    phoneme_pairs,
    epoch_arrays: dict,
    epoch_metadata: dict,
    target: str,
    strategy: str,
    return_outcomes: bool,
    randomize: bool,
) -> tuple[dict, dict, dict, dict]:
    """Fit all windows × phoneme_pairs for one electrode. Called by Parallel."""
    train_scores: dict = {}
    test_scores: dict = {}
    outcomes: dict = {}
    models: dict = {}

    X_subj = epoch_arrays[row.subject]
    md = epoch_metadata[row.subject]

    for smin, smax in windows:
        for phoneme_pair in phoneme_pairs:
            selection = (md.phoneme_pair == phoneme_pair).values
            if selection.sum() == 0:
                continue

            X = X_subj[selection, row.electrode_idx, smin:smax]

            if target == "acoustic":
                y = md.categorical_acoustic_cue[selection].values
            elif target == "lexical_evidence":
                y = (md.word_end.str[0] == phoneme_pair[0])[selection].values
            elif target == "mismatch":
                y = md.mismatch[selection].values
            elif target == "mismatch_left_right":
                y = md.mismatch_left_right[selection].values
                X = X[y != 0]
                y = y[y != 0]
            elif target in ("behavior_categorical", "behavior_categorical_forced"):
                y = md.behavior_dummy_forced[selection].values

            num_classes = len(set(y))

            if randomize:
                y = np.random.permutation(y)

            scoring = (
                ["roc_auc", "neg_log_loss", "f1_macro", "accuracy"]
                if num_classes == 2
                else ["f1_macro", "accuracy"]
            )

            if strategy == "nested-cv":
                fitted = fit_nested_cv(X, y, num_classes=num_classes, scoring=scoring)
            elif strategy == "train-test":
                fitted = fit_train_test(
                    X,
                    y,
                    num_classes=num_classes,
                    scoring=scoring,
                    stratify=y,
                    num_repeats=5,
                    n_jobs=1,
                )

            if fitted is None:
                continue

            result_key = (row.subject, row.electrode_idx, phoneme_pair, smin, smax)

            if isinstance(scoring, list):
                train_scores[result_key] = {k: fitted["train_" + k] for k in scoring}
                test_scores[result_key] = {k: fitted["test_" + k] for k in scoring}
            else:
                train_scores[result_key] = fitted["train_score"]
                test_scores[result_key] = fitted["test_score"]

            if return_outcomes:
                fold_results = []
                for fold, (test_idxs, estimator) in enumerate(
                    zip(fitted["test_idxs"], fitted["estimator"])
                ):
                    test_epoch_idxs = md.index[selection][test_idxs]
                    fold_results.append(
                        pd.DataFrame(
                            {
                                "decoder_target": y[test_idxs],
                                "decoder_prediction": estimator.predict(X[test_idxs]),
                                "decoder_proba": estimator.predict_proba(
                                    X[test_idxs]
                                )[:, 1],
                                "fold": fold,
                                "epoch_idx": test_epoch_idxs,
                            }
                        )
                    )
                outcomes[result_key] = pd.concat(fold_results)

            models[result_key] = fitted["estimator"]

    return train_scores, test_scores, outcomes, models


def run_decoding_searchlight_single_electrode(
    epochs,
    electrode_df,
    stride: int,
    window_size: int,
    global_min_sample: int = 0,
    global_max_sample: Optional[int] = None,
    target: Literal[
        "acoustic",
        "lexical_evidence",
        "mismatch",
        "mismatch_left_right",
        "behavior_categorical",
        "behavior_categorical_forced",
    ] = "lexical_evidence",
    strategy: Literal["nested-cv", "train-test"] = "nested-cv",
    filter_speech_responsive=True,
    return_outcomes=True,
    include_only_full_windows=True,
    smoke_test=False,
    randomize=False,
    n_jobs=None,
) -> tuple[
    dict[DecoderFitKey, dict[str, float]],
    dict[DecoderFitKey, dict[str, float]],
    dict[DecoderFitKey, pd.DataFrame],
    dict[DecoderFitKey, list[BaseEstimator]],
]:
    """
    stride: in samples
    window_size: in samples
    """

    if target not in [
        "acoustic",
        "lexical_evidence",
        "mismatch",
        "mismatch_left_right",
        "behavior_categorical",
        "behavior_categorical_forced",
    ]:
        raise ValueError(f"Invalid target {target}")
    if strategy not in ["nested-cv", "train-test"]:
        raise ValueError(f"Invalid strategy {strategy}")

    if global_max_sample is not None:
        assert global_max_sample > global_min_sample, (
            f"global_max_sample ({global_max_sample}) must be greater than global_min_sample ({global_min_sample})"
        )

    data_max_sample = min([epoch.get_data().shape[2] for epoch in epochs.values()])
    if global_max_sample is None:
        global_max_sample = cast(int, data_max_sample)
    else:
        global_max_sample = min(global_max_sample, data_max_sample)

    if global_max_sample - global_min_sample < window_size:
        raise ValueError(
            f"Window size ({window_size}) is larger than the available data range "
            f"({global_max_sample - global_min_sample}). Please adjust the parameters."
        )

    windows_left = np.arange(global_min_sample, global_max_sample, stride)
    windows_right = windows_left + window_size
    windows = np.array(list(zip(windows_left, windows_right)))
    if include_only_full_windows:
        windows = windows[windows[:, 1] <= global_max_sample]

    phoneme_pairs = next(iter(epochs.values())).metadata.phoneme_pair.unique()

    if filter_speech_responsive:
        electrodes = electrode_df.query("speech_responsive").reset_index()
    else:
        # include all electrodes
        electrodes = electrode_df.reset_index()

    if smoke_test:
        electrodes = electrodes.iloc[:5]

    # Pre-extract epoch arrays once per subject so workers share memory-mapped arrays
    # rather than calling get_data() inside the innermost loop.
    epoch_arrays = {subj: epochs[subj].get_data() for subj in epochs}
    epoch_metadata = {subj: epochs[subj].metadata for subj in epochs}

    per_electrode = Parallel(n_jobs=n_jobs)(
        delayed(_fit_searchlight_electrode)(
            row, windows, phoneme_pairs, epoch_arrays, epoch_metadata,
            target, strategy, return_outcomes, randomize,
        )
        for _, row in tqdm(electrodes.iterrows(), total=len(electrodes))
    )

    train_scores, test_scores, outcomes, models = {}, {}, {}, {}
    for ts, vs, oc, mo in per_electrode:
        train_scores.update(ts)
        test_scores.update(vs)
        outcomes.update(oc)
        models.update(mo)

    return train_scores, test_scores, outcomes, models


def fit_nested_cv(
    X,
    y,
    num_classes: int,
    scoring: list[str],
    stratify: Optional[np.ndarray] = None,
    num_outer_folds=2,
    num_inner_folds=2,
    pca_num_components: Optional[float] = None,
    random_state=42,
):
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
    pipeline.append(
        LogisticRegressionCV(
            Cs=Cs,
            cv=cv_inner,
            max_iter=100000,
            class_weight="balanced",
            fit_intercept=False,
            solver=solver,
        )
    )
    model = make_pipeline(*pipeline)
    fitted = cross_validate(
        model,
        X,
        y,
        cv=cv_outer,
        scoring=scoring,
        return_estimator=True,
        return_train_score=True,
    )

    # add information about the item idxs in each train/test fold
    fitted["train_idxs"] = []
    fitted["test_idxs"] = []
    for train_idxs, test_idxs in cv_outer.split(X, y):
        fitted["train_idxs"].append(train_idxs)
        fitted["test_idxs"].append(test_idxs)

    return fitted


def fit_train_test(
    X,
    y,
    num_classes: int,
    scoring: list[str],
    stratify: Optional[np.ndarray] = None,
    test_fraction=0.2,
    num_folds=3,
    reg_range: tuple[float, float] = (-8, 3),
    reg_grid_size: int = 10,
    pca_num_components: Optional[float | Literal["auto"]] = None,
    pca_dimensions: Optional[np.ndarray] = None,
    baseline_results: Optional[dict] = None,
    fixed_params_list: Optional[list[dict]] = None,
    num_repeats=1,
    n_jobs=None,
    random_state=42,
):
    """
    Args:
        reg_range: tuple of (min_exp, max_exp) for log10 regularization strength
            grid search
    """
    assert isinstance(scoring, list)
    if "neg_log_loss" not in scoring:
        scoring = scoring + ["neg_log_loss"]

    seeds = np.random.RandomState(random_state).randint(0, 10000, num_repeats)

    # Ensure there are enough of each label class for num_folds inner CV folds.
    # Inner CV operates on the training set ((1 - test_fraction) of total), so check that
    # the minority class has at least num_folds training examples.
    min_label_count = np.min(np.unique(y, return_counts=True)[1])
    if (1 - test_fraction) * min_label_count < num_folds:
        num_folds = max(1, int(np.floor((1 - test_fraction) * min_label_count)))
        L.warning(
            f"Reducing num_folds to {num_folds} due to limited class samples per fold."
        )

    results = []
    for i, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test, idxs_train, idxs_test = train_test_split(
            X,
            y,
            np.arange(len(X)),
            test_size=test_fraction,
            stratify=y,
            shuffle=True,
            random_state=seed,
        )

        # if baseline_results is not None:
        #     np.testing.assert_array_equal(idxs_train, baseline_results["train_idxs"][i])
        #     np.testing.assert_array_equal(idxs_test, baseline_results["test_idxs"][i])

        num_folds_i = num_folds
        class_counts_train = np.unique(y_train, return_counts=True)[1]
        class_counts_test = np.unique(y_test, return_counts=True)[1]
        if np.any(class_counts_train < num_folds) or np.any(
            class_counts_test < num_folds
        ):
            if min(np.min(class_counts_train), np.min(class_counts_test)) >= 2:
                num_folds_i = np.min(class_counts_test)
                L.warning(
                    f"Reducing num_folds to {num_folds_i} due to limited class samples "
                    f"in train/test split: train counts {class_counts_train}, "
                    f"test counts {class_counts_test}."
                )
            else:
                L.warning(
                    f"Skipping repeat with seed {seed} due to insufficient class samples "
                    f"in train/test split: train counts {class_counts_train}, "
                    f"test counts {class_counts_test}."
                )
                continue

        # Prepare stratified CV inner splits
        if stratify is None:
            cv_inner = KFold(num_folds_i, shuffle=True, random_state=seed)
            splits = list(cv_inner.split(X_train))
        else:
            cv_inner = StratifiedKFold(num_folds_i, shuffle=True, random_state=seed)
            splits = list(cv_inner.split(X_train, stratify[idxs_train]))

        # If we have baseline outcomes, validate that these inner splits match as well
        # if baseline_results is not None:
        #     for j, (train_idxs, test_idxs) in enumerate(splits):
        #         np.testing.assert_array_equal(
        #             idxs_train[train_idxs],
        #             baseline_results[f"fold_{j}_train_idxs"][i]
        #         )
        #         np.testing.assert_array_equal(
        #             idxs_train[test_idxs],
        #             baseline_results[f"fold_{j}_test_idxs"][i]
        #         )

        Cs = np.logspace(*reg_range, reg_grid_size).tolist()
        pca_num_components = (
            [0.25, 0.5, 0.9] if pca_num_components == "auto" else pca_num_components
        )

        pipeline = []
        if pca_num_components is not None and X.shape[1] > 1:
            # When fixed_params_list is given, extract the scalar n_components from the stored
            # params so PCA() receives a valid value at construction time. For grid search, the
            # list value is fine because GridSearchCV overrides via set_params before fitting.
            if fixed_params_list is not None:
                params_i = fixed_params_list[i]
                n_comp_key = next((k for k in params_i if "n_components" in k), None)
                pca_init = (
                    params_i[n_comp_key]
                    if n_comp_key
                    else (
                        pca_num_components[0]
                        if isinstance(pca_num_components, list)
                        else pca_num_components
                    )
                )
            else:
                pca_init = pca_num_components
            pca_m = PCA(n_components=pca_init)
            if pca_dimensions is not None:
                non_pca_dimensions = np.setdiff1d(np.arange(X.shape[1]), pca_dimensions)
                pipeline.append(
                    (
                        "prep",
                        ColumnTransformer(
                            [
                                ("baseline", StandardScaler(), non_pca_dimensions),
                                (
                                    "pca",
                                    make_pipeline(StandardScaler(), pca_m),
                                    pca_dimensions,
                                ),
                            ]
                        ),
                    )
                )
            else:
                pipeline.append(("prep", make_pipeline(StandardScaler(), pca_m)))

        solver = "liblinear" if num_classes == 2 else "saga"
        logreg_kwargs = dict(
            max_iter=100000, class_weight="balanced", fit_intercept=False, solver=solver
        )
        pipeline.append(("clf", LogisticRegression(**logreg_kwargs)))
        model = Pipeline(pipeline)

        if fixed_params_list is not None:
            params_i = fixed_params_list[i]
            model.set_params(**params_i)
            model.fit(X_train, y_train)
            gs = _FixedHParamEstimator(model, params_i)
        else:
            param_grid = {"clf__C": Cs}
            if pca_num_components is not None:
                if "pca" in model.named_steps:
                    param_grid["pca__n_components"] = pca_num_components
                elif "prep" in model.named_steps:
                    param_grid["prep__pca__pca__n_components"] = pca_num_components

            gs = GridSearchCV(
                model,
                param_grid,
                cv=splits,
                scoring="roc_auc" if num_classes == 2 else "accuracy",
                refit=True,
                n_jobs=n_jobs,
            )
            gs.fit(X_train, y_train)

        scorers = _check_multimetric_scoring(gs, scoring)
        # _check_refit_for_multimetric(scorers)
        scorers = _MultimetricScorer(scorers=scorers)

        scores_dict = {}

        train_scores = scorers(gs, X_train, y_train)
        test_scores = scorers(gs, X_test, y_test)

        # # Also evaluate on individual folds, for later Kfold runs of extended models
        # for j, (fold_train_idxs, fold_test_idxs) in enumerate(splits):
        #     fold_train_scores = scorers(gs, X_train[fold_train_idxs], y_train[fold_train_idxs])
        #     fold_test_scores = scorers(gs, X_train[fold_test_idxs], y_train[fold_test_idxs])
        #     scores_dict.update({
        #         **{f"fold_{j}_train_{k}": np.array([v]) for k, v in fold_train_scores.items()},
        #         **{f"fold_{j}_test_{k}": np.array([v]) for k, v in fold_test_scores.items()},
        #         f"fold_{j}_train_idxs": [idxs_train[fold_train_idxs]],
        #         f"fold_{j}_test_idxs": [idxs_train[fold_test_idxs]],
        #     })

        scores_dict.update(
            {
                **{f"train_{k}": np.array([v]) for k, v in train_scores.items()},
                **{f"test_{k}": np.array([v]) for k, v in test_scores.items()},
                "train_idxs": [idxs_train],
                "test_idxs": [idxs_test],
                "estimator": [gs],
            }
        )
        results.append(scores_dict)

    # Concatenate results from all repeats
    if len(results) == 0:
        return None

    fitted = {
        k: np.concatenate([r[k] for r in results])
        if isinstance(results[0][k], np.ndarray)
        else list(itertools.chain.from_iterable(r[k] for r in results))
        for k in results[0].keys()
    }
    return fitted


def get_ensemble_predictions(
    model_key: DecoderFitKey,
    models: list[ClassifierLike],
    epochs: dict[str, mne.Epochs],
    target: Literal["categorical_acoustic_cue", "subject_specific_acoustics"],
) -> pd.DataFrame:
    """
    Get predictions from an ensemble of fit single-electrode models on held-out epochs,
    subsetting appropriately to match the properties of the data the model
    was fit on.

    Args:
        model_key: tuple of `(subject, electrode_idx, phoneme_pair, smin, smax)`;
            i.e. the keys of the dictionary returned by `run_decoding_analysis_single_electrode`
        models: list of fitted models
        epochs: dict mapping from subject to mne.Epochs containing the held-out epochs
        target: which target variable to evaluate the model on
    """
    subject, electrode_idx, phoneme_pair, smin, smax = model_key

    epochs_ij = epochs[subject]
    assert epochs_ij.metadata is not None
    assert target in epochs_ij.metadata.columns
    selection = epochs_ij.metadata.phoneme_pair == phoneme_pair
    if selection.sum() == 0:
        raise ValueError(
            f"No epochs found for subject {subject}, "
            f"phoneme pair {phoneme_pair} in the given epochs."
        )

    X = epochs_ij.get_data(picks=[electrode_idx])[selection][:, 0, smin:smax]

    outcomes = []
    for i, model in enumerate(models):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        outcomes.append(
            pd.DataFrame(
                {
                    "epoch_idx": epochs_ij.metadata.index[selection].values,
                    "decoder_target": epochs_ij.metadata[target][selection].values,
                    "decoder_prediction": y_pred,
                    "decoder_proba": y_proba,
                    "fold": i,
                }
            )
        )

    return pd.concat(outcomes)


def get_population_ensemble_predictions(
    model_key: PopulationDecoderFitKey,
    models: list[ClassifierLike],
    electrode_idxs: list[int],
    epochs: mne.Epochs,
) -> pd.DataFrame:
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
        raise ValueError(
            f"No epochs found for subject {subject}, "
            f"phoneme pair {phoneme_pair} in the given epochs."
        )

    X = epochs.get_data(picks=electrode_idxs)[selection][:, :, smin:smax]
    X = X.reshape(X.shape[0], -1)  # flatten space * time

    outcomes = []
    for i, model in enumerate(models):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        outcomes.append(
            pd.DataFrame(
                {
                    "epoch_idx": epochs.metadata.index[selection].values,
                    "decoder_target": epochs.metadata.categorical_acoustic_cue[
                        selection
                    ].values,
                    "decoder_prediction": y_pred,
                    "decoder_proba": y_proba,
                    "fold": i,
                }
            )
        )

    return pd.concat(outcomes)
