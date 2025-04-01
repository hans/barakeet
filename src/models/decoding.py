
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from tqdm.auto import tqdm


def run_decoding_analysis_single_electrode(
        epochs, electrode_df, stride, window_size,
        target: Literal["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right"] = "lexical_evidence",
        filter_speech_responsive=True,
        return_outcomes=True,
        include_only_full_windows=True,
        smoke_test=False,
        randomize=False):
    """
    stride: in samples
    window_size: in samples
    """

    if target not in ["acoustic", "lexical_evidence", "mismatch", "mismatch_left_right"]:
        raise ValueError(f"Invalid target {target}")

    global_min_sample = 0
    global_max_sample = min([epoch.get_data().shape[2] for epoch in epochs.values()])
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

                cv_inner = StratifiedKFold(2, shuffle=True, random_state=42)
                cv_outer = StratifiedKFold(2, shuffle=True, random_state=42)

                Cs = np.logspace(-3, 2, 6)

                pipeline = [StandardScaler()]

                solver = "liblinear" if num_classes == 2 else "saga"
                pipeline.append(LogisticRegressionCV(
                    Cs=Cs, cv=cv_inner, max_iter=100000, n_jobs=1,
                    class_weight="balanced", fit_intercept=False,
                    solver=solver))
                model = make_pipeline(*pipeline)
                scoring = ["roc_auc", "f1_macro", "accuracy"] if num_classes == 2 else ["f1_macro", "accuracy"]
                fitted = cross_validate(model, X, y, cv=cv_outer, scoring=scoring,
                                        return_estimator=True,
                                        return_train_score=True,
                                        n_jobs=2)

                result_key = (row.subject, row.electrode_idx, phoneme_pair, smin, smax)

                if isinstance(scoring, list):
                    train_scores[result_key] = {k: fitted["train_" + k] for k in scoring}
                    test_scores[result_key] = {k: fitted["test_" + k] for k in scoring}
                else:
                    train_scores[result_key] = fitted["train_score"]
                    test_scores[result_key] = fitted["test_score"]

                if return_outcomes:
                    # only store outcomes on test folds
                    outcomes[result_key] = pd.concat([
                        pd.DataFrame({"decoder_target": y[test_idxs],
                                    "decoder_prediction": estimator.predict(X[test_idxs]),
                                    "decoder_proba": estimator.predict_proba(X[test_idxs])[:, 1],
                                    "fold": fold},
                                    index=test_idxs)
                        for fold, ((_, test_idxs), estimator) in enumerate(zip(cv_outer.split(X, y), fitted["estimator"]))
                    ])

                models[result_key] = fitted["estimator"]

    return train_scores, test_scores, outcomes, models