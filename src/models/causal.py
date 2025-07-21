import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils.extmath import softmax


def stratified_nested_cv(X, Y, metadata, idxs_val):
    # This section is more complicated because we need to do nested CV with an auxiliary
    # stratification variable (not the regressor but a metadata attribute).
    # This has to be done manually.

    # Define outer and inner cross-validation
    cv_outer = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Define alpha values for Ridge regression
    alphas = np.logspace(-2, 5, 6)

    # Storage for results
    outer_scores = []
    outer_models = []
    outer_scores_baseline = []
    outer_models_baseline = []

    for train_idx, test_idx in cv_outer.split(X, metadata.iloc[idxs_val].resampled.values):
        # Split data for outer CV
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        X_train_base, X_test_base = X_train[:, :-1], X_test[:, :-1]  # For baseline model

        # Inner CV for hyperparameter tuning
        best_alpha = None
        best_score = -np.inf
        
        for train_inner_idx, val_inner_idx in cv_inner.split(X_train, metadata.iloc[idxs_val[train_idx]].resampled.values):
            X_train_inner, X_val_inner = X_train[train_inner_idx], X_train[val_inner_idx]
            Y_train_inner, Y_val_inner = Y_train[train_inner_idx], Y_train[val_inner_idx]

            # Train RidgeCV on inner loop
            model_inner = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, store_cv_values=True))
            model_inner.fit(X_train_inner, Y_train_inner)
            
            # Evaluate performance on validation set
            score = model_inner.score(X_val_inner, Y_val_inner)
            
            # Store the best performing alpha
            if score > best_score:
                best_score = score
                best_alpha = model_inner.named_steps["ridgecv"].alpha_

        # Train final model on full training set with best alpha
        model_final = make_pipeline(StandardScaler(), RidgeCV(alphas=[best_alpha]))
        model_final.fit(X_train, Y_train)
        score_final = model_final.score(X_test, Y_test)

        # Train baseline model (without last feature)
        model_baseline = make_pipeline(StandardScaler(), RidgeCV(alphas=[best_alpha]))
        model_baseline.fit(X_train_base, Y_train)
        score_baseline = model_baseline.score(X_test_base, Y_test)

        # Store results
        outer_scores.append(score_final)
        outer_models.append(model_final)
        outer_scores_baseline.append(score_baseline)
        outer_models_baseline.append(model_baseline)

    # Convert results to match cross_validate output format
    fitted2 = {
        'test_score': np.array(outer_scores),
        'estimator': np.array(outer_models),
        'num_outputs': Y.shape[1],
    }

    fitted2_base = {
        'test_score': np.array(outer_scores_baseline),
        'estimator': np.array(outer_models_baseline),
        'num_outputs': Y.shape[1],
    }

    print("Final Nested CV Model Scores:", fitted2["test_score"])
    print("Final Nested CV Baseline Scores:", fitted2_base["test_score"])

    return fitted2, fitted2_base


def run_phase1(epochs, subject, phoneme_pair, 
               population, time_window,
               split_strategy="stratify",
               pca_num_components=6):
    """
    Train a decoder of $P(\text{phone})$ on population $A$.
    Hold out a substantial portion of the data for use in the next phase.
    """
    prediction_target = "categorical_acoustic_cue"

    if split_strategy not in ["stratify", "extreme"]:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    epochs_i = epochs[subject][f"phoneme_pair == '{phoneme_pair}'"]
    epochs_ij = epochs_i.copy() \
        .pick(population).crop(*time_window)

    X = epochs_ij.get_data().reshape(len(epochs_ij), -1)
    features_i = [(subject, electrode_idx, time_window[0] + time)
                  for electrode_idx in population
                  for time in epochs_ij.times]
    metadata = epochs_ij.metadata

    features_i = pd.DataFrame(features_i, columns=["subject", "electrode_idx", "time"])
    assert len(features_i) == X.shape[1]

    y = metadata[prediction_target].values

    ####

    if split_strategy == "stratify":
        X_train, X_val, y_train, y_val, idxs_train, idxs_val = \
            train_test_split(X, y, np.arange(X.shape[0]),
                            test_size=0.5, stratify=metadata.resampled.values)
    elif split_strategy == "extreme":
        resampled_extremes = metadata.resampled.min(), metadata.resampled.max()
        mdr = metadata.reset_index()
        idxs_train = mdr[mdr.resampled.isin(resampled_extremes)].index
        idxs_val = mdr[~mdr.resampled.isin(resampled_extremes)].index
        X_train, y_train = X[idxs_train], y[idxs_train]
        X_val, y_val = X[idxs_val], y[idxs_val]

    cv_inner = StratifiedKFold(2, shuffle=True)
    cv_outer = StratifiedKFold(2, shuffle=True)

    pipeline = [StandardScaler()]
    if pca_num_components is not None:
        pipeline.append(PCA(n_components=min(pca_num_components, X.shape[1])))

    Cs = np.logspace(-3, 2, 6)
    pipeline.append(LogisticRegressionCV(Cs=Cs, cv=cv_inner, max_iter=1000))
    model = make_pipeline(*pipeline)
    fitted = cross_validate(model, X_train, y_train, cv=cv_outer, scoring="roc_auc", return_estimator=True)

    # Now get ensembled predictions on the held-out val set; average in logit space
    logits_val = np.stack([est.predict_log_proba(X_val) for est in fitted["estimator"]]).mean(0)
    y_proba_val = softmax(logits_val)
    y_pred_val = np.take(fitted["estimator"][0].classes_,
                        y_proba_val.argmax(axis=1),
                        axis=0)
    
    return fitted, metadata, idxs_val, y_proba_val


def run_phase2(epochs, subject, phoneme_pair,
               population, intervening_window, time_window,
               metadata, idxs_val, y_proba_val, 
               prediction_target="categorical_acoustic_cue",
               pca_num_components=32):
    """"
    Use the output phone probabilities to predict the activity of population B neurons.
    Compare to a model which uses prior population B activity to predict population B activity.

    time window: `(0.6, 0.8)`
    linear regression predicts mean

    baseline: `B ~ prev_B + resampled`

    full model: `B ~ prev_B + resampled + P(phone = d | A)`
    """
    pca_num_components = 32
    X2, Y2, features2_i, metadata2 = [], [], [], []

    epochs_ij = epochs_i = epochs[subject][f"phoneme_pair == '{phoneme_pair}'"] \
        .pick(population)

    epochs_ij_intervening = epochs_ij.copy().crop(*intervening_window)
    epochs_ij_B = epochs_ij.copy().crop(*time_window)

    X2_ij = epochs_ij_intervening.get_data().reshape(len(epochs_ij_intervening), -1)

    Y2_ij = epochs_ij_B.get_data()
    # average over time
    Y2_ij = Y2_ij.mean(axis=2)
    assert Y2_ij.ndim == 2

    X2.append(X2_ij)
    Y2.append(Y2_ij)
    features2_i.extend(population)
    metadata2.append(epochs_ij_B.metadata)

    X2 = np.concatenate(X2, axis=1)
    Y2 = np.concatenate(Y2, axis=1)

    # Subset / reorder Y2 and the X2 nuisance features so that it aligns with the same trial properties as each row of X2
    # TODO this is not right -- need to cross-reference metadata I think
    Y2 = Y2[idxs_val, :]
    X2 = np.concatenate([X2[idxs_val], y_proba_val[:, [1]]], axis=1)

    features2_i = pd.DataFrame(features2_i, columns=["electrode_idx"]).assign(subject=subject)
    assert len(features2_i) == Y2.shape[1]
    assert X2.shape[0] == Y2.shape[0]

    # make sure trials are properly lined up
    for metadata2_i in metadata2[1:]:
        assert (metadata2[0].iloc[idxs_val][[prediction_target, "resampled", "word_end"]].values == metadata2_i.iloc[idxs_val][[prediction_target, "resampled", "word_end"]].values).all()
        assert (metadata.iloc[idxs_val][[prediction_target, "resampled", "word_end"]].values == metadata2_i.iloc[idxs_val][[prediction_target, "resampled", "word_end"]].values).all()

    ####

    return stratified_nested_cv(X2, Y2, metadata2[0], idxs_val)


def run_causal_analysis(epochs, subject, phoneme_pair,
                        population_A, population_A_window,
                        population_B, population_B_window,
                        max_intervening_window_size = 0.1,
                        split_strategy="stratify",
                        pca_num_components=32):
    """
    Run the causal analysis pipeline.
    """
    assert population_A_window[1] < population_B_window[0], "Time windows should have a gap"

    intervening_window = (population_A_window[1], population_B_window[0])
    if intervening_window[1] - intervening_window[0] > max_intervening_window_size:
        intervening_window = (population_B_window[0] - max_intervening_window_size, population_B_window[0])
    
    fitted, metadata, idxs_val, y_proba_val = run_phase1(
        epochs, subject, phoneme_pair,
        population_A, population_A_window,
        split_strategy=split_strategy,)
    fitted2, fitted2_base = run_phase2(epochs, subject, phoneme_pair,
               population_B, intervening_window, population_B_window,
               metadata, idxs_val, y_proba_val)
    
    return fitted, fitted2, fitted2_base, \
        (idxs_val, y_proba_val)
    