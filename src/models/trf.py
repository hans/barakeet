
from typing import Optional

import mne
from mne.decoding import ReceptiveField
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, check_is_fitted
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


class TRF(BaseEstimator):
    """
    combines data prep + model fitting in a single estimator
    """
    def __init__(self, estimator, ep: mne.Epochs, features_per_phoneme_pair,
                 fit_electrodes: Optional[list[int]] = None,
                 global_features: Optional[list[tuple[str, str]]] = None):
        self.estimator = estimator
        self.ep = ep
        self.features_per_phoneme_pair = features_per_phoneme_pair
        self.fit_electrodes = fit_electrodes
        self.global_features = global_features or []

        self._update_metadata()

    def _update_metadata(self):
        self.phoneme_pairs = sorted(set(self.ep.metadata.phoneme_pair))

        global_feature_names = [feature_name for feature_name, _ in self.global_features]
        per_phoneme_feature_names = [
            f"{feature_name}-{phoneme_pair}"
            for feature_name, _ in self.features_per_phoneme_pair
            for phoneme_pair in self.phoneme_pairs
        ]
        self.feature_names = global_feature_names + per_phoneme_feature_names

    def set_params(self, **params):
        super().set_params(**params)
    
        if "features_per_phoneme_pair" in params or "ep" in params:
            self._update_metadata()

        return self

    def _prepare_design_matrix(self, idxs):
        # HACK this is incorporated here rather than in a Pipeline because we need to
        # modify both X and Y

        X, Y = [], []
        md = self.ep.metadata

        for idx in idxs:
            ep_i = self.ep[idx].get_data().squeeze(0)
            # ep_i: n_channels * n_samples
            md_i = md.iloc[idx]

            # build up design matrix for this trial by column
            Xi = []
            for feature_name, feature_alignment in self.global_features:
                Xij = np.zeros((ep_i.shape[1], 1))
                if feature_alignment == "onset":
                    onset_time = 0. - self.ep.tmin
                    onset_sample = int(onset_time * self.ep.info["sfreq"])
                    Xij[onset_sample] = md_i[feature_name]
                else:
                    raise ValueError(f"Unknown feature alignment: {feature_alignment}")
                Xi.append(Xij)

            for feature_name, feature_alignment in self.features_per_phoneme_pair:
                for phoneme_pair in self.phoneme_pairs:
                    Xij = np.zeros((ep_i.shape[1], 1))
                    if phoneme_pair == md_i.phoneme_pair:
                        if feature_alignment == "onset":
                            onset_time = 0. - self.ep.tmin
                            onset_sample = int(onset_time * self.ep.info["sfreq"])
                            Xij[onset_sample] = md_i[feature_name]
                        elif feature_alignment == "PoD":
                            PoD_time = md_i.point_of_disambiguation - self.ep.tmin
                            PoD_sample = int(PoD_time * self.ep.info["sfreq"])
                            Xij[PoD_sample] = md_i[feature_name]
                        else:
                            raise ValueError(f"Unknown feature alignment: {feature_alignment}")
                    Xi.append(Xij)
            Xi = np.concatenate(Xi, axis=1)
            X.append(Xi)

            Y.append(ep_i)

        if len(X) == 0:
            raise ValueError("No features")

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=1)

        if self.fit_electrodes is not None:
            Y = Y[self.fit_electrodes, :]

        Y = Y.T

        return X, Y

    def fit(self, idxs, y=None):
        X, Y = self._prepare_design_matrix(idxs)

        est = clone(self.estimator)
        est.feature_names = self.feature_names
        self.estimator_ = est.fit(X, Y)

        return self

    def score_multidimensional(self, idxs, y=None):
        check_is_fitted(self)
        X, Y = self._prepare_design_matrix(idxs)

        # returns one score per output channel
        scores = self.estimator_.score(X, Y)
        return scores
    
    def score(self, idxs, y=None):
        check_is_fitted(self)
        scores = self.score_multidimensional(idxs)

        # take mean across electrodes, ignoring negative results
        scores[scores < 0] = np.nan
        if np.isnan(scores).all():
            return 0
        return np.nanmean(scores)

    @property
    def coef_(self):
        return self.estimator_.coef_
    
    def get_feature_names(self, feature_spec):
        feature_name, _ = feature_spec
        return [f"{feature_name}-{phoneme_pair}" for phoneme_pair in self.phoneme_pairs]

    def predict(self, idxs):
        check_is_fitted(self)
        X, _ = self._prepare_design_matrix(idxs)
        return self.estimator_.predict(X)


def estimate_trf_unique_variance(feature_blocks, features_per_phoneme_pair,
                                 estimator: TRF, train_idxs, test_idxs):
    overall_score = estimator.score_multidimensional(test_idxs)
    
    scored_blocks, scores = [], []
    for feature_block_name, features in feature_blocks.items():
        estimator_modified = clone(estimator)
        estimator_modified.set_params(
            features_per_phoneme_pair=[f for f in features_per_phoneme_pair if f[0] not in features])
        
        estimator_modified.fit(train_idxs)
        score = estimator_modified.score_multidimensional(test_idxs)
        # ignore subzero score
        score[score < 0] = np.nan

        scored_blocks.append(feature_block_name)
        scores.append(score)

    # electrodes with subzero score should not be considered
    reference_score = overall_score.copy()
    reference_score[reference_score < 0] = np.nan
    score_deltas = reference_score[None, :] - np.array(scores)

    return overall_score, scored_blocks, score_deltas


def estimate_trf(ep: mne.Epochs, feature_blocks, features_per_phoneme_pair, num_folds=4):
    Cs = np.logspace(-2, 3, 5)
    md = ep.metadata
    epoch_idxs = md.index.values

    estimator = ReceptiveField(tmin=-0.1, tmax=0.7, sfreq=ep.info["sfreq"])
    outer_cv = StratifiedKFold(num_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(num_folds, shuffle=True, random_state=42)

    pipeline = TRF(estimator, ep, features_per_phoneme_pair)
    param_grid = {"estimator__estimator": Cs}

    from sklearn.model_selection import GridSearchCV, cross_validate
    from sklearn.metrics import r2_score, make_scorer
    clf = GridSearchCV(pipeline, param_grid, cv=inner_cv, n_jobs=1)
    stratify_class = md.loc[epoch_idxs].stratify_class

    cv_results = cross_validate(clf, X=epoch_idxs, y=stratify_class,
                                cv=outer_cv, n_jobs=4,
                                return_estimator=True,
                                return_train_score=True,
                                return_indices=True,
                                verbose=100)

    # Estimate unique variance explained per feature on the outer folds.
    overall_scores, unique_variance_estimates = [], []
    for gs, train_idxs, test_idxs in zip(tqdm(cv_results["estimator"], desc="Estimate unique variance", unit="fold"),
                                         cv_results["indices"]["train"],
                                         cv_results["indices"]["test"]):
        overall_score, block_names, per_block_uv = estimate_trf_unique_variance(
            feature_blocks, features_per_phoneme_pair, gs.best_estimator_, train_idxs, test_idxs
        )
        overall_scores.append(overall_score)
        unique_variance_estimates.append(dict(zip(block_names, per_block_uv)))

    overall_scores = pd.DataFrame(overall_scores)
    overall_scores.index.name = "fold"
    overall_scores.columns.name = "electrode"
    unique_variance_estimates = pd.concat({
            fold: pd.DataFrame.from_dict(fold_uv_estimates, orient="index")
            for fold, fold_uv_estimates in enumerate(unique_variance_estimates)
        }, names=["fold", "feature_block"])
    unique_variance_estimates.columns.name = "electrode"
    
    return cv_results, overall_scores, unique_variance_estimates