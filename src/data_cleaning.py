from matplotlib import axis
import mne
import numpy as np
import pandas as pd
from scipy import stats


# Post-hoc relabeling of particular ROIs, after manual inspection
# This really should happen prior to unifying the As, but for now
# we are sticking to the current results and relabeling post-hoc
roi_updates = {
    # TODO
    # EC279 [] pb: 158

    ("EC287", "[]", "dn"): "precentral",
    ("EC287", "Left-Cerebral-White-Matter", "dn"): "insula",
    ("EC287", "Left-Cerebral-White-Matter", "pb"): "cingulate",
    ("EC279", "ctx_rh_S_circular_insula_sup", "dn"): "frontal operculum",
    ("EC279", "ctx_rh_S_circular_insula_inf", "dn"): "temporal operculum",
    ("EC279", "ctx_rh_G_temp_sup-G_T_transv", "dn"): "temporal operculum",
    ("EC279", "Right-Cerebral-White-Matter", "bm"): "temporal operculum",
    ("EC279", "Right-Cerebral-White-Matter", "dn"): "temporal operculum",
    ("EC279", "Unknown", "bm"): "temporal operculum",
}


def prepare_ABC_results(A_path, B_path, C_path,
                        acoustic_decoding_path=None,
                        trf_results_path=None):
    """
    Do post-hoc cleaning on the results of the causal4 pipeline.
    """
    A_results = pd.read_csv(A_path)
    B_results = pd.read_csv(B_path)
    C_results = pd.read_csv(C_path)

    C_results = C_results.drop_duplicates(subset=["subject", "electrode_idx", "phoneme_pair"])
    C_results = C_results.loc[~C_results["Temporal pattern"].isna()]

    if trf_results_path is not None:
        trf_results = pd.read_csv(trf_results_path)
        trf_results = trf_results.dropna()
        # average across folds
        trf_results = trf_results.rename(columns={"output_dim": "electrode_idx"}) \
            .groupby(["subject", "electrode_idx"]).score.mean()

        # Merge this into B_results
        B_results = B_results.join(trf_results.rename("trf_r2"), on=["subject", "electrode_idx"])

        # Merge this into C_results
        C_results = C_results.join(trf_results.rename("trf_r2"), on=["subject", "electrode_idx"])

    if acoustic_decoding_path is not None:
        acoustic_decoding_scores = pd.read_csv(acoustic_decoding_path)
        acoustic_decoding_scores["tmin"] = acoustic_decoding_scores.smin / 100 - 0.4
        acoustic_decoding_scores["tmax"] = acoustic_decoding_scores.smax / 100 - 0.4
        acoustic_decoding_scores = acoustic_decoding_scores.groupby(["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "tmin", "tmax"]).roc_auc.mean().reset_index().set_index(["subject", "electrode_idx", "phoneme_pair"])

        # Merge into B_results
        B_results["acoustic_decoding_roc_auc"] = B_results.apply(
            lambda row: acoustic_decoding_scores.loc[row.subject, row.electrode_idx, row.phoneme_pair].query("tmax < @row.window_start").roc_auc.max(), axis=1)
        
        # Merge into C_results
        C_results["acoustic_decoding_roc_auc"] = C_results.apply(
            lambda row: acoustic_decoding_scores.loc[row.subject, row.electrode_idx, row.phoneme_pair].query("tmax < @row.window_start").roc_auc.max(), axis=1)

    integration_dfs = {"B": B_results, "C": C_results}
    new_integration_dfs = {}
    for key, integration_df in integration_dfs.items():
        # Check schema.
        assert {"Temporal pattern", "Morphology", "Left polarity", "Right polarity",
                "Tracking resampled in A window?", "Timit tuning"} < set(integration_df.columns)

        polarity_columns = ["Left polarity", "Right polarity"]
        for col in polarity_columns:
            assert set(integration_df[col].unique()) == {"+", "-", np.nan}
            new_col = col.lower().replace(" ", "_")
            integration_df[new_col] = integration_df[col].replace({"+": 1, "-": -1, np.nan: 0}).astype(int)

        integration_df["one_sided_positive"] = False
        integration_df["one_sided_negative"] = False
        integration_df.loc[((integration_df["left_polarity"] == 1) & (integration_df["right_polarity"] == 0)) |
                    ((integration_df["left_polarity"] == 0) & (integration_df["right_polarity"] == 1)),
                    "one_sided_positive"] = True
        integration_df.loc[((integration_df["left_polarity"] == -1) & (integration_df["right_polarity"] == 0)) |
                    ((integration_df["left_polarity"] == 0) & (integration_df["right_polarity"] == -1)),
                    "one_sided_negative"] = True
        integration_df["one_sided"] = integration_df.one_sided_negative | integration_df.one_sided_positive

        # Make sure one_sided_negative and one_sided_positive are exclusive
        assert not (integration_df["one_sided_negative"] & integration_df["one_sided_positive"]).any()

        def simplify_tracking_resampled(value):
            if value == "n": return False
            elif value == np.nan: return None
            else: return True
        integration_df["tracking_resampled_A"] = integration_df["Tracking resampled in A window?"] \
            .map(simplify_tracking_resampled)

        integration_df = integration_df[~integration_df["Temporal pattern"].isin(("drop", "dupe")) &
                                        ~integration_df["Temporal pattern"].isna()]
        
        integration_df[["left_phoneme", "right_phoneme"]] = integration_df.phoneme_pair.str.extract(r"([a-z])([a-z])")
        integration_df["left_dominance_str"] = "/" + integration_df.left_phoneme + "/ > /" + integration_df.right_phoneme + "/"
        integration_df["right_dominance_str"] = "/" + integration_df.right_phoneme + "/ > /" + integration_df.left_phoneme + "/"

        integration_df["timit_left_dominant"] = integration_df["Timit tuning"] == integration_df.left_dominance_str
        integration_df["timit_right_dominant"] = integration_df["Timit tuning"] == integration_df.right_dominance_str
        integration_df = integration_df.drop(columns=["left_dominance_str", "right_dominance_str"])
        integration_df["timit_one"] = integration_df["timit_left_dominant"] | integration_df["timit_right_dominant"]
        integration_df["timit_both"] = integration_df["Timit tuning"].str.contains("=")

        assert not (integration_df.timit_left_dominant & integration_df.timit_right_dominant).any()
        assert not (integration_df.timit_one & integration_df.timit_both).any()

        # Drop duplicates (effects at the same site + phoneme pair)
        duplicate_effects = integration_df.groupby(["subject", "electrode_idx", "phoneme_pair", "roi"]).size()
        duplicate_effects = duplicate_effects[duplicate_effects > 1]
        print(f"Sites with duplicate effects ({len(duplicate_effects)}, {len(duplicate_effects) / len(integration_df.groupby(['subject', 'electrode_idx', 'phoneme_pair'])) * 100:.2f}%):")
        print(duplicate_effects.sort_values())

        new_integration_dfs[key] = integration_df

    B_results = new_integration_dfs["B"]
    C_results = new_integration_dfs["C"]

    # Take the strongest effect for each site + phoneme-pair
    B_results = B_results.sort_values("p_val_min").groupby(["subject", "electrode_idx", "phoneme_pair"], as_index=False).first()
    C_results = C_results.sort_values("stim_control_p_val_min").groupby(["subject", "electrode_idx", "phoneme_pair"], as_index=False).first()

    # Post-hoc aggregate A-site values according to updated ROI information
    # We should integrate this into the early unify-A pipeline eventually
    A_results["population_name_fixed"] = A_results["population_name"]
    for (subject, population_name, phoneme_pair), fixed_roi in roi_updates.items():
        A_results.loc[(A_results.subject == subject) & (A_results.population_name == population_name) & (A_results.phoneme_pair == phoneme_pair), "population_name_fixed"] = fixed_roi

    B_results["population_name_fixed"] = B_results["population_name"]
    for (subject, population_name, phoneme_pair), fixed_roi in roi_updates.items():
        B_results.loc[(B_results.subject == subject) & (B_results.population_name == population_name) & (B_results.phoneme_pair == phoneme_pair), "population_name_fixed"] = fixed_roi

    ####

    return A_results, B_results, C_results



def compute_stimulus_correlation(As: pd.DataFrame,
                                 A_decoders: dict[str, dict],
                                 epochs: dict[str, mne.epochs.EpochsFIF],
                                 metric: str = "spearmanr",
                                 return_outcomes=False
                                 ) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """
    For each A site, compute correlation between A population decoder
    output and ground-truth stimulus feature (resampled).
    """
    if metric not in ("spearmanr", "pearsonr"):
        raise ValueError(f"Invalid metric {metric}")
    if As.empty:
        if return_outcomes:
            return pd.Series(dtype=float), pd.DataFrame()
        else:
            return pd.Series(dtype=float)

    results = {}
    all_outcomes = {}

    for row in As.itertuples():
        subject = str(row.subject)
        outcomes = pd.concat([
            # decoder outputs computed on held-out test folds of the training data distribution
            A_decoders[subject]["outcomes"][subject, row.electrode_idx, row.phoneme_pair, row.smin, row.smax],
            # decoder outputs on never-seen held-out test data
            # (typically the non-extreme stimulus steps)
            A_decoders[subject]["held_out_outcomes"][subject, row.electrode_idx, row.phoneme_pair, row.smin, row.smax]
        ])

        # Take mean over predictions from multiple folds
        outcomes = outcomes.groupby("epoch_idx").decoder_proba.mean().reset_index()

        # Now merge with epoch information so we can run correlation
        outcomes = pd.merge(outcomes, epochs[subject].metadata,
                            how="left", left_on="epoch_idx", right_index=True,
                            validate="1:1")

        if metric == "spearmanr":
            corr, pval = stats.spearmanr(outcomes.decoder_proba, outcomes.resampled)
        elif metric == "pearsonr":
            corr, pval = stats.pearsonr(outcomes.decoder_proba, outcomes.resampled)

        results[row.Index] = corr
        all_outcomes[row.Index] = outcomes

    if return_outcomes:
        return pd.Series(results), pd.concat(all_outcomes, names=["A_idx"]).droplevel(-1).set_index("epoch_idx", append=True)
    else:
        return pd.Series(results)