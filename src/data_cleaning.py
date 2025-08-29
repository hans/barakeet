import numpy as np
import pandas as pd


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


def prepare_AB_results(A_path, B_path,
                       acoustic_decoding_path=None,
                       trf_results_path=None):
    """
    Do post-hoc cleaning on the results of the causal4 pipeline.
    """
    A_results = pd.read_csv(A_path)
    B_results = pd.read_csv(B_path)

    if trf_results_path is not None:
        trf_results = pd.read_csv(trf_results_path)
        trf_results = trf_results.dropna()
        # average across folds
        trf_results = trf_results.rename(columns={"output_dim": "electrode_idx"}) \
            .groupby(["subject", "electrode_idx"]).score.mean()

        # Merge this into B_results
        B_results = B_results.join(trf_results.rename("trf_r2"), on=["subject", "electrode_idx"])

    if acoustic_decoding_path is not None:
        acoustic_decoding_scores = pd.read_csv(acoustic_decoding_path)
        acoustic_decoding_scores["tmin"] = acoustic_decoding_scores.smin / 100 - 0.4
        acoustic_decoding_scores["tmax"] = acoustic_decoding_scores.smax / 100 - 0.4
        acoustic_decoding_scores = acoustic_decoding_scores.groupby(["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "tmin", "tmax"]).roc_auc.mean().reset_index().set_index(["subject", "electrode_idx", "phoneme_pair"])

        # Merge into B_results
        B_results["acoustic_decoding_roc_auc"] = B_results.apply(
            lambda row: acoustic_decoding_scores.loc[row.subject, row.electrode_idx, row.phoneme_pair].query("tmax < @row.window_start").roc_auc.max(), axis=1)

    # Check B schema.
    assert {"Temporal pattern", "Morphology", "Left polarity", "Right polarity",
            "Tracking resampled in A window?", "Timit tuning"} < set(B_results.columns)

    polarity_columns = ["Left polarity", "Right polarity"]
    for col in polarity_columns:
        assert set(B_results[col].unique()) == {"+", "-", np.nan}
        new_col = col.lower().replace(" ", "_")
        B_results[new_col] = B_results[col].replace({"+": 1, "-": -1, np.nan: 0}).astype(int)

    B_results["one_sided_positive"] = False
    B_results["one_sided_negative"] = False
    B_results.loc[((B_results["left_polarity"] == 1) & (B_results["right_polarity"] == 0)) |
                ((B_results["left_polarity"] == 0) & (B_results["right_polarity"] == 1)),
                "one_sided_positive"] = True
    B_results.loc[((B_results["left_polarity"] == -1) & (B_results["right_polarity"] == 0)) |
                ((B_results["left_polarity"] == 0) & (B_results["right_polarity"] == -1)),
                "one_sided_negative"] = True

    # Make sure one_sided_negative and one_sided_positive are exclusive
    assert not (B_results["one_sided_negative"] & B_results["one_sided_positive"]).any()

    def simplify_tracking_resampled(value):
        if value == "n": return False
        elif value == np.nan: return None
        else: return True
    B_results["tracking_resampled_A"] = B_results["Tracking resampled in A window?"] \
        .map(simplify_tracking_resampled)

    B_results = B_results[(B_results["Temporal pattern"] != "drop") &
                        ~B_results["Temporal pattern"].isna()]
    
    B_results[["left_phoneme", "right_phoneme"]] = B_results.phoneme_pair.str.extract(r"([a-z])([a-z])")
    B_results["left_dominance_str"] = "/" + B_results.left_phoneme + "/ > /" + B_results.right_phoneme + "/"
    B_results["right_dominance_str"] = "/" + B_results.right_phoneme + "/ > /" + B_results.left_phoneme + "/"

    B_results["timit_left_dominant"] = B_results["Timit tuning"] == B_results.left_dominance_str
    B_results["timit_right_dominant"] = B_results["Timit tuning"] == B_results.right_dominance_str
    B_results = B_results.drop(columns=["left_dominance_str", "right_dominance_str"])
    B_results["timit_one"] = B_results["timit_left_dominant"] | B_results["timit_right_dominant"]
    B_results["timit_both"] = B_results["Timit tuning"].str.contains("=")

    assert not (B_results.timit_left_dominant & B_results.timit_right_dominant).any()
    assert not (B_results.timit_one & B_results.timit_both).any()

    # Drop duplicates (effects at the same site + phoneme pair)
    duplicate_effects = B_results.groupby(["subject", "electrode_idx", "phoneme_pair", "roi"]).size()
    duplicate_effects = duplicate_effects[duplicate_effects > 1]
    print(f"Sites with duplicate effects ({len(duplicate_effects)}, {len(duplicate_effects) / len(B_results.groupby(['subject', 'electrode_idx', 'phoneme_pair'])) * 100:.2f}%):")
    print(duplicate_effects.sort_values())

    # Take the strongest effect for each site + phoneme-pair
    B_results = B_results.sort_values("p_val_min").groupby(["subject", "electrode_idx", "phoneme_pair"], as_index=False).first()

    # Post-hoc aggregate A-site values according to updated ROI information
    # We should integrate this into the early unify-A pipeline eventually
    A_results["population_name_fixed"] = A_results["population_name"]
    for (subject, population_name, phoneme_pair), fixed_roi in roi_updates.items():
        A_results.loc[(A_results.subject == subject) & (A_results.population_name == population_name) & (A_results.phoneme_pair == phoneme_pair), "population_name_fixed"] = fixed_roi

    B_results["population_name_fixed"] = B_results["population_name"]
    for (subject, population_name, phoneme_pair), fixed_roi in roi_updates.items():
        B_results.loc[(B_results.subject == subject) & (B_results.population_name == population_name) & (B_results.phoneme_pair == phoneme_pair), "population_name_fixed"] = fixed_roi

    ####

    return A_results, B_results