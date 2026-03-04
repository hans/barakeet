# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Summarize behavior decoding outcome on a single subject.

# %%
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm
import mne
import re

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from src.data import add_metadata_features
from src.data_cleaning import prepare_ABC_results, compute_stimulus_correlation

# %% tags=["parameters"]
subject = "EC243"

result_path = f"outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt"
groupby = ["word_end"]

# single-electrode stimulus decoding results
A_individual_results_path = f"outputs/causal4/find_As/{subject}_results.csv"
A_individual_decoder_path = f"outputs/causal4/find_As/{subject}_decoders.pt"

A_results_path = "outputs/causal4/unify_As/results.csv"
A_stimulus_decoders_path = "outputs/causal4/unify_As/unified_decoders.pt"
B_results_path = "outputs/causal4/annotated_B_results.csv"
C_results_path = "outputs/causal4/annotated_C_results.csv"

trf_results_path = "/userdata/jgauthier/projects/big-trf/outputs/encoder_summary/timit-no_repeats/vanilla_aud.csv"
acoustic_decoding_scores_path = "outputs/single_electrode_decoding/30/acoustic/scores.csv"

electrodes_path = f"outputs/causal4/find_speech_responsive/{subject}_results.csv"

epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"

textgrid_dir = "textgrids"

# don't include behavioral decoding results from windows before this point
min_decoding_sample = 0
# don't include behavioral decoding results from windows beyond this point
max_decoding_sample = 290 # ~2.5 seconds post onset
# for A's, keep it an earlier window -- we want to see whether the early acoustic response in
# particular is predictive of behavior
A_max_decoding_sample = 70 # 0.3 seconds post onset
# for B's and C's, don't include the acoustic window
BC_min_decoding_sample = 70

phoneme_pair_order = ["bm", "dn", "pb"]

outdir = "."

# %%
# subject may not have been updated in injected params; induce from path
subject = re.findall(r"/behavior_decoding_single_electrode/([^/]+)/", result_path)[0]

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
electrode_df = pd.read_csv(electrodes_path).assign(subject=subject).set_index(["subject", "electrode_idx"])

# %%
A_results, B_results, C_results = prepare_ABC_results(A_results_path, B_results_path, C_results_path,
                                                      trf_results_path=trf_results_path,
                                                      acoustic_decoding_path=acoustic_decoding_scores_path)

# %%
individual_A_results = pd.read_csv(A_individual_results_path).query("A")

# %%
A_individual_stimulus_decoder = torch.load(A_individual_decoder_path)

# %%
individual_A_results["stimulus_correlation"] = compute_stimulus_correlation(
    individual_A_results,
    {subject: A_individual_stimulus_decoder},
    {subject: epochs},
)

# %%
# Merge in TRF information for As
trf_results = pd.read_csv(trf_results_path)
individual_A_results = pd.merge(individual_A_results, trf_results.rename(columns={"output_dim": "electrode_idx"}).groupby(["subject", "electrode_idx"]).score.mean().rename("trf_r2"),
            on=["subject", "electrode_idx"], how="left")

# %%
A_stimulus_decoders = torch.load(A_stimulus_decoders_path)

# %%
behav_decoder_result = torch.load(result_path)

A_decoding_results = behav_decoder_result["A_decoding_results"]
B_decoding_results = behav_decoder_result["B_decoding_results"]
C_decoding_results = behav_decoder_result["C_decoding_results"]

A_decoders = behav_decoder_result["A_decoders"]
B_decoders = behav_decoder_result["B_decoders"]
C_decoders = behav_decoder_result["C_decoders"]

# %%
dec_columns = ['subject', 'population', 'phoneme_pair', 'smin', 'smax', 'fold',
       'baseline_roc_auc', 'full_roc_auc', 'baseline_precision',
       'full_precision', 'baseline_recall', 'full_recall', 'word_end',
       'baseline_clf__C', 'full_clf__C', 'full_prep__pca__pca__n_components']
if len(A_decoding_results) > 0:
    A_results_df = pd.concat(A_decoding_results, names=["subject", "population_name", "phoneme_pair"], ignore_index=True)
else:
    A_results_df = pd.DataFrame(columns=dec_columns)

if len(B_decoding_results) > 0:
    B_results_df = pd.concat(B_decoding_results, names=["subject", "population_name", "phoneme_pair"], ignore_index=True)
else:
    B_results_df = pd.DataFrame(columns=dec_columns)

if len(C_decoding_results) > 0:
    C_results_df = pd.concat(C_decoding_results, names=["subject", "population_name", "phoneme_pair"], ignore_index=True)
else:
    C_results_df = pd.DataFrame(columns=dec_columns)

# %%
A_early_results_df = A_results_df[(A_results_df.smin >= min_decoding_sample) & (A_results_df.smin <= A_max_decoding_sample)]
A_late_results_df = A_results_df[(A_results_df.smin >= A_max_decoding_sample) & (A_results_df.smax <= max_decoding_sample)]
B_results_df = B_results_df[(B_results_df.smin >= BC_min_decoding_sample) & (B_results_df.smax <= max_decoding_sample)]
C_results_df = C_results_df[(C_results_df.smin >= BC_min_decoding_sample) & (C_results_df.smax <= max_decoding_sample)]

# %%
# early and late windows will overlap a little, but we want to make sure we partition the space of smins and smaxes
target_smax = set(smax for smax in A_results_df.smax.unique() if min_decoding_sample <= smax <= max_decoding_sample)
assert set(A_early_results_df.smax) | set(A_late_results_df.smax) == target_smax, \
    f"Missing: {sorted(target_smax - (set(A_early_results_df.smax) | set(A_late_results_df.smax)))}"
assert set(A_early_results_df.smin).isdisjoint(set(A_late_results_df.smin))

# %%
# Make sure we don't actually refer to the old A_results now.
del A_results_df

# %%
A_early_results_df["diff"] = A_early_results_df["full_roc_auc"] - A_early_results_df["baseline_roc_auc"]
A_late_results_df["diff"] = A_late_results_df["full_roc_auc"] - A_late_results_df["baseline_roc_auc"]
B_results_df["diff"] = B_results_df["full_roc_auc"] - B_results_df["baseline_roc_auc"]
C_results_df["diff"] = C_results_df["full_roc_auc"] - C_results_df["baseline_roc_auc"]

# %% [markdown]
# ## Find peaks, filter

# %%
A_early_summary = A_early_results_df \
    .groupby(["subject", "population", "phoneme_pair", "smin", "smax"] + groupby) \
    [["baseline_roc_auc", "full_roc_auc", "diff"]].mean()
A_early_max_points = A_early_summary \
    .groupby(["subject", "population", "phoneme_pair"] + groupby)["diff"].idxmax()
A_early_final_summary = A_early_summary.loc[A_early_max_points]
A_early_final_summary["electrode_idx"] = A_early_final_summary.index.get_level_values("population").astype(int)

# Merge in results from searchlight
A_early_final_summary = pd.merge(
    A_early_final_summary.reset_index(),
    individual_A_results[["subject", "electrode_idx", "phoneme_pair",
                          "roc_auc", "trf_r2", "stimulus_correlation"]].rename(columns={"roc_auc": "searchlight_roc_auc"}),
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="left",
    validate="m:1").set_index(A_early_final_summary.index.names)

# %%
A_summary = A_late_results_df \
    .groupby(["subject", "population", "phoneme_pair", "smin", "smax"] + groupby) \
    [["baseline_roc_auc", "full_roc_auc", "diff"]].mean()
A_max_points = A_summary \
    .groupby(["subject", "population", "phoneme_pair"] + groupby)["diff"].idxmax()
A_final_summary = A_summary.loc[A_max_points]
A_final_summary["electrode_idx"] = A_final_summary.index.get_level_values("population").astype(int)

# Merge in results from searchlight
A_final_summary = pd.merge(
    A_final_summary.reset_index(),
    individual_A_results[["subject", "electrode_idx", "phoneme_pair",
                          "roc_auc", "trf_r2", "stimulus_correlation"]].rename(columns={"roc_auc": "searchlight_roc_auc"}),
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="left",
    validate="m:1").set_index(A_final_summary.index.names)

# %%
B_summary = B_results_df.groupby(["subject", "population", "phoneme_pair", "smin", "smax"] + groupby)[["baseline_roc_auc", "full_roc_auc", "diff"]].mean()
B_max_points = B_summary.groupby(["subject", "population", "phoneme_pair"] + groupby)["diff"].idxmax()
B_final_summary = B_summary.loc[B_max_points]
B_final_summary["electrode_idx"] = B_final_summary.index.get_level_values("population").astype(int)
# B_final_summary.sort_values("diff")

# Merge in results from searchlight
B_final_summary = pd.merge(
    B_final_summary.reset_index(), B_results,
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="inner", validate="m:1").set_index(B_final_summary.index.names)

# %%
C_summary = C_results_df.groupby(["subject", "population", "phoneme_pair", "smin", "smax"] + groupby)[["baseline_roc_auc", "full_roc_auc", "diff"]].mean()
C_max_points = C_summary.groupby(["subject", "population", "phoneme_pair"] + groupby)["diff"].idxmax()
C_final_summary = C_summary.loc[C_max_points]
C_final_summary["electrode_idx"] = C_final_summary.index.get_level_values("population").astype(int)
# C_final_summary.sort_values("diff")

# Merge in results from searchlight
C_final_summary = pd.merge(
    C_final_summary.reset_index(), C_results,
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="inner", validate="m:1").set_index(C_final_summary.index.names)

# %%
words = {
    "bm": ["bountiful", "mountains"],
    "dn": ["desolate", "necessary"],
    "pb": ["penecillin", "beneficial"],
}
def get_active_words(rows):
    active_words = []
    if (rows.left_polarity != 0).any():
        active_words.append(words[rows.phoneme_pair.iloc[0]][0])
    if (rows.right_polarity != 0).any():
        active_words.append(words[rows.phoneme_pair.iloc[0]][1])
    return pd.Series(active_words).rename("word_end")

active_B_words = B_results.groupby(["subject", "electrode_idx", "phoneme_pair"]).apply(get_active_words).reset_index()

# Only keep decoding results on the subset of words for which we saw meaningful B responses
B_final_summary = pd.merge(active_B_words, B_final_summary.reset_index(),
                           on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="inner")
B_final_summary = B_final_summary.set_index(["subject", "population", "phoneme_pair", "smin", "smax", "word_end"])

# %%
active_C_words = C_results.groupby(["subject", "electrode_idx", "phoneme_pair"]).apply(get_active_words).reset_index()
C_final_summary = pd.merge(active_C_words, C_final_summary.reset_index(),
                           on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="inner")
C_final_summary = C_final_summary.set_index(["subject", "population", "phoneme_pair", "smin", "smax", "word_end"])

# %%
all_summary = pd.concat({"A": A_final_summary,
                         "A_early": A_early_final_summary,
                         "B": B_final_summary, "C": C_final_summary,
                         },
                        names=["source"]).reset_index()

# %%
A_late_results_df.to_csv(Path(outdir) / f"A_results.csv")
A_early_results_df.to_csv(Path(outdir) / f"A_early_results.csv")
B_results_df.to_csv(Path(outdir) / f"B_results.csv")
C_results_df.to_csv(Path(outdir) / f"C_results.csv")

# %%
A_final_summary.to_csv(Path(outdir) / f"A_final_summary.csv")
A_early_final_summary.to_csv(Path(outdir) / f"A_early_final_summary.csv")
B_final_summary.to_csv(Path(outdir) / f"B_final_summary.csv")
C_final_summary.to_csv(Path(outdir) / f"C_final_summary.csv")
all_summary.to_csv(Path(outdir) / f"all_summary.csv")

# %% [markdown]
# ## Compare predictions at max point

# %%
A_decoder_predictions, B_decoder_predictions, C_decoder_predictions = [], [], []
for decs in A_decoders.values():
    for (subject, electrode_idx, phoneme_pair, (word_end,), smin, smax, fold), dec_detail in decs.items():
        # legacy
        if "test_predictions" not in dec_detail:
            raise
        A_decoder_predictions.append(dec_detail["test_predictions"].assign(
            subject=subject, electrode_idx=int(electrode_idx),
            phoneme_pair=phoneme_pair, word_end=word_end,
            smin=smin, smax=smax,
            fold=fold))
for decs in B_decoders.values():
    for (subject, electrode_idx, phoneme_pair, (word_end,), smin, smax, fold), dec_detail in decs.items():
        if "test_predictions" not in dec_detail:
            raise
        B_decoder_predictions.append(dec_detail["test_predictions"].assign(
            subject=subject, electrode_idx=int(electrode_idx),
            phoneme_pair=phoneme_pair, word_end=word_end,
            smin=smin, smax=smax,
            fold=fold))
for decs in C_decoders.values():
    for (subject, electrode_idx, phoneme_pair, (word_end,), smin, smax, fold), dec_detail in decs.items():
        if "test_predictions" not in dec_detail:
            raise
        C_decoder_predictions.append(dec_detail["test_predictions"].assign(
            subject=subject, electrode_idx=int(electrode_idx),
            phoneme_pair=phoneme_pair, word_end=word_end,
            smin=smin, smax=smax,
            fold=fold))


pred_columns = ['decoder_target', 'baseline_decoder_prediction',
       'baseline_decoder_proba', 'full_decoder_prediction',
       'full_decoder_proba', 'fold', 'epoch_idx', 'subject', 'electrode_idx',
       'phoneme_pair', 'word_end', 'smin', 'smax']
if len(A_decoder_predictions) > 0:
    A_decoder_predictions = pd.concat(A_decoder_predictions, ignore_index=True)

    A_early_decoder_predictions = A_decoder_predictions[
        (A_decoder_predictions.smin >= min_decoding_sample) & (A_decoder_predictions.smin <= A_max_decoding_sample)
    ]
    A_late_decoder_predictions = A_decoder_predictions[
        (A_decoder_predictions.smin >= A_max_decoding_sample) & (A_decoder_predictions.smax <= max_decoding_sample)
    ]

    # make sure we partitioned the space of smins correctly, matching the overall results
    assert set(A_early_decoder_predictions.smin) == set(A_early_results_df.smin), \
        f"Mismatch in early smins: {sorted(set(A_early_decoder_predictions.smin) - set(A_early_results_df.smin))}"
    assert set(A_late_decoder_predictions.smin) == set(A_late_results_df.smin), \
        f"Mismatch in late smins: {sorted(set(A_late_decoder_predictions.smin) - set(A_late_results_df.smin))}"
    # just for sanity
    assert set(A_late_decoder_predictions.smin).isdisjoint(set(A_early_decoder_predictions.smin)), \
        f"Overlap in smins: {sorted(set(A_late_decoder_predictions.smin) & set(A_early_decoder_predictions.smin))}"
else:
    A_decoder_predictions = pd.DataFrame(columns=pred_columns)
    A_early_decoder_predictions = pd.DataFrame(columns=pred_columns)

if len(B_decoder_predictions) > 0:
    B_decoder_predictions = pd.concat(B_decoder_predictions, ignore_index=True)

    B_decoder_predictions = B_decoder_predictions[
        (B_decoder_predictions.smin >= BC_min_decoding_sample) & (B_decoder_predictions.smax <= max_decoding_sample)
    ]
else:
    B_decoder_predictions = pd.DataFrame(columns=pred_columns)

if len(C_decoder_predictions) > 0:
    C_decoder_predictions = pd.concat(C_decoder_predictions, ignore_index=True)

    C_decoder_predictions = C_decoder_predictions[
        (C_decoder_predictions.smin >= BC_min_decoding_sample) & (C_decoder_predictions.smax <= max_decoding_sample)
    ]
else:
    C_decoder_predictions = pd.DataFrame(columns=pred_columns)

# %%
decoder_site_key = ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]


# %%
def inspect_behavior_trials(predictions_df, behavior_df, epochs, ax=None):
    all_metadata = epochs.metadata.copy().rename_axis("epoch_idx").reset_index().assign(subject=subject)
    ensembled_predictions = predictions_df.groupby(decoder_site_key + ["epoch_idx"])[["full_decoder_proba", "baseline_decoder_proba", "decoder_target"]].mean()
    study_ensemble = pd.merge(behavior_df, ensembled_predictions.reset_index(), on=decoder_site_key, how="left")
    study_ensemble = pd.merge(study_ensemble, all_metadata[["subject", "epoch_idx", "resampled", "behavior_based_belief_update"]],
                              on=["subject", "epoch_idx"], how="left", validate="m:1")

    study_ensemble["baseline_prediction_new"] = study_ensemble.baseline_decoder_proba > study_ensemble.baseline_decoder_proba.mean()
    study_ensemble["full_prediction"] = study_ensemble.full_decoder_proba > 0.5
    study_ensemble["changed"] = study_ensemble.full_prediction != study_ensemble.baseline_prediction_new
    study_ensemble["correct"] = study_ensemble.full_prediction == study_ensemble.decoder_target

    # print(study_ensemble.groupby(["word_end_x", "changed"]).resampled.std())
    study_ensemble["abs_behavior_based_belief_update"] = study_ensemble.behavior_based_belief_update.abs()

    # print(study_ensemble.groupby(["word_end_x", "changed"]).abs_behavior_based_belief_update.mean())

    # sns.catplot(data=study_ensemble, x="word_end_x", y="resampled", hue="changed", kind="box", aspect=2)
    # sns.catplot(data=study_ensemble, x="word_end_x", y="abs_behavior_based_belief_update", hue="changed", kind="violin", aspect=2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(data=study_ensemble, x="resampled", hue="changed",
                 stat="density", common_norm=False, bins=30, discrete=True, ax=ax)
    ax.set_xlabel("Stimulus step")
    ax.set_xticks([1,2,3,4,5,6])
    # set legend title
    legend_handles = ax.get_legend().legend_handles
    ax.legend(title="Decoder prediction\nchanged", handles=legend_handles, labels=["No", "Yes"],
              loc="upper left", bbox_to_anchor=(0.7, 1))
    
    return ax, study_ensemble


# %%
if len(A_early_decoder_predictions) > 0:
    ax, A_early_trial_analysis = inspect_behavior_trials(A_early_decoder_predictions, A_early_final_summary, epochs)
else:
    A_early_trial_analysis = pd.DataFrame()

# %%
if len(A_decoder_predictions) > 0:
    ax, A_trial_analysis = inspect_behavior_trials(A_decoder_predictions, A_final_summary, epochs)
else:
    A_trial_analysis = pd.DataFrame()

# %%
if len(B_decoder_predictions) > 0:
    ax, B_trial_analysis = inspect_behavior_trials(B_decoder_predictions, B_final_summary, epochs)
else:
    B_trial_analysis = pd.DataFrame()

# %%
if len(C_decoder_predictions) > 0:
    ax, C_trial_analysis = inspect_behavior_trials(C_decoder_predictions, C_final_summary, epochs)
else:
    C_trial_analysis = pd.DataFrame()


# %% [markdown]
# ## Store all predictions and change analysis

# %%
def ensemble_trial_predictions(predictions_df, epochs):
    all_metadata = epochs.metadata.copy().rename_axis("epoch_idx").reset_index().assign(subject=subject)
    ensembled_predictions = predictions_df.groupby(decoder_site_key + ["epoch_idx"])[["full_decoder_proba", "baseline_decoder_proba", "decoder_target"]].mean()
    study_ensemble = pd.merge(ensembled_predictions.reset_index(), all_metadata[["subject", "epoch_idx", "resampled", "behavior_based_belief_update"]],
                              on=["subject", "epoch_idx"], how="left", validate="m:1")

    study_ensemble["baseline_prediction_new"] = study_ensemble.baseline_decoder_proba > study_ensemble.baseline_decoder_proba.mean()
    study_ensemble["full_prediction"] = study_ensemble.full_decoder_proba > 0.5
    study_ensemble["changed"] = study_ensemble.full_prediction != study_ensemble.baseline_prediction_new
    study_ensemble["correct"] = study_ensemble.full_prediction == study_ensemble.decoder_target

    # print(study_ensemble.groupby(["word_end_x", "changed"]).resampled.std())
    study_ensemble["abs_behavior_based_belief_update"] = study_ensemble.behavior_based_belief_update.abs()

    # print(study_ensemble.groupby(["word_end_x", "changed"]).abs_behavior_based_belief_update.mean())
    
    return study_ensemble


# %%
A_early_decoder_predictions.to_parquet(Path(outdir) / "A_early-predictions.parquet")
A_late_decoder_predictions.to_parquet(Path(outdir) / "A-predictions.parquet")
B_decoder_predictions.to_parquet(Path(outdir) / "B-predictions.parquet")
C_decoder_predictions.to_parquet(Path(outdir) / "C-predictions.parquet")

# %%
ensemble_trial_predictions(A_early_decoder_predictions, epochs) \
    .to_csv(Path(outdir) / f"A_early-trial_analysis-ensembled.csv")
ensemble_trial_predictions(A_late_decoder_predictions, epochs) \
    .to_csv(Path(outdir) / f"A-trial_analysis-ensembled.csv")
ensemble_trial_predictions(B_decoder_predictions, epochs) \
    .to_csv(Path(outdir) / f"B-trial_analysis-ensembled.csv")
ensemble_trial_predictions(C_decoder_predictions, epochs) \
    .to_csv(Path(outdir) / f"C-trial_analysis-ensembled.csv")
