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
# # Sliding-window multivariate ambiguity decoder
#
# Decode trial-level ambiguity (behaviorally ambiguous vs. unambiguous) from a
# sliding-window feature vector built over acoustically-selective STG sites,
# per phoneme pair. Stratified CV balances fold composition on the
# `(resampled, word_end)` composite — implicitly holding the ambiguity label,
# step identity, and completion fixed across folds. See the plan at
# `plans/multivariate-ambiguity-decoding.md`.

# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# %%
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from src.data import add_metadata_features, get_ambiguous_resampled_steps
from src.models.decoding import _prepare_decoding_population, fit_train_test

# %% tags=["parameters"]
subject = "EC250"

epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
phon_peaks_path = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet"
all_md_path = "outputs/causal5/prepare_neurometrics/all_md.parquet"
outdir = "."

window_sizes = [10]
stride = 2
pca_num_components = "auto"
n_jobs = 4

phon_response_peak_threshold = 0.65
ambiguous_response_threshold = 2
stratum_min_trials = 5

epoch_tmin = -0.4
epoch_sfreq = 100

# %%
subject = Path(epochs_path).name.split("_")[0]
outdir = Path(outdir)

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
phon_peaks_df = pd.read_parquet(phon_peaks_path)
all_md = pl.read_parquet(all_md_path)

# %% [markdown]
# ## Trial-level ambiguity labels
#
# `get_ambiguous_resampled_steps(all_md)` returns `{(subject, phoneme_pair, word_end): [steps]}`
# — steps where this subject gave variable reports in this completion. Trials whose
# `(subject, pair, word_end, resampled)` matches are ambiguous; everything else
# (endpoints + middle steps the subject resolved) is unambiguous. Mirrors
# [notebooks/causal4/A_ambiguity_selective_decoding.py:90-136](notebooks/causal4/A_ambiguity_selective_decoding.py#L90-L136).

# %%
ambig_map = get_ambiguous_resampled_steps(
    all_md, ambiguous_response_threshold=ambiguous_response_threshold
)

ambig_rows = []
for (subj_i, pair_i, word_end_i), steps in ambig_map.items():
    if subj_i != subject:
        continue
    for step in steps:
        ambig_rows.append(
            {
                "phoneme_pair": pair_i,
                "word_end": word_end_i,
                "resampled": float(step),
                "is_ambiguous": True,
            }
        )

ambig_df = pd.DataFrame(
    ambig_rows,
    columns=["phoneme_pair", "word_end", "resampled", "is_ambiguous"],
)

md_orig = epochs.metadata.copy()
md_merged = md_orig.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
md_merged = md_merged.merge(
    ambig_df,
    on=["phoneme_pair", "word_end", "resampled"],
    how="left",
)
md_merged["is_ambiguous"] = md_merged["is_ambiguous"].fillna(False).astype(bool)
md_merged["strat_key"] = (
    md_merged["resampled"].astype(int).astype(str)
    + "__"
    + md_merged["word_end"].astype(str)
)
md_merged = md_merged.sort_values("_orig_idx").reset_index(drop=True)

epochs.metadata["is_ambiguous"] = md_merged["is_ambiguous"].values
epochs.metadata["strat_key"] = md_merged["strat_key"].values

# %%
ambiguity_labels_out = epochs.metadata[
    ["phoneme_pair", "word_end", "resampled", "behavior_dummy_forced", "is_ambiguous", "strat_key"]
].copy()
ambiguity_labels_out["subject"] = subject
ambiguity_labels_out["epoch_idx"] = np.arange(len(ambiguity_labels_out))
ambiguity_labels_out.to_parquet(outdir / "ambiguity_labels.parquet")

print(
    f"Ambiguity labels: n_ambig={int(epochs.metadata['is_ambiguous'].sum())}, "
    f"n_unambig={int((~epochs.metadata['is_ambiguous']).sum())}"
)

# %% [markdown]
# ## Electrode selection (acoustically-selective sites)

# %%
acoustic_sites = phon_peaks_df.query(
    "subject == @subject and phon_roc_auc >= @phon_response_peak_threshold"
)
acoustic_sites_by_pp = (
    acoustic_sites.groupby("phoneme_pair")["electrode_idx"].apply(list).to_dict()
)
print(
    f"Acoustic sites per phoneme pair: "
    f"{ {k: len(v) for k, v in acoustic_sites_by_pp.items()} }"
)

# %% [markdown]
# ## Sliding-window decoder

# %%
max_sample = epochs.times.shape[0]

scores_rows: list[dict] = []
outcomes_rows: list[pd.DataFrame] = []
fold_balance_rows: list[dict] = []
all_models: dict = {}


# %%
def _run_one_pair(epochs_pair, electrode_idxs, phoneme_pair, window_size):
    """Sliding-window ambiguity decoder for one (subject, phoneme_pair)."""
    md_pair = epochs_pair.metadata
    strat_counts = md_pair["strat_key"].value_counts()
    viable_strata = set(strat_counts[strat_counts >= stratum_min_trials].index)
    dropped_keys = set(strat_counts.index) - viable_strata
    keep_mask = md_pair["strat_key"].isin(viable_strata).values
    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        print(
            f"  [{phoneme_pair}] dropped {n_dropped} trials from sparse strata: "
            f"{sorted(dropped_keys)}"
        )
    epochs_kept = epochs_pair[keep_mask]
    if len(epochs_kept) < 2 * stratum_min_trials:
        print(f"  [{phoneme_pair}] skipping, too few trials after pruning")
        return
    md_kept = epochs_kept.metadata
    if md_kept["is_ambiguous"].nunique() < 2:
        print(f"  [{phoneme_pair}] skipping, only one class after pruning")
        return

    strat_full = md_kept["strat_key"].values

    total, gen = _prepare_decoding_population(
        epochs_i=epochs_kept,
        electrode_idxs=electrode_idxs,
        phoneme_pair=phoneme_pair,
        stride=stride,
        window_size=window_size,
        global_min_sample=0,
        global_max_sample=max_sample,
        target="is_ambiguous",
    )

    for name, smin, smax, selection, X_window, y in tqdm(
        gen, total=total, desc=f"{phoneme_pair}/ws={window_size}"
    ):
        if len(np.unique(y)) < 2:
            continue

        stratify_arr = strat_full[selection]
        unique_strata, counts = np.unique(stratify_arr, return_counts=True)
        viable = set(unique_strata[counts >= 5])
        inner_keep = np.isin(stratify_arr, list(viable))
        if not inner_keep.all():
            X_window = X_window[inner_keep]
            y = y[inner_keep]
            stratify_arr = stratify_arr[inner_keep]
        if len(np.unique(y)) < 2:
            continue

        fitted = fit_train_test(
            X_window,
            y,
            num_classes=2,
            pca_num_components=pca_num_components,
            scoring=["roc_auc", "f1_macro", "accuracy"],
            num_repeats=5,
            stratify=stratify_arr,
            n_jobs=n_jobs,
        )
        if fitted is None:
            continue

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        n_repeats = len(fitted["test_roc_auc"])
        for rep_idx in range(n_repeats):
            scores_rows.append(
                {
                    "window_size": window_size,
                    "subject": subject,
                    "phoneme_pair": phoneme_pair,
                    "smin": int(smin),
                    "smax": int(smax),
                    "repeat": rep_idx,
                    "roc_auc": float(fitted["test_roc_auc"][rep_idx]),
                    "f1_macro": float(fitted["test_f1_macro"][rep_idx]),
                    "accuracy": float(fitted["test_accuracy"][rep_idx]),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n_dropped_strata": len(dropped_keys),
                }
            )

        # Per-fold (i.e. per-repeat held-out) outcomes, with stratification
        # variables attached for downstream auditing.
        kept_positions = np.where(selection)[0]
        if not inner_keep.all():
            kept_positions = kept_positions[inner_keep]

        for rep_idx, (test_idxs, estimator) in enumerate(
            zip(fitted["test_idxs"], fitted["estimator"])
        ):
            epoch_positions = kept_positions[test_idxs]
            sub_md = md_kept.iloc[epoch_positions]
            outcomes_rows.append(
                pd.DataFrame(
                    {
                        "window_size": window_size,
                        "subject": subject,
                        "phoneme_pair": phoneme_pair,
                        "smin": int(smin),
                        "smax": int(smax),
                        "repeat": rep_idx,
                        "epoch_idx": sub_md.index.values,
                        "y_true": y[test_idxs],
                        "y_pred": estimator.predict(X_window[test_idxs]),
                        "y_proba": estimator.predict_proba(X_window[test_idxs])[:, 1],
                        "word_end": sub_md["word_end"].values,
                        "resampled": sub_md["resampled"].values,
                        "behavior_dummy_forced": sub_md["behavior_dummy_forced"].values,
                        "strat_key": sub_md["strat_key"].values,
                    }
                )
            )

            # Per-(repeat, stratum, class) counts for the held-out set.
            for (strat, cls), n in (
                sub_md.assign(_y=y[test_idxs])
                .groupby(["strat_key", "_y"])
                .size()
                .items()
            ):
                fold_balance_rows.append(
                    {
                        "window_size": window_size,
                        "subject": subject,
                        "phoneme_pair": phoneme_pair,
                        "smin": int(smin),
                        "smax": int(smax),
                        "repeat": rep_idx,
                        "strat_key": str(strat),
                        "is_ambiguous": bool(cls),
                        "n_trials": int(n),
                    }
                )

        all_models[(window_size, subject, phoneme_pair, int(smin), int(smax))] = fitted[
            "estimator"
        ]


# %%
for window_size in window_sizes:
    for phoneme_pair in sorted(acoustic_sites_by_pp.keys()):
        elecs = acoustic_sites_by_pp[phoneme_pair]
        if len(elecs) < 2:
            print(f"Skipping {phoneme_pair}: only {len(elecs)} acoustic sites")
            continue
        pair_epochs = epochs[epochs.metadata.phoneme_pair == phoneme_pair]
        print(
            f"=== {phoneme_pair}: {len(elecs)} electrodes, {len(pair_epochs)} trials"
        )
        _run_one_pair(pair_epochs, elecs, phoneme_pair, window_size)

# %% [markdown]
# ## Save outputs

# %%
scores_cols = [
    "window_size",
    "subject",
    "phoneme_pair",
    "smin",
    "smax",
    "repeat",
    "roc_auc",
    "f1_macro",
    "accuracy",
    "n_pos",
    "n_neg",
    "n_dropped_strata",
]
pd.DataFrame(scores_rows, columns=scores_cols).to_parquet(outdir / "scores.parquet")
print(f"scores.parquet: {len(scores_rows)} rows")

# %%
outcomes_cols = [
    "window_size",
    "subject",
    "phoneme_pair",
    "smin",
    "smax",
    "repeat",
    "epoch_idx",
    "y_true",
    "y_pred",
    "y_proba",
    "word_end",
    "resampled",
    "behavior_dummy_forced",
    "strat_key",
]
if outcomes_rows:
    outcomes_df = pd.concat(outcomes_rows, ignore_index=True)[outcomes_cols]
else:
    outcomes_df = pd.DataFrame(columns=outcomes_cols)
outcomes_df.to_parquet(outdir / "outcomes.parquet")
print(f"outcomes.parquet: {len(outcomes_df)} rows")

# %%
fold_balance_cols = [
    "window_size",
    "subject",
    "phoneme_pair",
    "smin",
    "smax",
    "repeat",
    "strat_key",
    "is_ambiguous",
    "n_trials",
]
pd.DataFrame(fold_balance_rows, columns=fold_balance_cols).to_parquet(
    outdir / "fold_balance.parquet"
)
print(f"fold_balance.parquet: {len(fold_balance_rows)} rows")

# %%
joblib.dump(all_models, outdir / "models.joblib")
print("models.joblib written")
