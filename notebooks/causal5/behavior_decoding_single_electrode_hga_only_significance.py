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
# Significance test for HGA-only behavior decoding — per-subject score-shuffle
# with max-statistic correction over the temporal searchlight.
#
# Reads `results.joblib` (decoders' `test_predictions` at every searchlight
# window) and `A_final_summary.csv` (real peak fold-mean AUC per site), and
# emits a parquet of per-site p-values. No model refitting: under each
# permutation we shuffle trial labels against the already-fitted OOF
# `full_decoder_proba` values and recompute fold-mean AUC at every window,
# taking the max across windows as the per-permutation null statistic. The
# standard `(#{T_k >= T_obs} + 1) / (K + 1)` one-tailed rule gives the p-value.
#
# Max-stat correction absorbs peak-window selection bias; BH-FDR across sites
# is applied downstream in the aggregate rule.
#
# AUC is computed from rank-sum / Mann-Whitney U, vectorised across windows
# and permutations, so K=10,000 is fast.
#
# Output columns (one row per (subject, electrode_idx, phoneme_pair, word_end)):
#   peak_smin, peak_smax     — argmax window reconstructed from test_predictions
#   ref_smin, ref_smax       — peak window recorded in A_final_summary.csv
#   full_roc_auc             — peak fold-mean AUC from A_final_summary.csv
#   T_obs                    — same statistic recomputed here (sanity check)
#   p_value                  — one-tailed score-shuffle p, max-stat corrected
#   n_permutations           — K
#   null_q05/q50/q95/q99     — null-distribution diagnostics
#   n_trials, n_pos          — trial universe size & positive class count

# %%
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# %% tags=["parameters"]
subject = "EC243"
result_path = (
    f"outputs/causal5/behavior_decoding_single_electrode_hga_only/{subject}/results.joblib"
)
final_summary_path = (
    f"outputs/causal5/behavior_decoding_single_electrode_hga_only_summarize/"
    f"{subject}/A_final_summary.csv"
)
outdir = "."
n_permutations = 10000
seed = 42

# %%
subject = re.findall(
    r"/behavior_decoding_single_electrode_hga_only/([^/]+)/", result_path
)[0]
outdir = Path(outdir)

# %%
behav = joblib.load(result_path)
decoders = behav["decoders"]

A_final = pd.read_csv(final_summary_path)
A_final_idx = A_final.set_index(["subject", "electrode_idx", "phoneme_pair", "word_end"])


# %%
def auc_from_ranks(ranks_wt, y_kt):
    """Vectorised AUC across windows and permutations via Mann-Whitney U.

    Args:
        ranks_wt: (n_windows, n_test) — ascending ranks of `decoder_proba` per
            window, columns aligned with the fold's test trials.
        y_kt: (K, n_test) — permutation-generated binary labels for the same
            test trials.

    Returns:
        aucs: (n_windows, K). NaN where a permutation's test set has zero
        class variance (n_pos in {0, n_test}).
    """
    n_test = ranks_wt.shape[1]
    rank_sum = ranks_wt @ y_kt.T                    # (n_windows, K)
    n_pos = y_kt.sum(axis=1)                        # (K,)
    n_neg = n_test - n_pos
    denom = n_pos * n_neg                           # (K,)
    u = rank_sum - (n_pos * (n_pos + 1) / 2)        # (n_windows, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        aucs = u / denom                            # broadcasts
    aucs[:, denom == 0] = np.nan
    return aucs


# %% [markdown]
# ## Per-decoder score-shuffle
#
# CV seed is fixed (`seed=42` in `src/models/decoding.py:run_decoding_model_comparison_population`),
# so each fold's test-trial set is stable across windows for a given
# (electrode, phoneme_pair, word_end). We reuse that structure to stack
# per-window probas into a matrix per fold and rank once.

# %%
# Group decoder test_predictions by (electrode_idx, phoneme_pair, word_end, fold, (smin, smax))
by_site_we = defaultdict(lambda: defaultdict(dict))
for outer_key, inner in decoders.items():
    _, electrode_idx, phoneme_pair = outer_key
    for inner_key, det in inner.items():
        _, _, _, (word_end,), smin, smax, fold = inner_key
        by_site_we[(electrode_idx, phoneme_pair, word_end)][fold][(smin, smax)] = (
            det["test_predictions"]
        )

rng = np.random.RandomState(seed)
rows = []

for (electrode_idx, phoneme_pair, word_end), by_fold in tqdm(
    list(by_site_we.items()), unit="decoder"
):
    folds = sorted(by_fold.keys())

    # Windows must be identical across folds; take fold-0's window list as canonical.
    windows = sorted(by_fold[folds[0]].keys())
    for f in folds[1:]:
        assert sorted(by_fold[f].keys()) == windows, (
            f"Window set mismatch across folds for "
            f"({electrode_idx}, {phoneme_pair}, {word_end})"
        )
    n_windows = len(windows)

    # Build the trial universe for this decoder: union of test trials across folds.
    # The (epoch_idx -> label) map must be consistent wherever a trial recurs.
    trial_labels = {}
    for f in folds:
        # Any window for this fold contains the same epoch_idx set; use the first.
        td0 = by_fold[f][windows[0]]
        for epoch_idx, y in zip(td0["epoch_idx"].values, td0["decoder_target"].values):
            prior = trial_labels.get(int(epoch_idx))
            if prior is not None and prior != int(y):
                raise AssertionError(
                    f"Inconsistent label for epoch {epoch_idx} in "
                    f"({electrode_idx}, {phoneme_pair}, {word_end})"
                )
            trial_labels[int(epoch_idx)] = int(y)

    trials = sorted(trial_labels.keys())
    trial_to_pos = {e: i for i, e in enumerate(trials)}
    y_trial = np.array([trial_labels[e] for e in trials], dtype=np.int64)
    n_trials = y_trial.size
    n_pos_true = int(y_trial.sum())
    if n_pos_true in (0, n_trials):
        # Degenerate site: only one class. Skip.
        continue

    # Pre-permute labels once: shape (K, n_trials).
    shuffled = np.empty((n_permutations, n_trials), dtype=np.int64)
    for k in range(n_permutations):
        shuffled[k] = rng.permutation(y_trial)

    # For each fold: canonical test-trial order + rank matrix across windows.
    fold_positions = {}          # fold -> trial-universe positions (np.int64 array)
    ranks_per_fold = {}          # fold -> (n_windows, n_test_f) ranks
    auc_true_per_fold = np.full((n_windows, len(folds)), np.nan)

    for fi, f in enumerate(folds):
        td0 = by_fold[f][windows[0]]
        test_epochs = sorted(td0["epoch_idx"].astype(int).tolist())
        assert len(set(test_epochs)) == len(test_epochs), (
            f"Duplicate epoch in fold {f} test set for "
            f"({electrode_idx}, {phoneme_pair}, {word_end})"
        )
        pos_f = np.array([trial_to_pos[e] for e in test_epochs], dtype=np.int64)
        fold_positions[f] = pos_f
        n_test_f = pos_f.size

        proba_mat = np.empty((n_windows, n_test_f), dtype=np.float64)
        y_col = None
        for wi, key in enumerate(windows):
            td = by_fold[f][key].set_index("epoch_idx").loc[test_epochs]
            proba_mat[wi] = td["full_decoder_proba"].to_numpy()
            if y_col is None:
                y_col = td["decoder_target"].to_numpy().astype(np.int64)
        ranks_per_fold[f] = rankdata(proba_mat, method="average", axis=1)

        # Real-data AUC per (window, fold). Reuse sklearn for a sanity-checkable
        # path that matches what was written to A_results.csv / A_final_summary.csv.
        if y_col.sum() in (0, n_test_f):
            auc_true_per_fold[:, fi] = np.nan
        else:
            for wi in range(n_windows):
                auc_true_per_fold[wi, fi] = roc_auc_score(y_col, proba_mat[wi])

    # Real T_obs = max over windows of fold-mean AUC
    auc_true_fold_mean = np.nanmean(auc_true_per_fold, axis=1)  # (n_windows,)
    T_obs = float(np.nanmax(auc_true_fold_mean))
    peak_wi = int(np.nanargmax(auc_true_fold_mean))
    peak_smin, peak_smax = windows[peak_wi]

    # Null distribution
    auc_null_per_fold = np.empty(
        (len(folds), n_windows, n_permutations), dtype=np.float64
    )
    for fi, f in enumerate(folds):
        y_kt = shuffled[:, fold_positions[f]]  # (K, n_test_f)
        auc_null_per_fold[fi] = auc_from_ranks(ranks_per_fold[f], y_kt)

    auc_null_fold_mean = np.nanmean(auc_null_per_fold, axis=0)  # (n_windows, K)
    T_null = np.nanmax(auc_null_fold_mean, axis=0)              # (K,)

    p_value = (np.sum(T_null >= T_obs) + 1) / (n_permutations + 1)

    # Reference values from A_final_summary.csv
    row_key = (subject, int(electrode_idx), phoneme_pair, word_end)
    if row_key in A_final_idx.index:
        a_row = A_final_idx.loc[row_key]
        full_auc_reference = float(a_row["full_roc_auc"])
        ref_smin = int(a_row["smin"])
        ref_smax = int(a_row["smax"])
    else:
        full_auc_reference = np.nan
        ref_smin = ref_smax = -1

    rows.append(
        {
            "subject": subject,
            "electrode_idx": int(electrode_idx),
            "phoneme_pair": phoneme_pair,
            "word_end": word_end,
            "peak_smin": int(peak_smin),
            "peak_smax": int(peak_smax),
            "ref_smin": ref_smin,
            "ref_smax": ref_smax,
            "full_roc_auc": full_auc_reference,
            "T_obs": T_obs,
            "p_value": float(p_value),
            "n_permutations": int(n_permutations),
            "null_q05": float(np.nanpercentile(T_null, 5)),
            "null_q50": float(np.nanpercentile(T_null, 50)),
            "null_q95": float(np.nanpercentile(T_null, 95)),
            "null_q99": float(np.nanpercentile(T_null, 99)),
            "n_trials": int(n_trials),
            "n_pos": int(n_pos_true),
        }
    )

# %% [markdown]
# ## Save
#
# Aggregation across subjects and BH-FDR correction are applied downstream in
# `behavior_decoding_single_electrode_hga_only_significance_aggregate.py`.

# %%
out_df = pd.DataFrame(rows)
out_df.to_parquet(outdir / "significance.parquet")
print(f"Wrote {len(out_df)} rows to {outdir / 'significance.parquet'}")
