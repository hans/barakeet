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
# Permutation-based NHST on behavior_decoding_single_electrode results.
#
# For each unique decoder (subject × electrode × phoneme_pair × word_end × window):
#   - True statistic: mean over folds of Δ ROC-AUC = full_roc_auc − baseline_roc_auc
#   - Null distribution: same statistic computed under K permutations of the labels
#   - p-value (one-sided, upper): (#{permuted ≥ true} + 1) / (K + 1)
#   - Multiple-comparisons correction: Benjamini-Hochberg FDR across all decoders
#
# Outputs a CSV with columns: subject, population, phoneme_pair, word_end, smin, smax,
# true_delta_roc_auc, p_value, q_value, significant.

# %%
from pathlib import Path

import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests

# %% tags=["parameters"]
all_true_results = [
    "outputs/causal4/behavior_decoding_single_electrode/EC260/results.pt"
]  # paths to results.pt from behavior_decoding_single_electrode
all_permutation_results = [
    "outputs/causal4/behavior_decoding_single_electrode_permutation/EC260/permutation_results.parquet"
]  # paths to permutation_results.parquet
fdr_alpha = 0.05
outdir = "."

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load true decoding results

# %%
true_dfs = []
for path in all_true_results:
    d = torch.load(path)
    for df in d["A_decoding_results"].values():
        if len(df) > 0:
            true_dfs.append(df)

true_results = pd.concat(true_dfs, ignore_index=True)

# %% [markdown]
# ## Load permuted decoding results

# %%
perm_dfs = []
for path in all_permutation_results:
    perm_dfs.append(pd.read_parquet(path))

perm_results = pd.concat(perm_dfs, ignore_index=True)

n_permutations = perm_results["permutation_idx"].nunique()
print(
    f"Loaded {n_permutations} permutations across {len(all_permutation_results)} subjects."
)

# %% [markdown]
# ## Compute Δ ROC-AUC and aggregate over folds
#
# The test statistic is the mean (over CV folds) of full − baseline ROC-AUC.

# %%
groupby_cols = ["subject", "population", "phoneme_pair", "word_end", "smin", "smax"]

true_results["delta_roc_auc"] = (
    true_results["full_roc_auc"] - true_results["baseline_roc_auc"]
)
true_summary = (
    true_results.groupby(groupby_cols, observed=True)["delta_roc_auc"]
    .mean()
    .reset_index()
    .rename(columns={"delta_roc_auc": "true_delta_roc_auc"})
)

perm_results["delta_roc_auc"] = (
    perm_results["full_roc_auc"] - perm_results["baseline_roc_auc"]
)
perm_summary = (
    perm_results.groupby(groupby_cols + ["permutation_idx"], observed=True)[
        "delta_roc_auc"
    ]
    .mean()
    .reset_index()
    .rename(columns={"delta_roc_auc": "perm_delta_roc_auc"})
)

# %% [markdown]
# ## NHST: one-sided permutation p-values
#
# p = (#{permuted Δ ≥ true Δ} + 1) / (K + 1)
# The "+1" in numerator and denominator is a standard correction that keeps
# p-values bounded away from 0 and ensures validity under the null.

# %%
merged = true_summary.merge(perm_summary, on=groupby_cols)

pvalue_rows = (
    merged.groupby(groupby_cols, observed=True)
    .apply(
        lambda g: (
            (
                (
                    g["perm_delta_roc_auc"].values >= g["true_delta_roc_auc"].iloc[0]
                ).sum()
                + 1
            )
            / (n_permutations + 1)
        )
    )
    .reset_index(name="p_value")
)

# %% [markdown]
# ## FDR correction (Benjamini-Hochberg) across all decoder tests

# %%
_, q_values, _, _ = multipletests(
    pvalue_rows["p_value"].values, alpha=fdr_alpha, method="fdr_bh"
)
pvalue_rows["q_value"] = q_values
pvalue_rows["significant"] = pvalue_rows["q_value"] < fdr_alpha

# %% [markdown]
# ## Combine and save

# %%
results = true_summary.merge(pvalue_rows, on=groupby_cols)

print(f"Total decoders tested: {len(results)}")
print(
    f"Significant (q < {fdr_alpha}): {results['significant'].sum()} "
    f"({100 * results['significant'].mean():.1f}%)"
)

results.to_csv(outdir / "results.csv", index=False)
