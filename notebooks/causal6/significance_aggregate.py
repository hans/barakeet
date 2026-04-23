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
# causal6: aggregate per-subject significance parquets and apply BH-FDR.
#
# Shared by all three decoders (acoustic / behavior_full / behavior_hga_only);
# parametrized on `result_paths` + `outdir` + `fdr_alpha`. Mirrors
# notebooks/causal5/behavior_decoding_single_electrode_hga_only_significance_aggregate.py.

# %%
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

# %% tags=["parameters"]
result_paths = []  # list[str]; annotation stripped for ploomber static_analysis
outdir = "."
fdr_alpha = 0.05

# %%
outdir = Path(outdir)

# %%
dfs = [pd.read_parquet(p) for p in result_paths]
combined = pd.concat(dfs, ignore_index=True)

# %%
_, q_values, _, _ = multipletests(
    combined["p_value"].values, alpha=fdr_alpha, method="fdr_bh"
)
combined["q_value"] = q_values
combined["significant"] = combined["q_value"] < fdr_alpha

# %%
print(f"Total decoders tested: {len(combined)}")
print(
    f"Significant (q < {fdr_alpha}): {combined['significant'].sum()} "
    f"({100 * combined['significant'].mean():.1f}%)"
)
print(
    "Per-subject significant counts:\n"
    f"{combined.groupby('subject')['significant'].agg(['sum', 'size']).to_string()}"
)

# %%
combined.to_parquet(outdir / "significance_all.parquet")
