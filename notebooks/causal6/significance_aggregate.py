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
# causal6: aggregate per-subject peak/significance parquets and apply BH-FDR.
#
# Shared by all three decoders (acoustic / behavior_full / behavior_hga_only)
# and any future decoder whose per-subject parquet contains a `p_value`
# column (as produced by src/models/significance.py:null_standardized_peak_test).
# Parametrized on `result_paths` + `outdir` + `output_name` + `fdr_alpha`.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# %% tags=["parameters"]
result_paths = []  # list[str]; annotation stripped for ploomber static_analysis
outdir = "."
output_name = "significance_all.parquet"
fdr_alpha = 0.05
fdr_rois = []
electrode_dfs_paths = []

# %%
outdir = Path(outdir)

# %%
dfs = [pd.read_parquet(p) for p in result_paths]
combined = pd.concat(dfs, ignore_index=True)

# %%
# Apply ROI restriction if configured. Sites outside the family get q=NaN.
if fdr_rois and electrode_dfs_paths:
    import polars as pl
    from src.models.causal6_aggregates import restrict_to_rois

    electrode_dfs = [pl.from_pandas(pd.read_csv(p)) for p in electrode_dfs_paths]
    combined_pl = pl.from_pandas(combined)
    in_family, n_roi = restrict_to_rois(
        combined_pl, electrode_dfs, fdr_rois,
        site_keys=("subject", "electrode_idx"),
    )
    in_family_keys = set(zip(
        in_family["subject"].to_list(),
        in_family["electrode_idx"].to_list(),
    ))
    combined["in_fdr_family"] = [
        (s, e) in in_family_keys
        for s, e in zip(combined["subject"], combined["electrode_idx"])
    ]
    print(
        f"ROI restriction: {combined['in_fdr_family'].sum()} / {len(combined)} "
        f"rows in FDR family across {n_roi} sites"
    )
else:
    combined["in_fdr_family"] = True

# %%
# FDR correction.
#
# When ROI restriction is active (acoustic decoder): hierarchical Simes + BH +
# Holm so the BH family size is N=n_electrodes rather than N=n_electrodes×n_pairs.
#
# Otherwise (behavior decoders, no per-pair grouping): flat BH over all in-family
# rows, preserving prior behaviour while JON-49 is still open.
if fdr_rois and electrode_dfs_paths:
    family = combined[combined["in_fdr_family"]].copy()

    # Step 1 — Simes: electrode-level p from K per-pair p-values.
    def _simes(ps: "pd.Series") -> float:
        ps_s = np.sort(ps.values)
        k = np.arange(1, len(ps_s) + 1)
        return float(np.min(len(ps_s) * ps_s / k))

    elec_simes = (
        family.groupby(["subject", "electrode_idx"])["p_value"]
        .agg(_simes)
        .reset_index()
        .rename(columns={"p_value": "electrode_p"})
    )

    # Step 2 — BH across electrodes.
    _, elec_q, _, _ = multipletests(
        elec_simes["electrode_p"].values, alpha=fdr_alpha, method="fdr_bh"
    )
    elec_simes["electrode_q_value"] = elec_q
    elec_simes["electrode_significant"] = elec_q < fdr_alpha

    n_elec_sig = int(elec_simes["electrode_significant"].sum())
    print(
        f"Electrode-level BH significant: {n_elec_sig} / {len(elec_simes)} "
        f"({100 * n_elec_sig / len(elec_simes):.1f}%)"
    )

    combined = combined.merge(
        elec_simes[["subject", "electrode_idx", "electrode_q_value", "electrode_significant"]],
        on=["subject", "electrode_idx"],
        how="left",
    )
    # Out-of-family electrodes were not in elec_simes → NaN / False after merge.
    combined["electrode_q_value"] = np.where(
        combined["in_fdr_family"], combined["electrode_q_value"], np.nan
    )
    combined["electrode_significant"] = combined["electrode_significant"].fillna(False).astype(bool)

    # Step 3 — Holm within each electrode_significant electrode.
    q_values = np.full(len(combined), np.nan)
    significant = np.zeros(len(combined), dtype=bool)
    sig_mask = combined["electrode_significant"].values
    for _, grp in combined[sig_mask].groupby(["subject", "electrode_idx"]):
        _, holm_q, _, _ = multipletests(grp["p_value"].values, alpha=fdr_alpha, method="holm")
        q_values[grp.index] = holm_q
        significant[grp.index] = holm_q < fdr_alpha
    combined["q_value"] = q_values
    combined["significant"] = significant

else:
    # Flat BH over all in-family rows.
    mask = combined["in_fdr_family"].values
    q_values = np.full(len(combined), np.nan)
    _, q_in, _, _ = multipletests(
        combined.loc[mask, "p_value"].values, alpha=fdr_alpha, method="fdr_bh"
    )
    q_values[mask] = q_in
    combined["q_value"] = q_values
    combined["significant"] = (combined["q_value"] < fdr_alpha).fillna(False)

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
combined.to_parquet(outdir / output_name)
