"""One-shot validation for JON-50.

Runs the new hierarchical Simes+BH+Holm aggregate against existing prod
per-subject phon_peaks parquets, then compares the resulting `significant`
column to the existing prod aggregate (flat BH baseline).

Reads:
  outputs_prod_causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet
  outputs_prod_causal6/find_speech_responsive/{subject}_results.csv
  outputs_prod_causal6/acoustic_decoding_peaks/phon_peaks_all.parquet  (old)

Writes:
  outputs_prod_causal6/acoustic_decoding_peaks/phon_peaks_all_new.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from statsmodels.stats.multitest import multipletests

from src.models.causal6_aggregates import restrict_to_rois


PROD = Path("outputs_prod_causal6")
PEAKS_DIR = PROD / "acoustic_decoding_peaks"
ELEC_DIR = PROD / "find_speech_responsive"
OLD_AGG = PEAKS_DIR / "phon_peaks_all.parquet"
NEW_AGG = PEAKS_DIR / "phon_peaks_all_new.parquet"

config = yaml.safe_load(open("config.yaml"))
subjects = config["data"]["subjects"]
fdr_alpha = config["analysis"]["fdr_alpha"]
fdr_rois = config["analysis"]["fdr_rois"]
print(f"alpha={fdr_alpha}  rois={fdr_rois}  subjects={subjects}")


def run_hierarchical(per_subject_parquets, electrode_csvs):
    combined = pd.concat(
        [pd.read_parquet(p) for p in per_subject_parquets], ignore_index=True
    )

    electrode_dfs = [pl.from_pandas(pd.read_csv(p)) for p in electrode_csvs]
    in_family, n_roi = restrict_to_rois(
        pl.from_pandas(combined), electrode_dfs, fdr_rois,
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
        f"rows in family across {n_roi} sites"
    )

    family = combined[combined["in_fdr_family"]].copy()

    def _simes(ps):
        ps_s = np.sort(ps.values)
        k = np.arange(1, len(ps_s) + 1)
        return float(np.min(len(ps_s) * ps_s / k))

    elec_simes = (
        family.groupby(["subject", "electrode_idx"])["p_value"]
        .agg(_simes)
        .reset_index()
        .rename(columns={"p_value": "electrode_p"})
    )
    _, elec_q, _, _ = multipletests(
        elec_simes["electrode_p"].values, alpha=fdr_alpha, method="fdr_bh"
    )
    elec_simes["electrode_q_value"] = elec_q
    elec_simes["electrode_significant"] = elec_q < fdr_alpha
    print(
        f"Electrode-level BH significant: {int(elec_simes['electrode_significant'].sum())} "
        f"/ {len(elec_simes)}"
    )

    combined = combined.merge(
        elec_simes[["subject", "electrode_idx", "electrode_q_value", "electrode_significant"]],
        on=["subject", "electrode_idx"], how="left",
    )
    combined["electrode_q_value"] = np.where(
        combined["in_fdr_family"], combined["electrode_q_value"], np.nan
    )
    combined["electrode_significant"] = combined["electrode_significant"].fillna(False).astype(bool)

    q_values = np.full(len(combined), np.nan)
    significant = np.zeros(len(combined), dtype=bool)
    sig_mask = combined["electrode_significant"].values
    for _, grp in combined[sig_mask].groupby(["subject", "electrode_idx"]):
        _, holm_q, _, _ = multipletests(grp["p_value"].values, alpha=fdr_alpha, method="holm")
        q_values[grp.index] = holm_q
        significant[grp.index] = holm_q < fdr_alpha
    combined["q_value"] = q_values
    combined["significant"] = significant
    return combined


per_subj = [PEAKS_DIR / s / "phon_peaks.parquet" for s in subjects]
elec_csvs = [ELEC_DIR / f"{s}_results.csv" for s in subjects]
missing = [p for p in per_subj + elec_csvs if not p.exists()]
if missing:
    raise FileNotFoundError(f"missing inputs: {missing}")

new = run_hierarchical(per_subj, elec_csvs)
new.to_parquet(NEW_AGG)
print(f"\nWrote {NEW_AGG}")

old = pd.read_parquet(OLD_AGG)

key = ["subject", "electrode_idx", "phoneme_pair"]
merged = old[key + ["significant"]].rename(columns={"significant": "old_sig"}).merge(
    new[key + ["significant", "in_fdr_family", "electrode_significant"]].rename(
        columns={"significant": "new_sig"}
    ),
    on=key, how="outer", indicator=True,
)
print(f"\nMerge: {merged['_merge'].value_counts().to_dict()}")
assert (merged["_merge"] == "both").all(), "row mismatch between old and new"

print(f"\nOld significant (flat BH, N={len(old)}): {int(old['significant'].sum())}")
print(f"New significant (hierarchical, ROI-restricted): {int(new['significant'].sum())}")
print(f"  in_fdr_family rows: {int(new['in_fdr_family'].sum())}")
print(f"  electrode_significant rows (BH layer): {int(new['electrode_significant'].sum())}")

losses = merged[merged["old_sig"] & ~merged["new_sig"]]
gains = merged[~merged["old_sig"] & merged["new_sig"]]
agree_sig = merged[merged["old_sig"] & merged["new_sig"]]
agree_ns = merged[~merged["old_sig"] & ~merged["new_sig"]]
print(f"\nAgreement:")
print(f"  both significant:   {len(agree_sig)}")
print(f"  both non-sig:       {len(agree_ns)}")
print(f"  lost (old→new):     {len(losses)}")
print(f"  gained (new→old):   {len(gains)}")

if len(losses):
    print("\nLOSSES (significant under old flat BH, not under new):")
    print(losses[key + ["in_fdr_family", "electrode_significant"]].to_string(index=False))
if len(gains):
    print("\nGAINS (significant under new, not under old):")
    print(gains[key + ["in_fdr_family", "electrode_significant"]].to_string(index=False))

old_sites = old[old["significant"]].groupby(["subject", "electrode_idx"]).size()
new_sites = new[new["significant"]].groupby(["subject", "electrode_idx"]).size()
print(f"\nElectrodes with ≥1 significant pair — old: {len(old_sites)}  new: {len(new_sites)}")
multi_new = (new_sites > 1).sum()
multi_old = (old_sites > 1).sum()
print(f"Electrodes with >1 significant pair — old: {multi_old}  new: {multi_new}")
