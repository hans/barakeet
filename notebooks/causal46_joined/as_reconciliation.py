# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # causal4/causal6 AS-site reconciliation
#
# Classifies every (subject, electrode_idx, phoneme_pair) tuple evaluated by
# either pipeline into one of five buckets, then renders summary stats and
# four star-plot PDFs (losses, gains-eligible, gains-newly-eligible, both) for
# visual inspection. Final canonical AS-site list is written to
# `outputs/causal46_joined/canonical_AS_sites.csv`.
#
# See `docs/superpowers/plans/2026-05-14-causal46-as-reconciliation.md` and
# Linear JON-42.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

# %%
HOME = Path(os.path.expanduser("~"))
# Resolve REPO from this notebook's location so it works in any worktree.
REPO = Path(__file__).resolve().parents[2]
CAUSAL4_DIR = HOME / "freesurfer_subjects/barakeet/causal4_pipeline/prepare_neurometrics"
CAUSAL6_DIR = HOME / "freesurfer_subjects/barakeet/causal6_speech_responsive_pipeline/acoustic_decoding_peaks"
OUT_DIR = REPO / "outputs/causal46_joined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSAL4_AUC_THRESHOLD = 0.65
CAUSAL6_P_THRESHOLD = 0.05

print(f"REPO:        {REPO}")
print(f"CAUSAL4_DIR: {CAUSAL4_DIR}")
print(f"CAUSAL6_DIR: {CAUSAL6_DIR}")
print(f"OUT_DIR:     {OUT_DIR}")

# %% [markdown]
# ## Load causal4 outputs

# %%
# phon_peaks_df.parquet contains the peak window per (subject, electrode_idx,
# phoneme_pair) over the full causal4 search range -- it is NOT pre-filtered
# to AUC>=0.65. Apply the causal4 AS criterion explicitly here.
c4_peaks = pl.read_parquet(CAUSAL4_DIR / "phon_peaks_df.parquet")
c4_peaks = c4_peaks.with_columns(
    pl.col("subject").cast(pl.Utf8),
    pl.col("phoneme_pair").cast(pl.Utf8),
).rename({
    "phon_roc_auc": "causal4_peak_auc",
    "smin": "causal4_smin",
    "smax": "causal4_smax",
}).drop("word_end_offset_sample")
print(f"causal4 phon_peaks_df rows (unfiltered): {c4_peaks.shape[0]}")
c4_AS = c4_peaks.filter(pl.col("causal4_peak_auc") >= CAUSAL4_AUC_THRESHOLD)

c4_eligible = (
    pl.read_parquet(CAUSAL4_DIR / "phon_roc_auc_searchlight_df.parquet")
    .with_columns(
        pl.col("subject").cast(pl.Utf8),
        pl.col("phoneme_pair").cast(pl.Utf8),
    )
    .select(["subject", "electrode_idx", "phoneme_pair"])
    .unique()
)

print(f"causal4 AS sites: {c4_AS.shape[0]}")
print(f"causal4 evaluated tuples: {c4_eligible.shape[0]}")
print(f"causal4 subjects (in AS): {sorted(c4_AS['subject'].unique().to_list())}")

# %% [markdown]
# ## Load causal6 outputs

# %%
c6_paths = sorted(CAUSAL6_DIR.glob("*/phon_peaks.parquet"))
c6_subjects_present = [p.parent.name for p in c6_paths]
print(f"causal6 subjects in prod: {c6_subjects_present}")

c6_all = pl.concat([pl.read_parquet(p) for p in c6_paths])
c6_all = c6_all.rename({
    "test_roc_auc": "causal6_test_roc_auc",
    "p_value": "causal6_p_value",
    "n_permutations": "causal6_n_perm",
    "smin": "causal6_smin",
    "smax": "causal6_smax",
}).select([
    "subject", "electrode_idx", "phoneme_pair",
    "causal6_test_roc_auc", "causal6_p_value", "causal6_n_perm",
    "causal6_smin", "causal6_smax",
])
print(f"causal6 evaluated tuples: {c6_all.shape[0]}")
print(f"causal6 significant (p<0.05): {int((c6_all['causal6_p_value'] < CAUSAL6_P_THRESHOLD).sum())}")

# %% [markdown]
# ## Subject coverage warning

# %%
c4_subj = set(c4_AS["subject"].unique().to_list())
c6_subj = set(c6_subjects_present)
missing_in_c6 = sorted(c4_subj - c6_subj)
if missing_in_c6:
    print(
        f"WARNING: {len(missing_in_c6)} causal4 subjects absent from causal6 prod: "
        f"{missing_in_c6}. Their sites are excluded from reconciliation."
    )
    c4_AS = c4_AS.filter(~pl.col("subject").is_in(missing_in_c6))
    c4_eligible = c4_eligible.filter(~pl.col("subject").is_in(missing_in_c6))

# %% [markdown]
# ## Build the reconciliation table

# %%
KEYS = ["subject", "electrode_idx", "phoneme_pair"]

# Universe = union of every tuple either pipeline evaluated.
universe = pl.concat([
    c4_eligible.select(KEYS),
    c6_all.select(KEYS),
]).unique()

recon = (
    universe
    .join(
        c4_eligible.with_columns(pl.lit(True).alias("causal4_eligible")),
        on=KEYS, how="left",
    )
    .with_columns(pl.col("causal4_eligible").fill_null(False))
    .join(
        c4_AS.with_columns(pl.lit(True).alias("causal4_AS")),
        on=KEYS, how="left",
    )
    .with_columns(pl.col("causal4_AS").fill_null(False))
    .join(c6_all, on=KEYS, how="left")
    .with_columns(
        (pl.col("causal6_p_value") < CAUSAL6_P_THRESHOLD)
            .fill_null(False)
            .alias("causal6_AS"),
    )
)


def assign_bucket(c4_elig: bool, c4_AS_: bool, c6_AS_: bool) -> str:
    if c4_AS_ and c6_AS_:
        return "both"
    if c4_AS_ and not c6_AS_:
        return "causal4_only"
    if c6_AS_ and c4_elig:
        return "causal6_only_eligible"
    if c6_AS_ and not c4_elig:
        return "causal6_only_newly_eligible"
    return "neither_AS"


recon = recon.with_columns(
    pl.struct(["causal4_eligible", "causal4_AS", "causal6_AS"])
      .map_elements(
          lambda s: assign_bucket(s["causal4_eligible"], s["causal4_AS"], s["causal6_AS"]),
          return_dtype=pl.Utf8,
      )
      .alias("bucket")
)

print("Bucket counts:")
print(recon.group_by("bucket").len().sort("len", descending=True))

recon.write_parquet(OUT_DIR / "reconciliation.parquet")
print(f"Written: {OUT_DIR / 'reconciliation.parquet'}  ({recon.shape[0]} rows)")
