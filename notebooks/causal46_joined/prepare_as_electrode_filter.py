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
# # causal46_joined: build the AS-electrode filter
#
# Reads the causal6 acoustic peak parquet (`phon_peaks_all.parquet`) plus the
# per-subject `find_speech_responsive` CSVs; writes per-subject CSVs identical
# to the inputs but with a new boolean column `acoustic_significant`, and a
# manifest text file listing subjects with at least one AS electrode.
#
# AS definition (the one used by every joined decoder):
#   - uncorrected `p_value < as_p_threshold` (default 0.05)
#   - in `phon_peaks_all.parquet` (v1 foldmean_maxstat flavor)
#   - collapsed to electrode level via OR over `phoneme_pair`
#   - AND'd with `speech_responsive` so we never promote a non-responsive site
#
# All real logic lives in `src.causal46_joined.compute_as_filter` so it can be
# unit-tested without ploomber.

# %%
from pathlib import Path

import pandas as pd
import polars as pl

from src.causal46_joined import compute_as_filter

# %% tags=["parameters"]
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
electrode_csv_paths = []  # list[str]; annotation stripped for ploomber static_analysis
outdir = "."
as_p_threshold = 0.05

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load inputs

# %%
phon = pl.read_parquet(phon_peaks_path)
print(f"phon_peaks_all rows: {phon.shape[0]}")
print(
    f"  uncorrected p < {as_p_threshold}: "
    f"{int((phon['p_value'] < as_p_threshold).sum())} rows"
)

# %%
electrode_dfs_by_subject: dict[str, pd.DataFrame] = {}
for csv_path in electrode_csv_paths:
    sr = pd.read_csv(csv_path)
    subj = sr["subject"].iloc[0]
    if subj in electrode_dfs_by_subject:
        raise ValueError(
            f"Duplicate subject {subj} in electrode_csv_paths "
            f"(already loaded from a different path)"
        )
    electrode_dfs_by_subject[subj] = sr

print(f"loaded electrode tables for {len(electrode_dfs_by_subject)} subjects")

# %% [markdown]
# ## Compute the AS filter

# %%
annotated, subjects_with_as = compute_as_filter(
    phon, electrode_dfs_by_subject, as_p_threshold=as_p_threshold,
)

# %% [markdown]
# ## Write outputs

# %%
for subject, sr_out in annotated.items():
    n_total = len(sr_out)
    n_sr = int(sr_out["speech_responsive"].sum())
    n_as = int(sr_out["acoustic_significant"].sum())
    print(
        f"  {subject}: {n_total} electrodes, "
        f"speech_responsive={n_sr}, acoustic_significant={n_as}"
    )
    sr_out.to_csv(outdir / f"{subject}_results.csv", index=False)

# %%
manifest_path = outdir / "subjects_with_as.txt"
manifest_path.write_text("\n".join(subjects_with_as) + ("\n" if subjects_with_as else ""))
print(f"wrote manifest with {len(subjects_with_as)} AS-positive subjects: {manifest_path}")
