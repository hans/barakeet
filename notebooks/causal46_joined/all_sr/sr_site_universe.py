# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet
#     language: python
#     name: python3
# ---

# %% [markdown]
# # All-speech-responsive site universe (unconditioned on acoustic response)
#
# Step 1 of the all-SR perceptual fork
# (`docs/superpowers/plans/2026-08-27-all-speech-responsive-perceptual.md`).
# Parallel to `prepare_as_electrode_filter.py`, but the site universe is
# every **speech-responsive** electrode x phoneme_pair the subject saw —
# not just the acoustic-significant (AS) subset. `acoustic_significant` is
# attached as a label (never a filter): every SR site survives, so downstream
# perceptual testing can run without the acoustic pre-selection that the
# AS-restricted pipeline structurally requires.
#
# All real logic lives in `src.causal46_joined.compute_sr_site_universe` so
# it's unit-tested without ploomber (`tests/test_all_sr_perceptual.py`).
#
# Outputs:
# - `outputs/causal46_joined/sr_site_universe/sr_site_universe.parquet` —
#   one row per (subject, electrode_idx, phoneme_pair), columns
#   `acoustic_significant`, `phon_smin`, `phon_smax`, `acoustic_peak_auc`
#   (the latter three null when not acoustic-significant).

# %%
from pathlib import Path

import mne
import pandas as pd
import polars as pl

from src.causal46_joined import compute_sr_site_universe

# %% tags=["parameters"]
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
electrode_csv_paths = []  # list[str]; annotation stripped for ploomber static_analysis
epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/sr_site_universe"
ac_p_value_threshold = 0.01

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
EPOCH_DIR = Path(epoch_dir)

# %% [markdown]
# ## Load per-subject speech-responsive electrode tables

# %%
sr_by_subject: dict[str, pd.DataFrame] = {}
for csv_path in electrode_csv_paths:
    sr = pd.read_csv(csv_path)
    assert sr["subject"].nunique() == 1, f"{csv_path} contains rows for multiple subjects"
    subj = sr["subject"].iloc[0]
    if subj in sr_by_subject:
        raise ValueError(
            f"Duplicate subject {subj} in electrode_csv_paths "
            f"(already loaded from a different path)"
        )
    sr_by_subject[subj] = sr

print(f"loaded electrode tables for {len(sr_by_subject)} subjects")
for subj, sr in sr_by_subject.items():
    n_sr = int(sr["speech_responsive"].sum())
    print(f"  {subj}: {len(sr)} electrodes, speech_responsive={n_sr}")

# %% [markdown]
# ## Phoneme pairs each subject saw (from epoch metadata, metadata-only load)

# %%
subject_phoneme_pairs: dict[str, list[str]] = {}
for subject in sr_by_subject:
    path = EPOCH_DIR / f"{subject}_epo.fif"
    if not path.exists():
        print(f"  ⚠ {subject}: {path} missing — no phoneme_pairs, contributes 0 rows")
        subject_phoneme_pairs[subject] = []
        continue
    epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
    pairs = sorted(epochs.metadata["phoneme_pair"].dropna().unique().tolist())
    subject_phoneme_pairs[subject] = pairs

print("phoneme_pairs per subject:")
for subj, pairs in subject_phoneme_pairs.items():
    print(f"  {subj}: {pairs}")

# %% [markdown]
# ## Build the universe

# %%
phon_peaks = pl.read_parquet(phon_peaks_path)
print(f"phon_peaks_all rows: {phon_peaks.height}")

universe = compute_sr_site_universe(
    sr_by_subject, subject_phoneme_pairs, phon_peaks,
    ac_p_value_threshold=ac_p_value_threshold,
)
print(f"sr_site_universe: {universe.height} rows "
      f"(subject, electrode_idx, phoneme_pair)")
print(
    "  acoustic_significant: "
    f"{int(universe['acoustic_significant'].sum())} / {universe.height} "
    f"(p < {ac_p_value_threshold})"
)

# %% [markdown]
# ## Electrode-level collapse (diagnostic — matches compute_as_filter's OR-across-pair view)

# %%
electrode_level = (
    universe
    .group_by(["subject", "electrode_idx"])
    .agg(pl.col("acoustic_significant").any().alias("acoustic_significant_electrode"))
)
n_sr_electrodes = electrode_level.height
n_as_electrodes = int(electrode_level["acoustic_significant_electrode"].sum())
print(
    f"SR electrodes: {n_sr_electrodes}  "
    f"(AS at p<{ac_p_value_threshold}, OR-across-pair: {n_as_electrodes})"
)

# %% [markdown]
# ## Write outputs

# %%
universe.write_parquet(OUT_DIR / "sr_site_universe.parquet")
print(f"wrote {OUT_DIR / 'sr_site_universe.parquet'}")

electrode_level.write_csv(OUT_DIR / "sr_site_universe_electrode_level.csv")
print(f"wrote {OUT_DIR / 'sr_site_universe_electrode_level.csv'}")

summary = pl.DataFrame({
    "n_sr_electrode_pairs": [universe.height],
    "n_sr_electrodes": [n_sr_electrodes],
    "n_as_electrode_pairs": [int(universe["acoustic_significant"].sum())],
    "n_as_electrodes": [n_as_electrodes],
    "ac_p_value_threshold": [ac_p_value_threshold],
})
summary.write_csv(OUT_DIR / "summary.csv")
print(summary)
