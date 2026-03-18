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
# # AX discrimination: adjacent-step decoders
#
# For each acoustically selective site and each adjacent step pair (1v2, 2v3, ..., 5v6),
# train a fresh binary decoder on the raw HGA at the site's peak acoustic window.
# This tests whether the neural signal itself can distinguish adjacent morph steps —
# the hallmark of categorical perception is high discrimination at the category
# boundary and low discrimination within categories.
#
# Inputs:
#   phon_peaks_df.parquet — peak acoustic window per site (from acoustic_decoding_peaks)
#   {subject}_epo.fif     — epoch files for metadata and HGA data
#
# Outputs:
#   ax_discrimination_df.parquet — per-(site, step-pair) discrimination AUC

# %%
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

# %%
from src.data import add_metadata_features
from src.models.decoding import fit_train_test
from src.viz_paper import phoneme_pair_enum, subject_enum

# %% tags=["parameters"]
all_epochs = list(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))
phon_peaks_path = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet"

epoch_tmin = -0.4
epoch_sfreq = 100

phon_response_peak_threshold = 0.65

outdir = "outputs/causal5/acoustic_ax_discrimination"

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load acoustic sites and peak windows

# %%
phon_peaks = pl.read_parquet(phon_peaks_path).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)
acoustic_sites = phon_peaks.filter(
    pl.col("phon_roc_auc") >= phon_response_peak_threshold
)
print(f"Acoustic sites: {len(acoustic_sites)}")

# %% [markdown]
# ## Load epoch data and metadata

# %%
all_md_rows = []
epoch_data_cache: dict[str, np.ndarray] = {}

for epoch_path in tqdm(sorted(all_epochs), desc="Loading epochs"):
    subject = re.findall(r"(EC\d+)_epo", str(epoch_path))[0]
    ep = mne.read_epochs(epoch_path, preload=True, verbose=False)
    ep.apply_baseline()
    epoch_data_cache[subject] = ep.get_data()   # (n_trials, n_channels, n_samples)
    md = add_metadata_features(ep.metadata).assign(subject=subject)
    del ep
    md.index.name = "epoch_idx"
    all_md_rows.append(
        md.reset_index()[
            [
                "subject",
                "epoch_idx",
                "phoneme_pair",
                "resampled",
            ]
        ]
    )

all_md = pl.from_pandas(pd.concat(all_md_rows, ignore_index=True)).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)

# %% [markdown]
# ## Train adjacent-step decoders

# %%
site_keys = acoustic_sites.select(
    ["subject", "electrode_idx", "phoneme_pair", "phon_roc_auc", "smin", "smax"]
).to_pandas()

ax_rows = []
step_pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

for _, site_row in tqdm(site_keys.iterrows(), total=len(site_keys), desc="AX discrimination"):
    sub = site_row["subject"]
    ei = int(site_row["electrode_idx"])
    pp = site_row["phoneme_pair"]
    smin_w, smax_w = int(site_row["smin"]), int(site_row["smax"])

    data_arr = epoch_data_cache[sub]

    # Get metadata for this site's trials (need resampled step)
    site_md = all_md.filter(
        (pl.col("subject") == sub) & (pl.col("phoneme_pair") == pp)
    ).to_pandas()

    for step_a, step_b in step_pairs:
        mask_a = site_md["resampled"] == step_a
        mask_b = site_md["resampled"] == step_b
        idx_a = site_md.loc[mask_a, "epoch_idx"].values.astype(int)
        idx_b = site_md.loc[mask_b, "epoch_idx"].values.astype(int)

        if len(idx_a) < 5 or len(idx_b) < 5:
            continue

        # Extract windowed HGA for this electrode
        X_a = data_arr[idx_a, ei, smin_w:smax_w]  # (n_a, window_size)
        X_b = data_arr[idx_b, ei, smin_w:smax_w]  # (n_b, window_size)
        X = np.vstack([X_a, X_b])
        y = np.array([0] * len(idx_a) + [1] * len(idx_b))

        fitted = fit_train_test(
            X, y,
            num_classes=2,
            scoring=["roc_auc"],
            stratify=y,
            num_repeats=5,
            n_jobs=1,
        )

        if fitted is None:
            continue

        test_aucs = fitted["test_roc_auc"]
        ax_rows.append({
            "subject": sub,
            "electrode_idx": ei,
            "phoneme_pair": pp,
            "step_a": step_a,
            "step_b": step_b,
            "n_a": len(idx_a),
            "n_b": len(idx_b),
            "roc_auc": float(np.mean(test_aucs)),
            "roc_auc_std": float(np.std(test_aucs)),
        })

ax_discrimination_df = pd.DataFrame(ax_rows)
ax_discrimination_df.to_parquet(outdir / "ax_discrimination_df.parquet", index=False)
print(f"ax_discrimination_df: {len(ax_discrimination_df)} rows")
ax_discrimination_df.head()
