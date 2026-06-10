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
# # AX discrimination — causal46_joined edition
#
# Adjacent-step binary decoders (1v2, 2v3, ..., 5v6) at each manifest-curated
# site's peak acoustic window. Hallmark of categorical perception: high AUC
# at the category boundary, low AUC within categories. Adapted from
# notebooks/causal5/acoustic_ax_discrimination.py with two changes:
#   - Electrode pool from filtered_manifest.csv (any annotated cell qualifies
#     the (subject, electrode_idx, phoneme_pair))
#   - Peak window from causal6 phon_peaks.parquet (null-standardized)
#
# Completions are pooled (matches causal5).

# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# %%
import sys
from pathlib import Path

import mne
import pandas as pd
from tqdm.auto import tqdm

from src.data import add_metadata_features
from src.models.decoding import run_ax_discrimination

# %% tags=["parameters"]
subject = "EC250"

epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/EC250/phon_peaks.parquet"
outdir = "."

ac_p_value_threshold = 0.001  # uncorrected; matches t_tests AC_P_VALUE_THRESHOLD
n_jobs = 4

# %%
subject = Path(epochs_path).name.split("_")[0]
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Build site pool

# %%
pool = (
    pd.read_parquet(phon_peaks_path)
    .query("p_value < @ac_p_value_threshold")
    [["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]]
    .drop_duplicates(subset=["subject", "electrode_idx", "phoneme_pair"])
    .reset_index(drop=True)
)
print(f"{subject}: {len(pool)} (electrode, phoneme_pair) sites "
      f"with p_value < {ac_p_value_threshold}")

# %% [markdown]
# ## Load epochs

# %%
epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)
epochs.apply_baseline()
data_arr = epochs.get_data()  # (n_trials, n_channels, n_samples)
md = add_metadata_features(epochs.metadata).reset_index(drop=True)
md.index.name = "epoch_idx"
md = md.reset_index()

# %% [markdown]
# ## Train adjacent-step decoders

# %%
rows = []

for _, site_row in tqdm(pool.iterrows(), total=len(pool), desc="AX discrimination"):
    ei = int(site_row["electrode_idx"])
    pp = site_row["phoneme_pair"]
    smin_w, smax_w = int(site_row["smin"]), int(site_row["smax"])

    site_md = md[md["phoneme_pair"] == pp].reset_index(drop=True)
    if len(site_md) == 0:
        continue
    epoch_idxs = site_md["epoch_idx"].values.astype(int)
    site_X = data_arr[epoch_idxs, ei, smin_w:smax_w]  # (n_trials, window_size)

    site_rows = run_ax_discrimination(
        metadata=site_md,
        get_X=lambda idx: site_X[idx],
        phoneme_pair=pp,
        fit_kw=dict(n_jobs=n_jobs),
    )
    for row in site_rows:
        row.update(
            subject=subject,
            electrode_idx=ei,
            phoneme_pair=pp,
            smin=smin_w,
            smax=smax_w,
        )
    rows.extend(site_rows)

_ax_schema = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax",
              "step_a", "step_b", "n_a", "n_b", "roc_auc", "roc_auc_std"]
ax_discrimination_df = (
    pd.DataFrame(rows) if rows else pd.DataFrame(columns=_ax_schema)
)
ax_discrimination_df.to_parquet(outdir / "ax_discrimination_df.parquet", index=False)
print(f"ax_discrimination_df: {len(ax_discrimination_df)} rows")
ax_discrimination_df.head()
