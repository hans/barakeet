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
# causal6 reg_lambda sweep on the tuning subject.
#
# Runs each of the three decoders (acoustic, behavior-full, behavior-HGA-only)
# at every reg_lambda in `reg_lambda_grid`. Picks the argmax mean test AUC per
# decoder and writes the winners to `reg_lambda_winners.json`.
#
# Decoder rules in the Snakefile read `reg_lambda_winners.json` directly as a
# Snakemake input — no config edit required.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %%
import json
import re
from pathlib import Path

import mne
import pandas as pd
import polars as pl
import torch
from loguru import logger as L

# %%
from src.data import add_metadata_features
from src.models.causal6 import (
    make_windows,
    run_acoustic_searchlight,
    run_behavior_hga_only,
    run_behavior_with_control,
)

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
outdir = "."

min_sample = 1
window_size = 15
stride = 2

reg_lambda_grid = [0.01, 0.1, 1.0, 10.0]
n_folds = 5
cv_random_state = 42
device = "cuda"
tol = 1e-6
max_iter = 50

# %%
subject = re.findall(r"(EC[\d]+)_epo", str(epochs_path))[0]
L.info(f"Tuning on subject: {subject}")

# %%
electrode_df = pd.read_csv(electrodes_path)
speech_responsive_idxs = sorted(
    electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"].unique().astype(int)
)

# %%
epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)
max_sample = epochs.times.shape[0]
windows = make_windows(min_sample, max_sample, window_size, stride)

# %% [markdown]
# Sweep each decoder independently.

# %%
def _mean_auc(scores: pl.DataFrame) -> float:
    """Mean test AUC across all (problem × fold) rows, skipping NaN."""
    return scores.select(pl.col("test_roc_auc").mean()).item()


sweep_rows = []

for rl in reg_lambda_grid:
    L.info(f"[acoustic] reg_lambda={rl}")
    s, _, _ = run_acoustic_searchlight(
        epochs, subject=subject,
        electrode_idxs=speech_responsive_idxs,
        windows=windows,
        reg_lambda=rl,
        target="categorical_acoustic_cue",
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=torch.float32,
        tol=tol, max_iter=max_iter,
    )
    sweep_rows.append(dict(decoder="acoustic", reg_lambda=rl, mean_test_auc=_mean_auc(s)))

    L.info(f"[behavior_full] reg_lambda={rl}")
    s, _, _ = run_behavior_with_control(
        epochs, subject=subject,
        electrode_idxs=speech_responsive_idxs,
        windows=windows,
        reg_lambda=rl,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=torch.float32,
        tol=tol, max_iter=max_iter,
    )
    # Only score the full model (not baseline) when ranking reg_lambda for the full-model AUC
    full_auc = _mean_auc(s.filter(pl.col("model") == "full"))
    sweep_rows.append(dict(decoder="behavior_full", reg_lambda=rl, mean_test_auc=full_auc))

    L.info(f"[behavior_hga_only] reg_lambda={rl}")
    s, _, _ = run_behavior_hga_only(
        epochs, subject=subject,
        electrode_idxs=speech_responsive_idxs,
        windows=windows,
        reg_lambda=rl,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=torch.float32,
        tol=tol, max_iter=max_iter,
    )
    sweep_rows.append(dict(decoder="behavior_hga_only", reg_lambda=rl, mean_test_auc=_mean_auc(s)))

sweep = pl.DataFrame(sweep_rows)
print(sweep.sort(["decoder", "reg_lambda"]))

# %% [markdown]
# Pick winners: argmax mean_test_auc per decoder.

# %%
winners = {
    row["decoder"]: float(row["reg_lambda"])
    for row in (
        sweep.group_by("decoder")
        .agg(pl.all().sort_by("mean_test_auc", descending=True).first())
        .to_dicts()
    )
}
L.info(f"Winners: {winners}")

# %%
outdir = Path(outdir)
sweep.write_parquet(outdir / "sweep_results.parquet")
(outdir / "reg_lambda_winners.json").write_text(json.dumps({
    "subject": subject,
    "reg_lambda_acoustic": winners["acoustic"],
    "reg_lambda_behavior_full": winners["behavior_full"],
    "reg_lambda_behavior_hga_only": winners["behavior_hga_only"],
}, indent=2))
L.info(f"Wrote sweep_results.parquet + reg_lambda_winners.json to {outdir}")
