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
    audit_class_balance,
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
# Audit class balance on this tuning subject before running the sweep. Uses the
# same StratifiedKFold seed/config as the fits, so per-fold minority counts
# reported here are exactly what the decoder will see. Any "low" or "skipped"
# group here contributes noisy per-fold AUCs to λ-winner selection.

# %%
outdir = Path(outdir)
audit = audit_class_balance(
    epochs, subject=subject,
    n_folds=n_folds, cv_random_state=cv_random_state,
)
L.info(f"Class-balance audit for tuning subject {subject}:\n{audit}")

status_counts = (
    audit.group_by(["decoder", "status"]).len().sort(["decoder", "status"])
)
L.info(f"Per-decoder status counts:\n{status_counts}")

audit.write_parquet(outdir / "class_balance_audit.parquet")

# %% [markdown]
# Sweep each decoder independently.

# %%
def _tag(scores: pl.DataFrame, decoder: str, rl: float) -> pl.DataFrame:
    """Add decoder name + reg_lambda columns for later aggregation."""
    return scores.with_columns(
        pl.lit(decoder).alias("decoder"),
        pl.lit(rl).alias("reg_lambda"),
    )


all_scores: list[pl.DataFrame] = []

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
    all_scores.append(_tag(s, "acoustic", rl))

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
    # Keep both full + baseline rows; use the `model` column to filter downstream.
    all_scores.append(_tag(s, "behavior_full", rl))

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
    all_scores.append(_tag(s, "behavior_hga_only", rl))

all_scores_df = pl.concat(all_scores, how="diagonal_relaxed")

# For winner selection: match how λ will actually be USED downstream — each
# site picks its own peak window — so the tuning metric is:
#   1. mean test AUC across folds per (site, window)
#   2. max across windows per site  (= the "peak" window for that site)
#   3. mean across sites per (decoder, reg_lambda)
# A plain mean over every (site × window × fold) cell is dominated by the many
# windows on many electrodes with no signal; stronger λ shrinks those toward
# 0.5 more tightly, which spuriously raises the mean.
#
# A "site" = (phoneme_pair, electrode_idx, word_end). word_end only exists for
# the behavior decoders — it is null for acoustic rows and forms its own
# (single-valued) group there, which gives the right grouping for both.
winner_input = all_scores_df.filter(
    (pl.col("decoder") != "behavior_full") | (pl.col("model") == "full")
)

site_cols = ["decoder", "reg_lambda", "phoneme_pair", "electrode_idx", "word_end"]
window_cols = site_cols + ["smin", "smax"]

per_site_per_window = (
    winner_input.group_by(window_cols)
    .agg(pl.col("test_roc_auc").mean().alias("auc_cv_mean"))
)
per_site_peak = (
    per_site_per_window.group_by(site_cols)
    .agg(pl.col("auc_cv_mean").max().alias("peak_auc"))
)
sweep = (
    per_site_peak.group_by(["decoder", "reg_lambda"])
    .agg(pl.col("peak_auc").mean().alias("mean_peak_auc"))
    .sort(["decoder", "reg_lambda"])
)
print(sweep)

# Fold-level AUC SD at each site's peak window, averaged across sites — a
# direct read on how noisy the per-λ AUCs that drive winner selection are. If
# the winner's `mean_peak_auc` lead over runners-up is smaller than
# `mean_site_fold_sd`, the pick is inside the noise floor.
peak_windows = (
    per_site_peak.join(per_site_per_window, on=site_cols, how="left")
    .filter(pl.col("auc_cv_mean") == pl.col("peak_auc"))
    .select(window_cols)
    .unique()
)
fold_variance = (
    winner_input.join(peak_windows, on=window_cols, how="inner")
    .group_by(window_cols)
    .agg(pl.col("test_roc_auc").std().alias("auc_fold_sd"))
    .group_by(["decoder", "reg_lambda"])
    .agg(pl.col("auc_fold_sd").mean().alias("mean_site_fold_sd"))
    .sort(["decoder", "reg_lambda"])
)
print(fold_variance)
L.info(f"Mean across-fold AUC SD at peak window per (decoder, λ):\n{fold_variance}")

# %% [markdown]
# Pick winners: argmax mean_peak_auc per decoder.

# %%
winners = {
    row["decoder"]: float(row["reg_lambda"])
    for row in (
        sweep.group_by("decoder")
        .agg(pl.all().sort_by("mean_peak_auc", descending=True).first())
        .to_dicts()
    )
}
L.info(f"Winners: {winners}")

# %%
sweep.write_parquet(outdir / "sweep_results.parquet")
all_scores_df.write_parquet(outdir / "sweep_all_scores.parquet")
(outdir / "reg_lambda_winners.json").write_text(json.dumps({
    "subject": subject,
    "reg_lambda_acoustic": winners["acoustic"],
    "reg_lambda_behavior_full": winners["behavior_full"],
    "reg_lambda_behavior_hga_only": winners["behavior_hga_only"],
}, indent=2))
L.info(
    f"Wrote sweep_results.parquet ({sweep.height} rows), "
    f"sweep_all_scores.parquet ({all_scores_df.height} rows), "
    f"reg_lambda_winners.json to {outdir}"
)
