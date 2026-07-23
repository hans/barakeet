# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Acoustic transfer: phonemic peak window → behavioral window
#
# For each row in b_windows.parquet (one per cell × window_id), fits an
# acoustic decoder (categorical_acoustic_cue, unambiguous trials) on BOTH
# the phonemic peak window and the behavioral target window in a single
# `run_acoustic_searchlight` call, then reports fold-level AUCs for each.
#
# Behavioral sub-window selection (fixed width = window_size):
# - `narrower_than_decoder=True`: use pre-computed `behav_decoder_smin/smax`.
# - `narrower_than_decoder=False` (union ≥ window_size): slide a window_size
#   window over `[smin, smax]` and pick the position that maximises
#   `|mean_HGA(acoustic_cue==0) − mean_HGA(acoustic_cue==1)|` on unambiguous
#   trials (resampled ∈ {1, 6}) for this phoneme pair.
#
# Output: one row per (b_windows row × fold).

# %% tags=["parameters"]
subject = "EC282"
epochs_path = "outputs/epochs_preprocessed/EC282_epo.fif"
late_projection_path = "outputs/causal46_joined/late_perceptual_projection/late_perceptual_projection_results.csv"
reg_lambda_winners_path = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json"
outdir = "."

window_size = 15
n_folds = 5
cv_random_state = 42
device = "cpu"
tol = 1e-6
max_iter = 50

# %%
import json
import sys
from pathlib import Path

import mne
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(".").resolve()))

from src.data import add_metadata_features
from src.models.causal6 import run_acoustic_searchlight

# %%
_winners = json.loads(Path(reg_lambda_winners_path).read_text())
reg_lambda = float(_winners["reg_lambda_acoustic"])
print(f"reg_lambda={reg_lambda}  window_size={window_size}  n_folds={n_folds}  device={device!r}")

# %%
epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)

# %%
late_perceptual_df = pl.read_csv(late_projection_path).filter(pl.col("subject") == subject)
print(f"late_perceptual rows for {subject}: {late_perceptual_df.height}")


# %% [markdown]
# ## Sub-window selection helper

# %%
def pick_best_subwindow(
    epoch_data: np.ndarray,
    md,
    phoneme_pair: str,
    union_smin: int,
    union_smax: int,
    window_size: int,
) -> tuple:
    """Slide a window_size window over [union_smin, union_smax) and return the
    position with the highest |mean(class_0) - mean(class_1)| on unambiguous trials."""
    mask = (md["phoneme_pair"] == phoneme_pair) & (md["resampled"].isin([1, 6]))
    X = epoch_data[mask.values]  # (N_unambig, N_times)
    y = md.loc[mask, "categorical_acoustic_cue"].values.astype(int)

    best_smin = union_smin
    best_sep = -1.0
    for s in range(union_smin, union_smax - window_size + 1):
        X_win = X[:, s:s + window_size]
        m0 = float(X_win[y == 0].mean()) if (y == 0).any() else 0.0
        m1 = float(X_win[y == 1].mean()) if (y == 1).any() else 0.0
        sep = abs(m0 - m1)
        if sep > best_sep:
            best_sep = sep
            best_smin = s

    return best_smin, best_smin + window_size


# %% [markdown]
# ## Main processing loop

# %%
output_rows: list = []
md = epochs.metadata

for (electrode_idx, phoneme_pair), group in late_perceptual_df.group_by(
    ["electrode_idx", "phoneme_pair"], maintain_order=True
):
    electrode_idx = int(electrode_idx)

    # Phonemic peak window is the same for every row in this group.
    phon_smin = int(group["phon_smin"][0])
    phon_smax = int(group["phon_smax"][0])

    # Epoch data for this electrode (needed for argmax selection).
    ep_data = epochs.get_data(picks=[electrode_idx]).squeeze(1)  # (N_epochs, N_times)

    # Resolve behavioral sub-window
    row_behav_windows: list = []
    for row in group.iter_rows(named=True):
        bsmin, bsmax = pick_best_subwindow(
            ep_data, md,
            phoneme_pair=phoneme_pair,
            union_smin=int(row["smin"]),
            union_smax=int(row["smax"]),
            window_size=window_size,
        )
        row_behav_windows.append((bsmin, bsmax))

    # Unique windows: phonemic peak + all behavioral sub-windows.
    unique_windows = sorted({(phon_smin, phon_smax)} | set(row_behav_windows))
    windows_arr = np.array(unique_windows, dtype=np.int64)

    assert (windows_arr[:, 1] - windows_arr[:, 0] == window_size).all(), (
        f"Window width mismatch for e{electrode_idx} {phoneme_pair}: "
        f"{(windows_arr[:, 1] - windows_arr[:, 0]).tolist()}"
    )

    print(
        f"  {subject} e{electrode_idx} {phoneme_pair}: "
        f"{len(unique_windows)} window(s) for "
        f"{group.height} b_windows row(s)"
    )

    scores, _, _ = run_acoustic_searchlight(
        epochs,
        subject=subject,
        electrode_idxs=[electrode_idx],
        windows=windows_arr,
        target="categorical_acoustic_cue",
        reg_lambda=reg_lambda,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
        device=device,
        tol=tol,
        max_iter=max_iter,
    )
    scores_pp = scores.filter(pl.col("phoneme_pair") == phoneme_pair)

    # Index by (smin, smax, fold) for O(1) lookup.
    scores_idx = {
        (int(r["smin"]), int(r["smax"]), int(r["fold"])): float(r["test_roc_auc"])
        for r in scores_pp.iter_rows(named=True)
    }

    for row, (behav_smin, behav_smax) in zip(
        group.iter_rows(named=True), row_behav_windows
    ):
        for fold in range(n_folds):
            phon_auc = scores_idx.get((phon_smin, phon_smax, fold))
            behav_auc = scores_idx.get((behav_smin, behav_smax, fold))
            if phon_auc is None or behav_auc is None:
                continue
            output_rows.append({
                "subject": subject,
                "electrode_idx": electrode_idx,
                "phoneme_pair": phoneme_pair,
                "word_end": row["word_end"],
                "window_id": row["window_id"],
                "phon_smin": phon_smin,
                "phon_smax": phon_smax,
                "behav_smin": behav_smin,
                "behav_smax": behav_smax,
                "fold": fold,
                "phon_roc_auc": phon_auc,
                "behav_roc_auc": behav_auc,

                # facts about the source
                "late_projection": row["projection"],
                "late_projection_p_value": row["projection_p_value"],
                "late_projection_q_value": row["projection_q_value"],
            })

print(f"Total output rows: {len(output_rows)}")

# %% [markdown]
# ## Write output

# %%
EXPECTED_COLS = [
    "subject", "electrode_idx", "phoneme_pair", "word_end", "window_id",
    "phon_smin", "phon_smax", "behav_smin", "behav_smax",
    "late_projection", "late_projection_p_value", "late_projection_q_value",
    "fold", "phon_roc_auc", "behav_roc_auc",
]

if output_rows:
    out_df = pl.DataFrame(output_rows)
else:
    out_df = pl.DataFrame(
        {col: pl.Series([], dtype=pl.Utf8) for col in EXPECTED_COLS}
    ).cast({
        "electrode_idx": pl.Int64,
        "window_id": pl.Int64,
        "phon_smin": pl.Int64,
        "phon_smax": pl.Int64,
        "behav_smin": pl.Int64,
        "behav_smax": pl.Int64,
        "fold": pl.Int64,
        "phon_roc_auc": pl.Float64,
        "behav_roc_auc": pl.Float64,

        "late_projection": pl.Float64,
        "late_projection_p_value": pl.Float64,
        "late_projection_q_value": pl.Float64,
    })

missing = set(EXPECTED_COLS) - set(out_df.columns)
assert not missing, f"Output missing columns: {missing}"

out_df.write_parquet(Path(outdir) / "scores.parquet")
print(f"Wrote {out_df.height} rows to {outdir}/scores.parquet")
print(out_df.head(5))
