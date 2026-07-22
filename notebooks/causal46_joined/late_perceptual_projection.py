# ---
# jupyter:
#   jupytext:
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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import os
import sys
from math import comb
from pathlib import Path

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

from src.data import add_metadata_features
from src.stimuli import PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    extract_hga,
    per_step_class_counts,
    resolve_behavior_col,
)

# %%
subject = "EC250"
# Computed site-type table (early_window_site_types); consumed ONLY for its
# A_significant column to define the projection site pool. NOT the manual
# type1-5 authority (that is early_acoustic_window.csv, read by the aggregate).
site_pool_path = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv"

# Individual b4 bootstrap results per window
b4_windows_path = "outputs/causal46_joined/t_tests/b4_per_window.parquet"

# Individual acoustic endpoint bootstrap results per window
a_windows_path = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_full_all.parquet"

# Unified behaviorally discriminative windows
# TODO use TFCE windows instead
b_windows_path = "outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet"

epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/early_perceptual_projection/EC250"
min_class_k = 3

# window parameters for the HGA sampling
window_size = 4
stride = 4

search_smin = 45
search_smax = 150 # TODO get the word_end-specific logic from decoding etc.

# super-window parameters for the projection comparison
super_window_size = 10
super_window_stride = 10

n_perms = 1000
master_seed = 42
fdr_alpha = 0.05

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = int(window_size)
STRIDE = int(stride)
K = int(min_class_k)
N_PERMS = int(n_perms)
SMIN = int(search_smin)
SMAX = int(search_smax)
MASTER_SEED = int(master_seed)
FDR_ALPHA = float(fdr_alpha)

# Window grid: same range() as searchlight_mean_diff
WINDOW_STARTS = list(range(SMIN, SMAX - WINDOW_SIZE + 1, STRIDE))
N_WINDOWS = len(WINDOW_STARTS)
WINDOW_TMINS = [s / epoch_sfreq + epoch_tmin for s in WINDOW_STARTS]

print(f"subject={subject}  K={K}  N_PERMS={N_PERMS}")
print(f"Window: [{SMIN},{SMAX}] samples = [{SMIN/epoch_sfreq+epoch_tmin:.3f},{SMAX/epoch_sfreq+epoch_tmin:.3f}] s")
print(f"Window grid smin values: {WINDOW_STARTS}  (N_WINDOWS={N_WINDOWS})")

# %%
b_windows = pd.read_parquet(b_windows_path)

# %%
b4_windows = pd.read_parquet("outputs/causal46_joined/t_tests/b4_per_window.parquet")

# %%
a_windows = pd.read_parquet(a_windows_path)

# %%
# Site pool: A_significant sites (same universe as other causal46_joined analyses).
# A_significant = True means the acoustic searchlight test was significant in early_window_site_types.
site_pool_all = pd.read_csv(site_pool_path)
site_pool_subj = site_pool_all[site_pool_all["subject"] == subject]
included_sites = (
    site_pool_subj[site_pool_subj["A_significant"]]
    [["subject", "electrode_idx", "phoneme_pair"]]
    .reset_index(drop=True)
)
n_total_in_pool = len(site_pool_subj)
print(f"Sites in pool for {subject}: {n_total_in_pool}")
print(f"A_significant sites (included): {len(included_sites)}")
if len(included_sites) > 0:
    print(included_sites[["electrode_idx", "phoneme_pair"]].to_string(index=False))

# %%
cell_pool = pd.merge(
    included_sites,
    b_windows.query("ci_excludes_zero"),
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="left",
    indicator=True
)

cell_pool_counts = cell_pool._merge.value_counts()
assert cell_pool_counts.get("right_only", 0) == 0, f"Some behaviorally significant sites are missing from site pool: {cell_pool_counts}"

print(f"Behaviorally significant sites in pool for {subject}: {cell_pool_counts.get('both', 0)}")
cell_pool = cell_pool.query("_merge == 'both'").drop(columns="_merge")

# %%
epochs_path = Path(epoch_dir) / f"{subject}_epo.fif"
ep = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
md_raw = ep.metadata
md = add_metadata_features(md_raw).reset_index(drop=True)
md["subject"] = subject
ep.metadata = md
bhv_col = resolve_behavior_col(md)
print(f"Loaded {len(ep)} epochs; behavior col: {bhv_col}")


# %%
def get_qualifying_steps(md_pp, *, word_end, group_col, ambiguous_threshold=2):
    """Ambiguous steps for one (phoneme_pair, word_end) cell.

    Step s qualifies if: not in endpoints {1,6}, both behavior classes
    present, and minority class count > ambiguous_threshold.
    Matches src.data.get_ambiguous_resampled_steps criterion.
    """
    we_mask = md_pp["word_end"] == word_end
    ambiguous_mask = ~md_pp["resampled"].isin([1, 6])
    steps = sorted(md_pp.loc[we_mask & ambiguous_mask, "resampled"].unique())
    qualifying = []
    for s in steps:
        step_mask = we_mask & ambiguous_mask & (md_pp["resampled"] == s)
        counts = md_pp.loc[step_mask, group_col].value_counts()
        if len(counts) >= 2 and int(counts.min()) > ambiguous_threshold:
            qualifying.append(int(s))
    return qualifying


def compute_a_vector(hga, md_pp):
    """Acoustic template a(w) = mean HGA[step6] - mean HGA[step1], pooled word_ends.

    Positive direction = phoneme_pair[1] (the 'step6' phoneme). No sign flip.
    """
    idx1 = md_pp.index[md_pp["resampled"] == 1].tolist()
    idx6 = md_pp.index[md_pp["resampled"] == 6].tolist()
    a = np.array([
        (hga[idx6, s:s + WINDOW_SIZE].mean() - hga[idx1, s:s + WINDOW_SIZE].mean())
        if (len(idx6) > 0 and len(idx1) > 0) else 0.0
        for s in WINDOW_STARTS
    ])
    return a


def compute_p(hga, md_pp, *, word_end, group_col):
    """
    Compute perceptual contrast:

    p(w) = Σ_s [min_class[s]/N_we] * [mean HGA[class1,s,w] - mean HGA[class0,s,w]]

    class1 = behavior_dummy_forced=1 = heard phoneme_pair[1].

    Returns
    -------
    p : np.ndarray
        Perceptual contrast for each window (N_WINDOWS,).
    min_classes : dict
        Dictionary mapping each qualifying step to the minimum class count for that step.
    per_step_filtered : dict
        Dictionary mapping each qualifying step to a dictionary of class counts for that step.
    N : int
        Total number of trials across all qualifying steps.
    """
    qualifying = get_qualifying_steps(md_pp, word_end=word_end, group_col=group_col)
    if not qualifying:
        return None, None, None, 0

    per_step = per_step_class_counts(
        md_pp, word_end=word_end, qualifying_steps=qualifying, group_col=group_col
    )

    # Apply min_class_k gate
    per_step_filtered = {
        s: by_cls
        for s, by_cls in per_step.items()
        if 0 in by_cls and 1 in by_cls
        and min(len(by_cls[0]), len(by_cls[1])) >= K
    }
    if not per_step_filtered:
        # No qualifying steps after filtering by min_class_k
        return None, None, None, 0

    min_classes = {
        s: min(len(by_cls[0]), len(by_cls[1]))
        for s, by_cls in per_step_filtered.items()
    }
    N = int(sum(min_classes.values()))

    p = np.zeros(N_WINDOWS)
    for s, by_cls in per_step_filtered.items():
        w_s = min_classes[s] / N
        idx1 = by_cls[1]
        idx0 = by_cls[0]
        for i, smin in enumerate(WINDOW_STARTS):
            smax_w = smin + WINDOW_SIZE
            diff = (
                hga[idx1, smin:smax_w].mean()
                - hga[idx0, smin:smax_w].mean()
            )
            p[i] += w_s * diff

    return p, min_classes, per_step_filtered, N


def compute_permutation_null(hga, cell_data, a_hat, rng):
    """Pooled and per-we permutation null distributions (N_PERMS each).

    cell_data: list of (idx1, idx0, weight_pooled, weight_we) tuples.
    Shuffles report labels independently within each cell,
    preserving the cell's observed (n1, n0) split. Same shuffle used
    for all windows within a cell (labels fixed per permutation replicate).

    Returns:
        pi_perm: (N_PERMS,) null
    """
    pi_perm = np.zeros(N_PERMS)

    for idx1, idx0, w_pooled, w_we in cell_data:
        n1 = len(idx1)
        all_idx = np.concatenate([idx1, idx0])
        n_total = len(all_idx)

        # Vectorized permutations: N_PERMS × n_total
        u = rng.random((N_PERMS, n_total))
        perm_matrix = np.argsort(u, axis=1)  # uniform random permutations

        for w_idx, smin in enumerate(WINDOW_STARTS):
            smax_w = smin + WINDOW_SIZE
            X = hga[all_idx, smin:smax_w].mean(axis=1)  # (n_total,)
            X_perm = X[perm_matrix]                       # (N_PERMS, n_total)
            diff_perm = (
                X_perm[:, :n1].mean(axis=1) - X_perm[:, n1:].mean(axis=1)
            )  # (N_PERMS,)
            pi_perm += a_hat[w_idx] * w_pooled * diff_perm

    return pi_perm


# %% [markdown]
# ## Per-cell projection

# %%
results = []

for _, site_row in tqdm(cell_pool.iterrows(), total=len(cell_pool), desc="Cells"):
    subject = site_row["subject"]
    electrode_idx = site_row["electrode_idx"]
    phoneme_pair = site_row["phoneme_pair"]
    word_end = site_row["word_end"]

    # Filter epochs for this cell
    assert ep.metadata is not None
    ep_i = ep[(ep.metadata["phoneme_pair"] == phoneme_pair)
              & (ep.metadata["word_end"] == word_end)]
    md_i = ep_i.metadata
    if len(md_i) == 0:
        continue
    print(len(ep_i), len(md_i), subject, electrode_idx, phoneme_pair, word_end)

    # Compute perceptual template
    b4_matches = b4_windows[
        (b4_windows["subject"] == subject)
        & (b4_windows["electrode_idx"] == electrode_idx)
        & (b4_windows["phoneme_pair"] == phoneme_pair)
        & (b4_windows["word_end"] == word_end)
        & (b4_windows["smin"] >= site_row.smin)
        & (b4_windows["smax"] <= site_row.smax)
    ].sort_values("smin")
    
    # Compute matching acoustic windowed data
    # TODO this needs to be word-end-specific ..
    a_matches = a_windows[
        (a_windows["subject"] == subject)
        & (a_windows["electrode_idx"] == electrode_idx)
        & (a_windows["phoneme_pair"] == phoneme_pair)
        & (a_windows["smin"] >= site_row.smin)
        & (a_windows["smax"] <= site_row.smax)
    ].sort_values("smin")

    assert len(b4_matches) == len(a_matches), f"Mismatch in number of matching windows: {len(b4_matches)} vs {len(a_matches)}"
    assert (b4_matches["smin"].values == a_matches["smin"].values).all(), "Mismatch in smin values between b4 and a matches"

    # Acoustic and perceptual vectors
    a_vec = a_matches.mean_diff_raw_med
    p_vec = b4_matches.mean_diff_raw_med

    projection = np.dot(p_vec, a_vec) / np.linalg.norm(a_vec)

    results.append({
        "subject": subject,
        "electrode_idx": electrode_idx,
        "phoneme_pair": phoneme_pair,
        "word_end": word_end,

        "n_windows": len(b4_matches),

        "projection": projection,
    })


results_df = pd.DataFrame(results)
print(f"\nTotal sites processed: {len(results_df)}")
if len(results_df) > 0:
    print(results_df[["electrode_idx", "phoneme_pair", "projection"]].to_string(index=False))
