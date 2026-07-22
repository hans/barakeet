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
window_size = 2
stride = 2

n_perms = 500000
master_seed = 42
fdr_alpha = 0.05

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = int(window_size)
STRIDE = int(stride)
K = int(min_class_k)
N_PERMS = int(n_perms)
MASTER_SEED = int(master_seed)
FDR_ALPHA = float(fdr_alpha)

print(f"subject={subject}  K={K}  N_PERMS={N_PERMS}")

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


def compute_a_vector(hga, md_pp, smin, smax):
    """Acoustic template a(w) = mean HGA[step6] - mean HGA[step1], pooled word_ends.

    Positive direction = phoneme_pair[1] (the 'step6' phoneme). No sign flip.
    """
    assert len(hga) == len(md_pp)

    step1_mask = md_pp["resampled"] == 1
    step6_mask = md_pp["resampled"] == 6

    if step1_mask.sum() == 0 or step6_mask.sum() == 0:
        raise ValueError(f"No trials for one of the endpoints: step 1 = {step1_mask.sum()}, step 6 = {step6_mask.sum()}")

    window_starts = np.arange(smin, smax - WINDOW_SIZE + 1, STRIDE)
    a = np.array([
        (hga[step6_mask, s:s + WINDOW_SIZE].mean() - hga[step1_mask, s:s + WINDOW_SIZE].mean())
        for s in window_starts
    ])
    return a


def compute_a_vector_null(hga, md_pp, smin, smax, rng):
    """
    Compute null distribution of acoustic template a(w) under label shuffling.

    Shuffles acoustic labels preserving the observed (n_step1, n_step6) split.

    Returns:
        a_perm: (N_PERMS, N_WINDOWS) null
    """
    assert len(hga) == len(md_pp)

    step1_mask = md_pp["resampled"] == 1
    step6_mask = md_pp["resampled"] == 6

    if step1_mask.sum() == 0 or step6_mask.sum() == 0:
        raise ValueError(f"No trials for one of the endpoints: step 1 = {step1_mask.sum()}, step 6 = {step6_mask.sum()}")

    window_starts = np.arange(smin, smax - WINDOW_SIZE + 1, STRIDE)
    n_windows = len(window_starts)
    a_perm = np.zeros((N_PERMS, n_windows))

    idx_step1 = np.where(step1_mask)[0]
    idx_step6 = np.where(step6_mask)[0]
    idx_all = np.concatenate([idx_step1, idx_step6])
    n_step1 = len(idx_step1)
    n_total = n_step1 + len(idx_step6)

    # Vectorized permutations: N_PERMS × n_total
    u = rng.random((N_PERMS, n_total))
    perm_matrix = np.argsort(u, axis=1)  # uniform random permutations

    for w_idx, smin_w in enumerate(window_starts):
        smax_w = smin_w + WINDOW_SIZE
        X = hga[idx_all, smin_w:smax_w].mean(axis=1)  # (n_total,)
        X_perm = X[perm_matrix]               # (N_PERMS, n_total)
        diff_perm = (
            X_perm[:, :n_step1].mean(axis=1) - X_perm[:, n_step1:].mean(axis=1)
        )  # (N_PERMS,)
        a_perm[:, w_idx] = diff_perm

    return a_perm


def compute_p(hga, md_pp, word_end, group_col,
              smin, smax):
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
    assert len(hga) == len(md_pp)

    qualifying = get_qualifying_steps(md_pp, word_end=word_end, group_col=group_col)
    if not qualifying:
        return None, None, None, 0, None

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
        return None, None, None, 0, None

    min_classes = {
        s: min(len(by_cls[0]), len(by_cls[1]))
        for s, by_cls in per_step_filtered.items()
    }
    N = int(sum(min_classes.values()))

    window_starts = np.arange(smin, smax - WINDOW_SIZE + 1, STRIDE)
    print(smin, smax, window_starts)
    n_windows = len(window_starts)
    p = np.zeros(n_windows)
    traces = np.zeros((2, len(ep.times)))
    for s, by_cls in per_step_filtered.items():
        w_s = min_classes[s] / N
        idx1 = by_cls[1]
        idx0 = by_cls[0]
        for i, smin in enumerate(window_starts):
            smax_w = smin + window_size
            diff = (
                hga[idx1, smin:smax_w].mean()
                - hga[idx0, smin:smax_w].mean()
            )
            p[i] += w_s * diff
        
        traces[0] += w_s * hga[idx0].mean(axis=0)
        traces[1] += w_s * hga[idx1].mean(axis=0)

    return p, min_classes, per_step_filtered, N, traces


def compute_permutation_null(hga, cell_data, a_hat, rng):
    """
    Compute permutation null distributions.

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
hga = ep.copy().apply_baseline((None, 0)).get_data()

# %%
cell_pool

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
    assert md_i is not None

    if len(md_i) == 0:
        continue
    hga_i = extract_hga(ep_i, electrode_idx)
    print(len(ep_i), len(md_i), subject, electrode_idx, phoneme_pair, word_end)

    smin, smax = int(site_row.smin), int(site_row.smax)

    # Compute perceptual template
    p_vec, min_classes, per_step_filtered, N, p_traces = compute_p(
        hga=hga_i,
        md_pp=md_i,
        word_end=word_end,
        group_col=bhv_col,
        smin=smin,
        smax=smax,
    )
    if p_vec is None:
        print(f"No qualifying steps for {subject} {electrode_idx} {phoneme_pair} {word_end}")
        continue

    # Compute acoustic vector
    a_vec = compute_a_vector(hga_i, md_i, smin, smax)

    projection = np.dot(p_vec, a_vec)

    # Compute null distribution of projection
    a_null = compute_a_vector_null(hga_i, md_i, smin, smax, rng=np.random.default_rng(MASTER_SEED))
    projection_null = np.dot(p_vec, a_null.T)
    projection_null_ci_low, projection_null_ci_high = \
        np.percentile(projection_null, [2.5, 97.5])

    pval = (np.sum(projection_null >= projection) + 1) / (N_PERMS + 1)

    results.append({
        "subject": subject,
        "electrode_idx": electrode_idx,
        "phoneme_pair": phoneme_pair,
        "word_end": word_end,

        "n_windows": p_vec.shape[0],
        "smin": smin,
        "smax": smax,
        "projection": projection,

        "projection_null_mean": projection_null.mean(),
        "projection_null_ci_low": projection_null_ci_low,
        "projection_null_ci_high": projection_null_ci_high,
        "projection_p_value": pval,
    })


results_df = pd.DataFrame(results)
print(f"\nTotal sites processed: {len(results_df)}")

# %%
results_df

# %%
site_row

# %%
xs = ep_i.times

for step in [1, 6]:
    step_mask = md_i["resampled"] == step
    plt.plot(xs, hga_i[step_mask].mean(axis=0), label=f"Step {step}")

plt.xlim(-0.05, 0.8)
ax = plt.gca()
ax.axvspan(site_row.smin / 100 - 0.4, site_row.smax / 100 - 0.4,
           alpha=0.3)
ax.legend()

# %%
plt.plot(xs, p_traces[0], label="Perceptual class 0", color="blue")
plt.plot(xs, p_traces[1], label="Perceptual class 1", color="orange")
plt.xlim(-0.05, 0.8)
plt.axvspan(site_row.smin / 100 - 0.4, site_row.smax / 100 - 0.4,
              alpha=0.3)

# %%
p_vec

# %%
a_vec

# %%
site_row[["smin", "smax"]]

# %%
