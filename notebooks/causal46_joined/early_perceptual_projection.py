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
# # Early Perceptual Projection (per subject)
#
# Projection-based detection of perceptual early responses.
# Per site (subject, electrode_idx, phoneme_pair), compute how much the
# report-driven contrast on ambiguous trials resembles the acoustic contrast
# on unambiguous trials, integrated over the early window.
#
# - `â(w)` = unit-L2-normalized acoustic template: mean HGA[step6] − mean HGA[step1]
# - `p(w)` = deterministic B4 min_class-weighted within-completion perceptual contrast
# - `π = ⟨â, p⟩` = projection
#
# Template uses unambiguous trials; testing uses ambiguous trials — structurally
# independent by design. No per-site sign flip (avoids acoustic_sign circularity).
#
# See: `docs/superpowers/plans/2026-07-16-early-perceptual-projection-spec.md`

# %%
from __future__ import annotations

import os
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
from _projection import get_qualifying_steps  # noqa: E402

# %% tags=["parameters"]
subject = "EC250"
# Computed site-type table (early_window_site_types); consumed ONLY for its
# A_significant column to define the projection site pool. NOT the manual
# type1-5 authority (that is early_acoustic_window.csv, read by the aggregate).
site_pool_path = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv"
epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/early_perceptual_projection/EC250"
min_class_k = 3
window_size = 5
stride = 5
ac_search_smin = 45
ac_search_smax = 68
n_perms = 10000
master_seed = 42
fdr_alpha = 0.05

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = int(window_size)
STRIDE = int(stride)
K = int(min_class_k)
N_PERMS = int(n_perms)
SMIN = int(ac_search_smin)
SMAX = int(ac_search_smax)
MASTER_SEED = int(master_seed)
FDR_ALPHA = float(fdr_alpha)

# Window grid: same range() as searchlight_mean_diff
WINDOW_STARTS = list(range(SMIN, SMAX - WINDOW_SIZE + 1, STRIDE))
N_WINDOWS = len(WINDOW_STARTS)
WINDOW_TMINS = [s / epoch_sfreq + epoch_tmin for s in WINDOW_STARTS]

print(f"subject={subject}  K={K}  N_PERMS={N_PERMS}")
print(f"Early window: [{SMIN},{SMAX}] samples = [{SMIN/epoch_sfreq+epoch_tmin:.3f},{SMAX/epoch_sfreq+epoch_tmin:.3f}] s")
print(f"Window grid smin values: {WINDOW_STARTS}  (N_WINDOWS={N_WINDOWS})")

# %% [markdown]
# ## Load inputs

# %%
# Site pool: A_significant sites (same universe as other causal46_joined analyses).
# A_significant = True means the acoustic searchlight test was significant in early_window_site_types.
# This is broader than phon_peaks_all 'significant' (global FDR across all subjects/sites/times),
# which would exclude many subjects entirely.
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
epochs_path = Path(epoch_dir) / f"{subject}_epo.fif"
ep = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
md_raw = ep.metadata
md = add_metadata_features(md_raw).reset_index(drop=True)
md["subject"] = subject
ep.metadata = md
bhv_col = resolve_behavior_col(md)
print(f"Loaded {len(ep)} epochs; behavior col: {bhv_col}")

# %% [markdown]
# ## Helper functions

# %%
# get_qualifying_steps is imported from _projection (shared with the late
# projection notebook; moved there verbatim — issue #22).


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


def compute_p_we(hga, md_pp, *, word_end, group_col):
    """Deterministic B4 min_class-weighted perceptual contrast for one word_end.

    Returns (p_we, min_classes_dict, per_step_dict, N_we) or
    (None, None, None, 0) if no valid cells.

    p_we(w) = Σ_s [min_class[s]/N_we] * [mean HGA[class1,s,w] - mean HGA[class0,s,w]]
    class1 = behavior_dummy_forced=1 = heard phoneme_pair[1].
    All trials in each class used (no resampling). min_class[s] only enters as weight.
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
        return None, None, None, 0

    min_classes = {
        s: min(len(by_cls[0]), len(by_cls[1]))
        for s, by_cls in per_step_filtered.items()
    }
    N_we = int(sum(min_classes.values()))

    p_we = np.zeros(N_WINDOWS)
    for s, by_cls in per_step_filtered.items():
        w_s = min_classes[s] / N_we
        idx1 = by_cls[1]
        idx0 = by_cls[0]
        for i, smin in enumerate(WINDOW_STARTS):
            smax_w = smin + WINDOW_SIZE
            diff = (
                hga[idx1, smin:smax_w].mean()
                - hga[idx0, smin:smax_w].mean()
            )
            p_we[i] += w_s * diff

    return p_we, min_classes, per_step_filtered, N_we


def compute_permutation_null(hga, cell_data, a_hat, rng):
    """Pooled and per-we permutation null distributions (N_PERMS each).

    cell_data: list of (idx1, idx0, weight_pooled, weight_we, we_key) tuples.
    Shuffles report labels independently within each (we, step) cell,
    preserving the cell's observed (n1, n0) split. Same shuffle used
    for all windows within a cell (labels fixed per permutation replicate).

    Returns:
        pi_perm: (N_PERMS,) pooled null
        pi_perm_by_we: {we_key: (N_PERMS,)} per-word_end null
    """
    we_keys = list({c[4] for c in cell_data})
    pi_perm = np.zeros(N_PERMS)
    pi_perm_by_we = {we: np.zeros(N_PERMS) for we in we_keys}

    for idx1, idx0, w_pooled, w_we, we_key in cell_data:
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
            pi_perm_by_we[we_key] += a_hat[w_idx] * w_we * diff_perm

    return pi_perm, pi_perm_by_we


# %% [markdown]
# ## Per-site projection loop

# %%
results = []
null_arrays = {}  # site_key → (N_PERMS,) null π array

if len(included_sites) == 0:
    print("No FDR-significant sites for this subject — writing empty outputs.")
else:
    for site_i, site_row in tqdm(
        included_sites.iterrows(), total=len(included_sites), desc="sites"
    ):
        elec_idx = int(site_row["electrode_idx"])
        pp = site_row["phoneme_pair"]
        word_ends = PHONEME_PAIR_TO_WORD_ENDS[pp]
        site_key = f"{subject}_{elec_idx}_{pp}"

        # Independent per-site RNG stream (master + offset)
        site_offset = site_i
        rng = np.random.default_rng(MASTER_SEED + int(site_offset))

        # Subset epochs to this phoneme pair
        ep_pp = ep[ep.metadata["phoneme_pair"] == pp]
        md_pp = ep_pp.metadata.reset_index(drop=True)
        hga = extract_hga(ep_pp, elec_idx)

        n_step1 = int((md_pp["resampled"] == 1).sum())
        n_step6 = int((md_pp["resampled"] == 6).sum())

        # Acoustic template â
        a = compute_a_vector(hga, md_pp)
        a_norm = float(np.linalg.norm(a))

        base_rec = dict(
            subject=subject,
            electrode_idx=elec_idx,
            phoneme_pair=pp,
            a_norm=a_norm,
            n_step1=n_step1,
            n_step6=n_step6,
            window_smin=SMIN,
            window_smax=SMAX,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            n_perms=N_PERMS,
            master_seed=MASTER_SEED,
            site_offset=int(site_offset),
        )

        if a_norm < 1e-12:
            print(f"  [{site_key}] near-zero acoustic template — skipping")
            results.append({**base_rec, **{
                k: np.nan for k in [
                    "pi_pooled", "pi_raw_pooled", "pi_we0", "pi_we1",
                    "p_one_tailed", "p_two_tailed",
                    "p_one_tailed_we0", "p_two_tailed_we0",
                    "p_one_tailed_we1", "p_two_tailed_we1",
                    "null_mean", "null_sd",
                    "n_qualifying_cells_we0", "n_qualifying_cells_we1",
                    "n_we0_total", "n_we1_total",
                    "exhaustive", "perm_space", "min_p",
                ]
            }})
            continue

        a_hat = a / a_norm

        # Perceptual contrast per word_end
        p_by_we, N_by_we, per_step_by_we, min_classes_by_we = {}, {}, {}, {}
        n_cells_by_we = {}
        for we in word_ends:
            p_we, min_classes, per_step, N_we = compute_p_we(
                hga, md_pp, word_end=we, group_col=bhv_col
            )
            if p_we is not None:
                p_by_we[we] = p_we
                N_by_we[we] = N_we
                per_step_by_we[we] = per_step
                min_classes_by_we[we] = min_classes
                n_cells_by_we[we] = len(per_step)

        n_we0_total = N_by_we.get(word_ends[0], 0)
        n_we1_total = N_by_we.get(word_ends[1], 0)
        n_cells_we0 = n_cells_by_we.get(word_ends[0], 0)
        n_cells_we1 = n_cells_by_we.get(word_ends[1], 0)

        if not p_by_we:
            print(f"  [{site_key}] no valid perceptual cells — skipping")
            results.append({**base_rec, **{
                k: np.nan for k in [
                    "pi_pooled", "pi_raw_pooled", "pi_we0", "pi_we1",
                    "p_one_tailed", "p_two_tailed",
                    "p_one_tailed_we0", "p_two_tailed_we0",
                    "p_one_tailed_we1", "p_two_tailed_we1",
                    "null_mean", "null_sd",
                    "exhaustive", "perm_space", "min_p",
                ]
            }, "n_qualifying_cells_we0": n_cells_we0,
               "n_qualifying_cells_we1": n_cells_we1,
               "n_we0_total": n_we0_total, "n_we1_total": n_we1_total})
            continue

        # Pool p across word_ends (min_class-weighted average)
        total_N = sum(N_by_we.values())
        p_pooled = sum(N_by_we[we] * p_by_we[we] for we in p_by_we) / total_N

        pi_pooled = float(np.dot(a_hat, p_pooled))
        pi_raw_pooled = float(np.dot(a, p_pooled))
        pi_by_we = {we: float(np.dot(a_hat, p_by_we[we])) for we in p_by_we}

        # Cell data for permutation null
        cell_data = []
        for we in p_by_we:
            N_we = N_by_we[we]
            for s, by_cls in per_step_by_we[we].items():
                min_c = min_classes_by_we[we][s]
                w_pooled = min_c / total_N
                w_we = min_c / N_we
                cell_data.append((by_cls[1], by_cls[0], w_pooled, w_we, we))

        # Permutation space size (per-cell product of C(n1+n0, n1))
        _space = 1
        for idx1_c, idx0_c, *_ in cell_data:
            _space *= comb(len(idx1_c) + len(idx0_c), len(idx1_c))
            if _space > N_PERMS * 100:  # cap to avoid integer overflow
                _space = N_PERMS * 100 + 1
                break
        perm_space = int(_space)
        is_exhaustive = perm_space <= N_PERMS
        min_p = 1.0 / perm_space if perm_space > 0 else np.nan

        # Permutation null
        pi_perm, pi_perm_by_we = compute_permutation_null(hga, cell_data, a_hat, rng)

        # Per-site p-values (one-tailed positive, two-tailed)
        p_one = float(np.mean(pi_perm >= pi_pooled))
        p_two = float(np.mean(np.abs(pi_perm) >= abs(pi_pooled)))

        # Per-word_end p-values (descriptive)
        def _pvals(obs, null):
            p1 = float(np.mean(null >= obs))
            p2 = float(np.mean(np.abs(null) >= abs(obs)))
            return p1, p2

        we0 = word_ends[0]
        we1 = word_ends[1]
        if we0 in pi_by_we and we0 in pi_perm_by_we:
            p1_we0, p2_we0 = _pvals(pi_by_we[we0], pi_perm_by_we[we0])
        else:
            p1_we0, p2_we0 = np.nan, np.nan
        if we1 in pi_by_we and we1 in pi_perm_by_we:
            p1_we1, p2_we1 = _pvals(pi_by_we[we1], pi_perm_by_we[we1])
        else:
            p1_we1, p2_we1 = np.nan, np.nan

        null_arrays[site_key] = pi_perm

        results.append({
            **base_rec,
            "pi_pooled": pi_pooled,
            "pi_raw_pooled": pi_raw_pooled,
            "pi_we0": pi_by_we.get(we0, np.nan),
            "pi_we1": pi_by_we.get(we1, np.nan),
            "p_one_tailed": p_one,
            "p_two_tailed": p_two,
            "p_one_tailed_we0": p1_we0,
            "p_two_tailed_we0": p2_we0,
            "p_one_tailed_we1": p1_we1,
            "p_two_tailed_we1": p2_we1,
            "null_mean": float(pi_perm.mean()),
            "null_sd": float(pi_perm.std()),
            "n_qualifying_cells_we0": n_cells_we0,
            "n_qualifying_cells_we1": n_cells_we1,
            "n_we0_total": n_we0_total,
            "n_we1_total": n_we1_total,
            "exhaustive": is_exhaustive,
            "perm_space": perm_space,
            "min_p": min_p,
        })

results_df = pd.DataFrame(results)
print(f"\nTotal sites processed: {len(results_df)}")
if len(results_df) > 0:
    print(results_df[["electrode_idx", "phoneme_pair", "pi_pooled", "p_one_tailed", "p_two_tailed"]].to_string(index=False))

# %% [markdown]
# ## Valid sites summary (cross-subject FDR in aggregate notebook)

# %%
valid_mask = results_df["pi_pooled"].notna() if len(results_df) > 0 else pd.Series([], dtype=bool)
valid_df = results_df[valid_mask].copy() if len(results_df) > 0 else results_df.copy()

n_sig_uncorr_1t = int((valid_df["p_one_tailed"] < FDR_ALPHA).sum()) if len(valid_df) > 0 else 0
n_sig_uncorr_2t = int((valid_df["p_two_tailed"] < FDR_ALPHA).sum()) if len(valid_df) > 0 else 0
print(f"Valid sites: {len(valid_df)} / {len(results_df)}")
print(f"Uncorrected p < {FDR_ALPHA}  one-tailed: {n_sig_uncorr_1t}  two-tailed: {n_sig_uncorr_2t}")
print("(Cross-subject BH-FDR applied in aggregate notebook.)")
if len(valid_df) > 0:
    n_exhaustive = int(valid_df["exhaustive"].sum()) if "exhaustive" in valid_df.columns else 0
    print(f"Exhaustive permutation sites: {n_exhaustive} / {len(valid_df)}")

# %% [markdown]
# ## Null calibration diagnostic

# %%
if null_arrays:
    null_means = [arr.mean() for arr in null_arrays.values()]
    null_sds = [arr.std() for arr in null_arrays.values()]
    null_mean_max_abs = float(np.max(np.abs(null_means))) if null_means else 0.0
    print(f"Null calibration (per-site null mean):")
    print(f"  max |mean|: {null_mean_max_abs:.4f}  (should be ≈ 0)")
    flagged = [k for k, arr in null_arrays.items() if abs(arr.mean()) > 0.1 * arr.std()]
    if flagged:
        print(f"  WARNING: {len(flagged)} sites with null mean > 0.1×SD: {flagged}")
    else:
        print("  All sites pass null-mean check.")

# %% [markdown]
# ## Plots

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f"{subject} — early perceptual projection diagnostics")

ax = axes[0]
if len(valid_df) > 0:
    pi_vals = valid_df["pi_pooled"].dropna().values
    # Pooled null for display only — all inference uses per-site nulls
    pooled_null = np.concatenate(list(null_arrays.values())) if null_arrays else np.array([])
    if len(pooled_null) > 0:
        ax.hist(pooled_null, bins=60, density=True, alpha=0.4, color="gray", label="pooled null (display)")
    ax.scatter(pi_vals, np.zeros_like(pi_vals), zorder=5, s=40, c="steelblue", label="observed π")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("π (pooled)")
    ax.set_ylabel("density")
    ax.set_title("π distribution\n(per-site nulls for inference)")
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, "no sites", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("π distribution")

ax = axes[1]
if null_arrays:
    null_sds_plot = [arr.std() for arr in null_arrays.values()]
    n_trials_plot = valid_df["n_we0_total"].fillna(0) + valid_df["n_we1_total"].fillna(0)
    n_trials_plot = n_trials_plot.values[:len(null_sds_plot)]
    ax.scatter(n_trials_plot, null_sds_plot, s=30, alpha=0.7, color="darkorange")
    ax.set_xlabel("qualifying trial count (we0 + we1)")
    ax.set_ylabel("null SD")
    ax.set_title("null SD vs trial count\n(expect ~1/√n)")
else:
    ax.text(0.5, 0.5, "no sites", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("null SD vs trial count")

ax = axes[2]
if len(valid_df) > 0:
    pi_v = valid_df["pi_pooled"].dropna().values
    a_v = valid_df["a_norm"].dropna().values[:len(pi_v)]
    sc = ax.scatter(a_v, pi_v, s=30, alpha=0.7, c="steelblue")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("‖a‖ (acoustic template magnitude)")
    ax.set_ylabel("π (pooled)")
    ax.set_title("π vs ‖a‖")
else:
    ax.text(0.5, 0.5, "no sites", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("π vs ‖a‖")

plt.tight_layout()
fig.savefig(OUT_DIR / "pi_dist.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Saved pi_dist.png")

# %% [markdown]
# ## Save outputs

# %%
results_df.to_csv(OUT_DIR / "site_results.csv", index=False)
print(f"Saved site_results.csv  ({len(results_df)} rows)")

np.savez_compressed(str(OUT_DIR / "null_pi.npz"), **{k: v for k, v in null_arrays.items()})
print(f"Saved null_pi.npz  ({len(null_arrays)} arrays)")

print("\nDone.")
