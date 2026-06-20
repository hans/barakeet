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
#     display_name: barakeet
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Acoustic transfer — aggregate
#
# Concatenates per-subject acoustic transfer scores and produces summary plots.
#
# Plots:
# 1. Paired scatter: fold-mean phon vs. behav AUC per cell (color = ci_excludes_zero).
# 2. Transfer drop distribution: (phon − behav) AUC, stratified by ci_excludes_zero.
# 3. Transfer drop by phoneme pair.

# %% tags=["parameters"]
per_subject_paths = []   # list of per-subject scores.parquet paths
outdir = "."

# %%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(".").resolve()))
from src.viz_paper import epoch_sfreq, epoch_tmin

# %% [markdown]
# ## Concatenate per-subject scores

# %%
dfs = [pl.read_parquet(p) for p in per_subject_paths]
non_empty = [d for d in dfs if d.height > 0]
if non_empty:
    scores_all = pl.concat(non_empty)
else:
    scores_all = dfs[0] if dfs else pl.DataFrame()

print(f"Total rows: {scores_all.height} from {len(dfs)} subjects "
      f"({len(non_empty)} non-empty)")
if scores_all.height > 0:
    print(scores_all.head(5))

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
scores_all.write_parquet(OUT_DIR / "scores_all.parquet")

# %% [markdown]
# ## Derive cell-level (fold-mean) summary

# %%
CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end", "window_id",
             "phon_smin", "phon_smax", "behav_smin", "behav_smax",
             "is_fallback", "ci_excludes_zero"]

if scores_all.height > 0:
    cell_means = (
        scores_all
        .group_by(CELL_KEYS)
        .agg(
            pl.col("phon_roc_auc").mean().alias("phon_auc_mean"),
            pl.col("behav_roc_auc").mean().alias("behav_auc_mean"),
            pl.col("phon_roc_auc").std().alias("phon_auc_std"),
            pl.col("behav_roc_auc").std().alias("behav_auc_std"),
        )
        .with_columns(
            (pl.col("phon_auc_mean") - pl.col("behav_auc_mean")).alias("transfer_drop"),
            ((pl.col("phon_smin") + pl.col("phon_smax")) / 2 / epoch_sfreq + epoch_tmin)
                .alias("phon_tcenter"),
            ((pl.col("behav_smin") + pl.col("behav_smax")) / 2 / epoch_sfreq + epoch_tmin)
                .alias("behav_tcenter"),
        )
    )
    print(f"Cell-level summary: {cell_means.height} rows")
    print(cell_means.select([
        "subject", "electrode_idx", "phoneme_pair", "word_end", "window_id",
        "phon_auc_mean", "behav_auc_mean", "transfer_drop",
        "is_fallback", "ci_excludes_zero",
    ]).sort("transfer_drop").head(10))
else:
    cell_means = pl.DataFrame()

# %% [markdown]
# ## Summary plots

# %%
if cell_means.height == 0:
    print("No data — skipping plots.")
else:
    phon = cell_means["phon_auc_mean"].to_numpy()
    behav = cell_means["behav_auc_mean"].to_numpy()
    drop = cell_means["transfer_drop"].to_numpy()
    ci_excl = cell_means["ci_excludes_zero"].to_numpy()
    fallback = cell_means["is_fallback"].to_numpy()
    pps = cell_means["phoneme_pair"].to_list()

    sig_mask = ci_excl & ~fallback
    fallback_mask = fallback
    insig_mask = ~ci_excl & ~fallback

    # --- Panel layout ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Acoustic transfer: phonemic peak → behavioral window", fontsize=10)

    # Panel 1: paired scatter phon vs. behav AUC
    ax = axes[0]
    _lims = [min(phon.min(), behav.min()) - 0.02,
             max(phon.max(), behav.max()) + 0.02]
    for mask, label, color, marker, zorder in [
        (sig_mask,     "CI excl. 0",   "#2166ac", "o", 3),
        (insig_mask,   "CI ∩ 0",       "#92c5de", "o", 2),
        (fallback_mask,"fallback",      "#d1e5f0", "^", 1),
    ]:
        if mask.any():
            ax.scatter(phon[mask], behav[mask], c=color, marker=marker,
                       s=30, alpha=0.8, label=label, zorder=zorder)
    ax.plot(_lims, _lims, "k--", lw=0.8, alpha=0.5, label="unity")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.axvline(0.5, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Phon peak AUC (fold mean)")
    ax.set_ylabel("Behav window AUC (fold mean)")
    ax.set_xlim(_lims)
    ax.set_ylim(_lims)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Phon vs. behav AUC per cell")

    # Panel 2: transfer drop histogram stratified by significance
    ax = axes[1]
    bins = np.linspace(drop.min() - 0.02, drop.max() + 0.02, 30)
    for mask, label, color, alpha in [
        (sig_mask,     "CI excl. 0",  "#2166ac", 0.75),
        (insig_mask,   "CI ∩ 0",      "#92c5de", 0.6),
        (fallback_mask,"fallback",     "#d1e5f0", 0.5),
    ]:
        if mask.any():
            ax.hist(drop[mask], bins=bins, color=color, alpha=alpha, label=label)
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("Transfer drop  (phon AUC − behav AUC)")
    ax.set_ylabel("Cells")
    ax.legend(fontsize=7)
    ax.set_title("Transfer drop distribution")

    # Panel 3: transfer drop by phoneme pair (violin / strip)
    ax = axes[2]
    unique_pps = sorted(set(pps))
    pp_drops = [drop[np.array(pps) == pp] for pp in unique_pps]
    vp = ax.violinplot(pp_drops, positions=range(len(unique_pps)),
                       showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.55)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax.set_xticks(range(len(unique_pps)))
    ax.set_xticklabels(unique_pps, fontsize=8)
    ax.set_ylabel("Transfer drop  (phon − behav AUC)")
    ax.set_title("Transfer drop by phoneme pair")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "transfer_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved transfer_summary.pdf")

    # ── Panel 4 (separate fig): window timing scatter ──
    phon_t = cell_means["phon_tcenter"].to_numpy()
    behav_t = cell_means["behav_tcenter"].to_numpy()

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for mask, label, color, marker in [
        (sig_mask,     "CI excl. 0",  "#2166ac", "o"),
        (insig_mask,   "CI ∩ 0",      "#92c5de", "o"),
        (fallback_mask,"fallback",     "#d1e5f0", "^"),
    ]:
        if mask.any():
            sc = ax2.scatter(phon_t[mask], behav_t[mask],
                             c=drop[mask], cmap="RdBu_r",
                             vmin=-0.3, vmax=0.3,
                             marker=marker, s=35, alpha=0.85, label=label)
    plt.colorbar(sc, ax=ax2, label="transfer drop")
    ax2.set_xlabel("Phon peak window center (s)")
    ax2.set_ylabel("Behav window center (s)")
    ax2.legend(fontsize=7)
    ax2.set_title("Window timing: phon vs. behav\n(color = transfer drop)")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "transfer_timing.pdf", bbox_inches="tight")
    plt.close(fig2)
    print("Saved transfer_timing.pdf")
