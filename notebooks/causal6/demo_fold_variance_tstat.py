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
# # Demo: does a t-like statistic beat fold-mean for null-standardization?
#
# Question: we have 5 fold AUCs per (site, window) in real and per
# (site, window, permutation) in null. Currently we aggregate to
# `fold_mean` before doing significance testing. This throws away
# fold-variance information. In principle, real signal should be
# **consistent across folds** (low fold-std) while null permutations
# often capture transient features that look different per fold
# (higher fold-std).
#
# This notebook empirically checks that intuition on the acoustic
# decoder outputs, and compares pointwise permutation p-values
# computed under two statistics:
#
#   1. `fold_mean` — the current choice
#   2. `t_stat = (fold_mean - 0.5) / (fold_std / sqrt(n_folds))` —
#      the proposed variance-normalized choice
#
# If the assumption holds, the t-stat will give smaller pointwise p
# than fold_mean at sites with real signal, i.e. more effective
# separation between real and null.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# %% tags=["parameters"]
subject = "EC278"
scores_path = f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet"
null_scores_path = f"outputs/causal6/acoustic_decoding_null/{subject}/null_scores.parquet"

target = "categorical_acoustic_cue"
# Restrict to the same window range as the acoustic peaks rule uses;
# change to (0, 290) to see the full searchlight.
peak_search_smin = 50
peak_search_smax = 75

# Small floor on fold_std so the t-stat doesn't blow up at sites with
# near-zero fold variance.
fold_std_floor = 0.01

# Centering value for the t-stat (AUC chance level; changes the sign
# of t but not its rank structure under permutation test).
center = 0.5

outdir = "."

# %%
outdir = Path(outdir)

# %% [markdown]
# ## Load data

# %%
site_keys = ["subject", "electrode_idx", "phoneme_pair"]
window_keys = site_keys + ["smin", "smax"]

real = (
    pl.read_parquet(scores_path)
    .filter(pl.col("target") == target)
    .filter(
        (pl.col("smin") >= peak_search_smin) & (pl.col("smax") <= peak_search_smax)
    )
)
null = (
    pl.read_parquet(null_scores_path)
    .filter(pl.col("target") == target)
    .filter(
        (pl.col("smin") >= peak_search_smin) & (pl.col("smax") <= peak_search_smax)
    )
)
print(f"real rows: {real.height:,}  null rows: {null.height:,}")
print(f"sites × windows × folds: {real.height}")
print(f"nulls = {null.height / real.height:.1f}× real rows (~K permutations)")

# %% [markdown]
# ## Aggregate to per-(site, window) fold-mean + fold-std
#
# Real: one row per (site, window). Null: one row per (site, window, perm).

# %%
def _tstat(fold_mean: pl.Expr, fold_std: pl.Expr, n_folds: pl.Expr) -> pl.Expr:
    """t-like statistic: signed distance from `center` in fold-SEM units."""
    sem = (pl.max_horizontal(fold_std, pl.lit(fold_std_floor))
           / n_folds.cast(pl.Float64).sqrt())
    return (fold_mean - center) / sem


real_agg = real.group_by(window_keys).agg(
    pl.col("test_roc_auc").mean().alias("fold_mean"),
    pl.col("test_roc_auc").std().alias("fold_std"),
    pl.col("test_roc_auc").len().alias("n_folds"),
).with_columns(
    _tstat(pl.col("fold_mean"), pl.col("fold_std"), pl.col("n_folds")).alias("t_stat")
)
null_agg = null.group_by(window_keys + ["permutation_idx"]).agg(
    pl.col("test_roc_auc").mean().alias("fold_mean"),
    pl.col("test_roc_auc").std().alias("fold_std"),
    pl.col("test_roc_auc").len().alias("n_folds"),
).with_columns(
    _tstat(pl.col("fold_mean"), pl.col("fold_std"), pl.col("n_folds")).alias("t_stat")
)

print(f"real_agg: {real_agg.height} (site, window) rows")
print(f"null_agg: {null_agg.height} (site, window, perm) rows")

# %% [markdown]
# ## Q1: do nulls systematically have higher fold-variance than real?
#
# Overlay the marginal histograms of `fold_std` across all (site,
# window) in real and all (site, window, perm) in null. If the
# variance-exploitation assumption holds, the null distribution
# should sit to the right of the real distribution.

# %%
real_std = real_agg["fold_std"].to_numpy()
null_std = null_agg["fold_std"].to_numpy()

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, max(real_std.max(), null_std.max()), 60)
ax.hist(real_std, bins=bins, density=True, alpha=0.55, label=f"real (n={len(real_std)})")
ax.hist(null_std, bins=bins, density=True, alpha=0.55, label=f"null (n={len(null_std):,})")
ax.axvline(np.median(real_std), color="C0", ls="--", alpha=0.8,
           label=f"real median = {np.median(real_std):.3f}")
ax.axvline(np.median(null_std), color="C1", ls="--", alpha=0.8,
           label=f"null median = {np.median(null_std):.3f}")
ax.set_xlabel("fold-AUC std across 5 folds")
ax.set_ylabel("density")
ax.set_title(f"{subject} acoustic: fold-std distributions")
ax.legend()
fig.tight_layout()
fig.savefig(outdir / "q1_fold_std_distributions.png", dpi=120)
plt.show()

print(f"real fold_std:  mean={real_std.mean():.3f}  median={np.median(real_std):.3f}")
print(f"null fold_std:  mean={null_std.mean():.3f}  median={np.median(null_std):.3f}")
print(f"ratio of medians (null/real): {np.median(null_std) / np.median(real_std):.2f}")

# %% [markdown]
# ### Controlled comparison: fold-std at matched fold-mean
#
# The marginal above mixes sites with very different signal levels. A
# cleaner test: within each (site, window), compare the real fold_std
# to the null fold_std distribution at that SAME (site, window).

# %%
paired = real_agg.select(window_keys + ["fold_mean", "fold_std", "t_stat"]).rename({
    "fold_mean": "real_fold_mean",
    "fold_std":  "real_fold_std",
    "t_stat":    "real_t_stat",
}).join(
    null_agg.group_by(window_keys).agg(
        pl.col("fold_std").mean().alias("null_fold_std_mean"),
        pl.col("fold_std").median().alias("null_fold_std_median"),
        pl.col("fold_mean").mean().alias("null_fold_mean_mean"),
    ),
    on=window_keys, how="inner",
)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    paired["real_fold_std"].to_numpy(),
    paired["null_fold_std_median"].to_numpy(),
    s=5, alpha=0.3,
)
lim = max(paired["real_fold_std"].max(), paired["null_fold_std_median"].max())
ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y = x")
ax.set_xlabel("real fold_std (one point per site × window)")
ax.set_ylabel("null fold_std median (same site × window)")
ax.set_title(f"{subject}: per-(site, window) fold_std — real vs null median")
ax.legend()
fig.tight_layout()
fig.savefig(outdir / "q1b_paired_fold_std.png", dpi=120)
plt.show()

frac_null_higher = (
    paired["null_fold_std_median"] > paired["real_fold_std"]
).mean()
print(f"Fraction of (site, window) where null median fold_std > real fold_std: {frac_null_higher:.3f}")
print("If the assumption holds, this should be well above 0.5.")

# %% [markdown]
# ## Q2: does the t-stat give a different (better?) pointwise p?
#
# For each (site, window), compute the empirical pointwise p under
# both statistics:
#   p_mean  = (#{null_mean  >= real_mean}  + 1) / (K + 1)
#   p_tstat = (#{null_tstat >= real_tstat} + 1) / (K + 1)
#
# Then compare.

# %%
K = int(null_agg.group_by(window_keys).len()["len"].min())
print(f"K (min permutations per (site, window)) = {K}")


# Join real onto null and aggregate-count (explicit, no list.eval closure gymnastics).
merged_mean = real_agg.select(window_keys + ["fold_mean"]).rename({"fold_mean": "real_mean"}).join(
    null_agg.select(window_keys + ["fold_mean"]).rename({"fold_mean": "null_mean"}),
    on=window_keys, how="inner",
)
p_mean = merged_mean.group_by(window_keys + ["real_mean"]).agg(
    (pl.col("null_mean") >= pl.col("real_mean")).sum().alias("_ge"),
    pl.len().alias("_n"),
).with_columns(
    ((pl.col("_ge") + 1) / (pl.col("_n") + 1)).alias("p_mean"),
).drop("_ge", "_n", "real_mean")

merged_t = real_agg.select(window_keys + ["t_stat"]).rename({"t_stat": "real_t"}).join(
    null_agg.select(window_keys + ["t_stat"]).rename({"t_stat": "null_t"}),
    on=window_keys, how="inner",
)
p_tstat = merged_t.group_by(window_keys + ["real_t"]).agg(
    (pl.col("null_t") >= pl.col("real_t")).sum().alias("_ge"),
    pl.len().alias("_n"),
).with_columns(
    ((pl.col("_ge") + 1) / (pl.col("_n") + 1)).alias("p_tstat"),
).drop("_ge", "_n", "real_t")

compare = (
    real_agg.select(window_keys + ["fold_mean", "fold_std", "t_stat"])
    .join(p_mean, on=window_keys, how="inner")
    .join(p_tstat, on=window_keys, how="inner")
)

# %%
p_m = compare["p_mean"].to_numpy()
p_t = compare["p_tstat"].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(p_m, p_t, s=5, alpha=0.3)
ax.plot([1 / (K + 1), 1], [1 / (K + 1), 1], "k--", alpha=0.5, label="y = x")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("pointwise p (fold_mean)")
ax.set_ylabel("pointwise p (t_stat)")
ax.set_title("per-(site, window): does t_stat give smaller p?")
ax.legend()

ax = axes[1]
# Per-site minimum pointwise p (peak window candidate)
per_site_min = (
    compare.group_by(site_keys)
    .agg(pl.col("p_mean").min().alias("best_p_mean"),
         pl.col("p_tstat").min().alias("best_p_tstat"))
)
bpm = per_site_min["best_p_mean"].to_numpy()
bpt = per_site_min["best_p_tstat"].to_numpy()
ax.scatter(bpm, bpt, s=15, alpha=0.6)
lim_lo = 1 / (K + 1) * 0.8
ax.plot([lim_lo, 1], [lim_lo, 1], "k--", alpha=0.5, label="y = x")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("best pointwise p (fold_mean)")
ax.set_ylabel("best pointwise p (t_stat)")
ax.set_title("per-site best window: does t_stat do better?")
ax.legend()

fig.tight_layout()
fig.savefig(outdir / "q2_pointwise_p_comparison.png", dpi=120)
plt.show()

frac_t_better = (p_t < p_m).mean()
frac_t_much_better = (p_t < 0.5 * p_m).mean()
print(f"Fraction of (site, window) where p_tstat < p_mean:       {frac_t_better:.3f}")
print(f"Fraction of (site, window) where p_tstat < 0.5 * p_mean: {frac_t_much_better:.3f}")
print()
print("Per-site best window:")
print(f"  p_tstat best < p_mean best at {(bpt < bpm).mean():.3f} of sites")
print(f"  median best p_mean:  {np.median(bpm):.4f}")
print(f"  median best p_tstat: {np.median(bpt):.4f}")

# %% [markdown]
# ## Q3: close-up on strong sites
#
# Pick the top few sites by real fold_mean and look at where the real
# point sits inside the null cloud in (fold_mean, fold_std) space.
# If the assumption holds, the real point should sit down-and-right
# of the null cloud (higher mean, lower std).

# %%
top_sites = (
    real_agg.sort("fold_mean", descending=True)
    .head(6)
    .select(site_keys + ["smin", "smax", "fold_mean", "fold_std"])
)
print(top_sites)

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=False, sharey=False)
for ax, row in zip(axes.ravel(), top_sites.iter_rows(named=True)):
    flt = (
        (pl.col("electrode_idx") == row["electrode_idx"])
        & (pl.col("phoneme_pair") == row["phoneme_pair"])
        & (pl.col("smin") == row["smin"]) & (pl.col("smax") == row["smax"])
    )
    null_cloud = null_agg.filter(flt).select(["fold_mean", "fold_std"]).to_numpy()
    ax.scatter(null_cloud[:, 0], null_cloud[:, 1], s=5, alpha=0.3, label="null")
    ax.scatter([row["fold_mean"]], [row["fold_std"]], s=80, c="red", marker="*",
               label="real", zorder=5)
    ax.axvline(row["fold_mean"], ls=":", c="red", alpha=0.4)
    ax.set_title(
        f"e{row['electrode_idx']} {row['phoneme_pair']} "
        f"[{row['smin']},{row['smax']}]\nreal: μ={row['fold_mean']:.3f}, σ={row['fold_std']:.3f}",
        fontsize=9,
    )
    ax.set_xlabel("fold_mean"); ax.set_ylabel("fold_std")
    ax.legend(fontsize=8)

fig.suptitle(f"{subject}: top-6 sites by real fold_mean — real (★) vs null cloud (·)")
fig.tight_layout()
fig.savefig(outdir / "q3_top_sites_cloud.png", dpi=120)
plt.show()

# %% [markdown]
# ## Headline takeaways
#
# Read the three output plots + printed stats:
#   - **Q1** (`q1_fold_std_distributions.png`, `q1b_paired_fold_std.png`):
#     is the null's fold-std systematically higher?
#   - **Q2** (`q2_pointwise_p_comparison.png`): does the t-stat
#     produce smaller pointwise p-values — and by how much at saturated
#     sites where current K=500 hits the floor?
#   - **Q3** (`q3_top_sites_cloud.png`): at the signal-bearing sites,
#     does the real point visibly sit at lower std than the null cloud
#     at its own fold_mean?
#
# Decision rule: if Q2 shows `p_tstat < p_mean` for say 70%+ of
# (site, window) pairs, and median best per-site `p_tstat` is
# noticeably below `p_mean`, the t-stat is worth adopting. If the
# improvement is mild or none, the fold variance structure isn't
# carrying much extra signal and we should stick with fold_mean.
