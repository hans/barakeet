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
# # Acoustic gradient figures — causal46_joined population
#
# Population-level visualization of:
#   - AX discrimination: adjacent-step decoders at each site's peak acoustic window
#   - Sigmoid neurometric fits: steepness k distinguishes categorical from graded
#
# Inputs (aggregate parquets from upstream rules):
#   - trial_df_all.parquet            — per-trial hga_norm at each site's peak window
#   - model_comparison_df_all.parquet — per-site sigmoid fit params
#   - ax_discrimination_df_all.parquet — per-site adjacent-step AUCs (step_a, step_b, roc_auc)
#   - phon_peaks_all.parquet          — peak acoustic window + phon_roc_auc per site
#
# All figures written to outdir/. Figures that require behavior labels (Fig 3
# behavior-agreement) or epoch reload (Fig 6 timecourse) are out of scope here.

# %%
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from src.models.sigmoid import (
    EFFECTIVELY_LINEAR_K,
    sigmoid_model_2p,
)

# %% tags=["parameters"]
trial_df_path            = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet"
model_comparison_df_path = "outputs/causal46_joined/acoustic_univariate_gradient/model_comparison_df_all.parquet"
ax_discrimination_path   = "outputs/causal46_joined/acoustic_ax_discrimination/ax_discrimination_df_all.parquet"
phon_peaks_path          = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
outdir                   = "outputs/causal46_joined/acoustic_gradient_figures"
n_sample                 = 24  # sites for catplot galleries

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load data

# %%
trial_df   = pd.read_parquet(trial_df_path)
model_df   = pd.read_parquet(model_comparison_df_path)
ax_df      = pd.read_parquet(ax_discrimination_path)

# Keep one row per (subject, electrode_idx, phoneme_pair) from phon_peaks: highest AUC
peaks_all  = (
    pd.read_parquet(phon_peaks_path)[
        ["subject", "electrode_idx", "phoneme_pair", "phon_roc_auc"]
    ]
    .sort_values("phon_roc_auc", ascending=False)
    .drop_duplicates(subset=["subject", "electrode_idx", "phoneme_pair"])
)

SITE_KEY = ["subject", "electrode_idx", "phoneme_pair"]

trial_df = trial_df.merge(peaks_all, on=SITE_KEY, how="left")
model_df = model_df.merge(peaks_all, on=SITE_KEY, how="left")

print(f"trial_df:  {len(trial_df)} trial rows, "
      f"{trial_df[SITE_KEY].drop_duplicates().shape[0]} sites")
print(f"model_df:  {len(model_df)} sites")
print(f"ax_df:     {len(ax_df)} step-pair rows, "
      f"{ax_df[SITE_KEY].drop_duplicates().shape[0]} sites")

# %%
steps_all   = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
step_pairs  = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

step_cmap   = plt.cm.RdBu_r
step_colors = {int(s): step_cmap(i / 5) for i, s in enumerate(range(1, 7))}

rng = np.random.default_rng(0)

# %% [markdown]
# ## Section 1 — HGA confidence

# %%
trial_df["confidence"] = (trial_df["hga_norm"] - 0.5).abs()

site_stats = (
    trial_df
    .groupby(SITE_KEY)
    .apply(lambda g: pd.Series({
        "mean_endpoint_confidence": g.loc[g["resampled"].isin([1, 6]), "confidence"].mean(),
        "mean_ambig_confidence":    g.loc[g["resampled"].isin([2, 3, 4, 5]), "confidence"].mean(),
        "phon_roc_auc":             g["phon_roc_auc"].iloc[0],
    }), include_groups=False)
    .reset_index()
    .dropna(subset=["mean_endpoint_confidence", "mean_ambig_confidence"])
)
print(f"site_stats: {len(site_stats)} sites")

# %% [markdown]
# ### Fig 1 — HGA confidence by morph step

# %%
fig, ax = plt.subplots(figsize=(7, 4))

conf_by_step = [
    trial_df.loc[trial_df["resampled"] == s, "confidence"].dropna().values
    for s in range(1, 7)
]

bp = ax.boxplot(
    conf_by_step,
    positions=list(range(1, 7)),
    widths=0.6,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
)
for patch, step in zip(bp["boxes"], range(1, 7)):
    patch.set_facecolor("steelblue" if step in (1, 6) else "lightyellow")
    patch.set_edgecolor("black")

ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="midpoint (hga_norm=0.5)")
ax.set_xlabel("Morph step (1 = clear /d/ endpoint, 6 = clear /n/ endpoint)")
ax.set_ylabel("HGA confidence  |hga_norm − 0.5|")
ax.set_title("Acoustic HGA confidence by morph step\n(across all manifest sites)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(outdir / "confidence_by_step.pdf")
plt.close(fig)
print("Saved confidence_by_step.pdf")

# %% [markdown]
# ### Fig 2 — Endpoint vs. ambiguous confidence per site

# %%
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(
    site_stats["mean_endpoint_confidence"],
    site_stats["mean_ambig_confidence"],
    c=site_stats["phon_roc_auc"],
    cmap="viridis",
    alpha=0.7,
    edgecolors="k",
    linewidths=0.5,
)
lim = [0, site_stats[["mean_endpoint_confidence", "mean_ambig_confidence"]].values.max() * 1.05]
ax.plot(lim, lim, "k--", linewidth=1, label="equal confidence")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Mean confidence on endpoints (steps 1 & 6)")
ax.set_ylabel("Mean confidence on ambiguous trials (steps 2–5)")
ax.set_title("Representational commitment:\nendpoints vs. ambiguous steps")
ax.legend(fontsize=9)
plt.colorbar(sc, ax=ax, label="phon_roc_auc")
plt.tight_layout()
fig.savefig(outdir / "confidence_scatter.pdf")
plt.close(fig)
print("Saved confidence_scatter.pdf")

# %% [markdown]
# ## Section 2 — Population AX discrimination curve

# %%
fig, ax = plt.subplots(figsize=(6, 4))

pair_labels    = [f"{a}v{b}" for a, b in step_pairs]
pair_midpoints = [(a + b) / 2.0 for a, b in step_pairs]

mean_aucs, sem_aucs = [], []
for step_a, step_b in step_pairs:
    vals = ax_df.loc[(ax_df["step_a"] == step_a) & (ax_df["step_b"] == step_b), "roc_auc"]
    mean_aucs.append(vals.mean())
    sem_aucs.append(vals.std() / np.sqrt(len(vals)))

mean_aucs = np.array(mean_aucs)
sem_aucs  = np.array(sem_aucs)

ax.plot(pair_midpoints, mean_aucs, "D-", color="green", linewidth=1.5, markersize=6, zorder=5)
ax.fill_between(pair_midpoints, mean_aucs - sem_aucs, mean_aucs + sem_aucs,
                color="green", alpha=0.2)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
ax.set_xticks(pair_midpoints)
ax.set_xticklabels(pair_labels)
ax.set_xlabel("Adjacent step pair")
ax.set_ylabel("Mean AX discrimination AUC")
ax.set_title("Population AX discrimination: can single electrodes\n"
             "distinguish adjacent morph steps?")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(outdir / "ax_discrimination_population.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved ax_discrimination_population.pdf")

# %% [markdown]
# ## Section 3 — Sample catplots with AX overlay
#
# 24 sites sampled evenly across the phon_roc_auc range.
# Scatter colored by morph step (behavior label not available in trial_df_all).

# %%
_mc_sorted = (
    model_df.dropna(subset=["phon_roc_auc"])
    .sort_values("phon_roc_auc")
    .reset_index(drop=True)
)
sample_idx   = np.round(np.linspace(0, len(_mc_sorted) - 1, n_sample)).astype(int)
sample_sites = _mc_sorted.iloc[sample_idx].copy().reset_index(drop=True)
sample_sites["site_label"] = sample_sites.apply(
    lambda r: f"{r['subject']} e{int(r['electrode_idx'])} {r['phoneme_pair']}", axis=1
)
label_map = {
    (r["subject"], r["electrode_idx"], r["phoneme_pair"]): r["site_label"]
    for _, r in sample_sites.iterrows()
}
sample_trials = trial_df[
    trial_df.apply(lambda r: (r["subject"], r["electrode_idx"], r["phoneme_pair"]) in label_map, axis=1)
].copy()
sample_trials["site_label"] = sample_trials.apply(
    lambda r: label_map[(r["subject"], r["electrode_idx"], r["phoneme_pair"])], axis=1
)

# %%
n_cols = 4
n_rows = int(np.ceil(n_sample / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
axes_flat = axes.flatten()

for ax_idx, (_, site_row) in enumerate(sample_sites.iterrows()):
    ax = axes_flat[ax_idx]
    site_data = sample_trials[sample_trials["site_label"] == site_row["site_label"]].dropna(
        subset=["resampled", "hga_norm"]
    )

    # Per-step scatter colored by step
    for step in range(1, 7):
        mask = site_data["resampled"] == step
        xvals = step + rng.uniform(-0.2, 0.2, mask.sum())
        ax.scatter(xvals, site_data.loc[mask, "hga_norm"],
                   c=[step_colors[step]], alpha=0.3, s=8, linewidths=0)

    # Mean neurometric line
    means = site_data.groupby("resampled")["hga_norm"].mean().sort_index()
    ax.plot(means.index, means.values, "k-o", linewidth=1.5, markersize=4, zorder=5)

    # AX discrimination on secondary y-axis
    site_ax = ax_df[
        (ax_df["subject"] == site_row["subject"])
        & (ax_df["electrode_idx"] == site_row["electrode_idx"])
        & (ax_df["phoneme_pair"] == site_row["phoneme_pair"])
    ]
    if len(site_ax) > 0:
        ax2 = ax.twinx()
        midpoints = (site_ax["step_a"].values + site_ax["step_b"].values) / 2.0
        ax2.plot(midpoints, site_ax["roc_auc"].values, "D--",
                 color="green", linewidth=1.2, markersize=4, alpha=0.8, zorder=6)
        ax2.set_ylim(0.4, 1.0)
        ax2.tick_params(axis="y", labelcolor="green", labelsize=6)
        if ax_idx % n_cols == n_cols - 1:
            ax2.set_ylabel("AX discrim. AUC", color="green", fontsize=7)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.5, 6.5)
    ax.set_title(f"{site_row['site_label']}\nAUC={site_row['phon_roc_auc']:.2f}", fontsize=7.5)
    ax.set_xlabel("Morph step")
    if ax_idx % n_cols == 0:
        ax.set_ylabel("Normalized HGA")

step_handles = [
    plt.Line2D([0], [0], color=step_colors[s], linewidth=2, label=f"step {s}")
    for s in range(1, 7)
]
step_handles.append(
    plt.Line2D([0], [0], color="green", linewidth=1.2, linestyle="--",
               marker="D", markersize=4, label="AX discrim. AUC")
)
fig.legend(handles=step_handles, loc="lower right", fontsize=8, ncol=4)
fig.suptitle("Neurometric function at individual acoustic sites (sorted by AUC, low→high)\n"
             "colored by morph step (percept label not available)", fontsize=11)
plt.tight_layout()
fig.savefig(outdir / "catplots_sample.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved catplots_sample.pdf")

# %% [markdown]
# ## Section 4 — Per-site gallery: neurometric + AX discrimination
#
# One page per (subject × electrode × phoneme_pair) site.
# Top panel: hga_norm scatter + mean + sigmoid overlay.
# Bottom panel: AX discrimination AUC per step pair with std bars.
# Sorted by phoneme_pair, then phon_roc_auc descending.

# %%
# Build a per-site index from model_df (has sigmoid params + phon_roc_auc)
gallery_sites = (
    model_df.dropna(subset=["phon_roc_auc"])
    .sort_values(["phoneme_pair", "phon_roc_auc"], ascending=[True, False])
    .reset_index(drop=True)
)

# Pre-index trial_df and ax_df by site key for fast lookup
_td_idx  = trial_df.set_index(SITE_KEY)
_ax_idx  = ax_df.set_index(SITE_KEY)

print(f"Per-site gallery: {len(gallery_sites)} pages")

# %%
x_fine = np.linspace(1, 6, 200)

with PdfPages(outdir / "ax_per_site_gallery.pdf") as pdf:
    for _, site_row in gallery_sites.iterrows():
        sub = site_row["subject"]
        ei  = int(site_row["electrode_idx"])
        pp  = site_row["phoneme_pair"]
        key = (sub, ei, pp)

        # --- trial data for this site ---
        try:
            site_trials = _td_idx.loc[[key]].reset_index().dropna(subset=["resampled", "hga_norm"])
        except KeyError:
            site_trials = pd.DataFrame()

        # --- AX data for this site ---
        try:
            site_ax = _ax_idx.loc[[key]].reset_index().sort_values("step_a")
        except KeyError:
            site_ax = pd.DataFrame()

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
        fig.subplots_adjust(hspace=0.4)

        # ---------- Top panel: neurometric + sigmoid ----------
        if len(site_trials) > 0:
            for step in range(1, 7):
                mask = site_trials["resampled"] == step
                xvals = step + rng.uniform(-0.2, 0.2, mask.sum())
                ax_top.scatter(xvals, site_trials.loc[mask, "hga_norm"],
                               c=[step_colors[step]], alpha=0.3, s=8, linewidths=0)

            means = site_trials.groupby("resampled")["hga_norm"].mean().sort_index()
            ax_top.plot(means.index, means.values, "k-o", linewidth=1.5, markersize=4, zorder=5)

        # sigmoid overlay
        sig_label = ""
        x0, k, r2 = site_row.get("sigmoid_x0"), site_row.get("sigmoid_k"), site_row.get("sigmoid_r2")
        if x0 is not None and k is not None and not (np.isnan(x0) or np.isnan(k)):
            ax_top.plot(x_fine, sigmoid_model_2p(x_fine, x0, k),
                        color="tomato", linewidth=2.0, zorder=4, label="sigmoid fit")
            sig_label = f"  PSE={x0:.2f}  k={k:.2f}  R²={r2:.2f}"
            ax_top.legend(fontsize=7, loc="upper left")

        ax_top.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax_top.set_xticks([1, 2, 3, 4, 5, 6])
        ax_top.set_xlim(0.5, 6.5)
        ax_top.set_ylabel("Normalized HGA")
        ax_top.set_xlabel("Morph step")
        ax_top.set_title(
            f"Neurometric function\n{sig_label}",
            fontsize=9,
        )

        # ---------- Bottom panel: AX discrimination ----------
        if len(site_ax) > 0:
            midpoints = (site_ax["step_a"].values + site_ax["step_b"].values) / 2.0
            ax_bot.errorbar(
                midpoints, site_ax["roc_auc"].values,
                yerr=site_ax["roc_auc_std"].values,
                fmt="D-", color="green", linewidth=1.5, markersize=5, capsize=3,
            )
            ax_bot.set_xticks(midpoints)
            ax_bot.set_xticklabels([f"{int(a)}v{int(b)}" for a, b in
                                    zip(site_ax["step_a"], site_ax["step_b"])])

        ax_bot.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
        ax_bot.set_xlim(1.0, 6.0)
        ax_bot.set_ylim(0.35, 1.05)
        ax_bot.set_ylabel("AX discrimination AUC")
        ax_bot.set_xlabel("Step pair")
        ax_bot.set_title("Adjacent-step discrimination", fontsize=9)
        ax_bot.legend(fontsize=7)

        fig.suptitle(
            f"{sub} / elec {ei} / {pp}  |  phon_roc_auc={site_row['phon_roc_auc']:.3f}",
            fontsize=10, fontweight="bold",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("Saved ax_per_site_gallery.pdf")

# %% [markdown]
# ## Section 5 — Sigmoid figures

# %% [markdown]
# ### Sigmoid Fig 1 — Ideal shapes at different steepness values

# %%
fig, ax = plt.subplots(figsize=(6, 4))

k_examples = [0.1, 0.2, 0.5, 1.0, 2.0]
cmap_sig = plt.cm.Reds
for i, k_val in enumerate(k_examples):
    color = cmap_sig(0.4 + 0.5 * i / (len(k_examples) - 1))
    ax.plot(x_fine, 1.0 / (1.0 + np.exp(-(x_fine - 3.5) / k_val)),
            color=color, linewidth=2.0, label=f"k={k_val}")

ax.set_xlabel("Morph step")
ax.set_ylabel("Normalized model prediction")
ax.set_title("Sigmoid shapes: small k = categorical, large k ≈ linear")
ax.set_xticks(range(1, 7))
ax.set_xlim(0.5, 6.5)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8, title="Steepness k")
plt.tight_layout()
fig.savefig(outdir / "ideal_model_shapes.pdf")
plt.close(fig)
print("Saved ideal_model_shapes.pdf")

# %% [markdown]
# ### Sigmoid Fig 2 — Fitted parameter distributions

# %%
PSE_RANGE = (2.0, 5.0)
valid_all = model_df.dropna(subset=["sigmoid_k", "sigmoid_r2"])
valid     = valid_all[valid_all["sigmoid_x0"].between(*PSE_RANGE)]
print(f"PSE filter: {len(valid)}/{len(valid_all)} sites "
      f"(excluded {len(valid_all) - len(valid)} with x0 outside {PSE_RANGE})")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

k_thresh     = EFFECTIVELY_LINEAR_K
k_sigmoidal  = valid.loc[valid["sigmoid_k"] <= k_thresh, "sigmoid_k"]
n_linear     = (valid["sigmoid_k"] > k_thresh).sum()
bins_sig     = np.linspace(0, k_thresh, 20)

ax = axes[0]
ax.hist(k_sigmoidal, bins=bins_sig, color="tomato", edgecolor="k", alpha=0.8)
if n_linear > 0:
    bar_w = bins_sig[1] - bins_sig[0]
    ax.bar(k_thresh + bar_w / 2, n_linear, width=bar_w,
           color="gray", edgecolor="k", alpha=0.8,
           label=f"linear (k>{k_thresh:.0f}, n={n_linear})")
    ax.legend(fontsize=8)
ax.set_xlabel("Fitted sigmoid k (steepness)")
ax.set_ylabel("Count")
ax.set_title(f"Steepness distribution (n={len(valid)})\nsmall k = categorical")

ax = axes[1]
ax.hist(valid["sigmoid_x0"], bins=25, color="mediumpurple", edgecolor="k", alpha=0.8)
ax.axvline(3.5, color="gray", linestyle="--", linewidth=1, label="midpoint (3.5)")
ax.set_xlabel("Fitted PSE (x0)")
ax.set_ylabel("Count")
ax.set_title("PSE distribution")
ax.legend(fontsize=8)

ax = axes[2]
ax.hist(valid["sigmoid_r2"], bins=25, color="steelblue", edgecolor="k", alpha=0.8)
med_r2 = valid["sigmoid_r2"].median()
ax.axvline(med_r2, color="red", linestyle="--", label=f"median={med_r2:.2f}")
ax.set_xlabel("Sigmoid R²")
ax.set_ylabel("Count")
ax.set_title("Goodness of fit")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(outdir / "sigmoid_parameter_distributions.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved sigmoid_parameter_distributions.pdf")

# %% [markdown]
# ### Sigmoid Fig 2b — PSE distribution by subject and phoneme pair

# %%
_valid_pse = model_df.dropna(subset=["sigmoid_x0"]).copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
sns.boxplot(data=_valid_pse, x="subject", y="sigmoid_x0",
            color="lightgray", fliersize=0, ax=ax)
sns.stripplot(data=_valid_pse, x="subject", y="sigmoid_x0", hue="phoneme_pair",
              dodge=True, alpha=0.7, size=5, ax=ax)
ax.axhline(3.5, color="gray", linestyle="--", linewidth=1, label="midpoint (3.5)")
ax.set_xlabel("Subject")
ax.set_ylabel("Fitted PSE (x0)")
ax.set_title("PSE distribution by subject")
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=7, title="Phoneme pair", loc="upper right")

ax = axes[1]
sns.violinplot(data=_valid_pse, x="phoneme_pair", y="sigmoid_x0",
               color="lightgray", inner=None, alpha=0.4, ax=ax)
sns.stripplot(data=_valid_pse, x="phoneme_pair", y="sigmoid_x0", hue="subject",
              dodge=True, alpha=0.7, size=4, ax=ax)
ax.axhline(3.5, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Phoneme pair")
ax.set_ylabel("Fitted PSE (x0)")
ax.set_title("PSE distribution by phoneme pair")
ax.legend(fontsize=6, title="Subject", loc="upper right", ncol=2)

fig.suptitle(
    "PSE spread across subjects and phoneme pairs\n"
    "(diverse PSEs → population gradedness is not an averaging artifact)",
    fontsize=11,
)
plt.tight_layout()
fig.savefig(outdir / "pse_by_subject_phoneme.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved pse_by_subject_phoneme.pdf")

# %% [markdown]
# ### Sigmoid Fig 2c — Multi-electrode PSE overlay
#
# Find (subject × phoneme_pair) combos with ≥3 categorical electrodes
# (k < 1, R² > 0.05, PSE ∈ [1.5, 5.5]) and PSE spread ≥ 0.5.

# %%
_R2_THRESH   = 0.05
_K_THRESH    = 1.0
_PSE_RANGE   = (1.5, 5.5)
_MIN_ELECS   = 3
_MIN_SPREAD  = 0.5

_valid_sites = model_df.dropna(subset=["sigmoid_k", "sigmoid_x0", "sigmoid_r2"])
_valid_sites = _valid_sites[
    (_valid_sites["sigmoid_r2"] > _R2_THRESH)
    & (_valid_sites["sigmoid_k"] < _K_THRESH)
    & (_valid_sites["sigmoid_x0"].between(*_PSE_RANGE))
]

_candidates = []
for (sub, pp), grp in _valid_sites.groupby(["subject", "phoneme_pair"]):
    if len(grp) < _MIN_ELECS:
        continue
    spread = grp["sigmoid_x0"].max() - grp["sigmoid_x0"].min()
    if spread < _MIN_SPREAD:
        continue
    _candidates.append({
        "subject": sub, "phoneme_pair": pp,
        "n_electrodes": len(grp), "pse_spread": spread,
        "mean_r2": grp["sigmoid_r2"].mean(), "mean_k": grp["sigmoid_k"].mean(),
    })

_candidates_df = (
    pd.DataFrame(_candidates)
    .sort_values(["n_electrodes", "pse_spread"], ascending=False)
    .head(6)
    .reset_index(drop=True)
    if _candidates else pd.DataFrame()
)
print(f"Found {len(_candidates_df)} candidate combos for PSE overlay")
if len(_candidates_df) > 0:
    print(_candidates_df.to_string(index=False))

# %%
if len(_candidates_df) > 0:
    _palette     = plt.cm.tab10.colors
    _n_cands     = len(_candidates_df)
    _n_cols_ov   = min(3, _n_cands)
    _n_rows_ov   = int(np.ceil(_n_cands / _n_cols_ov))

    fig, axes = plt.subplots(_n_rows_ov, _n_cols_ov,
                             figsize=(5 * _n_cols_ov, 4.5 * _n_rows_ov), squeeze=False)
    _ax_flat = axes.flatten()

    for ci, (_, cand) in enumerate(_candidates_df.iterrows()):
        ax = _ax_flat[ci]
        sub, pp = cand["subject"], cand["phoneme_pair"]

        combo_sites = _valid_sites[
            (_valid_sites["subject"] == sub) & (_valid_sites["phoneme_pair"] == pp)
        ].sort_values("sigmoid_x0")

        _dodge_total = 0.6
        n_elec = len(combo_sites)
        _dodge_pos = np.linspace(-_dodge_total / 2, _dodge_total / 2, n_elec)
        _jitter_h  = _dodge_total / n_elec / 2 * 0.8

        for ei_idx, (_, site) in enumerate(combo_sites.iterrows()):
            color = _palette[ei_idx % len(_palette)]
            x0_s, k_s = site["sigmoid_x0"], site["sigmoid_k"]
            ei = site["electrode_idx"]

            try:
                st = _td_idx.loc[[(sub, ei, pp)]].reset_index().dropna(subset=["resampled", "hga_norm"])
            except KeyError:
                st = pd.DataFrame()

            if len(st) > 0:
                xvals = (st["resampled"].values + _dodge_pos[ei_idx]
                         + rng.uniform(-_jitter_h, _jitter_h, len(st)))
                ax.scatter(xvals, st["hga_norm"].values,
                           c=[color], alpha=0.3, s=8, linewidths=0, zorder=1)

            ax.plot(x_fine, sigmoid_model_2p(x_fine, x0_s, k_s),
                    color=color, linewidth=2.0, zorder=3)
            ax.axvline(x0_s, color=color, linestyle=":", linewidth=1.0, alpha=0.6, zorder=2)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("Morph step")
        if ci % _n_cols_ov == 0:
            ax.set_ylabel("Normalized HGA")
        ax.set_title(f"{sub} / {pp} — {n_elec} electrodes", fontsize=9)

    for ai in range(_n_cands, len(_ax_flat)):
        _ax_flat[ai].set_visible(False)

    fig.suptitle(
        "Multi-electrode PSE overlay: categorical curves with diverse PSEs\n"
        "(each color = one electrode; dotted lines = PSE)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(outdir / "pse_overlay_candidates.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved pse_overlay_candidates.pdf")
else:
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, "No candidates found", ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    fig.savefig(outdir / "pse_overlay_candidates.pdf")
    plt.close(fig)
    print("Saved pse_overlay_candidates.pdf (no candidates found)")

# %% [markdown]
# ### Sigmoid Fig 3 — Steepness vs. acoustic decoding quality

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

valid_all_auc = model_df.dropna(subset=["sigmoid_k", "sigmoid_r2", "phon_roc_auc"])
valid_auc     = valid_all_auc[valid_all_auc["sigmoid_x0"].between(*PSE_RANGE)]
n_excluded    = len(valid_all_auc) - len(valid_auc)
print(f"PSE filter for sigmoid_vs_auc: {len(valid_auc)}/{len(valid_all_auc)} sites "
      f"(excluded {n_excluded})")

k_thresh   = EFFECTIVELY_LINEAR_K
k_display  = valid_auc["sigmoid_k"].clip(upper=k_thresh)
is_linear  = valid_auc["sigmoid_k"] > k_thresh

ax = axes[0]
sc = ax.scatter(
    valid_auc.loc[~is_linear, "phon_roc_auc"], k_display[~is_linear],
    c=valid_auc.loc[~is_linear, "sigmoid_r2"],
    cmap="viridis", alpha=0.6, edgecolors="k", linewidths=0.4, s=30,
)
ax.scatter(
    valid_auc.loc[is_linear, "phon_roc_auc"], k_display[is_linear],
    c="gray", alpha=0.6, edgecolors="k", linewidths=0.4, s=30, marker="^",
    label=f"linear (k>{k_thresh:.0f}, n={is_linear.sum()})",
)
ax.axhline(k_thresh, color="gray", linestyle=":", linewidth=1, alpha=0.5)
valid_sig = valid_auc[~is_linear]
if len(valid_sig) > 2:
    r_k, p_k = scipy.stats.pearsonr(valid_sig["phon_roc_auc"], valid_sig["sigmoid_k"])
    ax.set_title(f"Steepness vs. AUC  r={r_k:.2f}, p={p_k:.3g}\n"
                 f"(excluding {is_linear.sum()} linear sites)")
else:
    ax.set_title("Steepness vs. AUC")
ax.set_xlabel("phon_roc_auc")
ax.set_ylabel("Fitted sigmoid k (steepness)")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="sigmoid R²")

ax = axes[1]
sc = ax.scatter(
    valid_auc["phon_roc_auc"], valid_auc["sigmoid_x0"],
    c=valid_auc["sigmoid_r2"], cmap="viridis",
    alpha=0.6, edgecolors="k", linewidths=0.4, s=30,
)
ax.axhline(3.5, color="gray", linestyle="--", linewidth=1)
if len(valid_auc) > 2:
    r_x0, p_x0 = scipy.stats.pearsonr(valid_auc["phon_roc_auc"], valid_auc["sigmoid_x0"])
    ax.set_title(f"PSE vs. AUC  r={r_x0:.2f}, p={p_x0:.3g}")
else:
    ax.set_title("PSE vs. AUC")
ax.set_xlabel("phon_roc_auc")
ax.set_ylabel("Fitted PSE (x0)")
plt.colorbar(sc, ax=ax, label="sigmoid R²")

plt.tight_layout()
fig.savefig(outdir / "sigmoid_vs_auc.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved sigmoid_vs_auc.pdf")

# %% [markdown]
# ### Sigmoid Fig 4 — Catplots with sigmoid fit overlays
#
# Same 24 sample sites as Section 3, with sigmoid overlay.

# %%
mc_lookup = {
    (r["subject"], r["electrode_idx"], r["phoneme_pair"]): r
    for _, r in model_df.iterrows()
}

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
axes_flat = axes.flatten()

for ax_idx, (_, site_row) in enumerate(sample_sites.iterrows()):
    ax = axes_flat[ax_idx]
    site_data = sample_trials[sample_trials["site_label"] == site_row["site_label"]].dropna(
        subset=["resampled", "hga_norm"]
    )

    # Per-step scatter
    for step in range(1, 7):
        mask = site_data["resampled"] == step
        xvals = step + rng.uniform(-0.2, 0.2, mask.sum())
        ax.scatter(xvals, site_data.loc[mask, "hga_norm"],
                   c=[step_colors[step]], alpha=0.3, s=8, linewidths=0)

    # Mean line
    means = site_data.groupby("resampled")["hga_norm"].mean().sort_index()
    ax.plot(means.index, means.values, "k-o", linewidth=1.5, markersize=4, zorder=3)

    # Sigmoid overlay
    key = (site_row["subject"], site_row["electrode_idx"], site_row["phoneme_pair"])
    mc_row = mc_lookup.get(key)
    sig_label = ""
    if mc_row is not None:
        x0_v = mc_row.get("sigmoid_x0")
        k_v  = mc_row.get("sigmoid_k")
        r2_v = mc_row.get("sigmoid_r2")
        if x0_v is not None and k_v is not None and not (np.isnan(float(x0_v)) or np.isnan(float(k_v))):
            ax.plot(x_fine, sigmoid_model_2p(x_fine, float(x0_v), float(k_v)),
                    color="tomato", linewidth=2.0, zorder=4)
            sig_label = (f"  PSE={float(x0_v):.2f}  k={float(k_v):.2f}\n"
                         f"R²={float(r2_v):.2f}")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.5, 6.5)
    ax.set_title(f"{site_row['site_label']}\nAUC={site_row['phon_roc_auc']:.2f}{sig_label}",
                 fontsize=7.5)
    ax.set_xlabel("Morph step")
    if ax_idx % n_cols == 0:
        ax.set_ylabel("Normalized HGA")

fig.legend(
    handles=[
        plt.Line2D([0], [0], color="tomato", linewidth=2.0, label="sigmoid fit"),
        plt.Line2D([0], [0], color="k", linewidth=1.5, marker="o", markersize=4, label="data mean"),
    ],
    loc="lower right", fontsize=9, ncol=2,
)
fig.suptitle("Neurometric functions with sigmoid fits (sorted by AUC, low→high)", fontsize=11)
plt.tight_layout()
fig.savefig(outdir / "catplots_sigmoid_fits.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved catplots_sigmoid_fits.pdf")
