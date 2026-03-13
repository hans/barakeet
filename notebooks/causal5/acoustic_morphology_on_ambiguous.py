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
# # Acoustic response morphology on ambiguous inputs
#
# At acoustically selective electrodes (significant phon_roc_auc), this notebook
# applies the endpoint-trained acoustic decoder to ambiguous trials (steps 2–5)
# and asks: does the decoder remain confident (categorical account) or collapse
# toward chance (intermediate / graded account)?
#
# Key measures per site:
#   - decoder_confidence = abs(decoder_proba - 0.5) on ambiguous vs. endpoint trials
#   - ROC-AUC of acoustic decoder predicting behavior_categorical_forced on ambiguous trials
#     (AUC ≈ 0.5 = acoustic representation dissociates from percept;
#      AUC >> 0.5 = acoustic representation aligns with percept)
#
# Inputs:
#   all_outcomes.parquet  — acoustic decoder predictions on all trials (from acoustic_decoding_single_electrode)
#   phon_peaks_df.parquet — peak acoustic window per site (from acoustic_decoding_peaks)
#   {subject}_epo.fif     — epoch files for metadata
#
# Outputs:
#   trial_df.parquet   — per-trial decoder outputs with metadata at peak acoustic window
#   site_stats.parquet — per-site summary statistics

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# %%
# %load_ext autoreload
# %autoreload 2
# %%
from src.data import add_metadata_features
from src.stimuli import POD_dict
from src.viz_paper import phoneme_pair_enum, subject_enum

# %% tags=["parameters"]
all_epochs = list(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))
all_outcomes_paths = list(
    Path("outputs/causal5/acoustic_decoding_single_electrode").glob(
        "*/all_outcomes.parquet"
    )
)
phon_peaks_path = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet"

epoch_tmin = -0.4
epoch_sfreq = 100

phon_response_peak_threshold = 0.65

outdir = "outputs/causal5/acoustic_morphology_on_ambiguous"

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load acoustic sites and peak windows

# %%
phon_peaks = pl.read_parquet(phon_peaks_path).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)
acoustic_sites = phon_peaks.filter(
    pl.col("phon_roc_auc") >= phon_response_peak_threshold
)
print(f"Acoustic sites: {len(acoustic_sites)}")

# %% [markdown]
# ## Load decoder outcomes and filter to peak acoustic windows

# %%
outcomes = pl.concat(
    [pl.read_parquet(p) for p in tqdm(all_outcomes_paths, desc="Loading all_outcomes")]
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)

# keep only the acoustic category decoder (categorical_acoustic_cue measure)
outcomes = outcomes.filter(pl.col("measure") == "categorical_acoustic_cue")

# Filter to peak acoustic window per site by joining on (subject, electrode_idx, phoneme_pair)
# and keeping only rows where (smin, smax) matches the peak window
outcomes = outcomes.join(
    acoustic_sites.select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]),
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="inner",
)

# Per-fold version: keep fold column for per-fold neurometric estimates
# (each epoch appears once per fold as a held-out test trial)
proba_by_trial_fold = outcomes  # already one row per (site, epoch, fold)

# Average decoder_proba across CV folds per (site, epoch)
proba_by_trial = outcomes.group_by(
    ["subject", "electrode_idx", "phoneme_pair", "epoch_idx"]
).agg(pl.col("decoder_proba").mean())

print(f"Trials × sites: {len(proba_by_trial)}")

# %% [markdown]
# ## Load epoch metadata

# %%
all_md_rows = []
epoch_data_cache: dict[str, np.ndarray] = {}
epoch_times_cache: dict[str, np.ndarray] = {}

for epoch_path in tqdm(sorted(all_epochs), desc="Loading epochs"):
    subject = re.findall(r"(EC\d+)_epo", str(epoch_path))[0]
    ep = mne.read_epochs(epoch_path, preload=True, verbose=False)
    ep.apply_baseline()
    epoch_data_cache[subject] = ep.get_data()   # (n_trials, n_channels, n_samples)
    epoch_times_cache[subject] = ep.times
    md = add_metadata_features(ep.metadata).assign(subject=subject)
    del ep
    md.index.name = "epoch_idx"
    all_md_rows.append(
        md.reset_index()[
            [
                "subject",
                "epoch_idx",
                "phoneme_pair",
                "resampled",
                "behavior_dummy_forced",
                "word_end",
                "ambiguity",
            ]
        ]
    )

all_md = pl.from_pandas(pd.concat(all_md_rows, ignore_index=True)).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)

# %%
trial_df = proba_by_trial.join(
    all_md, on=["subject", "epoch_idx", "phoneme_pair"], how="left"
).with_columns(
    (pl.col("decoder_proba") - 0.5).abs().alias("confidence"),
    pl.col("resampled").is_in([1, 6]).alias("is_endpoint"),
    pl.col("resampled").is_in([2, 3, 4, 5]).alias("is_ambiguous"),
)

trial_df.write_parquet(outdir / "trial_df.parquet")
print(f"trial_df saved: {len(trial_df)} rows")

# Per-fold trial frame: same as trial_df but with fold column preserved
trial_df_fold = proba_by_trial_fold.join(
    all_md, on=["subject", "epoch_idx", "phoneme_pair"], how="left"
).with_columns(
    (pl.col("decoder_proba") - 0.5).abs().alias("confidence"),
    pl.col("resampled").is_in([1, 6]).alias("is_endpoint"),
    pl.col("resampled").is_in([2, 3, 4, 5]).alias("is_ambiguous"),
)
trial_df_fold_pd = trial_df_fold.to_pandas()
print(f"trial_df_fold: {len(trial_df_fold_pd)} rows (incl. fold dim)")

trial_df.head()

# %% [markdown]
# ## Per-site statistics

# %%
site_records = []
site_keys = acoustic_sites.select(
    ["subject", "electrode_idx", "phoneme_pair", "phon_roc_auc", "smin", "smax"]
).to_pandas()

for _, row in tqdm(
    site_keys.iterrows(), total=len(site_keys), desc="Computing site stats"
):
    sub = row["subject"]
    ei = row["electrode_idx"]
    pp = row["phoneme_pair"]

    site = trial_df.filter(
        (pl.col("subject") == sub)
        & (pl.col("electrode_idx") == ei)
        & (pl.col("phoneme_pair") == pp)
    ).to_pandas()

    endpoints = site[site["is_endpoint"]]
    ambig = site[site["is_ambiguous"]]

    if len(ambig) < 5 or len(endpoints) < 5:
        continue

    # --- Confidence test ---
    # Mann-Whitney: are ambiguous trials less confident than endpoint trials?
    stat, confidence_drop_p = scipy.stats.mannwhitneyu(
        ambig["confidence"].values,
        endpoints["confidence"].values,
        alternative="less",
    )
    # AUC: 0.5 = same confidence, 0.0 = ambig uniformly less confident
    confidence_drop_auc = stat / (len(ambig) * len(endpoints))

    # --- Decoder-behavior agreement on ambiguous trials ---
    ambig_nona = ambig.dropna(subset=["behavior_dummy_forced"])
    behavior_labels = (ambig_nona["behavior_dummy_forced"] > 0).astype(int)
    auc_behavior_on_ambig = roc_auc_score(behavior_labels, ambig_nona["decoder_proba"])

    # Same on endpoints (sanity check: should be ~1.0 for good acoustic sites)
    ep_nona = endpoints.dropna(subset=["behavior_dummy_forced"])
    # On endpoints behavior and acoustic cue are almost always aligned; use acoustic_cue
    # via decoder_proba directly
    ep_behavior_labels = (ep_nona["behavior_dummy_forced"] > 0).astype(int)
    auc_behavior_on_endpoints = roc_auc_score(
        ep_behavior_labels, ep_nona["decoder_proba"]
    )

    site_records.append(
        {
            "subject": sub,
            "electrode_idx": ei,
            "phoneme_pair": pp,
            "phon_roc_auc": row["phon_roc_auc"],
            "smin": row["smin"],
            "smax": row["smax"],
            "n_endpoint_trials": len(endpoints),
            "n_ambig_trials": len(ambig),
            "mean_endpoint_confidence": endpoints["confidence"].mean(),
            "mean_ambig_confidence": ambig["confidence"].mean(),
            "confidence_drop_auc": confidence_drop_auc,
            "confidence_drop_p": confidence_drop_p,
            "behavior_auc_on_ambig": auc_behavior_on_ambig,
            "behavior_auc_on_endpoints": auc_behavior_on_endpoints,
        }
    )

site_stats = pd.DataFrame(site_records)
site_stats.to_parquet(outdir / "site_stats.parquet", index=False)
print(f"site_stats: {len(site_stats)} sites")
site_stats.describe()

# %% [markdown]
# ## AX discrimination: adjacent-step decoders
#
# For each site and each adjacent step pair (1v2, 2v3, ..., 5v6), train a fresh
# binary decoder on the raw HGA at the site's peak acoustic window. This tests
# whether the neural signal itself can distinguish adjacent morph steps — the
# hallmark of categorical perception is high discrimination at the category
# boundary and low discrimination within categories.

# %%
from src.models.decoding import fit_train_test

ax_rows = []
step_pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

for _, site_row in tqdm(site_keys.iterrows(), total=len(site_keys), desc="AX discrimination"):
    sub = site_row["subject"]
    ei = int(site_row["electrode_idx"])
    pp = site_row["phoneme_pair"]
    smin_w, smax_w = int(site_row["smin"]), int(site_row["smax"])

    data_arr = epoch_data_cache[sub]

    # Get metadata for this site's trials (need resampled step)
    site_md = all_md.filter(
        (pl.col("subject") == sub) & (pl.col("phoneme_pair") == pp)
    ).to_pandas()

    for step_a, step_b in step_pairs:
        mask_a = site_md["resampled"] == step_a
        mask_b = site_md["resampled"] == step_b
        idx_a = site_md.loc[mask_a, "epoch_idx"].values.astype(int)
        idx_b = site_md.loc[mask_b, "epoch_idx"].values.astype(int)

        if len(idx_a) < 5 or len(idx_b) < 5:
            continue

        # Extract windowed HGA for this electrode
        X_a = data_arr[idx_a, ei, smin_w:smax_w]  # (n_a, window_size)
        X_b = data_arr[idx_b, ei, smin_w:smax_w]  # (n_b, window_size)
        X = np.vstack([X_a, X_b])
        y = np.array([0] * len(idx_a) + [1] * len(idx_b))

        fitted = fit_train_test(
            X, y,
            num_classes=2,
            scoring=["roc_auc"],
            stratify=y,
            num_repeats=5,
            n_jobs=1,
        )

        if fitted is None:
            continue

        test_aucs = fitted["test_roc_auc"]
        ax_rows.append({
            "subject": sub,
            "electrode_idx": ei,
            "phoneme_pair": pp,
            "step_a": step_a,
            "step_b": step_b,
            "n_a": len(idx_a),
            "n_b": len(idx_b),
            "roc_auc": float(np.mean(test_aucs)),
            "roc_auc_std": float(np.std(test_aucs)),
        })

ax_discrimination_df = pd.DataFrame(ax_rows)
ax_discrimination_df.to_parquet(outdir / "ax_discrimination_df.parquet", index=False)
print(f"ax_discrimination_df: {len(ax_discrimination_df)} rows")
ax_discrimination_df.head()

# %% [markdown]
# ## Figures

# %%
trial_pd = trial_df.to_pandas()

# %% [markdown]
# ### Fig 1 — Decoder confidence by morph step

# %%
fig, ax = plt.subplots(figsize=(7, 4))

steps = sorted(trial_pd["resampled"].dropna().unique())
conf_by_step = [
    trial_pd.loc[trial_pd["resampled"] == s, "confidence"].dropna().values
    for s in steps
]

bp = ax.boxplot(
    conf_by_step,
    positions=steps,
    widths=0.6,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=2),
)
for patch, step in zip(bp["boxes"], steps):
    patch.set_facecolor("steelblue" if step in (1, 6) else "lightyellow")
    patch.set_edgecolor("black")

ax.axhline(
    0, color="gray", linestyle="--", linewidth=1, label="chance confidence (proba=0.5)"
)
ax.set_xlabel("Morph step (1 = clear /d/ endpoint, 6 = clear /n/ endpoint)")
ax.set_ylabel("Decoder confidence  |proba − 0.5|")
ax.set_title("Acoustic decoder confidence by morph step\n(across all acoustic sites)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(outdir / "confidence_by_step.pdf")
plt.close(fig)

# %% [markdown]
# ### Fig 2 — Endpoint vs. ambiguous confidence per site

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(
    site_stats["mean_endpoint_confidence"],
    site_stats["mean_ambig_confidence"],
    c=site_stats["phon_roc_auc"],
    cmap="viridis",
    alpha=0.7,
    edgecolors="k",
    linewidths=0.5,
)
lim = [
    0,
    site_stats[["mean_endpoint_confidence", "mean_ambig_confidence"]].values.max()
    * 1.05,
]
ax.plot(lim, lim, "k--", linewidth=1, label="equal confidence")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Mean confidence on endpoints (steps 1 & 6)")
ax.set_ylabel("Mean confidence on ambiguous trials (steps 2–5)")
ax.set_title("Representational commitment:\nendpoints vs. ambiguous steps")
ax.legend(fontsize=9)
sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=plt.Normalize(
        site_stats["phon_roc_auc"].min(), site_stats["phon_roc_auc"].max()
    ),
)
plt.colorbar(sm, ax=ax, label="phon_roc_auc")
plt.tight_layout()
fig.savefig(outdir / "confidence_scatter.pdf")
plt.close(fig)

# %% [markdown]
# ### Fig 3 — Decoder-behavior agreement distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

for ax, col, title in [
    (axes[0], "behavior_auc_on_ambig", "Ambiguous trials (steps 2–5)"),
    (axes[1], "behavior_auc_on_endpoints", "Endpoint trials (steps 1 & 6)"),
]:
    vals = site_stats[col].dropna()
    ax.hist(vals, bins=20, color="steelblue", edgecolor="k", alpha=0.8)
    ax.axvline(
        0.5, color="red", linestyle="--", linewidth=1.5, label="chance (AUC=0.5)"
    )
    t, p = scipy.stats.ttest_1samp(vals, popmean=0.5)
    ax.set_title(f"{title}\nmean={vals.mean():.3f}, t={t:.2f}, p={p:.3g}")
    ax.set_xlabel("Acoustic decoder ROC-AUC\n(predicting behavior_categorical_forced)")
    ax.set_ylabel("Number of sites")
    ax.legend(fontsize=9)

plt.suptitle("Does the acoustic decoder predict behavior?", fontsize=12)
plt.tight_layout()
fig.savefig(outdir / "behavior_agreement.pdf")
plt.close(fig)

# %% [markdown]
# ### Fig 4 — Single-site example

# %%
# Select exemplar: site with highest ambiguous confidence (most categorical-looking)
exemplar_idx = site_stats["mean_ambig_confidence"].idxmax()
exemplar = site_stats.iloc[exemplar_idx]

site_trials = trial_pd[
    (trial_pd["subject"] == exemplar["subject"])
    & (trial_pd["electrode_idx"] == exemplar["electrode_idx"])
    & (trial_pd["phoneme_pair"] == exemplar["phoneme_pair"])
].dropna(subset=["resampled", "behavior_dummy_forced"])

fig, ax = plt.subplots(figsize=(6, 4))
colors = {0: "steelblue", 1: "tomato"}
labels = {0: "Heard /d/ (or /b/ or /p/)", 1: "Heard /n/ (or /m/ or /b/)"}
for beh, grp in site_trials.groupby("behavior_dummy_forced"):
    ax.scatter(
        grp["resampled"] + np.random.uniform(-0.15, 0.15, len(grp)),
        grp["decoder_proba"],
        c=colors.get(beh, "gray"),
        alpha=0.35,
        s=15,
        label=labels.get(beh, str(beh)),
    )

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Morph step")
ax.set_ylabel("Acoustic decoder P(/d/)")
ax.set_title(
    f"Example site: {exemplar['subject']} elec {exemplar['electrode_idx']}"
    f" {exemplar['phoneme_pair']}\n"
    f"endpoint conf={exemplar['mean_endpoint_confidence']:.2f}, "
    f"ambig conf={exemplar['mean_ambig_confidence']:.2f}"
)
ax.legend(fontsize=9)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
fig.savefig(outdir / "site_example.pdf")
plt.close(fig)

print("All figures saved.")

# %% [markdown]
# ### Fig 5 — Sample catplots: decoder output by morph step (individual electrodes)
#
# Select a set of sites spanning the phon_roc_auc range. For each site, plot
# per-trial decoder_proba by morph step, colored by reported percept, with the
# mean neurometric function overlaid as a line.

# %%
n_sample = 24

# Sample sites evenly across the AUC range
site_stats_sorted = site_stats.sort_values("phon_roc_auc").reset_index(drop=True)
sample_idx = np.round(np.linspace(0, len(site_stats_sorted) - 1, n_sample)).astype(int)
sample_sites = site_stats_sorted.iloc[sample_idx].copy().reset_index(drop=True)
sample_sites["site_label"] = sample_sites.apply(
    lambda r: f"{r['subject']} e{int(r['electrode_idx'])} {r['phoneme_pair']}", axis=1
)

# Build a lookup from (subject, electrode_idx, phoneme_pair) → site_label
label_map = {
    (r["subject"], r["electrode_idx"], r["phoneme_pair"]): r["site_label"]
    for _, r in sample_sites.iterrows()
}

sample_trials = trial_pd[
    trial_pd.apply(
        lambda r: (r["subject"], r["electrode_idx"], r["phoneme_pair"]) in label_map,
        axis=1,
    )
].copy()
sample_trials["site_label"] = sample_trials.apply(
    lambda r: label_map[(r["subject"], r["electrode_idx"], r["phoneme_pair"])], axis=1
)

# Per-fold version: used for fold-level neurometric lines in Fig 5
sample_trials_fold = trial_df_fold_pd[
    trial_df_fold_pd.apply(
        lambda r: (r["subject"], r["electrode_idx"], r["phoneme_pair"]) in label_map,
        axis=1,
    )
].copy()
sample_trials_fold["site_label"] = sample_trials_fold.apply(
    lambda r: label_map[(r["subject"], r["electrode_idx"], r["phoneme_pair"])], axis=1
)

# %%
rng = np.random.default_rng(0)
n_cols = 4
n_rows = int(np.ceil(n_sample / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
axes_flat = axes.flatten()

beh_colors = {0: "steelblue", 1: "tomato"}
beh_labels = {0: "heard cat-0", 1: "heard cat-1"}

for ax_idx, (_, site_row) in enumerate(sample_sites.iterrows()):
    ax = axes_flat[ax_idx]
    site_data = sample_trials[
        sample_trials["site_label"] == site_row["site_label"]
    ].dropna(subset=["resampled", "decoder_proba"])

    # Per-behavior jittered scatter
    for beh, color in beh_colors.items():
        mask = site_data["behavior_dummy_forced"] == beh
        xvals = site_data.loc[mask, "resampled"] + rng.uniform(-0.2, 0.2, mask.sum())
        ax.scatter(
            xvals,
            site_data.loc[mask, "decoder_proba"],
            c=color,
            alpha=0.3,
            s=8,
            linewidths=0,
            label=beh_labels[beh],
        )

    # Per-fold neurometric lines (thin, behind the mean)
    site_fold_data = sample_trials_fold[
        sample_trials_fold["site_label"] == site_row["site_label"]
    ].dropna(subset=["resampled", "decoder_proba", "fold"])
    for fold_id, fold_grp in site_fold_data.groupby("fold"):
        fold_means = fold_grp.groupby("resampled")["decoder_proba"].mean().sort_index()
        ax.plot(
            fold_means.index,
            fold_means.values,
            "-",
            color="gray",
            alpha=0.4,
            linewidth=0.7,
            zorder=3,
        )

    # Overall mean neurometric line (across folds)
    means = site_data.groupby("resampled")["decoder_proba"].mean().sort_index()
    ax.plot(means.index, means.values, "k-o", linewidth=1.5, markersize=4, zorder=5)

    # AX discrimination curve on secondary y-axis
    site_ax = ax_discrimination_df[
        (ax_discrimination_df["subject"] == site_row["subject"])
        & (ax_discrimination_df["electrode_idx"] == site_row["electrode_idx"])
        & (ax_discrimination_df["phoneme_pair"] == site_row["phoneme_pair"])
    ]
    if len(site_ax) > 0:
        ax2 = ax.twinx()
        midpoints = (site_ax["step_a"].values + site_ax["step_b"].values) / 2.0
        ax2.plot(
            midpoints, site_ax["roc_auc"].values,
            "D--", color="green", linewidth=1.2, markersize=4, alpha=0.8, zorder=6,
        )
        ax2.set_ylim(0.4, 1.0)
        ax2.tick_params(axis="y", labelcolor="green", labelsize=6)
        if ax_idx % n_cols == n_cols - 1:
            ax2.set_ylabel("AX discrim. AUC", color="green", fontsize=7)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.5, 6.5)
    # ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"{site_row['site_label']}\nAUC={site_row['phon_roc_auc']:.2f}", fontsize=7.5
    )
    ax.set_xlabel("Morph step")
    if ax_idx % n_cols == 0:
        ax.set_ylabel("Decoder P(cat-1)")

# Global legend using last axis
handles = [
    plt.scatter([], [], c=beh_colors[0], alpha=0.6, s=20, label="heard cat-0"),
    plt.scatter([], [], c=beh_colors[1], alpha=0.6, s=20, label="heard cat-1"),
    plt.Line2D([0], [0], color="gray", linewidth=0.8, label="per-fold mean"),
    plt.Line2D(
        [0],
        [0],
        color="k",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label="overall mean",
    ),
    plt.Line2D(
        [0],
        [0],
        color="green",
        linewidth=1.2,
        linestyle="--",
        marker="D",
        markersize=4,
        label="AX discrim. AUC",
    ),
]
fig.legend(handles=handles, loc="lower right", fontsize=9, ncol=5)
fig.suptitle(
    "Neurometric function at individual acoustic sites (sorted by AUC, low→high)",
    fontsize=11,
)
plt.tight_layout()
fig.savefig(outdir / "catplots_sample.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved catplots_sample.pdf")

# %% [markdown]
# ### Fig 5b — Population AX discrimination curve
#
# Average AX discrimination AUC across all sites for each adjacent step pair.
# Categorical prediction: peak discrimination at boundary pairs (3v4), low at
# within-category pairs (1v2, 5v6).

# %%
fig, ax = plt.subplots(figsize=(6, 4))

pair_labels = [f"{a}v{b}" for a, b in step_pairs]
pair_midpoints = [(a + b) / 2.0 for a, b in step_pairs]

# Mean ± SEM across sites for each step pair
for i, (step_a, step_b) in enumerate(step_pairs):
    pair_data = ax_discrimination_df[
        (ax_discrimination_df["step_a"] == step_a)
        & (ax_discrimination_df["step_b"] == step_b)
    ]["roc_auc"]
    mean_auc = pair_data.mean()
    sem_auc = pair_data.std() / np.sqrt(len(pair_data))
    ax.bar(
        pair_midpoints[i], mean_auc - 0.5, bottom=0.5,
        width=0.6, color="steelblue", edgecolor="k", alpha=0.8,
    )
    ax.errorbar(
        pair_midpoints[i], mean_auc, yerr=sem_auc,
        fmt="none", ecolor="k", capsize=4, linewidth=1.5,
    )

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
ax.set_xticks(pair_midpoints)
ax.set_xticklabels(pair_labels)
ax.set_xlabel("Adjacent step pair")
ax.set_ylabel("Mean AX discrimination AUC")
ax.set_title(
    "Population AX discrimination: can single electrodes\n"
    "distinguish adjacent morph steps?"
)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(outdir / "ax_discrimination_population.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved ax_discrimination_population.pdf")

# %% [markdown]
# ### Fig 6 — HGA timecourse by morph step (sample sites)
#
# For the same sample of sites, load raw preloaded epochs and compute the
# mean HGA trace at each morph step. Steps are color-coded blue→red (1→6).
# The peak acoustic decoding window is shaded; POD is marked.

# %%
# epoch_data_cache and epoch_times_cache were populated during the metadata loading step above
step_cmap = plt.cm.RdBu_r
step_colors = {s: step_cmap(i / 5) for i, s in enumerate(range(1, 7))}

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True)
axes_flat = axes.flatten()

for ax_idx, (_, site_row) in enumerate(sample_sites.iterrows()):
    ax = axes_flat[ax_idx]
    sub = site_row["subject"]
    ei = int(site_row["electrode_idx"])
    pp = site_row["phoneme_pair"]
    smin_w = int(site_row["smin"])
    smax_w = int(site_row["smax"])

    data_arr = epoch_data_cache[sub]
    times = epoch_times_cache[sub]

    site_data = sample_trials[
        (sample_trials["site_label"] == site_row["site_label"])
        & sample_trials["resampled"].notna()
    ]

    for step in range(1, 7):
        ep_idxs = (
            site_data[site_data["resampled"] == step]["epoch_idx"]
            .dropna()
            .astype(int)
            .values
        )
        if len(ep_idxs) == 0:
            continue
        traces = data_arr[ep_idxs, ei, :]  # (n, n_samples)
        mean_trace = traces.mean(axis=0)
        sem_trace = traces.std(axis=0) / np.sqrt(len(ep_idxs))
        ax.plot(
            times,
            mean_trace,
            color=step_colors[step],
            linewidth=1.3,
            label=f"step {step}",
        )
        ax.fill_between(
            times,
            mean_trace - sem_trace,
            mean_trace + sem_trace,
            color=step_colors[step],
            alpha=0.12,
        )

    # Acoustic window shading
    tmin_w = epoch_tmin + smin_w / epoch_sfreq
    tmax_w = epoch_tmin + smax_w / epoch_sfreq
    ax.axvspan(
        tmin_w, tmax_w, alpha=0.18, color="goldenrod", zorder=0, label="acoustic window"
    )

    # POD
    pod = POD_dict.get(pp, 0.295)
    ax.axvline(pod, color="black", linestyle="--", linewidth=0.9, label="POD")
    ax.axhline(0, color="gray", linewidth=0.4)

    ax.set_title(
        f"{site_row['site_label']}\nAUC={site_row['phon_roc_auc']:.2f}", fontsize=7.5
    )
    if ax_idx >= 8:
        ax.set_xlabel("Time (s)")
    if ax_idx % 4 == 0:
        ax.set_ylabel("HGA (z-scored)")

# Legend on last axis
step_handles = [
    plt.Line2D([0], [0], color=step_colors[s], linewidth=2, label=f"step {s}")
    for s in range(1, 7)
]
step_handles += [
    plt.matplotlib.patches.Patch(color="goldenrod", alpha=0.4, label="acoustic window"),
    plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="POD"),
]
fig.legend(handles=step_handles, loc="lower right", ncol=4, fontsize=8)
fig.suptitle(
    "HGA timecourse by morph step — sample sites (low AUC → high AUC)", fontsize=11
)
plt.tight_layout()
fig.savefig(outdir / "hga_timecourse_sample.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved hga_timecourse_sample.pdf")

# %% [markdown]
# ## HGA-based acoustic polarity per site
#
# Polarity = sign(mean HGA at step 1 − mean HGA at step 6) within each site's
# peak acoustic window. Positive = electrode responds more strongly to cat-0 (step 1).
# Computed from raw HGA, not decoder outputs (which are always oriented P(cat-1)).

# %%
site_polarity: dict[tuple, float] = {}
trial_pd_by_subject = trial_pd.groupby("subject")

for subject, subj_sites in acoustic_sites.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
).to_pandas().groupby("subject", observed=True):
    data_arr = epoch_data_cache[subject]
    subj_trials = trial_pd_by_subject.get_group(subject) if subject in trial_pd["subject"].values else None
    if subj_trials is None:
        continue
    for _, site_row in subj_sites.iterrows():
        ei = int(site_row["electrode_idx"])
        pp = site_row["phoneme_pair"]
        smin_w, smax_w = int(site_row["smin"]), int(site_row["smax"])
        site_t = subj_trials[
            (subj_trials["electrode_idx"] == ei) & (subj_trials["phoneme_pair"] == pp)
        ]
        idx1 = site_t[site_t["resampled"] == 1.0]["epoch_idx"].dropna().astype(int).values
        idx6 = site_t[site_t["resampled"] == 6.0]["epoch_idx"].dropna().astype(int).values
        if len(idx1) == 0 or len(idx6) == 0:
            site_polarity[(subject, ei, pp)] = np.nan
            continue
        mean1 = data_arr[idx1, ei, smin_w:smax_w].mean()
        mean6 = data_arr[idx6, ei, smin_w:smax_w].mean()
        site_polarity[(subject, ei, pp)] = float(np.sign(mean1 - mean6))

print(f"Polarity computed for {len(site_polarity)} sites")

# %% [markdown]
# ## Neurometric analysis: sigmoid fits to per-site decoder response functions
#
# For each acoustic site, we characterize the shape of its neurometric function —
# the mapping from morph step to mean decoder_proba — by fitting a logistic sigmoid:
#
#   f(x; x0, k) = 1 / (1 + exp(-(x - x0) / k))
#
# where x0 is the PSE (point of subjective equality, inflection point of the curve)
# and k controls slope steepness (smaller k = steeper near x0 = more sigmoid-like).
#
# Before fitting we normalize the curve direction: all sites are flipped so that
# the curve is monotonically decreasing (step 1 high → step 6 low), making x0
# and k directly comparable across sites regardless of phoneme-pair polarity.
#
# Hypothesis (graded tuning / compression):
#   Strong acoustic sites (high phon_roc_auc) should show a steep transition near
#   the decision boundary (steps 3–4) with flat extremes — i.e., smaller k. Sites
#   with weaker tuning may show shallower, more linear functions (larger k).


# %%
def sigmoid(x, x0, k):
    return 1.0 / (1.0 + np.exp(-(x - x0) / k))


neurometric_rows = []
steps_all = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

for _, site_row in tqdm(
    site_stats.iterrows(), total=len(site_stats), desc="Fitting neurometrics"
):
    sub = site_row["subject"]
    ei = site_row["electrode_idx"]
    pp = site_row["phoneme_pair"]

    site = trial_pd[
        (trial_pd["subject"] == sub)
        & (trial_pd["electrode_idx"] == ei)
        & (trial_pd["phoneme_pair"] == pp)
    ].dropna(subset=["resampled", "decoder_proba"])

    means_s = site.groupby("resampled")["decoder_proba"].mean().reindex(steps_all)
    valid = means_s.dropna()

    if len(valid) < 4:
        continue

    x = valid.index.values.astype(float)
    y = valid.values

    # Polarity from HGA means in the acoustic window (not decoder outputs)
    early_polarity = site_polarity.get((sub, ei, pp), np.nan)

    y_fit = (1.0 - y) if (not np.isnan(early_polarity) and early_polarity < 0) else y

    try:
        popt, _ = curve_fit(
            sigmoid,
            x,
            y_fit,
            p0=[3.5, 1.0],
            bounds=([1.0, 0.05], [6.0, 10.0]),
            maxfev=2000,
        )
        x0, k = popt
        y_pred = sigmoid(x, x0, k)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    except Exception:
        x0, k, r2 = np.nan, np.nan, np.nan

    # Per-fold sigmoid fits (use same polarity convention as aggregate fit)
    site_fold = trial_df_fold_pd[
        (trial_df_fold_pd["subject"] == sub)
        & (trial_df_fold_pd["electrode_idx"] == ei)
        & (trial_df_fold_pd["phoneme_pair"] == pp)
    ].dropna(subset=["resampled", "decoder_proba", "fold"])

    pse_folds, k_folds = [], []
    for fold_id, fold_grp in site_fold.groupby("fold"):
        fold_means = (
            fold_grp.groupby("resampled")["decoder_proba"].mean().reindex(steps_all)
        )
        fold_valid = fold_means.dropna()
        if len(fold_valid) < 4:
            continue
        x_f = fold_valid.index.values.astype(float)
        y_f = fold_valid.values
        y_f_fit = (
            (1.0 - y_f)
            if (not np.isnan(early_polarity) and early_polarity < 0)
            else y_f
        )
        try:
            popt_f, _ = curve_fit(
                sigmoid,
                x_f,
                y_f_fit,
                p0=[3.5, 1.0],
                bounds=([1.0, 0.05], [6.0, 10.0]),
                maxfev=2000,
            )
            pse_folds.append(popt_f[0])
            k_folds.append(popt_f[1])
        except Exception:
            pass

    norm_probas = {
        f"norm_proba_step{int(s)}": (
            float(1.0 - means_s[s])
            if (early_polarity is not np.nan and early_polarity < 0)
            else float(means_s[s])
        )
        for s in steps_all
    }

    neurometric_rows.append(
        {
            "subject": sub,
            "electrode_idx": ei,
            "phoneme_pair": pp,
            "smin": site_row["smin"],
            "smax": site_row["smax"],
            "phon_roc_auc": site_row["phon_roc_auc"],
            "pse": x0,
            "slope_k": k,
            "sigmoid_r2": r2,
            "early_polarity": early_polarity,
            "pse_fold_mean": float(np.nanmean(pse_folds)) if pse_folds else np.nan,
            "pse_fold_std": float(np.nanstd(pse_folds)) if pse_folds else np.nan,
            "slope_k_fold_mean": float(np.nanmean(k_folds)) if k_folds else np.nan,
            "slope_k_fold_std": float(np.nanstd(k_folds)) if k_folds else np.nan,
            "n_folds_fit": len(pse_folds),
            **norm_probas,
        }
    )

neurometrics_df = pd.DataFrame(neurometric_rows)
neurometrics_df.to_parquet(outdir / "neurometrics_df.parquet", index=False)
print(f"neurometrics_df: {len(neurometrics_df)} sites")
neurometrics_df.describe()

# %% [markdown]
# ### Neurometric Fig 1 — PSE and slope vs. AUC

# %%
nm = neurometrics_df.dropna(subset=["pse", "slope_k", "sigmoid_r2"])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# PSE vs AUC — errorbars show fold std
ax = axes[0]
sc = ax.scatter(
    nm["phon_roc_auc"],
    nm["pse"],
    c=nm["sigmoid_r2"],
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
    linewidths=0.4,
    s=30,
    zorder=3,
)
ax.errorbar(
    nm["phon_roc_auc"],
    nm["pse"],
    yerr=nm["pse_fold_std"].fillna(0),
    fmt="none",
    ecolor="gray",
    elinewidth=0.6,
    alpha=0.5,
    zorder=2,
)
ax.axhline(3.5, color="gray", linestyle="--", linewidth=1, label="midpoint (step 3.5)")
ax.set_xlabel("phon_roc_auc")
ax.set_ylabel("PSE (sigmoid midpoint, steps)")
ax.set_title("PSE vs. acoustic AUC")
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="sigmoid R²")

# slope k vs AUC — errorbars show fold std
ax = axes[1]
sc = ax.scatter(
    nm["phon_roc_auc"],
    nm["slope_k"],
    c=nm["sigmoid_r2"],
    cmap="viridis",
    alpha=0.6,
    edgecolors="k",
    linewidths=0.4,
    s=30,
    zorder=3,
)
ax.errorbar(
    nm["phon_roc_auc"],
    nm["slope_k"],
    yerr=nm["slope_k_fold_std"].fillna(0),
    fmt="none",
    ecolor="gray",
    elinewidth=0.6,
    alpha=0.5,
    zorder=2,
)
ax.set_xlabel("phon_roc_auc")
ax.set_ylabel("Slope parameter k (smaller = steeper)")
ax.set_title("Neurometric steepness vs. acoustic AUC\n(low k → compressed extremes)")
plt.colorbar(sc, ax=ax, label="sigmoid R²")

r_pse, p_pse = scipy.stats.pearsonr(nm["phon_roc_auc"], nm["pse"].fillna(3.5))
r_k, p_k = scipy.stats.pearsonr(
    nm.dropna(subset=["slope_k"])["phon_roc_auc"],
    nm.dropna(subset=["slope_k"])["slope_k"],
)
axes[0].set_title(f"PSE vs. AUC  r={r_pse:.2f}, p={p_pse:.3g}")
axes[1].set_title(
    f"Slope k vs. AUC  r={r_k:.2f}, p={p_k:.3g}\n(low k → steeper at boundary)"
)

# Sigmoid R² distribution
ax = axes[2]
ax.hist(nm["sigmoid_r2"], bins=20, color="steelblue", edgecolor="k", alpha=0.8)
ax.axvline(
    nm["sigmoid_r2"].median(),
    color="red",
    linestyle="--",
    label=f"median={nm['sigmoid_r2'].median():.2f}",
)
ax.set_xlabel("Sigmoid R²  (goodness of fit to neurometric)")
ax.set_ylabel("Number of sites")
ax.set_title("How well does a sigmoid describe the neurometric?")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(outdir / "neurometrics_scatter.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved neurometrics_scatter.pdf")

# %% [markdown]
# ### Neurometric Fig 2 — Average neurometric curves by AUC tertile
#
# Split sites into three AUC groups. Show the mean ± SEM normalized neurometric
# curve for each group. A steeper, more sigmoid-shaped curve in the high-AUC group
# would support the compression hypothesis.

# %%
nm_valid = neurometrics_df.dropna(
    subset=[f"norm_proba_step{int(s)}" for s in steps_all]
)
nm_valid = nm_valid.assign(
    auc_tertile=pd.qcut(
        nm_valid["phon_roc_auc"], 3, labels=["low AUC", "mid AUC", "high AUC"]
    )
)

fig, ax = plt.subplots(figsize=(6, 4))
tertile_colors = {"low AUC": "lightcoral", "mid AUC": "gold", "high AUC": "steelblue"}

for tertile, grp in nm_valid.groupby("auc_tertile", observed=True):
    curves = grp[[f"norm_proba_step{int(s)}" for s in steps_all]].values  # (n_sites, 6)
    mean_curve = np.nanmean(curves, axis=0)
    sem_curve = np.nanstd(curves, axis=0) / np.sqrt(np.sum(~np.isnan(curves), axis=0))
    auc_range = f"[{grp['phon_roc_auc'].min():.2f}–{grp['phon_roc_auc'].max():.2f}]"
    label = f"{tertile} {auc_range} (n={len(grp)})"
    color = tertile_colors[str(tertile)]
    ax.plot(
        steps_all, mean_curve, "o-", color=color, linewidth=2, markersize=5, label=label
    )
    ax.fill_between(
        steps_all,
        mean_curve - sem_curve,
        mean_curve + sem_curve,
        color=color,
        alpha=0.2,
    )

ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax.set_xticks(range(1, 7))
ax.set_xlim(0.5, 6.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Morph step")
ax.set_ylabel(
    "Mean normalized decoder P(cat-0)\n(polarity-corrected so step-1 is high)"
)
ax.set_title(
    "Average neurometric function by AUC tertile\n(compression hypothesis: high-AUC sites steepest near boundary)"
)
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
fig.savefig(outdir / "neurometrics_by_auc.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved neurometrics_by_auc.pdf")

# %% [markdown]
# ## Electrode clustering by neurometric curve shape
#
# Cluster electrodes by the *shape* of their raw neurometric curve — mean
# decoder_proba per morph step, computed directly from trial_pd with no polarity
# correction. (Polarity correction is appropriate for sigmoid fitting but not
# here: it would artificially invert some curves and create spurious clusters.)
#
# Before clustering we subtract each site's curve minimum to remove y-intercept
# differences (driven by overall HGA level or decoder calibration), leaving
# only the trajectory shape. We then divide by the per-site range so that
# amplitude differences don't dominate distance.
#
# k is chosen via silhouette score over k=2..6.

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Raw mean decoder_proba per (site, step) — no polarity correction
_raw_means = (
    trial_df.group_by(["subject", "electrode_idx", "phoneme_pair", "resampled"])
    .agg(pl.col("decoder_proba").mean())
    .sort(["subject", "electrode_idx", "phoneme_pair", "resampled"])
    .to_pandas()
)
_raw_wide = _raw_means.pivot_table(
    index=["subject", "electrode_idx", "phoneme_pair"],
    columns="resampled",
    values="decoder_proba",
)
_raw_wide.columns = [f"raw_proba_step{int(c)}" for c in _raw_wide.columns]
_raw_wide = _raw_wide.reset_index()

# Pivot AX discrimination to wide format: one column per step pair
_ax_wide = ax_discrimination_df.assign(
    pair_label=lambda df: df["step_a"].astype(str) + "v" + df["step_b"].astype(str)
).pivot_table(
    index=["subject", "electrode_idx", "phoneme_pair"],
    columns="pair_label",
    values="roc_auc",
).reset_index()
_ax_pair_cols = [f"{a}v{b}" for a, b in step_pairs]
_ax_wide.columns.name = None

# Merge identification curves, AX curves, and metadata
_auc_lookup = neurometrics_df[["subject", "electrode_idx", "phoneme_pair", "phon_roc_auc", "smin", "smax"]]
nm_clust = (
    _raw_wide
    .merge(_auc_lookup, on=["subject", "electrode_idx", "phoneme_pair"], how="inner")
    .merge(_ax_wide, on=["subject", "electrode_idx", "phoneme_pair"], how="inner")
)

_step_cols = [f"raw_proba_step{int(s)}" for s in steps_all]
nm_clust = nm_clust.dropna(subset=_step_cols + _ax_pair_cols).copy().reset_index(drop=True)

_curves = nm_clust[_step_cols].values  # (n_sites, 6)

# Remove y-intercept (subtract min) then normalize range for identification curves
_curves_norm = _curves - _curves.min(axis=1, keepdims=True)
_ranges = _curves_norm.max(axis=1, keepdims=True)
_ranges[_ranges == 0] = 1  # avoid divide-by-zero for flat sites
_curves_norm = _curves_norm / _ranges

# AX discrimination: use raw AUC values (absolute level matters)
_ax_curves = nm_clust[_ax_pair_cols].values  # (n_sites, 5)

# Store shape-normalized values for plotting (identification only)
_norm_cols = [f"_cn_step{int(s)}" for s in steps_all]
for _i, _c in enumerate(_norm_cols):
    nm_clust[_c] = _curves_norm[:, _i]

# Cluster on concatenation of normalized identification + raw AX discrimination
_joint = np.hstack([_curves_norm, _ax_curves])  # (n_sites, 11)

# Select k by silhouette
_k_range = range(3, 8)
_sil = []
for _k in _k_range:
    _labels = KMeans(n_clusters=_k, random_state=0, n_init=20).fit_predict(_joint)
    _sil.append(silhouette_score(_joint, _labels))

best_k = list(_k_range)[int(np.argmax(_sil))]
print(f"Best k={best_k}  (silhouette scores: { {k: f'{s:.3f}' for k, s in zip(_k_range, _sil)} })")

km_final = KMeans(n_clusters=best_k, random_state=0, n_init=20)
nm_clust["cluster"] = km_final.fit_predict(_joint)

nm_clust["tmax"] = nm_clust.smax / epoch_sfreq + epoch_tmin

# Sort cluster labels so cluster 0 has the largest mean AUC (cosmetic convenience)
_cl_mean_auc = nm_clust.groupby("cluster")["phon_roc_auc"].mean().sort_values(ascending=False)
_cl_remap = {old: new for new, old in enumerate(_cl_mean_auc.index)}
nm_clust["cluster"] = nm_clust["cluster"].map(_cl_remap)

print(nm_clust.groupby("cluster")[["phon_roc_auc", "smin", "smax"]].agg(["count", "mean", "std"]).round(3))

# %%
import seaborn as sns
sns.displot(data=nm_clust, x="tmax", kind="kde", hue="cluster",
            cut=0)

# %%
# Fig — Cluster counts and AUC score distribution
import seaborn as sns
_cluster_colors = sns.color_palette("tab10", n_colors=best_k)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

_counts = nm_clust["cluster"].value_counts().sort_index()
axes[0].bar(
    _counts.index,
    _counts.values,
    color=[_cluster_colors[i] for i in _counts.index],
    edgecolor="k",
)
axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("Number of sites")
axes[0].set_title("Electrode counts per cluster")
axes[0].set_xticks(_counts.index)

_shared_bins = np.linspace(nm_clust["phon_roc_auc"].min(), nm_clust["phon_roc_auc"].max(), 16)
for _cl in sorted(nm_clust["cluster"].unique()):
    _aucs = nm_clust.loc[nm_clust["cluster"] == _cl, "phon_roc_auc"]
    axes[1].hist(
        _aucs,
        bins=_shared_bins,
        alpha=0.6,
        color=_cluster_colors[_cl],
        edgecolor="k",
        label=f"Cluster {_cl}  (n={len(_aucs)}, mean={_aucs.mean():.2f})",
    )
axes[1].axvline(phon_response_peak_threshold, color="gray", linestyle=":", linewidth=1)
axes[1].set_xlabel("phon_roc_auc")
axes[1].set_ylabel("Count")
axes[1].set_title("Acoustic AUC distribution by cluster")
axes[1].legend(fontsize=8)

plt.tight_layout()
fig.savefig(outdir / "cluster_stats.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved cluster_stats.pdf")

# %%
# Fig — Cluster mean curves (thick black) + sample member curves
# Top row: identification (neurometric) curves; bottom row: AX discrimination curves
_n_sample_per_cluster = 20
_rng_cl = np.random.default_rng(42)
_pair_midpoints = [(a + b) / 2.0 for a, b in step_pairs]

fig, axes = plt.subplots(2, best_k, figsize=(5 * best_k, 7), sharey="row")
if best_k == 1:
    axes = axes.reshape(2, 1)

for _cl in sorted(nm_clust["cluster"].unique()):
    _members = nm_clust[nm_clust["cluster"] == _cl]
    _n_draw = min(_n_sample_per_cluster, len(_members))
    _sampled = _members.sample(n=_n_draw, random_state=int(_rng_cl.integers(1000)))
    _mean_auc = _members["phon_roc_auc"].mean()

    # --- Top row: identification curves ---
    ax = axes[0, _cl]
    for _, _row in _sampled.iterrows():
        _c = np.array([_row[_nc] for _nc in _norm_cols])
        ax.plot(steps_all, _c, "-", color=_cluster_colors[_cl], alpha=0.3, linewidth=0.9)

    _mean_c = np.array([_members[_nc].mean() for _nc in _norm_cols])
    ax.plot(steps_all, _mean_c, "k-", linewidth=2.5, zorder=5, label="cluster mean")

    ax.set_title(
        f"Cluster {_cl}  (n={len(_members)}, AUC={_mean_auc:.2f})", fontsize=9
    )
    if _cl == 0:
        ax.set_ylabel("Identification\n(normalized decoder output)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.5, 6.5)
    ax.legend(fontsize=8)

    # --- Bottom row: AX discrimination curves (raw AUC values) ---
    ax = axes[1, _cl]
    for _, _row in _sampled.iterrows():
        _c = np.array([_row[_p] for _p in _ax_pair_cols])
        ax.plot(_pair_midpoints, _c, "-", color=_cluster_colors[_cl], alpha=0.3, linewidth=0.9)

    _mean_ax = np.array([_members[_p].mean() for _p in _ax_pair_cols])
    ax.plot(_pair_midpoints, _mean_ax, "k-D", linewidth=2.5, markersize=5, zorder=5, label="cluster mean")

    ax.set_xlabel("Step pair midpoint")
    if _cl == 0:
        ax.set_ylabel("AX discrimination\n(ROC-AUC)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xticks(_pair_midpoints)
    ax.set_xticklabels([f"{a}v{b}" for a, b in step_pairs], fontsize=8)
    ax.set_xlim(1.0, 6.0)
    ax.legend(fontsize=8)

fig.suptitle(
    f"Joint identification + discrimination clusters (k={best_k})",
    fontsize=11,
)
plt.tight_layout()
fig.savefig(outdir / "cluster_curves.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved cluster_curves.pdf")

# %%
# Export cluster assignments + key neurometric properties for downstream analysis
# (used by A_neurometrics to relate cluster membership to latency / ROC-AUC)
_cluster_export = (
    nm_clust[["subject", "electrode_idx", "phoneme_pair", "cluster", "phon_roc_auc"]]
    .merge(
        neurometrics_df[
            ["subject", "electrode_idx", "phoneme_pair", "pse", "slope_k", "sigmoid_r2"]
        ],
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="left",
    )
    .merge(
        site_stats[
            [
                "subject",
                "electrode_idx",
                "phoneme_pair",
                "mean_ambig_confidence",
                "confidence_drop_auc",
                "behavior_auc_on_ambig",
            ]
        ],
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="left",
    )
)
_cluster_export.to_parquet(outdir / "neurometrics_clusters.parquet", index=False)
print(f"Saved neurometrics_clusters.parquet: {len(_cluster_export)} sites")
