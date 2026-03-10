# ---
# jupyter:
#   jupytext:
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
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import mne

from src.data import add_metadata_features
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

phon_response_peak_threshold = 0.6

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
acoustic_sites = phon_peaks.filter(pl.col("phon_roc_auc") >= phon_response_peak_threshold)
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
    acoustic_sites.select(
        ["subject", "electrode_idx", "phoneme_pair", "phon_smin", "phon_smax"]
    ),
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="inner",
).filter(
    (pl.col("smin") == pl.col("phon_smin")) & (pl.col("smax") == pl.col("phon_smax"))
)

# Average decoder_proba across CV folds per (site, epoch)
proba_by_trial = outcomes.group_by(
    ["subject", "electrode_idx", "phoneme_pair", "epoch_idx"]
).agg(pl.col("decoder_proba").mean())

print(f"Trials × sites: {len(proba_by_trial)}")

# %% [markdown]
# ## Load epoch metadata

# %%
all_md_rows = []
for epoch_path in tqdm(sorted(all_epochs), desc="Loading epoch metadata"):
    subject = re.findall(r"(EC\d+)_epo", str(epoch_path))[0]
    ep = mne.read_epochs(epoch_path, preload=False, verbose=False)
    md = add_metadata_features(ep.metadata).assign(subject=subject)
    md.index.name = "epoch_idx"
    all_md_rows.append(
        md.reset_index()[
            [
                "subject",
                "epoch_idx",
                "phoneme_pair",
                "resampled",
                "behavior_categorical_forced",
                "word_end",
                "ambiguity",
            ]
        ]
    )

all_md = pl.from_pandas(
    pd.concat(all_md_rows, ignore_index=True)
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)

# %%
trial_df = (
    proba_by_trial.join(
        all_md, on=["subject", "epoch_idx", "phoneme_pair"], how="left"
    )
    .with_columns(
        (pl.col("decoder_proba") - 0.5).abs().alias("confidence"),
        pl.col("resampled").is_in([1, 6]).alias("is_endpoint"),
        pl.col("resampled").is_in([2, 3, 4, 5]).alias("is_ambiguous"),
    )
)

trial_df.write_parquet(outdir / "trial_df.parquet")
print(f"trial_df saved: {len(trial_df)} rows")
trial_df.head()

# %% [markdown]
# ## Per-site statistics

# %%
site_records = []
site_keys = acoustic_sites.select(
    ["subject", "electrode_idx", "phoneme_pair", "phon_roc_auc", "phon_smin", "phon_smax"]
).to_pandas()

for _, row in tqdm(site_keys.iterrows(), total=len(site_keys), desc="Computing site stats"):
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
    ambig_nona = ambig.dropna(subset=["behavior_categorical_forced"])
    behavior_labels = (ambig_nona["behavior_categorical_forced"] > 0).astype(int)
    auc_behavior_on_ambig = roc_auc_score(behavior_labels, ambig_nona["decoder_proba"])

    # Same on endpoints (sanity check: should be ~1.0 for good acoustic sites)
    ep_nona = endpoints.dropna(subset=["behavior_categorical_forced"])
    # On endpoints behavior and acoustic cue are almost always aligned; use acoustic_cue
    # via decoder_proba directly
    ep_behavior_labels = (ep_nona["behavior_categorical_forced"] > 0).astype(int)
    auc_behavior_on_endpoints = roc_auc_score(ep_behavior_labels, ep_nona["decoder_proba"])

    site_records.append(
        {
            "subject": sub,
            "electrode_idx": ei,
            "phoneme_pair": pp,
            "phon_roc_auc": row["phon_roc_auc"],
            "phon_smin": row["phon_smin"],
            "phon_smax": row["phon_smax"],
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

bp = ax.boxplot(conf_by_step, positions=steps, widths=0.6, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, step in zip(bp["boxes"], steps):
    patch.set_facecolor("steelblue" if step in (1, 6) else "lightyellow")
    patch.set_edgecolor("black")

ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="chance confidence (proba=0.5)")
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
lim = [0, site_stats[["mean_endpoint_confidence", "mean_ambig_confidence"]].values.max() * 1.05]
ax.plot(lim, lim, "k--", linewidth=1, label="equal confidence")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Mean confidence on endpoints (steps 1 & 6)")
ax.set_ylabel("Mean confidence on ambiguous trials (steps 2–5)")
ax.set_title("Representational commitment:\nendpoints vs. ambiguous steps")
ax.legend(fontsize=9)
sm = plt.cm.ScalarMappable(cmap="viridis",
    norm=plt.Normalize(site_stats["phon_roc_auc"].min(), site_stats["phon_roc_auc"].max()))
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
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="chance (AUC=0.5)")
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
].dropna(subset=["resampled", "behavior_categorical_forced"])

fig, ax = plt.subplots(figsize=(6, 4))
colors = {-1.0: "steelblue", 1.0: "tomato"}
labels = {-1.0: "Heard /d/ (or /b/ or /p/)", 1.0: "Heard /n/ (or /m/ or /b/)"}
for beh, grp in site_trials.groupby("behavior_categorical_forced"):
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
