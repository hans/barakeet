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
# # Ganong decoding: population-level inspection
#
# Compares population-level Ganong decoding results (across-completion behavioral
# decoder) with within-completion behavioral decoding results.
#
# Analyses:
#   1. Distribution of peak Δ ROC-AUC (full − baseline) across all sites
#   2. Ganong decoder performance split by lexical evidence (completion identity):
#      is the Ganong effect symmetric across the two completions?
#   3. Timing of peak Ganong window vs. Δ ROC-AUC
#   4. Ganong peak Δ vs. behavioral peak Δ: do sites that decode within-completion
#      behavior well also decode the Ganong boundary shift well?
#
# Inputs:
#   ganong_peaks.parquet         — peak Ganong window per site (from ganong_decoding_summarize)
#   ganong_predictions.parquet   — trial-level Ganong predictions (from ganong_decoding_summarize)
#   A_final_summary.csv (×N)    — behavioral peak windows per site per subject
#   {subject}_epo.fif            — epoch files for trial metadata (lexical_evidence)

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# %%
from src.data import add_metadata_features
from src.stimuli import POD_dict

# %% tags=["parameters"]
ganong_peaks_path = "outputs/causal5/ganong_decoding/ganong_peaks.parquet"
ganong_predictions_path = "outputs/causal5/ganong_decoding/ganong_predictions.parquet"

behav_summary_paths = list(
    Path("outputs/causal5/behavior_decoding_single_electrode_summarize").glob(
        "*/A_final_summary.csv"
    )
)

all_epochs = list(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))

epoch_tmin = -0.4
epoch_sfreq = 100
window_size = 15
behav_peak_post_offset_s = 0.2

outdir = "outputs/causal5/ganong_decoding_inspect_population"

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load Ganong peaks and predictions

# %%
ganong_peaks = pd.read_parquet(ganong_peaks_path)
ganong_peaks["window_center_s"] = (
    ganong_peaks["smin"] + window_size / 2
) / epoch_sfreq + epoch_tmin

print(f"Ganong peaks: {len(ganong_peaks)} sites")
ganong_peaks.head()

# %%
ganong_predictions = pd.read_parquet(ganong_predictions_path)
print(f"Ganong predictions: {len(ganong_predictions)} rows")

# %% [markdown]
# ## Load behavioral peak summaries (all subjects)

# %%
behav_summaries = []
for p in sorted(behav_summary_paths):
    df = pd.read_csv(p)
    behav_summaries.append(df)

behav_peaks = pd.concat(behav_summaries, ignore_index=True)
print(f"Behavioral peaks: {len(behav_peaks)} (subject × electrode × phoneme_pair × word_end)")
behav_peaks.head()

# %% [markdown]
# ## Load epoch metadata (for lexical_evidence)

# %%
all_md_rows = []
for epoch_path in tqdm(sorted(all_epochs), desc="Loading epoch metadata"):
    subject = re.findall(r"(EC\d+)_epo", str(epoch_path))[0]
    ep = mne.read_epochs(epoch_path, preload=False, verbose=False)
    md = add_metadata_features(ep.metadata).assign(subject=subject)
    del ep
    md.index.name = "epoch_idx"
    all_md_rows.append(
        md.reset_index()[
            ["subject", "epoch_idx", "phoneme_pair", "resampled", "lexical_evidence", "word_end"]
        ]
    )

all_md = pd.concat(all_md_rows, ignore_index=True)
print(f"Epoch metadata: {len(all_md)} trials")

# %% [markdown]
# ---
# ## Analysis 1: Distribution of peak Δ ROC-AUC

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram colored by phoneme pair
ax = axes[0]
for pp, grp in ganong_peaks.groupby("phoneme_pair"):
    ax.hist(grp["diff"], bins=20, alpha=0.5, label=pp)
ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlabel("Peak Δ ROC-AUC (full − baseline)")
ax.set_ylabel("Number of sites")
ax.set_title(f"Ganong peak Δ distribution (n={len(ganong_peaks)})")
ax.legend(title="phoneme_pair")

# Full ROC-AUC distribution
ax = axes[1]
for pp, grp in ganong_peaks.groupby("phoneme_pair"):
    ax.hist(grp["full_roc_auc"], bins=20, alpha=0.5, label=pp)
ax.axvline(0.5, color="k", linestyle="--", linewidth=0.8)
ax.set_xlabel("Peak full ROC-AUC")
ax.set_ylabel("Number of sites")
ax.set_title("Ganong peak performance (full model)")
ax.legend(title="phoneme_pair")

plt.tight_layout()
fig.savefig(outdir / "ganong_peak_distributions.pdf")
plt.show()

# %% [markdown]
# ---
# ## Analysis 2: Performance split by lexical evidence
#
# For each site's peak window, compute ROC-AUC on trials where
# `lexical_evidence == 0` vs `lexical_evidence == 1` separately.
# This tests whether the Ganong effect is symmetric across the two completions.

# %%
# Filter predictions to peak windows
peak_preds = ganong_predictions.merge(
    ganong_peaks[["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]],
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="inner",
)
print(f"Peak-window predictions: {len(peak_preds)} rows")

# Merge in lexical_evidence from epoch metadata
peak_preds = peak_preds.merge(
    all_md[["subject", "epoch_idx", "phoneme_pair", "lexical_evidence"]],
    on=["subject", "epoch_idx", "phoneme_pair"],
    how="left",
)
print(f"After metadata merge: {len(peak_preds)} rows")
print(f"lexical_evidence distribution:\n{peak_preds['lexical_evidence'].value_counts()}")

# %%
# Compute ROC-AUC per site × lexical_evidence, averaging across folds first
site_key = ["subject", "electrode_idx", "phoneme_pair"]

# Ensemble predictions across folds (average proba per trial)
ensembled = (
    peak_preds.groupby(site_key + ["epoch_idx", "lexical_evidence"])[
        ["full_decoder_proba", "baseline_decoder_proba", "decoder_target"]
    ]
    .mean()
    .reset_index()
)

lex_auc_records = []
for (subj, ei, pp), grp in ensembled.groupby(site_key):
    for lex_val in [0, 1]:
        subset = grp[grp["lexical_evidence"] == lex_val]
        if len(subset) < 5 or subset["decoder_target"].nunique() < 2:
            continue

        full_auc = roc_auc_score(subset["decoder_target"], subset["full_decoder_proba"])
        base_auc = roc_auc_score(subset["decoder_target"], subset["baseline_decoder_proba"])

        lex_auc_records.append({
            "subject": subj,
            "electrode_idx": ei,
            "phoneme_pair": pp,
            "lexical_evidence": lex_val,
            "full_roc_auc": full_auc,
            "baseline_roc_auc": base_auc,
            "diff": full_auc - base_auc,
            "n_trials": len(subset),
        })

lex_auc_df = pd.DataFrame(lex_auc_records)
print(f"Lexical-evidence split AUC: {len(lex_auc_df)} rows")

# %%
# Pivot to wide format for scatter plot
lex_wide = lex_auc_df.pivot_table(
    index=site_key,
    columns="lexical_evidence",
    values=["full_roc_auc", "diff"],
).reset_index()
lex_wide.columns = [
    f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
    for col in lex_wide.columns
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Full ROC-AUC: lex_evidence 0 vs 1
ax = axes[0]
ax.scatter(
    lex_wide["full_roc_auc_0"], lex_wide["full_roc_auc_1"],
    alpha=0.3, s=15, edgecolors="none",
)
lim = [
    min(lex_wide["full_roc_auc_0"].min(), lex_wide["full_roc_auc_1"].min()) - 0.02,
    max(lex_wide["full_roc_auc_0"].max(), lex_wide["full_roc_auc_1"].max()) + 0.02,
]
ax.plot(lim, lim, "k--", linewidth=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Full ROC-AUC (lexical_evidence = 0)")
ax.set_ylabel("Full ROC-AUC (lexical_evidence = 1)")
ax.set_title("Ganong decoding: symmetry across completions\n(full model)")
ax.set_aspect("equal")

# Δ ROC-AUC: lex_evidence 0 vs 1
ax = axes[1]
ax.scatter(
    lex_wide["diff_0"], lex_wide["diff_1"],
    alpha=0.3, s=15, edgecolors="none",
)
lim_d = [
    min(lex_wide["diff_0"].min(), lex_wide["diff_1"].min()) - 0.01,
    max(lex_wide["diff_0"].max(), lex_wide["diff_1"].max()) + 0.01,
]
ax.plot(lim_d, lim_d, "k--", linewidth=0.8)
ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlim(lim_d)
ax.set_ylim(lim_d)
ax.set_xlabel("Δ ROC-AUC (lexical_evidence = 0)")
ax.set_ylabel("Δ ROC-AUC (lexical_evidence = 1)")
ax.set_title("Ganong decoding: symmetry across completions\n(neural improvement)")
ax.set_aspect("equal")

plt.tight_layout()
fig.savefig(outdir / "ganong_lexical_evidence_symmetry.pdf")
plt.show()

# %%
# Distribution of asymmetry
lex_wide["full_auc_asymmetry"] = lex_wide["full_roc_auc_1"] - lex_wide["full_roc_auc_0"]
lex_wide["diff_asymmetry"] = lex_wide["diff_1"] - lex_wide["diff_0"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(lex_wide["full_auc_asymmetry"].dropna(), bins=30, color="steelblue", edgecolor="k", alpha=0.8)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
mean_asym = lex_wide["full_auc_asymmetry"].mean()
ax.set_xlabel("Full AUC asymmetry (lex=1 − lex=0)")
ax.set_ylabel("Number of sites")
ax.set_title(f"Full AUC asymmetry (mean={mean_asym:.4f})")

ax = axes[1]
ax.hist(lex_wide["diff_asymmetry"].dropna(), bins=30, color="tomato", edgecolor="k", alpha=0.8)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
mean_asym_d = lex_wide["diff_asymmetry"].mean()
ax.set_xlabel("Δ AUC asymmetry (lex=1 − lex=0)")
ax.set_ylabel("Number of sites")
ax.set_title(f"Δ AUC asymmetry (mean={mean_asym_d:.4f})")

plt.tight_layout()
fig.savefig(outdir / "ganong_lexical_evidence_asymmetry.pdf")
plt.show()

# %% [markdown]
# ---
# ## Analysis 3: Timing vs. peak Δ ROC-AUC

# %%
fig, ax = plt.subplots(figsize=(7, 5))

for pp, grp in ganong_peaks.groupby("phoneme_pair"):
    ax.scatter(
        grp["window_center_s"], grp["diff"],
        alpha=0.4, s=20, label=pp,
    )
    # Mark POD for this phoneme pair
    pod = POD_dict.get(pp)
    if pod is not None:
        ax.axvline(pod, linestyle=":", linewidth=1, alpha=0.6)

ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlabel("Peak window center (s from word onset)")
ax.set_ylabel("Peak Δ ROC-AUC (full − baseline)")
ax.set_title("Ganong peak timing vs. Δ performance")
ax.legend(title="phoneme_pair")
plt.tight_layout()
fig.savefig(outdir / "ganong_timing_vs_delta.pdf")
plt.show()

# %%
ganong_peaks.sort_values("diff", ascending=False).head(10)

# %% [markdown]
# ---
# ## Analysis 4: Ganong Δ vs. behavioral Δ
#
# The behavioral decoder has peaks per (subject, electrode, phoneme_pair, word_end),
# while the Ganong decoder has one peak per (subject, electrode, phoneme_pair).
# We compute both the average and max behavioral Δ across word_end values.

# %%
# Aggregate behavioral peaks: average and max across word_end
behav_agg = (
    behav_peaks.groupby(["subject", "electrode_idx", "phoneme_pair"])["diff"]
    .agg(behav_diff_avg="mean", behav_diff_max="max")
    .reset_index()
)
print(f"Behavioral aggregated peaks: {len(behav_agg)}")

# Merge with Ganong peaks
comparison = ganong_peaks.merge(
    behav_agg,
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="inner",
)
comparison = comparison.rename(columns={"diff": "ganong_diff"})
print(f"Matched sites: {len(comparison)}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, behav_col, title in [
    (axes[0], "behav_diff_avg", "Behavioral Δ (avg across completions)"),
    (axes[1], "behav_diff_max", "Behavioral Δ (max across completions)"),
]:
    for pp, grp in comparison.groupby("phoneme_pair"):
        ax.scatter(
            grp[behav_col], grp["ganong_diff"],
            alpha=0.4, s=20, label=pp,
        )

    # Correlation
    valid = comparison[[behav_col, "ganong_diff"]].dropna()
    r = np.corrcoef(valid[behav_col], valid["ganong_diff"])[0, 1]
    ax.set_xlabel(title)
    ax.set_ylabel("Ganong Δ ROC-AUC")
    ax.set_title(f"Ganong Δ vs. {title}\nr = {r:.3f}, n = {len(valid)}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(title="phoneme_pair", fontsize=8)

plt.tight_layout()
fig.savefig(outdir / "ganong_vs_behavioral_delta.pdf")
plt.show()

# %% [markdown]
# ### Timing comparison: Ganong peak vs. behavioral peak

# %%
# Behavioral peak timing (average across word_end)
behav_timing = (
    behav_peaks.assign(
        window_center_s=lambda df: (df["smin"] + window_size / 2) / epoch_sfreq + epoch_tmin
    )
    .groupby(["subject", "electrode_idx", "phoneme_pair"])["window_center_s"]
    .agg(behav_time_avg="mean", behav_time_max="max")
    .reset_index()
)

timing_comparison = ganong_peaks.merge(
    behav_timing,
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="inner",
)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    timing_comparison["behav_time_avg"],
    timing_comparison["window_center_s"],
    alpha=0.3, s=15, edgecolors="none",
)
lim_t = [
    min(timing_comparison["behav_time_avg"].min(), timing_comparison["window_center_s"].min()) - 0.1,
    max(timing_comparison["behav_time_avg"].max(), timing_comparison["window_center_s"].max()) + 0.1,
]
ax.plot(lim_t, lim_t, "k--", linewidth=0.8)
ax.set_xlim(lim_t)
ax.set_ylim(lim_t)
ax.set_xlabel("Behavioral peak time (avg across completions, s)")
ax.set_ylabel("Ganong peak time (s)")
ax.set_title("Peak timing: Ganong vs. behavioral decoder")
ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(outdir / "ganong_vs_behavioral_timing.pdf")
plt.show()

# %% [markdown]
# ## Summary statistics

# %%
print("=== Ganong peaks ===")
print(ganong_peaks[["full_roc_auc", "diff", "window_center_s"]].describe().round(4))
print()
print("=== By phoneme pair ===")
print(
    ganong_peaks.groupby("phoneme_pair")[["full_roc_auc", "diff", "window_center_s"]]
    .describe()
    .round(4)
)
