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
#   5. Behavioral Ganong effect size per (subject, phoneme_pair): PSE shift between
#      completions from sigmoid psychometric fits
#   6. Neural-behavioral Ganong correlation: does the behavioral Ganong effect
#      predict the neural Ganong effect?
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
from scipy.optimize import curve_fit

from src.data import add_metadata_features
from src.stimuli import PHONEME_PAIR_TO_WORD_ENDS, POD_dict

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
            ["subject", "epoch_idx", "phoneme_pair", "resampled", "lexical_evidence", "word_end", "behavior_dummy_forced"]
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
# ---
# ## Analysis 5: Behavioral Ganong effect (PSE shift between completions)
#
# For each (subject, phoneme_pair), fit a sigmoid psychometric function
# `behavior_dummy_forced ~ resampled` separately for each completion (word_end).
# The behavioral Ganong effect = PSE_lex0 − PSE_lex1, where lex0/lex1 follow
# the ordering in `PHONEME_PAIR_TO_WORD_ENDS`.
# Also compute a simpler non-parametric measure: mean behavioral response
# difference at the most ambiguous steps (3–4).

# %%
# Sigmoid fitting utilities (from acoustic_morphology_on_ambiguous.py)


def sigmoid_model(x, a, x0, k):
    """Sigmoid constrained so f(x0) = 0.5 (x0 is the true PSE)."""
    return a / (1.0 + np.exp(-(x - x0) / k)) + (0.5 - a / 2.0)


def fit_model(func, x, y, p0_list, bounds, maxfev=5000):
    """Try multiple initial guesses, return (best_params, best_rss)."""
    best_rss = np.inf
    best_popt = None
    for p0 in p0_list:
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
            rss = float(np.sum((y - func(x, *popt)) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_popt = popt
        except Exception:
            pass
    return best_popt, best_rss


SIGMOID_P0_LIST = [
    [1.0, 3.5, 0.5],
    [-1.0, 3.5, 0.5],
    [1.0, 3.5, 1.5],
    [-1.0, 3.5, 1.5],
    [1.0, 3.5, 5.0],
    [-1.0, 3.5, 5.0],
]
SIGMOID_BOUNDS = ([-3, 0.5, 0.05], [3, 6.5, 20.0])

# %%
steps_all = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

behav_ganong_records = []

for (subj, pp), grp in all_md.groupby(["subject", "phoneme_pair"]):
    word_ends = PHONEME_PAIR_TO_WORD_ENDS[pp]
    # word_ends[0] = lex0 completion, word_ends[1] = lex1 completion

    pse_by_lex = {}
    sigmoid_params_by_lex = {}
    mean_ambig_by_lex = {}
    fit_success = {}

    for lex_val, we in enumerate(word_ends):
        subset = grp[grp["word_end"] == we]

        # Simple measure: mean behavior at steps 3-4
        ambig = subset[subset["resampled"].isin([3, 4])]
        mean_ambig_by_lex[lex_val] = (
            ambig["behavior_dummy_forced"].mean() if len(ambig) > 0 else np.nan
        )

        # Sigmoid fit: compute mean behavior per step
        step_means = subset.groupby("resampled")["behavior_dummy_forced"].mean()
        step_means = step_means.reindex(steps_all)
        valid = step_means.dropna()

        if len(valid) < 3:
            pse_by_lex[lex_val] = np.nan
            sigmoid_params_by_lex[lex_val] = (np.nan, np.nan, np.nan)
            fit_success[lex_val] = False
            continue

        x = valid.index.values.astype(float)
        y = valid.values

        popt, rss = fit_model(sigmoid_model, x, y, SIGMOID_P0_LIST, SIGMOID_BOUNDS)

        if popt is not None:
            pse_by_lex[lex_val] = float(popt[1])
            sigmoid_params_by_lex[lex_val] = tuple(float(p) for p in popt)
            fit_success[lex_val] = True
        else:
            pse_by_lex[lex_val] = np.nan
            sigmoid_params_by_lex[lex_val] = (np.nan, np.nan, np.nan)
            fit_success[lex_val] = False

    # PSE shift: lex0 - lex1
    pse_shift = pse_by_lex.get(0, np.nan) - pse_by_lex.get(1, np.nan)
    # Simple measure: lex1 - lex0 at ambiguous steps
    # (positive = lex1 completion elicits more "right phoneme" responses)
    simple_ganong = mean_ambig_by_lex.get(1, np.nan) - mean_ambig_by_lex.get(0, np.nan)

    behav_ganong_records.append({
        "subject": subj,
        "phoneme_pair": pp,
        "pse_lex0": pse_by_lex.get(0, np.nan),
        "pse_lex1": pse_by_lex.get(1, np.nan),
        "pse_shift": pse_shift,
        "pse_shift_abs": abs(pse_shift) if not np.isnan(pse_shift) else np.nan,
        "sigmoid_a_lex0": sigmoid_params_by_lex.get(0, (np.nan,))[0],
        "sigmoid_k_lex0": sigmoid_params_by_lex.get(0, (np.nan, np.nan, np.nan))[2],
        "sigmoid_a_lex1": sigmoid_params_by_lex.get(1, (np.nan,))[0],
        "sigmoid_k_lex1": sigmoid_params_by_lex.get(1, (np.nan, np.nan, np.nan))[2],
        "fit_success_lex0": fit_success.get(0, False),
        "fit_success_lex1": fit_success.get(1, False),
        "simple_ganong": simple_ganong,
        "simple_ganong_abs": abs(simple_ganong) if not np.isnan(simple_ganong) else np.nan,
        "mean_behav_lex0_ambig": mean_ambig_by_lex.get(0, np.nan),
        "mean_behav_lex1_ambig": mean_ambig_by_lex.get(1, np.nan),
    })

behav_ganong_df = pd.DataFrame(behav_ganong_records)
both_fit = behav_ganong_df["fit_success_lex0"] & behav_ganong_df["fit_success_lex1"]
print(f"Behavioral Ganong: {len(behav_ganong_df)} (subject × phoneme_pair)")
print(f"  Both sigmoids fit successfully: {both_fit.sum()}")
print(f"  At least one fit failed: {(~both_fit).sum()}")
behav_ganong_df

# %%
# Visualization: PSE shift and simple Ganong measure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for pp, grp in behav_ganong_df[both_fit].groupby("phoneme_pair"):
    ax.scatter(
        [pp] * len(grp), grp["pse_shift"],
        alpha=0.7, s=50, label=pp, zorder=3,
    )
ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlabel("Phoneme pair")
ax.set_ylabel("PSE shift (lex0 − lex1)")
ax.set_title(f"Behavioral Ganong: PSE shift (n={both_fit.sum()})")

ax = axes[1]
for pp, grp in behav_ganong_df.groupby("phoneme_pair"):
    ax.scatter(
        [pp] * len(grp), grp["simple_ganong"],
        alpha=0.7, s=50, label=pp, zorder=3,
    )
ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax.set_xlabel("Phoneme pair")
ax.set_ylabel("P(right) lex1 − P(right) lex0 at steps 3–4")
ax.set_title(f"Behavioral Ganong: simple measure (n={len(behav_ganong_df)})")

plt.tight_layout()
fig.savefig(outdir / "behavioral_ganong_effect.pdf")
plt.show()

# %%
# Validation: correlation between the two behavioral Ganong measures
valid_both = behav_ganong_df[both_fit].dropna(subset=["pse_shift", "simple_ganong"])
r_val = np.corrcoef(valid_both["pse_shift"], valid_both["simple_ganong"])[0, 1]
fig, ax = plt.subplots(figsize=(5, 5))
for pp, grp in valid_both.groupby("phoneme_pair"):
    ax.scatter(grp["pse_shift"], grp["simple_ganong"], alpha=0.7, s=50, label=pp)
ax.set_xlabel("PSE shift (sigmoid)")
ax.set_ylabel("Simple Ganong measure (steps 3–4)")
ax.set_title(f"Two behavioral Ganong measures\nr = {r_val:.3f}, n = {len(valid_both)}")
ax.legend(title="phoneme_pair", fontsize=8)
plt.tight_layout()
fig.savefig(outdir / "behavioral_ganong_measures_comparison.pdf")
plt.show()

# %%
# Example psychometric curves: show sigmoid fits for each (subject, phoneme_pair)
# with the largest |PSE shift|
n_examples = min(6, both_fit.sum())
examples = (
    behav_ganong_df[both_fit]
    .sort_values("pse_shift_abs", ascending=False)
    .head(n_examples)
)

n_cols = min(3, n_examples)
n_rows = int(np.ceil(n_examples / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes_flat = np.atleast_1d(axes).flat

x_curve = np.linspace(1, 6, 100)
for ax_idx, (_, erow) in enumerate(examples.iterrows()):
    ax = axes_flat[ax_idx]
    subj, pp = erow["subject"], erow["phoneme_pair"]
    word_ends = PHONEME_PAIR_TO_WORD_ENDS[pp]

    for lex_val, (we, color) in enumerate(zip(word_ends, ["steelblue", "tomato"])):
        subset = all_md[
            (all_md["subject"] == subj)
            & (all_md["phoneme_pair"] == pp)
            & (all_md["word_end"] == we)
        ]
        step_means = subset.groupby("resampled")["behavior_dummy_forced"].mean()
        ax.plot(step_means.index, step_means.values, "o-", color=color, label=we,
                linewidth=1.5, markersize=5, alpha=0.8)

        # Overlay sigmoid fit
        a_key = f"sigmoid_a_lex{lex_val}"
        k_key = f"sigmoid_k_lex{lex_val}"
        pse_key = f"pse_lex{lex_val}"
        a_val, pse_val, k_val = erow[a_key], erow[pse_key], erow[k_key]
        if not any(np.isnan(v) for v in [a_val, pse_val, k_val]):
            ax.plot(x_curve, sigmoid_model(x_curve, a_val, pse_val, k_val),
                    "--", color=color, linewidth=1.5, alpha=0.6)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"{subj} {pp}\n"
        f"PSE shift = {erow['pse_shift']:.2f}",
        fontsize=9,
    )
    ax.set_xlabel("Morph step")
    if ax_idx % n_cols == 0:
        ax.set_ylabel("P(right phoneme)")
    ax.legend(fontsize=7)

for ax_idx in range(n_examples, len(list(np.atleast_1d(axes).flat))):
    np.atleast_1d(axes).flat[ax_idx].set_visible(False)

fig.suptitle("Largest behavioral Ganong effects: psychometric curves", fontsize=11)
plt.tight_layout()
fig.savefig(outdir / "behavioral_ganong_example_curves.pdf", bbox_inches="tight")
plt.show()

# %%
behav_ganong_df.to_parquet(outdir / "behavioral_ganong.parquet", index=False)
print(f"Saved behavioral_ganong.parquet ({len(behav_ganong_df)} rows)")

# %% [markdown]
# ---
# ## Analysis 6: Neural-behavioral Ganong correlation
#
# The neural Ganong effect (`ganong_diff`) is per (subject, electrode, phoneme_pair).
# The behavioral Ganong effect is per (subject, phoneme_pair).
# We show both a per-electrode view (behavioral effect broadcast) and an
# aggregated view (mean `ganong_diff` per subject × phoneme_pair).

# %%
# Per-electrode view: broadcast behavioral effect to all electrodes
neural_behav = ganong_peaks.rename(columns={"diff": "ganong_diff"}).merge(
    behav_ganong_df[["subject", "phoneme_pair", "pse_shift", "pse_shift_abs",
                     "simple_ganong", "simple_ganong_abs"]],
    on=["subject", "phoneme_pair"],
    how="inner",
)
print(f"Neural-behavioral merge (per-electrode): {len(neural_behav)} sites")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# |PSE shift| vs ganong_diff
ax = axes[0]
valid_nb = neural_behav[["pse_shift_abs", "ganong_diff"]].dropna()
for pp, grp in neural_behav.dropna(subset=["pse_shift_abs"]).groupby("phoneme_pair"):
    ax.scatter(grp["pse_shift_abs"], grp["ganong_diff"], alpha=0.3, s=15, label=pp)
r_nb = np.corrcoef(valid_nb["pse_shift_abs"], valid_nb["ganong_diff"])[0, 1]
ax.set_xlabel("|PSE shift| (behavioral Ganong)")
ax.set_ylabel("Neural Ganong Δ ROC-AUC")
ax.set_title(f"Per-electrode: |PSE shift| vs. neural Ganong\nr = {r_nb:.3f}, n = {len(valid_nb)}")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.legend(title="phoneme_pair", fontsize=8)

# |Simple Ganong| vs ganong_diff
ax = axes[1]
valid_ns = neural_behav[["simple_ganong_abs", "ganong_diff"]].dropna()
for pp, grp in neural_behav.dropna(subset=["simple_ganong_abs"]).groupby("phoneme_pair"):
    ax.scatter(grp["simple_ganong_abs"], grp["ganong_diff"], alpha=0.3, s=15, label=pp)
r_ns = np.corrcoef(valid_ns["simple_ganong_abs"], valid_ns["ganong_diff"])[0, 1]
ax.set_xlabel("|Simple Ganong| (behavioral)")
ax.set_ylabel("Neural Ganong Δ ROC-AUC")
ax.set_title(f"Per-electrode: |simple Ganong| vs. neural Ganong\nr = {r_ns:.3f}, n = {len(valid_ns)}")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.legend(title="phoneme_pair", fontsize=8)

plt.tight_layout()
fig.savefig(outdir / "neural_vs_behavioral_ganong_per_electrode.pdf")
plt.show()

# %%
# Aggregated view: average ganong_diff per (subject, phoneme_pair)
neural_agg = (
    ganong_peaks.rename(columns={"diff": "ganong_diff"})
    .groupby(["subject", "phoneme_pair"])["ganong_diff"]
    .agg(ganong_diff_mean="mean", ganong_diff_max="max", n_electrodes="count")
    .reset_index()
)

neural_behav_agg = neural_agg.merge(
    behav_ganong_df[["subject", "phoneme_pair", "pse_shift", "pse_shift_abs",
                     "simple_ganong", "simple_ganong_abs"]],
    on=["subject", "phoneme_pair"],
    how="inner",
)
print(f"Neural-behavioral merge (aggregated): {len(neural_behav_agg)} (subject × phoneme_pair)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, behav_col, neural_col, xlabel, ylabel in [
    (axes[0], "pse_shift_abs", "ganong_diff_mean",
     "|PSE shift| (behavioral Ganong)", "Mean neural Ganong Δ ROC-AUC"),
    (axes[1], "simple_ganong_abs", "ganong_diff_mean",
     "|Simple Ganong| (behavioral)", "Mean neural Ganong Δ ROC-AUC"),
]:
    valid_agg = neural_behav_agg[[behav_col, neural_col]].dropna()
    for pp, grp in neural_behav_agg.dropna(subset=[behav_col]).groupby("phoneme_pair"):
        ax.scatter(grp[behav_col], grp[neural_col], alpha=0.7, s=50, label=pp)
    r_agg = np.corrcoef(valid_agg[behav_col], valid_agg[neural_col])[0, 1]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Aggregated: r = {r_agg:.3f}, n = {len(valid_agg)}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.legend(title="phoneme_pair", fontsize=8)

plt.tight_layout()
fig.savefig(outdir / "neural_vs_behavioral_ganong_aggregated.pdf")
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
print()
print("=== Behavioral Ganong ===")
print(behav_ganong_df[["pse_shift", "simple_ganong"]].describe().round(4))
print()
print("=== Behavioral Ganong by phoneme pair ===")
print(
    behav_ganong_df.groupby("phoneme_pair")[["pse_shift", "pse_shift_abs", "simple_ganong"]]
    .describe()
    .round(4)
)
print()
print("=== Neural-behavioral correlation (per-electrode) ===")
valid_pse = neural_behav[["pse_shift_abs", "ganong_diff"]].dropna()
print(f"  |PSE shift|:    r = {np.corrcoef(valid_pse['pse_shift_abs'], valid_pse['ganong_diff'])[0, 1]:.4f}, n = {len(valid_pse)}")
valid_simple = neural_behav[["simple_ganong_abs", "ganong_diff"]].dropna()
print(f"  |Simple Ganong|: r = {np.corrcoef(valid_simple['simple_ganong_abs'], valid_simple['ganong_diff'])[0, 1]:.4f}, n = {len(valid_simple)}")
print()
print("=== Neural-behavioral correlation (aggregated) ===")
valid_agg_pse = neural_behav_agg[["pse_shift_abs", "ganong_diff_mean"]].dropna()
print(f"  |PSE shift|:    r = {np.corrcoef(valid_agg_pse['pse_shift_abs'], valid_agg_pse['ganong_diff_mean'])[0, 1]:.4f}, n = {len(valid_agg_pse)}")
valid_agg_simple = neural_behav_agg[["simple_ganong_abs", "ganong_diff_mean"]].dropna()
print(f"  |Simple Ganong|: r = {np.corrcoef(valid_agg_simple['simple_ganong_abs'], valid_agg_simple['ganong_diff_mean'])[0, 1]:.4f}, n = {len(valid_agg_simple)}")
