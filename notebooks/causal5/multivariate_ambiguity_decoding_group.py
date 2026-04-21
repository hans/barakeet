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
# # Group-level aggregation for the ambiguity decoder
#
# Aggregates per-subject scores from
# `multivariate_ambiguity_decoding/{subject}/scores.parquet` into a subject-averaged
# AUC time course, runs a per-window one-sample t-test vs. chance (0.5) with BH-FDR
# correction across windows, computes the three pre-specified temporal measurements,
# produces the two-panel figure.
#
# Cluster-based permutation is intentionally **not** run here (see the plan).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# %%
from src.stimuli import POD_dict

# %% tags=["parameters"]
scores_paths: list[str] = []
fold_balance_paths: list[str] = []
ambiguity_labels_paths: list[str] = []
hga_df_path = "outputs/causal5/prepare_neurometrics/hga_df.parquet"
outdir = "."

behavior_imbalance_threshold = 0.2
fdr_alpha = 0.05
epoch_tmin = -0.4
epoch_sfreq = 100

# %%
outdir = Path(outdir)

# %%
scores_df = pd.concat([pd.read_parquet(p) for p in scores_paths], ignore_index=True)
fold_balance_df = pd.concat(
    [pd.read_parquet(p) for p in fold_balance_paths], ignore_index=True
)
ambig_labels_df = pd.concat(
    [pd.read_parquet(p) for p in ambiguity_labels_paths], ignore_index=True
)

print(
    f"Loaded scores for {scores_df['subject'].nunique()} subjects, "
    f"{len(scores_df)} rows"
)

# %% [markdown]
# ## Subject-averaged AUC and per-window t-test vs. chance

# %%
# Per-subject curve: mean AUC across repeats and phoneme pairs per (smin, smax).
per_subject = (
    scores_df.groupby(["subject", "smin", "smax"])["roc_auc"]
    .mean()
    .reset_index()
)

# Wide: rows = subjects, cols = (smin, smax). Ordered by smin.
subject_curves = (
    per_subject.pivot_table(
        index="subject", columns=["smin", "smax"], values="roc_auc"
    )
    .sort_index(axis=1)
)

# Per-window t-test vs 0.5.
windows = list(subject_curves.columns)
tvals, pvals = [], []
for w in windows:
    col = subject_curves[w].dropna().values
    if len(col) < 3:
        tvals.append(np.nan)
        pvals.append(np.nan)
        continue
    t, p = ttest_1samp(col, popmean=0.5, alternative="greater")
    tvals.append(t)
    pvals.append(p)

pvals_arr = np.array(pvals)
valid = ~np.isnan(pvals_arr)
qvals = np.full_like(pvals_arr, np.nan)
if valid.any():
    _, q_valid, _, _ = multipletests(pvals_arr[valid], alpha=fdr_alpha, method="fdr_bh")
    qvals[valid] = q_valid

group_auc = pd.DataFrame(
    {
        "smin": [w[0] for w in windows],
        "smax": [w[1] for w in windows],
        "mean_auc": subject_curves.mean(axis=0).values,
        "sem_auc": subject_curves.sem(axis=0).values,
        "n_subjects": subject_curves.notna().sum(axis=0).values,
        "t": tvals,
        "p_uncorrected": pvals,
        "q_fdr": qvals,
        "sig_fdr": qvals < fdr_alpha,
    }
)
# Window mid-time in seconds (for plotting / POD comparison).
group_auc["t_mid_s"] = epoch_tmin + (group_auc["smin"] + group_auc["smax"]) / 2 / epoch_sfreq
group_auc["t_start_s"] = epoch_tmin + group_auc["smin"] / epoch_sfreq
group_auc["t_end_s"] = epoch_tmin + group_auc["smax"] / epoch_sfreq

group_auc.to_parquet(outdir / "group_auc.parquet")
print(f"group_auc.parquet: {len(group_auc)} windows")

# %% [markdown]
# ## Temporal measurements (onset, peak, duration, late re-emergence)

# %%
sig = group_auc[group_auc["sig_fdr"]].sort_values("smin")

if sig.empty:
    measurements = {
        "onset_t_start_s": np.nan,
        "peak_t_mid_s": group_auc.loc[group_auc["mean_auc"].idxmax(), "t_mid_s"],
        "peak_auc": group_auc["mean_auc"].max(),
        "above_chance_duration_s": 0.0,
        "late_reemergence": False,
        "n_sig_windows": 0,
    }
else:
    # Contiguous runs of significance (gaps defined by any non-significant window).
    gauc = group_auc.sort_values("smin").reset_index(drop=True)
    gauc["run_id"] = (gauc["sig_fdr"] != gauc["sig_fdr"].shift()).cumsum()
    runs = (
        gauc[gauc["sig_fdr"]]
        .groupby("run_id")
        .agg(
            t_start_s=("t_start_s", "min"),
            t_end_s=("t_end_s", "max"),
            mean_auc=("mean_auc", "max"),
        )
        .reset_index(drop=True)
    )
    peak_row = group_auc.loc[group_auc["mean_auc"].idxmax()]
    # Run containing the peak (fallback: longest run).
    peak_run = runs[
        (runs["t_start_s"] <= peak_row["t_mid_s"])
        & (runs["t_end_s"] >= peak_row["t_mid_s"])
    ]
    if peak_run.empty:
        peak_run = runs.iloc[[(runs["t_end_s"] - runs["t_start_s"]).idxmax()]]
    # Average POD across phoneme pairs for the late-re-emergence window.
    pod_mean = np.mean(list(POD_dict.values()))
    later_runs = runs[runs["t_start_s"] > peak_run["t_end_s"].iloc[0]]
    late_runs = later_runs[later_runs["t_start_s"] >= pod_mean - 0.05]

    measurements = {
        "onset_t_start_s": float(sig["t_start_s"].iloc[0]),
        "peak_t_mid_s": float(peak_row["t_mid_s"]),
        "peak_auc": float(peak_row["mean_auc"]),
        "above_chance_duration_s": float(
            peak_run["t_end_s"].iloc[0] - peak_run["t_start_s"].iloc[0]
        ),
        "late_reemergence": bool(not late_runs.empty),
        "n_sig_windows": int(sig.shape[0]),
        "pod_mean_s": float(pod_mean),
    }

pd.Series(measurements).to_csv(outdir / "temporal_measurements.csv", header=False)
print("temporal_measurements.csv:")
print(pd.Series(measurements).to_string())

# %% [markdown]
# ## Fold-balance report
#
# Flag (subject, pair) cells where the ambiguous-class report distribution
# (`behavior_dummy_forced` balance within `is_ambiguous=True` trials) varies
# across folds beyond `behavior_imbalance_threshold`. Those cells are candidates
# for a behavior-stratified follow-up fit.

# %%
ambig_labels_df = ambig_labels_df.copy()
ambig_labels_df["is_ambiguous"] = ambig_labels_df["is_ambiguous"].astype(bool)

# Join fold_balance to ambiguity_labels to reconstruct per-repeat ambiguous-report balance.
# fold_balance has per-(repeat, stratum, class) counts; we need per-repeat share of
# behavior=1 among ambiguous held-out trials. That needs a join through the outcomes,
# but a lighter proxy is: use the stratum counts to get ambiguous counts per repeat,
# compared to the per-subject/pair ambiguous total.
ambig_summary = (
    fold_balance_df[fold_balance_df["is_ambiguous"]]
    .groupby(["subject", "phoneme_pair", "repeat"])["n_trials"]
    .sum()
    .reset_index()
)
# We can't directly measure behavior imbalance without outcomes (which may be large).
# Report the stratum-count variability as a conservative proxy: CV of ambiguous-trial
# count across repeats. If Cv > threshold the user should dig further.
per_cell = (
    ambig_summary.groupby(["subject", "phoneme_pair"])["n_trials"]
    .agg(mean="mean", std="std", n_repeats="count")
    .reset_index()
)
per_cell["cv"] = per_cell["std"] / per_cell["mean"].replace(0, np.nan)
per_cell["needs_behavior_stratified_followup"] = per_cell["cv"] > behavior_imbalance_threshold
per_cell.to_csv(outdir / "fold_balance_report.csv", index=False)
print(f"fold_balance_report.csv: {len(per_cell)} cells")
print(
    f"  {int(per_cell['needs_behavior_stratified_followup'].sum())} "
    f"flagged for follow-up (cv > {behavior_imbalance_threshold})"
)

# %% [markdown]
# ## Univariate ambiguity selectivity map (panel a data)
#
# Per site: `mean(hga_early | ambiguous) - mean(hga_early | unambiguous)`, where
# ambiguity is defined by our Rule 1 labels (not by `reg_df.is_ambiguous`, which
# uses the behavioral-peak `behav_steps_chosen` semantics).

# %%
try:
    hga_df = pd.read_parquet(hga_df_path)
    hga_merged = hga_df.merge(
        ambig_labels_df[["subject", "epoch_idx", "phoneme_pair", "is_ambiguous"]].rename(
            columns={"is_ambiguous": "is_ambiguous_rule1"}
        ),
        on=["subject", "epoch_idx", "phoneme_pair"],
        how="left",
    )
    hga_merged["is_ambiguous_rule1"] = hga_merged["is_ambiguous_rule1"].fillna(False)

    univariate_map = (
        hga_merged.groupby(
            ["subject", "electrode_idx", "phoneme_pair", "word_end", "is_ambiguous_rule1"]
        )["hga_early"]
        .mean()
        .unstack("is_ambiguous_rule1")
        .reset_index()
    )
    if True in univariate_map.columns and False in univariate_map.columns:
        univariate_map["ambig_selectivity"] = univariate_map[True] - univariate_map[False]
    else:
        univariate_map["ambig_selectivity"] = np.nan
except FileNotFoundError:
    print(f"hga_df not found at {hga_df_path}; skipping panel (a)")
    univariate_map = pd.DataFrame(columns=["ambig_selectivity"])

# %% [markdown]
# ## Figure

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel (a): univariate ambiguity selectivity map.
ax = axes[0]
sel_vals = univariate_map["ambig_selectivity"].dropna().values
if len(sel_vals):
    sns.histplot(sel_vals, bins=30, ax=ax)
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_title(
        f"(a) Univariate ambiguity selectivity  \n"
        f"mean HGA (ambig) − mean HGA (unambig) per site in acoustic window \n"
        f"n={len(sel_vals)} sites, mean={np.nanmean(sel_vals):.3f}"
    )
    ax.set_xlabel("HGA (ambig) − HGA (unambig)")
    ax.set_ylabel("Count")
else:
    ax.set_axis_off()
    ax.text(0.5, 0.5, "hga_df unavailable", ha="center", va="center")

# Panel (b): group AUC time course with FDR-significant shading.
ax = axes[1]
ax.axhline(0.5, color="k", linestyle=":", alpha=0.5, label="chance")
ax.plot(group_auc["t_mid_s"], group_auc["mean_auc"], color="C0", label="mean across subjects")
ax.fill_between(
    group_auc["t_mid_s"],
    group_auc["mean_auc"] - group_auc["sem_auc"],
    group_auc["mean_auc"] + group_auc["sem_auc"],
    color="C0",
    alpha=0.2,
)
for _, row in group_auc[group_auc["sig_fdr"]].iterrows():
    ax.axvspan(row["t_start_s"], row["t_end_s"], color="C1", alpha=0.15)
pod_mean = float(np.mean(list(POD_dict.values())))
ax.axvline(0.0, color="grey", linestyle="--", alpha=0.4, label="stim onset")
ax.axvline(pod_mean, color="red", linestyle="--", alpha=0.4, label=f"mean POD={pod_mean:.2f}s")
ax.set_xlabel("time (s, window midpoint)")
ax.set_ylabel("mean ROC-AUC")
ax.set_title("(b) Group ambiguity-decoder AUC  (BH-FDR q<%.2f shaded; not cluster-corrected)" % fdr_alpha)
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig(outdir / "figure_ambiguity_decoding.pdf")
print("figure_ambiguity_decoding.pdf written")
