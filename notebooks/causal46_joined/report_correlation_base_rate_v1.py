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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Report-correlation base-rate test — Variant 1 (synthetic HGA)
#
# **Question**: Under the null (no perceptual signal in neural data), how many of
# the 99 acoustic site×pair cells would be falsely classified as one-sided or
# two-sided perceptual, given the finite sample sizes and trial structure of the
# actual experiment?
#
# **Observed distribution** (from `early_acoustic_window.csv`, B1/B2_aligned_sig):
#   - Absent:     70 cells (neither word-end flagged)
#   - One-sided:  18 cells (exactly one word-end flagged)
#   - Two-sided:  11 cells (both word-ends flagged)
#
# **Null model (Variant 1)**: Replace per-trial HGA with synthetic draws from
# N(µ_step, σ) where µ_step = real per-step mean (preserving acoustic variation)
# and σ = pooled within-site SD (no perceptual signal). Real report labels and
# trial structure are unchanged.
#
# **Classification rule**: For each powered (subject × electrode × phoneme_pair ×
# word_end) cell, fit a scalar HGA metric (mean over a behavioral time window) →
# compute aligned Mann-Whitney AUC (class matched to acoustic tuning) on
# ambiguous trials → flag if AUC exceeds 95th percentile of within-step
# permutation null. Sites with ≥ 1 flagged word-end = perceptual; ≥ 2 = two-sided.
#
# Permutation threshold is precomputed once from real HGA, then reused for all M
# null iterations (valid for AUC because the rank-null distribution depends only
# on cell sizes, not HGA values).

# %%
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import mne
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# Ensure we run from repo root regardless of Jupyter cwd
REPO = Path("/workdir")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from src.data import add_metadata_features, get_ambiguous_resampled_steps
from src.stimuli import POD_dict, OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_tmin, epoch_sfreq  # -0.4, 100

# %% tags=["parameters"]
early_acoustic_window_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/report_correlation_base_rate_v1"

MIN_CLASS_K = 4    # minimum total trials per class across qualifying steps
N_PERM = 1000      # permutation replicates for AUC threshold (per cell)
M_NULL = 1000      # synthetic-null iterations
RNG_SEED = 42

# %%
OUTDIR = Path(outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

# %%
# Epoch-relative sample index for a time (seconds post word-onset)
def t_to_sample(t_s):
    return int(round((t_s - epoch_tmin) * epoch_sfreq))

# Behavioral window per (phoneme_pair, word_end): from POD to word_offset + 200ms
POD_SAMPLE = {pp: t_to_sample(pod) for pp, pod in POD_dict.items()}
OFFSET_SAMPLE = {we: t_to_sample(offset) for we, offset in OFFSET_DICT.items()}

def behav_window(phoneme_pair, word_end):
    """[smin, smax) for the per-trial scalar HGA."""
    smin = POD_SAMPLE[phoneme_pair]
    smax = OFFSET_SAMPLE[word_end] + 20  # +200ms past word offset
    return smin, min(smax, 340)          # cap at epoch end

# Acoustic window: fixed early window for all subjects/sites
ACOUSTIC_SMIN = t_to_sample(0.05)   # 50ms post-onset
ACOUSTIC_SMAX = t_to_sample(0.25)   # 250ms post-onset

print(f"epoch_tmin={epoch_tmin}, epoch_sfreq={epoch_sfreq}")
print(f"Acoustic window: samples {ACOUSTIC_SMIN}–{ACOUSTIC_SMAX} "
      f"({ACOUSTIC_SMIN/epoch_sfreq+epoch_tmin:.3f}–{ACOUSTIC_SMAX/epoch_sfreq+epoch_tmin:.3f}s)")
for pp, pod in POD_dict.items():
    for we in PHONEME_PAIR_TO_WORD_ENDS[pp]:
        s, e = behav_window(pp, we)
        print(f"  {pp}/{we}: behavioral samples {s}–{e} "
              f"({s/epoch_sfreq+epoch_tmin:.3f}–{e/epoch_sfreq+epoch_tmin:.3f}s)")

# %% [markdown]
# ## Observed distribution

# %%
ea = pd.read_csv(early_acoustic_window_path)
ea["n_we_sig"] = ea["B1_aligned_sig"].astype(int) + ea["B2_aligned_sig"].astype(int)
ea["late_category"] = ea["n_we_sig"].map({0: "absent", 1: "one-sided", 2: "two-sided"})

obs_counts = ea["late_category"].value_counts()
print("Observed distribution (early_acoustic_window.csv):")
print(obs_counts)
print(f"  N_sites = {len(ea)}")

OBS_TWOSIDED = int(obs_counts.get("two-sided", 0))
OBS_ONESIDED = int(obs_counts.get("one-sided", 0))
OBS_ABSENT = int(obs_counts.get("absent", 0))

# %% [markdown]
# ## Load epoch metadata (all subjects) for ambiguous step computation
#
# We avoid holding all subjects' data arrays in memory simultaneously.
# Load metadata only first; load data arrays per-subject during extraction.

# %%
subjects = sorted({str(s) for s in ea["subject"].unique()})
print(f"Subjects in early_acoustic_window: {subjects}")

# Collect metadata across all subjects (metadata only, no data arrays)
all_md_parts = []
for subj in tqdm(subjects, desc="loading metadata"):
    ep_path = Path(epoch_dir) / f"{subj}_epo.fif"
    if not ep_path.exists():
        print(f"  WARNING: {ep_path} not found")
        continue
    ep = mne.read_epochs(str(ep_path), verbose=False, preload=False)
    ep.metadata = add_metadata_features(ep.metadata)
    md = ep.metadata[["phoneme_pair", "word_end", "resampled", "behavior_dummy_forced"]].copy()
    md["subject"] = subj
    all_md_parts.append(md)

all_md_pd = pd.concat(all_md_parts, ignore_index=True)
all_md_pl = pl.from_pandas(
    all_md_pd[["subject", "phoneme_pair", "word_end", "resampled", "behavior_dummy_forced"]]
)
del all_md_parts

# Uses ambiguous_response_threshold=2 (minority class ≥ 2 trials per step)
ambig_steps = get_ambiguous_resampled_steps(all_md_pl, ambiguous_response_threshold=2)
print(f"Qualifying ambiguous step groups: {len(ambig_steps)}")

# %% [markdown]
# ## Precompute per-cell data

# %%
def extract_scalar_hga(ep, eidx, pp, word_end):
    """Mean baseline-corrected HGA in behavioral window for all trials of this
    (phoneme_pair × word_end) combination.

    Returns (hga_scalar, steps, classes) as 1-D numpy arrays aligned by trial.
    """
    md = ep.metadata
    pp_mask = (md["phoneme_pair"] == pp).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)

    # baseline correct and pick electrode
    hga_full = (
        ep_pp.copy()
        .apply_baseline((None, 0))
        .get_data(picks=[eidx])
        .squeeze(1)
    )  # shape: (n_pp_trials, n_times)

    smin, smax = behav_window(pp, word_end)
    we_mask = (md_pp["word_end"] == word_end).values
    hga_we = hga_full[we_mask, smin:smax].mean(axis=1)

    steps_we = md_pp.loc[we_mask, "resampled"].values.astype(int)
    classes_we = md_pp.loc[we_mask, "behavior_dummy_forced"].values.astype(int)
    return hga_we, steps_we, classes_we


def acoustic_preferred_class(ep, eidx, pp, word_end):
    """Preferred behavior class aligned with acoustic tuning.

    Compare step-1 vs step-6 mean HGA in the fixed acoustic window; the
    endpoint with higher HGA gives the preferred step, whose modal
    behavior_dummy_forced value is the preferred class. Returns None if tied.
    """
    md = ep.metadata
    pp_mask = (md["phoneme_pair"] == pp).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    we_mask = (md_pp["word_end"] == word_end).values

    hga_full = (
        ep_pp.copy()
        .apply_baseline((None, 0))
        .get_data(picks=[eidx])
        .squeeze(1)
    )

    lo_mask = we_mask & (md_pp["resampled"] == 1).values
    hi_mask = we_mask & (md_pp["resampled"] == 6).values
    if not lo_mask.any() or not hi_mask.any():
        return None

    m_lo = hga_full[lo_mask, ACOUSTIC_SMIN:ACOUSTIC_SMAX].mean()
    m_hi = hga_full[hi_mask, ACOUSTIC_SMIN:ACOUSTIC_SMAX].mean()
    if m_lo == m_hi:
        return None

    pref_mask = hi_mask if m_hi > m_lo else lo_mask
    modal = md_pp.loc[pref_mask, "behavior_dummy_forced"].mode()
    return int(modal.iloc[0]) if len(modal) == 1 else None


# %%
cell_data = []  # one dict per (subject, electrode, phoneme_pair, word_end) cell

# Process one subject at a time to bound memory (~600 MB peak per subject)
for subj in tqdm(subjects, desc="subjects"):
    ep_path = Path(epoch_dir) / f"{subj}_epo.fif"
    if not ep_path.exists():
        continue
    ep = mne.read_epochs(str(ep_path), verbose=False)
    ep.metadata = add_metadata_features(ep.metadata)

    subj_rows = ea[ea["subject"] == subj]

    for _, row in subj_rows.iterrows():
        eidx = int(row["electrode_idx"])
        pp   = row["phoneme_pair"]

        for we_key, we_col in [("B1_word_end", "B1_n_per_class"),
                                ("B2_word_end", "B2_n_per_class")]:
            we = row[we_key]
            n_per_class_ref = row[we_col]

            q_steps = ambig_steps.get((subj, pp, we), [])

            try:
                hga_all, steps_all, classes_all = extract_scalar_hga(ep, eidx, pp, we)
            except Exception as exc:
                print(f"  ERROR extracting {subj} e{eidx} {pp} {we}: {exc}")
                continue

            # Filter to ambiguous trials only
            ambig_mask = np.isin(steps_all, q_steps) if q_steps else np.zeros(len(steps_all), bool)
            hga_amb    = hga_all[ambig_mask]
            steps_amb  = steps_all[ambig_mask]
            classes_amb = classes_all[ambig_mask]

            # Total per-class sample size across qualifying steps
            n_per_class_actual = sum(
                min(int(np.sum((steps_amb == s) & (classes_amb == 0))),
                    int(np.sum((steps_amb == s) & (classes_amb == 1))))
                for s in q_steps
            ) if q_steps else 0

            powered = (len(np.unique(classes_amb)) == 2
                       and n_per_class_actual >= MIN_CLASS_K)

            # Per-step mean HGA (no report conditioning — null model params)
            mu_step = {}
            sigma_vals = []
            for s in (q_steps if q_steps else []):
                s_mask = steps_amb == s
                if s_mask.any():
                    mu_step[s] = float(hga_amb[s_mask].mean())
                    sigma_vals.extend(hga_amb[s_mask].tolist())
            sigma = float(np.std(sigma_vals)) if sigma_vals else 1.0
            sigma = max(sigma, 1e-6)

            try:
                pref_class = acoustic_preferred_class(ep, eidx, pp, we)
            except Exception:
                pref_class = None

            cell_data.append({
                "subject": subj, "electrode_idx": eidx,
                "phoneme_pair": pp, "word_end": we, "we_key": we_key,
                "hga_amb": hga_amb,
                "steps_amb": steps_amb,
                "classes_amb": classes_amb,
                "q_steps": list(q_steps),
                "pref_class": pref_class,
                "mu_step": mu_step,
                "sigma": sigma,
                "powered": powered,
                "n_per_class_actual": n_per_class_actual,
                "n_per_class_ref": n_per_class_ref,
            })

    del ep  # free memory before loading next subject

print(f"\nTotal cells precomputed: {len(cell_data)}")
n_powered = sum(c["powered"] for c in cell_data)
print(f"Powered (≥{MIN_CLASS_K} per class): {n_powered}")
print(f"Underpowered: {len(cell_data) - n_powered}")

# %% [markdown]
# ## Aligned AUC helper

# %%
def aligned_auc(hga, classes, pref_class):
    """AUC for the preferred class having higher HGA.

    pref_class=1 → AUC = P(HGA[class=1] > HGA[class=0])
    pref_class=0 → AUC = 1 - P(HGA[class=1] > HGA[class=0])
    pref_class=None → raw AUC(class=1 as positive)
    """
    if len(np.unique(classes)) < 2:
        return 0.5
    raw = roc_auc_score(classes, hga)
    if pref_class == 0:
        return 1.0 - raw
    return raw  # pref_class==1 or None

# %% [markdown]
# ## Permutation-null AUC thresholds
#
# For each powered cell, permute report labels within step × word_end cells
# (preserving per-step class balance) and compute AUC. The 95th percentile
# of this null gives the per-cell significance threshold.
#
# Key property: for a rank-based test (AUC), the permutation-null distribution
# depends only on (n_class, n_trials) per step, NOT on HGA values. The same
# threshold is therefore valid for synthetic HGA draws in the null model.

# %%
rng = np.random.default_rng(RNG_SEED)

for cell in tqdm(cell_data, desc="perm thresholds"):
    if not cell["powered"]:
        cell["threshold"] = None
        cell["obs_auc_aligned"] = np.nan
        cell["obs_flagged"] = False
        continue

    hga = cell["hga_amb"]
    steps = cell["steps_amb"]
    classes = cell["classes_amb"]
    q_steps = cell["q_steps"]
    pref = cell["pref_class"]

    # Observed aligned AUC
    cell["obs_auc_aligned"] = aligned_auc(hga, classes, pref)

    # Permutation null: shuffle classes within each step
    null_aucs = np.empty(N_PERM)
    perm_cls = classes.copy()
    for i in range(N_PERM):
        for s in q_steps:
            idx = np.where(steps == s)[0]
            perm_cls[idx] = rng.permutation(perm_cls[idx])
        null_aucs[i] = aligned_auc(hga, perm_cls, pref)
        perm_cls[:] = classes  # reset

    cell["threshold"] = float(np.percentile(null_aucs, 95))
    cell["obs_flagged"] = bool(cell["obs_auc_aligned"] > cell["threshold"])

n_obs_flagged = sum(c["obs_flagged"] for c in cell_data)
print(f"Observed flagged (powered) cells: {n_obs_flagged} / {n_powered}")

# %% [markdown]
# ## Observed scalar-AUC classification vs original B1/B2_aligned_sig

# %%
# Map (subject, electrode_idx, phoneme_pair, word_end) → obs_flagged
obs_flag_map = {
    (c["subject"], c["electrode_idx"], c["phoneme_pair"], c["word_end"]): c["obs_flagged"]
    for c in cell_data
}

ea_check = ea.copy()
ea_check["b1_scalar_flag"] = [
    obs_flag_map.get((r.subject, int(r.electrode_idx), r.phoneme_pair, r.B1_word_end), False)
    for _, r in ea_check.iterrows()
]
ea_check["b2_scalar_flag"] = [
    obs_flag_map.get((r.subject, int(r.electrode_idx), r.phoneme_pair, r.B2_word_end), False)
    for _, r in ea_check.iterrows()
]
ea_check["n_scalar_sig"] = ea_check["b1_scalar_flag"].astype(int) + ea_check["b2_scalar_flag"].astype(int)
ea_check["scalar_category"] = ea_check["n_scalar_sig"].map({0: "absent", 1: "one-sided", 2: "two-sided"})

print("Scalar-AUC classification (observed data, simplified window):")
print(ea_check["scalar_category"].value_counts())
print()
print("Crosstab vs original B1/B2_aligned_sig classification:")
print(pd.crosstab(ea_check["late_category"], ea_check["scalar_category"],
                  rownames=["original"], colnames=["scalar-AUC"]))

# %% [markdown]
# ## Variant 1 null: M synthetic-HGA iterations

# %%
# Build fast lookup from ea rows → (cell_B1, cell_B2)
cell_lookup = {
    (c["subject"], c["electrode_idx"], c["phoneme_pair"], c["word_end"]): c
    for c in cell_data
}

null_two = np.zeros(M_NULL, dtype=int)
null_one = np.zeros(M_NULL, dtype=int)
null_absent = np.zeros(M_NULL, dtype=int)

for m in tqdm(range(M_NULL), desc="null iterations"):
    site_flags = {}

    for cell in cell_data:
        if not cell["powered"]:
            site_flags[id(cell)] = False
            continue

        steps = cell["steps_amb"]
        classes = cell["classes_amb"]
        mu_step = cell["mu_step"]
        sigma = cell["sigma"]
        pref = cell["pref_class"]
        thresh = cell["threshold"]

        # Synthetic HGA: N(mu_step[trial_step], sigma), independent of report
        synth = np.array([
            rng.normal(mu_step.get(int(s), 0.0), sigma)
            for s in steps
        ])

        auc = aligned_auc(synth, classes, pref)
        site_flags[id(cell)] = bool(auc > thresh)

    # Count two-sided / one-sided across all 99 rows
    n_two = n_one = n_abs = 0
    for _, row in ea.iterrows():
        k1 = (row["subject"], int(row["electrode_idx"]), row["phoneme_pair"], row["B1_word_end"])
        k2 = (row["subject"], int(row["electrode_idx"]), row["phoneme_pair"], row["B2_word_end"])
        c1 = cell_lookup.get(k1)
        c2 = cell_lookup.get(k2)
        f1 = site_flags.get(id(c1), False) if c1 is not None else False
        f2 = site_flags.get(id(c2), False) if c2 is not None else False
        n = int(f1) + int(f2)
        if n == 2:
            n_two += 1
        elif n == 1:
            n_one += 1
        else:
            n_abs += 1

    null_two[m] = n_two
    null_one[m] = n_one
    null_absent[m] = n_abs

print(f"\nNull complete (M={M_NULL}):")
print(f"  Two-sided: mean={null_two.mean():.2f} ± {null_two.std():.2f}")
print(f"  One-sided: mean={null_one.mean():.2f} ± {null_one.std():.2f}")
print(f"  Absent:    mean={null_absent.mean():.2f} ± {null_absent.std():.2f}")

# %%
# Empirical p-values (one-sided: how often does null ≥ observed?)
p_two = float(np.mean(null_two >= OBS_TWOSIDED))
p_one = float(np.mean(null_one >= OBS_ONESIDED))
p_absent = float(np.mean(null_absent <= OBS_ABSENT))  # absent: observed < null means fewer perceptual

print("\n=== HEADLINE READOUT ===")
print(f"  Observed: {OBS_TWOSIDED} two-sided, {OBS_ONESIDED} one-sided, {OBS_ABSENT} absent")
print(f"  Null mean: {null_two.mean():.1f} two-sided, {null_one.mean():.1f} one-sided, {null_absent.mean():.1f} absent")
print(f"  95th pct of null: {np.percentile(null_two, 95):.1f} two-sided, {np.percentile(null_one, 95):.1f} one-sided")
print(f"  p (null ≥ observed): two-sided p={p_two:.4f}, one-sided p={p_one:.4f}")

# %% [markdown]
# ## Figures

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Base-rate test — Variant 1 null (synthetic HGA ~ N(µ_step, σ))", fontsize=11)

panels = [
    (null_two,   OBS_TWOSIDED, "Two-sided",  "#1b7837"),
    (null_one,   OBS_ONESIDED, "One-sided",  "#762a83"),
    (null_absent, OBS_ABSENT,  "Absent",     "#7fbfff"),
]
for ax, (null_arr, obs_n, label, color) in zip(axes, panels):
    ax.hist(null_arr, bins=range(null_arr.min(), null_arr.max() + 2),
            color=color, alpha=0.65, edgecolor="white", linewidth=0.4)
    ax.axvline(obs_n, color="crimson", lw=2.2, zorder=5,
               label=f"Observed (n={obs_n})")
    ax.set_xlabel(f"N {label} sites")
    ax.set_ylabel("Null iterations" if ax is axes[0] else "")
    ax.set_title(f"{label}\nnull mean={null_arr.mean():.1f}")
    ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUTDIR / "null_distribution.pdf", bbox_inches="tight")
plt.show()
print(f"Saved → {OUTDIR}/null_distribution.pdf")

# %% [markdown]
# ## 2-D histogram: (one-sided, two-sided) joint null

# %%
fig2, ax2 = plt.subplots(figsize=(6, 5))
h = ax2.hist2d(
    null_one, null_two,
    bins=[range(0, null_one.max() + 2), range(0, null_two.max() + 2)],
    cmap="Blues",
)
plt.colorbar(h[3], ax=ax2, label="Null iterations")
ax2.scatter(OBS_ONESIDED, OBS_TWOSIDED, color="crimson", s=120, zorder=5,
            marker="*", label=f"Observed ({OBS_ONESIDED}, {OBS_TWOSIDED})")
ax2.set_xlabel("N one-sided sites")
ax2.set_ylabel("N two-sided sites")
ax2.set_title("Joint null distribution of one-sided and two-sided counts")
ax2.legend(fontsize=9)
plt.tight_layout()
fig2.savefig(OUTDIR / "null_2d.pdf", bbox_inches="tight")
plt.show()
print(f"Saved → {OUTDIR}/null_2d.pdf")

# %% [markdown]
# ## Save null arrays for downstream use

# %%
null_df = pd.DataFrame({
    "null_iter": np.arange(M_NULL),
    "n_two_sided": null_two,
    "n_one_sided": null_one,
    "n_absent": null_absent,
})
null_df.to_parquet(OUTDIR / "null_counts.parquet", index=False)

summary = {
    "obs_two_sided": OBS_TWOSIDED,
    "obs_one_sided": OBS_ONESIDED,
    "obs_absent": OBS_ABSENT,
    "null_two_mean": float(null_two.mean()),
    "null_two_std": float(null_two.std()),
    "null_two_p95": float(np.percentile(null_two, 95)),
    "null_one_mean": float(null_one.mean()),
    "null_one_std": float(null_one.std()),
    "null_one_p95": float(np.percentile(null_one, 95)),
    "p_two_sided": p_two,
    "p_one_sided": p_one,
    "M_NULL": M_NULL,
    "N_PERM": N_PERM,
    "MIN_CLASS_K": MIN_CLASS_K,
    "n_powered_cells": n_powered,
    "n_total_cells": len(cell_data),
}
pd.Series(summary).to_csv(OUTDIR / "summary.csv", header=False)
print(summary)
