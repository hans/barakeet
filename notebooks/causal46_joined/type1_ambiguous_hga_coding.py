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
# # Type-1 sites: HGA coding of ambiguous input (graded vs committed)
#
# **Question**: At type-1 (acoustic-only) sites — where the early acoustic window shows
# no detectable perceptual selectivity — how does the HGA represent ambiguous input?
#
# Three hypotheses, evaluated within the peak acoustic window at qualifying ambiguous steps:
# - **O1 — graded intermediate**: unimodal HGA distribution centred between endpoints (loc ≈ 0.5).
# - **O2a — committed, fixed**: unimodal HGA at an endpoint (loc near 0 or 1), endpoint-like spread.
# - **O2b — committed, trial-varying**: bimodal, both endpoints visited across trials → inflated spread.
#
# **Primary test (O1 vs O2a)**: per qualifying step, `loc = mean(hga_norm)` ∈ [0,1], with
# bootstrap CIs and one-sample t-test against 0.5 (midpoint) and nearer endpoint.
#
# **Secondary test (O2a vs O2b)**: `var_ratio = var(hga_dprime_corr at step)` — already in
# pooled-endpoint-SD² units, so ≈1 → O2a/O1 and ≫1 → O2b. Computed ONCE; not re-divided.
#
# **Corroboration only**: sigmoid k (tuning categoricity across steps) + AX adjacent-step AUC.
#
# **Normalization fix**: existing `hga_endpoint_std` uses combined step-1+step-6 SD (inflated).
# This notebook recomputes `sigma_pooled` as the pooled within-condition SD (ddof=1), giving a
# corrected d-prime ruler used for all d′-scaled axes.
#
# **Cross-completion pooling**: the acoustic test pools across completions because the early
# acoustic window precedes the POD — verified for bm and dn. For pb (POD = 0.21 s), 12 of 15
# type-1 sites have smax > POD; those sites are flagged `confound_possible=True` and excluded
# from the cross-completion no-confound assertion.
#
# **Behavior split** (secondary): within-completion contrast only — epoch metadata merged on
# epoch_idx; one panel per word_end.

# %% tags=["parameters"]
annotations_path    = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
trial_balance_path  = "outputs/causal46_joined/trial_balance_index.csv"
trial_df_path       = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet"
ax_discrimination_path = "outputs/causal46_joined/acoustic_ax_discrimination/ax_discrimination_df_all.parquet"
phon_peaks_path     = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
epoch_dir           = "outputs/epochs_preprocessed"
outdir              = "outputs/causal46_joined/type1_ambiguous_hga_coding"
R_boot              = 2000
high_dprime_threshold = 1.0

# %% [markdown]
# ## Imports

# %%
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne
import numpy as np
import pandas as pd
from loguru import logger as L
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks" / "causal46_joined"))

from src.data import add_metadata_features
from src.models.sigmoid import fit_sigmoid, sigmoid_model_2p, EFFECTIVELY_LINEAR_K
from src.stimuli import POD_dict, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin
from _within_completion import resolve_behavior_col

mne.set_log_level("ERROR")
rng = np.random.default_rng(42)

OUT = Path(outdir)
OUT.mkdir(parents=True, exist_ok=True)

# %%
matplotlib.rcParams.update(
    {
        "figure.dpi": 300,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.25,
        "ytick.minor.width": 0.25,
        "lines.linewidth": 1.0,
        "font.family": "Helvetica",
        "font.sans-serif": ["Helvetica", "Arial"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    }
)

# %% [markdown]
# ## Load type-1 sites

# %%
annotations = pd.read_csv(annotations_path)
type1 = annotations[annotations["site_type_relabel"] == "type1_acoustic_only"].copy()
type1 = type1.reset_index(drop=True)
SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]
L.info(f"Type-1 sites: {len(type1)}")
if len(type1) == 0:
    raise RuntimeError("No type-1 sites found in annotations")

# %%
# Load phon_peaks for (a) confound check and (b) epoch-fallback window
phon_peaks = pd.read_parquet(phon_peaks_path)
# Best window per site (highest AUC among all peaks)
phon_peaks_best = (
    phon_peaks.sort_values("test_roc_auc", ascending=False)
    .drop_duplicates(subset=SITE_KEYS)
    .reset_index(drop=True)
)

# %% [markdown]
# ## POD confound check (per-site, per phoneme pair)
#
# Cross-completion pooling is valid only where the acoustic window ends before the
# point of disambiguation (POD_dict). Verified for bm and dn. For pb (POD=0.21 s),
# many sites have smax_t > 0.21 s — those are flagged `confound_possible=True`.

# %%
type1_with_peaks = type1.merge(
    phon_peaks_best[SITE_KEYS + ["smin", "smax"]],
    on=SITE_KEYS, how="left",
)
type1_with_peaks["smax_t"] = (
    type1_with_peaks["smax"] / epoch_sfreq + epoch_tmin
)
type1_with_peaks["pod"] = type1_with_peaks["phoneme_pair"].map(POD_dict)
type1_with_peaks["confound_possible"] = (
    type1_with_peaks["smax_t"] > type1_with_peaks["pod"]
)

for pp, grp in type1_with_peaks.groupby("phoneme_pair"):
    pod = POD_dict[pp]
    n_ok = (~grp["confound_possible"]).sum()
    n_flag = grp["confound_possible"].sum()
    L.info(f"{pp}: POD={pod:.3f}s — {n_ok} sites window<POD, {n_flag} flagged confound_possible")

# Assertion: bm and dn sites never exceed their POD
for pp in ("bm", "dn"):
    grp = type1_with_peaks[type1_with_peaks["phoneme_pair"] == pp]
    n_viol = grp["confound_possible"].sum()
    assert n_viol == 0, (
        f"{pp}: {n_viol} sites have smax_t > POD — cross-completion pooling not valid"
    )
L.info("Cross-completion pooling valid for bm and dn (confirmed). pb sites with confound flagged.")

# %% [markdown]
# ## Qualifying ambiguous steps (union across completions per site)
#
# `trial_balance_index` has one row per (site × word_end × step). The union of
# `is_ambiguous_step` across word_ends gives the steps that are ambiguous in at
# least one completion.

# %%
tbi = pd.read_csv(trial_balance_path)
# Restrict to type-1 sites
tbi_type1 = tbi.merge(type1[SITE_KEYS], on=SITE_KEYS, how="inner")

qualifying_steps_map = (
    tbi_type1[tbi_type1["is_ambiguous_step"]]
    .groupby(SITE_KEYS)["resampled"]
    .apply(lambda s: sorted(s.unique()))
    .to_dict()
)
L.info(
    f"Qualifying steps coverage: {len(qualifying_steps_map)}/{len(type1)} sites "
    f"have at least one qualifying step"
)

# %% [markdown]
# ## Load trial_df_all and identify coverage

# %%
trial_df_all = pd.read_parquet(trial_df_path)
# Filter to type-1 sites
trial_df = trial_df_all.merge(type1[SITE_KEYS], on=SITE_KEYS, how="inner")
trial_df_site_keys = set(map(tuple, trial_df[SITE_KEYS].drop_duplicates().values))
type1_tuples = set(map(tuple, type1[SITE_KEYS].values))

fallback_sites = type1_tuples - trial_df_site_keys
covered_sites = type1_tuples & trial_df_site_keys
L.info(
    f"trial_df coverage: {len(covered_sites)}/{len(type1)} type-1 sites present; "
    f"{len(fallback_sites)} need epoch fallback"
)
for s in sorted(fallback_sites):
    L.warning(f"Fallback site: {s}")

# %% [markdown]
# ## Epoch fallback: extract HGA for sites not in trial_df_all
#
# For missing sites, we read `smin/smax` from phon_peaks_best, load the epoch file,
# and extract baseline-corrected HGA in that window. Same normalization follows.

# %%
fallback_subjects = {s[0] for s in fallback_sites}
fallback_rows = []

for subject in sorted(fallback_subjects):
    epo_path = Path(epoch_dir) / f"{subject}_epo.fif"
    if not epo_path.exists():
        L.warning(f"{subject}: epoch file not found — epoch fallback sites skipped")
        continue

    subj_fallback = [s for s in fallback_sites if s[0] == subject]
    L.info(f"{subject}: loading epochs for {len(subj_fallback)} fallback site(s)")

    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)
    epochs.apply_baseline()
    data_arr = epochs.get_data()  # (n_trials, n_channels, n_samples)
    md = add_metadata_features(epochs.metadata).reset_index(drop=True)
    md.index.name = "epoch_idx"
    md = md.reset_index()

    for _, electrode_idx, phoneme_pair in subj_fallback:
        row = phon_peaks_best[
            (phon_peaks_best["subject"] == subject)
            & (phon_peaks_best["electrode_idx"] == electrode_idx)
            & (phon_peaks_best["phoneme_pair"] == phoneme_pair)
        ]
        if len(row) == 0:
            L.warning(f"  No phon_peaks entry for {subject}/e{electrode_idx}/{phoneme_pair}")
            continue
        smin_w = int(row.iloc[0]["smin"])
        smax_w = int(row.iloc[0]["smax"])
        ei = electrode_idx

        pp_md = md[md["phoneme_pair"] == phoneme_pair]
        if len(pp_md) == 0:
            L.warning(f"  No {phoneme_pair} epochs for {subject}/e{electrode_idx}")
            continue

        epoch_idxs = pp_md["epoch_idx"].values.astype(int)
        hga_raw = data_arr[epoch_idxs, ei, smin_w:smax_w].mean(axis=1)

        chunk = pd.DataFrame({
            "subject": subject,
            "electrode_idx": ei,
            "phoneme_pair": phoneme_pair,
            "smin": smin_w,
            "smax": smax_w,
            "epoch_idx": epoch_idxs,
            "resampled": pp_md["resampled"].values,
            "hga_raw": hga_raw,
        })
        fallback_rows.append(chunk)
        L.info(f"  Extracted {len(chunk)} trials for {subject}/e{ei}/{phoneme_pair}")

if fallback_rows:
    fallback_df = pd.concat(fallback_rows, ignore_index=True)
    # Compute hga_polarity and hga_norm for fallback sites (same logic as upstream)
    ep_fb = fallback_df[fallback_df["resampled"].isin([1, 6])]
    fb_means = (
        ep_fb.groupby(SITE_KEYS + ["resampled"])["hga_raw"]
        .mean().unstack("resampled")
        .rename(columns={1: "m1", 6: "m6"})
    ).reset_index()
    fb_means["hga_polarity"] = np.where(fb_means["m6"] > fb_means["m1"], 1, -1)
    fb_means["hga_low"] = fb_means[["m1", "m6"]].min(axis=1)
    fb_means["hga_high"] = fb_means[["m1", "m6"]].max(axis=1)
    fallback_df = fallback_df.merge(
        fb_means[SITE_KEYS + ["hga_polarity", "hga_low", "hga_high"]],
        on=SITE_KEYS, how="left",
    )
    _raw_norm = (fallback_df["hga_raw"] - fallback_df["hga_low"]) / (
        fallback_df["hga_high"] - fallback_df["hga_low"]
    )
    fallback_df["hga_norm"] = np.where(
        fallback_df["hga_polarity"] < 0, 1.0 - _raw_norm, _raw_norm
    )
    fallback_df["hga_endpoint_dprime"] = np.nan  # old inflated d′ not available
    fallback_df = fallback_df.drop(columns=["hga_low", "hga_high"])
    fallback_df["coverage"] = "epoch_fallback"
else:
    fallback_df = pd.DataFrame()

# %% [markdown]
# ## Combine trial_df with fallback; annotate coverage and confound flag

# %%
trial_df["coverage"] = "trial_df"

all_cols = SITE_KEYS + ["smin", "smax", "epoch_idx", "resampled",
                         "hga_raw", "hga_polarity", "hga_norm",
                         "hga_endpoint_dprime", "coverage"]

if len(fallback_df) > 0:
    combined = pd.concat(
        [trial_df[all_cols], fallback_df[[c for c in all_cols if c in fallback_df.columns]]],
        ignore_index=True,
    )
else:
    combined = trial_df[all_cols].copy()

combined = combined.merge(
    type1_with_peaks[SITE_KEYS + ["confound_possible"]],
    on=SITE_KEYS, how="left",
)
combined["confound_possible"] = combined["confound_possible"].fillna(False)

L.info(
    f"Combined: {len(combined)} trial rows, "
    f"{combined.groupby(SITE_KEYS).ngroups} sites"
)

# %% [markdown]
# ## Per-site normalization fix: pooled within-condition σ
#
# `hga_endpoint_std` in the upstream parquet uses `.std()` over the combined
# step-1+step-6 pool, which absorbs the between-endpoint gap and inflates the
# denominator. Here we recompute the pooled within-condition SD (ddof=1):
#
#     sigma_pooled = sqrt( ((n1-1)*var1 + (n6-1)*var6) / (n1+n6-2) )
#
# `d_prime_corr = |mean6 - mean1| / sigma_pooled` — corrected d-prime ruler used
# throughout this notebook for all d′-scaled axes.

# %%
site_dprime_rows = []

for site_key, grp in combined.groupby(SITE_KEYS):
    subject, electrode_idx, phoneme_pair = site_key
    ep1 = grp[grp["resampled"] == 1]["hga_raw"].values
    ep6 = grp[grp["resampled"] == 6]["hga_raw"].values
    n1, n6 = len(ep1), len(ep6)
    if n1 < 2 or n6 < 2:
        L.warning(f"{site_key}: insufficient endpoint trials (n1={n1}, n6={n6}), skipping")
        continue

    var1 = float(np.var(ep1, ddof=1))
    var6 = float(np.var(ep6, ddof=1))
    sigma_pooled = float(np.sqrt(((n1 - 1) * var1 + (n6 - 1) * var6) / (n1 + n6 - 2)))
    if sigma_pooled == 0:
        L.warning(f"{site_key}: sigma_pooled=0, skipping")
        continue

    mean1 = float(np.mean(ep1))
    mean6 = float(np.mean(ep6))
    midpoint = (mean1 + mean6) / 2.0
    d_prime_corr = abs(mean6 - mean1) / sigma_pooled
    sigma_ratio = float(np.sqrt(var1) / np.sqrt(var6)) if var6 > 0 else np.nan

    hga_polarity = int(grp["hga_polarity"].iloc[0])
    coverage = str(grp["coverage"].iloc[0])
    confound_possible = bool(grp["confound_possible"].iloc[0])

    _dp_existing_val = grp["hga_endpoint_dprime"].iloc[0]
    d_prime_existing = float(_dp_existing_val) if not pd.isna(_dp_existing_val) else np.nan

    site_dprime_rows.append({
        "subject": subject,
        "electrode_idx": electrode_idx,
        "phoneme_pair": phoneme_pair,
        "n_step1": n1,
        "n_step6": n6,
        "mean1_raw": mean1,
        "mean6_raw": mean6,
        "midpoint_raw": midpoint,
        "sigma_pooled": sigma_pooled,
        "sigma1": float(np.sqrt(var1)),
        "sigma6": float(np.sqrt(var6)),
        "sigma_ratio": sigma_ratio,
        "d_prime_corr": d_prime_corr,
        "d_prime_existing": d_prime_existing,
        "hga_polarity": hga_polarity,
        "coverage": coverage,
        "confound_possible": confound_possible,
    })

site_dprime = pd.DataFrame(site_dprime_rows)
L.info(
    f"site_dprime: {len(site_dprime)} sites, "
    f"d_prime_corr median={site_dprime['d_prime_corr'].median():.2f}, "
    f"sigma_ratio median={site_dprime['sigma_ratio'].median():.3f}"
)

# Sanity check: sigma_ratio far from 1 → unequal variance (pooled SD already handles it)
n_unequal = (site_dprime["sigma_ratio"] < 0.5).sum() + (site_dprime["sigma_ratio"] > 2.0).sum()
if n_unequal > 0:
    L.warning(f"{n_unequal} sites have sigma_ratio outside [0.5, 2.0] — pooled SD is still valid")

# %% [markdown]
# ## Per-trial hga_dprime_corr
#
# `hga_dprime_corr = (hga_raw - midpoint) / sigma_pooled * hga_polarity`
#
# Sign convention (enforced by hga_polarity): /d/-endpoint → negative, /n/-endpoint → positive.
# The variance of hga_dprime_corr at any step equals var(hga_raw) / sigma_pooled² — the variance
# ratio for the spread test. This is computed ONCE below; no further division by sigma_pooled².

# %%
combined = combined.merge(
    site_dprime[SITE_KEYS + ["sigma_pooled", "midpoint_raw", "d_prime_corr"]],
    on=SITE_KEYS, how="inner",
)
combined["hga_dprime_corr"] = (
    (combined["hga_raw"] - combined["midpoint_raw"])
    / combined["sigma_pooled"]
    * combined["hga_polarity"]
)

# Endpoint sanity: var(hga_dprime_corr at endpoints) should be ≈1
for ep_step, label in [(1, "step1"), (6, "step6")]:
    ep_var = combined[combined["resampled"] == ep_step]["hga_dprime_corr"].var(ddof=1)
    L.info(f"Population var(hga_dprime_corr) at {label}: {ep_var:.3f} (expect ≈1)")

# %% [markdown]
# ## Sigmoid fits per site (corroboration: tuning categoricity)

# %%
SIG_STEPS = [1, 2, 3, 4, 5, 6]

sigmoid_rows = []
for site_key, grp in combined.groupby(SITE_KEYS):
    step_means = grp.groupby("resampled")["hga_norm"].mean()
    x = np.array([s for s in SIG_STEPS if s in step_means.index], dtype=float)
    y = np.array([step_means[s] for s in x])
    if len(x) < 4:
        sigmoid_rows.append({"subject": site_key[0], "electrode_idx": site_key[1],
                              "phoneme_pair": site_key[2],
                              "sig_x0": np.nan, "sig_k": np.nan, "sig_r2": np.nan,
                              "sig_effectively_linear": np.nan})
        continue
    result = fit_sigmoid(x, y)
    if result is None:
        sigmoid_rows.append({"subject": site_key[0], "electrode_idx": site_key[1],
                              "phoneme_pair": site_key[2],
                              "sig_x0": np.nan, "sig_k": np.nan, "sig_r2": np.nan,
                              "sig_effectively_linear": np.nan})
    else:
        sigmoid_rows.append({
            "subject": site_key[0], "electrode_idx": site_key[1],
            "phoneme_pair": site_key[2],
            "sig_x0": result["x0"], "sig_k": result["k"],
            "sig_r2": result["r2"],
            "sig_effectively_linear": result["effectively_linear"],
        })

sigmoid_df = pd.DataFrame(sigmoid_rows)
site_dprime = site_dprime.merge(sigmoid_df, on=SITE_KEYS, how="left")

n_cat = (~sigmoid_df["sig_effectively_linear"].isna() &
         ~sigmoid_df["sig_effectively_linear"].astype(bool)).sum()
n_lin = (~sigmoid_df["sig_effectively_linear"].isna() &
         sigmoid_df["sig_effectively_linear"].astype(bool)).sum()
L.info(f"Sigmoid fits: {n_cat} categorical (k<={EFFECTIVELY_LINEAR_K}), {n_lin} linear (k>{EFFECTIVELY_LINEAR_K})")

# %% [markdown]
# ## Save site_dprime.parquet

# %%
site_dprime.to_parquet(OUT / "site_dprime.parquet", index=False)
L.info(f"Saved site_dprime.parquet ({len(site_dprime)} rows)")

# %% [markdown]
# ## Per-step location test (O1 vs O2a) and variance ratio (O2a vs O2b)
#
# **Location** `loc = mean(hga_norm at step)` ∈ [0,1]: SD-invariant, not affected by the
# normalization fix. Bootstrap CIs + two-tailed one-sample t-tests vs 0.5 and vs nearer endpoint.
#
# **Variance ratio** `var_ratio = var(hga_dprime_corr at step)` — already in sigma_pooled² units.
# Expected ≈1 at endpoints; ≫1 → O2b. Computed ONCE (no re-division).
#
# Only sites in `qualifying_steps_map` with at least one qualifying step are included.

# %%
site_dprime_lookup = site_dprime.set_index(SITE_KEYS)["d_prime_corr"]
ambig_step_rows = []

for site_key, grp in combined.groupby(SITE_KEYS):
    qs = qualifying_steps_map.get(site_key, [])
    if not qs:
        continue
    dp_corr = site_dprime_lookup.get(site_key, np.nan)

    for step in qs:
        step_grp = grp[grp["resampled"] == step]
        hga_norm_vals = step_grp["hga_norm"].dropna().values
        hga_dp_vals = step_grp["hga_dprime_corr"].dropna().values
        n = len(hga_norm_vals)
        if n < 3:
            continue

        loc_obs = float(np.mean(hga_norm_vals))

        # Bootstrap CIs for loc
        boot_locs = np.array([
            np.mean(rng.choice(hga_norm_vals, size=n, replace=True))
            for _ in range(R_boot)
        ])
        ci_lo, ci_hi = float(np.percentile(boot_locs, 2.5)), float(np.percentile(boot_locs, 97.5))

        # Two-tailed t-test vs midpoint (0.5)
        t_mid, p_vs_midpoint = stats.ttest_1samp(hga_norm_vals, 0.5)

        # Two-tailed t-test vs nearer endpoint (0 if loc < 0.5, 1 if loc >= 0.5)
        nearer = 0.0 if loc_obs < 0.5 else 1.0
        t_near, p_vs_nearer = stats.ttest_1samp(hga_norm_vals, nearer)

        # Variance ratio (O2b test): var of hga_dprime_corr at this step
        # This is var(hga_raw)/sigma_pooled² — the spread relative to endpoint noise.
        # Computed once; do NOT divide by sigma_pooled² again.
        var_ratio = float(np.var(hga_dp_vals, ddof=1)) if len(hga_dp_vals) >= 2 else np.nan

        ambig_step_rows.append({
            "subject": site_key[0],
            "electrode_idx": site_key[1],
            "phoneme_pair": site_key[2],
            "resampled": step,
            "n": n,
            "loc": loc_obs,
            "loc_ci_lo": ci_lo,
            "loc_ci_hi": ci_hi,
            "p_vs_midpoint": float(p_vs_midpoint),
            "p_vs_nearer_endpoint": float(p_vs_nearer),
            "var_ratio": var_ratio,
            "mean_hga_dprime_corr": float(np.mean(hga_dp_vals)),
            "d_prime_corr": float(dp_corr),
        })

ambig_step_df = pd.DataFrame(ambig_step_rows)

# Merge confound flag from site_dprime
ambig_step_df = ambig_step_df.merge(
    site_dprime[SITE_KEYS + ["confound_possible", "coverage"]],
    on=SITE_KEYS, how="left",
)
ambig_step_df.to_parquet(OUT / "ambiguous_step_stats.parquet", index=False)
L.info(f"Saved ambiguous_step_stats.parquet ({len(ambig_step_df)} rows, {ambig_step_df.groupby(SITE_KEYS).ngroups} sites)")

# Summary of location distribution
loc_all = ambig_step_df["loc"].dropna().values
L.info(
    f"loc across all type-1 sites × qualifying steps: "
    f"mean={loc_all.mean():.3f}, median={np.median(loc_all):.3f}, "
    f"fraction near midpoint (|loc−0.5|<0.15): {(np.abs(loc_all - 0.5) < 0.15).mean():.2f}"
)

high_dp = ambig_step_df[ambig_step_df["d_prime_corr"] >= high_dprime_threshold]
L.info(
    f"High-d′ subset (d_prime_corr >= {high_dprime_threshold}): "
    f"{high_dp.groupby(SITE_KEYS).ngroups} sites, "
    f"{len(high_dp)} site×step observations"
)

# %% [markdown]
# ## Fig: within-step location per site (sorted by d′)
#
# Heatmap: sites × ambiguous steps, colour = loc ∈ [0,1] (white at 0.5 = midpoint).
# Sorted by d_prime_corr descending.

# %%
ambig_pivot = ambig_step_df.pivot_table(
    index=SITE_KEYS, columns="resampled", values="loc"
)
site_dp_indexed = site_dprime.set_index(SITE_KEYS)["d_prime_corr"]
# Sort sites by d_prime_corr descending
site_order_dp = (
    site_dp_indexed.reindex(ambig_pivot.index)
    .sort_values(ascending=False)
)
site_order = site_order_dp.index
ambig_pivot = ambig_pivot.reindex(site_order)
dp_vals = site_order_dp.values
step_cols = sorted(ambig_pivot.columns)
Z = ambig_pivot[step_cols].values

fig_loc_heat, ax = plt.subplots(figsize=(6, max(4, len(site_order) * 0.18)))
im = ax.imshow(
    Z, aspect="auto", vmin=0, vmax=1,
    cmap="RdBu_r", interpolation="nearest",
)
ax.set_xticks(range(len(step_cols)))
ax.set_xticklabels([f"step {s}" for s in step_cols], fontsize=7)
ax.set_yticks(range(len(site_order)))
ax.set_yticklabels(
    [f"{idx[0]}/{idx[1]}/{idx[2]} d′={d:.1f}" for idx, d in zip(site_order, dp_vals)],
    fontsize=5,
)
ax.set_xlabel("Ambiguous step", fontsize=8)
ax.set_ylabel("Site (sorted by d′)", fontsize=8)
ax.set_title("Within-step location — type-1 sites\nloc=0 → /d/ endpoint, loc=1 → /n/ endpoint, loc=0.5 → midpoint",
             fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.6, label="loc")
fig_loc_heat.tight_layout()
fig_loc_heat.savefig(OUT / "location_per_site.pdf", bbox_inches="tight")
plt.close(fig_loc_heat)
L.info("Saved location_per_site.pdf")

# %% [markdown]
# ## Fig: loc histogram (headline O1/O2a read)
#
# Distribution of loc across all type-1 sites × qualifying steps.
# Peaked near 0.5 → O1 (graded intermediate); bimodal at 0/1 → O2a (committed).

# %%
fig_loc_hist, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(
    ambig_step_df["loc"].dropna(), bins=20, range=(0, 1),
    color="#4dac26", edgecolor="white", linewidth=0.4, alpha=0.8,
)
ax.axvline(0.5, color="k", lw=1.2, ls="--", label="midpoint (O1)")
ax.axvline(0.0, color="#2166ac", lw=1.0, ls=":", alpha=0.7, label="/d/ endpoint (O2a)")
ax.axvline(1.0, color="#d73027", lw=1.0, ls=":", alpha=0.7, label="/n/ endpoint (O2a)")
ax.set_xlabel("loc (fraction of endpoint separation)", fontsize=9)
ax.set_ylabel("Site × step count", fontsize=9)
ax.set_title(
    f"Location distribution — type-1 sites × qualifying steps (n={len(ambig_step_df['loc'].dropna())})\n"
    f"Peaked at 0.5 → O1 (graded); bimodal at edges → O2a (committed)",
    fontsize=8,
)
ax.legend(fontsize=7, framealpha=0.7)
fig_loc_hist.tight_layout()
fig_loc_hist.savefig(OUT / "location_histogram.pdf", bbox_inches="tight")
plt.close(fig_loc_hist)
L.info("Saved location_histogram.pdf")

# %% [markdown]
# ## Fig: location × spread scatter (O1/O2a/O2b map)
#
# Per-site means (across qualifying steps): x = mean loc, y = mean var_ratio.
# Colour = d_prime_corr. Quadrant lines at loc=0.5, var_ratio=1.
# High-d′ sites (filled) vs low-d′ (open circles) — spread test only powered for high-d′.

# %%
site_summary = (
    ambig_step_df.groupby(SITE_KEYS)
    .agg(
        mean_loc=("loc", "mean"),
        mean_var_ratio=("var_ratio", "mean"),
        d_prime_corr=("d_prime_corr", "first"),
        confound_possible=("confound_possible", "first"),
    )
    .reset_index()
)

fig_scatter, ax = plt.subplots(figsize=(5, 4.5))
high_dp_mask = site_summary["d_prime_corr"] >= high_dprime_threshold
confound_mask = site_summary["confound_possible"]

sc_high = ax.scatter(
    site_summary.loc[high_dp_mask & ~confound_mask, "mean_loc"],
    site_summary.loc[high_dp_mask & ~confound_mask, "mean_var_ratio"],
    c=site_summary.loc[high_dp_mask & ~confound_mask, "d_prime_corr"],
    cmap="viridis", vmin=0.5, vmax=4,
    s=40, marker="o", zorder=3, label=f"high d′ (≥{high_dprime_threshold})"
)
ax.scatter(
    site_summary.loc[~high_dp_mask & ~confound_mask, "mean_loc"],
    site_summary.loc[~high_dp_mask & ~confound_mask, "mean_var_ratio"],
    c=site_summary.loc[~high_dp_mask & ~confound_mask, "d_prime_corr"],
    cmap="viridis", vmin=0.5, vmax=4,
    s=30, marker="o", facecolors="none", linewidth=0.8,
    zorder=3, label=f"low d′ (<{high_dprime_threshold})"
)
if confound_mask.any():
    ax.scatter(
        site_summary.loc[confound_mask, "mean_loc"],
        site_summary.loc[confound_mask, "mean_var_ratio"],
        c=site_summary.loc[confound_mask, "d_prime_corr"],
        cmap="viridis", vmin=0.5, vmax=4,
        s=35, marker="^", zorder=3, label="pb confound_possible"
    )

ax.axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.6)
ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)

# Annotate quadrants
_kw = dict(fontsize=7, alpha=0.55, ha="center")
ymax_scatter = site_summary["mean_var_ratio"].quantile(0.98) * 1.3
ax.text(0.25, ymax_scatter * 0.92, "O2a\n(committed, fixed)", color="#2166ac", **_kw)
ax.text(0.75, ymax_scatter * 0.92, "O2a\n(other endpoint)", color="#d73027", **_kw)
ax.text(0.5, ymax_scatter * 0.70, "O2b\n(trial-varying)", color="#7b3294", **_kw)
ax.text(0.5, 0.15, "O1\n(graded)", color="#4dac26", **_kw)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, ymax_scatter)
ax.set_xlabel("mean loc (across qualifying steps per site)", fontsize=9)
ax.set_ylabel("mean var_ratio [var(hga_dprime)/σ²_pooled]", fontsize=9)
ax.set_title("Location × spread — type-1 sites\n(O1/O2a/O2b map; var_ratio computed once)", fontsize=8)
plt.colorbar(sc_high, ax=ax, label="d′ corr", shrink=0.7)
ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
fig_scatter.tight_layout()
fig_scatter.savefig(OUT / "location_spread_scatter.pdf", bbox_inches="tight")
plt.close(fig_scatter)
L.info("Saved location_spread_scatter.pdf")

# %% [markdown]
# ## Fig: per-step spread (high-d′ subset)
#
# Violin + strip plot of hga_dprime_corr at each qualifying ambiguous step, restricted to
# high-d′ sites. Horizontal bands at ±1 show the endpoint noise floor (var_ratio=1 → spread
# at ambiguous step equals endpoint spread). Spread inflated beyond ±1 → O2b signature.

# %%
ambig_high_dp = ambig_step_df[
    (ambig_step_df["d_prime_corr"] >= high_dprime_threshold)
]
# For the violin we need trial-level data
combined_high_dp = combined.merge(
    ambig_high_dp[SITE_KEYS + ["resampled"]].drop_duplicates(),
    on=SITE_KEYS + ["resampled"], how="inner",
)

ambig_steps_present = sorted(ambig_high_dp["resampled"].unique())

fig_spread, ax = plt.subplots(figsize=(5, 4))
if len(ambig_steps_present) > 0 and len(combined_high_dp) > 0:
    data_per_step = [
        combined_high_dp.loc[
            combined_high_dp["resampled"] == s, "hga_dprime_corr"
        ].dropna().values
        for s in ambig_steps_present
    ]
    parts = ax.violinplot(
        data_per_step,
        positions=range(len(ambig_steps_present)),
        showmedians=True, showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#9ecae1")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("k")
    parts["cmedians"].set_linewidth(1.2)

    # Individual points (jitter)
    for i, (s, dat) in enumerate(zip(ambig_steps_present, data_per_step)):
        jitter = rng.uniform(-0.08, 0.08, size=min(len(dat), 300))
        ax.scatter(i + jitter[:len(dat)], dat[:300],
                   s=3, alpha=0.25, color="#2166ac", zorder=2)

# Reference bands at ±1 (endpoint noise floor)
ax.axhline(1.0, color="#d73027", lw=1.0, ls="--", alpha=0.7, label="±1 σ_pooled (endpoint floor)")
ax.axhline(-1.0, color="#d73027", lw=1.0, ls="--", alpha=0.7)
ax.axhline(0.0, color="k", lw=0.6, ls=":", alpha=0.5)

ax.set_xticks(range(len(ambig_steps_present)))
ax.set_xticklabels([f"step {s}" for s in ambig_steps_present], fontsize=8)
ax.set_xlabel("Ambiguous step", fontsize=9)
ax.set_ylabel("hga_dprime_corr [σ_pooled units]", fontsize=9)
ax.set_title(
    f"Per-step spread — high-d′ type-1 sites (d′≥{high_dprime_threshold}, "
    f"n={ambig_high_dp.groupby(SITE_KEYS).ngroups} sites)\n"
    "Spread ≈ endpoint floor → O2a/O1; inflated spread → O2b",
    fontsize=8,
)
ax.legend(fontsize=7, framealpha=0.7)
fig_spread.tight_layout()
fig_spread.savefig(OUT / "per_step_spread.pdf", bbox_inches="tight")
plt.close(fig_spread)
L.info("Saved per_step_spread.pdf")

# %% [markdown]
# ## Fig: tuning categoricity (corroboration only)
#
# **Top panel**: population mean ± SEM of hga_norm per step (type-1 sites), with sigmoid
# overlay fit to the population mean. Labelled as corroboration, not primary adjudication.
#
# **Bottom panel**: AX adjacent-step discrimination (2v3, 3v4, 4v5) at type-1 sites.
# Population mean ± SEM of roc_auc.

# %%
ax_df = pd.read_parquet(ax_discrimination_path)
ax_type1 = ax_df.merge(type1[SITE_KEYS], on=SITE_KEYS, how="inner")
ax_ambig = ax_type1[
    ax_type1.apply(
        lambda r: (min(r["step_a"], r["step_b"]) >= 2)
                  and (max(r["step_a"], r["step_b"]) <= 5)
                  and (abs(r["step_a"] - r["step_b"]) == 1),
        axis=1,
    )
].copy()
ax_ambig["pair_label"] = ax_ambig.apply(
    lambda r: f"{int(r['step_a'])}v{int(r['step_b'])}", axis=1
)

fig_cat, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5, 6))

# Top: population neurometric
pop_step_means = combined.groupby("resampled")["hga_norm"].agg(["mean", "sem", "count"])
pop_steps = pop_step_means.index.values
pop_means = pop_step_means["mean"].values
pop_sems = pop_step_means["sem"].values

ax_top.errorbar(
    pop_steps, pop_means, yerr=pop_sems,
    fmt="o", color="#4dac26", ms=5, lw=1.5, label="population mean ± SEM"
)

# Sigmoid overlay on population mean
if len(pop_steps) >= 4:
    sig_pop = fit_sigmoid(pop_steps, pop_means)
    if sig_pop is not None:
        x_smooth = np.linspace(1, 6, 100)
        y_smooth = sigmoid_model_2p(x_smooth, sig_pop["x0"], sig_pop["k"])
        ax_top.plot(x_smooth, y_smooth, color="#7b3294", lw=1.5, ls="--",
                    label=f"sigmoid k={sig_pop['k']:.2f} x0={sig_pop['x0']:.2f} r²={sig_pop['r2']:.2f}")
        cat_label = "categorical" if not sig_pop["effectively_linear"] else "linear"
        ax_top.text(0.02, 0.92, f"Tuning: {cat_label}", transform=ax_top.transAxes,
                    fontsize=7, color="#7b3294")

ax_top.axhline(0.0, color="k", lw=0.5, ls=":")
ax_top.axhline(1.0, color="k", lw=0.5, ls=":")
ax_top.set_xlim(0.5, 6.5)
ax_top.set_ylim(-0.05, 1.05)
ax_top.set_xlabel("Acoustic step", fontsize=8)
ax_top.set_ylabel("hga_norm (population mean)", fontsize=8)
ax_top.set_title("Tuning categoricity (corroboration only)\nDoes NOT adjudicate O1 vs O2a — see location test",
                 fontsize=8)
ax_top.legend(fontsize=7, framealpha=0.7)

# Bottom: AX adjacent-step discrimination
if len(ax_ambig) > 0:
    ax_ambig_grouped = ax_ambig.groupby("pair_label")["roc_auc"].agg(["mean", "sem"]).reset_index()
    pair_order = sorted(ax_ambig_grouped["pair_label"].unique())
    ax_ambig_grouped = ax_ambig_grouped.set_index("pair_label").reindex(pair_order)
    ax_bot.bar(
        range(len(pair_order)), ax_ambig_grouped["mean"],
        yerr=ax_ambig_grouped["sem"],
        color="#9ecae1", edgecolor="k", linewidth=0.5,
        capsize=4, width=0.6,
    )
    # Individual site points
    for i, pair in enumerate(pair_order):
        vals = ax_ambig[ax_ambig["pair_label"] == pair]["roc_auc"].values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax_bot.scatter(i + jitter, vals, s=5, alpha=0.4, color="#2166ac", zorder=3)
    ax_bot.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.6, label="chance (AUC=0.5)")
    ax_bot.set_xticks(range(len(pair_order)))
    ax_bot.set_xticklabels(pair_order, fontsize=8)
    ax_bot.set_ylim(0.3, 1.0)
    ax_bot.set_ylabel("AX ROC-AUC", fontsize=8)
    ax_bot.set_title(f"AX adjacent-step discrimination — type-1 sites (n={ax_ambig.groupby(SITE_KEYS).ngroups} sites)", fontsize=8)
    ax_bot.legend(fontsize=7, framealpha=0.7)
else:
    ax_bot.text(0.5, 0.5, "No AX data for type-1 sites", ha="center", va="center",
                transform=ax_bot.transAxes, fontsize=9)

fig_cat.tight_layout()
fig_cat.savefig(OUT / "tuning_categoricity.pdf", bbox_inches="tight")
plt.close(fig_cat)
L.info("Saved tuning_categoricity.pdf")

# %% [markdown]
# ## Epoch metadata for behavior split
#
# Load metadata from epoch files to retrieve `word_end` and behavior class per trial.
# Merge with `combined` on (subject, epoch_idx).

# %%
all_subjects = sorted(combined["subject"].unique())
behav_frames = []

for subject in all_subjects:
    epo_path = Path(epoch_dir) / f"{subject}_epo.fif"
    if not epo_path.exists():
        L.warning(f"{subject}: epoch file not found — excluded from behavior panel")
        continue
    epochs = mne.read_epochs(str(epo_path), preload=False, verbose=False)
    md = add_metadata_features(epochs.metadata).reset_index(drop=True)
    md.index.name = "epoch_idx"
    md = md.reset_index()
    bhv_col = resolve_behavior_col(md)
    subj_df = md[["epoch_idx", "phoneme_pair", "word_end", bhv_col]].copy()
    subj_df["subject"] = subject
    subj_df = subj_df.rename(columns={bhv_col: "behavior_class"})
    behav_frames.append(subj_df)

if behav_frames:
    behav_md = pd.concat(behav_frames, ignore_index=True)
    combined_behav = combined.merge(
        behav_md[["subject", "epoch_idx", "phoneme_pair", "word_end", "behavior_class"]],
        on=["subject", "epoch_idx", "phoneme_pair"], how="inner",
    )
    L.info(f"Behavior merge: {len(combined_behav)} rows with word_end + behavior_class")
else:
    combined_behav = pd.DataFrame()
    L.warning("No epoch files found — behavior panel will be empty")

# %% [markdown]
# ## Fig: behavior split within-completion (secondary, perceptual claim)
#
# For ambiguous qualifying steps only: distribution of hga_dprime_corr split by reported
# percept, separately per word_end (within-completion constraint).
# Positive hga_dprime_corr → /n/-aligned early encoding; negative → /d/-aligned.

# %%
if len(combined_behav) > 0:
    # Restrict to qualifying ambiguous steps per site — build lookup table for merge
    qs_rows = [
        {"subject": k[0], "electrode_idx": k[1], "phoneme_pair": k[2], "resampled": s}
        for k, steps in qualifying_steps_map.items()
        for s in steps
    ]
    qs_lookup = pd.DataFrame(qs_rows) if qs_rows else pd.DataFrame(
        columns=SITE_KEYS + ["resampled"]
    )
    qs_lookup["_is_qs"] = True
    combined_behav_ambig = combined_behav.merge(
        qs_lookup, on=SITE_KEYS + ["resampled"], how="inner"
    ).drop(columns=["_is_qs"])
    L.info(f"Behavior panel: {len(combined_behav_ambig)} trial rows at qualifying steps")

    word_ends_present = sorted(combined_behav_ambig["word_end"].dropna().unique())
    behav_classes = sorted(combined_behav_ambig["behavior_class"].dropna().unique())
    BEHAV_COLORS = {behav_classes[0]: "#2166ac", behav_classes[-1]: "#d73027"}

    n_we = len(word_ends_present)
    fig_behav, axes = plt.subplots(
        1, n_we, figsize=(3.5 * n_we, 4.5), sharey=True
    )
    if n_we == 1:
        axes = [axes]

    for ax_we, we in zip(axes, word_ends_present):
        we_data = combined_behav_ambig[combined_behav_ambig["word_end"] == we]
        ambig_steps_we = sorted(we_data["resampled"].unique())
        positions = {s: i for i, s in enumerate(ambig_steps_we)}

        for bhv in behav_classes:
            color = BEHAV_COLORS.get(bhv, "gray")
            for step in ambig_steps_we:
                vals = we_data.loc[
                    (we_data["resampled"] == step) & (we_data["behavior_class"] == bhv),
                    "hga_dprime_corr",
                ].dropna().values
                if len(vals) == 0:
                    continue
                x_pos = positions[step] + (0.15 if bhv == behav_classes[-1] else -0.15)
                # Violin
                if len(vals) >= 5:
                    parts = ax_we.violinplot(
                        vals, positions=[x_pos], widths=0.25,
                        showmedians=True, showextrema=False,
                    )
                    for pc in parts["bodies"]:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.45)
                    parts["cmedians"].set_color(color)
                else:
                    ax_we.scatter([x_pos] * len(vals), vals, s=8, alpha=0.5, color=color)

        ax_we.axhline(0, color="k", lw=0.6, ls=":")
        ax_we.axhline(1, color="#d73027", lw=0.8, ls="--", alpha=0.5)
        ax_we.axhline(-1, color="#2166ac", lw=0.8, ls="--", alpha=0.5)
        ax_we.set_xticks(range(len(ambig_steps_we)))
        ax_we.set_xticklabels([f"s{s}" for s in ambig_steps_we], fontsize=7)
        ax_we.set_xlabel("Ambiguous step", fontsize=8)
        if ax_we is axes[0]:
            ax_we.set_ylabel("hga_dprime_corr [σ_pooled units]", fontsize=8)
        ax_we.set_title(f"word_end={we}\n(within-completion)", fontsize=8)
        for bhv, color in BEHAV_COLORS.items():
            ax_we.plot([], [], color=color, label=f"resp={bhv}", lw=2)
        ax_we.legend(fontsize=7, framealpha=0.7)

    fig_behav.suptitle(
        "Behavior split (within-completion) — type-1 sites × qualifying steps\n"
        "hga_dprime_corr > 0 → /n/-aligned encoding, < 0 → /d/-aligned",
        fontsize=8, y=1.01,
    )
    fig_behav.tight_layout()
    fig_behav.savefig(OUT / "behavior_split.pdf", bbox_inches="tight")
    plt.close(fig_behav)
    L.info("Saved behavior_split.pdf")
else:
    L.warning("Behavior panel skipped (no epoch metadata available)")

# %% [markdown]
# ## Summary

# %%
n_sites = len(site_dprime)
n_high_dp = (site_dprime["d_prime_corr"] >= high_dprime_threshold).sum()
n_fallback = (site_dprime["coverage"] == "epoch_fallback").sum()
n_confound = site_dprime["confound_possible"].sum()

L.info(
    f"\n=== TYPE-1 AMBIGUOUS HGA CODING SUMMARY ===\n"
    f"Total type-1 sites: {n_sites}\n"
    f"  trial_df coverage: {n_sites - n_fallback}\n"
    f"  epoch fallback: {n_fallback} (weak-gradient, low d′)\n"
    f"  confound_possible (pb smax>POD): {n_confound}\n"
    f"  high d′ (≥{high_dprime_threshold}): {n_high_dp}\n"
    f"\nLocation (O1 vs O2a): see location_histogram.pdf\n"
    f"  mean loc = {ambig_step_df['loc'].mean():.3f} "
    f"(midpoint=0.5 → O1 if concentrated, edge → O2a)\n"
    f"Spread (O2a vs O2b, high-d′): see per_step_spread.pdf\n"
    f"  mean var_ratio (high-d′) = "
    f"{ambig_step_df[ambig_step_df['d_prime_corr']>=high_dprime_threshold]['var_ratio'].mean():.3f} "
    f"(≈1 → O2a, ≫1 → O2b)\n"
    f"===========================================\n"
)

# %% [markdown]
# ## Demo plots

# %%
plot_subject = 'EC250'
plot_electrode_idx = 216
plot_phoneme_pair = 'pb'

# %%
plot_df = combined.query(f"subject == '{plot_subject}' and electrode_idx == {plot_electrode_idx} and phoneme_pair == '{plot_phoneme_pair}'")
plot_df

# %%
import seaborn as sns
g = sns.catplot(data=plot_df.astype({"resampled": "int"}), x="resampled", y="hga_dprime_corr",
                kind="swarm", height=3, aspect=1.75)
ax = g.ax
ax.plot(plot_df.groupby("resampled")["hga_dprime_corr"].mean().values,
        color="k", lw=3, ls="--", label="mean", zorder=100)
ax.axhline(0, color="k", lw=0.6, ls=":")

ax2 = ax.twinx()
plot_steps = np.array(sorted(plot_ax_df["step_a"].unique())) - 0.5
ax2.plot(plot_steps, plot_ax_df.groupby("step_a")["roc_auc"].mean().values,
         color="C2", lw=1.5, ls="-", label="AX mean AUC")
ax2.errorbar(plot_steps, plot_ax_df.groupby("step_a")["roc_auc"].mean().values,
             yerr=plot_ax_df.groupby("step_a")["roc_auc_std"].mean() / np.sqrt(48),
             fmt="o", color="C2", ms=5, lw=1.5, label="AX mean ± SEM")

ax2.spines["top"].set_visible(False)

ax.set_xlabel("Acoustic step")
ax.set_ylabel("$d'$", color="k", rotation=0, labelpad=10, ha="right")
ax2.set_ylabel("AX discrimination\n(ROC-AUC)", color="C2", rotation=0,
               labelpad=10, ha="left")


# Sigmoid on the d' axis. Mean d' per step, normalized to [0,1] endpoints to match fit_sigmoid's precondition.
row = sigmoid_df.set_index(SITE_KEYS).loc[(plot_subject, plot_electrode_idx, plot_phoneme_pair)]   # the site you're plotting
x0, k = row["sig_x0"], row["sig_k"]

if np.isfinite(x0) and np.isfinite(k):
    # site's d' per step, to anchor the normalized curve back onto the d' axis
    dp = plot_df.groupby("resampled")["hga_dprime_corr"].mean()
    d0, d1 = dp.loc[min(SIG_STEPS)], dp.loc[max(SIG_STEPS)]

    xs = np.linspace(min(SIG_STEPS), max(SIG_STEPS), 200)
    ys = sigmoid_model_2p(xs, x0, k) * (d1 - d0) + d0   # normalized -> d' scale
    xpos = xs - min(SIG_STEPS)                          # step 1 -> swarm position 0
    ax.plot(xpos, ys, color="C1", lw=2, zorder=101,
            label="sigmoid fit")

ax.legend(loc="upper left", frameon=False)

# %%
print("y_tr range:", y_tr_norm.min(), y_tr_norm.max(), "mean:", y_tr_norm.mean())
print("yhat range:", yhat.min(), yhat.max())
