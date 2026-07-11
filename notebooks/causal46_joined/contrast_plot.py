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
# # Continuous-time HGA contrast plot (causal46_joined)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import ast
from collections import defaultdict
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _contrast import (  # noqa: E402
    acoustic_endpoint_means,
    aggregate_trajectories,
    behavioral_bootstrap_meandiff,
    oriented_group_band,
    plot_contrast_axis,
    sliding_ttest,
)
from _acoustic_step_bootstrap import per_cell_best  # noqa: E402

from src.stimuli import OFFSET_DICT, POD_dict
from src.viz_provisional import load_epochs_dict

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

# %% tags=["parameters"]
annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
filtered_manifest_path = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
# Endpoint (step6 − step1) bootstrap over ALL acoustic sites — fixes the acoustic
# sign/window from unambiguous trials, independent of the ambiguous data plotted.
a_per_window_all_path = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_all.parquet"

output_dir = "outputs/causal46_joined/contrast_plot"
phoneme_pair = None   # None = aggregate all pairs; "bm"/"dn"/"pb" for per-pair

epochs_dir = "outputs/epochs_preprocessed"
# "annotated": sign-correct using consensus tuning letter from manifest
# "abs":       take absolute value of mean diff (no manifest label needed)
behav_polarity_mode = "annotated"
n_perm = 1000
null_seed = 0

min_class_k = 3
bootstrap_r = 1000
bootstrap_seed = null_seed

# %% [markdown]
# ### Prepare bootstrap results

# %%
# bootstrap estimates of difference evoked by BEHAVIORAL contrasts on ambiguous trials,
# matching acoustic distribution across the behavioral comparison
b4_by_behavior = pd.read_parquet("outputs/causal46_joined/t_tests/b4_bootstrap.parquet") \
    .set_index(["subject", "electrode_idx", "phoneme_pair", "word_end"])

# %%
b4_by_behavior_windows = pd.read_parquet("outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet")

# %%
# bootstrap estimates of difference evoked by acoustic contrasts on ambiguous trials,
# matching behavior distribution across the acoustic comparison
b4_by_acoustic = pd.read_parquet("outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_bootstrap.parquet") \
    .set_index(["subject", "electrode_idx", "phoneme_pair", "word_end"])

# %%
# endpoint (step6 − step1) bootstrap over all acoustic sites, in the early acoustic
# window. Used to fix the acoustic panel's sign/window from UNAMBIGUOUS trials,
# independent of the ambiguous data plotted (see "Plot contrast by acoustics").
a_per_window_all = pl.read_parquet(a_per_window_all_path)

# %% [markdown]
# There is an important mismatch between the two bootstrapping pipelines we need to rectify.
#
# While both are looking for contrasts between blocks of trials at ambiguous acoustic steps, the behavioral contrast can operate *within-step* -- looking for differences in HGA by behavioral report at matched acoustic step -- while the acoustic contrast necessarily operates *between-step* -- needs to have at least two ambiguous acoustic steps to compare.
#
# So the cells that can be used to evaluate acoustic contrasts are a subset of those that can be used to evaluate behavioral contrasts. Let's validate this and explicitly mark the cells that get dropped -- so if there are data issues later on, we can attribute to this known mismatch vs other bugs.

# %%
behavior_site_keys = set(b4_by_behavior.index.to_list())
acoustic_site_keys = set(b4_by_acoustic.index.to_list())

# %%
assert acoustic_site_keys <= behavior_site_keys, "acoustic sites should be a subset of behavior sites"

# %%
missing_acoustic_site_keys = behavior_site_keys - acoustic_site_keys
print(f"Behavioral sites that are missing from acoustic sites: {len(missing_acoustic_site_keys)}")

# %%
PAIR_PHONEMES = {"bm": ("b", "m"), "dn": ("d", "n"), "pb": ("p", "b")}

PDF_SAVEFIG_KWARGS = dict(
    bbox_inches="tight",
    dpi=300,
)

OUT_DIR = Path(output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### Prepare manifest results

# %%
early = pd.read_csv(annotations_path)
manifest = pd.read_csv(filtered_manifest_path)
phon_peaks = pd.read_parquet(phon_peaks_path)

# "etc" is a catch-all for all the untyped cases, which we don't have enough of any one type to justify separate categories for.
# We can break it down into subtypes in the future if we want, but for now it's easier to lump them together and give them a single color.
EARLY_TYPE_GROUPS = {
    "etc": ["A_unsigned", "problematic", "interesting"],
}
# Ordered bottom→top in the Sankey: typed categories first, then "other" above.
EARLY_TYPES = [
    "type1_acoustic_only",
    "type2_early_perceptual",
    "type3_asymmetric",
    "type4_early_perceptual_mirrored",
    "type5_behav_only",
    "etc",
]
EARLY_LABELS = {
    "type1_acoustic_only":              "Acoustic",
    "type2_early_perceptual":           "Perceptual",
    "type3_asymmetric":                 None,#"Acoustic+perceptual\n(one-sided)",
    "type4_early_perceptual_mirrored":  None,#"Acoustic+perceptual\n(mirrored)",
    "type5_behav_only":                 None,#"Perceptual only",
    "etc":                              None,#"Other",
}
EARLY_COLORS = {
    "type1_acoustic_only":             "#4E79A7",
    "type2_early_perceptual":          "#F27200",
    "type3_asymmetric":                "#F28E2B",
    "type4_early_perceptual_mirrored": "#B07AA1",
    "type5_behav_only":                "#E15759",
    "etc":                             "#AAAAAA",
}
# Right column: absent → one-sided → two-sided (bottom to top)
LATE_ORDER  = ["absent", "one-sided", "two-sided"]
LATE_LABELS = {
    "absent":     "Absent",
    "one-sided":  "One-sided",
    "two-sided":  "Two-sided",
}
LATE_COLORS = {
    "absent":    "#BAB0AC",
    "one-sided": "#76B7B2",
    "two-sided": "#4E9F97",
}

CONJUNCTION_CATEGORIES = {
    "Acoustic-only": {
        "early_type": "type1_acoustic_only",
        "late_type":  "absent",
    },
    "Acoustic + integration": {
        "early_type": "type1_acoustic_only",
        "late_type": ["two-sided", "one-sided"],
    },
    "Perceptual": {
        "early_type": "type2_early_perceptual",
        "late_type":  ["two-sided", "one-sided"],
    }
}

CAT_PLOT_ORDER = ["Perceptual", "Acoustic + integration", "Acoustic-only"]


# %%
def _late_category(x: pd.Series) -> str:
    n = int(x.notna().sum())
    if n == 0:
        return "absent"
    elif n == 1:
        return "one-sided"
    else:
        return "two-sided"

late_pres = (
    manifest
    .groupby(["subject", "electrode_idx", "phoneme_pair"])["behav @late"]
    .apply(_late_category)
    .reset_index()
    .rename(columns={"behav @late": "late_category"})
)

merged = early.merge(
    late_pres, on=["subject", "electrode_idx", "phoneme_pair"], how="left"
)

early_category_map = {}
for group, members in EARLY_TYPE_GROUPS.items():
    for m in members:
        early_category_map[m] = group
merged["early_category"] = merged["site_type_relabel"].replace(early_category_map)
merged["late_category"] = merged["late_category"].fillna("absent")

merged["early_label"] = merged["early_category"].replace(EARLY_LABELS)
merged["late_label"] = merged["late_category"].replace(LATE_LABELS)

# Add phon peaks
merged = merged.merge(
    phon_peaks[["subject", "electrode_idx", "phoneme_pair", "test_roc_auc", "smax"]].rename(columns={"test_roc_auc": "phon_peak_roc_auc", "smax": "phon_peak_smax"}),
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="left",
    validate="1:1",
)

# Track the sites that will be dropped because they were assigned a null early_label or a null late_label
dropped_sites = merged[merged["early_label"].isna() | merged["late_label"].isna()][["subject", "electrode_idx", "phoneme_pair"]]
dropped_sites = {tuple(xs) for xs in dropped_sites.values}
merged = merged.dropna(subset=["early_category", "early_label", "late_category", "late_label"])

merged["conjunction_category"] = None
for cat_name, cat_def in CONJUNCTION_CATEGORIES.items():
    early_type = cat_def["early_type"]
    late_type = cat_def["late_type"]
    if isinstance(late_type, str):
        late_type = [late_type]
    merged.loc[
        (merged["early_category"] == early_type) & (merged["late_category"].isin(late_type)),
        "conjunction_category"
    ] = cat_name

print(f"Site×pair cells total: {len(merged)}")
print("\nFlow table (rows=early type, cols=late category):")
ct = (
    merged
    .groupby(["early_category", "late_category"])
    .size()
    .unstack(fill_value=0)
    .reindex(EARLY_TYPES)
    [LATE_ORDER]
)
print(ct)

# %%
# Load epochs (one-time eager load, matches t_tests.py pattern)
epochs_dict = load_epochs_dict(Path(epochs_dir))
print(f"epochs loaded: {sorted(epochs_dict)}")

# %% [markdown]
# ## Plot contrast by behavior

# %%
epoch_tmin = next(iter(epochs_dict.values())).tmin
epoch_sfreq = next(iter(epochs_dict.values())).info["sfreq"]
window_size = 0.05

# %%
# prepare to extract contrast time series per electrode×phoneme_pair×word_end, for plotting
b4_by_behavior_diffs = b4_by_behavior.pivot_table(
    index=["subject", "electrode_idx", "phoneme_pair", "word_end", "replicate"],
    columns=["smin"],
    values=["mean_diff_raw"]
)

# %%
# TODO how to handle sites with multiple behav windows? we'll just take the earliest for now
behavior_plot_guide_df = pd.merge(
    b4_by_behavior_windows
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .first(),
    merged.set_index(["subject", "electrode_idx", "phoneme_pair"]),
    how="outer", left_index=True, right_index=True, indicator=True
)

# Build per-category cell lists for oriented_group_band.
# Cells lacking a behavioral window (word_end is NaN / not in b_windows) are
# excluded: without a discriminative window we cannot define the orientation
# sign, so including them in both observed and null is not possible.
# TODO update above comment about cells lacking behavioral window
cells_per_category = defaultdict(list)
for (subject, electrode_idx, phoneme_pair, word_end), row in behavior_plot_guide_df.iterrows():
    if pd.isna(row.conjunction_category):
        continue
    cells_per_category[row.conjunction_category].append({
        "subject": subject,
        "electrode_idx": int(electrode_idx),
        "phoneme_pair": phoneme_pair,
        "word_end": word_end,
        "smin": int(row.smin) if not pd.isna(row.smin) else None,
        "smax": int(row.smax) if not pd.isna(row.smax) else None,
    })

# %%
# Compute per-category oriented grand means + matched-permutation null bands.
# Each null replicate uses within-step label permutation so per-step trial
# counts are preserved; the sign is recomputed from the permuted trajectory to
# capture the rectification floor.
# NOTE on interpretation: the null calibrates the orientation (rectification)
# bias, not selection bias. Groups selected on "behav @late" will still show
# late-window observed > null by construction because that is the selection
# criterion; the null band is informative primarily outside the selection window
# and for groups (Acoustic-only) that were NOT selected on behavioral criteria.
ep_times = next(iter(epochs_dict.values())).times
behav_band_results = {}
for cat, cells in tqdm(cells_per_category.items()):
    if any(cell["smin"] is None for cell in cells):
        # We are missing sample bounds here -- this means we don't
        # have a valid behavioral window. So we can't re-orient based on
        # behavioral window; instead we will just take absolute value of
        # the mean difference.
        # TODO implement
    else:
        obs_mean, obs_sem, null_mat, n_valid = oriented_group_band(
            cells, epochs_dict,
            n_perm=n_perm, seed=null_seed,
            min_class_k=min_class_k,
            bootstrap_r=bootstrap_r,
            bootstrap_seed=bootstrap_seed,
        )
    behav_band_results[cat] = (obs_mean, obs_sem, null_mat, n_valid)
    print(f"{cat}: {n_valid} valid cells")

# %%
fig_bh, ax_bh = plt.subplots(figsize=(3, 2))
ax_bh.axvline(0, color="k", linestyle="--", alpha=0.5)
ax_bh.axhline(0, color="k", linestyle="--", alpha=0.5)

_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, cat in enumerate(CAT_PLOT_ORDER):
    if cat not in behav_band_results:
        print(f"Warning: {cat} not found in behav_band_results")
        continue
    obs_mean, obs_sem, null_mat, n_valid = behav_band_results[cat]

    if obs_mean is None:
        continue
    color = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
    label = cat.replace(" + ", "\n+ ")
    ax_bh.plot(ep_times, obs_mean, color=color, lw=2, label=label)

    if obs_sem is not None:
        ax_bh.fill_between(ep_times, obs_mean - obs_sem, obs_mean + obs_sem,
                           color=color, alpha=0.3, lw=0)

    null_lo = np.percentile(null_mat, 2.5, axis=0)
    null_hi = np.percentile(null_mat, 97.5, axis=0)

    # Mark timepoints where observed exits the null band
    sig_mask = (obs_mean > null_hi) | (obs_mean < null_lo)
    if sig_mask.any():
        sig_y = np.where(sig_mask, obs_mean, np.nan)
        sig_y_change = np.diff((~np.isnan(sig_y)).astype(int))

        # Find continuous runs of significant points
        run_starts = np.where(sig_y_change == 1)[0] + 1
        run_ends = np.where(sig_y_change == -1)[0] + 1

        if len(run_starts) > len(run_ends):
            run_ends = np.append(run_ends, len(sig_y))
        
        ymin, ymax = ax_bh.get_ylim()
        bar_h = 0.02
        bar_y = 0.98 - i * bar_h * 1.5
        from matplotlib.transforms import blended_transform_factory
        for start, end in zip(run_starts, run_ends):
            start_time, end_time = ep_times[[start, end - 1]]
            ax_bh.barh(y=bar_y, width=end_time - start_time,
                       left=start_time, height=bar_h,
                       color=color, alpha=0.6,
                       edgecolor="none", zorder=5,
                       transform=blended_transform_factory(ax_bh.transData, ax_bh.transAxes))

        # ax_bh.scatter(ep_times[sig_mask], sig_y[sig_mask], color=color,
        #               s=6, zorder=5, linewidths=0)

ax_bh.legend(loc="lower left", bbox_to_anchor=(-0.8, -0.2))
ax_bh.set_xlim(-0.05, 0.8)
ax_bh.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
ax_bh.set_xlabel("Time (s)")
ax_bh.set_ylabel("HGA contrast by\nperceptual state\n($z$)", rotation=0, labelpad=10, ha="right")
sns.despine(ax=ax_bh)

fig_bh.savefig(OUT_DIR / "behavioral_null_band.pdf", bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Plot contrast by acoustics
#
# Unlike the behavioral panel, the acoustic window and sign are fixed from the
# **unambiguous endpoint** contrast (step6 − step1) measured in the early
# acoustic window by `acoustic_bootstrap.py` — independent of the ambiguous data
# plotted here. Each site is oriented by `sign(median(step6 − step1))` at its best
# endpoint window, and kept only if that window is reliable (bootstrap CI excludes
# zero). We then plot the ambiguous, behavior-balanced acoustic contrast
# (`b4_by_acoustic`, s_hi − s_lo) under that fixed orientation. Positive ⇒ the
# ambiguous acoustic contrast runs in the same direction as the clean endpoint
# tuning — an out-of-sample test of acoustic-code consistency. The
# `ad_windows`/`b4_by_acoustic_windows` (ambiguous-derived) contrast is
# deliberately NOT used to select or orient here: doing so would double-dip.

# %%
b4_by_acoustic_diffs = b4_by_acoustic.pivot_table(
    index=["subject", "electrode_idx", "phoneme_pair", "word_end", "replicate"],
    columns=["smin"],
    values=["mean_diff_raw"]
)

# %%
# Endpoint-derived sign + selection gate, per site (pooled over word ends).
# `a_per_window_all` is loaded up in "Prepare bootstrap results".
# `per_cell_best` picks the largest-|median| endpoint window per site;
# `best_ci_aligned_excludes_zero` is the reliability gate; the sign of
# `best_mean_diff_aligned_med` (= median(step6 − step1)) is the acoustic tuning.
a_best = per_cell_best(
    a_per_window_all, ["subject", "electrode_idx", "phoneme_pair"]
).to_pandas()
a_best["acoustic_sign_endpoint"] = np.sign(a_best["best_mean_diff_aligned_med"])

endpoint_sign = (
    a_best[a_best["best_ci_aligned_excludes_zero"]]
    .set_index(["subject", "electrode_idx", "phoneme_pair"])["acoustic_sign_endpoint"]
)
print(f"acoustic sites with a reliable endpoint window: {len(endpoint_sign)}")

# %%
# Guide: sites with (i) a conjunction category from the manifest and (ii) a
# reliable endpoint sign. Sign/window come from endpoints; category from manifest.
acoustic_plot_guide_df = (
    merged
    .set_index(["subject", "electrode_idx", "phoneme_pair"])
    .join(endpoint_sign, how="inner")
)

# %%
acoustic_contrasts = defaultdict(list)
n_no_endpoint = 0
n_no_category = 0
acoustic_cells = b4_by_acoustic_diffs.index.droplevel("replicate").unique()
for (subject, electrode_idx, phoneme_pair, word_end) in acoustic_cells:
    site_key = (subject, electrode_idx, phoneme_pair)
    if site_key not in acoustic_plot_guide_df.index:
        # No reliable endpoint acoustic window (or no manifest row) → can't orient.
        n_no_endpoint += 1
        continue
    row = acoustic_plot_guide_df.loc[site_key]
    if pd.isna(row.conjunction_category):
        n_no_category += 1
        continue

    # which word end should we attend to? if we're looking at a one-sided site, we should only look at the word end that has a behavioral window. if we're looking at a two-sided site, we should look at both word ends and average them together. if we're looking at an acoustic-only site, we should look at both word ends and average them together.
    if row.late_category == "one-sided":
        target_word_ends = manifest.loc[
            (
                (manifest.subject == subject)
                & (manifest.electrode_idx == electrode_idx)
                & (manifest.phoneme_pair == phoneme_pair)
                & (manifest["behav @late"].notna())
            ),
            "word_end"
        ].unique()
        if word_end not in target_word_ends:
            continue

    b_diffs = b4_by_acoustic_diffs.loc[(subject, electrode_idx, phoneme_pair, word_end)].values
    b_diff_mean = (row.acoustic_sign_endpoint * b_diffs).mean(0)
    acoustic_contrasts[row.conjunction_category].append(b_diff_mean)

print(f"acoustic cells dropped (no reliable endpoint sign): {n_no_endpoint}")
print(f"acoustic cells dropped (no conjunction category):   {n_no_category}")
for cat, contrasts in acoustic_contrasts.items():
    print(f"  {cat}: {len(contrasts)} cells")

# %% [markdown]
# ### Acoustic null band (sign-flip permutation)
#
# For each category, the observed grand mean is `mean_c[endpoint_sign_c ×
# contrast_c(t)]`; the null flips each cell's `endpoint_sign` independently
# per replicate.  The null is centered at zero — **no rectification floor** —
# because the orientation source (endpoint unambiguous data) is disjoint from
# the tested quantity (ambiguous-trial acoustic contrast).  Stars mark where
# the observed curve exits the 2.5–97.5 th percentile band.
#
# Non-circularity: selection (endpoint reliability gate + morphology type) is
# on unambiguous endpoint data; the tested quantity is the behavior-balanced
# acoustic contrast on ambiguous trials.  The sign-flip significance is
# therefore non-circular, unlike the behavioral panel where selection and
# tested quantity share data.

# %%
# Compute sign-flip null bands for each category.
# cell_trajectories are the already-oriented per-cell contrasts (endpoint_sign
# baked in); no epoch reload needed.
xs = b4_by_acoustic_diffs.columns.get_level_values("smin").values / epoch_sfreq + epoch_tmin + window_size / 2
acoustic_band_results = {}
for cat, contrasts in acoustic_contrasts.items():
    obs_mean, obs_sem, null_mat, n_valid = oriented_group_band(
        null_mode="sign_flip",
        cell_trajectories=contrasts,
        n_perm=n_perm,
        seed=null_seed,
    )
    acoustic_band_results[cat] = (obs_mean, null_mat, n_valid)
    print(f"{cat}: {n_valid} valid cells")

# %%
f, ax = plt.subplots(figsize=(3, 2))

ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.axhline(0, color="k", linestyle="--", alpha=0.5)

for i, cat in enumerate(CAT_PLOT_ORDER):
    if cat not in acoustic_band_results:
        continue
    obs_mean, null_mat, n_valid = acoustic_band_results[cat]
    if obs_mean is None:
        continue
    color = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]

    ax.plot(xs, obs_mean, color=color, lw=1.5,
            label=f"{cat.replace(' + ', chr(10) + '+ ')} (n={n_valid})")

    null_lo = np.percentile(null_mat, 2.5, axis=0)
    null_hi = np.percentile(null_mat, 97.5, axis=0)
    ax.fill_between(xs, null_lo, null_hi, color=color, alpha=0.18)

    sig_mask = (obs_mean > null_hi) | (obs_mean < null_lo)
    if sig_mask.any():
        ax.scatter(xs[sig_mask], obs_mean[sig_mask],
                   color=color, s=6, zorder=5, linewidths=0)

ax.legend(loc="lower left", bbox_to_anchor=(-0.8, -0.2))
ax.set_xlim(-0.05, 0.8)
ax.set_xlabel("Time from word onset (s)")
ax.set_ylabel("HGA contrast by\nacoustic input\n($z$)", rotation=0, labelpad=10, ha="right")
ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])

sns.despine(ax=ax)

f.savefig(OUT_DIR / "hga_contrast_acoustic.pdf", **PDF_SAVEFIG_KWARGS)

# %% [markdown]
# ## Plot individual site, acoustics vs behavior

# %%
plot_subject = "EC287"
plot_electrode_idx = 199
plot_phoneme_pair = "pb"
plot_word_end = "beneficial"

# %%
plot_acoustic_diffs = b4_by_acoustic_diffs.loc[(plot_subject, plot_electrode_idx, plot_phoneme_pair, plot_word_end)].values
plot_behavior_diffs = b4_by_behavior_diffs.loc[(plot_subject, plot_electrode_idx, plot_phoneme_pair, plot_word_end)].values

# %%
# b4_by_acoustic.loc[(plot_subject, plot_electrode_idx, plot_phoneme_pair, plot_word_end)]
# b4_by_behavior.loc[(plot_subject, plot_electrode_idx, plot_phoneme_pair, plot_word_end)]

# %%
f, ax = plt.subplots(figsize=(6, 4))

ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.axhline(0, color="k", linestyle="--", alpha=0.5)

xs = b4_by_behavior_diffs.columns.get_level_values("smin").values / epoch_sfreq + epoch_tmin + window_size / 2

ax.plot(xs, plot_behavior_diffs.mean(0), label="behavioral contrast")
ax.plot(xs, plot_acoustic_diffs.mean(0), label="acoustic contrast")

ax.legend()
