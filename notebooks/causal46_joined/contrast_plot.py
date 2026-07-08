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
from __future__ import annotations

import ast
from collections import defaultdict
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _contrast import (  # noqa: E402
    acoustic_endpoint_means,
    aggregate_trajectories,
    behavioral_bootstrap_meandiff,
    plot_contrast_axis,
    sliding_ttest,
)

from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS, POD_dict
from src.viz_provisional import load_epochs_dict

# %% tags=["parameters"]
annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
filtered_manifest_path = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"

output_dir = "outputs/causal46_joined/contrast_plot"
phoneme_pair = None   # None = aggregate all pairs; "bm"/"dn"/"pb" for per-pair
bootstrap_r = 1000
bootstrap_seed = 42
min_class_k = 3
ttest_window_size = 5
ttest_window_stride = 5
pval_thresholds = (0.00001, 0.0001, 0.001)
epochs_dir = "outputs/epochs_preprocessed"
# "annotated": sign-correct using consensus tuning letter from manifest
# "abs":       take absolute value of mean diff (no manifest label needed)
behav_polarity_mode = "annotated"

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
b4_by_acoustic_windows = pd.read_parquet("outputs/causal46_joined/acoustic_discriminative_windows/ad_windows.parquet")

# DEV
b4_by_acoustic_windows = b4_by_acoustic_windows.query("smax <= 68")

# %%
# TODO paste in significance check from cc here
ac_bootstrap_sites = b4_by_acoustic.reset_index()[["subject", "electrode_idx", "phoneme_pair", "word_end"]].drop_duplicates()
ac_significant_sites = b4_by_acoustic_windows[["subject", "electrode_idx", "phoneme_pair", "word_end"]].drop_duplicates()

ac_nonsig_sites = pd.merge(
    ac_bootstrap_sites, ac_significant_sites,
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    how="outer",
    indicator=True
).query('_merge == "left_only"').drop(columns="_merge")

# Record those sites which are fully missing from the acoustic windows (i.e., no significant windows for either word end)
b4_by_acoustic_windows_missing = set(
    pd.merge(
        ac_bootstrap_sites, ac_significant_sites,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="outer",
        indicator=True
    )
    .groupby(["subject", "electrode_idx", "phoneme_pair"])
    .apply(lambda df: df._merge.unique().tolist() == ["left_only"])
    .index.to_list()
)

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

# papermill serializes tuples as a single string; parse then coerce
if isinstance(pval_thresholds, str):
    pval_thresholds = ast.literal_eval(pval_thresholds)
pval_thresholds = tuple(float(p) for p in pval_thresholds)

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

behavior_contrasts = defaultdict(list)
for (subject, electrode_idx, phoneme_pair, word_end), row in behavior_plot_guide_df.iterrows():
    # # DEV
    # keep = [
    #     ("EC250", 191, "dn")
    # ]
    # if (subject, electrode_idx, phoneme_pair) not in keep:
    #     continue

    ep = epochs_dict[subject]
    md = ep.metadata
    assert md is not None

    conjunction_category = row.conjunction_category
    sign = row.sign
    if pd.isna(word_end):
        # this row was present in `merged` but not in `b4_by_behavior_windows`, because
        # it doesn't have a behavioral window!
        assert row._merge == "right_only", "assumption violated"

        # .. but we need to draw differences from somewhere. SO we'll pool across the word ends
        # which are available
        # (a word end may not be available if there was no sufficiently powered balanced ambiguous cell)
        b_diffs = b4_by_behavior_diffs.loc[(subject, electrode_idx, phoneme_pair, slice(None))].values
        conjunction_category = "Acoustic-only"

        b_diff_mean = b_diffs.mean(0)

        print(f"Warning: site×pair {subject}×{electrode_idx}×{phoneme_pair} has no behavioral window; pooling across word ends")
    else:
        if pd.isna(row.conjunction_category):
            # print(f"Warning: site×pair×word_end {subject}×{electrode_idx}×{phoneme_pair}×{word_end} has no conjunction category; skipping")
            continue

        b_diffs = b4_by_behavior_diffs.loc[(subject, electrode_idx, phoneme_pair, word_end)].values
        b_diffs = row.sign * b_diffs
        b_diff_mean = b_diffs.mean(0)

    behavior_contrasts[conjunction_category].append(b_diff_mean)

# %%
plt.axvline(0, color="k", linestyle="--", alpha=0.5)
plt.axhline(0, color="k", linestyle="--", alpha=0.5)

for cat, contrasts in behavior_contrasts.items():
    ys = np.stack(contrasts).mean(0)
    xs = b4_by_behavior_diffs.columns.get_level_values("smin").values / epoch_sfreq + epoch_tmin
    plt.plot(xs, ys, label=cat)

    yerr = np.stack(contrasts).std(0) / np.sqrt(len(contrasts))
    plt.fill_between(xs, ys - yerr, ys + yerr, alpha=0.2)

plt.legend()


# %% [markdown]
# ## Plot contrast by acoustics

# %%
b4_by_acoustic_diffs = b4_by_acoustic.pivot_table(
    index=["subject", "electrode_idx", "phoneme_pair", "word_end", "replicate"],
    columns=["smin"],
    values=["mean_diff_raw"]
)

# %%
acoustic_plot_guide_df = pd.merge(
    b4_by_acoustic_windows
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .first(),
    merged.set_index(["subject", "electrode_idx", "phoneme_pair"]),
    how="outer", left_index=True, right_index=True, indicator=True
)

# Sanity check: all the sites that are in the acoustic contrast but missing in merged
# were intentionally dropped because they were assigned a null early_label or a null late_label
assert set(acoustic_plot_guide_df.query("_merge == 'left_only'").index.droplevel(-1)) <= dropped_sites

# Sanity check: all the sites that are in merged but are missing in the acoustic contrast
# are because we didn't have a cell that supported the acoustic contrast
# OR we had a cell but there was no simple contrast that showed up in the bootstrap contrast
for (subject, electrode_idx, phoneme_pair, _), _ in acoustic_plot_guide_df.query("_merge == 'right_only'").iterrows():
    word_ends = PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]
    print(f"Warning: site×pair {subject}×{electrode_idx}×{phoneme_pair} has no acoustic window; skipping")
    assert (subject, electrode_idx, phoneme_pair, word_ends[0]) in missing_acoustic_site_keys \
        or (subject, electrode_idx, phoneme_pair, word_ends[1]) in missing_acoustic_site_keys \
        or (subject, electrode_idx, phoneme_pair) in b4_by_acoustic_windows_missing

acoustic_plot_guide_df = acoustic_plot_guide_df.query("_merge == 'both'").drop(columns=["_merge"])

# %%
acoustic_contrasts = defaultdict(list)
for (subject, electrode_idx, phoneme_pair, word_end), row in acoustic_plot_guide_df.iterrows():
    conjunction_category = row.conjunction_category
    b_diffs = b4_by_acoustic_diffs.loc[(subject, electrode_idx, phoneme_pair, word_end)].values
    b_diff_mean = (row.sign * b_diffs).mean(0)

    # # DEV
    # if len(acoustic_contrasts[conjunction_category]) > 0:
    #     continue
    # else:
    #     print("DEV: plotting only one site per conjunction category for now")
    #     print(conjunction_category, subject, electrode_idx, phoneme_pair, word_end)

    acoustic_contrasts[conjunction_category].append(b_diff_mean)

# %%
plt.axvline(0, color="k", linestyle="--", alpha=0.5)
plt.axhline(0, color="k", linestyle="--", alpha=0.5)

for cat, contrasts in acoustic_contrasts.items():
    ys = np.stack(contrasts).mean(0)
    xs = b4_by_acoustic_diffs.columns.get_level_values("smin").values / epoch_sfreq + epoch_tmin
    plt.plot(xs, ys, label=cat)

    yerr = np.stack(contrasts).std(0) / np.sqrt(len(contrasts))
    plt.fill_between(xs, ys - yerr, ys + yerr, alpha=0.2)

plt.legend()
