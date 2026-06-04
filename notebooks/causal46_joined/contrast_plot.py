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
#
# Population-level acoustic and behavioral HGA contrast trajectories.
# Acoustic pool: sites with a valid `acoustic tuning` letter in filtered_manifest.csv.
# Behavioral pool: cells with manual behavioral annotations (behav @ac, etc.).
#
# Outputs:
# - `contrast_plot.pdf` (aggregate) or `{pair}_contrast_plot.pdf` (per-pair)

# %%
from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
manifest_path = "outputs_prod/causal46_joined/manual_annotations/filtered_manifest.csv"
output_dir = "outputs/causal46_joined/contrast_plot"
phoneme_pair = None   # None = aggregate all pairs; "bm"/"dn"/"pb" for per-pair
bootstrap_r = 1000
bootstrap_seed = 42
min_class_k = 4
ttest_window_size = 15
ttest_window_stride = 15
pval_thresholds = (0.00001, 0.0001, 0.001)
epochs_dir = "outputs/epochs_preprocessed"
# "annotated": sign-correct using consensus tuning letter from manifest
# "abs":       take absolute value of mean diff (no manifest label needed)
behav_polarity_mode = "annotated"

# %%
PAIR_PHONEMES = {"bm": ("b", "m"), "dn": ("d", "n"), "pb": ("p", "b")}

# papermill serializes tuples as a single string; parse then coerce
if isinstance(pval_thresholds, str):
    pval_thresholds = ast.literal_eval(pval_thresholds)
pval_thresholds = tuple(float(p) for p in pval_thresholds)

OUT_DIR = Path(output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
manifest = pd.read_csv(manifest_path)
print(f"manifest: {len(manifest)} rows")
print(f"manifest columns: {manifest.columns.tolist()}")

# %%
# Load epochs (one-time eager load, matches t_tests.py pattern)
epochs_dict = load_epochs_dict(Path(epochs_dir))
print(f"epochs loaded: {sorted(epochs_dict)}")

# %% [markdown]
# ## Acoustic pool

# %%
def build_acoustic_pool(manifest: pd.DataFrame, phoneme_pair_filter):
    """Return rows for the acoustic pool.

    One row per unique (subject, electrode_idx, phoneme_pair).
    """
    df = manifest.copy()
    # Filter: acoustic tuning is a single lowercase letter
    mask_acoustic = df["acoustic tuning"].str.match(r'^[a-z]$', na=False)
    df = df[mask_acoustic].copy()
    print(f"  acoustic pool: {len(df)} rows after tuning-letter filter")

    if phoneme_pair_filter is not None:
        df = df[df["phoneme_pair"] == phoneme_pair_filter].copy()
        print(f"  acoustic pool: {len(df)} rows after phoneme_pair={phoneme_pair_filter} filter")

    # Dedup to (subject, electrode_idx, phoneme_pair) — both word_ends share same acoustic window
    df = df.drop_duplicates(subset=["subject", "electrode_idx", "phoneme_pair"]).copy()
    print(f"  acoustic pool: {len(df)} unique sites after dedup")
    return df


# %%
def get_behav_letter(row):
    """Return consensus tuning letter from behavioral annotation columns, or None."""
    cols = ["behav @ac", "behav @ac slightly late", "behav @late"]
    letters = set()
    for col in cols:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            letters.add(str(val).strip())
    if len(letters) == 1:
        return letters.pop()
    return None


def build_behavioral_pool(manifest: pd.DataFrame, phoneme_pair_filter):
    """Return rows for the behavioral pool.

    One row per unique (subject, electrode_idx, phoneme_pair, word_end).
    Applies the three-step filtering described in the spec.
    """
    df = manifest.copy()

    if phoneme_pair_filter is not None:
        df = df[df["phoneme_pair"] == phoneme_pair_filter].copy()
        print(f"  behav pool: {len(df)} rows after phoneme_pair={phoneme_pair_filter} filter")

    behav_cols = ["behav @ac", "behav @ac slightly late", "behav @late"]

    # Filter step 1: at least one behavioral annotation is non-empty
    def has_any_behav(row):
        for col in behav_cols:
            val = row.get(col, "")
            if pd.notna(val) and str(val).strip():
                return True
        return False

    mask1 = df.apply(has_any_behav, axis=1)
    df = df[mask1].copy()
    print(f"  behav pool: {len(df)} rows after step1 (any annotation present)")

    # Filter step 2: no within-column multi-peak (contains "then")
    def has_no_then(row):
        for col in behav_cols:
            val = row.get(col, "")
            if pd.notna(val) and "then" in str(val).lower():
                return False
        return True

    mask2 = df.apply(has_no_then, axis=1)
    df = df[mask2].copy()
    print(f"  behav pool: {len(df)} rows after step2 (no 'then' multi-peak)")

    # Filter step 3: cross-column tuning agreement — all non-empty letters must match
    df["_behav_letter"] = df.apply(get_behav_letter, axis=1)
    n_before = len(df)
    df = df[df["_behav_letter"].notna()].copy()
    print(f"  behav pool: {len(df)} rows after step3 (consensus letter, dropped {n_before - len(df)})")

    return df


# %%
acoustic_pool = build_acoustic_pool(manifest, phoneme_pair)
behavioral_pool = build_behavioral_pool(manifest, phoneme_pair)

print(f"\nAcoustic pool: {len(acoustic_pool)} sites")
print(f"Behavioral pool: {len(behavioral_pool)} cells")

# %% [markdown]
# ## Compute acoustic trajectories

# %%
acoustic_trajectories = []
acoustic_skipped = 0

for _, row in acoustic_pool.iterrows():
    subj = row["subject"]
    eidx = int(row["electrode_idx"])
    pair = row["phoneme_pair"]
    acoustic_tuning_letter = str(row["acoustic tuning"]).strip()

    if subj not in epochs_dict:
        print(f"  SKIP acoustic: no epochs for {subj}")
        acoustic_skipped += 1
        continue

    ep = epochs_dict[subj]
    md = ep.metadata
    pp_mask = (md["phoneme_pair"] == pair).values
    ep_pp = ep[pp_mask]

    # Determine polarity: class 0 = first phoneme, class 1 = second phoneme
    first_ph = PAIR_PHONEMES[pair][0]
    acoustic_sign = 1 if acoustic_tuning_letter == first_ph else -1

    # step 1 = first phoneme (clear), step 6 = second phoneme (clear)
    means = acoustic_endpoint_means(ep_pp, eidx)  # (mean_step1, mean_step6)
    if means is None:
        print(f"  SKIP acoustic: {subj} e{eidx} {pair} missing endpoint steps")
        acoustic_skipped += 1
        continue
    mean_step1, mean_step6 = means
    trajectory = acoustic_sign * (mean_step1 - mean_step6)
    acoustic_trajectories.append(trajectory)

print(f"\nAcoustic trajectories: {len(acoustic_trajectories)} (skipped: {acoustic_skipped})")

# %% [markdown]
# ## Compute behavioral trajectories

# %%
behavioral_trajectories = []
behavioral_skipped = 0
behavioral_no_qualifying = 0

# Use a sample epoch to get the times vector
_sample_ep = next(iter(epochs_dict.values()))
times = _sample_ep.times

for _, row in behavioral_pool.iterrows():
    subj = row["subject"]
    eidx = int(row["electrode_idx"])
    pair = row["phoneme_pair"]
    word_end = row["word_end"]
    behav_letter = str(row["_behav_letter"]).strip()

    if subj not in epochs_dict:
        print(f"  SKIP behav: no epochs for {subj}")
        behavioral_skipped += 1
        continue

    ep = epochs_dict[subj]
    md = ep.metadata
    pp_mask = (md["phoneme_pair"] == pair).values
    ep_pp = ep[pp_mask]

    mean_diff, status = behavioral_bootstrap_meandiff(
        ep_pp, eidx, word_end,
        min_class_k=min_class_k,
        bootstrap_r=bootstrap_r,
        bootstrap_seed=bootstrap_seed,
    )
    if status == "no_qualifying":
        behavioral_no_qualifying += 1
        continue
    if mean_diff is None:  # status == "skipped"
        behavioral_skipped += 1
        continue

    if behav_polarity_mode == "abs":
        trajectory = np.abs(mean_diff)
    else:  # "annotated"
        first_ph = PAIR_PHONEMES[pair][0]
        behav_sign = 1 if behav_letter == first_ph else -1
        trajectory = behav_sign * mean_diff
    behavioral_trajectories.append(trajectory)

print(f"\nBehavioral trajectories: {len(behavioral_trajectories)}"
      f" (skipped: {behavioral_skipped}, no_qualifying: {behavioral_no_qualifying})")

# %% [markdown]
# ## Aggregation and significance testing

# %%
# aggregate_trajectories / sliding_ttest are imported from _contrast.
ac_matrix, ac_mean, ac_sem = aggregate_trajectories(acoustic_trajectories)
bh_matrix, bh_mean, bh_sem = aggregate_trajectories(behavioral_trajectories)

print(f"Acoustic matrix shape: {ac_matrix.shape if ac_matrix is not None else None}")
print(f"Behavioral matrix shape: {bh_matrix.shape if bh_matrix is not None else None}")

ac_ttest = sliding_ttest(ac_matrix, times, ttest_window_size, ttest_window_stride) if ac_matrix is not None else []
bh_ttest = sliding_ttest(bh_matrix, times, ttest_window_size, ttest_window_stride) if bh_matrix is not None else []

# %% [markdown]
# ## Plot

# %%
# x-axis limit
if phoneme_pair is not None:
    _wes = PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]
    xlim = max(OFFSET_DICT[we] for we in _wes) + 0.1
else:
    xlim = max(OFFSET_DICT.values()) + 0.1

# %%
fig, ax = plt.subplots(figsize=(7, 4))

n_acoustic = len(acoustic_trajectories)
n_behav = len(behavioral_trajectories)

pod_vline = POD_dict[phoneme_pair] if (phoneme_pair is not None and phoneme_pair in POD_dict) else None

plot_contrast_axis(
    ax, times,
    ac_mean=ac_mean, ac_sem=ac_sem, ac_ttest=ac_ttest, n_acoustic=n_acoustic,
    bh_mean=bh_mean, bh_sem=bh_sem, bh_ttest=bh_ttest, n_behav=n_behav,
    pval_thresholds=pval_thresholds,
    pod_vline=pod_vline,
)

# Axis labels and title
ax.set_xlabel("Time (s, post word onset)")
ax.set_ylabel("HGA (z, baseline-corrected)")

if phoneme_pair is None:
    title = "All pairs — acoustic vs behavioral contrast"
else:
    title = f"{phoneme_pair} — acoustic vs behavioral contrast"
ax.set_title(title, fontsize=11)
ax.set_xlim(0.0, xlim)

fig.tight_layout()

# %%
# Save figure
if phoneme_pair is None:
    out_path = OUT_DIR / "contrast_plot.pdf"
else:
    out_path = OUT_DIR / f"{phoneme_pair}_contrast_plot.pdf"

fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")

# %% [markdown]
# ## Summary

# %%
print("=" * 60)
print("CONTRAST PLOT SUMMARY")
print("=" * 60)
print(f"phoneme_pair filter:   {phoneme_pair!r}")
print()
print("ACOUSTIC POOL")
print(f"  manifest rows:        {len(manifest)}")
ac_letter_mask = manifest['acoustic tuning'].str.match(r'^[a-z]$', na=False)
print(f"  after tuning-letter:  {ac_letter_mask.sum()}")
if phoneme_pair is not None:
    n_pp = (ac_letter_mask & (manifest['phoneme_pair'] == phoneme_pair)).sum()
    print(f"  after pair filter:    {n_pp}")
print(f"  unique sites (dedup): {len(acoustic_pool)}")
print(f"  trajectories built:   {n_acoustic}")
print(f"  skipped:              {acoustic_skipped}")
print()
print("BEHAVIORAL POOL")
print(f"  after phoneme_pair:   {len(behavioral_pool)}")
print(f"  trajectories built:   {n_behav}")
print(f"  skipped (other):      {behavioral_skipped}")
print(f"  no qualifying steps:  {behavioral_no_qualifying}")
print()
if ac_mean is not None:
    print(f"Acoustic grand mean (late, >0.3s): {ac_mean[times > 0.3].mean():.4f}")
if bh_mean is not None:
    print(f"Behavioral grand mean (late, >0.3s): {bh_mean[times > 0.3].mean():.4f}")
print("=" * 60)
print("Done.")
