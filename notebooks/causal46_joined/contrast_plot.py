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
import scipy.stats  # noqa: F401 (used in sliding_ttest)

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    extract_hga,
    per_step_class_counts,
    resolve_behavior_col,
    select_cell_trials_bootstrap,
)

from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS, POD_dict
from src.viz_provisional import load_epochs_dict

# %% tags=["parameters"]
manifest_path = "outputs_prod/causal46_joined/filtered_manifest.csv"
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
    md_pp = ep_pp.metadata.reset_index(drop=True)

    hga = extract_hga(ep_pp, eidx)

    # Determine polarity: class 0 = first phoneme, class 1 = second phoneme
    first_ph = PAIR_PHONEMES[pair][0]
    acoustic_sign = 1 if acoustic_tuning_letter == first_ph else -1

    # step 1 = first phoneme (clear), step 6 = second phoneme (clear)
    step1_mask = (md_pp["resampled"] == 1).values
    step6_mask = (md_pp["resampled"] == 6).values
    if not step1_mask.any() or not step6_mask.any():
        print(f"  SKIP acoustic: {subj} e{eidx} {pair} missing endpoint steps")
        acoustic_skipped += 1
        continue

    raw_diff = hga[step1_mask].mean(0) - hga[step6_mask].mean(0)
    trajectory = acoustic_sign * raw_diff
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
    md_pp = ep_pp.metadata.reset_index(drop=True)

    hga = extract_hga(ep_pp, eidx)

    # Determine qualifying steps
    bhv_col = resolve_behavior_col(md_pp)
    we_mask = (md_pp["word_end"] == word_end).values
    candidate_steps = [
        s for s in [2, 3, 4, 5]
        if (we_mask & (md_pp["resampled"] == s).values).any()
    ]
    per_step = per_step_class_counts(
        md_pp, word_end=word_end,
        qualifying_steps=candidate_steps,
        group_col=bhv_col,
    )
    qualifying_steps = [
        s for s, by_class in per_step.items()
        if len(by_class) == 2 and min(len(v) for v in by_class.values()) >= min_class_k
    ]
    if not qualifying_steps:
        behavioral_no_qualifying += 1
        continue

    per_step_q = {s: per_step[s] for s in qualifying_steps}

    # Bootstrap — stream to avoid memory accumulation
    running_sum = np.zeros(hga.shape[1])
    valid_reps = 0
    for r in range(bootstrap_r):
        draws = select_cell_trials_bootstrap(
            per_step_q, rng=np.random.default_rng(bootstrap_seed + r)
        )
        if 0 not in draws or 1 not in draws:
            continue
        # class 0 = first phoneme, class 1 = second phoneme
        diff_r = hga[draws[0]].mean(0) - hga[draws[1]].mean(0)
        running_sum += diff_r
        valid_reps += 1

    if valid_reps == 0:
        behavioral_skipped += 1
        continue

    mean_diff = running_sum / valid_reps
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
def aggregate_trajectories(trajectories):
    """Return (matrix, grand_mean, sem) from a list of 1D trajectory arrays."""
    if not trajectories:
        return None, None, None
    matrix = np.stack(trajectories, axis=0)  # (n_sites, n_times)
    grand_mean = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    return matrix, grand_mean, sem


def sliding_ttest(matrix, times, window_size, window_stride):
    """One-sample t-test on sliding windows. Returns list of (t_start, t_end, p_val)."""
    n_times = matrix.shape[1]
    results = []
    for start in range(0, n_times - window_size + 1, window_stride):
        window_means = matrix[:, start:start + window_size].mean(axis=1)
        t_stat, p_val = scipy.stats.ttest_1samp(window_means, 0)
        end = min(start + window_size, n_times - 1)
        t_start = times[start]
        t_end = times[end]
        results.append((t_start, t_end, p_val))
    return results


# %%
ac_matrix, ac_mean, ac_sem = aggregate_trajectories(acoustic_trajectories)
bh_matrix, bh_mean, bh_sem = aggregate_trajectories(behavioral_trajectories)

print(f"Acoustic matrix shape: {ac_matrix.shape if ac_matrix is not None else None}")
print(f"Behavioral matrix shape: {bh_matrix.shape if bh_matrix is not None else None}")

ac_ttest = sliding_ttest(ac_matrix, times, ttest_window_size, ttest_window_stride) if ac_matrix is not None else []
bh_ttest = sliding_ttest(bh_matrix, times, ttest_window_size, ttest_window_stride) if bh_matrix is not None else []

# %% [markdown]
# ## Plot

# %%
ACOUSTIC_COLOR = "#2166ac"
BEHAVIORAL_COLOR = "#d73027"

# p_threshold height multipliers (from viz_paper.py style)
P_THRESHOLD_MULTS = [1.0, 0.5, 0.25]

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

# -- Plot acoustic mean + SEM ribbon
if ac_mean is not None:
    ax.plot(times, ac_mean, color=ACOUSTIC_COLOR, lw=2,
            label=f"Acoustic (n={n_acoustic} sites)")
    ax.fill_between(times, ac_mean - ac_sem, ac_mean + ac_sem,
                    color=ACOUSTIC_COLOR, alpha=0.18)

# -- Plot behavioral mean + SEM ribbon
if bh_mean is not None:
    ax.plot(times, bh_mean, color=BEHAVIORAL_COLOR, lw=2,
            label=f"Behavioral (n={n_behav} cells)")
    ax.fill_between(times, bh_mean - bh_sem, bh_mean + bh_sem,
                    color=BEHAVIORAL_COLOR, alpha=0.18)

ax.axhline(0, color="k", lw=0.5, ls=":")

# -- POD vertical line (per-pair only)
if phoneme_pair is not None and phoneme_pair in POD_dict:
    pod_time = POD_dict[phoneme_pair]
    ax.axvline(pod_time, color="gray", lw=1.2, ls="--", alpha=0.7, label="POD")

# -- Significance bars
# Two rows: acoustic (upper) and behavioral (lower)
ymin, ymax = ax.get_ylim()
base_bar_h = (ymax - ymin) * 0.04
bar_row_gap = base_bar_h * 1.5

bar_y_ac = ymin + (ymax - ymin) * 0.95
bar_y_bh = bar_y_ac - bar_row_gap

p_thresholds_sorted = sorted(pval_thresholds)   # ascending (smallest first = darkest)
for (t_start, t_end, p_val) in ac_ttest:
    for i, p_thresh in enumerate(p_thresholds_sorted):
        if p_val < p_thresh:
            mult = P_THRESHOLD_MULTS[i]
            ax.barh(y=bar_y_ac, width=t_end - t_start, left=t_start,
                    height=base_bar_h * mult,
                    color=ACOUSTIC_COLOR, alpha=0.5, edgecolor="none")
            break

for (t_start, t_end, p_val) in bh_ttest:
    for i, p_thresh in enumerate(p_thresholds_sorted):
        if p_val < p_thresh:
            mult = P_THRESHOLD_MULTS[i]
            ax.barh(y=bar_y_bh, width=t_end - t_start, left=t_start,
                    height=base_bar_h * mult,
                    color=BEHAVIORAL_COLOR, alpha=0.5, edgecolor="none")
            break

# Legend
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

class HandlerRect(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        rect = Rectangle([xdescent, ydescent], width, height,
                         facecolor=orig_handle.get_facecolor(),
                         alpha=orig_handle.get_alpha(),
                         edgecolor="none")
        return [rect]

p_handles = []
for i, (p_thresh, mult) in enumerate(zip(p_thresholds_sorted, P_THRESHOLD_MULTS)):
    h = Rectangle((0, 0), 1, mult, facecolor="gray", alpha=0.5,
                  label=f"p < {p_thresh:g}".replace("-0", "-"))
    p_handles.append(h)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles + p_handles,
    labels=labels + [h.get_label() for h in p_handles],
    handler_map={Rectangle: HandlerRect()},
    loc="lower left",
    fontsize=8,
    title="Sig bars: acoustic (blue), behavioral (red)\n[different site populations]",
    title_fontsize=7,
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
