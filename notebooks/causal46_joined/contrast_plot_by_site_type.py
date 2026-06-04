# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
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
# # Per-response-type HGA contrast plots (causal46_joined)
#
# One continuous-time HGA contrast plot, **aligned to word onset, per manually
# annotated response type** (`site_type_relabel` in
# `manual_annotations/early_acoustic_window.csv`). Sites are pooled across
# phoneme pairs (bm/dn/pb) within each type.
#
# Each page overlays two trajectories (as in `contrast_plot.py`):
# - **Acoustic contrast** — `acoustic_sign × (HGA[step6] − HGA[step1])` per site,
#   pooled over both word-ends' endpoint trials, averaged across sites.
# - **Behavioral contrast** — within-completion bootstrapped "heard-first vs
#   heard-second" contrast on ambiguous steps, oriented by `acoustic_sign`
#   (the "aligned" convention), averaged across cells.
#
# `acoustic_sign` (from the CSV) = sign of median(HGA[step6] − HGA[step1]) at the
# site's best acoustic window, so multiplying by it orients each site so the
# **tuned** response is positive — the polarity trick that lets sites with
# opposite tuning be averaged without cancelling. step1 = clear first phoneme,
# step6 = clear second phoneme.
#
# **Complex/bimodal tuning:** the manual `complex` tuning label is not trusted
# (sometimes two real peaks, sometimes a wash), so it does not gate the average.
# In the default `overlay` mode every orientable site is in the mean and faint
# per-site traces are drawn behind it (annotated-`complex` sites in a distinct
# colour) so bimodality-vs-wash is visible by eye.
#
# Output: `contrast_plot_by_site_type.pdf` — one page per response type.

# %%
from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _contrast import (  # noqa: E402
    acoustic_endpoint_means,
    aggregate_trajectories,
    behavioral_bootstrap_meandiff,
    plot_contrast_axis,
    sliding_ttest,
)

from src.stimuli import OFFSET_DICT, POD_dict
from src.viz_provisional import load_epochs_dict

# %% tags=["parameters"]
annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
output_dir = "outputs/causal46_joined/contrast_plot_by_site_type"
epochs_dir = "outputs/epochs_preprocessed"
bootstrap_r = 1000
bootstrap_seed = 42
min_class_k = 4
ttest_window_size = 15
ttest_window_stride = 15
pval_thresholds = (0.00001, 0.0001, 0.001)
# Complex/bimodal acoustic tuning handling for the AVERAGED acoustic line:
#   "overlay" — all orientable sites in the mean + faint per-site traces (default)
#   "include" — all orientable sites in the mean, no traces
#   "exclude" — drop complex-tuned sites from the mean (report N excluded)
complex_acoustic_mode = "overlay"
complex_tuning_values = ("both", "complex", "two peaks")
exclude_tuning_conflict = True   # only used when mode == "exclude"
# Normalize duplicate-intent relabels (future-proofing; "behav_only" == type5)
site_type_relabel_map = {"behav_only": "type5_behav_only"}
# Review-flag relabels — not real response types
review_flag_types = ("problematic", "interesting", "unknown", "discuss")
review_flags_mode = "skip"       # "skip" | "page" (one combined page)
# Asymmetric sites: restrict the behavioral line to the single word-end that
# carries the effect, named (by initial phoneme letter) in this manual column.
asymmetric_sig_col = "if asymmetric, which is sig?"
asymmetric_use_sig_we_only = True
# Mirrored sites: this column names the ALIGNED word-end (by initial phoneme
# letter); keep only the OTHER (anti/mirrored) word-end so the behavioral line
# reads negative. Values that name no word-end (e.g. "complex") → keep both.
mirrored_aligned_col = "if mirrored, which WE is aligned?"
mirrored_use_anti_we_only = True

# %%
# papermill serializes tuples/dicts as strings; parse then coerce
for _name in ("pval_thresholds", "complex_tuning_values", "review_flag_types"):
    _v = globals()[_name]
    if isinstance(_v, str):
        globals()[_name] = ast.literal_eval(_v)
pval_thresholds = tuple(float(p) for p in pval_thresholds)
complex_tuning_values = tuple(str(v).lower() for v in complex_tuning_values)
review_flag_types = tuple(review_flag_types)
if isinstance(site_type_relabel_map, str):
    site_type_relabel_map = ast.literal_eval(site_type_relabel_map)

OUT_DIR = Path(output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Preferred page order; any other plotted types are appended after these
PREFERRED_ORDER = [
    "type1_acoustic_only",
    "type2_early_perceptual",
    "type3_asymmetric",
    "type4_early_perceptual_mirrored",
    "type5_behav_only",
    "A_unsigned",
    "complex",
]

LABELS = {
    "type1_acoustic_only": "Acoustic only",
    "type2_early_perceptual": "Acoustic+perceptual",
    "type3_asymmetric": "Acoustic+perceptual one-sided",
    "type4_early_perceptual_mirrored": "Acoustic+perceptual mirrored",
    "type5_behav_only": "Perceptual only",
    "A_unsigned": "Complex acoustic",
}

# %%
def _to_bool(series: pd.Series) -> pd.Series:
    """Coerce a CSV TRUE/FALSE (or bool) column to a boolean Series."""
    return series.astype(str).str.strip().str.upper().eq("TRUE")


ann = pd.read_csv(annotations_path)
print(f"annotations: {len(ann)} rows")
print(f"columns: {ann.columns.tolist()}")

# Normalize relabel values
ann["site_type_relabel"] = (
    ann["site_type_relabel"].astype(str).str.strip().replace(site_type_relabel_map)
)

# Derived helper columns
ann["_acoustic_sign"] = pd.to_numeric(ann["acoustic_sign"], errors="coerce")
ann["_orientable"] = ann["_acoustic_sign"].notna()
_tuning = ann["manifest_tuning"].astype(str).str.strip().str.lower()
ann["_is_complex"] = _tuning.isin(complex_tuning_values)
if exclude_tuning_conflict and "tuning_conflict" in ann.columns:
    ann["_is_complex"] = ann["_is_complex"] | _to_bool(ann["tuning_conflict"])

print("\nsite_type_relabel value counts:")
print(ann["site_type_relabel"].value_counts())
print(f"\norientable (acoustic_sign present): {int(ann['_orientable'].sum())} / {len(ann)}")
print(f"complex-tuned: {int(ann['_is_complex'].sum())} / {len(ann)}")

# %%
# Derive the ordered list of response types from the data at runtime.
present_types = list(ann["site_type_relabel"].value_counts().index)
plotted_types = [t for t in present_types if t not in review_flag_types]
ordered_types = (
    [t for t in PREFERRED_ORDER if t in plotted_types]
    + [t for t in plotted_types if t not in PREFERRED_ORDER]
)
review_types_present = [t for t in present_types if t in review_flag_types]
print(f"plotted types (in order): {ordered_types}")
print(f"review-flag types present: {review_types_present} (mode={review_flags_mode})")

# %%
# Load epochs (one-time eager load, matches contrast_plot.py)
epochs_dict = load_epochs_dict(Path(epochs_dir))
print(f"epochs loaded: {sorted(epochs_dict)}")
_sample_ep = next(iter(epochs_dict.values()))
times = _sample_ep.times

# %% [markdown]
# ## Per-site / per-cell contrast computations

# %%
def _ep_pp(subj, pair):
    """Epochs restricted to one phoneme_pair for a subject, or None."""
    if subj not in epochs_dict:
        return None
    ep = epochs_dict[subj]
    pp_mask = (ep.metadata["phoneme_pair"] == pair).values
    if not pp_mask.any():
        return None
    return ep[pp_mask]


def compute_acoustic_traj(row):
    """Sign-corrected acoustic contrast for one site, or None.

    Returns (trajectory, is_complex). trajectory = acoustic_sign × (step6 − step1).
    """
    subj = row["subject"]
    eidx = int(row["electrode_idx"])
    pair = row["phoneme_pair"]
    sign = float(row["_acoustic_sign"])
    ep_pp = _ep_pp(subj, pair)
    if ep_pp is None:
        return None
    means = acoustic_endpoint_means(ep_pp, eidx)  # (mean_step1, mean_step6)
    if means is None:
        return None
    mean_step1, mean_step6 = means
    traj = sign * (mean_step6 - mean_step1)
    return traj, bool(row["_is_complex"])


def compute_behav_traj(cell):
    """Aligned within-completion behavioral contrast for one cell, or None.

    traj = −acoustic_sign × (class0 − class1) = acoustic_sign × (class1 − class0),
    the "aligned" convention (positive when the percept response matches the
    site's acoustic tuning).
    """
    subj = cell["subject"]
    eidx = int(cell["electrode_idx"])
    pair = cell["phoneme_pair"]
    word_end = cell["word_end"]
    sign = float(cell["_acoustic_sign"])
    ep_pp = _ep_pp(subj, pair)
    if ep_pp is None:
        return None
    mean_diff, status = behavioral_bootstrap_meandiff(
        ep_pp, eidx, word_end,
        min_class_k=min_class_k, bootstrap_r=bootstrap_r, bootstrap_seed=bootstrap_seed,
    )
    if mean_diff is None:
        return None
    return -sign * mean_diff


def behavioral_word_ends(row):
    """Word-ends to include in the behavioral line for one site.

    Default: both annotated word-ends (B1, B2). Two manual columns can restrict
    this to a single word-end (word-ends start with their identifying phoneme:
    bountiful/b, mountains/m, desolate/d, necessary/n, penecillin/p,
    beneficial/b — so a first-letter match is unambiguous within a pair):

    - Asymmetric: `asymmetric_sig_col` names the word-end carrying the effect;
      keep only it so the null word-end does not dilute the average.
    - Mirrored: `mirrored_aligned_col` names the ALIGNED word-end; keep only the
      OTHER (anti/mirrored) word-end so the behavioral line reads negative.

    A value that names no word-end (empty, or e.g. "complex") → keep both.
    """
    wes = []
    for we_col in ("B1_word_end", "B2_word_end"):
        we = row.get(we_col)
        if pd.isna(we) or str(we).strip() == "":
            continue
        wes.append(str(we).strip())

    def _letter(col):
        raw = row.get(col, "")
        return "" if pd.isna(raw) else str(raw).strip().lower()

    # Asymmetric → keep the named (significant) word-end.
    sig = _letter(asymmetric_sig_col)
    if asymmetric_use_sig_we_only and sig:
        matched = [we for we in wes if we[:1].lower() == sig[:1]]
        if matched:
            return matched
        print(f"  ⚠ asymmetric sig letter {sig!r} matched no word-end for "
              f"{row['subject']} e{int(row['electrode_idx'])} {row['phoneme_pair']} "
              f"(word_ends={wes}); falling back to both")

    # Mirrored → keep the OTHER (anti) word-end, i.e. the one NOT named aligned.
    aligned = _letter(mirrored_aligned_col)
    if mirrored_use_anti_we_only and aligned:
        anti = [we for we in wes if we[:1].lower() != aligned[:1]]
        if anti and len(anti) < len(wes):
            return anti
        print(f"  ⚠ mirrored aligned letter {aligned!r} named no word-end for "
              f"{row['subject']} e{int(row['electrode_idx'])} {row['phoneme_pair']} "
              f"(word_ends={wes}); falling back to both")

    return wes


# %%
# Sanity diagnostic: the sign convention assumes behavior class 0 = first phoneme
# (step-1 percept) and class 1 = second phoneme (step-6 percept). Check that the
# modal behavior at the clear endpoints follows that ordering across sites.
def _endpoint_class_encoding_ok(ann_orientable, n_check=40):
    from _within_completion import resolve_behavior_col
    ok = bad = 0
    for _, row in ann_orientable.head(n_check).iterrows():
        ep_pp = _ep_pp(row["subject"], row["phoneme_pair"])
        if ep_pp is None:
            continue
        md = ep_pp.metadata.reset_index(drop=True)
        col = resolve_behavior_col(md)
        m1 = md.loc[(md["resampled"] == 1).values, col].mode()
        m6 = md.loc[(md["resampled"] == 6).values, col].mode()
        if len(m1) == 1 and len(m6) == 1:
            if int(m1.iloc[0]) == 0 and int(m6.iloc[0]) == 1:
                ok += 1
            else:
                bad += 1
    return ok, bad


_ok, _bad = _endpoint_class_encoding_ok(ann[ann["_orientable"]])
print(f"endpoint class-encoding check: {_ok} consistent (class0@step1, class1@step6), "
      f"{_bad} inconsistent")
if _bad > _ok:
    print("  ⚠ behavior class ordering may be flipped vs assumption — "
          "verify behavioral line signs by eye.")

# %% [markdown]
# ## Aggregate per response type

# %%
def build_results_for_type(rows):
    """Compute acoustic + behavioral aggregates for one type's annotation rows."""
    orient = rows[rows["_orientable"]]

    # --- Acoustic pool (one row per site) ---
    ac_rows = orient
    if complex_acoustic_mode == "exclude":
        ac_rows = ac_rows[~ac_rows["_is_complex"]]
    ac_trajs, ac_site_traces, n_complex = [], [], 0
    for _, row in ac_rows.iterrows():
        res = compute_acoustic_traj(row)
        if res is None:
            continue
        traj, is_complex = res
        ac_trajs.append(traj)
        n_complex += int(is_complex)
        if complex_acoustic_mode == "overlay":
            ac_site_traces.append((traj, is_complex))

    # --- Behavioral pool (≤2 cells per site; asymmetric → only the sig WE) ---
    bh_trajs = []
    for _, row in orient.iterrows():
        for we in behavioral_word_ends(row):
            cell = {
                "subject": row["subject"], "electrode_idx": row["electrode_idx"],
                "phoneme_pair": row["phoneme_pair"], "word_end": we,
                "_acoustic_sign": row["_acoustic_sign"],
            }
            traj = compute_behav_traj(cell)
            if traj is not None:
                bh_trajs.append(traj)

    ac_matrix, ac_mean, ac_sem = aggregate_trajectories(ac_trajs)
    bh_matrix, bh_mean, bh_sem = aggregate_trajectories(bh_trajs)
    return {
        "ac_mean": ac_mean, "ac_sem": ac_sem,
        "ac_ttest": sliding_ttest(ac_matrix, times, ttest_window_size, ttest_window_stride),
        "ac_site_traces": ac_site_traces,
        "bh_mean": bh_mean, "bh_sem": bh_sem,
        "bh_ttest": sliding_ttest(bh_matrix, times, ttest_window_size, ttest_window_stride),
        "n_acoustic": len(ac_trajs), "n_acoustic_complex": n_complex,
        "n_behav": len(bh_trajs), "n_sites": len(orient),
    }


results = {}
for t in ordered_types:
    results[t] = build_results_for_type(ann[ann["site_type_relabel"] == t])
    r = results[t]
    print(f"{t:34s}  sites={r['n_sites']:3d}  acoustic={r['n_acoustic']:3d} "
          f"(complex={r['n_acoustic_complex']:2d})  behav_cells={r['n_behav']:3d}")

if review_flags_mode == "page" and review_types_present:
    results["review_flags"] = build_results_for_type(
        ann[ann["site_type_relabel"].isin(review_types_present)]
    )
    ordered_types = ordered_types + ["review_flags"]

# %%
counts = [results[t]["n_sites"] for t in ordered_types]

f, ax = plt.subplots(figsize=(3.5, 3.5))
# show pie chart with counts
ax.pie(counts, labels=[LABELS[t] for t in ordered_types], autopct=lambda x: int(x / 100. * sum(counts)))

# %% [markdown]
# ## Plot — one page per response type

# %%
XLIM = max(OFFSET_DICT.values()) + 0.1
POD_BAND = (min(POD_dict.values()), max(POD_dict.values()))


def plot_type_page(type_name, res):
    fig, ax = plt.subplots(figsize=(7, 4))
    if res["n_acoustic"] == 0 and res["n_behav"] == 0:
        ax.text(0.5, 0.5, f"{type_name}\n(no data)", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xlim(0.0, XLIM)
        return fig

    n_ac, n_bh, n_cx = res["n_acoustic"], res["n_behav"], res["n_acoustic_complex"]
    ac_label = f"Acoustic (n={n_ac} sites" + (f", {n_cx} complex)" if n_cx else ")")
    bh_label = f"Behavioral (n={n_bh} cells)"
    plot_contrast_axis(
        ax, times,
        ac_mean=res["ac_mean"], ac_sem=res["ac_sem"], ac_ttest=res["ac_ttest"], n_acoustic=n_ac,
        bh_mean=res["bh_mean"], bh_sem=res["bh_sem"], bh_ttest=res["bh_ttest"], n_behav=n_bh,
        pval_thresholds=pval_thresholds,
        acoustic_label=ac_label, behav_label=bh_label,
        pod_band=POD_BAND,
        site_traces=res["ac_site_traces"] if complex_acoustic_mode == "overlay" else None,
        site_trace_alpha=0.35, site_trace_lw=0.6,
        legend_title=("acoustic (blue), behavioral (red); sig bars below\n"
                      "thin lines = per-site acoustic (purple = complex tuning)\n"
                      "acoustic: 1 / site · behavioral: 1 / (site × completion)"),
        # place legend outside the axes so it never occludes the per-site traces
        legend_loc="upper left", legend_bbox_to_anchor=(1.01, 1.0),
    )
    ax.set_xlabel("Time (s, post word onset)")
    ax.set_ylabel("HGA (z, baseline-corrected)")
    ax.set_title(
        f"{type_name}  —  acoustic vs behavioral contrast\n"
        f"(same sites; acoustic pools completions, behavioral splits by completion)",
        fontsize=10,
    )
    ax.set_xlim(0.0, XLIM)
    fig.tight_layout()
    return fig


out_path = OUT_DIR / "contrast_plot_by_site_type.pdf"
with PdfPages(out_path) as pdf:
    for t in ordered_types:
        fig = plot_type_page(t, results[t])
        pdf.savefig(fig, bbox_inches="tight")
print(f"Saved: {out_path}  ({len(ordered_types)} pages)")

# %% [markdown]
# ## Summary

# %%
print("=" * 70)
print("CONTRAST PLOT BY SITE TYPE — SUMMARY")
print("=" * 70)
print(f"annotations:           {annotations_path}")
print(f"complex_acoustic_mode: {complex_acoustic_mode}")
print(f"orientation:           acoustic_sign (aligned convention)")
print()
print(f"{'response type':34s} {'sites':>6s} {'acoustic':>9s} {'complex':>8s} {'behav':>6s}"
      f" {'ac_late':>9s} {'bh_late':>9s}")
for t in ordered_types:
    r = results[t]
    ac_late = r["ac_mean"][times > 0.3].mean() if r["ac_mean"] is not None else float("nan")
    bh_late = r["bh_mean"][times > 0.3].mean() if r["bh_mean"] is not None else float("nan")
    print(f"{t:34s} {r['n_sites']:6d} {r['n_acoustic']:9d} {r['n_acoustic_complex']:8d}"
          f" {r['n_behav']:6d} {ac_late:9.4f} {bh_late:9.4f}")
print("=" * 70)
print("Done.")
