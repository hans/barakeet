# -*- coding: utf-8 -*-
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
# # Sankey: early-window site type → late behavioral response
#
# Maps all manually annotated early-window response types
# (`site_type_relabel` in `early_acoustic_window.csv`) onto whether each
# site×pair cell has a late behavioral response in `filtered_manifest.csv`
# (`behav @late` non-null in at least one word-end row).
#
# **Unit:** site×pair cell (subject × electrode × phoneme_pair). An electrode
# contributing bm, dn, and pb appears three times. Total = 99 cells.
#
# **Late window definition:** `behav @late` non-null for ≥1 word-end row
# in the filtered manifest. "@ac slightly late" is excluded by design;
# refine if needed.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import pandas as pd
import seaborn as sns

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
output_dir = "outputs/causal46_joined/sankey_early_late"

# %%
Path(output_dir).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load and join

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
    "type2_early_perceptual":          "#59A14F",
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

merged = merged.dropna(subset=["early_category", "early_label", "late_category", "late_label"])

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

# %% [markdown]
# ## Draw decoding stats

# %%
g = sns.catplot(data=merged, x="early_category", y="phon_peak_roc_auc", order=EARLY_TYPES, height=3, aspect=3)
g.set_xticklabels(rotation=25, ha="right")

# %%
sns.displot(data=merged, x="phon_peak_roc_auc", hue="early_category", hue_order=EARLY_TYPES, kind="kde", height=3, aspect=3)


# %% [markdown]
# ## Draw Sankey

# %%
def _ribbon(ax, x0, y0b, y0t, x1, y1b, y1t, color, alpha=0.45):
    """Bezier ribbon between left (x0, [y0b, y0t]) and right (x1, [y1b, y1t])."""
    cx = (x0 + x1) / 2
    verts = [
        (x0, y0b),
        (cx, y0b), (cx, y1b), (x1, y1b),
        (x1, y1t),
        (cx, y1t), (cx, y0t), (x0, y0t),
        (x0, y0b),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none",
                  alpha=alpha, zorder=1)
    )


# Layout
NODE_GAP   = 0.04
NODE_W     = 0.03
X_LEFT     = 0.35
X_RIGHT    = 0.65
PLOT_H     = 1.0

active_left  = [t for t in EARLY_TYPES if (merged["early_category"] == t).any()]
active_right = LATE_ORDER
N_LEFT  = len(active_left)
N_RIGHT = len(active_right)
grand   = len(merged)

left_totals  = {t:  (merged["early_category"] == t).sum()      for t in active_left}
right_totals = {lc: (merged["late_category"] == lc).sum()      for lc in active_right}

left_scale  = (PLOT_H - NODE_GAP * (N_LEFT  - 1)) / grand
right_scale = (PLOT_H - NODE_GAP * (N_RIGHT - 1)) / grand


def _positions(labels, totals, scale):
    pos = {}
    y = 0.0
    for lbl in labels:
        pos[lbl] = y
        y += totals[lbl] * scale + NODE_GAP
    return pos


left_y  = _positions(active_left,  left_totals,  left_scale)
right_y = _positions(active_right, right_totals, right_scale)

fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Nodes
for t in active_left:
    ax.add_patch(Rectangle(
        (X_LEFT - NODE_W / 2, left_y[t]), NODE_W, left_totals[t] * left_scale,
        color=EARLY_COLORS[t], zorder=3,
    ))
for lc in active_right:
    ax.add_patch(Rectangle(
        (X_RIGHT - NODE_W / 2, right_y[lc]), NODE_W, right_totals[lc] * right_scale,
        color=LATE_COLORS[lc], zorder=3,
    ))

# Ribbons (iterate left→right to fill each node from bottom to top)
left_fill  = {t:  0.0 for t in active_left}
right_fill = {lc: 0.0 for lc in active_right}

for t in active_left:
    for lc in active_right:
        n = ((merged["early_category"] == t) & (merged["late_category"] == lc)).sum()
        if n == 0:
            continue
        lh = n * left_scale
        rh = n * right_scale
        y0b = left_y[t]  + left_fill[t]
        y1b = right_y[lc] + right_fill[lc]
        _ribbon(ax,
                X_LEFT  + NODE_W / 2, y0b, y0b + lh,
                X_RIGHT - NODE_W / 2, y1b, y1b + rh,
                EARLY_COLORS[t])
        left_fill[t]   += lh
        right_fill[lc] += rh

# Left labels
for t in active_left:
    h    = left_totals[t] * left_scale
    ymid = left_y[t] + h / 2
    ax.text(
        X_LEFT - NODE_W / 2 - 0.02, ymid,
        f"{EARLY_LABELS[t]}  (n={left_totals[t]})",
        ha="right", va="center", fontsize=9,
    )

# Right labels
for lc in active_right:
    h    = right_totals[lc] * right_scale
    ymid = right_y[lc] + h / 2
    ax.text(
        X_RIGHT + NODE_W / 2 + 0.02, ymid,
        f"{LATE_LABELS[lc]}  (n={right_totals[lc]})",
        ha="left", va="center", fontsize=9,
    )

# Column headers
header_y = PLOT_H + 0.07
ax.text(X_LEFT,  header_y, "Early\nwindow", ha="center", va="bottom",
        fontsize=10, fontweight="bold")
ax.text(X_RIGHT, header_y, "Late\nwindow",  ha="center", va="bottom",
        fontsize=10, fontweight="bold")

ax.set_xlim(0, 1)
ax.set_ylim(-0.05, PLOT_H + 0.13)
ax.axis("off")

plt.tight_layout()
out_path = Path(output_dir) / "sankey_early_late.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
