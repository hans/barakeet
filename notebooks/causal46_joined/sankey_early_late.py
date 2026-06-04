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
# Maps five manually annotated early-window response types
# (`site_type_relabel` in `early_acoustic_window.csv`) onto whether each
# site×pair cell has a late behavioral response in `filtered_manifest.csv`
# (`behav @late` non-null in at least one word-end row).
#
# **Unit:** site×pair cell (subject × electrode × phoneme_pair). An electrode
# contributing bm, dn, and pb appears three times.
#
# **Late window definition:** `behav @late` non-null for ≥1 word-end row
# in the filtered manifest. "@ac slightly late" is excluded by design;
# refine if needed.
#
# **Status note:** ~8 `status=unclassifiable_B_power` type1 cells are
# included — their early-window type is well-defined even though the
# behavioral window classification was unreliable.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import pandas as pd

# %% tags=["parameters"]
annotations_path = "outputs_prod/causal46_joined/manual_annotations/early_acoustic_window.csv"
filtered_manifest_path = "outputs_prod/causal46_joined/manual_annotations/filtered_manifest.csv"
output_dir = "outputs/causal46_joined/sankey_early_late"

# %%
Path(output_dir).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load and join

# %%
early = pd.read_csv(annotations_path)
manifest = pd.read_csv(filtered_manifest_path)

EARLY_TYPES = [
    "type1_acoustic_only",
    "type2_early_perceptual",
    "type3_asymmetric",
    "type4_early_perceptual_mirrored",
    "type5_behav_only",
]
EARLY_LABELS = {
    "type1_acoustic_only":              "Acoustic only",
    "type2_early_perceptual":           "Acoustic+perceptual",
    "type3_asymmetric":                 "Acoustic+perceptual\n(one-sided)",
    "type4_early_perceptual_mirrored":  "Acoustic+perceptual\n(mirrored)",
    "type5_behav_only":                 "Perceptual only",
}
EARLY_COLORS = {
    "type1_acoustic_only":             "#4E79A7",
    "type2_early_perceptual":          "#59A14F",
    "type3_asymmetric":                "#F28E2B",
    "type4_early_perceptual_mirrored": "#B07AA1",
    "type5_behav_only":                "#E15759",
}
LATE_COLORS = {True: "#76B7B2", False: "#BAB0AC"}
LATE_LABELS = {True: "Late window\npresent", False: "Late window\nabsent"}

# %%
early_filt = early[early["site_type_relabel"].isin(EARLY_TYPES)].copy()

late_pres = (
    manifest
    .groupby(["subject", "electrode_idx", "phoneme_pair"])["behav @late"]
    .apply(lambda x: x.notna().any())
    .reset_index()
    .rename(columns={"behav @late": "late_present"})
)

merged = early_filt.merge(
    late_pres, on=["subject", "electrode_idx", "phoneme_pair"], how="left"
)
merged["late_present"] = merged["late_present"].fillna(False)

print(f"Site×pair cells included: {len(merged)}")
print("\nFlow table (rows=early type, cols=late present):")
ct = (
    merged
    .groupby(["site_type_relabel", "late_present"])
    .size()
    .unstack(fill_value=0)
    .reindex(EARLY_TYPES)
)
print(ct)

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

active_left  = [t for t in EARLY_TYPES if (merged["site_type_relabel"] == t).any()]
active_right = [True, False]
N_LEFT  = len(active_left)
N_RIGHT = len(active_right)
grand   = len(merged)

left_totals  = {t:  (merged["site_type_relabel"] == t).sum()  for t in active_left}
right_totals = {lp: (merged["late_present"] == lp).sum()      for lp in active_right}

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

fig, ax = plt.subplots(figsize=(7, 6))

# Nodes
for t in active_left:
    ax.add_patch(Rectangle(
        (X_LEFT - NODE_W / 2, left_y[t]), NODE_W, left_totals[t] * left_scale,
        color=EARLY_COLORS[t], zorder=3,
    ))
for lp in active_right:
    ax.add_patch(Rectangle(
        (X_RIGHT - NODE_W / 2, right_y[lp]), NODE_W, right_totals[lp] * right_scale,
        color=LATE_COLORS[lp], zorder=3,
    ))

# Ribbons (iterate left→right to fill each node from bottom to top)
left_fill  = {t:  0.0 for t in active_left}
right_fill = {lp: 0.0 for lp in active_right}

for t in active_left:
    for lp in active_right:
        n = ((merged["site_type_relabel"] == t) & (merged["late_present"] == lp)).sum()
        if n == 0:
            continue
        lh = n * left_scale
        rh = n * right_scale
        y0b = left_y[t]  + left_fill[t]
        y1b = right_y[lp] + right_fill[lp]
        _ribbon(ax,
                X_LEFT  + NODE_W / 2, y0b, y0b + lh,
                X_RIGHT - NODE_W / 2, y1b, y1b + rh,
                EARLY_COLORS[t])
        left_fill[t]   += lh
        right_fill[lp] += rh

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
for lp in active_right:
    h    = right_totals[lp] * right_scale
    ymid = right_y[lp] + h / 2
    ax.text(
        X_RIGHT + NODE_W / 2 + 0.02, ymid,
        f"{LATE_LABELS[lp]}  (n={right_totals[lp]})",
        ha="left", va="center", fontsize=9,
    )

# Column headers
header_y = PLOT_H + 0.07
ax.text(X_LEFT,  header_y, "Early window", ha="center", va="bottom",
        fontsize=10, fontweight="bold")
ax.text(X_RIGHT, header_y, "Late window",  ha="center", va="bottom",
        fontsize=10, fontweight="bold")

ax.set_xlim(0, 1)
ax.set_ylim(-0.05, PLOT_H + 0.13)
ax.axis("off")

plt.tight_layout()
out_path = Path(output_dir) / "sankey_early_late.pdf"
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
