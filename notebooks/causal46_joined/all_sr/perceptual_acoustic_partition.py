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
# # Perceptual x acoustic partition — the scientifically load-bearing output
#
# Step 4 of the all-SR perceptual fork
# (`docs/superpowers/plans/2026-08-27-all-speech-responsive-perceptual.md`).
#
# Cross-tabulates perceptual significance against `acoustic_significant`,
# across ALL speech-responsive sites. The new cell — **perceptually
# significant but NOT acoustic-significant** — directly bears on the
# CLAUDE.md claim that distal integration is "ruled out by co-localization
# (91%)": that claim was only ever testable within the AS-restricted
# population, which structurally cannot see a non-acoustic perceptual site.
# Either outcome here is a result:
#
# - a large non-acoustic-but-perceptual cell REVISES the co-localization claim
# - a near-empty one CONFIRMS it more strongly than the AS-restricted design
#   could, since this design can actually see the alternative
#
# **Headline metric**: `best_ci_raw_excludes_zero` from `t_tests_all_sr.py`
# — the same raw, uncorrected bootstrap CI that `t_tests.py`'s output is
# consumed as everywhere downstream in the real pipeline (`plot_for_paper`,
# `behavioral_discriminative_windows.py`, `late_perceptual_projection.py`'s
# candidate gate). No maxstat / BH-FDR / TFCE correction is applied here —
# an earlier draft added one, modeled on `late_integration_maxstat_significance.py`,
# but that notebook has no Snakefile rule and doesn't feed `plot_for_paper`;
# it isn't the pipeline's actual precedent for this statistic. Matching how
# `t_tests.py`'s own output is used elsewhere keeps this fork's headline
# consistent with the rest of the codebase instead of inventing a new
# standard for it alone.
#
# Gated on `t_tests_all_sr_reconciliation.py` passing — this table is only
# meaningful once the AS cells reproduce the existing AS-restricted test (see
# that notebook). Read the reconciliation summary and hard-fail if it did
# not pass.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# %% tags=["parameters"]
b4_per_cell_path = "outputs/causal46_joined/t_tests_all_sr/b4_per_cell.parquet"
sr_site_universe_electrode_path = "outputs/causal46_joined/sr_site_universe/sr_site_universe_electrode_level.csv"
reconciliation_summary_path = "outputs/causal46_joined/t_tests_all_sr_reconciliation/reconciliation_summary.csv"
outdir = "outputs/causal46_joined/perceptual_acoustic_partition"

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

paper_reported_as_electrode_n = 44

# %% [markdown]
# ## Gate: reconciliation must have passed

# %%
recon = pl.read_csv(reconciliation_summary_path)
recon_passed = bool(recon["passed"][0])
print(recon)
if not recon_passed:
    raise AssertionError(
        "perceptual_acoustic_partition: t_tests_all_sr_reconciliation did NOT "
        "pass. The AS subset of this pipeline does not reproduce t_tests.py, "
        "so this partition would compare two different tests, not one test "
        "partitioned. Fix the reconciliation first "
        "(outputs/causal46_joined/t_tests_all_sr_reconciliation/)."
    )
print("✓ reconciliation passed — partition is meaningful")

# %% [markdown]
# ## Load tested cells

# %%
b4_per_cell = pl.read_parquet(b4_per_cell_path)
b4_per_cell = b4_per_cell.with_columns([
    pl.col("acoustic_significant").fill_null(False),
    pl.col("best_ci_raw_excludes_zero").fill_null(False).alias("perceptual_significant"),
])
print(f"tested B4 cells: {b4_per_cell.height}")

# %% [markdown]
# ## Cell-level partition (subject x electrode x phoneme_pair x word_end)

# %%
cell_partition = (
    b4_per_cell
    .group_by(["acoustic_significant", "perceptual_significant"])
    .agg(pl.len().alias("n_cells"))
    .sort(["acoustic_significant", "perceptual_significant"])
)
print("Cell-level partition:")
print(cell_partition)
cell_partition.write_csv(OUT_DIR / "partition_cell_level.csv")

# The new cell: perceptually significant, NOT acoustic-significant.
new_cell_n = b4_per_cell.filter(
    (~pl.col("acoustic_significant")) & pl.col("perceptual_significant")
).height
print(f"\nNEW CELL — perceptual-significant AND NOT acoustic-significant: "
      f"{new_cell_n} / {b4_per_cell.height}")

# %% [markdown]
# ## Electrode-level partition (OR over phoneme_pair x word_end cells)
#
# Collapses the tested-cell population to one row per (subject,
# electrode_idx): an electrode is `acoustic_significant_electrode` /
# `perceptual_significant_electrode` if ANY of its tested cells qualifies.
# This is the population comparable to the AS-restricted pipeline's
# electrode-level counts.

# %%
electrode_partition = (
    b4_per_cell
    .group_by(["subject", "electrode_idx"])
    .agg(
        pl.col("acoustic_significant").any().alias("acoustic_significant_electrode"),
        pl.col("perceptual_significant").any().alias("perceptual_significant_electrode"),
        pl.len().alias("n_cells_tested"),
    )
)
electrode_partition.write_csv(OUT_DIR / "partition_electrode_level.csv")
print(f"\nelectrodes with ≥1 tested B4 cell: {electrode_partition.height}")

electrode_2x2 = (
    electrode_partition
    .group_by(["acoustic_significant_electrode", "perceptual_significant_electrode"])
    .agg(pl.len().alias("n_electrodes"))
    .sort(["acoustic_significant_electrode", "perceptual_significant_electrode"])
)
print("\nElectrode-level 2x2:")
print(electrode_2x2)
electrode_2x2.write_csv(OUT_DIR / "partition_electrode_level_2x2.csv")

new_cell_electrodes = electrode_partition.filter(
    (~pl.col("acoustic_significant_electrode")) & pl.col("perceptual_significant_electrode")
).height
print(f"\nNEW CELL (electrode-level) — perceptual-significant, NOT "
      f"acoustic-significant: {new_cell_electrodes} / {electrode_partition.height}")

# %% [markdown]
# ## Denominator context — against the full SR / AS site universe
#
# `t_tests_all_sr`'s tested population may be smaller than the full SR
# universe (cells drop for underpowered trial counts, missing epochs, or a
# search range narrower than one window) — report both counts rather than
# silently treating "tested" as "all SR". `paper_reported_as_electrode_n` is
# printed for reference, never asserted.

# %%
sr_electrode_level = pl.read_csv(sr_site_universe_electrode_path)
n_sr_total = sr_electrode_level.height
n_as_total = int(sr_electrode_level["acoustic_significant_electrode"].sum())
n_sr_tested = electrode_partition.height
n_dropped = n_sr_total - n_sr_tested

print(f"SR electrodes in universe:        {n_sr_total}")
print(f"  of which acoustic_significant:  {n_as_total}  "
      f"(paper-reported AS denominator, for reference only: {paper_reported_as_electrode_n})")
print(f"SR electrodes with ≥1 tested cell: {n_sr_tested}  "
      f"(dropped from universe: {n_dropped})")

# %% [markdown]
# ## Report

# %%
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.axis("off")
lines = [
    "Perceptual x acoustic partition — all speech-responsive sites",
    "",
    f"SR electrodes in universe: {n_sr_total}   (acoustic_significant: {n_as_total};",
    f"  paper-reported AS denominator, reference only: {paper_reported_as_electrode_n})",
    f"SR electrodes with ≥1 tested B4 cell: {n_sr_tested}  (dropped: {n_dropped})",
    "",
    "Electrode-level 2x2:",
]
for row in electrode_2x2.iter_rows(named=True):
    lines.append(
        f"  acoustic_significant={row['acoustic_significant_electrode']!s:5}  "
        f"perceptual_significant={row['perceptual_significant_electrode']!s:5}  "
        f"n={row['n_electrodes']}"
    )
lines += [
    "",
    "NEW CELL (perceptual, NOT acoustic) — electrode-level:",
    f"  {new_cell_electrodes} / {n_sr_tested}",
    "",
    "Cell-level (subject x electrode x phoneme_pair x word_end):",
    f"  NEW CELL: {new_cell_n} / {b4_per_cell.height}",
    "",
    "Caveats:",
    "  - perceptual_significant = best_ci_raw_excludes_zero (raw bootstrap CI,",
    "    self-selected best window), matching how t_tests.py's output is used",
    "    everywhere else downstream (plot_for_paper, behavioral_discriminative_windows,",
    "    late_perceptual_projection's candidate gate). No cross-window or",
    "    cross-cell multiple-comparisons correction is applied to it, same as",
    "    the rest of the pipeline.",
    "  - Denominator is the TESTED population, not the full SR universe",
    "    (see dropped count above).",
]
ax.text(0.03, 0.97, "\n".join(lines), ha="left", va="top",
        family="monospace", fontsize=9.5)
pdf_path = OUT_DIR / "partition_report.pdf"
fig.savefig(pdf_path)
plt.close(fig)
print(f"\nwrote {pdf_path}")

# %% [markdown]
# ## Done

# %%
print("=" * 70)
print("Perceptual x acoustic partition summary")
print("=" * 70)
print("NEW CELL (perceptual-significant, NOT acoustic-significant):")
print(f"  cell-level:      {new_cell_n} / {b4_per_cell.height}")
print(f"  electrode-level: {new_cell_electrodes} / {n_sr_tested}")
print(f"See {OUT_DIR}")
print("=" * 70)
