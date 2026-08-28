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
# **Headline metric**: `maxstat_reject` from `t_tests_all_sr.py`
# (`cell_maxstat_fdr_test` — max-|z| permutation correction per cell, then
# BH-FDR across cells, mirroring `late_integration_maxstat_significance.py`'s
# established method). `best_ci_raw_excludes_zero` (naive, self-selected
# best window, no cross-cell correction) is reported alongside labeled
# CIRCULAR, never as the headline — this is a deviation from the design
# doc's literal Step 3 text (`ci_raw_excludes_zero` as headline), decided by
# the implementer rather than confirmed with Jon beforehand; see
# `t_tests_all_sr.py`'s "NOTE ON HEADLINE CHOICE" and this plan's
# "Implementation notes". Flagged here again so it's visible at the point
# where the numbers get used, not just in a docstring upstream.
#
# BH-FDR here is one family across all tested cells; it is NOT further
# corrected across the electrode-level collapse below (an electrode with
# several tested cells gets several independent chances to pass).
#
# Gated on `t_tests_all_sr_reconciliation.py` passing — this table is only
# meaningful once the AS cells reproduce the existing AS-restricted test (see
# that notebook). Read the reconciliation summary and hard-fail if it did
# not pass.
#
# Also reads `t_tests_all_sr.py`'s `maxstat_floor_check.csv`: BH-FDR can only
# reject a rank-1 p if `p <= alpha/n_cells`, and a permutation p floors at
# `1/(R+1)` — if the floor exceeds that threshold, NO cell can survive
# correction regardless of true effect size. A "NEW CELL = 0" result under
# that condition is flagged (not silently reported as confirmation of
# co-localization) — see `src.causal46_joined.maxstat_floor_check`.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

# %% tags=["parameters"]
b4_per_cell_path = "outputs/causal46_joined/t_tests_all_sr/b4_per_cell.parquet"
sr_site_universe_electrode_path = "outputs/causal46_joined/sr_site_universe/sr_site_universe_electrode_level.csv"
reconciliation_summary_path = "outputs/causal46_joined/t_tests_all_sr_reconciliation/reconciliation_summary.csv"
maxstat_floor_check_path = "outputs/causal46_joined/t_tests_all_sr/maxstat_floor_check.csv"
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
# ## Permutation-floor adequacy (from `t_tests_all_sr.py`)
#
# BH-FDR rejects a rank-1 p only if `p <= alpha/n_cells`; a permutation p
# floors at `1/(R+1)`. If the floor exceeds that threshold, NO cell can EVER
# survive correction regardless of true effect size — a "0 in the new cell"
# result below would look like confirmation of co-localization when it may
# just be permutation censoring. Read once here so it can qualify the
# headline number, not just sit in `t_tests_all_sr.py`'s own report.

# %%
floor_check = pl.read_csv(maxstat_floor_check_path)
floor_limits_rejection = bool(floor_check["floor_limits_rejection"][0])
print(floor_check)
if floor_limits_rejection:
    print("*** permutation floor limits BH-FDR rejection at this family size — "
          "a '0 in the new cell' result below is NOT reliable evidence of "
          "confirmation; it may be permutation-censored. See "
          "t_tests_all_sr/maxstat_floor_check.csv. ***")

# %% [markdown]
# ## Load tested cells

# %%
b4_per_cell = pl.read_parquet(b4_per_cell_path)
b4_per_cell = b4_per_cell.with_columns([
    pl.col("acoustic_significant").fill_null(False),
    pl.col("maxstat_reject").fill_null(False).alias("perceptual_significant"),
    pl.col("best_ci_raw_excludes_zero").fill_null(False).alias("perceptual_significant_circular"),
])
print(f"tested B4 cells: {b4_per_cell.height}")

# %% [markdown]
# ## Cell-level partition (subject x electrode x phoneme_pair x word_end)

# %%
def crosstab(df: pl.DataFrame, perceptual_col: str) -> pl.DataFrame:
    return (
        df.group_by(["acoustic_significant", perceptual_col])
        .agg(pl.len().alias("n_cells"))
        .sort(["acoustic_significant", perceptual_col])
    )


cell_partition_headline = crosstab(b4_per_cell, "perceptual_significant")
cell_partition_naive = crosstab(b4_per_cell, "perceptual_significant_circular")
print("Cell-level partition (headline: maxstat + BH-FDR):")
print(cell_partition_headline)
print("\nCell-level partition (CIRCULAR — naive best-window CI, not a test — for comparison only):")
print(cell_partition_naive)

cell_partition_headline.write_csv(OUT_DIR / "partition_cell_level_headline.csv")
cell_partition_naive.write_csv(OUT_DIR / "partition_cell_level_naive.csv")

# The new cell: perceptually significant, NOT acoustic-significant.
new_cell_n = b4_per_cell.filter(
    (~pl.col("acoustic_significant")) & pl.col("perceptual_significant")
).height
new_cell_circular_n = b4_per_cell.filter(
    (~pl.col("acoustic_significant")) & pl.col("perceptual_significant_circular")
).height
print(f"\nNEW CELL — perceptual-significant AND NOT acoustic-significant "
      f"(headline: maxstat + BH-FDR): {new_cell_n} / {b4_per_cell.height}")
print(f"NEW CELL — same, CIRCULAR (naive, not a test):                    "
      f"{new_cell_circular_n} / {b4_per_cell.height}")
if new_cell_n == 0 and floor_limits_rejection:
    print("*** NEW CELL = 0, but the permutation floor limits rejection at this "
          "family size — do NOT read this as confirmation of co-localization. "
          "See the floor-adequacy section above. ***")

# %% [markdown]
# ## Electrode-level partition (OR over phoneme_pair x word_end cells)
#
# Collapses the tested-cell population to one row per (subject,
# electrode_idx): an electrode is `acoustic_significant_electrode` /
# `perceptual_significant_electrode` if ANY of its tested cells qualifies.
# This is the population comparable to the AS-restricted pipeline's
# electrode-level counts. NOT BH-FDR corrected at this collapsed level — see
# module docstring.

# %%
electrode_partition = (
    b4_per_cell
    .group_by(["subject", "electrode_idx"])
    .agg(
        pl.col("acoustic_significant").any().alias("acoustic_significant_electrode"),
        pl.col("perceptual_significant").any().alias("perceptual_significant_electrode"),
        pl.col("perceptual_significant_circular").any().alias("perceptual_significant_electrode_circular"),
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
print("\nElectrode-level 2x2 (headline: maxstat + BH-FDR):")
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
# printed for reference, never asserted (see param comment / plan
# Implementation notes on why this dev-container run can't reproduce "64").

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
    "Electrode-level 2x2 (headline: maxstat + BH-FDR):",
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
    f"  headline (maxstat + BH-FDR): {new_cell_electrodes} / {n_sr_tested}",
    "",
    "Cell-level (subject x electrode x phoneme_pair x word_end):",
    f"  NEW CELL headline (maxstat + BH-FDR): {new_cell_n} / {b4_per_cell.height}",
    f"  NEW CELL circular (naive, not a test): {new_cell_circular_n} / {b4_per_cell.height}",
]
if new_cell_n == 0 and floor_limits_rejection:
    lines += [
        "",
        "  *** WARNING: NEW CELL = 0, but the permutation floor limits BH-FDR",
        "  rejection at this family size (see below) — this is NOT reliable",
        "  evidence of confirmation. It may be permutation-censored. ***",
    ]
lines += [
    "",
    f"Permutation-floor adequacy (from t_tests_all_sr.py):",
    f"  floor = {floor_check['floor'][0]:.2e}   "
    f"rank-1 BH threshold = {floor_check['rank1_bh_threshold'][0]:.2e}",
    f"  min p = {floor_check['min_p'][0]:.2e}   "
    f"cells pinned at floor = {int(floor_check['n_at_floor'][0])}",
    f"  floor_limits_rejection = {floor_limits_rejection}",
    "",
    "Caveats:",
    "  - Headline metric (maxstat + BH-FDR) is an implementer decision beyond",
    "    the design doc's literal Step 3 text; not confirmed with Jon before",
    "    running. See t_tests_all_sr.py and the plan's Implementation notes.",
    "  - BH-FDR family = all tested cells; NOT further corrected across the",
    "    electrode-level collapse above.",
    "  - 'circular' = per-window CI at the best self-selected window; per",
    "    late_integration_maxstat_significance.py's own standard, this must",
    "    not be reported as a test. Kept for comparison only.",
    "  - Denominator is the TESTED population, not the full SR universe",
    "    (see dropped count above).",
    "  - If floor_limits_rejection is True, a null (0) result above may be",
    "    permutation-censored, not genuine — see the warning above if",
    "    triggered, and consider a higher n_bootstrap before trusting it.",
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
print(f"  cell-level:      {new_cell_n} / {b4_per_cell.height}  (headline: maxstat + BH-FDR)")
print(f"  electrode-level: {new_cell_electrodes} / {n_sr_tested}  (headline: maxstat + BH-FDR)")
print(f"See {OUT_DIR}")
print("=" * 70)
