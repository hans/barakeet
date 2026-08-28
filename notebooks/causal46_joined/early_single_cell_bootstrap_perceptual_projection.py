# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Early single-cell bootstrap: recovering perceptual-projection sites
#
# `early_perceptual_projection.py` calls a site "early perceptual"
# (`early_response_class == "type2_aligned"`) via a deterministic projection
# statistic (`pi_pooled = <a_hat, p_pooled>`, permutation null, one-tailed
# `p_one_tailed < 0.05`). `early_perceptual_windows.py` then runs the B4
# **single-cell bootstrap** (`bootstrap_cell` / `b4_bootstrap.parquet`) in the
# early region `[t=0, phon_smax]` — but only on cells whose site *already*
# passed the projection gate (`b4_per_cell.join(passing_sites, ..., how="semi")`).
# That's a one-way gate: bootstrap-significant cells are a subset of
# projection-passing sites by construction, so it can never independently
# confirm or contradict the projection's site calls.
#
# This notebook removes that gate. For every B4-qualified cell belonging to an
# `A_significant` site (the same universe `early_perceptual_projection.py`
# itself starts from — `included_sites`, before any class split), it searches
# `[t=0, phon_smax]` for a significant single-cell bootstrap window using the
# EXISTING precomputed `b4_bootstrap.parquet` replicates — no new bootstrap
# computation, no epoch access. It then asks: does this ungated bootstrap test
# recover the same site list (`type2_aligned`) the projection method found,
# and at what false-positive rate against `acoustic_only`/`neither` sites?
#
# This is the mirror of `late_single_step_perceptual_projection.py`'s
# late-window check — there we asked whether a *restricted* version of the
# pooled projection test survives; here we ask whether an *independent*
# method (bootstrap CI, not projection+permutation) agrees with the
# projection's site classification at all.
#
# Standalone notebook, not wired into the Snakefile — re-analysis of already
# -computed pipeline outputs only.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import summarize_replicate_array  # noqa: E402
from _windows import _find_maximal_runs  # noqa: E402

# %% tags=["parameters"]
# Site pool: A_significant sites — same universe early_perceptual_projection.py
# itself starts from (included_sites), before any early_response_class split.
site_pool_path = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv"

# early_response_class per site (type2_aligned / acoustic_only / neither),
# from the deterministic projection method — what we're checking recovery of.
site_class_path = "outputs/causal46_joined/early_perceptual_projection/site_class.parquet"

# The existing B4 single-cell bootstrap (t_tests.py bootstrap_cell) — reused
# unchanged, just re-sliced to the early candidate window with no projection
# pre-gate.
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
b4_bootstrap_path = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"

outdir = "outputs/causal46_joined/early_single_cell_bootstrap_perceptual_projection"

# "aligned" (default): mean_diff_aligned, one-tailed (ci_excludes_zero AND
#   median > 0) — the direct analog of the projection's one-tailed pi>0 gate.
#   Cells where acoustic_preferred_class was undetermined (no clear
#   single-peak acoustic tuning; bootstrap_cell writes NaN for every replicate)
#   are reported separately, not folded into "not significant".
# "raw": mean_diff_raw, two-tailed ci_excludes_zero — unsigned "any effect"
#   test, for sites without a clean single-peak acoustic direction.
direction_mode = "aligned"

ci_low = 2.5
ci_high = 97.5

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]
CELL_KEYS = SITE_KEYS + ["word_end"]

assert direction_mode in {"aligned", "raw"}
VALUE_COL = {"aligned": "mean_diff_aligned", "raw": "mean_diff_raw"}[direction_mode]

# t=0 in sample space (matches early_perceptual_windows.py's SAMPLE_T0).
SAMPLE_T0 = int(round((0.0 - epoch_tmin) * epoch_sfreq))
print(f"direction_mode={direction_mode}  VALUE_COL={VALUE_COL}  SAMPLE_T0={SAMPLE_T0}")

# %% [markdown]
# ## Load inputs

# %%
site_pool = pd.read_csv(site_pool_path)
included_sites = (
    site_pool[site_pool["A_significant"]][SITE_KEYS]
    .reset_index(drop=True)
)
print(f"A_significant sites (included_sites universe): {len(included_sites)}")

site_class = pd.read_parquet(site_class_path)
print(f"site_class: {len(site_class)} rows; early_response_class counts:")
print(site_class["early_response_class"].value_counts())

# %%
b4_per_cell = pl.read_parquet(b4_per_cell_path)
b4_bootstrap = pl.read_parquet(b4_bootstrap_path)
print(f"b4_per_cell: {b4_per_cell.height} rows   b4_bootstrap: {b4_bootstrap.height:,} rows")

included_pl = pl.from_pandas(included_sites).with_columns(pl.col("electrode_idx").cast(pl.Int64))
b4_pc_incl = b4_per_cell.join(included_pl, on=SITE_KEYS, how="semi")
print(f"b4_per_cell cells within A_significant universe (no projection gate): {b4_pc_incl.height}")

n_sites_with_cell = b4_pc_incl.select(SITE_KEYS).unique().height
print(f"A_significant sites with >=1 B4-qualified cell: {n_sites_with_cell} / {len(included_sites)}")

# %% [markdown]
# ## Global grid validation

# %%
all_grid_windows: list[tuple[int, int]] = sorted(
    {(int(r[0]), int(r[1])) for r in b4_bootstrap.select(["smin", "smax"]).iter_rows()},
    key=lambda t: t[0],
)
assert len(all_grid_windows) >= 1, "b4_bootstrap has no windows."
widths = {smax - smin for smin, smax in all_grid_windows}
assert len(widths) == 1, f"Non-uniform grid window widths: {widths}"
GRID_WINDOW_SIZE = next(iter(widths))
print(f"Grid OK: {len(all_grid_windows)} windows, width={GRID_WINDOW_SIZE}, "
      f"range=[{all_grid_windows[0][0]}, {all_grid_windows[-1][1]})")

# %% [markdown]
# ## Per-cell: ungated single-cell bootstrap search over `[t=0, phon_smax]`
#
# Same candidate-window + union-of-adjacent-significant-windows logic as
# `early_perceptual_windows.py` (`_find_maximal_runs`), applied to every
# B4-qualified cell in the `A_significant` universe — not just
# projection-passing ones.

# %%
_boot_partitioned: dict[tuple, pl.DataFrame] = {}
for row in b4_pc_incl.iter_rows(named=True):
    key = (row["subject"], row["electrode_idx"], row["phoneme_pair"], row["word_end"])
    if key not in _boot_partitioned:
        _boot_partitioned[key] = b4_bootstrap.filter(
            (pl.col("subject") == key[0]) & (pl.col("electrode_idx") == key[1]) &
            (pl.col("phoneme_pair") == key[2]) & (pl.col("word_end") == key[3])
        )

# %%
cell_rows: list[dict] = []

for cell_row in b4_pc_incl.iter_rows(named=True):
    subj, eidx, pp, we = (
        cell_row["subject"], cell_row["electrode_idx"],
        cell_row["phoneme_pair"], cell_row["word_end"],
    )
    phon_smin, phon_smax = int(cell_row["phon_smin"]), int(cell_row["phon_smax"])
    key = (subj, eidx, pp, we)

    base = dict(subject=subj, electrode_idx=eidx, phoneme_pair=pp, word_end=we,
                phon_smin=phon_smin, phon_smax=phon_smax)

    cell_boot = _boot_partitioned.get(key)
    if cell_boot is None or cell_boot.height == 0:
        cell_rows.append({**base, "status": "missing_bootstrap"})
        continue

    # Candidate windows: [SAMPLE_T0, phon_smax] — "that early region".
    cand_windows = [(s, x) for s, x in all_grid_windows if s >= SAMPLE_T0 and x <= phon_smax]
    if not cand_windows:
        cell_rows.append({**base, "status": "no_candidate_windows"})
        continue

    cand_smins = {s for s, _ in cand_windows}
    cell_cand_boot = cell_boot.filter(pl.col("smin").is_in(list(cand_smins)))

    if direction_mode == "aligned":
        # bootstrap_cell writes NaN for every mean_diff_aligned replicate when
        # acoustic_preferred_class is undetermined (tied endpoint means) —
        # i.e. the site has no clear single-peak acoustic direction to align
        # to. Report separately rather than as "not significant".
        vals_all = cell_cand_boot[VALUE_COL].to_numpy()
        if vals_all.size == 0 or np.all(np.isnan(vals_all)):
            cell_rows.append({**base, "status": "undetermined_direction"})
            continue

    w_medians: dict[int, float] = {}
    w_sig: dict[int, bool] = {}
    for smin, smax in cand_windows:
        arr = cell_cand_boot.filter(pl.col("smin") == smin)[VALUE_COL].to_numpy()
        if arr.size == 0:
            continue
        stats = summarize_replicate_array(arr, ci_low=ci_low, ci_high=ci_high)
        w_medians[smin] = stats["median"]
        if direction_mode == "aligned":
            # One-tailed analog of the projection's pi > 0 gate.
            w_sig[smin] = bool(stats["ci_excludes_zero"] and stats["median"] > 0)
        else:
            w_sig[smin] = bool(stats["ci_excludes_zero"])

    cand_windows = [(s, x) for s, x in cand_windows if s in w_medians]
    if not cand_windows:
        cell_rows.append({**base, "status": "no_candidate_windows"})
        continue

    sig_windows = [(s, x) for s, x in cand_windows if w_sig[s]]
    if not sig_windows:
        cell_rows.append({**base, "status": "tested", "cell_significant": False,
                           "n_sig_windows": 0})
        continue

    union_list = _find_maximal_runs(sig_windows, w_medians)
    best_union = max(union_list, key=lambda comp: abs(w_medians[comp[0][0]]))
    peak_smin = max(best_union, key=lambda w: abs(w_medians[w[0]]))[0]

    cell_rows.append({
        **base, "status": "tested", "cell_significant": True,
        "n_sig_windows": len(sig_windows),
        "n_unions": len(union_list),
        "best_union_smin": best_union[0][0], "best_union_smax": best_union[-1][1],
        "best_union_median": w_medians[peak_smin],
    })

cell_df = pd.DataFrame(cell_rows)
cell_df["tmin"] = np.nan
cell_df["tmax"] = np.nan
_has_union = cell_df.get("best_union_smin") is not None
if _has_union:
    mask = cell_df["best_union_smin"].notna()
    cell_df.loc[mask, "tmin"] = cell_df.loc[mask, "best_union_smin"] / epoch_sfreq + epoch_tmin
    cell_df.loc[mask, "tmax"] = cell_df.loc[mask, "best_union_smax"] / epoch_sfreq + epoch_tmin

print(f"\nCells processed: {len(cell_df)}")
print(cell_df["status"].value_counts())
print("\ncell_significant among status=='tested':")
print(cell_df.loc[cell_df["status"] == "tested", "cell_significant"].value_counts())

# %%
cell_df.to_parquet(OUT_DIR / "early_bootstrap_cells.parquet", index=False)

# %% [markdown]
# ## Site-level rollup and cross-check against `early_response_class`
#
# Site-level "recovered" = OR across a site's word_ends among **tested**
# cells (same convention as `late_category`/`single_step_site` elsewhere in
# this codebase). A site is **untestable** only if none of its B4-qualified
# cells reached `status == "tested"` (e.g. no candidate window, or — in
# `aligned` mode — undetermined acoustic direction at every cell).

# %%
tested = cell_df[cell_df["status"] == "tested"]
site_tested = (
    tested.groupby(SITE_KEYS)
    .agg(n_cells_tested=("cell_significant", "size"),
         any_significant=("cell_significant", "any"))
    .reset_index()
)

site_summary = included_sites.merge(site_tested, on=SITE_KEYS, how="left")
site_summary["n_cells_tested"] = site_summary["n_cells_tested"].fillna(0).astype(int)
site_summary["bootstrap_status"] = np.select(
    [site_summary["n_cells_tested"] == 0, site_summary["any_significant"] == True],  # noqa: E712
    ["untestable", "recovered"],
    default="not_recovered",
)

merged = site_summary.merge(
    site_class[SITE_KEYS + ["early_response_class"]], on=SITE_KEYS, how="left"
)
merged["early_response_class"] = merged["early_response_class"].fillna("unclassified")
merged.to_parquet(OUT_DIR / "early_bootstrap_sites.parquet", index=False)

print("Site bootstrap_status counts:")
print(merged["bootstrap_status"].value_counts())

print("\n=== Cross-tab: early_response_class (projection) x bootstrap_status ===")
ct = pd.crosstab(merged["early_response_class"], merged["bootstrap_status"])
print(ct)

# %%
testable = merged[merged["bootstrap_status"] != "untestable"]

n_type2 = int((testable["early_response_class"] == "type2_aligned").sum())
n_type2_recovered = int((
    (testable["early_response_class"] == "type2_aligned")
    & (testable["bootstrap_status"] == "recovered")
).sum())
n_type2_untestable = int((
    (merged["early_response_class"] == "type2_aligned")
    & (merged["bootstrap_status"] == "untestable")
).sum())

n_other = int(testable["early_response_class"].isin(["acoustic_only", "neither"]).sum())
n_other_fp = int((
    testable["early_response_class"].isin(["acoustic_only", "neither"])
    & (testable["bootstrap_status"] == "recovered")
).sum())

print("=== Recovery summary (ungated single-cell bootstrap vs. projection's type2_aligned) ===")
print(f"Recall:            {n_type2_recovered} / {n_type2}  type2_aligned sites recovered (testable)")
print(f"Untestable type2:  {n_type2_untestable}")
print(f"False positives:   {n_other_fp} / {n_other}  acoustic_only+neither sites also flagged (testable)")
print(f"  acoustic_only false positives: "
      f"{int(((testable.early_response_class=='acoustic_only') & (testable.bootstrap_status=='recovered')).sum())}"
      f" / {int((testable.early_response_class=='acoustic_only').sum())}")
print(f"  neither false positives:       "
      f"{int(((testable.early_response_class=='neither') & (testable.bootstrap_status=='recovered')).sum())}"
      f" / {int((testable.early_response_class=='neither').sum())}")
