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
# # Early perceptual windows
#
# For each B4 cell `(subject, electrode_idx, phoneme_pair, word_end)` that has a
# **`behav @ac`** annotation in the filtered manifest, find time window(s) in
# **`[t=0, phon_smax]`** carrying a reliable within-completion behavioral (percept)
# contrast. This is the mirror of `behavioral_discriminative_windows.py`: that
# notebook searches *beyond* the acoustic peak (`smin ≥ phon_smax`); this one
# searches *up to and including* the acoustic window.
#
# **Algorithm summary** (see plan for full rationale):
# 1. Filter to cells annotated `behav @ac` in the manual manifest.
# 2. Candidate windows: `smin >= SAMPLE_T0` (=40, corresponding to t=0) AND
#    `smax <= phon_smax` (acoustic-peak window end from b4_per_cell).
# 3. Significant windows: bootstrap CI of `mean_diff_raw` excludes zero.
# 4. Union runs: maximal groups of adjacent + significant + same-sign windows.
# 5. **No fallback**: cells with no significant window in this region emit zero rows.
# 6. Union β = per-replicate mean of component `mean_diff_raw` values.
#
# Reference fixed: /n/−/d/ (`mean_diff_raw`); never `mean_diff_aligned`.
#
# See: docs/superpowers/plans/2026-06-20-causal46-early-perceptual-windows.md

# %% tags=["parameters"]
b4_bootstrap_path = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
filtered_manifest_path = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv"
early_annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
outdir = "outputs/causal46_joined/early_perceptual_windows"

ci_low = 2.5
ci_high = 97.5

# %%
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import summarize_replicate_array  # noqa: E402
from _windows import _find_maximal_runs, _window_sign  # noqa: E402

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

# t=0 in sample space: round((0 - epoch_tmin) * epoch_sfreq) = round(0.4 * 100) = 40
SAMPLE_T0 = int(round((0.0 - epoch_tmin) * epoch_sfreq))
print(f"SAMPLE_T0 = {SAMPLE_T0} (t=0 in samples, epoch_tmin={epoch_tmin}, sfreq={epoch_sfreq})")

# %% [markdown]
# ## Load and validate inputs

# %%
b4_bootstrap = pl.read_parquet(b4_bootstrap_path)
b4_per_cell = pl.read_parquet(b4_per_cell_path)

print(f"b4_bootstrap: {b4_bootstrap.height:,} rows, cols: {b4_bootstrap.columns}")
print(f"b4_per_cell:  {b4_per_cell.height} rows, cols: {b4_per_cell.columns}")

for col in ("phon_smin", "phon_smax"):
    assert col in b4_per_cell.columns, (
        f"b4_per_cell missing '{col}'. "
        "Re-run t_tests with complete data."
    )

# %% [markdown]
# ## Filter cells to those with a `behav @ac` annotation
#
# Only cells with a non-null `behav @ac` value in the filtered manifest qualify.
# The letter encodes the behavioral category with higher HGA at the acoustic window.

# %%
early_annotation_df = pl.read_csv(early_annotations_path)
print(f"early_annotation_df: {early_annotation_df.height} rows, cols: {early_annotation_df.columns}")

# %%
manifest = pl.read_csv(filtered_manifest_path)
print(f"filtered_manifest: {manifest.height} rows")

behav_ac = manifest.filter(pl.col("behav @ac").is_not_null())
behav_keys: set[tuple] = {
    (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"], r["word_end"])
    for r in behav_ac.iter_rows(named=True)
}
# Carry the tuning letter per cell for the output schema.
behav_ac_tuning: dict[tuple, str] = {
    (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"], r["word_end"]): r["behav @ac"]
    for r in behav_ac.iter_rows(named=True)
}
print(f"cells with behav @ac: {len(behav_keys)}")

b4_per_cell_all = b4_per_cell  # full table, needed for type1 comparison below
n_before = b4_per_cell_all.height
b4_per_cell = b4_per_cell_all.filter(
    pl.struct(["subject", "electrode_idx", "phoneme_pair", "word_end"]).map_elements(
        lambda r: (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"], r["word_end"])
                  in behav_keys,
        return_dtype=pl.Boolean,
    )
)
print(f"b4_per_cell after manifest filter: {b4_per_cell.height} / {n_before} cells")

# %% [markdown]
# ## Global grid validation
#
# Assert stride == window_size and contiguity.

# %%
all_grid_windows: list[tuple[int, int]] = sorted(
    {(int(r[0]), int(r[1])) for r in b4_bootstrap.select(["smin", "smax"]).iter_rows()},
    key=lambda t: t[0],
)
assert len(all_grid_windows) >= 1, "b4_bootstrap has no windows."

widths = [smax - smin for smin, smax in all_grid_windows]
assert len(set(widths)) == 1, (
    f"Non-uniform grid window widths detected: {set(widths)}. "
    "Plan requires stride==window_size and a contiguous grid."
)
GRID_WINDOW_SIZE = widths[0]

for i in range(len(all_grid_windows) - 1):
    assert all_grid_windows[i][1] == all_grid_windows[i + 1][0], (
        f"Grid gap between {all_grid_windows[i]} and {all_grid_windows[i+1]}. "
        "Grid must be contiguous (smax_i == smin_{i+1})."
    )

print(
    f"Grid OK: {len(all_grid_windows)} windows, width={GRID_WINDOW_SIZE}, "
    f"range=[{all_grid_windows[0][0]}, {all_grid_windows[-1][1]})"
)

# %%
# Pre-index b4_bootstrap by cell key for fast per-cell slicing.
_boot_partitioned: dict[tuple, pl.DataFrame] = {}
for row in b4_per_cell.iter_rows(named=True):
    key = (row["subject"], row["electrode_idx"], row["phoneme_pair"], row["word_end"])
    if key not in _boot_partitioned:
        _boot_partitioned[key] = b4_bootstrap.filter(
            (pl.col("subject") == key[0]) &
            (pl.col("electrode_idx") == key[1]) &
            (pl.col("phoneme_pair") == key[2]) &
            (pl.col("word_end") == key[3])
        )

# %% [markdown]
# ## Per-cell processing

# %%
EXPECTED_SUMMARY_COLS = [
    "subject", "electrode_idx", "phoneme_pair", "word_end", "window_id",
    "smin", "smax", "n_component_windows", "component_smins", "sign",
    "beta_ambig_median", "beta_ambig_ci_low", "beta_ambig_ci_high", "ci_excludes_zero",
    "phon_smin", "phon_smax", "n_per_class", "acoustic_peak_auc", "R", "behav_ac_tuning",
]

summary_rows: list[dict] = []
n_no_candidates = 0
n_no_sig = 0

for cell_row in b4_per_cell.iter_rows(named=True):
    subj       = cell_row["subject"]
    eidx       = int(cell_row["electrode_idx"])
    pp         = cell_row["phoneme_pair"]
    we         = cell_row["word_end"]
    phon_smin  = int(cell_row["phon_smin"])
    phon_smax  = int(cell_row["phon_smax"])
    n_per_class       = int(cell_row["n_per_class"])
    acoustic_peak_auc = float(cell_row["acoustic_peak_auc"])

    cell_boot = _boot_partitioned.get((subj, eidx, pp, we))
    if cell_boot is None or cell_boot.height == 0:
        warnings.warn(f"No bootstrap data for {subj} e{eidx} {pp} {we}")
        continue

    R = int(cell_boot["replicate"].max()) + 1  # replicates are 0-indexed

    # Candidate windows: [SAMPLE_T0, phon_smax].
    # smin >= SAMPLE_T0 and smax <= phon_smax.
    cand_windows = [
        (smin, smax) for smin, smax in all_grid_windows
        if smin >= SAMPLE_T0 and smax <= phon_smax
    ]
    if not cand_windows:
        n_no_candidates += 1
        continue

    # Filter bootstrap to candidate windows only.
    cand_smins_set = {smin for smin, _ in cand_windows}
    cell_cand_boot = cell_boot.filter(pl.col("smin").is_in(list(cand_smins_set)))

    # Per-window significance via bootstrap CI.
    w_medians: dict[int, float] = {}
    w_ci_excl_zero: dict[int, bool] = {}
    for smin, smax in cand_windows:
        arr = cell_cand_boot.filter(pl.col("smin") == smin)["mean_diff_raw"].to_numpy()
        if arr.size == 0:
            continue
        stats = summarize_replicate_array(arr, ci_low=ci_low, ci_high=ci_high)
        w_medians[smin] = stats["median"]
        w_ci_excl_zero[smin] = stats["ci_excludes_zero"]

    cand_windows = [(smin, smax) for smin, smax in cand_windows if smin in w_medians]
    if not cand_windows:
        n_no_candidates += 1
        continue

    sig_windows = [(smin, smax) for smin, smax in cand_windows if w_ci_excl_zero[smin]]

    if not sig_windows:
        # No fallback: cells with no significant window emit zero rows.
        n_no_sig += 1
        continue

    union_list = _find_maximal_runs(sig_windows, w_medians)
    cell_tuning = behav_ac_tuning.get((subj, eidx, pp, we), "")

    for window_id, comp_windows in enumerate(union_list):
        component_smins = [smin for smin, _ in comp_windows]
        union_smin = comp_windows[0][0]
        union_smax = comp_windows[-1][1]
        n_comp = len(comp_windows)

        # Union β: per-replicate mean across component windows.
        union_boot = cell_cand_boot.filter(pl.col("smin").is_in(component_smins))
        assert union_boot.height == R * n_comp, (
            f"Expected {R * n_comp} rows (R={R} × {n_comp} component windows) "
            f"for {subj} e{eidx} {pp} {we} union {component_smins}, "
            f"got {union_boot.height}. Check that all component windows ran."
        )
        union_beta_df = (
            union_boot
            .group_by("replicate")
            .agg(pl.col("mean_diff_raw").mean().alias("beta"))
            .sort("replicate")
        )
        beta_arr = union_beta_df["beta"].to_numpy()
        assert len(beta_arr) == R, f"Expected {R} replicate β values, got {len(beta_arr)}"

        stats = summarize_replicate_array(beta_arr, ci_low=ci_low, ci_high=ci_high)

        summary_rows.append({
            "subject": subj,
            "electrode_idx": eidx,
            "phoneme_pair": pp,
            "word_end": we,
            "window_id": window_id,
            "smin": union_smin,
            "smax": union_smax,
            "n_component_windows": n_comp,
            "component_smins": component_smins,
            "sign": _window_sign(stats["median"]),
            "beta_ambig_median": stats["median"],
            "beta_ambig_ci_low": stats["ci_lo"],
            "beta_ambig_ci_high": stats["ci_hi"],
            "ci_excludes_zero": stats["ci_excludes_zero"],
            "phon_smin": phon_smin,
            "phon_smax": phon_smax,
            "n_per_class": n_per_class,
            "acoustic_peak_auc": acoustic_peak_auc,
            "R": R,
            "behav_ac_tuning": cell_tuning,
        })

print(
    f"Cells: {b4_per_cell.height} total, "
    f"{n_no_candidates} with no candidate windows, "
    f"{n_no_sig} with no significant window (no fallback)"
)

# %% [markdown]
# ## Write outputs

# %%
if summary_rows:
    ep_windows = pl.DataFrame(summary_rows)
else:
    # Empty output — preserve expected schema for downstream consumers.
    ep_windows = pl.DataFrame(
        {col: pl.Series([], dtype=pl.Utf8) for col in EXPECTED_SUMMARY_COLS}
    ).cast({
        "electrode_idx": pl.Int64,
        "window_id": pl.Int64,
        "smin": pl.Int64,
        "smax": pl.Int64,
        "n_component_windows": pl.Int64,
        "component_smins": pl.List(pl.Int64),
        "sign": pl.Int64,
        "beta_ambig_median": pl.Float64,
        "beta_ambig_ci_low": pl.Float64,
        "beta_ambig_ci_high": pl.Float64,
        "ci_excludes_zero": pl.Boolean,
        "phon_smin": pl.Int64,
        "phon_smax": pl.Int64,
        "n_per_class": pl.Int64,
        "acoustic_peak_auc": pl.Float64,
        "R": pl.Int64,
    })

missing_cols = set(EXPECTED_SUMMARY_COLS) - set(ep_windows.columns)
assert not missing_cols, f"ep_windows missing expected columns: {missing_cols}"

ep_windows.write_parquet(OUT_DIR / "ep_windows.parquet")

print(f"ep_windows: {ep_windows.height} rows")
if ep_windows.height > 0:
    print(f"  ci_excludes_zero: {ep_windows['ci_excludes_zero'].sum()}")

    print(ep_windows.select(CELL_KEYS + ["window_id", "smin", "smax", "ci_excludes_zero", "behav_ac_tuning"]))

# %% [markdown]
# ## Type1 (acoustic-only) processing — same bootstrap algorithm
#
# Run the identical window-finding procedure on acoustic-only sites so that the
# comparison with early perceptual sites is apples-to-apples: both groups use
# the behavioral bootstrap CI to find their earliest significant window in
# [t=0, phon_smax], rather than mixing bootstrap onsets with decoder-peak onsets.

# %%
type1_site_keys: set[tuple] = {
    (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"])
    for r in early_annotation_df.filter(
        pl.col("site_type_relabel") == "type1_acoustic_only"
    ).iter_rows(named=True)
}

b4_per_cell_type1 = b4_per_cell_all.filter(
    pl.struct(["subject", "electrode_idx", "phoneme_pair"]).map_elements(
        lambda r: (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"])
                  in type1_site_keys,
        return_dtype=pl.Boolean,
    )
)
print(f"b4_per_cell_type1: {b4_per_cell_type1.height} cells")

# %%
_boot_partitioned_type1: dict[tuple, pl.DataFrame] = {}
for row in b4_per_cell_type1.iter_rows(named=True):
    key = (row["subject"], row["electrode_idx"], row["phoneme_pair"], row["word_end"])
    if key not in _boot_partitioned_type1:
        _boot_partitioned_type1[key] = b4_bootstrap.filter(
            (pl.col("subject") == key[0]) &
            (pl.col("electrode_idx") == key[1]) &
            (pl.col("phoneme_pair") == key[2]) &
            (pl.col("word_end") == key[3])
        )

# %%
type1_rows: list[dict] = []
n_type1_no_candidates = 0
n_type1_no_sig = 0

for cell_row in b4_per_cell_type1.iter_rows(named=True):
    subj      = cell_row["subject"]
    eidx      = int(cell_row["electrode_idx"])
    pp        = cell_row["phoneme_pair"]
    we        = cell_row["word_end"]
    phon_smin = int(cell_row["phon_smin"])
    phon_smax = int(cell_row["phon_smax"])

    cell_boot = _boot_partitioned_type1.get((subj, eidx, pp, we))
    if cell_boot is None or cell_boot.height == 0:
        continue

    R = int(cell_boot["replicate"].max()) + 1

    cand_windows = [
        (smin, smax) for smin, smax in all_grid_windows
        if smin >= SAMPLE_T0 and smax <= phon_smax
    ]
    if not cand_windows:
        n_type1_no_candidates += 1
        continue

    cand_smins_set = {smin for smin, _ in cand_windows}
    cell_cand_boot = cell_boot.filter(pl.col("smin").is_in(list(cand_smins_set)))

    w_medians: dict[int, float] = {}
    w_ci_excl_zero: dict[int, bool] = {}
    for smin, smax in cand_windows:
        arr = cell_cand_boot.filter(pl.col("smin") == smin)["mean_diff_raw"].to_numpy()
        if arr.size == 0:
            continue
        stats = summarize_replicate_array(arr, ci_low=ci_low, ci_high=ci_high)
        w_medians[smin] = stats["median"]
        w_ci_excl_zero[smin] = stats["ci_excludes_zero"]

    cand_windows = [(smin, smax) for smin, smax in cand_windows if smin in w_medians]
    if not cand_windows:
        n_type1_no_candidates += 1
        continue

    sig_windows = [(smin, smax) for smin, smax in cand_windows if w_ci_excl_zero[smin]]
    if not sig_windows:
        n_type1_no_sig += 1
        continue

    for window_id, comp_windows in enumerate(_find_maximal_runs(sig_windows, w_medians)):
        component_smins = [smin for smin, _ in comp_windows]
        union_smin = comp_windows[0][0]
        union_smax = comp_windows[-1][1]
        n_comp = len(comp_windows)

        union_boot = cell_cand_boot.filter(pl.col("smin").is_in(component_smins))
        union_beta_df = (
            union_boot
            .group_by("replicate")
            .agg(pl.col("mean_diff_raw").mean().alias("beta"))
            .sort("replicate")
        )
        beta_arr = union_beta_df["beta"].to_numpy()
        stats = summarize_replicate_array(beta_arr, ci_low=ci_low, ci_high=ci_high)

        type1_rows.append({
            "subject": subj,
            "electrode_idx": eidx,
            "phoneme_pair": pp,
            "word_end": we,
            "window_id": window_id,
            "smin": union_smin,
            "smax": union_smax,
            "phon_smin": phon_smin,
            "phon_smax": phon_smax,
            "ci_excludes_zero": stats["ci_excludes_zero"],
        })

type1_windows = pl.DataFrame(type1_rows) if type1_rows else pl.DataFrame(
    {c: pl.Series([], dtype=pl.Utf8)
     for c in ["subject", "electrode_idx", "phoneme_pair", "word_end",
               "window_id", "smin", "smax", "phon_smin", "phon_smax", "ci_excludes_zero"]}
).cast({"electrode_idx": pl.Int64, "window_id": pl.Int64,
        "smin": pl.Int64, "smax": pl.Int64,
        "phon_smin": pl.Int64, "phon_smax": pl.Int64,
        "ci_excludes_zero": pl.Boolean})

print(
    f"Type1 cells: {b4_per_cell_type1.height} total, "
    f"{n_type1_no_candidates} no candidates, "
    f"{n_type1_no_sig} no significant window"
)
print(f"type1_windows: {type1_windows.height} rows")

# %%
ep_windows_perceptual = (
    ep_windows
    .join(early_annotation_df.select(["subject", "electrode_idx", "phoneme_pair", "site_type_relabel"]),
          on=["subject", "electrode_idx", "phoneme_pair"], how="left")
    .filter((pl.col("site_type_relabel") == "type2_early_perceptual") | (pl.col("site_type_relabel") == "type3_asymmetric"))
    .group_by(CELL_KEYS).first()
)

# %%
ep_windows_acoustic = type1_windows.group_by(CELL_KEYS).first()

# %%
from scipy.stats import ttest_ind

ttest_ind(ep_windows_acoustic["smin"].to_numpy(), ep_windows_perceptual["smin"].to_numpy())

# %%
sns.displot(data=pd.concat([
    ep_windows_acoustic.select(["smin"]).to_pandas().assign(type="acoustic"),
    ep_windows_perceptual.select(["smin"]).to_pandas().assign(type="perceptual")]),
    x="smin", hue="type", kind="kde")

# %% [markdown]
# ### Acoustic vs perceptual timing within-site

# %%
ttest_df = ep_windows_perceptual.select(["smin", "smax", "phon_smin", "phon_smax"]).to_pandas()
from scipy.stats import ttest_rel
ttest_res = ttest_rel(ttest_df["smin"], ttest_df["phon_smin"])
print(f"t-test smin vs phon_smin: t={ttest_res.statistic:.3f}, p={ttest_res.pvalue:.3e}")

# %%
g = sns.lmplot(data=ep_windows_perceptual.to_pandas(), x="smin", y="phon_smin")
g.ax.plot(list(g.ax.get_xlim()), list(g.ax.get_xlim()), color="gray", linestyle="--")

# %% [markdown]
# ## Optional QC figures

# %%
try:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Early perceptual windows — QC", fontsize=11)

    # Panel 1: windows per cell
    ax = axes[0]
    if ep_windows.height > 0:
        n_wins_per_cell = (
            ep_windows.group_by(CELL_KEYS).len()["len"].to_numpy()
        )
        mx = int(n_wins_per_cell.max())
        ax.hist(n_wins_per_cell, bins=range(1, mx + 2), align="left", color="steelblue")
    ax.set_xlabel("Windows per cell")
    ax.set_ylabel("Count")
    ax.set_title("Windows per B4 cell")

    # Panel 2: β_ambig_median distribution
    ax = axes[1]
    if ep_windows.height > 0:
        betas = ep_windows["beta_ambig_median"].to_numpy()
        excl = ep_windows["ci_excludes_zero"].to_numpy()
        ax.hist(betas[~excl], bins=20, alpha=0.5, label="CI ∩ 0", color="gray")
        ax.hist(betas[excl], bins=20, alpha=0.7, label="CI excl. 0", color="steelblue")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=7)
    ax.set_xlabel("β_ambig_median (/n/−/d/)")
    ax.set_title("β distribution")

    # Panel 3: timing scatter (smin vs smax, relative to phon_smin/phon_smax)
    ax = axes[2]
    if ep_windows.height > 0:
        t_smin = ep_windows["smin"].to_numpy() / epoch_sfreq + epoch_tmin
        t_smax = ep_windows["smax"].to_numpy() / epoch_sfreq + epoch_tmin
        t_psmin = ep_windows["phon_smin"].to_numpy() / epoch_sfreq + epoch_tmin
        t_psmax = ep_windows["phon_smax"].to_numpy() / epoch_sfreq + epoch_tmin
        excl = ep_windows["ci_excludes_zero"].to_numpy().astype(float)
        sc = ax.scatter(t_smin, t_smax, c=excl, cmap="coolwarm", vmin=0, vmax=1,
                        s=25, alpha=0.8)
        # Mark acoustic window bounds per cell
        for ps, pe in zip(t_psmin, t_psmax):
            ax.axvline(ps, color="green", lw=0.3, alpha=0.3)
            ax.axvline(pe, color="orange", lw=0.3, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="ci_excludes_zero")
    ax.set_xlabel("Window start (s)")
    ax.set_ylabel("Window end (s)")
    ax.set_title("Timing (green=phon_smin, orange=phon_smax)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ep_windows_summary.pdf")
    print("Saved ep_windows_summary.pdf")
except Exception as _qc_exc:
    warnings.warn(f"QC PDF skipped: {_qc_exc}")
