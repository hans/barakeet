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
# For each B4 cell `(subject, electrode_idx, phoneme_pair, word_end)` whose
# **site** passes the **perceptual-projection gate** (uncorrected one-tailed
# pooled `p_one_tailed < gate_alpha`), find time window(s) in
# **`[t=0, phon_smax]`** carrying a reliable within-completion behavioral (percept)
# contrast. This is the mirror of `behavioral_discriminative_windows.py`: that
# notebook searches *beyond* the acoustic peak (`smin ≥ phon_smax`); this one
# searches *up to and including* the acoustic window.
#
# **Gate change (2026-07-20):** the site-selection gate is the automated
# `early_perceptual_projection` statistic (π), replacing the manual `behav @ac`
# annotation. The gate is per *site* (subject, electrode_idx, phoneme_pair),
# pooled across completions; both of a passing site's B4 cells enter the window
# search. "If any site passes" is the empty-case guard — no passing site ⇒ zero
# rows. See docs/adr/0001-early-perceptual-window-gate.md.
#
# **Algorithm summary** (see plan for full rationale):
# 1. Gate: sites with valid π and `p_one_tailed < gate_alpha` from the per-subject
#    projection `site_results.csv`; select both B4 cells of each passing site.
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
#      docs/superpowers/plans/2026-07-16-early-perceptual-projection-spec.md

# %% tags=["parameters"]
b4_bootstrap_path = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
projection_results_dir = "outputs/causal46_joined/early_perceptual_projection"
early_annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
a_windows_path = "outputs/causal46_joined/acoustic_endpoint_windows/a_windows.parquet"
outdir = "outputs/causal46_joined/early_perceptual_windows"

gate_alpha = 0.05
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
# ## Projection gate: select cells whose site passes the projection
#
# The site-selection gate is the automated `early_perceptual_projection` statistic
# (π), not the manual `behav @ac` annotation. A *site*
# `(subject, electrode_idx, phoneme_pair)` passes when its pooled π is valid
# (`pi_pooled` not null) and its uncorrected one-tailed p is below `gate_alpha`.
# The gate is one-tailed (π > 0, report tracks the acoustic direction) and
# **uncorrected** — no BH-FDR (Test 1 in the aggregate applies FDR separately).
# Both B4 cells (completions) of each passing site enter the window search.

# %%
# early_annotation_df carries the manual site_type_relabel labels, still used
# below for the type1 (acoustic-only) comparison group — not for the gate.
early_annotation_df = pl.read_csv(early_annotations_path)
print(f"early_annotation_df: {early_annotation_df.height} rows, cols: {early_annotation_df.columns}")

# %%
# Load per-subject projection site_results.csv and concatenate. Reading the
# per-subject files (not the aggregate all_sites.csv) keeps the gate independent
# of the aggregate's FDR pass. Skip empty (0-site) subject CSVs.
proj_paths = sorted(Path(projection_results_dir).glob("*/site_results.csv"))
assert proj_paths, (
    f"No projection site_results.csv under {projection_results_dir}. "
    "Run early_perceptual_projection first."
)
_proj_frames = []
for p in proj_paths:
    try:
        df = pl.read_csv(p)
    except pl.exceptions.NoDataError:
        # 0-site subjects write an empty CSV (no header) — skip.
        continue
    if df.height > 0:
        _proj_frames.append(df)
projection = pl.concat(_proj_frames, how="diagonal_relaxed") if _proj_frames else pl.DataFrame()
print(f"projection site_results: {projection.height} rows from {len(proj_paths)} subjects")

# Passing sites: valid pooled π AND uncorrected one-tailed p < gate_alpha.
passing = projection.filter(
    pl.col("pi_pooled").is_not_null() & (pl.col("p_one_tailed") < gate_alpha)
)
SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]
passing_sites = passing.select(SITE_KEYS).with_columns(
    pl.col("electrode_idx").cast(pl.Int64)
).unique()
# Carry the gate values (site-level) for the output schema.
site_gate_stats: dict[tuple, tuple[float, float]] = {
    (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"]): (
        float(r["pi_pooled"]), float(r["p_one_tailed"])
    )
    for r in passing.iter_rows(named=True)
}
print(f"passing sites (one-tailed p < {gate_alpha}): {passing_sites.height}")

# Select both completions of each passing site: semi-join b4_per_cell on site keys.
n_before = b4_per_cell.height
b4_per_cell = b4_per_cell.join(passing_sites, on=SITE_KEYS, how="semi")
print(f"b4_per_cell after projection gate: {b4_per_cell.height} / {n_before} cells")

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
    "phon_smin", "phon_smax", "n_per_class", "acoustic_peak_auc", "R",
    "pi_pooled", "p_one_tailed",
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
    site_pi, site_p = site_gate_stats.get((subj, eidx, pp), (np.nan, np.nan))

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
            "pi_pooled": site_pi,
            "p_one_tailed": site_p,
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
        "pi_pooled": pl.Float64,
        "p_one_tailed": pl.Float64,
    })

missing_cols = set(EXPECTED_SUMMARY_COLS) - set(ep_windows.columns)
assert not missing_cols, f"ep_windows missing expected columns: {missing_cols}"

ep_windows.write_parquet(OUT_DIR / "ep_windows.parquet")

print(f"ep_windows: {ep_windows.height} rows")
if ep_windows.height > 0:
    print(f"  ci_excludes_zero: {ep_windows['ci_excludes_zero'].sum()}")

    print(ep_windows.select(CELL_KEYS + ["window_id", "smin", "smax", "ci_excludes_zero", "pi_pooled", "p_one_tailed"]))

# %% [markdown]
# ## Type1 (acoustic-only) windows — from acoustic_endpoint_windows
#
# Load pre-computed unified endpoint windows (step6 − step1, unambiguous trials)
# produced by `acoustic_endpoint_windows.py` and filter to the type1 subset.
# (SITE_KEYS defined above in the projection-gate section.)

# %%
a_windows = pl.read_parquet(a_windows_path)
type1_sites = (
    early_annotation_df
    .filter(pl.col("site_type_relabel") == "type1_acoustic_only")
    .select(SITE_KEYS)
    .with_columns(pl.col("electrode_idx").cast(pl.Int64))
    .unique()
)
type1_windows = a_windows.join(type1_sites, on=SITE_KEYS, how="semi")
print(f"a_windows: {a_windows.height} rows; type1 subset: {type1_windows.height} rows")

# The projection gate now *defines* the Perceptual set: every ep_windows row is a
# projection-passing cell, so no site_type_relabel re-filter. One row per cell.
# %%
ep_windows_perceptual = ep_windows.group_by(CELL_KEYS).first()

# %%
ep_windows_acoustic = type1_windows

# %%
window_size = GRID_WINDOW_SIZE
plot_df = pd.concat([
    ep_windows_acoustic.select(["smin", "smax"]).to_pandas().assign(**{"Site type": "Acoustic"}),
    ep_windows_perceptual.select(["smin", "smax"]).to_pandas().assign(**{"Site type": "Perceptual"})
])
plot_df["window_center_sec"] = ((plot_df["smin"] + plot_df["smax"]) / 2) / epoch_sfreq + epoch_tmin
g = sns.displot(data=plot_df,
    x="window_center_sec", hue="Site type", kind="kde", common_norm=False,
    height=2.5, aspect=2, lw=3
)
g.set_axis_labels("Peak center (sec after word onset)", "Density")
g.legend.set_bbox_to_anchor((0.7, 0.7))

# %%
from scipy.stats import ttest_ind

ttest_ind(plot_df.query("`Site type` == 'Acoustic'")["window_center_sec"], plot_df.query("`Site type` == 'Perceptual'")["window_center_sec"])

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
