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
# # Acoustic discriminative windows
#
# Companion to `behavioral_discriminative_windows.py`. For each B4 cell
# `(subject, electrode_idx, phoneme_pair, word_end)`, discover time window(s)
# carrying a reliable **acoustic-step** HGA contrast on ambiguous trials. Pure
# post-processing over `b4_acoustic_bootstrap.parquet` (from
# `acoustic_on_ambiguous.py`) — no epoch reload.
#
# The acoustic-step bootstrap contrasts the extreme qualifying steps
# `s_hi − s_lo` with behavioral report balanced 50/50 within each step, so the
# contrast isolates the acoustic-cue effect independent of reported percept.
# Reference is `mean_diff_raw` (= `mean_diff_aligned`; step order fixes polarity),
# **s_hi − s_lo**: positive = stronger HGA to the higher (more /n/-like) step.
#
# **Disanalogies vs the behavioral notebook** (see plan):
# - Candidate region is the **full range `[onset, PAIR_SMAX]`** — the acoustic
#   peak is *included* (no `smin >= phon_smax` post-acoustic restriction).
# - **No manifest filter**: every acoustic-ok cell in `b4_acoustic_per_cell` is
#   processed.
# - **No decoder-window placement** (the acoustic decoder window already exists
#   as causal6 `phon_smin`/`phon_smax`).
# - **Fallback off by default** (`use_fallback=False`): unfiltered discovery, so a
#   cell with no significant window emits zero rows (matches
#   `early_perceptual_windows.py`).
#
# **Algorithm** (mirror of behavioral):
# 1. Validate grid is contiguous with stride==window_size.
# 2. Candidate windows: `smin >= SAMPLE_T0` (t=0, word onset).
# 3. Significant windows: bootstrap CI of `mean_diff_raw` excludes zero.
# 4. Union runs: maximal groups of adjacent + significant + same-sign windows.
# 5. Union β = per-replicate mean of component `mean_diff_raw` values.
# 6. (Optional) fallback when no significant window: seed at max |median|.
#
# See: docs/superpowers/plans/2026-07-07-causal46-acoustic-discriminative-windows.md

# %% tags=["parameters"]
ac_bootstrap_path = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_bootstrap.parquet"
ac_per_cell_path = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_per_cell.parquet"
outdir = "outputs/causal46_joined/acoustic_discriminative_windows"

ci_low = 2.5
ci_high = 97.5
use_fallback = False

# %%
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.stimuli import OFFSET_DICT
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import summarize_replicate_array  # noqa: E402
from _windows import _fallback_run, _find_maximal_runs, _window_sign  # noqa: E402

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

# t=0 in sample space: round((0 - epoch_tmin) * epoch_sfreq) = round(0.4 * 100) = 40.
# The acoustic bootstrap grid starts at sample 0 (= t−0.4 s, pre-onset baseline);
# we exclude everything before onset to avoid spurious pre-onset candidates.
SAMPLE_T0 = int(round((0.0 - epoch_tmin) * epoch_sfreq))
print(f"SAMPLE_T0 = {SAMPLE_T0} (t=0 in samples, epoch_tmin={epoch_tmin}, sfreq={epoch_sfreq})")
print(f"use_fallback = {use_fallback}")

# %% [markdown]
# ## Load and validate inputs

# %%
ac_bootstrap = pl.read_parquet(ac_bootstrap_path)
ac_per_cell = pl.read_parquet(ac_per_cell_path)

print(f"ac_bootstrap: {ac_bootstrap.height:,} rows, cols: {ac_bootstrap.columns}")
print(f"ac_per_cell:  {ac_per_cell.height} rows, cols: {ac_per_cell.columns}")

# phon_smin/phon_smax/s_lo/s_hi are joined onto b4_acoustic_per_cell from the
# acoustic cell manifest (acoustic_on_ambiguous.py). Fail loudly if absent.
for col in ("phon_smin", "phon_smax", "s_lo", "s_hi", "n_per_class", "acoustic_peak_auc"):
    assert col in ac_per_cell.columns, (
        f"ac_per_cell missing '{col}'. Re-run acoustic_on_ambiguous with complete data."
    )

# %% [markdown]
# ## Global grid validation
#
# Asserts stride == window_size and contiguity from the parquet.

# %%
all_grid_windows: list[tuple[int, int]] = sorted(
    {(int(r[0]), int(r[1])) for r in ac_bootstrap.select(["smin", "smax"]).iter_rows()},
    key=lambda t: t[0],
)
assert len(all_grid_windows) >= 1, "ac_bootstrap has no windows."

widths = [smax - smin for smin, smax in all_grid_windows]
assert len(set(widths)) == 1, (
    f"Non-uniform grid window widths detected: {set(widths)}. "
    "Requires stride==window_size and a contiguous grid."
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
# Pre-index ac_bootstrap by cell key for fast per-cell slicing (N_cells is small).
_boot_partitioned: dict[tuple, pl.DataFrame] = {}
for row in ac_per_cell.iter_rows(named=True):
    key = (row["subject"], row["electrode_idx"], row["phoneme_pair"], row["word_end"])
    if key not in _boot_partitioned:
        _boot_partitioned[key] = ac_bootstrap.filter(
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
    "beta_acoustic_mean", "beta_acoustic_median",
    "beta_acoustic_ci_low", "beta_acoustic_ci_high",
    "ci_excludes_zero", "is_fallback",
    "phon_smin", "phon_smax", "acoustic_peak_auc", "n_per_class", "R",
    "s_lo", "s_hi", "post_word_offset",
]
EXPECTED_BOOT_COLS = [
    "subject", "electrode_idx", "phoneme_pair", "word_end",
    "window_id", "replicate", "beta",
]

summary_rows: list[dict] = []
bootstrap_rows: list[dict] = []
n_no_candidates = 0
n_no_sig = 0
n_fallback = 0

for cell_row in ac_per_cell.iter_rows(named=True):
    subj       = cell_row["subject"]
    eidx       = int(cell_row["electrode_idx"])
    pp         = cell_row["phoneme_pair"]
    we         = cell_row["word_end"]
    phon_smin  = int(cell_row["phon_smin"])
    phon_smax  = int(cell_row["phon_smax"])
    n_per_class       = int(cell_row["n_per_class"])
    acoustic_peak_auc = float(cell_row["acoustic_peak_auc"])
    s_lo = int(cell_row["s_lo"])
    s_hi = int(cell_row["s_hi"])

    cell_boot = _boot_partitioned.get((subj, eidx, pp, we))
    if cell_boot is None or cell_boot.height == 0:
        warnings.warn(f"No bootstrap data for {subj} e{eidx} {pp} {we}")
        continue

    R = int(cell_boot["replicate"].max()) + 1  # replicates are 0-indexed

    # Per-WE word-offset in sample space (for the post_word_offset flag only —
    # candidate windows are NOT restricted to it; full-range discovery).
    we_offset_s = OFFSET_DICT.get(we)
    we_offset_sample: int | None = (
        int(round((we_offset_s - epoch_tmin) * epoch_sfreq))
        if we_offset_s is not None else None
    )

    # Candidate windows: full range from word onset (SAMPLE_T0), acoustic peak
    # included. No post-acoustic (phon_smax) lower bound.
    cand_windows = [
        (smin, smax) for smin, smax in all_grid_windows
        if smin >= SAMPLE_T0
    ]
    if not cand_windows:
        n_no_candidates += 1
        continue

    # Filter bootstrap to candidate windows only (efficiency + safety).
    cand_smins_set = {smin for smin, _ in cand_windows}
    cell_cand_boot = cell_boot.filter(pl.col("smin").is_in(list(cand_smins_set)))

    # Per-window significance via bootstrap CI on mean_diff_raw.
    w_medians: dict[int, float] = {}
    w_ci_excl_zero: dict[int, bool] = {}
    for smin, smax in cand_windows:
        arr = cell_cand_boot.filter(pl.col("smin") == smin)["mean_diff_raw"].to_numpy()
        if arr.size == 0:
            continue
        stats = summarize_replicate_array(arr, ci_low=ci_low, ci_high=ci_high)
        w_medians[smin] = stats["median"]
        w_ci_excl_zero[smin] = stats["ci_excludes_zero"]

    # Keep only windows that have data.
    cand_windows = [(smin, smax) for smin, smax in cand_windows if smin in w_medians]
    if not cand_windows:
        n_no_candidates += 1
        continue

    sig_windows = [(smin, smax) for smin, smax in cand_windows if w_ci_excl_zero[smin]]

    if sig_windows:
        union_list = _find_maximal_runs(sig_windows, w_medians)
        is_fallback = False
    elif use_fallback:
        union_list = [_fallback_run(cand_windows, w_medians)]
        is_fallback = True
        n_fallback += 1
    else:
        # No significant window and fallback disabled → emit zero rows.
        n_no_sig += 1
        continue

    for window_id, comp_windows in enumerate(union_list):
        component_smins = [smin for smin, _ in comp_windows]
        union_smin = comp_windows[0][0]
        union_smax = comp_windows[-1][1]
        n_comp = len(comp_windows)

        # Union β: per-replicate mean across component windows. For a contiguous
        # equal-width grid this is bit-identical to re-running the bootstrap with
        # the same seeds.
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

        post_wo = bool(we_offset_sample is not None and union_smax > we_offset_sample)

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
            "beta_acoustic_mean": stats["mean"],
            "beta_acoustic_median": stats["median"],
            "beta_acoustic_ci_low": stats["ci_lo"],
            "beta_acoustic_ci_high": stats["ci_hi"],
            "ci_excludes_zero": stats["ci_excludes_zero"],
            "is_fallback": is_fallback,
            "phon_smin": phon_smin,
            "phon_smax": phon_smax,
            "acoustic_peak_auc": acoustic_peak_auc,
            "n_per_class": n_per_class,
            "R": R,
            "s_lo": s_lo,
            "s_hi": s_hi,
            "post_word_offset": post_wo,
        })

        rep_arr = union_beta_df["replicate"].to_numpy()
        for rep, beta in zip(rep_arr, beta_arr):
            bootstrap_rows.append({
                "subject": subj,
                "electrode_idx": eidx,
                "phoneme_pair": pp,
                "word_end": we,
                "window_id": window_id,
                "replicate": int(rep),
                "beta": float(beta),
            })

print(
    f"Cells: {ac_per_cell.height} total, "
    f"{n_no_candidates} with no candidates, "
    f"{n_no_sig} with no significant window (fallback off), "
    f"{n_fallback} fallback cells"
)

# %% [markdown]
# ## Write outputs

# %%
if summary_rows:
    ad_windows = pl.DataFrame(summary_rows)
    ad_windows_boot = pl.DataFrame(bootstrap_rows)
else:
    # Empty output — preserve expected schema for downstream consumers.
    ad_windows = pl.DataFrame(
        {col: pl.Series([], dtype=pl.Utf8) for col in EXPECTED_SUMMARY_COLS}
    ).cast({
        "electrode_idx": pl.Int64,
        "window_id": pl.Int64,
        "smin": pl.Int64,
        "smax": pl.Int64,
        "n_component_windows": pl.Int64,
        "component_smins": pl.List(pl.Int64),
        "sign": pl.Int64,
        "beta_acoustic_mean": pl.Float64,
        "beta_acoustic_median": pl.Float64,
        "beta_acoustic_ci_low": pl.Float64,
        "beta_acoustic_ci_high": pl.Float64,
        "ci_excludes_zero": pl.Boolean,
        "is_fallback": pl.Boolean,
        "phon_smin": pl.Int64,
        "phon_smax": pl.Int64,
        "acoustic_peak_auc": pl.Float64,
        "n_per_class": pl.Int64,
        "R": pl.Int64,
        "s_lo": pl.Int64,
        "s_hi": pl.Int64,
        "post_word_offset": pl.Boolean,
    })
    ad_windows_boot = pl.DataFrame(
        {col: pl.Series([], dtype=pl.Utf8) for col in EXPECTED_BOOT_COLS}
    ).cast({
        "electrode_idx": pl.Int64,
        "window_id": pl.Int64,
        "replicate": pl.Int64,
        "beta": pl.Float64,
    })

missing_cols = set(EXPECTED_SUMMARY_COLS) - set(ad_windows.columns)
assert not missing_cols, f"ad_windows missing expected columns: {missing_cols}"

ad_windows.write_parquet(OUT_DIR / "ad_windows.parquet")
ad_windows_boot.write_parquet(OUT_DIR / "ad_windows_bootstrap.parquet")

print(f"ad_windows:           {ad_windows.height} rows")
print(f"ad_windows_bootstrap: {ad_windows_boot.height} rows")
if ad_windows.height > 0:
    print(f"  ci_excludes_zero:  {ad_windows['ci_excludes_zero'].sum()}")
    print(f"  is_fallback:       {ad_windows['is_fallback'].sum()}")
    print(f"  post_word_offset:  {ad_windows['post_word_offset'].sum()}")
    print(ad_windows.select(CELL_KEYS + ["window_id", "smin", "smax", "ci_excludes_zero"]))

# %% [markdown]
# ## Optional QC figures

# %%
try:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Acoustic discriminative windows — QC", fontsize=11)

    # Panel 1: windows per cell
    ax = axes[0, 0]
    if ad_windows.height > 0:
        n_wins_per_cell = (
            ad_windows.group_by(CELL_KEYS).len()["len"].to_numpy()
        )
        mx = int(n_wins_per_cell.max())
        ax.hist(n_wins_per_cell, bins=range(1, mx + 2), align="left", color="darkorange")
    ax.set_xlabel("Windows per cell")
    ax.set_ylabel("Count")
    ax.set_title("Windows per B4 cell")

    # Panel 2: β_acoustic_median distribution
    ax = axes[0, 1]
    if ad_windows.height > 0:
        betas = ad_windows["beta_acoustic_median"].to_numpy()
        excl = ad_windows["ci_excludes_zero"].to_numpy()
        ax.hist(betas[~excl], bins=30, alpha=0.5, label="CI ∩ 0", color="gray")
        ax.hist(betas[excl], bins=30, alpha=0.7, label="CI excl. 0", color="darkorange")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=7)
    ax.set_xlabel("β_acoustic_median (s_hi − s_lo)")
    ax.set_title("β distribution")

    # Panel 3: timing scatter, acoustic-peak bounds marked
    ax = axes[1, 0]
    if ad_windows.height > 0:
        t_smin = ad_windows["smin"].to_numpy() / epoch_sfreq + epoch_tmin
        t_smax = ad_windows["smax"].to_numpy() / epoch_sfreq + epoch_tmin
        t_psmin = ad_windows["phon_smin"].to_numpy() / epoch_sfreq + epoch_tmin
        t_psmax = ad_windows["phon_smax"].to_numpy() / epoch_sfreq + epoch_tmin
        excl = ad_windows["ci_excludes_zero"].to_numpy().astype(float)
        sc = ax.scatter(t_smin, t_smax, c=excl, cmap="coolwarm", vmin=0, vmax=1,
                        s=25, alpha=0.8)
        for ps, pe in zip(t_psmin, t_psmax):
            ax.axvline(ps, color="green", lw=0.3, alpha=0.3)
            ax.axvline(pe, color="orange", lw=0.3, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="ci_excludes_zero")
    ax.set_xlabel("Window start (s)")
    ax.set_ylabel("Window end (s)")
    ax.set_title("Timing (green=phon_smin, orange=phon_smax)")

    # Panel 4: summary counts
    ax = axes[1, 1]
    if ad_windows.height > 0:
        n_fb = int(ad_windows["is_fallback"].sum())
        n_sig = int(ad_windows["ci_excludes_zero"].sum())
        n_tot = ad_windows.height
        ax.bar(["Total", "CI excl. 0", "Fallback"], [n_tot, n_sig, n_fb],
               color=["gray", "darkorange", "steelblue"])
        ax.set_ylabel("Windows")
    ax.set_title("Counts")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ad_windows_summary.pdf")
    plt.close(fig)
    print("Saved ad_windows_summary.pdf")
except Exception as _qc_exc:
    warnings.warn(f"QC PDF skipped: {_qc_exc}")
