# %% [markdown]
# # Behavioral discriminative windows
#
# For each B4 cell `(subject, electrode_idx, phoneme_pair, word_end)`, infer
# time window(s) **beyond the acoustic peak** that carry a reliable behavioral
# (percept) difference. Pure post-processing over `b4_bootstrap.parquet` —
# no epoch reload needed.
#
# **Algorithm summary** (see plan for full rationale):
# 1. Validate grid is contiguous with stride==window_size (required for union β
#    averaging to be bit-identical to re-running the bootstrap, plan decision 1).
# 2. Candidate windows: `smin >= phon_smax` (acoustic boundary from b4_per_cell).
# 3. Significant windows: bootstrap CI of `mean_diff_raw` excludes zero.
# 4. Union runs: maximal groups of adjacent + significant + same-sign windows.
# 5. Fallback when no significant window: seed at max |median|, grow same-sign.
# 6. Union β = per-replicate mean of component `mean_diff_raw` values.
# 7. Decoder placement for narrow unions (`smax−smin < decoder_window_size`).
#
# Reference fixed: /n/−/d/ (`mean_diff_raw`); never `mean_diff_aligned`.
#
# See: docs/superpowers/plans/2026-06-19-causal46-behavioral-discriminative-windows.md

# %% tags=["parameters"]
b4_bootstrap_path: str = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"
b4_per_cell_path: str = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
outdir: str = "outputs/causal46_joined/behavioral_discriminative_windows"
ci_low: float = 2.5
ci_high: float = 97.5
decoder_window_size: int = 15
manual_override_path: str | None = None

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

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

# %% [markdown]
# ## Load and validate inputs

# %%
b4_bootstrap = pl.read_parquet(b4_bootstrap_path)
b4_per_cell = pl.read_parquet(b4_per_cell_path)

print(f"b4_bootstrap: {b4_bootstrap.height:,} rows, cols: {b4_bootstrap.columns}")
print(f"b4_per_cell:  {b4_per_cell.height} rows, cols: {b4_per_cell.columns}")

# phon_smin/phon_smax are written to b4_per_cell only when both b4_per_pair
# and b4_per_cell are non-empty (t_tests.py:722-732). Fail loudly if absent.
for col in ("phon_smin", "phon_smax"):
    assert col in b4_per_cell.columns, (
        f"b4_per_cell missing '{col}'. "
        "This column is only written when t_tests.py has non-empty paired data "
        "(t_tests.py:722-732). Re-run t_tests with complete data."
    )

# %% [markdown]
# ## Global grid validation
#
# Asserts stride == window_size and contiguity from the parquet (plan decision 1).

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
# Pre-index b4_bootstrap by (subject, electrode_idx, phoneme_pair, word_end)
# for fast per-cell slicing (N_cells is small, ~16–32).
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
# ## Helper functions

# %%
def _window_sign(median: float) -> int:
    return 1 if median >= 0 else -1


def _find_maximal_runs(
    sig_windows: list[tuple[int, int]],
    medians: dict[int, float],
) -> list[list[tuple[int, int]]]:
    """Maximal runs of adjacent + significant + same-sign candidate windows."""
    runs: list[list[tuple[int, int]]] = []
    if not sig_windows:
        return runs
    current = [sig_windows[0]]
    for w in sig_windows[1:]:
        prev = current[-1]
        adjacent = (prev[1] == w[0])
        same_sign = (_window_sign(medians[prev[0]]) == _window_sign(medians[w[0]]))
        if adjacent and same_sign:
            current.append(w)
        else:
            runs.append(current)
            current = [w]
    runs.append(current)
    return runs


def _fallback_run(
    cand_windows: list[tuple[int, int]],
    medians: dict[int, float],
) -> list[tuple[int, int]]:
    """Seed at max |median|; grow over adjacent same-sign windows."""
    abs_meds = [abs(medians[smin]) for smin, _ in cand_windows]
    seed_idx = int(np.argmax(abs_meds))
    seed_sign = _window_sign(medians[cand_windows[seed_idx][0]])

    # Grow left (toward smaller smin)
    left_indices = [seed_idx]
    for i in range(seed_idx - 1, -1, -1):
        if cand_windows[i][1] != cand_windows[i + 1][0]:  # gap
            break
        if _window_sign(medians[cand_windows[i][0]]) != seed_sign:
            break
        left_indices.insert(0, i)

    # Grow right (toward larger smin)
    right_indices = [seed_idx]
    for i in range(seed_idx + 1, len(cand_windows)):
        if cand_windows[i - 1][1] != cand_windows[i][0]:  # gap
            break
        if _window_sign(medians[cand_windows[i][0]]) != seed_sign:
            break
        right_indices.append(i)

    all_indices = sorted(set(left_indices + right_indices))
    return [cand_windows[i] for i in all_indices]


# %% [markdown]
# ## Per-cell processing

# %%
EXPECTED_SUMMARY_COLS = [
    "subject", "electrode_idx", "phoneme_pair", "word_end", "window_id",
    "smin", "smax", "n_component_windows", "component_smins", "sign",
    "beta_ambig_mean", "beta_ambig_median", "beta_ambig_ci_low", "beta_ambig_ci_high",
    "ci_excludes_zero", "is_fallback",
    "phon_smin", "phon_smax", "acoustic_peak_auc", "n_per_class", "R",
    "narrower_than_decoder", "post_word_offset", "behav_decoder_smin", "behav_decoder_smax",
]
EXPECTED_BOOT_COLS = [
    "subject", "electrode_idx", "phoneme_pair", "word_end",
    "window_id", "replicate", "beta",
]

summary_rows: list[dict] = []
bootstrap_rows: list[dict] = []
n_no_candidates = 0
n_fallback = 0

for cell_row in b4_per_cell.iter_rows(named=True):
    subj       = cell_row["subject"]
    eidx       = int(cell_row["electrode_idx"])
    pp         = cell_row["phoneme_pair"]
    we         = cell_row["word_end"]
    phon_smin  = int(cell_row["phon_smin"])
    phon_smax  = int(cell_row["phon_smax"])
    n_per_class        = int(cell_row["n_per_class"])
    acoustic_peak_auc  = float(cell_row["acoustic_peak_auc"])

    cell_boot = _boot_partitioned.get((subj, eidx, pp, we))
    if cell_boot is None or cell_boot.height == 0:
        warnings.warn(f"No bootstrap data for {subj} e{eidx} {pp} {we}")
        continue

    R = int(cell_boot["replicate"].max()) + 1  # replicates are 0-indexed

    # Candidate windows: grid windows with smin >= phon_smax
    cand_windows = [(smin, smax) for smin, smax in all_grid_windows if smin >= phon_smax]
    if not cand_windows:
        n_no_candidates += 1
        continue

    # Filter bootstrap to candidate windows only (efficiency + safety)
    cand_smins_set = {smin for smin, _ in cand_windows}
    cell_cand_boot = cell_boot.filter(pl.col("smin").is_in(list(cand_smins_set)))

    # Per-window summary using np.percentile (linear interpolation).
    # NOTE: t_tests.py per_window_summary uses polars .quantile() (nearest),
    # so CIs here may differ by interpolation at bound-near-zero. Gated on
    # outputs_prod/ mount (see plan "GATING" section).
    w_medians: dict[int, float] = {}
    w_ci_excl_zero: dict[int, bool] = {}
    for smin, smax in cand_windows:
        arr = cell_cand_boot.filter(pl.col("smin") == smin)["mean_diff_raw"].to_numpy()
        if arr.size == 0:
            continue
        stats = summarize_replicate_array(arr, ci_low=ci_low, ci_high=ci_high)
        w_medians[smin] = stats["median"]
        w_ci_excl_zero[smin] = stats["ci_excludes_zero"]

    # Keep only windows that have data
    cand_windows = [(smin, smax) for smin, smax in cand_windows if smin in w_medians]
    if not cand_windows:
        n_no_candidates += 1
        continue

    sig_windows = [(smin, smax) for smin, smax in cand_windows if w_ci_excl_zero[smin]]

    if sig_windows:
        union_list = _find_maximal_runs(sig_windows, w_medians)
        is_fallback = False
    else:
        union_list = [_fallback_run(cand_windows, w_medians)]
        is_fallback = True
        n_fallback += 1

    # Word-end offset sample for post_word_offset flag
    we_offset_s = OFFSET_DICT.get(we)
    we_offset_sample: int | None = (
        int(round((we_offset_s - epoch_tmin) * epoch_sfreq))
        if we_offset_s is not None else None
    )

    for window_id, comp_windows in enumerate(union_list):
        component_smins = [smin for smin, _ in comp_windows]
        union_smin = comp_windows[0][0]
        union_smax = comp_windows[-1][1]
        n_comp = len(comp_windows)

        # Union β: per-replicate mean across component windows.
        # For a contiguous equal-width grid, this is bit-identical to re-running
        # the bootstrap with the same seeds (plan decision 1).
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

        # Decoder window placement (plan §"Behavioral decoder window placement")
        union_width = union_smax - union_smin
        narrower = union_width < decoder_window_size
        if narrower:
            center = round((union_smin + union_smax) / 2.0)
            dec_smin = center - decoder_window_size // 2
            dec_smax = dec_smin + decoder_window_size
            # Lower clamp only: never extend back into the acoustic decoding region
            if dec_smin < phon_smax:
                dec_smin = phon_smax
                dec_smax = phon_smax + decoder_window_size
            behav_dec_smin: int | None = dec_smin
            behav_dec_smax: int | None = dec_smax
        else:
            behav_dec_smin = None
            behav_dec_smax = None

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
            "beta_ambig_mean": stats["mean"],
            "beta_ambig_median": stats["median"],
            "beta_ambig_ci_low": stats["ci_lo"],
            "beta_ambig_ci_high": stats["ci_hi"],
            "ci_excludes_zero": stats["ci_excludes_zero"],
            "is_fallback": is_fallback,
            "phon_smin": phon_smin,
            "phon_smax": phon_smax,
            "acoustic_peak_auc": acoustic_peak_auc,
            "n_per_class": n_per_class,
            "R": R,
            "narrower_than_decoder": narrower,
            "post_word_offset": post_wo,
            "behav_decoder_smin": behav_dec_smin,
            "behav_decoder_smax": behav_dec_smax,
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
    f"Cells: {b4_per_cell.height} total, "
    f"{n_no_candidates} with no candidates, "
    f"{n_fallback} fallback cells"
)

# %% [markdown]
# ## Write outputs

# %%
if summary_rows:
    b_windows = pl.DataFrame(summary_rows)
    b_windows_boot = pl.DataFrame(bootstrap_rows)
else:
    # Empty output — preserve expected schema for downstream consumers
    b_windows = pl.DataFrame(
        {col: pl.Series([], dtype=pl.Utf8) for col in EXPECTED_SUMMARY_COLS}
    ).cast({
        "electrode_idx": pl.Int64,
        "window_id": pl.Int64,
        "smin": pl.Int64,
        "smax": pl.Int64,
        "n_component_windows": pl.Int64,
        "component_smins": pl.List(pl.Int64),
        "sign": pl.Int64,
        "beta_ambig_mean": pl.Float64,
        "beta_ambig_median": pl.Float64,
        "beta_ambig_ci_low": pl.Float64,
        "beta_ambig_ci_high": pl.Float64,
        "ci_excludes_zero": pl.Boolean,
        "is_fallback": pl.Boolean,
        "phon_smin": pl.Int64,
        "phon_smax": pl.Int64,
        "acoustic_peak_auc": pl.Float64,
        "n_per_class": pl.Int64,
        "R": pl.Int64,
        "narrower_than_decoder": pl.Boolean,
        "post_word_offset": pl.Boolean,
        "behav_decoder_smin": pl.Int64,
        "behav_decoder_smax": pl.Int64,
    })
    b_windows_boot = pl.DataFrame(
        {col: pl.Series([], dtype=pl.Utf8) for col in EXPECTED_BOOT_COLS}
    ).cast({
        "electrode_idx": pl.Int64,
        "window_id": pl.Int64,
        "replicate": pl.Int64,
        "beta": pl.Float64,
    })

missing_cols = set(EXPECTED_SUMMARY_COLS) - set(b_windows.columns)
assert not missing_cols, f"b_windows missing expected columns: {missing_cols}"

b_windows.write_parquet(OUT_DIR / "b_windows.parquet")
b_windows_boot.write_parquet(OUT_DIR / "b_windows_bootstrap.parquet")

print(f"b_windows:           {b_windows.height} rows")
print(f"b_windows_bootstrap: {b_windows_boot.height} rows")
if b_windows.height > 0:
    print(f"  ci_excludes_zero:  {b_windows['ci_excludes_zero'].sum()}")
    print(f"  is_fallback:       {b_windows['is_fallback'].sum()}")
    print(f"  narrower_than_dec: {b_windows['narrower_than_decoder'].sum()}")
    print(f"  post_word_offset:  {b_windows['post_word_offset'].sum()}")
    print(b_windows.select(CELL_KEYS + ["window_id", "smin", "smax", "ci_excludes_zero", "is_fallback"]))

# %% [markdown]
# ## Optional QC figures

# %%
try:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Behavioral discriminative windows — QC", fontsize=11)

    # Panel 1: windows per cell
    ax = axes[0, 0]
    if b_windows.height > 0:
        n_wins_per_cell = (
            b_windows.group_by(CELL_KEYS).len()["len"].to_numpy()
        )
        mx = int(n_wins_per_cell.max())
        ax.hist(n_wins_per_cell, bins=range(1, mx + 2), align="left", color="steelblue")
    ax.set_xlabel("Windows per cell")
    ax.set_ylabel("Count")
    ax.set_title("Windows per B4 cell")

    # Panel 2: β_ambig_median distribution
    ax = axes[0, 1]
    if b_windows.height > 0:
        betas = b_windows["beta_ambig_median"].to_numpy()
        excl = b_windows["ci_excludes_zero"].to_numpy()
        ax.hist(betas[~excl], bins=30, alpha=0.5, label="CI ∩ 0", color="gray")
        ax.hist(betas[excl], bins=30, alpha=0.7, label="CI excl. 0", color="steelblue")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=7)
    ax.set_xlabel("β_ambig_median (/n/−/d/)")
    ax.set_title("β distribution")

    # Panel 3: timing scatter
    ax = axes[1, 0]
    if b_windows.height > 0:
        t_smin = b_windows["smin"].to_numpy() / epoch_sfreq + epoch_tmin
        t_smax = b_windows["smax"].to_numpy() / epoch_sfreq + epoch_tmin
        excl = b_windows["ci_excludes_zero"].to_numpy().astype(float)
        fallback = b_windows["is_fallback"].to_numpy()
        sc = ax.scatter(t_smin[~fallback], t_smax[~fallback],
                        c=excl[~fallback], cmap="coolwarm", vmin=0, vmax=1,
                        s=20, alpha=0.8, label="main")
        ax.scatter(t_smin[fallback], t_smax[fallback],
                   c=excl[fallback], cmap="coolwarm", vmin=0, vmax=1,
                   s=20, alpha=0.8, marker="^", label="fallback")
        t_range = [min(t_smin.min(), t_smax.min()), max(t_smin.max(), t_smax.max())]
        ax.plot(t_range, t_range, "k--", lw=0.5)
        ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax, label="ci_excludes_zero")
    ax.set_xlabel("Window start (s)")
    ax.set_ylabel("Window end (s)")
    ax.set_title("Timing scatter")

    # Panel 4: summary counts
    ax = axes[1, 1]
    if b_windows.height > 0:
        n_fb = int(b_windows["is_fallback"].sum())
        n_sig = int(b_windows["ci_excludes_zero"].sum())
        n_tot = b_windows.height
        ax.bar(["Total", "CI excl. 0", "Fallback"], [n_tot, n_sig, n_fb],
               color=["gray", "steelblue", "orange"])
        ax.set_ylabel("Windows")
    ax.set_title("Counts")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "b_windows_summary.pdf")
    plt.close(fig)
    print("Saved b_windows_summary.pdf")
except Exception as _qc_exc:
    warnings.warn(f"QC PDF skipped: {_qc_exc}")

# %% [markdown]
# ## Manual override hook (guarded stub)
#
# Schema: `subject, electrode_idx, phoneme_pair, word_end, window_id, action, smin, smax`
# `action ∈ {add, drop, edit}`
# Absent file → automated output only.

# %%
if manual_override_path and Path(manual_override_path).exists():
    overrides = pl.read_csv(manual_override_path)
    required_override_cols = {
        "subject", "electrode_idx", "phoneme_pair", "word_end",
        "window_id", "action", "smin", "smax",
    }
    missing_oc = required_override_cols - set(overrides.columns)
    assert not missing_oc, f"Override CSV missing columns: {missing_oc}"
    # TODO: implement add/drop/edit merge (plan §"Manual override hook")
    raise NotImplementedError(
        f"Manual override merge not yet implemented. "
        f"Found {overrides.height} row(s) at {manual_override_path}. "
        "Remove the override file or implement the merge body."
    )
