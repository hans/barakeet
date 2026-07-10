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
# # Acoustic endpoint windows
#
# Companion to `acoustic_discriminative_windows.py` for the **endpoint**
# (unambiguous, step6 − step1) acoustic contrast.  Pure post-processing over
# `a_bootstrap_all.parquet` (from `acoustic_bootstrap.py`) — no epoch reload.
#
# Applies the same `_find_maximal_runs` unification used by
# `behavioral_discriminative_windows.py` and `acoustic_discriminative_windows.py`
# so that acoustic endpoint timing is expressed on the same basis as perceptual
# timing (`b_windows.parquet`) and acoustic-on-ambiguous timing (`ad_windows.parquet`).
#
# **Differences vs `acoustic_discriminative_windows.py`**:
# - Source: `a_bootstrap_all.parquet` (endpoint contrast, unambiguous trials,
#   pooled across both word_ends per pair) rather than `b4_acoustic_bootstrap.parquet`.
# - Cell keys: `(subject, electrode_idx, phoneme_pair)` — no `word_end` dimension.
# - Candidate region: `[onset, phon_smax]` — same upper bound as
#   `acoustic_bootstrap.py`'s search, so we stay within the endpoint-search region.
# - No `s_lo`/`s_hi`/`acoustic_peak_auc` (not applicable to endpoint contrast).
# - Beta columns named `beta_endpoint_*`.
# - `post_word_offset`: flagged when unified window end exceeds the minimum
#   word offset for the phoneme pair (conservative; shorter word used as bound).
#
# **Algorithm** (mirror of behavioral/acoustic discriminative):
# 1. Validate grid is contiguous with stride == window_size.
# 2. Candidate windows: `smin >= SAMPLE_T0` (word onset) and `smax <= phon_smax`.
# 3. Significant windows: bootstrap CI of `mean_diff_raw` excludes zero.
# 4. Union runs: maximal groups of adjacent + significant + same-sign windows.
# 5. Union β = per-replicate mean of component `mean_diff_raw` values.
# 6. (Optional) fallback when no significant window: seed at max |median|.
#
# Outputs:
# - `a_windows.parquet`           — per (site × unified_window) summary
# - `a_windows_bootstrap.parquet` — per (site × unified_window × replicate) β

# %% tags=["parameters"]
ac_bootstrap_path = "outputs/causal46_joined/acoustic_bootstrap/a_bootstrap_all.parquet"
ac_per_site_path  = "outputs/causal46_joined/acoustic_bootstrap/a_per_site_all.parquet"
outdir = "outputs/causal46_joined/acoustic_endpoint_windows"

ci_low  = 2.5
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

from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import summarize_replicate_array  # noqa: E402
from _windows import _fallback_run, _find_maximal_runs, _window_sign  # noqa: E402

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]

# t=0 in sample space
SAMPLE_T0 = int(round((0.0 - epoch_tmin) * epoch_sfreq))
print(f"SAMPLE_T0 = {SAMPLE_T0} (t=0 in samples, epoch_tmin={epoch_tmin}, sfreq={epoch_sfreq})")
print(f"use_fallback = {use_fallback}")

# Minimum word offset per phoneme pair — used for post_word_offset flag.
PAIR_MIN_OFFSET_SAMPLE: dict[str, int] = {
    pp: int(round((min(OFFSET_DICT[we] for we in word_ends) - epoch_tmin) * epoch_sfreq))
    for pp, word_ends in PHONEME_PAIR_TO_WORD_ENDS.items()
}
print(f"pair min offset (samples): {PAIR_MIN_OFFSET_SAMPLE}")

# %% [markdown]
# ## Load and validate inputs

# %%
ac_bootstrap = pl.read_parquet(ac_bootstrap_path)
ac_per_site  = pl.read_parquet(ac_per_site_path)

print(f"ac_bootstrap: {ac_bootstrap.height:,} rows, cols: {ac_bootstrap.columns}")
print(f"ac_per_site:  {ac_per_site.height} rows, cols: {ac_per_site.columns}")

for col in ("phon_smin", "phon_smax", "n_per_class"):
    assert col in ac_per_site.columns, (
        f"ac_per_site missing '{col}'. Re-run acoustic_bootstrap with complete data."
    )

# %% [markdown]
# ## Global grid validation

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
        f"Grid gap between {all_grid_windows[i]} and {all_grid_windows[i + 1]}. "
        "Grid must be contiguous (smax_i == smin_{i+1})."
    )

print(
    f"Grid OK: {len(all_grid_windows)} windows, width={GRID_WINDOW_SIZE}, "
    f"range=[{all_grid_windows[0][0]}, {all_grid_windows[-1][1]})"
)

# %%
# Pre-index ac_bootstrap by site key for fast per-site slicing.
_boot_partitioned: dict[tuple, pl.DataFrame] = {}
for row in ac_per_site.iter_rows(named=True):
    key = (row["subject"], row["electrode_idx"], row["phoneme_pair"])
    if key not in _boot_partitioned:
        _boot_partitioned[key] = ac_bootstrap.filter(
            (pl.col("subject") == key[0]) &
            (pl.col("electrode_idx") == key[1]) &
            (pl.col("phoneme_pair") == key[2])
        )

# %% [markdown]
# ## Per-site processing

# %%
EXPECTED_SUMMARY_COLS = [
    "subject", "electrode_idx", "phoneme_pair", "window_id",
    "smin", "smax", "n_component_windows", "component_smins", "sign",
    "beta_endpoint_mean", "beta_endpoint_median",
    "beta_endpoint_ci_low", "beta_endpoint_ci_high",
    "ci_excludes_zero", "is_fallback",
    "phon_smin", "phon_smax", "n_per_class", "R",
    "post_word_offset",
]
EXPECTED_BOOT_COLS = [
    "subject", "electrode_idx", "phoneme_pair",
    "window_id", "replicate", "beta",
]

summary_rows: list[dict] = []
bootstrap_rows: list[dict] = []
n_no_candidates = 0
n_no_sig = 0
n_fallback = 0

for site_row in ac_per_site.iter_rows(named=True):
    subj      = site_row["subject"]
    eidx      = int(site_row["electrode_idx"])
    pp        = site_row["phoneme_pair"]
    phon_smin = int(site_row["phon_smin"])
    phon_smax = int(site_row["phon_smax"])
    n_per_class = int(site_row["n_per_class"])

    site_boot = _boot_partitioned.get((subj, eidx, pp))
    if site_boot is None or site_boot.height == 0:
        warnings.warn(f"No bootstrap data for {subj} e{eidx} {pp}")
        continue

    R = int(site_boot["replicate"].max()) + 1  # replicates are 0-indexed

    pair_min_offset = PAIR_MIN_OFFSET_SAMPLE.get(pp)

    # Candidate windows: from word onset up to phon_smax (the acoustic decoder
    # peak boundary used as the endpoint bootstrap's upper search limit).
    cand_windows = [
        (smin, smax) for smin, smax in all_grid_windows
        if smin >= SAMPLE_T0 and smax <= phon_smax
    ]
    if not cand_windows:
        n_no_candidates += 1
        continue

    cand_smins_set = {smin for smin, _ in cand_windows}
    site_cand_boot = site_boot.filter(pl.col("smin").is_in(list(cand_smins_set)))

    # Per-window significance via bootstrap CI on mean_diff_raw.
    w_medians: dict[int, float] = {}
    w_ci_excl_zero: dict[int, bool] = {}
    for smin, smax in cand_windows:
        arr = site_cand_boot.filter(pl.col("smin") == smin)["mean_diff_raw"].to_numpy()
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

    if sig_windows:
        union_list = _find_maximal_runs(sig_windows, w_medians)
        is_fallback = False
    elif use_fallback:
        union_list = [_fallback_run(cand_windows, w_medians)]
        is_fallback = True
        n_fallback += 1
    else:
        n_no_sig += 1
        continue

    for window_id, comp_windows in enumerate(union_list):
        component_smins = [smin for smin, _ in comp_windows]
        union_smin = comp_windows[0][0]
        union_smax = comp_windows[-1][1]
        n_comp = len(comp_windows)

        union_boot = site_cand_boot.filter(pl.col("smin").is_in(component_smins))
        assert union_boot.height == R * n_comp, (
            f"Expected {R * n_comp} rows (R={R} × {n_comp} component windows) "
            f"for {subj} e{eidx} {pp} union {component_smins}, "
            f"got {union_boot.height}."
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

        post_wo = bool(pair_min_offset is not None and union_smax > pair_min_offset)

        summary_rows.append({
            "subject": subj,
            "electrode_idx": eidx,
            "phoneme_pair": pp,
            "window_id": window_id,
            "smin": union_smin,
            "smax": union_smax,
            "n_component_windows": n_comp,
            "component_smins": component_smins,
            "sign": _window_sign(stats["median"]),
            "beta_endpoint_mean": stats["mean"],
            "beta_endpoint_median": stats["median"],
            "beta_endpoint_ci_low": stats["ci_lo"],
            "beta_endpoint_ci_high": stats["ci_hi"],
            "ci_excludes_zero": stats["ci_excludes_zero"],
            "is_fallback": is_fallback,
            "phon_smin": phon_smin,
            "phon_smax": phon_smax,
            "n_per_class": n_per_class,
            "R": R,
            "post_word_offset": post_wo,
        })

        rep_arr = union_beta_df["replicate"].to_numpy()
        for rep, beta in zip(rep_arr, beta_arr):
            bootstrap_rows.append({
                "subject": subj,
                "electrode_idx": eidx,
                "phoneme_pair": pp,
                "window_id": window_id,
                "replicate": int(rep),
                "beta": float(beta),
            })

print(
    f"Sites: {ac_per_site.height} total, "
    f"{n_no_candidates} with no candidates, "
    f"{n_no_sig} with no significant window (fallback off), "
    f"{n_fallback} fallback sites"
)

# %% [markdown]
# ## Write outputs

# %%
if summary_rows:
    a_windows      = pl.DataFrame(summary_rows)
    a_windows_boot = pl.DataFrame(bootstrap_rows)
else:
    a_windows = pl.DataFrame(
        {col: pl.Series([], dtype=pl.Utf8) for col in EXPECTED_SUMMARY_COLS}
    ).cast({
        "electrode_idx": pl.Int64,
        "window_id": pl.Int64,
        "smin": pl.Int64,
        "smax": pl.Int64,
        "n_component_windows": pl.Int64,
        "component_smins": pl.List(pl.Int64),
        "sign": pl.Int64,
        "beta_endpoint_mean": pl.Float64,
        "beta_endpoint_median": pl.Float64,
        "beta_endpoint_ci_low": pl.Float64,
        "beta_endpoint_ci_high": pl.Float64,
        "ci_excludes_zero": pl.Boolean,
        "is_fallback": pl.Boolean,
        "phon_smin": pl.Int64,
        "phon_smax": pl.Int64,
        "n_per_class": pl.Int64,
        "R": pl.Int64,
        "post_word_offset": pl.Boolean,
    })
    a_windows_boot = pl.DataFrame(
        {col: pl.Series([], dtype=pl.Utf8) for col in EXPECTED_BOOT_COLS}
    ).cast({
        "electrode_idx": pl.Int64,
        "window_id": pl.Int64,
        "replicate": pl.Int64,
        "beta": pl.Float64,
    })

missing_cols = set(EXPECTED_SUMMARY_COLS) - set(a_windows.columns)
assert not missing_cols, f"a_windows missing expected columns: {missing_cols}"

a_windows.write_parquet(OUT_DIR / "a_windows.parquet")
a_windows_boot.write_parquet(OUT_DIR / "a_windows_bootstrap.parquet")

print(f"a_windows:           {a_windows.height} rows")
print(f"a_windows_bootstrap: {a_windows_boot.height} rows")
if a_windows.height > 0:
    print(f"  ci_excludes_zero:  {a_windows['ci_excludes_zero'].sum()}")
    print(f"  is_fallback:       {a_windows['is_fallback'].sum()}")
    print(f"  post_word_offset:  {a_windows['post_word_offset'].sum()}")
    print(a_windows.select(SITE_KEYS + ["window_id", "smin", "smax", "ci_excludes_zero"]))

# %% [markdown]
# ## QC figures

# %%
try:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Acoustic endpoint windows — QC", fontsize=11)

    ax = axes[0, 0]
    if a_windows.height > 0:
        n_wins_per_site = (
            a_windows.group_by(SITE_KEYS).len()["len"].to_numpy()
        )
        mx = int(n_wins_per_site.max())
        ax.hist(n_wins_per_site, bins=range(1, mx + 2), align="left", color="steelblue")
    ax.set_xlabel("Windows per site")
    ax.set_ylabel("Count")
    ax.set_title("Windows per acoustic site")

    ax = axes[0, 1]
    if a_windows.height > 0:
        betas = a_windows["beta_endpoint_median"].to_numpy()
        excl  = a_windows["ci_excludes_zero"].to_numpy()
        ax.hist(betas[~excl], bins=30, alpha=0.5, label="CI ∩ 0", color="gray")
        ax.hist(betas[excl],  bins=30, alpha=0.7, label="CI excl. 0", color="steelblue")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=7)
    ax.set_xlabel("β_endpoint_median (step6 − step1)")
    ax.set_title("β distribution")

    ax = axes[1, 0]
    if a_windows.height > 0:
        t_smin  = a_windows["smin"].to_numpy() / epoch_sfreq + epoch_tmin
        t_smax  = a_windows["smax"].to_numpy() / epoch_sfreq + epoch_tmin
        t_psmin = a_windows["phon_smin"].to_numpy() / epoch_sfreq + epoch_tmin
        t_psmax = a_windows["phon_smax"].to_numpy() / epoch_sfreq + epoch_tmin
        excl    = a_windows["ci_excludes_zero"].to_numpy().astype(float)
        sc = ax.scatter(t_smin, t_smax, c=excl, cmap="coolwarm", vmin=0, vmax=1,
                        s=25, alpha=0.8)
        for ps, pe in zip(t_psmin, t_psmax):
            ax.axvline(ps, color="green", lw=0.3, alpha=0.3)
            ax.axvline(pe, color="orange", lw=0.3, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="ci_excludes_zero")
    ax.set_xlabel("Window start (s)")
    ax.set_ylabel("Window end (s)")
    ax.set_title("Timing (green=phon_smin, orange=phon_smax)")

    ax = axes[1, 1]
    if a_windows.height > 0:
        n_fb  = int(a_windows["is_fallback"].sum())
        n_sig = int(a_windows["ci_excludes_zero"].sum())
        n_tot = a_windows.height
        ax.bar(["Total", "CI excl. 0", "Fallback"], [n_tot, n_sig, n_fb],
               color=["gray", "steelblue", "darkorange"])
        ax.set_ylabel("Windows")
    ax.set_title("Counts")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "a_windows_summary.pdf")
    plt.close(fig)
    print("Saved a_windows_summary.pdf")
except Exception as _qc_exc:
    warnings.warn(f"QC PDF skipped: {_qc_exc}")
