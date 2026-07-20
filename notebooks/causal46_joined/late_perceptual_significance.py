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
# # Late within-completion perceptual significance (TFCE gate)
#
# For each of the powered B4 cells `(subject, electrode_idx, phoneme_pair,
# word_end)`, replaces the manual `behav @late` entry gate with a per-cell
# **TFCE permutation test** on the post-acoustic within-completion `/n/-/d/`
# contrast — pure post-processing over the already-persisted
# `b4_bootstrap.parquet` replicates (no epoch reload).
#
# Per cell:
# - Candidate windows: post-acoustic (`smin >= phon_smax`) and within the
#   per-word-end bound `smax <= offset + 0.1s` (same window logic as
#   `behavioral_discriminative_windows.py`).
# - Observed curve: median over bootstrap replicates of `mean_diff_raw`
#   (signed, two-tailed `/n/-/d/`, D1) per window.
# - Null: the R within-step label-permutation null curves recovered from
#   `mean_diff_aligned_null` (D2) — one coherent across-window curve per
#   replicate.
# - Gate statistic: two-tailed max-TFCE (`_windows.late_cell_significance`,
#   built on the #9 helpers), `E=0.5, H=2.0` pre-registered (D3). Also
#   reports a knob-free integrated-window robustness stat/p and an optional
#   descriptive split-half sign-agreement column (D7).
# - Tied cells (`preferred is None`, no usable null — D2) are dropped from
#   the gate but kept in the parquet with `is_tied=True`.
#
# Two operating points on the same per-cell p-values (D5):
# - **Gate** (feeds the downstream window-finding cascade, `#11`):
#   uncorrected `tfce_emp_p < 0.05`.
# - **Headline** (population claim): count of gate-passers vs
#   `Binomial(n_family, 0.05)`, plus BH-FDR across the family as the
#   conservative floor. Family = all powered cells with a computed
#   p-value (excludes tied / no-candidate-window / missing-data cells) —
#   NOT conditioned on the manual `behav @late` labels being replaced.
#
# `manual_behav_late` is joined read-only from `filtered_manifest.csv` for
# later calibration use (D6) — it never gates anything here.
#
# See: docs/superpowers/plans/2026-07-20-causal46-late-perceptual-significance.md
#
# Outputs:
# - `late_perceptual_significance/site_results.parquet` — one row per powered B4 cell
# - `late_perceptual_significance/population_summary.pdf` — count-vs-null headline + BH-FDR survivor count

# %% tags=["parameters"]
b4_bootstrap_path = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
filtered_manifest_path = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv"
outdir = "outputs/causal46_joined/late_perceptual_significance"

gate_alpha = 0.05
fdr_alpha = 0.05
binom_null_p = 0.05
tfce_E = 0.5
tfce_H = 2.0

# %%
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.stats.multitest import multipletests

from src.stimuli import OFFSET_DICT
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _windows import (  # noqa: E402
    assert_coherent_null_replicates,
    late_cell_significance,
    validate_contiguous_grid,
)

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
GATE_ALPHA = float(gate_alpha)
FDR_ALPHA = float(fdr_alpha)
BINOM_NULL_P = float(binom_null_p)
TFCE_E = float(tfce_E)
TFCE_H = float(tfce_H)

# %% [markdown]
# ## Load and validate inputs

# %%
b4_bootstrap = pl.read_parquet(b4_bootstrap_path)
b4_per_cell = pl.read_parquet(b4_per_cell_path)

print(f"b4_bootstrap: {b4_bootstrap.height:,} rows, cols: {b4_bootstrap.columns}")
print(f"b4_per_cell:  {b4_per_cell.height} rows (powered B4 cells)")

for col in ("phon_smin", "phon_smax"):
    assert col in b4_per_cell.columns, (
        f"b4_per_cell missing '{col}'. This column is only written when "
        "t_tests.py has non-empty paired data (t_tests.py:722-732). "
        "Re-run t_tests with complete data."
    )
for col in ("mean_diff_raw", "mean_diff_aligned_null", "replicate", "smin", "smax"):
    assert col in b4_bootstrap.columns, f"b4_bootstrap missing '{col}'."

# %% [markdown]
# ## Global grid validation
#
# Asserts stride == window_size and contiguity (mirrors
# `behavioral_discriminative_windows.py`).

# %%
all_grid_windows: list[tuple[int, int]] = sorted(
    {(int(r[0]), int(r[1])) for r in b4_bootstrap.select(["smin", "smax"]).iter_rows()},
    key=lambda t: t[0],
)
grid_window_size = validate_contiguous_grid(all_grid_windows)

print(
    f"Grid OK: {len(all_grid_windows)} windows, width={grid_window_size}, "
    f"range=[{all_grid_windows[0][0]}, {all_grid_windows[-1][1]})"
)

# %% [markdown]
# ## Per-WE post-acoustic search bound (D4)
#
# Same bound as `behavioral_discriminative_windows.py`: `offset + 0.1s`.

# %%
WE_SEARCH_SMAX: dict[str, int] = {
    we: int(round((offset_s + 0.1 - epoch_tmin) * epoch_sfreq))
    for we, offset_s in OFFSET_DICT.items()
}
print(f"word-end search_smax (samples): {WE_SEARCH_SMAX}")

# %%
# Pre-index b4_bootstrap by cell for fast per-cell slicing.
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
# ## Per-cell TFCE gate

# %%
rows: list[dict] = []
n_no_candidates = 0
n_tied = 0
n_missing = 0

for cell_row in b4_per_cell.iter_rows(named=True):
    subj = cell_row["subject"]
    eidx = int(cell_row["electrode_idx"])
    pp = cell_row["phoneme_pair"]
    we = cell_row["word_end"]
    phon_smin = int(cell_row["phon_smin"])
    phon_smax = int(cell_row["phon_smax"])

    base_row = {
        "subject": subj, "electrode_idx": eidx, "phoneme_pair": pp, "word_end": we,
        "phon_smin": phon_smin, "phon_smax": phon_smax,
        "search_smin": phon_smax, "search_smax": WE_SEARCH_SMAX.get(we),
        "n_windows": 0,
        "tfce_peak": None, "tfce_max_abs": None, "tfce_emp_p": None,
        "tfce_gate_pass": False,
        "integral_stat": None, "integral_emp_p": None,
        "splithalf_sign_agree": None,
        "is_tied": False,
    }

    cell_boot = _boot_partitioned.get((subj, eidx, pp, we))
    if cell_boot is None or cell_boot.height == 0:
        warnings.warn(f"No bootstrap data for {subj} e{eidx} {pp} {we}")
        n_missing += 1
        rows.append(base_row)
        continue

    we_search_smax = WE_SEARCH_SMAX.get(we)
    cand_windows = [
        (smin, smax) for smin, smax in all_grid_windows
        if smin >= phon_smax and (we_search_smax is None or smax <= we_search_smax)
    ]
    if not cand_windows:
        n_no_candidates += 1
        rows.append(base_row)
        continue

    cand_smins = [smin for smin, _ in cand_windows]
    cand_boot = cell_boot.filter(pl.col("smin").is_in(cand_smins))
    # Keep only windows that actually have data for this cell.
    present_smins = set(cand_boot["smin"].unique().to_list())
    cand_windows = [(smin, smax) for smin, smax in cand_windows if smin in present_smins]
    if not cand_windows:
        n_no_candidates += 1
        rows.append(base_row)
        continue

    n_windows = len(cand_windows)
    base_row["n_windows"] = n_windows
    R = int(cand_boot["replicate"].max()) + 1

    # mean_diff_aligned_null is written as float("nan") (not a polars null) for
    # tied cells (t_tests.py bootstrap_cell, D2) — to_numpy() maps both to NaN.
    is_tied = bool(np.all(np.isnan(cand_boot["mean_diff_aligned_null"].to_numpy())))
    base_row["is_tied"] = is_tied
    if is_tied:
        n_tied += 1
        rows.append(base_row)
        continue

    context = f"{subj} e{eidx} {pp} {we}"
    assert_coherent_null_replicates(cand_boot.height, R, n_windows, context=context)

    sorted_boot = cand_boot.sort(["replicate", "smin"])
    rep_curves = sorted_boot["mean_diff_raw"].to_numpy().reshape(R, n_windows)
    null_curves = sorted_boot["mean_diff_aligned_null"].to_numpy().reshape(R, n_windows)

    sig = late_cell_significance(rep_curves, null_curves, E=TFCE_E, H=TFCE_H)

    base_row.update({
        "tfce_peak": sig["tfce_peak"],
        "tfce_max_abs": sig["tfce_max_abs"],
        "tfce_emp_p": sig["tfce_emp_p"],
        "tfce_gate_pass": bool(sig["tfce_emp_p"] < GATE_ALPHA),
        "integral_stat": sig["integral_stat"],
        "integral_emp_p": sig["integral_emp_p"],
        "splithalf_sign_agree": sig["splithalf_sign_agree"],
    })
    rows.append(base_row)

print(
    f"Cells: {b4_per_cell.height} total, {n_missing} missing bootstrap data, "
    f"{n_no_candidates} with no candidate windows, {n_tied} tied (D2 drop-and-document)"
)

# %% [markdown]
# ## BH-FDR + count-vs-null population headline (D5)
#
# Family = powered cells with a computed p-value (excludes missing-data,
# no-candidate-window, and tied cells) — not conditioned on the manual
# `behav @late` labels this test replaces.

# %%
site_results = pl.DataFrame(
    rows,
    schema_overrides={
        "search_smax": pl.Int64,
        "tfce_peak": pl.Float64, "tfce_max_abs": pl.Float64, "tfce_emp_p": pl.Float64,
        "integral_stat": pl.Float64, "integral_emp_p": pl.Float64,
        "splithalf_sign_agree": pl.Boolean,
    },
)

has_p = site_results["tfce_emp_p"].is_not_null().to_numpy()
n_family = int(has_p.sum())

tfce_p_fdr = np.full(site_results.height, np.nan)
tfce_fdr_pass = np.zeros(site_results.height, dtype=bool)
if n_family > 0:
    pvals = site_results["tfce_emp_p"].to_numpy()[has_p]
    reject, qvals, _, _ = multipletests(pvals, method="fdr_bh", alpha=FDR_ALPHA)
    tfce_p_fdr[has_p] = qvals
    tfce_fdr_pass[has_p] = reject

site_results = site_results.with_columns([
    pl.Series("tfce_p_fdr", tfce_p_fdr),
    pl.Series("tfce_fdr_pass", tfce_fdr_pass),
])

n_gate_pass = int(site_results["tfce_gate_pass"].sum())
n_fdr_pass = int(site_results["tfce_fdr_pass"].sum())

if n_family > 0:
    binom_p = float(
        scipy.stats.binomtest(n_gate_pass, n_family, BINOM_NULL_P, alternative="greater").pvalue
    )
    binom_mean = n_family * BINOM_NULL_P
    binom_sd = float(np.sqrt(n_family * BINOM_NULL_P * (1 - BINOM_NULL_P)))
else:
    binom_p = float("nan")
    binom_mean = 0.0
    binom_sd = 0.0

print(f"Family (n_family, has p-value): {n_family}")
print(f"Gate pass (uncorrected p<{GATE_ALPHA}): {n_gate_pass} / {n_family}")
print(f"  Binomial({n_family}, {BINOM_NULL_P}) ref: mean={binom_mean:.1f}  SD={binom_sd:.1f}")
print(f"  Binomial exact p (greater): {binom_p:.4g}")
print(f"BH-FDR survivors (q<{FDR_ALPHA}): {n_fdr_pass} / {n_family}")

# %% [markdown]
# ## Join manual `behav @late` label (D6 — calibration only, read-only)

# %%
manifest = pl.read_csv(filtered_manifest_path)
manual_late = (
    manifest
    .select(CELL_KEYS + [pl.col("behav @late").alias("manual_behav_late")])
    .with_columns(pl.col("electrode_idx").cast(pl.Int64))
)
assert manual_late.height == manual_late.unique(subset=CELL_KEYS).height, (
    "filtered_manifest.csv has duplicate (subject, electrode_idx, phoneme_pair, "
    "word_end) keys — the manual_behav_late join below assumes one row per cell."
)

n_before_join = site_results.height
site_results = site_results.join(manual_late, on=CELL_KEYS, how="left")
assert site_results.height == n_before_join, (
    f"manual_behav_late join changed row count ({n_before_join} -> "
    f"{site_results.height}); expected a 1:1 left join (one row per powered B4 cell)."
)

print(f"site_results: {site_results.height} rows, cols: {site_results.columns}")

# %% [markdown]
# ## Write outputs

# %%
site_results.write_parquet(OUT_DIR / "site_results.parquet")
print(f"Wrote {OUT_DIR / 'site_results.parquet'}")

# %%
with PdfPages(OUT_DIR / "population_summary.pdf") as pdf:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.axis("off")
    lines = [
        "Late within-completion perceptual significance — TFCE gate",
        "",
        f"Powered B4 cells:            {b4_per_cell.height}",
        f"  missing bootstrap data:    {n_missing}",
        f"  no candidate windows:      {n_no_candidates}",
        f"  tied (dropped, D2):        {n_tied}",
        f"Family (has p-value):        {n_family}",
        "",
        f"Gate pass (p<{GATE_ALPHA}, uncorrected):  {n_gate_pass} / {n_family}",
        f"  Binomial({n_family}, {BINOM_NULL_P}) mean±SD: {binom_mean:.1f} ± {binom_sd:.1f}",
        f"  Binomial exact p (greater):             {binom_p:.4g}",
        "",
        f"BH-FDR survivors (q<{FDR_ALPHA}, conservative floor): {n_fdr_pass} / {n_family}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, va="top", ha="left", family="monospace")

    ax = axes[1]
    if n_family > 0:
        labels = ["observed\ngate pass", "expected\nunder null"]
        values = [n_gate_pass, binom_mean]
        errs = [0.0, binom_sd]
        ax.bar(labels, values, yerr=errs, color=["#2166ac", "#b2182b"],
               alpha=0.85, capsize=6)
        ax.set_ylabel("cell count")
        ax.set_title(f"count-vs-null (n_family={n_family})")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no family (n_family=0)", ha="center", va="center")
    fig.suptitle("Late perceptual significance — population headline", fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

print(f"Wrote {OUT_DIR / 'population_summary.pdf'}")
