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
# # Late within-completion perceptual significance — report (plan Step 6)
#
# Diagnostics for the TFCE late-gate (`#10`, `late_perceptual_significance.py`):
# is the manual-independent population headline trustworthy, or an artifact of
# the pre-registered TFCE `E`/`H` choice? Three checks, none of which touch the
# gate itself (read-only over `site_results.parquet` + a re-derivation from the
# raw bootstrap curves for the sensitivity panel):
#
# 1. **Population headline recap** — count-vs-null and BH-FDR, reprinted from
#    `site_results.parquet` (no recomputation of the gate itself).
# 2. **Integral-vs-TFCE agreement** — the knob-free integrated-window statistic
#    (D3's "documented, reproducible, zero-params" robustness column) compared
#    against the TFCE gate. Convergent counts/ranking here means the TFCE
#    result isn't a TFCE-specific quirk.
# 3. **`E`/`H` param-sensitivity panel** (D3, pre-registered, previously never
#    run) — rerun `late_cell_significance` at every combination of `E ∈
#    {0.5, 1.0}`, `H ∈ {1, 2}` over the real per-cell curves and compare
#    gate-pass counts.
# 4. **Manual-label calibration** (D6) — 2×2 concordance table against
#    `behav @late`, a Mann-Whitney/KS comparison of the TFCE p-value
#    distributions for manual-positive vs manual-negative cells, and the named
#    disagreement list (manual ∩ not-TFCE-pass; TFCE-pass ∩ not-manual).
#
# See: docs/superpowers/plans/2026-07-20-causal46-late-perceptual-significance.md
#
# Outputs:
# - `late_perceptual_significance_report/sensitivity_grid.csv` — one row per
#   (E, H) combination: n_family, n_gate_pass, binom_p, n_fdr_pass.
# - `late_perceptual_significance_report/calibration_disagreements.csv` — named
#   cells in the two disagreement quadrants (D6), for eyeball follow-up.
# - `late_perceptual_significance_report/report_summary.pdf` — headline text
#   panel, integral-vs-TFCE scatter, sensitivity bar chart, calibration
#   histogram.

# %% tags=["parameters"]
site_results_path = "outputs/causal46_joined/late_perceptual_significance/site_results.parquet"
b4_bootstrap_path = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
outdir = "outputs/causal46_joined/late_perceptual_significance_report"

gate_alpha = 0.05
binom_null_p = 0.05
sensitivity_E_values = [0.5, 1.0]
sensitivity_H_values = [1.0, 2.0]

# %%
import sys
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
    extract_cell_curves,
    late_cell_significance,
    validate_contiguous_grid,
)

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
GATE_ALPHA = float(gate_alpha)
BINOM_NULL_P = float(binom_null_p)

# %% [markdown]
# ## Load `site_results.parquet` (#10 output — read-only)

# %%
site_results = pl.read_parquet(site_results_path)
print(f"site_results: {site_results.height} rows, cols: {site_results.columns}")

for col in ("tfce_emp_p", "tfce_gate_pass", "integral_emp_p", "manual_behav_late"):
    assert col in site_results.columns, f"site_results missing '{col}'."

# %% [markdown]
# ## 1. Population headline recap (no recomputation of the gate)

# %%
has_p = site_results["tfce_emp_p"].is_not_null().to_numpy()
n_family = int(has_p.sum())
n_gate_pass = int(site_results["tfce_gate_pass"].sum())
n_fdr_pass = int(site_results["tfce_fdr_pass"].sum())
binom_mean = n_family * BINOM_NULL_P
binom_sd = float(np.sqrt(n_family * BINOM_NULL_P * (1 - BINOM_NULL_P)))
binom_p = float(
    scipy.stats.binomtest(n_gate_pass, n_family, BINOM_NULL_P, alternative="greater").pvalue
) if n_family > 0 else float("nan")

print(f"n_family: {n_family}")
print(f"Gate pass (uncorrected p<{GATE_ALPHA}): {n_gate_pass} / {n_family}")
print(f"  Binomial({n_family}, {BINOM_NULL_P}) mean±SD: {binom_mean:.1f} ± {binom_sd:.1f}")
print(f"  Binomial exact p (greater): {binom_p:.4g}")
print(f"BH-FDR survivors: {n_fdr_pass} / {n_family}")

# %% [markdown]
# ## 2. Integral-vs-TFCE agreement
#
# `integral_stat`/`integral_emp_p` (D3) is a knob-free alternative to TFCE —
# mean of the observed curve over all candidate windows, no `E`/`H`. Applying
# the same uncorrected gate to it and comparing against `tfce_gate_pass` tests
# whether the TFCE result is a TFCE-specific artifact.

# %%
family = site_results.filter(pl.col("tfce_emp_p").is_not_null())
family = family.with_columns((pl.col("integral_emp_p") < GATE_ALPHA).alias("integral_gate_pass"))

integral_crosstab = family.group_by(["tfce_gate_pass", "integral_gate_pass"]).len().sort(
    ["tfce_gate_pass", "integral_gate_pass"]
)
n_integral_pass = int(family["integral_gate_pass"].sum())
tfce_p_arr = family["tfce_emp_p"].to_numpy()
integral_p_arr = family["integral_emp_p"].to_numpy()
spearman_rho, spearman_p = scipy.stats.spearmanr(tfce_p_arr, integral_p_arr)

print(integral_crosstab)
print(f"\nintegral gate pass: {n_integral_pass} / {n_family}  (tfce: {n_gate_pass} / {n_family})")
print(f"Spearman rho(tfce_emp_p, integral_emp_p) = {spearman_rho:.3f}, p = {spearman_p:.3g}")

# %% [markdown]
# ## 3. `E`/`H` param-sensitivity panel (D3, pre-registered)
#
# Re-derives per-cell curves once via `extract_cell_curves` (shared with #10's
# notebook) and reruns `late_cell_significance` at each `(E, H)` combination —
# cheap once curves are extracted, since TFCE itself operates on ~10 windows.

# %%
b4_bootstrap = pl.read_parquet(b4_bootstrap_path)
b4_per_cell = pl.read_parquet(b4_per_cell_path)

all_grid_windows: list[tuple[int, int]] = sorted(
    {(int(r[0]), int(r[1])) for r in b4_bootstrap.select(["smin", "smax"]).iter_rows()},
    key=lambda t: t[0],
)
validate_contiguous_grid(all_grid_windows)

WE_SEARCH_SMAX: dict[str, int] = {
    we: int(round((offset_s + 0.1 - epoch_tmin) * epoch_sfreq))
    for we, offset_s in OFFSET_DICT.items()
}

cell_curves = extract_cell_curves(b4_bootstrap, b4_per_cell, all_grid_windows, WE_SEARCH_SMAX)
ok_curves = [c for c in cell_curves.values() if c["status"] == "ok"]
print(f"Cells with usable curves for sensitivity sweep: {len(ok_curves)} / {len(cell_curves)}")

# %%
sensitivity_rows: list[dict] = []
for E in sensitivity_E_values:
    for H in sensitivity_H_values:
        pvals = np.array([
            late_cell_significance(c["rep_curves"], c["null_curves"], E=E, H=H)["tfce_emp_p"]
            for c in ok_curves
        ])
        n_fam = len(pvals)
        n_pass = int((pvals < GATE_ALPHA).sum())
        p_binom = float(
            scipy.stats.binomtest(n_pass, n_fam, BINOM_NULL_P, alternative="greater").pvalue
        ) if n_fam > 0 else float("nan")
        reject, _, _, _ = multipletests(pvals, method="fdr_bh", alpha=0.05) if n_fam > 0 else (
            np.array([]), None, None, None
        )
        n_fdr = int(reject.sum())
        sensitivity_rows.append({
            "E": E, "H": H, "n_family": n_fam, "n_gate_pass": n_pass,
            "binom_p": p_binom, "n_fdr_pass": n_fdr,
            "is_preregistered": bool(E == 0.5 and H == 2.0),
        })

sensitivity_grid = pl.DataFrame(sensitivity_rows)
print(sensitivity_grid)
sensitivity_grid.write_csv(OUT_DIR / "sensitivity_grid.csv")
print(f"Wrote {OUT_DIR / 'sensitivity_grid.csv'}")

# %% [markdown]
# ## 4. Manual-label calibration (D6 — calibration, not validation)

# %%
family_manual = family.with_columns(pl.col("manual_behav_late").is_not_null().alias("manual_flag"))
calibration_crosstab = family_manual.group_by(["tfce_gate_pass", "manual_flag"]).len().sort(
    ["tfce_gate_pass", "manual_flag"]
)
print(calibration_crosstab)

manual_p = family_manual.filter(pl.col("manual_flag"))["tfce_emp_p"].to_numpy()
nonmanual_p = family_manual.filter(~pl.col("manual_flag"))["tfce_emp_p"].to_numpy()
mw_stat, mw_p = scipy.stats.mannwhitneyu(manual_p, nonmanual_p, alternative="less")
ks_stat, ks_p = scipy.stats.ks_2samp(manual_p, nonmanual_p)
print(
    f"\nmanual tfce_emp_p: n={len(manual_p)}, mean={manual_p.mean():.3f}, min={manual_p.min():.3f}"
)
print(
    f"non-manual tfce_emp_p: n={len(nonmanual_p)}, mean={nonmanual_p.mean():.3f}, "
    f"min={nonmanual_p.min():.3f}"
)
print(f"Mann-Whitney (manual < non-manual): U={mw_stat:.1f}, p={mw_p:.3g}")
print(f"KS 2-sample: D={ks_stat:.3f}, p={ks_p:.3g}")

# %%
# Named disagreement cells (D6): eyeball follow-up only, never re-gates anything.
manual_not_tfce = family_manual.filter(pl.col("manual_flag") & ~pl.col("tfce_gate_pass"))
tfce_not_manual = family_manual.filter(pl.col("tfce_gate_pass") & ~pl.col("manual_flag"))

disagreements = pl.concat([
    manual_not_tfce.select(CELL_KEYS + ["tfce_emp_p"]).with_columns(
        pl.lit("manual_only").alias("disagreement_kind")
    ),
    tfce_not_manual.select(CELL_KEYS + ["tfce_emp_p"]).with_columns(
        pl.lit("tfce_only").alias("disagreement_kind")
    ),
])
disagreements.write_csv(OUT_DIR / "calibration_disagreements.csv")
print(
    f"\nDisagreements: {manual_not_tfce.height} manual-only, {tfce_not_manual.height} tfce-only "
    f"-> {OUT_DIR / 'calibration_disagreements.csv'}"
)

# %% [markdown]
# ## Write summary PDF

# %%
with PdfPages(OUT_DIR / "report_summary.pdf") as pdf:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.axis("off")
    prereg = sensitivity_grid.filter(pl.col("is_preregistered")).row(0, named=True)
    lines = [
        "Late perceptual significance — report (Step 6)",
        "",
        f"n_family:                 {n_family}",
        f"Gate pass (p<{GATE_ALPHA}, uncorrected): {n_gate_pass} / {n_family}",
        f"  Binomial({n_family}, {BINOM_NULL_P}) mean±SD: {binom_mean:.1f} ± {binom_sd:.1f}",
        f"  Binomial exact p (greater):    {binom_p:.4g}",
        f"BH-FDR survivors:          {n_fdr_pass} / {n_family}",
        "",
        f"Integral gate pass:        {n_integral_pass} / {n_family}",
        f"Spearman(tfce, integral):  rho={spearman_rho:.3f}, p={spearman_p:.3g}",
        "",
        f"Manual behav @late cells:  {manual_not_tfce.height + tfce_not_manual.height + int((family_manual['manual_flag'] & family_manual['tfce_gate_pass']).sum())}",
        f"  manual-only disagreement: {manual_not_tfce.height}",
        f"  tfce-only disagreement:   {tfce_not_manual.height}",
        f"Mann-Whitney (manual<non-manual) p: {mw_p:.3g}",
        f"KS 2-sample p:             {ks_p:.3g}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top", ha="left", family="monospace")

    ax = axes[0, 1]
    ax.scatter(tfce_p_arr, integral_p_arr, s=12, alpha=0.6, color="#2166ac")
    ax.axhline(GATE_ALPHA, color="gray", lw=0.5, ls="--")
    ax.axvline(GATE_ALPHA, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("tfce_emp_p")
    ax.set_ylabel("integral_emp_p")
    ax.set_title("Integral vs TFCE p-values")

    ax = axes[1, 0]
    x = np.arange(len(sensitivity_grid))
    labels = [f"E={r['E']}\nH={r['H']}" for r in sensitivity_grid.iter_rows(named=True)]
    counts = sensitivity_grid["n_gate_pass"].to_list()
    colors = ["#b2182b" if r["is_preregistered"] else "#2166ac"
              for r in sensitivity_grid.iter_rows(named=True)]
    ax.bar(x, counts, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("n_gate_pass")
    ax.set_title("E/H sensitivity (red = pre-registered)")

    ax = axes[1, 1]
    bins = np.linspace(0, 1, 21)
    ax.hist(nonmanual_p, bins=bins, alpha=0.6, label="non-manual", color="gray")
    ax.hist(manual_p, bins=bins, alpha=0.6, label="manual behav@late", color="#2166ac")
    ax.axvline(GATE_ALPHA, color="black", lw=0.5, ls="--")
    ax.set_xlabel("tfce_emp_p")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("Calibration: manual vs non-manual p-values")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Wrote {OUT_DIR / 'report_summary.pdf'}")
