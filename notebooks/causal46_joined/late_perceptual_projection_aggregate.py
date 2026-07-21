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
# # Late Perceptual Projection — Aggregate + pre-registered go/no-go
#
# Combines per-subject outputs and applies the **LOCKED, confirmatory**
# pass criterion (issue #21, spec §§8–9, ADR-0003):
#
# - **Claim-bearing population** = **â-reliable** cells (`pi_anchored` non-NaN).
#   Claim statistic = `pi_anchored`.
# - **Population statistic** = **CPO count-vs-null** (early aggregate Test 2):
#   observed = # â-reliable cells with uncorrected **one-tailed** p < 0.05;
#   matched permutation null of that count; `p_cpo` = fraction of null counts ≥ observed.
# - **GO iff `p_cpo < 0.05`**, one-tailed (π > 0), **no minimum-count floor**.
#   NO-GO = `p_cpo ≥ 0.05` → integration section retreats to negative claims only.
# - **Binomial(N, 0.05)** and **BH-FDR** reported as references — **neither gates**.
# - **Diagnostics (non-gating):** reliable-vs-all comparison via `pi_peak` (the map's
#   spine), and a cross-tab vs the late manual annotation `behav @late` (Test-3
#   analogue; the late vocabulary is not ground truth for this statistic — mutable).
#
# The go/no-go value falls out mechanically; issue #23 records the paper-claim decision.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.multitest import multipletests

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

# %% tags=["parameters"]
results_dir = "outputs/causal46_joined/late_perceptual_projection"
# Late manual annotation (Test-3 diagnostic; non-gating). `behav @late` column.
late_annotations_path = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv"
outdir = "outputs/causal46_joined/late_perceptual_projection"
fdr_alpha = 0.05
cpo_p_threshold = 0.05

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(results_dir)
FDR_ALPHA = float(fdr_alpha)
CPO_P = float(cpo_p_threshold)
print(f"Aggregate: results_dir={RESULTS_DIR}  fdr_alpha={FDR_ALPHA}  cpo_p={CPO_P}")


def cell_key(row) -> str:
    return f"{row.subject}_{int(row.electrode_idx)}_{row.phoneme_pair}_{row.word_end}"


# %% [markdown]
# ## Load per-subject cell results + anchored nulls

# %%
csv_paths = sorted(RESULTS_DIR.glob("*/site_results.csv"))
print(f"Found {len(csv_paths)} per-subject result CSVs")
dfs = []
for p in csv_paths:
    try:
        df = pd.read_csv(p)
        if len(df.columns) > 0:
            dfs.append(df)
    except pd.errors.EmptyDataError:
        print(f"  Skipping empty: {p.parent.name}")
all_cells = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
print(f"Total cells: {len(all_cells)}")

null_arrays = {}
for npz_path in sorted(RESULTS_DIR.glob("*/null_pi.npz")):
    data = np.load(str(npz_path))
    for k in data.files:
        null_arrays[k] = data[k]
print(f"Anchored null arrays loaded: {len(null_arrays)} cells")

null_arrays_peak = {}
for npz_path in sorted(RESULTS_DIR.glob("*/null_pi_peak.npz")):
    data = np.load(str(npz_path))
    for k in data.files:
        null_arrays_peak[k] = data[k]

# %% [markdown]
# ## Claim-bearing population: â-reliable cells (π_anchored non-NaN)

# %%
if len(all_cells) == 0:
    reliable_df = pd.DataFrame()
    print("No cells — nothing to test.")
else:
    reliable_df = all_cells[all_cells["pi_anchored"].notna()].copy()
    reliable_df["cell_key"] = reliable_df.apply(cell_key, axis=1)
    n_reliable = len(reliable_df)
    n_estimable = int(all_cells["pi_peak"].notna().sum())
    print(f"cells total                          : {len(all_cells)}")
    print(f"â-estimable (π_peak non-NaN)          : {n_estimable}")
    print(f"â-reliable  (π_anchored non-NaN)      : {n_reliable}   <- CLAIM-BEARING POPULATION")
    # NB: the pre-reg anchored ~34/187 to strong_generator_scan (#16), whose
    # â-reliability = reliable at the single β_amb-peak window (center ≥ 0.30s).
    # Here â-reliability = ≥1 reliable window in the â-anchored run over the late
    # grid — a different (looser) definition, so N may differ. No floor gates GO.

# %% [markdown]
# ## Reference — BH-FDR (Test 1 analogue, non-gating)

# %%
if len(reliable_df) > 0:
    reject_1t, q_1t, _, _ = multipletests(
        reliable_df["p_one_tailed"].fillna(1.0), method="fdr_bh", alpha=FDR_ALPHA
    )
    reliable_df["q_one_tailed"] = q_1t
    reliable_df["fdr_sig_one_tailed"] = reject_1t
    n_fdr = int(reliable_df["fdr_sig_one_tailed"].sum())
    print(f"[reference, non-gating] BH-FDR one-tailed survivors: {n_fdr} / {len(reliable_df)}")

# %% [markdown]
# ## Population statistic — CPO count-vs-null (LOCKED gate)

# %%
p_cpo = np.nan
n_obs = 0
cpo_null_counts = None
available_keys = []
if len(reliable_df) > 0 and null_arrays:
    n_obs = int((reliable_df["p_one_tailed"] < CPO_P).sum())
    keys = reliable_df["cell_key"].tolist()
    available_keys = [k for k in keys if k in null_arrays]
    n_missing = len(keys) - len(available_keys)
    if n_missing:
        print(f"  WARNING: {n_missing} â-reliable cells missing null arrays; excluded from CPO.")

    if available_keys:
        null_matrix = np.stack([null_arrays[k] for k in available_keys])  # (N_cells, N_PERMS)
        N_PERMS = null_matrix.shape[1]
        thresh_95 = np.percentile(null_matrix, 95, axis=1)                # per-cell 95th
        cpo_null_counts = (null_matrix > thresh_95[:, None]).sum(axis=0)  # (N_PERMS,)
        p_cpo = float(np.mean(cpo_null_counts >= n_obs))

        binom_p = float(scipy.stats.binomtest(n_obs, len(available_keys), CPO_P,
                                              alternative="greater").pvalue)
        binom_mean = len(available_keys) * CPO_P
        print(f"\nCPO count-vs-null (â-reliable cells, one-tailed gate p<{CPO_P}):")
        print(f"  observed # cells passing gate : {n_obs} / {len(available_keys)}")
        print(f"  CPO null count: mean={cpo_null_counts.mean():.2f}  SD={cpo_null_counts.std():.2f}")
        print(f"  p_cpo = {p_cpo:.4f}   (fraction of null counts ≥ {n_obs})")
        print(f"  [ref, non-gating] Binomial(N={len(available_keys)}, {CPO_P}) mean≈{binom_mean:.2f}, "
              f"exact p={binom_p:.4f}")

        pd.DataFrame([dict(
            n_obs=n_obs, n_reliable=len(reliable_df), n_with_null=len(available_keys),
            cpo_threshold=CPO_P, cpo_null_mean=float(cpo_null_counts.mean()),
            cpo_null_sd=float(cpo_null_counts.std()), p_cpo=p_cpo,
            p_binom=binom_p, binom_mean=binom_mean,
        )]).to_csv(OUT_DIR / "cpo.csv", index=False)
        print("  Saved cpo.csv")

# %% [markdown]
# ## LOCKED go/no-go (spec §8.4)

# %%
print("\n" + "=" * 62)
print("PRE-REGISTERED GO/NO-GO — late context-gated perceptual reactivation")
print("=" * 62)
if np.isnan(p_cpo):
    decision = "INDETERMINATE"
    print("Could not compute p_cpo (no â-reliable cells / no null arrays).")
else:
    go = p_cpo < CPO_P
    decision = "GO" if go else "NO-GO"
    print(f"  claim-bearing population (â-reliable cells): {len(available_keys)}")
    print(f"  observed cells passing per-cell one-tailed gate (p<{CPO_P}): {n_obs}")
    print(f"  p_cpo = {p_cpo:.4f}   (threshold {CPO_P}, one-tailed π>0, no floor)")
    print(f"\n  DECISION: {decision}")
    if go:
        print("  → licenses context-gated reactivation of the PERCEPTUAL code along the")
        print("    acoustic-tuning direction (mechanism-1, tuning-direction sense).")
    else:
        print("  → integration section retreats to NEGATIVE claims only")
        print("    (not lexicality/mismatch/surprisal; not an extended acoustic-or-")
        print("     perceptual response to the word-initial sound).")
print("  (Binomial + BH-FDR are references only; neither gates. Issue #23 records the claim.)")
print("=" * 62)

pd.DataFrame([dict(decision=decision, p_cpo=p_cpo, n_obs=n_obs,
                   n_reliable=(len(reliable_df) if len(all_cells) else 0),
                   cpo_threshold=CPO_P)]).to_csv(OUT_DIR / "go_no_go.csv", index=False)

# %% [markdown]
# ## Diagnostic — reliable-vs-all (the map's spine)
#
# Same-sign / significance among **â-reliable** cells vs among **all â-estimable**
# cells (via π_peak). The map's spine: the answer swings on â-reliability.

# %%
if len(all_cells) > 0:
    est = all_cells[all_cells["pi_peak"].notna()].copy()
    n_est = len(est)
    n_est_pos = int((est["pi_peak"] > 0).sum())
    n_est_sig = int((est["p_one_tailed_peak"] < CPO_P).sum()) if "p_one_tailed_peak" in est else 0
    print(f"[diagnostic] â-estimable via π_peak      : {n_est}  (π_peak>0: {n_est_pos}, "
          f"one-tailed p<{CPO_P}: {n_est_sig})")
    if len(reliable_df) > 0:
        n_rel_pos = int((reliable_df["pi_anchored"] > 0).sum())
        print(f"[diagnostic] â-reliable via π_anchored   : {len(reliable_df)}  "
              f"(π_anchored>0: {n_rel_pos}, one-tailed p<{CPO_P}: {n_obs})")

# %% [markdown]
# ## Diagnostic — cross-tab vs late manual annotation `behav @late` (non-gating)

# %%
test3_df = None
ann_path = Path(late_annotations_path)
if len(reliable_df) > 0 and ann_path.exists():
    ann = pd.read_csv(ann_path)
    key_cols = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
    behav_col = "behav @late" if "behav @late" in ann.columns else None
    if behav_col is not None:
        ann_small = ann[key_cols + [behav_col]].copy()
        ann_small["has_late_behav"] = ann_small[behav_col].notna() & (ann_small[behav_col].astype(str).str.strip() != "")
        merged = pd.merge(
            reliable_df[key_cols + ["pi_anchored", "p_one_tailed"]],
            ann_small[key_cols + ["has_late_behav"]],
            on=key_cols, how="left",
        )
        merged["has_late_behav"] = merged["has_late_behav"].fillna(False)
        merged["projection_pos"] = (merged["p_one_tailed"] < CPO_P) & (merged["pi_anchored"] > 0)
        test3_df = merged
        ct = pd.crosstab(merged["has_late_behav"], merged["projection_pos"],
                         margins=True, margins_name="Total")
        print("\n[diagnostic, non-gating] π_anchored positive × manual `behav @late`:")
        print(ct.to_string())
        merged.to_csv(OUT_DIR / "test3_late_annotation.csv", index=False)
    else:
        print("  `behav @late` column not found in annotations — skipping Test-3 diagnostic.")
else:
    print("  Test-3 diagnostic skipped (no â-reliable cells or annotations missing).")

# %% [markdown]
# ## Diagnostic plots

# %%
with PdfPages(str(OUT_DIR / "diagnostics.pdf")) as pdf:
    # π_anchored strip + CPO null
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    if len(reliable_df) > 0:
        pi_vals = reliable_df["pi_anchored"].values
        pooled_null = np.concatenate(list(null_arrays.values())) if null_arrays else np.array([])
        if len(pooled_null):
            ax.hist(pooled_null, bins=80, density=True, alpha=0.35, color="gray", label="pooled null (display)")
        colors = ["firebrick" if (reliable_df["p_one_tailed"].iloc[i] < CPO_P and pi_vals[i] > 0)
                  else "steelblue" for i in range(len(pi_vals))]
        ax.scatter(pi_vals, np.zeros_like(pi_vals) + 0.02, s=35, c=colors, zorder=5)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("π_anchored"); ax.set_title("â-reliable cells (red = gate-passing π>0)")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no â-reliable cells", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1]
    if cpo_null_counts is not None:
        ax.hist(cpo_null_counts, bins=40, color="gray", alpha=0.6, density=True, label="CPO matched null")
        ax.axvline(n_obs, color="firebrick", lw=1.5, label=f"observed={n_obs}")
        ax.axvline(len(available_keys) * CPO_P, color="steelblue", lw=1, ls="--",
                   label=f"Binom mean={len(available_keys)*CPO_P:.1f}")
        ax.set_xlabel("# cells π_perm > 95th pct"); ax.set_ylabel("density")
        ax.set_title(f"CPO null (p_cpo={p_cpo:.3f})"); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no CPO", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # π_anchored vs ‖â‖ and reliability histogram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    if len(reliable_df) > 0:
        ax.scatter(reliable_df["a_raw_norm"].values, reliable_df["pi_anchored"].values,
                   s=30, alpha=0.7, c="steelblue")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("‖â_raw‖ over run"); ax.set_ylabel("π_anchored"); ax.set_title("π_anchored vs ‖â‖")
    else:
        ax.text(0.5, 0.5, "no cells", ha="center", va="center", transform=ax.transAxes)
    ax = axes[1]
    if all_cells["n_reliable_windows"].notna().any():
        maxr = int(np.nanmax(all_cells["n_reliable_windows"].values))
        ax.hist(all_cells["n_reliable_windows"].dropna().values, bins=range(0, maxr + 2),
                color="darkorange", alpha=0.7, edgecolor="white")
        ax.set_xlabel("# reliable late windows"); ax.set_ylabel("cells"); ax.set_title("â-reliability")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
print("Saved diagnostics.pdf")

# %% [markdown]
# ## Save all-cells CSV

# %%
if len(reliable_df) > 0:
    reliable_df.to_csv(OUT_DIR / "all_reliable_cells.csv", index=False)
    print(f"Saved all_reliable_cells.csv  ({len(reliable_df)} rows)")
else:
    pd.DataFrame().to_csv(OUT_DIR / "all_reliable_cells.csv", index=False)
    print("No â-reliable cells — wrote empty all_reliable_cells.csv")
if len(all_cells) > 0:
    all_cells.to_csv(OUT_DIR / "all_cells.csv", index=False)

print("\nDone.")
