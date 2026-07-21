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
# # Early Perceptual Projection — Aggregate analysis
#
# Combines per-subject outputs, runs Tests 1–3, and produces population-level plots.
#
# **Test 1** — BH-FDR across all sites on one-tailed p (candidate perceptual list).
# **Test 2** — CPO population count + Binomial analytic reference + LOO percentiles.
# **Test 3** — cross-tab vs site_type_relabel (type1–5); two-tailed detection.
#
# See: `docs/superpowers/plans/2026-07-16-early-perceptual-projection-spec.md`

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
results_dir = "outputs/causal46_joined/early_perceptual_projection"
site_type_relabel_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
# Automated (manual-free) site-type labels from early_window_site_types; its
# `site_type` column is the assign_site_type() output. Used to build the
# operational early_response_class; the manual relabel above is validation only.
site_type_computed_path = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv"
outdir = "outputs/causal46_joined/early_perceptual_projection"
fdr_alpha = 0.05
cpo_p_threshold = 0.05
# Type2-removal gate for early_response_class. "uncorrected": p_one_tailed <
# gate_alpha (matches the early_perceptual_windows operating point). "fdr":
# cross-subject BH-FDR one-tailed significant (subtracts only high-confidence
# aligned sites, retaining borderline type1s).
gate_alpha = 0.05
gate_mode = "uncorrected"

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(results_dir)
FDR_ALPHA = float(fdr_alpha)
CPO_P = float(cpo_p_threshold)
GATE_ALPHA = float(gate_alpha)
GATE_MODE = str(gate_mode)
assert GATE_MODE in {"uncorrected", "fdr"}, f"bad gate_mode: {GATE_MODE}"
print(f"Aggregate: results_dir={RESULTS_DIR}  fdr_alpha={FDR_ALPHA}")

# %% [markdown]
# ## Load per-subject site results

# %%
csv_paths = sorted(RESULTS_DIR.glob("*/site_results.csv"))
print(f"Found {len(csv_paths)} per-subject result CSVs")

if not csv_paths:
    print("No results found — writing empty outputs.")
    all_sites = pd.DataFrame()
else:
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            if len(df.columns) > 0:
                dfs.append(df)
            else:
                print(f"  Skipping empty (0-site subject): {p.parent.name}")
        except pd.errors.EmptyDataError:
            print(f"  Skipping empty (0-site subject): {p.parent.name}")
    all_sites = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print(f"Total sites: {len(all_sites)}")

# Load null distributions (needed for Test 2 CPO)
null_arrays = {}
for npz_path in sorted(RESULTS_DIR.glob("*/null_pi.npz")):
    try:
        data = np.load(str(npz_path))
        for k in data.files:
            null_arrays[k] = data[k]
    except Exception as e:
        print(f"Warning: could not load {npz_path}: {e}")
print(f"Null arrays loaded: {len(null_arrays)} sites")

# %% [markdown]
# ## Test 1 — BH-FDR across all sites

# %%
if len(all_sites) == 0:
    print("No sites — skipping Test 1.")
    valid_df = pd.DataFrame()
else:
    valid_mask = all_sites["pi_pooled"].notna()
    valid_df = all_sites[valid_mask].copy()
    print(f"Valid sites (non-null pi_pooled): {len(valid_df)} / {len(all_sites)}")

    if len(valid_df) > 0:
        # Primary family: one-tailed p per site (pooled π)
        reject_1t, q_1t, _, _ = multipletests(
            valid_df["p_one_tailed"].fillna(1.0), method="fdr_bh", alpha=FDR_ALPHA
        )
        valid_df["q_one_tailed"] = q_1t
        valid_df["fdr_sig_one_tailed"] = reject_1t

        # Two-tailed (for Test 3 "detected" = |π|-significant)
        reject_2t, q_2t, _, _ = multipletests(
            valid_df["p_two_tailed"].fillna(1.0), method="fdr_bh", alpha=FDR_ALPHA
        )
        valid_df["q_two_tailed"] = q_2t
        valid_df["fdr_sig_two_tailed"] = reject_2t

        # Per-word_end FDR — own BH pass, NOT folded into primary family
        for col_p, col_q, col_sig in [
            ("p_one_tailed_we0", "q_we0_one_tailed", "fdr_sig_we0_one_tailed"),
            ("p_two_tailed_we0", "q_we0_two_tailed", "fdr_sig_we0_two_tailed"),
            ("p_one_tailed_we1", "q_we1_one_tailed", "fdr_sig_we1_one_tailed"),
            ("p_two_tailed_we1", "q_we1_two_tailed", "fdr_sig_we1_two_tailed"),
        ]:
            vals = valid_df[col_p].fillna(1.0) if col_p in valid_df.columns else pd.Series([1.0]*len(valid_df))
            _, q, _, _ = multipletests(vals, method="fdr_bh", alpha=FDR_ALPHA)
            valid_df[col_q] = q
            valid_df[col_sig] = q <= FDR_ALPHA

        n_sig_1t = int(valid_df["fdr_sig_one_tailed"].sum())
        n_sig_2t = int(valid_df["fdr_sig_two_tailed"].sum())
        n_neg = int((valid_df["fdr_sig_two_tailed"] & (valid_df["pi_pooled"] < 0)).sum())
        n_pos = int((valid_df["fdr_sig_two_tailed"] & (valid_df["pi_pooled"] > 0)).sum())

        print(f"\nTest 1 — cross-subject BH-FDR (q≤{FDR_ALPHA}):")
        print(f"  One-tailed (π > 0): {n_sig_1t} / {len(valid_df)} sites")
        print(f"  Two-tailed (|π| > 0): {n_sig_2t} / {len(valid_df)} sites")
        print(f"    Positive (report tracks acoustic): {n_pos}")
        print(f"    Negative (report mirrors acoustic): {n_neg}")
    else:
        print("  No valid sites after filtering.")

# %% [markdown]
# ## Test 2 — CPO population count

# %%
if len(valid_df) > 0 and null_arrays:
    # Observed count: sites with uncorrected one-tailed p < CPO_P
    n_obs = int((valid_df["p_one_tailed"] < CPO_P).sum())
    n_sites_with_null = len(null_arrays)
    print(f"\nTest 2 — CPO population count (threshold={CPO_P}):")
    print(f"  Observed sites with p < {CPO_P}: {n_obs} / {len(valid_df)}")

    # Per-site 95th null percentile (LOO-approximated: for large N_PERMS the
    # difference between LOO and full percentile is negligible)
    site_keys = [
        f"{row.subject}_{row.electrode_idx}_{row.phoneme_pair}"
        for _, row in valid_df.iterrows()
    ]
    available_keys = [k for k in site_keys if k in null_arrays]
    n_missing = len(site_keys) - len(available_keys)
    if n_missing > 0:
        print(f"  Warning: {n_missing} sites missing null arrays; excluded from CPO.")

    if available_keys:
        # Stack null arrays: (n_sites_with_null, N_PERMS)
        null_matrix = np.stack([null_arrays[k] for k in available_keys])
        N_PERMS = null_matrix.shape[1]

        # Per-site 95th percentile (LOO approximation: use full distribution)
        thresh_95 = np.percentile(null_matrix, 95, axis=1)  # (n_sites_with_null,)

        # CPO matched null: for each perm i, count sites where π_perm[j,i] > thresh_95[j]
        cpo_null_counts = (null_matrix > thresh_95[:, None]).sum(axis=0)  # (N_PERMS,)

        p_cpo = float(np.mean(cpo_null_counts >= n_obs))
        print(f"  CPO null distribution: mean={cpo_null_counts.mean():.1f}  SD={cpo_null_counts.std():.1f}")
        print(f"  CPO p-value: {p_cpo:.4f}  (fraction of null counts ≥ {n_obs})")

        # Analytic reference: Binomial(n_sites, CPO_P)
        binom_p_exact = float(scipy.stats.binomtest(n_obs, len(available_keys), CPO_P, alternative="greater").pvalue)
        binom_mean = len(available_keys) * CPO_P
        binom_sd = np.sqrt(len(available_keys) * CPO_P * (1 - CPO_P))
        print(f"  Binomial(n={len(available_keys)}, p={CPO_P}) ref: mean={binom_mean:.1f}  SD={binom_sd:.1f}")
        print(f"  Binomial exact p: {binom_p_exact:.4f}")
        print(f"  LOO 95th percentile (approx): used full null (N_PERMS={N_PERMS})")

        # Save CPO results
        cpo_summary = {
            "n_obs": n_obs,
            "n_sites": len(valid_df),
            "n_sites_with_null": len(available_keys),
            "cpo_threshold": CPO_P,
            "cpo_null_mean": float(cpo_null_counts.mean()),
            "cpo_null_sd": float(cpo_null_counts.std()),
            "p_cpo": p_cpo,
            "p_binom": binom_p_exact,
            "binom_mean": binom_mean,
            "binom_sd": binom_sd,
        }
        pd.DataFrame([cpo_summary]).to_csv(OUT_DIR / "test2_cpo.csv", index=False)
        print("  Saved test2_cpo.csv")
    else:
        print("  No null arrays available — CPO skipped.")
        cpo_null_counts = None
else:
    cpo_null_counts = None
    if len(valid_df) == 0:
        print("\nTest 2 — CPO: no valid sites.")
    else:
        print("\nTest 2 — CPO: no null arrays loaded.")

# %% [markdown]
# ## Test 3 — cross-tab vs site_type_relabel

# %%
relabel_path = Path(site_type_relabel_path)
if not relabel_path.exists():
    print(f"\nTest 3: site_type_relabel not found at {relabel_path} — skipping.")
    test3_df = None
elif len(valid_df) == 0:
    print("\nTest 3: no valid sites — skipping.")
    test3_df = None
else:
    relabel = pd.read_csv(relabel_path)
    print("\nTest 3 — cross-tab vs site_type_relabel")
    print(f"  Relabel rows: {len(relabel)}")

    # Manual authority is the `site_type_relabel` column (type1–5); the computed
    # `site_type` column carries the retired grab_bag/complex vocabulary. Drop it
    # before renaming — otherwise the rename produces two columns both named
    # `site_type`, and `merged["site_type"]` returns a 2-column DataFrame instead
    # of a Series (crosstab then fails with a buffer-dimension error).
    relabel = relabel.drop(columns=["site_type"], errors="ignore")
    relabel = relabel.rename(columns={"site_type_relabel": "site_type"})

    # Merge on site keys
    merged = pd.merge(
        valid_df[["subject", "electrode_idx", "phoneme_pair",
                  "pi_pooled", "fdr_sig_two_tailed", "fdr_sig_one_tailed",
                  "p_one_tailed", "p_two_tailed"]],
        relabel[["subject", "electrode_idx", "phoneme_pair", "site_type"]],
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="left",
    )
    merged["site_type"] = merged["site_type"].fillna("unknown")

    # "Detected" = two-tailed FDR significant (|π| > 0), sign-annotated
    # (one-tailed positive list addresses the reactivation sub-question separately)
    merged["detected"] = "undetected"
    merged.loc[merged["fdr_sig_two_tailed"] & (merged["pi_pooled"] > 0), "detected"] = "detected_positive"
    merged.loc[merged["fdr_sig_two_tailed"] & (merged["pi_pooled"] < 0), "detected"] = "detected_negative"

    test3_df = merged.copy()

    # Cross-tab: detected(+/-/no) × site_type
    crosstab = pd.crosstab(
        merged["site_type"],
        merged["detected"],
        margins=True,
        margins_name="Total",
    )
    print("\nCross-tab: site_type × detected(+/-/undetected):")
    print(crosstab.to_string())

    n_matched = merged["site_type"].notna().sum()
    n_type5_excluded = (merged["site_type"] == "type5_behav_only").sum()
    print(f"\n  Sites with site_type label: {n_matched}")
    print(f"  type5_behav_only (excluded from projection): {n_type5_excluded}")

    # Report key comparisons
    for st in ["type1_acoustic_only", "type2_early_perceptual", "type3_asymmetric",
               "type4_early_perceptual_mirrored", "type5_behav_only"]:
        sub = merged[merged["site_type"] == st]
        if len(sub) == 0:
            continue
        n_det_pos = int((sub["detected"] == "detected_positive").sum())
        n_det_neg = int((sub["detected"] == "detected_negative").sum())
        n_undet = int((sub["detected"] == "undetected").sum())
        print(f"  {st} ({len(sub)}): +detected={n_det_pos}  -detected={n_det_neg}  undetected={n_undet}")

    print(
        "\nWindow-mismatch caveat: current labels span [word_onset, word_offset] "
        "while projection evaluates [ac_search_smin, ac_search_smax]. "
        "Misses may reflect post-280ms perceptual signal, not method disagreement."
    )

    crosstab.to_csv(OUT_DIR / "test3_crosstab.csv")
    test3_df.to_csv(OUT_DIR / "test3_detail.csv", index=False)
    print("  Saved test3_crosstab.csv + test3_detail.csv")

# %% [markdown]
# ## Diagnostic plots

# %%
with PdfPages(str(OUT_DIR / "diagnostics.pdf")) as pdf:
    # ── Plot 1: π distribution strip + histogram ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("π distribution — all subjects\n(all inference uses per-site nulls; pooled null display only)")

    ax = axes[0]
    if len(valid_df) > 0:
        pi_vals = valid_df["pi_pooled"].dropna().values
        pooled_null = np.concatenate(list(null_arrays.values())) if null_arrays else np.array([])
        if len(pooled_null):
            ax.hist(pooled_null, bins=80, density=True, alpha=0.35, color="gray", label="pooled null")
        colors = []
        for _, row in valid_df.iterrows():
            if pd.isna(row.get("fdr_sig_two_tailed", np.nan)):
                colors.append("steelblue")
            elif row.get("fdr_sig_one_tailed", False):
                colors.append("firebrick")
            elif row.get("fdr_sig_two_tailed", False):
                colors.append("darkorange")
            else:
                colors.append("steelblue")
        ax.scatter(pi_vals, np.zeros_like(pi_vals) + 0.02, s=35, c=colors, zorder=5)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("π (pooled)")
        ax.set_ylabel("density")
        ax.set_title("strip (red=1t-sig, orange=2t-only-sig)")
    else:
        ax.text(0.5, 0.5, "no sites", ha="center", va="center", transform=ax.transAxes)
    ax.legend(fontsize=7)

    ax = axes[1]
    if len(valid_df) > 0:
        pi_v = valid_df["pi_pooled"].dropna().values
        ax.hist(pi_v, bins=min(20, max(5, len(pi_v) // 3)), color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        if null_arrays:
            pooled_null_m = float(np.concatenate(list(null_arrays.values())).mean())
            ax.axvline(pooled_null_m, color="gray", lw=0.8, ls=":", label=f"null mean≈{pooled_null_m:.3f}")
        ax.set_xlabel("π (pooled)")
        ax.set_ylabel("count")
        ax.set_title("histogram")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no sites", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Plot 2: null calibration ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    if null_arrays:
        null_means = [arr.mean() for arr in null_arrays.values()]
        null_sds = [arr.std() for arr in null_arrays.values()]
        ax.scatter(null_sds, null_means, s=30, alpha=0.7, color="darkorange")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("null SD")
        ax.set_ylabel("null mean")
        ax.set_title("Null calibration: mean vs SD\n(mean should be ≈ 0)")
    else:
        ax.text(0.5, 0.5, "no null arrays", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Plot 3: null SD vs trial count ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    if null_arrays and len(valid_df) > 0:
        site_keys_plot = [
            f"{r.subject}_{r.electrode_idx}_{r.phoneme_pair}"
            for _, r in valid_df.iterrows()
            if f"{r.subject}_{r.electrode_idx}_{r.phoneme_pair}" in null_arrays
        ]
        n_trials = (
            valid_df.set_index(
                valid_df.apply(lambda r: f"{r.subject}_{r.electrode_idx}_{r.phoneme_pair}", axis=1)
            )
            .reindex(site_keys_plot)
        )
        n_tot = (n_trials["n_we0_total"].fillna(0) + n_trials["n_we1_total"].fillna(0)).values
        sd_vals = [null_arrays[k].std() for k in site_keys_plot]
        ax.scatter(n_tot, sd_vals, s=30, alpha=0.7, color="steelblue")
        ax.set_xlabel("qualifying trial count (we0 + we1)")
        ax.set_ylabel("null SD")
        ax.set_title("Null SD vs trial count\n(expect ~1/√n trend)")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Plot 4: π vs ‖a‖ ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(valid_df) > 0:
        pi_v = valid_df["pi_pooled"].dropna().values
        a_v = valid_df["a_norm"].dropna().values[:len(pi_v)]
        ax.scatter(a_v, pi_v, s=30, alpha=0.7, color="steelblue")
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("‖a‖ (acoustic template magnitude)")
        ax.set_ylabel("π (pooled)")
        ax.set_title("π vs ‖a‖\n(positive tail at low ‖a‖ → normalization artifact)")
    else:
        ax.text(0.5, 0.5, "no sites", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Plot 5: CPO null distribution ─────────────────────────────────────
    if cpo_null_counts is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        n_obs_plot = int((valid_df["p_one_tailed"] < CPO_P).sum())
        ax.hist(cpo_null_counts, bins=40, color="gray", alpha=0.6, density=True,
                label="CPO matched null")
        ax.axvline(n_obs_plot, color="firebrick", lw=1.5, label=f"observed={n_obs_plot}")
        ax.axvline(len(available_keys) * CPO_P, color="steelblue", lw=1, ls="--",
                   label=f"Binom mean={len(available_keys)*CPO_P:.1f}")
        ax.set_xlabel("# sites with π_perm > LOO 95th percentile")
        ax.set_ylabel("density")
        ax.set_title("Test 2: CPO null distribution")
        ax.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ── Plot 6: Test 3 cross-tab heatmap ──────────────────────────────────
    if test3_df is not None:
        ct = pd.crosstab(test3_df["site_type"], test3_df["detected"])
        if len(ct) > 0:
            fig, ax = plt.subplots(figsize=(8, max(3, len(ct) * 0.5 + 1)))
            im = ax.imshow(ct.values, aspect="auto", cmap="Blues")
            ax.set_xticks(range(len(ct.columns)))
            ax.set_xticklabels(ct.columns, rotation=20, ha="right")
            ax.set_yticks(range(len(ct.index)))
            ax.set_yticklabels(ct.index)
            for i in range(ct.shape[0]):
                for j in range(ct.shape[1]):
                    ax.text(j, i, str(ct.values[i, j]),
                            ha="center", va="center", fontsize=10)
            plt.colorbar(im, ax=ax, label="count")
            ax.set_title("Test 3: site_type × detected(+/-/undetected)")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # ── Plot 7: per-FDR-sig site null overlay ─────────────────────────────
    # For each one-tailed FDR-significant site: observed π vs its own null.
    if len(valid_df) > 0 and null_arrays:
        sig_sites = valid_df[valid_df.get("fdr_sig_one_tailed", pd.Series(False, index=valid_df.index)).astype(bool)]
        if len(sig_sites) > 0:
            ncols = min(3, len(sig_sites))
            nrows = int(np.ceil(len(sig_sites) / ncols))
            fig, axes_grid = plt.subplots(nrows, ncols,
                                          figsize=(5 * ncols, 3.5 * nrows),
                                          squeeze=False)
            fig.suptitle("FDR-significant sites: observed π vs per-site null", fontsize=11)
            for idx, (_, row) in enumerate(sig_sites.iterrows()):
                ax = axes_grid[idx // ncols][idx % ncols]
                site_key = f"{row.subject}_{row.electrode_idx}_{row.phoneme_pair}"
                null = null_arrays.get(site_key)
                if null is not None:
                    ax.hist(null, bins=60, density=True, color="gray", alpha=0.6,
                            label="permutation null")
                obs = row["pi_pooled"]
                ax.axvline(obs, color="firebrick", lw=2, label=f"π={obs:.3f}")
                ax.axvline(0, color="k", lw=0.7, ls="--")
                p_label = f"p={row['p_one_tailed']:.4f}, q={row['q_one_tailed']:.3f}"
                ax.set_title(f"{row.subject} e{int(row.electrode_idx)} {row.phoneme_pair}\n{p_label}",
                             fontsize=9)
                ax.set_xlabel("π")
                ax.legend(fontsize=7)
            # blank unused axes
            for idx in range(len(sig_sites), nrows * ncols):
                axes_grid[idx // ncols][idx % ncols].set_visible(False)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # ── Plot 8: π by phoneme_pair ─────────────────────────────────────────
    if len(valid_df) > 0:
        pairs = sorted(valid_df["phoneme_pair"].dropna().unique())
        fig, ax = plt.subplots(figsize=(max(5, len(pairs) * 2), 4))
        pair_order = sorted(pairs)
        jitter = 0.15
        for i, pp in enumerate(pair_order):
            sub = valid_df[valid_df["phoneme_pair"] == pp]["pi_pooled"].dropna()
            x = np.full(len(sub), i) + np.random.default_rng(7).uniform(-jitter, jitter, len(sub))
            sig_mask = valid_df.loc[valid_df["phoneme_pair"] == pp, "fdr_sig_one_tailed"].fillna(False)
            colors_pp = ["firebrick" if s else "steelblue"
                         for s in sig_mask.values]
            ax.scatter(x, sub.values, s=40, alpha=0.7, c=colors_pp, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [sub.median(), sub.median()],
                    color="k", lw=1.5, zorder=4)
            n_sig = int(valid_df.loc[valid_df["phoneme_pair"] == pp, "fdr_sig_one_tailed"].fillna(False).sum())
            ax.annotate(f"n={len(sub)}, sig={n_sig}", xy=(i, sub.max() if len(sub) else 0),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_xticks(range(len(pair_order)))
        ax.set_xticklabels(pair_order)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("phoneme pair")
        ax.set_ylabel("π (pooled)")
        ax.set_title("π by phoneme pair  (red = one-tailed FDR sig,  bar = median)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ── Plot 9: π by site_type ────────────────────────────────────────────
    if test3_df is not None and len(test3_df) > 0:
        type_order = [t for t in [
            "type1_acoustic_only", "type2_early_perceptual",
            "type3_asymmetric", "type4_early_perceptual_mirrored",
            "type5_behav_only", "complex", "grab_bag", "unknown",
        ] if t in test3_df["site_type"].values]
        fig, ax = plt.subplots(figsize=(max(6, len(type_order) * 1.5), 4))
        rng_jit = np.random.default_rng(13)
        for i, st in enumerate(type_order):
            sub = test3_df[test3_df["site_type"] == st]
            pi_v = sub["pi_pooled"].dropna().values
            x = np.full(len(pi_v), i) + rng_jit.uniform(-0.2, 0.2, len(pi_v))
            fdr_col = "fdr_sig_one_tailed"
            sig_col = [
                "firebrick" if (fdr_col in sub.columns and bool(row[fdr_col]))
                else "steelblue"
                for _, row in sub.iterrows()
            ]
            ax.scatter(x, pi_v, s=35, alpha=0.75, c=sig_col, zorder=3)
            if len(pi_v):
                ax.plot([i - 0.3, i + 0.3], [np.median(pi_v), np.median(pi_v)],
                        color="k", lw=1.5, zorder=4)
        ax.set_xticks(range(len(type_order)))
        ax.set_xticklabels(type_order, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("π (pooled)")
        ax.set_title("π by site type  (red = one-tailed FDR sig,  bar = median)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # ── Plot 10: qualifying trial count histogram ─────────────────────────
    if len(valid_df) > 0:
        n_trials_col = (valid_df["n_we0_total"].fillna(0) + valid_df["n_we1_total"].fillna(0))
        sig_mask_trials = valid_df.get("fdr_sig_one_tailed", pd.Series(False, index=valid_df.index)).fillna(False)
        fig, axes_tc = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("Qualifying perceptual trial count per site")

        ax = axes_tc[0]
        ax.hist(n_trials_col.values, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
        for v in n_trials_col[sig_mask_trials].values:
            ax.axvline(v, color="firebrick", lw=1.2, alpha=0.8)
        ax.set_xlabel("n_we0_total + n_we1_total")
        ax.set_ylabel("count")
        ax.set_title("histogram (red lines = FDR-sig sites)")

        ax = axes_tc[1]
        ax.scatter(n_trials_col.values,
                   valid_df["pi_pooled"].values,
                   c=["firebrick" if s else "steelblue" for s in sig_mask_trials],
                   s=35, alpha=0.7)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("qualifying trial count")
        ax.set_ylabel("π (pooled)")
        ax.set_title("π vs trial count  (red = FDR-sig)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print("Saved diagnostics.pdf")

# %% [markdown]
# ## Early response class (operational, manual-free)
#
# Three-way label composed from the automated `assign_site_type` output and the
# projection gate. Precedence:
#   1. `type2_aligned`  — projection flags an aligned early perceptual response
#      (`p_one_tailed < gate_alpha`, or FDR-significant one-tailed if gate_mode="fdr").
#   2. `acoustic_only`  — automated `site_type == type1_acoustic_only` AND not aligned.
#   3. `neither`        — everything else (complex / unknown / type3 / grab_bag /
#      A_unsigned, plus mirrored/out-of-pool type2 the one-tailed gate can't catch).
#
# Built over the automated site-type universe (left-joined with projection p),
# so a computed-type1 site with no projection result is still `acoustic_only`.
# No manual annotation. Emitted as `site_class.parquet` for downstream consumers.

# %%
SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]
comp_path = Path(site_type_computed_path)
if not comp_path.exists():
    print(f"Computed site types not found at {comp_path} — skipping early_response_class.")
    site_class = pd.DataFrame(columns=SITE_KEYS + ["computed_site_type", "early_response_class"])
    site_class.to_parquet(OUT_DIR / "site_class.parquet", index=False)
else:
    comp = pd.read_csv(comp_path)
    comp = comp[SITE_KEYS + ["site_type"]].rename(columns={"site_type": "computed_site_type"})
    comp["electrode_idx"] = comp["electrode_idx"].astype("int64")

    # Per-site projection p (and FDR flag) from valid_df; left-join keeps
    # computed-type1 sites even when the projection produced no valid π.
    if len(valid_df) > 0:
        proj_cols = [c for c in ["p_one_tailed", "fdr_sig_one_tailed", "pi_pooled"]
                     if c in valid_df.columns]
        proj_p = valid_df[SITE_KEYS + proj_cols].copy()
        proj_p["electrode_idx"] = proj_p["electrode_idx"].astype("int64")
    else:
        proj_p = pd.DataFrame(columns=SITE_KEYS + ["p_one_tailed", "fdr_sig_one_tailed", "pi_pooled"])

    site_class = comp.merge(proj_p, on=SITE_KEYS, how="left")

    if GATE_MODE == "uncorrected":
        aligned = (site_class["p_one_tailed"] < GATE_ALPHA).fillna(False)
    else:  # "fdr"
        aligned = site_class.get(
            "fdr_sig_one_tailed", pd.Series(False, index=site_class.index)
        ).fillna(False).astype(bool)

    is_type1 = site_class["computed_site_type"] == "type1_acoustic_only"
    site_class["early_response_class"] = np.select(
        [aligned, is_type1 & ~aligned],
        ["type2_aligned", "acoustic_only"],
        default="neither",
    )

    counts = site_class["early_response_class"].value_counts()
    print(f"early_response_class (gate_mode={GATE_MODE}, gate_alpha={GATE_ALPHA}):")
    for cls_name in ["type2_aligned", "acoustic_only", "neither"]:
        print(f"  {cls_name}: {int(counts.get(cls_name, 0))}")

    site_class.to_parquet(OUT_DIR / "site_class.parquet", index=False)
    print(f"Saved site_class.parquet  ({len(site_class)} rows)")

# %% [markdown]
# ## Save all-sites CSV

# %%
if len(valid_df) > 0:
    all_sites_out = valid_df.copy()
    if len(site_class) > 0:
        all_sites_out = all_sites_out.merge(
            site_class[SITE_KEYS + ["computed_site_type", "early_response_class"]],
            on=SITE_KEYS, how="left",
        )
    all_sites_out.to_csv(OUT_DIR / "all_sites.csv", index=False)
    print(f"Saved all_sites.csv  ({len(all_sites_out)} rows)")

    # Test 1: one-tailed significant list
    if "fdr_sig_one_tailed" in all_sites_out.columns:
        sig_list = all_sites_out[all_sites_out["fdr_sig_one_tailed"]].copy()
        sig_list.to_csv(OUT_DIR / "test1_one_tailed.csv", index=False)
        print(f"Saved test1_one_tailed.csv  ({len(sig_list)} sites)")
else:
    pd.DataFrame().to_csv(OUT_DIR / "all_sites.csv", index=False)
    pd.DataFrame().to_csv(OUT_DIR / "test1_one_tailed.csv", index=False)
    print("No sites — wrote empty all_sites.csv + test1_one_tailed.csv")

# %% [markdown]
# ## Summary

# %%
print("\n=== Early Perceptual Projection — Summary ===")
n_v = len(valid_df)
if n_v > 0:
    n_1t = int(valid_df["fdr_sig_one_tailed"].sum()) if "fdr_sig_one_tailed" in valid_df.columns else 0
    n_2t = int(valid_df["fdr_sig_two_tailed"].sum()) if "fdr_sig_two_tailed" in valid_df.columns else 0
    pi_mean = float(valid_df["pi_pooled"].mean())
    pi_sd = float(valid_df["pi_pooled"].std())
    print(f"Sites: {n_v}  |  π mean={pi_mean:.4f}  SD={pi_sd:.4f}")
    print(f"Test 1 — one-tailed FDR sig: {n_1t}/{n_v}  |  two-tailed: {n_2t}/{n_v}")
    pi_skew = float(scipy.stats.skew(valid_df["pi_pooled"].dropna()))
    null_all = np.concatenate(list(null_arrays.values())) if null_arrays else np.array([0.0])
    null_skew = float(scipy.stats.skew(null_all))
    print(f"Skewness: observed π={pi_skew:.3f}  pooled null={null_skew:.3f}  (right-skew = signal)")
else:
    print("No valid sites processed.")
print("=============================================")
print("\nDone.")
