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
# # Strong-generator test: β_ambig vs β_unamb for all behavioral discriminative windows
#
# For each row in `b_windows.parquet` (one union behavioral window per B4 cell),
# computes β_unamb — the endpoint step-6 minus step-1 HGA difference in the same
# `[smin, smax]` window, within the same word_end — and records it alongside the
# stored β_ambig.
#
# **Strong-generator prediction**: β_ambig ≈ β_unamb (both slopes produced by a
# single belief-driven generator). `difference ≫ 0` rules out a single generator
# for that cell.
#
# Both slopes use the fixed /n/−/d/ reference (step6−step1; no acoustic-polarity
# alignment), so signs are directly comparable across electrodes and subjects.

# %% tags=["parameters"]
b_windows_path = "outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet"
early_annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
filtered_manifest_path = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv"

epoch_dir = "outputs/epochs_preprocessed"
textgrid_dir = "textgrids"
outdir = "outputs/causal46_joined/strong_generator"
R_unamb = 1000
ci_low = 2.5
ci_high = 97.5
min_endpoint_n = 3
n_star_plot_examples = 20

include_fallback = False

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import mne

from src.data import add_metadata_features
from src.stimuli import OFFSET_DICT
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    bootstrap_endpoint_beta,
    extract_hga,
    matched_n_star_plot,
    summarize_replicate_array,
)
from sankey_early_late import EARLY_TYPES, EARLY_LABELS, early_category_map

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

b_windows_pd = pl.read_parquet(b_windows_path).to_pandas()

print(f"b_windows: {len(b_windows_pd)} rows  cols: {list(b_windows_pd.columns)}")
print(f"  subjects:      {sorted(b_windows_pd['subject'].unique())}")
print(f"  phoneme_pairs: {sorted(b_windows_pd['phoneme_pair'].unique())}")
print(f"  word_ends:     {sorted(b_windows_pd['word_end'].unique())}")

# %%
early_types_df = pd.read_csv(early_annotations_path)
early_types_df["early_category"] = early_types_df.site_type_relabel.replace(early_category_map)

filtered_manifest_df = pd.read_csv(filtered_manifest_path)

# Canonical qualifying-step set per cell, as used by the behavioral
# discriminative window search (strict `is_ambiguous_step`: min_class > 2 per
# step; derived from cell_manifest). Star plots MUST use this, not a looser
# on-the-fly recompute, so the displayed steps/trials match the steps the
# search and β_ambig actually used.
QUALIFYING_STEPS_LOOKUP: dict[tuple, list[int]] = {
    (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"], r["word_end"]): [
        int(s) for s in str(r["qualifying_steps"]).split(",") if s != ""
    ]
    for _, r in filtered_manifest_df.iterrows()
    if isinstance(r["qualifying_steps"], str) and r["qualifying_steps"] != ""
}
print(f"qualifying-step lookup: {len(QUALIFYING_STEPS_LOOKUP)} cells")

# %% [markdown]
# ## Compute β_unamb for each behavioral window
#
# Groups by (subject → phoneme_pair, electrode_idx) so epochs and HGA are
# extracted once per site. Within each site, `bootstrap_endpoint_beta` applies
# the word_end filter and the `[smin, smax]` window per row.

# %%
rows_out = []
n_nan = 0

for subject, subj_df in b_windows_pd.groupby("subject"):
    print(f"\n-- {subject}: {len(subj_df)} windows --")
    ep_path = Path(epoch_dir) / f"{subject}_epo.fif"
    ep_full = mne.read_epochs(str(ep_path), preload=True, verbose="WARNING")
    ep_full.metadata = add_metadata_features(ep_full.metadata.copy())
    md = ep_full.metadata

    for (pp, eidx), site_df in subj_df.groupby(["phoneme_pair", "electrode_idx"]):
        pp_mask = (md["phoneme_pair"] == pp).values
        ep_pp = ep_full[pp_mask]
        md_pp = md[pp_mask].reset_index(drop=True)
        hga = extract_hga(ep_pp, int(eidx))

        for _, row in site_df.iterrows():
            unamb_arr = bootstrap_endpoint_beta(
                hga, md_pp,
                word_end=row["word_end"],
                smin=int(row["smin"]),
                smax=int(row["smax"]),
                R=R_unamb,
                min_n=min_endpoint_n,
            )

            if unamb_arr is None:
                n_nan += 1
                u_mean = u_ci_lo = u_ci_hi = float("nan")
            else:
                s = summarize_replicate_array(unamb_arr, ci_low, ci_high)
                u_mean, u_ci_lo, u_ci_hi = s["mean"], s["ci_lo"], s["ci_hi"]

            rows_out.append({
                "subject": subject,
                "electrode_idx": int(eidx),
                "phoneme_pair": pp,
                "word_end": row["word_end"],
                "smin": int(row["smin"]),
                "smax": int(row["smax"]),
                "beta_unambig_mean": u_mean,
                "beta_unambig_ci_low": u_ci_lo,
                "beta_unambig_ci_high": u_ci_hi,
                "beta_ambig_mean": float(row["beta_ambig_mean"]),
                "beta_ambig_ci_low": float(row["beta_ambig_ci_low"]),
                "beta_ambig_ci_high": float(row["beta_ambig_ci_high"]),
            })

print(f"\nDone: {len(rows_out)} rows total, {n_nan} with NaN β_unamb (< {min_endpoint_n} endpoint trials).")

# %% [markdown]
# ## Save output parquet

# %%
_COLS = [
    "subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax",
    "beta_unambig_mean", "beta_unambig_ci_low", "beta_unambig_ci_high",
    "beta_ambig_mean", "beta_ambig_ci_low", "beta_ambig_ci_high",
]
result_df = (
    pd.DataFrame(rows_out, columns=_COLS)
    if rows_out
    else pd.DataFrame(columns=_COLS)
)

result_df = result_df.merge(
    early_types_df[["subject", "electrode_idx", "phoneme_pair", "early_category"]],
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="left")

out_path = OUT_DIR / "strong_generator.parquet"
result_df.to_parquet(str(out_path), index=False)
print(f"Saved {len(result_df)} rows → {out_path}")

# %% [markdown]
# ## Summary statistics

# %%
valid = result_df[result_df["beta_unambig_mean"].notna()].copy()
print(f"Rows with valid β_unamb : {len(valid)} / {len(result_df)}")
if len(valid):
    print(f"\nβ_ambig  (all rows)   mean={result_df['beta_ambig_mean'].mean():.3f}  "
          f"std={result_df['beta_ambig_mean'].std():.3f}")
    print(f"β_unamb  (valid only) mean={valid['beta_unambig_mean'].mean():.3f}  "
          f"std={valid['beta_unambig_mean'].std():.3f}")
    same_sign = np.sign(valid["beta_ambig_mean"].values) == np.sign(valid["beta_unambig_mean"].values)
    print(f"\nSame sign: {same_sign.sum()} / {len(valid)} ({100 * same_sign.mean():.1f}%)")

    by_pp = valid.groupby("phoneme_pair").agg(
        n=("beta_ambig_mean", "count"),
        beta_ambig_mean=("beta_ambig_mean", "mean"),
        beta_unamb_mean=("beta_unambig_mean", "mean"),
    )
    print(f"\nPer phoneme_pair:\n{by_pp.to_string()}")

# %% [markdown]
# ## Example star plots with behavioral window highlighted
#
# Top `n_star_plot_examples` cells by |β_ambig|. The orange span marks the
# behavioral discriminative window being tested. Green shading = acoustic peak.

# %%
_KEY = ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin"]
_bw_extra_cols = [c for c in
    ["phon_smin", "phon_smax", "acoustic_peak_auc", "n_per_class", "ci_excludes_zero"]
    if c in b_windows_pd.columns]
_bw_join = b_windows_pd[_KEY + ["smax"] + _bw_extra_cols].drop_duplicates(
    subset=_KEY + ["smax"]
)
result_joined = result_df.merge(_bw_join, on=_KEY + ["smax"], how="left")

_has_textgrid = (
    Path(textgrid_dir).exists()
    and bool(list(Path(textgrid_dir).glob("*.TextGrid"))[:1])
)

if not _has_textgrid:
    print(f"textgrid_dir='{textgrid_dir}' not found or empty — skipping star plots.")
else:
    _valid_joined = result_joined[result_joined["beta_unambig_mean"].notna()].copy()
    _valid_joined = _valid_joined.assign(
        abs_beta_ambig=_valid_joined["beta_ambig_mean"].abs()
    )

    # DEV
    _examples = _valid_joined.sample(n=min(n_star_plot_examples, len(_valid_joined)))
    # _examples = _valid_joined.assign(contrast=lambda x: x.beta_ambig_mean - x.beta_unambig_mean).nlargest(n_star_plot_examples, "contrast")

    _ep_cache = {}

    for _, ex in _examples.iterrows():
        subj = ex["subject"]
        pp = ex["phoneme_pair"]
        eidx = int(ex["electrode_idx"])
        we = ex["word_end"]
        smin_ex = int(ex["smin"])
        smax_ex = int(ex["smax"])
        t_smin = smin_ex / epoch_sfreq + epoch_tmin
        t_smax = smax_ex / epoch_sfreq + epoch_tmin

        if subj not in _ep_cache:
            ep = mne.read_epochs(
                str(Path(epoch_dir) / f"{subj}_epo.fif"),
                preload=True, verbose="WARNING",
            )
            ep.metadata = add_metadata_features(ep.metadata.copy())
            _ep_cache[subj] = ep
        ep_full_ex = _ep_cache[subj]

        qs = QUALIFYING_STEPS_LOOKUP.get((subj, eidx, pp, we))
        if not qs:
            print(f"  skip {subj} e{eidx} {pp}·{we}: no qualifying steps in manifest")
            continue

        phon_s = int(ex["phon_smin"]) if "phon_smin" in ex.index and pd.notna(ex["phon_smin"]) else None
        phon_e = int(ex["phon_smax"]) if "phon_smax" in ex.index and pd.notna(ex["phon_smax"]) else None
        ac_auc = ex.get("acoustic_peak_auc")
        n_pc = int(ex["n_per_class"]) if "n_per_class" in ex.index and pd.notna(ex["n_per_class"]) else 0

        try:
            fig = matched_n_star_plot(
                subj, eidx, pp, we, qs,
                epochs_dict={subj: ep_full_ex},
                n_per_class=n_pc,
                phon_smin=phon_s,
                phon_smax=phon_e,
                textgrid_dir=textgrid_dir,
                acoustic_peak_auc=float(ac_auc) if pd.notna(ac_auc) else None,
                R_plot=100,
            )
        except Exception as e:
            print(f"  star plot failed {subj} e{eidx} {pp}·{we}: {e}")
            plt.close("all")
            continue

        for ax in fig.axes:
            ax.axvspan(t_smin, t_smax, color="#fdae61", alpha=0.30, zorder=0)
            ax.axvline(t_smin, color="#b35806", lw=0.8, ls="--", alpha=0.7, zorder=5)
            ax.axvline(t_smax, color="#b35806", lw=0.8, ls="--", alpha=0.7, zorder=5)

        u_mean = ex["beta_unambig_mean"]
        a_mean = ex["beta_ambig_mean"]
        fig.suptitle(
            f"{subj} e{eidx} {pp}·{we}  window [{t_smin:.2f}, {t_smax:.2f}]s\n"
            f"β_ambig={a_mean:+.3f}  β_unamb={u_mean:+.3f}",
            fontsize=9, y=1.02,
        )
        plt.show()
        plt.close(fig)

# %% [markdown]
# ## Population scatter: β_ambig vs β_unamb
#
# Each point is one behavioral window (subject × electrode × phoneme_pair × word_end ×
# window_id). Error bars are 95 % bootstrap CIs. The dashed identity line (y = x) is
# the strong-generator prediction.

# %%
import seaborn as sns
g = sns.lmplot(data=result_df, x="beta_ambig_mean", y="beta_unambig_mean",
               hue="early_category", height=6, aspect=1.2)

# add gridlines
for ax in g.axes.flat:
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

# %%
PP_COLORS = {"dn": "#1b7837", "bm": "#762a83", "pb": "#e08214"}

fig_sc, ax_sc = plt.subplots(figsize=(5.5, 5.0))

valid_sc = result_df[result_df["beta_unambig_mean"].notna()].copy()

for pp_val in sorted(valid_sc["phoneme_pair"].unique()):
    pp_rows = valid_sc[valid_sc["phoneme_pair"] == pp_val]
    color = PP_COLORS.get(pp_val, "gray")

    x = pp_rows["beta_unambig_mean"].values
    y = pp_rows["beta_ambig_mean"].values
    xerr = [
        (pp_rows["beta_unambig_mean"] - pp_rows["beta_unambig_ci_low"]).values,
        (pp_rows["beta_unambig_ci_high"] - pp_rows["beta_unambig_mean"]).values,
    ]
    yerr = [
        (pp_rows["beta_ambig_mean"] - pp_rows["beta_ambig_ci_low"]).values,
        (pp_rows["beta_ambig_ci_high"] - pp_rows["beta_ambig_mean"]).values,
    ]

    ax_sc.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt="o", color=color, alpha=0.65, ms=4,
        elinewidth=0.6, capsize=2, label=pp_val,
        zorder=3,
    )

# identity line y = x (strong-generator prediction)
_all_vals = np.concatenate([
    valid_sc["beta_ambig_mean"].values if len(valid_sc) else np.array([0.0]),
    valid_sc["beta_unambig_mean"].values if len(valid_sc) else np.array([0.0]),
])
_finite = _all_vals[np.isfinite(_all_vals)]
if _finite.size >= 2:
    _span = _finite.max() - _finite.min()
    _lo = _finite.min() - 0.1 * _span
    _hi = _finite.max() + 0.1 * _span
else:
    _lo, _hi = -1.0, 1.0
ax_sc.plot([_lo, _hi], [_lo, _hi], "k--", lw=1.2, alpha=0.6, label="y = x (prediction)", zorder=2)
ax_sc.axhline(0, color="k", lw=0.5, ls=":", zorder=1)
ax_sc.axvline(0, color="k", lw=0.5, ls=":", zorder=1)

ax_sc.set_xlabel("β_unamb  (step 6 − step 1, endpoint trials)", fontsize=9)
ax_sc.set_ylabel("β_ambig  (heard /n/ − /d/, ambiguous trials)", fontsize=9)
ax_sc.set_title(
    f"Strong-generator test  —  β_ambig vs β_unamb\n"
    f"n={len(valid_sc)} windows with valid β_unamb  ({n_nan} NaN)",
    fontsize=9,
)
ax_sc.legend(fontsize=8, framealpha=0.7)
fig_sc.tight_layout()
plt.show()
