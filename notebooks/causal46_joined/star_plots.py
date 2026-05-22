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
# # causal46 AS-site star plots (B3 single-step + B4 class-balanced across-step)
#
# Two HGA star-plot galleries. AS sites are read directly from the causal6
# acoustic peaks parquet (filtered to `significant`) — that file is the
# authority; mirrors what `trial_balance_index.py` does. The
# (site, word_end, resampled) class counts come from
# `trial_balance_index.csv` (JON-42 / A2). Endpoint steps 1 and 6 are
# excluded from qualifying-step sets upstream (in `trial_balance_index.py`)
# since they calibrate the acoustic decoder.
#
# Uses `src.viz_provisional.provisional_star_plot` for B3; a notebook-local
# `matched_n_star_plot` helper for B4 (per-step class balance: at each
# ambiguous step, draw min_class[s] of each class; concat across steps).
# Both classes have the same step composition by construction, killing
# within-class step-acoustic confounds.
#
# See `docs/superpowers/plans/2026-05-20-causal46-star-plots.md` and
# Linear JON-43.

# %%
from __future__ import annotations

import os
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from src.stimuli import OFFSET_DICT
from src.viz_paper import add_textgrid, epoch_sfreq, epoch_tmin
from src.viz_provisional import (
    load_ambig_steps,
    load_epochs_dict,
    provisional_star_plot,
)

from _within_completion import extract_hga, resolve_behavior_col

# %%
REPO = Path(".").resolve()
OUT_DIR = REPO / "outputs/causal46_joined"
STAR_DIR = OUT_DIR / "star_plots"
SINGLE_DIR = STAR_DIR / "single_step"
MATCHED_DIR = STAR_DIR / "matched_n"
for d in (STAR_DIR, SINGLE_DIR, SINGLE_DIR / "per_site",
          MATCHED_DIR, MATCHED_DIR / "per_site"):
    d.mkdir(parents=True, exist_ok=True)

# AS sites: causal6 foldmean-maxstat peaks, FDR-significant. Same authority
# used by trial_balance_index.py.
CAUSAL6_PEAKS = REPO / "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"

EPOCH_DIR = Path(os.environ.get(
    "BARAKEET_EPOCH_DIR", str(REPO / "outputs/epochs_preprocessed"),
))

# Production default per A2 plan; expose K so downstream can tighten/loosen.
# K is the cell-level floor on the class-balanced subsample size (n_per_class
# in the summary). Per-step ambiguity is gated by `is_ambiguous_step` upstream.
K = 5
THRESHOLD_COL = f"meets_threshold_{K}"

# Acoustic search bounds (for dashed lines on the top panel).
_cfg = yaml.safe_load((REPO / "config.yaml").read_text())
AC_SEARCH_SMIN = int(_cfg["analysis"]["decoding"]["acoustic_peak_search_smin"])
AC_SEARCH_SMAX = int(_cfg["analysis"]["decoding"]["acoustic_peak_search_smax"])
print(f"REPO:      {REPO}")
print(f"EPOCH_DIR: {EPOCH_DIR}  (exists: {EPOCH_DIR.exists()})")
print(f"K={K}  AC_SEARCH=[{AC_SEARCH_SMIN}, {AC_SEARCH_SMAX}]")

# %% [markdown]
# ## Load AS sites (causal6 peaks) + JON-42 trial-balance outputs

# %%
_peaks_raw = pl.read_parquet(CAUSAL6_PEAKS)
if "significant" in _peaks_raw.columns:
    peaks = _peaks_raw.filter(pl.col("significant"))
else:
    peaks = _peaks_raw.filter(pl.col("p_value") < 0.05)
    print("⚠ no `significant` column — falling back to p_value < 0.05 (uncorrected)")

trial_balance = pl.read_csv(OUT_DIR / "trial_balance_index.csv")
trial_summary = pl.read_csv(OUT_DIR / "trial_balance_summary.csv")

print(f"AS sites: {peaks.height} across {peaks['subject'].n_unique()} subjects")
print(f"trial_balance: {trial_balance.height} rows")
print(f"trial_summary: {trial_summary.height} (site × word_end) rows")

# %% [markdown]
# ## Load epochs

# %%
needed_subjects = sorted(peaks["subject"].unique().to_list())
epochs_dict = load_epochs_dict(EPOCH_DIR)
missing = set(needed_subjects) - set(epochs_dict)
if missing:
    print(f"⚠ epoch files missing for subjects: {sorted(missing)}  "
          f"(those sites will be skipped)")
print(f"Epochs loaded: {sorted(epochs_dict)}")

ambig_steps_default = load_ambig_steps(epochs_dict)
print(f"ambig_steps_default: {len(ambig_steps_default)} (subject, pp, word_end) keys")

# %% [markdown]
# ## B3 — single-step cells (ambiguous step with min_class ≥ K)
#
# A single-step cell is the degenerate case of the cell-level rule: the
# pool over a 1-step set balanced = min_class for that step.

# %%
b3_cells = (
    trial_balance
    .filter(pl.col("is_ambiguous_step") & (pl.col("min_class") >= K))
    .join(
        peaks.select(["subject", "electrode_idx", "phoneme_pair",
                      "smin", "smax", "test_roc_auc"])
             .rename({"smin": "phon_smin", "smax": "phon_smax",
                      "test_roc_auc": "acoustic_peak_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end", "resampled"])
)
print(f"B3 cells (K={K}): {b3_cells.height} across "
      f"{b3_cells.select(['subject','electrode_idx','phoneme_pair']).unique().height} sites")
print(b3_cells.group_by("resampled").len().sort("resampled"))

# %%
sites_with_any_b3 = (
    b3_cells.select(["subject", "electrode_idx", "phoneme_pair"]).unique().height
)
print(f"AS sites with ≥1 B3 cell: {sites_with_any_b3}/{peaks.height}")
print(f"Sites with ZERO qualifying single-step cell at K={K}: "
      f"{peaks.height - sites_with_any_b3}")

# %% [markdown]
# ## Render B3 single-step star plots
#
# Reuses `provisional_star_plot` unchanged. For each B3 cell, we build a
# single-entry `ambig_steps` dict containing only that one step — the
# middle panel of the figure then shows that step alone, split by
# `behavior_dummy_forced`.

# %%
b3_failures: list[dict] = []
b3_manifest: list[dict] = []
b3_combined_pdf = SINGLE_DIR / "star_plots_all.pdf"

with PdfPages(b3_combined_pdf) as pdf:
    # Title page (always written; matplotlib >= 3.10 deletes empty PDFs).
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(0.5, 0.6,
            f"B3 single-step star plots\nK={K}\n"
            f"{b3_cells.height} cells across {sites_with_any_b3} AS sites",
            ha="center", va="center", fontsize=18)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

    for row in tqdm(b3_cells.iter_rows(named=True), total=b3_cells.height):
        subj = row["subject"]
        if subj not in epochs_dict:
            b3_failures.append({**row, "error": "no epochs for subject"})
            continue
        # Override ambig_steps for THIS cell only: a per-call dict containing
        # just the qualifying step at this (subject, pp, word_end).
        cell_ambig = {
            (subj, row["phoneme_pair"], row["word_end"]): [row["resampled"]],
        }
        try:
            fig = provisional_star_plot(
                subject=subj,
                electrode_idx=int(row["electrode_idx"]),
                phoneme_pair=row["phoneme_pair"],
                word_end=row["word_end"],
                epochs_dict=epochs_dict,
                ambig_steps=cell_ambig,
                phon_smin_c6=int(row["phon_smin"]),
                phon_smax_c6=int(row["phon_smax"]),
                phon_search_smin=AC_SEARCH_SMIN,
                phon_search_smax=AC_SEARCH_SMAX,
                acoustic_peak_auc=float(row["acoustic_peak_auc"]),
            )
            fig.suptitle(
                f"B3 step={row['resampled']}  |  {subj} e{row['electrode_idx']} "
                f"{row['phoneme_pair']} · {row['word_end']}\n"
                f"n_class0={row['n_class0']}  n_class1={row['n_class1']}  "
                f"min_class={row['min_class']}  ac={row['acoustic_peak_auc']:.3f}",
                y=1.01, fontsize=9,
            )
            site_pdf = (
                SINGLE_DIR / "per_site"
                / f"{subj}_{row['electrode_idx']}_{row['phoneme_pair']}_"
                  f"{row['word_end']}_step{row['resampled']}.pdf"
            )
            fig.savefig(site_pdf, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            b3_manifest.append({
                "subject": subj,
                "electrode_idx": row["electrode_idx"],
                "phoneme_pair": row["phoneme_pair"],
                "word_end": row["word_end"],
                "mode": "single_step",
                "resampled_step": row["resampled"],
                "qualifying_steps": "",
                "n_per_class": int(row["min_class"]),
                "n_total": int(row["n_total"]),
                "threshold_K": K,
                "status": "rendered",
            })
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"FAILED: {subj} e{row['electrode_idx']} {row['phoneme_pair']} "
                  f"{row['word_end']} step={row['resampled']}\n{tb}")
            b3_failures.append({**row, "error": repr(exc), "traceback": tb})
            plt.close("all")

if b3_failures:
    pl.DataFrame(b3_failures).write_csv(SINGLE_DIR / "failures.csv")
else:
    (SINGLE_DIR / "failures.csv").write_text("")
print(f"B3 rendered: {len(b3_manifest)} cells  |  failed: {len(b3_failures)}")

# %% [markdown]
# ## `matched_n_star_plot` — inline helper for B4
#
# Top: unambiguous trials at resampled 1 & 6 (acoustic anchor).
# Bottom: at each ambiguous step `s` in `qualifying_steps`, subsample
# `min_class[s]` trials per class (minority used in full; majority
# subsampled). Concat across steps. Both classes end up with the same
# step composition by construction — so the class-difference trace is
# free of within-class step-acoustic confounds. Total per-class N =
# `n_per_class = sum_s min_class[s]`. Plot mean HGA per class with SEM.
#
# Subsampling uses a deterministic seed for reproducibility. Raises if the
# per-step `min_class` summed by the caller disagrees with what the function
# recomputes from metadata — that's a counting bug, not bad data.


# %%
def matched_n_star_plot(
    subject,
    electrode_idx,
    phoneme_pair,
    word_end,
    qualifying_steps,
    *,
    epochs_dict,
    n_per_class,
    phon_smin=None,
    phon_smax=None,
    phon_search_smin=None,
    phon_search_smax=None,
    textgrid_dir="data/stimuli/textgrid",
    figsize=(6.5, 5.5),
    acoustic_peak_auc=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(0)
    ep = epochs_dict[subject]
    md = ep.metadata
    bhv_col = resolve_behavior_col(md)

    pp_mask = (md["phoneme_pair"] == phoneme_pair).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = extract_hga(ep_pp, electrode_idx)
    times = ep.times

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top: unambiguous step 1 & 6 (mirrors provisional_star_plot top).
    step_colors = {1: "#2166ac", 6: "#d73027"}
    for step, color in step_colors.items():
        mask = (md_pp["resampled"] == step).values
        if not mask.any():
            continue
        tr = hga[mask]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        ax_top.plot(times, m, color=color, lw=1.5,
                    label=f"step {step}  (n={mask.sum()})")
        ax_top.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    if phon_search_smin is not None and phon_search_smax is not None:
        for s in (phon_search_smin, phon_search_smax):
            ax_top.axvline(s / epoch_sfreq + epoch_tmin,
                           color="k", lw=0.6, ls="--", alpha=0.5)
    if phon_smin is not None:
        t_phon = np.array([phon_smin, phon_smax]) / epoch_sfreq + epoch_tmin
        ax_top.axvspan(*t_phon, color="#4dac26", alpha=0.20, label="acoustic peak")
    ax_top.axhline(0, color="k", lw=0.5, ls=":")
    ax_top.set_ylabel("HGA (z)")
    top_title = (
        f"{subject} e{electrode_idx} {phoneme_pair} · {word_end} — unambiguous"
    )
    if acoustic_peak_auc is not None:
        top_title += f"  (ac={acoustic_peak_auc:.3f})"
    ax_top.set_title(top_title, fontsize=9)
    ax_top.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # Bottom: at each ambiguous step, draw min_class[s] of each class
    # (minority in full; majority subsampled), then concat across steps.
    # Both classes end up with the same step composition by construction.
    bhv_colors = ["#762a83", "#1b7837"]
    we_mask = (md_pp["word_end"] == word_end).values
    bhv_vals = sorted(md_pp.loc[we_mask, bhv_col].dropna().unique())

    chosen_per_class: dict = {bhv: [] for bhv in bhv_vals}
    pool_check = 0
    for s in sorted(qualifying_steps):
        step_mask = we_mask & (md_pp["resampled"] == s).values
        step_idxs = {
            bhv: np.where(step_mask & (md_pp[bhv_col] == bhv).values)[0]
            for bhv in bhv_vals
        }
        n_s = min(len(v) for v in step_idxs.values())
        pool_check += n_s
        for bhv in bhv_vals:
            chosen_per_class[bhv].append(
                rng.choice(step_idxs[bhv], size=n_s, replace=False)
            )
    if pool_check != n_per_class:
        raise ValueError(
            f"pool size mismatch: per-step min_class summed to {pool_check}, "
            f"caller passed n_per_class={n_per_class} (data integrity bug)"
        )
    chosen_per_class = {
        bhv: np.concatenate(arrs) for bhv, arrs in chosen_per_class.items()
    }

    for i, bhv in enumerate(bhv_vals):
        tr = hga[chosen_per_class[bhv]]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(tr.shape[0])
        color = bhv_colors[i % len(bhv_colors)]
        ax_bot.plot(
            times, m, color=color, lw=1.5,
            label=f"resp={bhv}  (n={tr.shape[0]})",
        )
        ax_bot.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    ax_bot.axhline(0, color="k", lw=0.5, ls=":")
    ax_bot.set_ylabel("HGA (z)")
    ax_bot.set_xlabel("Time (s, post word onset)")
    ax_bot.set_title(
        f"Per-step class-balanced — steps {list(qualifying_steps)}  "
        f"({n_per_class} per class)",
        fontsize=9,
    )
    ax_bot.legend(fontsize=7, loc="upper left", framealpha=0.7)

    try:
        for ax in (ax_top, ax_bot):
            add_textgrid(
                ax,
                textgrid_dir=textgrid_dir,
                textgrid_file=f"11_{word_end}_dn_002.TextGrid",
                vline_extent=1.0,
            )
    except Exception:
        pass

    xlim = OFFSET_DICT.get(word_end, 1.0) + 0.1
    ax_top.set_xlim(0.0, xlim)
    fig.tight_layout()
    return fig


# %% [markdown]
# ## Sanity-check `matched_n_star_plot` on one site
#
# Pick the (site, word_end) with the largest qualifying-step count and
# render once before the full B4 loop.

# %%
_smoke = (
    trial_summary
    .filter(pl.col(THRESHOLD_COL) & (pl.col("n_ambiguous") >= 2))
    .sort("n_per_class", descending=True)
    .head(1)
)
if _smoke.height == 0:
    print(f"⚠ no (site, word_end) passes K={K} with ≥2 ambiguous steps "
          "— skipping smoke test")
else:
    _row = _smoke.row(0, named=True)
    _smoke_steps = [int(s) for s in _row["ambiguous_steps"].split(",")]
    _smoke_can = peaks.filter(
        (pl.col("subject") == _row["subject"])
        & (pl.col("electrode_idx") == _row["electrode_idx"])
        & (pl.col("phoneme_pair") == _row["phoneme_pair"])
    ).row(0, named=True)
    _smoke_n = int(_row["n_per_class"])
    print(f"smoke: {_row['subject']} e{_row['electrode_idx']} "
          f"{_row['phoneme_pair']} · {_row['word_end']}  "
          f"steps={_smoke_steps}  n_per_class={_smoke_n}")
    if _row["subject"] not in epochs_dict:
        print(f"  ⚠ epochs missing for {_row['subject']} — skipping smoke")
    else:
        fig = matched_n_star_plot(
            subject=_row["subject"],
            electrode_idx=int(_row["electrode_idx"]),
            phoneme_pair=_row["phoneme_pair"],
            word_end=_row["word_end"],
            qualifying_steps=_smoke_steps,
            epochs_dict=epochs_dict,
            n_per_class=_smoke_n,
            phon_smin=int(_smoke_can["smin"]),
            phon_smax=int(_smoke_can["smax"]),
            phon_search_smin=AC_SEARCH_SMIN,
            phon_search_smax=AC_SEARCH_SMAX,
            acoustic_peak_auc=float(_smoke_can["test_roc_auc"]),
        )
        fig.savefig(MATCHED_DIR / "_smoke.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {MATCHED_DIR / '_smoke.pdf'} — eyeball before the full gallery.")

# %% [markdown]
# ## B4 — class-balanced across-step cells
#
# Per (site, word_end), for each ambiguous step `s` (`is_ambiguous_step`)
# contribute `min_class[s]` trials per class to the cell pool (per-step
# class balance — minority in full, majority subsampled). Total per-class N
# = `n_per_class = sum_s min_class[s]`. Cell included iff `n_per_class ≥ K`
# AND ≥ 2 ambiguous steps. Per-step trial counts may differ; what matters
# is that both classes have the same step composition.

# %%
b4_per_step = (
    trial_balance
    .filter(pl.col("is_ambiguous_step"))
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.col("resampled").sort().alias("qualifying_steps"),
        pl.col("min_class").sum().alias("n_per_class"),
        pl.len().alias("n_qualifying"),
    )
    .filter((pl.col("n_qualifying") >= 2) & (pl.col("n_per_class") >= K))
    .join(
        peaks.select(["subject", "electrode_idx", "phoneme_pair",
                      "smin", "smax", "test_roc_auc"])
             .rename({"smin": "phon_smin", "smax": "phon_smax",
                      "test_roc_auc": "acoustic_peak_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
)
print(f"B4 cells (K={K}, ≥2 ambiguous steps, n_per_class ≥ K): {b4_per_step.height}")
print("n_per_class distribution:")
print(b4_per_step.group_by("n_per_class").len().sort("n_per_class"))

# %% [markdown]
# ## Render B4 class-balanced star plots

# %%
b4_failures: list[dict] = []
b4_manifest: list[dict] = []
b4_combined_pdf = MATCHED_DIR / "star_plots_all.pdf"

with PdfPages(b4_combined_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(0.5, 0.6,
            f"B4 class-balanced star plots\nK={K}  (≥2 ambiguous steps)\n"
            f"{b4_per_step.height} (site × word_end) cells",
            ha="center", va="center", fontsize=18)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

    for row in tqdm(b4_per_step.iter_rows(named=True), total=b4_per_step.height):
        subj = row["subject"]
        if subj not in epochs_dict:
            b4_failures.append({**row, "error": "no epochs for subject"})
            continue
        steps = [int(s) for s in row["qualifying_steps"]]
        try:
            fig = matched_n_star_plot(
                subject=subj,
                electrode_idx=int(row["electrode_idx"]),
                phoneme_pair=row["phoneme_pair"],
                word_end=row["word_end"],
                qualifying_steps=steps,
                epochs_dict=epochs_dict,
                n_per_class=int(row["n_per_class"]),
                phon_smin=int(row["phon_smin"]),
                phon_smax=int(row["phon_smax"]),
                phon_search_smin=AC_SEARCH_SMIN,
                phon_search_smax=AC_SEARCH_SMAX,
                acoustic_peak_auc=float(row["acoustic_peak_auc"]),
            )
            fig.suptitle(
                f"B4 class-balanced  |  {subj} e{row['electrode_idx']} "
                f"{row['phoneme_pair']} · {row['word_end']}  |  "
                f"steps={steps}  n_per_class={row['n_per_class']}  "
                f"ac={row['acoustic_peak_auc']:.3f}",
                y=1.01, fontsize=9,
            )
            site_pdf = (
                MATCHED_DIR / "per_site"
                / f"{subj}_{row['electrode_idx']}_{row['phoneme_pair']}_"
                  f"{row['word_end']}.pdf"
            )
            fig.savefig(site_pdf, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            b4_manifest.append({
                "subject": subj,
                "electrode_idx": row["electrode_idx"],
                "phoneme_pair": row["phoneme_pair"],
                "word_end": row["word_end"],
                "mode": "class_balanced",
                "resampled_step": None,
                "qualifying_steps": ",".join(str(s) for s in steps),
                "n_per_class": int(row["n_per_class"]),
                "n_total": int(row["n_per_class"] * 2),
                "threshold_K": K,
                "status": "rendered",
            })
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"FAILED: {subj} e{row['electrode_idx']} {row['phoneme_pair']} "
                  f"{row['word_end']}\n{tb}")
            b4_failures.append({**row, "error": repr(exc), "traceback": tb})
            plt.close("all")

if b4_failures:
    pl.DataFrame(b4_failures).write_csv(MATCHED_DIR / "failures.csv")
else:
    (MATCHED_DIR / "failures.csv").write_text("")
print(f"B4 rendered: {len(b4_manifest)} cells  |  failed: {len(b4_failures)}")

# %% [markdown]
# ## Combined manifest

# %%
manifest = pl.DataFrame(b3_manifest + b4_manifest)
manifest.write_csv(STAR_DIR / "star_plot_keys.csv")
print(f"Wrote manifest: {STAR_DIR / 'star_plot_keys.csv'}  ({manifest.height} rows)")
print(manifest.group_by("mode").len().sort("mode"))

# %% [markdown]
# ## Reviewer summary
#
# Use this section to decide whether B3 + B4 are sufficient evidence for the
# JON-41 Group B story, or whether to drop K to 4.

# %%
print(f"K={K} ({THRESHOLD_COL}) — production default")
print(f"\nAS sites (causal6 significant): {peaks.height}")
print(f"Sites with ≥1 B3 cell: {sites_with_any_b3}")
print(
    "Sites with ≥1 B4 cell (≥2 ambiguous steps, n_per_class ≥ K): "
    f"{b4_per_step.select(['subject','electrode_idx','phoneme_pair']).unique().height}"
)
print(f"\nB3 cells rendered: {sum(1 for m in b3_manifest if m['status']=='rendered')}")
print(f"B4 cells rendered: {sum(1 for m in b4_manifest if m['status']=='rendered')}")
print(
    f"Failures: B3={len(b3_failures)}, B4={len(b4_failures)} "
    "(must be 0 — investigate any > 0)"
)
print("\nNext: read outputs/causal46_joined/star_plots/"
      "{single_step,matched_n}/star_plots_all.pdf.")
print("If B3 coverage looks too sparse, re-run with K=4 (set K=4 at the top of the notebook).")
