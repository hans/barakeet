# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # causal46 AS-site star plots (B3 single-step + B4 matched-N across-step)
#
# Two HGA star-plot galleries driven by the JON-42 canonical-AS-site list
# and trial-balance index. Uses the `provisional_star_plot` helper from
# `as_reconciliation.py` for B3; a notebook-local `matched_n_star_plot`
# helper for B4 (subsamples to equal-N per step × class before pooling).
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

# %%
REPO = Path(".").resolve()
OUT_DIR = REPO / "outputs/causal46_joined"
STAR_DIR = OUT_DIR / "star_plots"
SINGLE_DIR = STAR_DIR / "single_step"
MATCHED_DIR = STAR_DIR / "matched_n"
for d in (STAR_DIR, SINGLE_DIR, SINGLE_DIR / "per_site",
          MATCHED_DIR, MATCHED_DIR / "per_site"):
    d.mkdir(parents=True, exist_ok=True)

EPOCH_DIR = Path(os.environ.get(
    "BARAKEET_EPOCH_DIR", str(REPO / "outputs/epochs_preprocessed"),
))

# Production default per A2 plan; expose K so downstream can tighten/loosen.
K = 5
THRESHOLD_COL = f"meets_threshold_{K}"
QUAL_COL = f"qualifying_steps_{K}"

# Acoustic search bounds (for dashed lines on the top panel).
_cfg = yaml.safe_load((REPO / "config.yaml").read_text())
AC_SEARCH_SMIN = int(_cfg["analysis"]["decoding"]["acoustic_peak_search_smin"])
AC_SEARCH_SMAX = int(_cfg["analysis"]["decoding"]["acoustic_peak_search_smax"])
print(f"REPO:      {REPO}")
print(f"EPOCH_DIR: {EPOCH_DIR}  (exists: {EPOCH_DIR.exists()})")
print(f"K={K}  AC_SEARCH=[{AC_SEARCH_SMIN}, {AC_SEARCH_SMAX}]")

# %% [markdown]
# ## Load JON-42 outputs

# %%
canonical = pl.read_csv(OUT_DIR / "canonical_AS_sites.csv")
trial_balance = pl.read_csv(OUT_DIR / "trial_balance_index.csv")
trial_summary = pl.read_csv(OUT_DIR / "trial_balance_summary.csv")

print(f"canonical: {canonical.height} sites across "
      f"{canonical['subject'].n_unique()} subjects")
print(f"trial_balance: {trial_balance.height} rows")
print(f"trial_summary: {trial_summary.height} (site × word_end) rows")
print(canonical.group_by("bucket").len().sort("len", descending=True))

# %% [markdown]
# ## Load epochs

# %%
needed_subjects = sorted(canonical["subject"].unique().to_list())
epochs_dict = load_epochs_dict(EPOCH_DIR)
missing = set(needed_subjects) - set(epochs_dict)
if missing:
    print(f"⚠ epoch files missing for subjects: {sorted(missing)}  "
          f"(those sites will be skipped)")
print(f"Epochs loaded: {sorted(epochs_dict)}")

ambig_steps_default = load_ambig_steps(epochs_dict)
print(f"ambig_steps_default: {len(ambig_steps_default)} (subject, pp, word_end) keys")

# %% [markdown]
# ## B3 — single-step cells (`meets_threshold_K`)

# %%
b3_cells = (
    trial_balance
    .filter(pl.col(THRESHOLD_COL))
    .join(
        canonical.select(["subject", "electrode_idx", "phoneme_pair",
                          "smin", "smax", "peak_auc", "bucket"])
                 .rename({"smin": "phon_smin", "smax": "phon_smax",
                          "peak_auc": "acoustic_peak_auc"}),
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
print(f"Canonical sites with ≥1 B3 cell: {sites_with_any_b3}/{canonical.height}")
print(f"Sites with ZERO qualifying single-step cell at K={K}: "
      f"{canonical.height - sites_with_any_b3}")

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
            f"{b3_cells.height} cells across {sites_with_any_b3} sites",
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
                f"{row['phoneme_pair']} · {row['word_end']}  |  bucket={row['bucket']}\n"
                f"n_class0={row['n_class0']}  n_class1={row['n_class1']}  "
                f"min_class={row['min_class']}",
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
                "n_per_step": int(row["min_class"]),
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
# Bottom: within (word_end), for each step in `qualifying_steps`, subsample to
# `n_per_step` trials of each behavior class; pool subsamples; plot mean HGA
# per class with SEM.
#
# Subsampling uses a deterministic seed for reproducibility. Raises if any
# (qualifying step × class) lacks `n_per_step` trials — the caller computes
# n_per_step = min over (qualifying steps × classes) of min_class, so the
# raise indicates a counting bug, not bad data.


# %%
def matched_n_star_plot(
    subject,
    electrode_idx,
    phoneme_pair,
    word_end,
    qualifying_steps,
    *,
    epochs_dict,
    n_per_step,
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
    bhv_col = (
        "behavior_dummy_forced"
        if "behavior_dummy_forced" in md.columns
        else "behavior_categorical"
    )

    pp_mask = (md["phoneme_pair"] == phoneme_pair).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = (
        ep_pp.copy()
        .apply_baseline((None, 0))
        .get_data(picks=[electrode_idx])
        .squeeze(1)
    )
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

    # Bottom: matched-N pooled across qualifying_steps within word_end.
    we_mask = (md_pp["word_end"] == word_end).values
    bhv_colors = ["#762a83", "#1b7837"]
    bhv_vals = sorted(md_pp.loc[we_mask, bhv_col].dropna().unique())

    pooled = {bhv: [] for bhv in bhv_vals}
    for step in qualifying_steps:
        step_mask = we_mask & (md_pp["resampled"] == step).values
        for bhv in bhv_vals:
            cell_mask = step_mask & (md_pp[bhv_col] == bhv).values
            idxs = np.where(cell_mask)[0]
            if len(idxs) < n_per_step:
                raise ValueError(
                    f"step {step} class {bhv}: only {len(idxs)} trials < "
                    f"n_per_step={n_per_step} (caller picked too large an N)"
                )
            chosen = rng.choice(idxs, size=n_per_step, replace=False)
            pooled[bhv].append(hga[chosen])

    for i, bhv in enumerate(bhv_vals):
        tr = np.concatenate(pooled[bhv], axis=0)
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(tr.shape[0])
        color = bhv_colors[i % len(bhv_colors)]
        ax_bot.plot(
            times, m, color=color, lw=1.5,
            label=f"resp={bhv}  (n={tr.shape[0]} "
                  f"= {n_per_step}×{len(qualifying_steps)} steps)",
        )
        ax_bot.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    ax_bot.axhline(0, color="k", lw=0.5, ls=":")
    ax_bot.set_ylabel("HGA (z)")
    ax_bot.set_xlabel("Time (s, post word onset)")
    ax_bot.set_title(
        f"Matched-N pooled — steps {list(qualifying_steps)}  "
        f"({n_per_step} per (step × class))",
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
    trial_summary.filter(pl.col("n_qualifying_5") >= 2)
                 .sort("n_qualifying_5", descending=True).head(1)
)
if _smoke.height == 0:
    print(f"⚠ no (site, word_end) has n_qualifying_5 ≥ 2 — skipping smoke test")
else:
    _row = _smoke.row(0, named=True)
    _smoke_steps = [int(s) for s in _row[QUAL_COL].split(",")]
    _smoke_can = canonical.filter(
        (pl.col("subject") == _row["subject"])
        & (pl.col("electrode_idx") == _row["electrode_idx"])
        & (pl.col("phoneme_pair") == _row["phoneme_pair"])
    ).row(0, named=True)
    _smoke_n = int(
        trial_balance.filter(
            (pl.col("subject") == _row["subject"])
            & (pl.col("electrode_idx") == _row["electrode_idx"])
            & (pl.col("phoneme_pair") == _row["phoneme_pair"])
            & (pl.col("word_end") == _row["word_end"])
            & (pl.col("resampled").is_in(_smoke_steps))
        )["min_class"].min()
    )
    print(f"smoke: {_row['subject']} e{_row['electrode_idx']} "
          f"{_row['phoneme_pair']} · {_row['word_end']}  "
          f"steps={_smoke_steps}  n_per_step={_smoke_n}")
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
            n_per_step=_smoke_n,
            phon_smin=int(_smoke_can["smin"]),
            phon_smax=int(_smoke_can["smax"]),
            phon_search_smin=AC_SEARCH_SMIN,
            phon_search_smax=AC_SEARCH_SMAX,
            acoustic_peak_auc=float(_smoke_can["peak_auc"]),
        )
        fig.savefig(MATCHED_DIR / "_smoke.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {MATCHED_DIR / '_smoke.pdf'} — eyeball before the full gallery.")

# %% [markdown]
# ## B4 — matched-N across-step cells
#
# Per (site, word_end), gather qualifying steps + the per-step n_per_step.
# n_per_step = min over (qualifying steps × classes) of n_class{0,1}.
# Drop cells with <2 qualifying steps — matched-N is meaningless with one step.

# %%
b4_per_step = (
    trial_balance
    .filter(pl.col(THRESHOLD_COL))
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.col("resampled").sort().alias("qualifying_steps"),
        pl.col("min_class").min().alias("n_per_step"),
        pl.len().alias("n_qualifying"),
    )
    .filter(pl.col("n_qualifying") >= 2)
    .join(
        canonical.select(["subject", "electrode_idx", "phoneme_pair",
                          "smin", "smax", "peak_auc", "bucket"])
                 .rename({"smin": "phon_smin", "smax": "phon_smax",
                          "peak_auc": "acoustic_peak_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
)
print(f"B4 cells (K={K}, ≥2 qualifying steps): {b4_per_step.height}")
print("n_per_step distribution:")
print(b4_per_step.group_by("n_per_step").len().sort("n_per_step"))

# %% [markdown]
# ## Render B4 matched-N star plots

# %%
b4_failures: list[dict] = []
b4_manifest: list[dict] = []
b4_combined_pdf = MATCHED_DIR / "star_plots_all.pdf"

with PdfPages(b4_combined_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(0.5, 0.6,
            f"B4 matched-N star plots\nK={K}  (≥2 qualifying steps)\n"
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
                n_per_step=int(row["n_per_step"]),
                phon_smin=int(row["phon_smin"]),
                phon_smax=int(row["phon_smax"]),
                phon_search_smin=AC_SEARCH_SMIN,
                phon_search_smax=AC_SEARCH_SMAX,
                acoustic_peak_auc=float(row["acoustic_peak_auc"]),
            )
            fig.suptitle(
                f"B4 matched-N  |  {subj} e{row['electrode_idx']} "
                f"{row['phoneme_pair']} · {row['word_end']}  |  "
                f"bucket={row['bucket']}  steps={steps}  "
                f"n_per_step={row['n_per_step']}",
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
                "mode": "matched_n",
                "resampled_step": None,
                "qualifying_steps": ",".join(str(s) for s in steps),
                "n_per_step": int(row["n_per_step"]),
                "n_total": int(row["n_per_step"] * len(steps) * 2),
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
print(f"\nCanonical sites: {canonical.height}")
print(f"Sites with ≥1 B3 cell: {sites_with_any_b3}")
print(
    "Sites with ≥1 B4 cell (≥2 qualifying steps): "
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
