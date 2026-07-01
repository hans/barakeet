# causal46 AS-Site Star Plots (single-step + matched-N across-step) — Implementation Plan

> **⚠ Sampling rule superseded (2026-07-01).** The B3/B4 per-step class-balance
> subsampling is now defined canonically in
> `notebooks/causal46_joined/_within_completion.py` (module docstring); pointer
> at `docs/superpowers/plans/2026-07-01-causal46-within-completion-subsampling.md`.
> Two things drifted since this plan: the gallery notebook `star_plots.py` was
> refactored into `_star_gallery.py` + `_within_completion.py`, and the
> single-draw "minority-in-full" visualisation was replaced by a both-classes
> bootstrap-with-replacement (the same draws the t-tests use). Historical record
> — not rewritten on purpose; where it differs from the module docstring, the
> code wins.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Linear:** [JON-43](https://linear.app/jonlab/issue/JON-43/star-plots-at-as-sites-single-step-matched-n-across-step), Group B items 3 & 4 of [JON-41](https://linear.app/jonlab/issue/JON-41). AS sites come directly from `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet` (filtered to `significant`) — the same authority `trial_balance_index.py` uses. Trial balance from [JON-42](https://linear.app/jonlab/issue/JON-42) (`trial_balance_index.csv`, `trial_balance_summary.csv` under `outputs/causal46_joined/`). Do **not** read `canonical_AS_sites.csv` — that CSV is a legacy artifact of the AS-reconciliation notebook and is not the authority.

**Goal:** For every causal6-significant AS site, render two flavors of within-completion HGA star plot driven by the JON-42 trial-balance index:
- **B3 (single-step):** one panel per (site, word_end, resampled) where `meets_threshold_5` is True — cleanest acoustic control, partial coverage.
- **B4 (matched-N across-step):** one panel per (site, word_end), pooling all qualifying steps after equal-N subsampling per step — broader coverage, residual step-driven-acoustic caveat.

Both use the **same visual style as `notebooks/causal46_joined/as_reconciliation.py`** (the `provisional_star_plot` helper from `src.viz_provisional`). Do not switch to the older `zoomin_hga` / `PaperData` path without checking first.

**Architecture:** One Jupytext percent notebook at `notebooks/causal46_joined/star_plots.py`. Loads causal6 peaks + trial-balance index + per-subject epochs once; renders B3 first (loop over qualifying single steps), then B4 (loop over (site, word_end), subsample, plot). Each pass writes per-site PDFs plus a combined multi-page PDF and a failures CSV. B3 reuses `provisional_star_plot` as-is via a per-call `ambig_steps` override. B4 uses a `matched_n_star_plot` helper **defined inline in the notebook** (not in `src/viz_provisional.py` — keeps `src/` untouched).

**Tech Stack:** Python, polars, mne, matplotlib. Local execution via `./.venv` (no GPU). All paths inside the notebook are resolved from `Path(".").resolve()` (same convention as `as_reconciliation.py`); the notebook works in any worktree.

---

## Context

- AS sites and trial balance are done. Inputs live at:
  - `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet` — causal6 foldmean-maxstat peaks aggregated with BH-FDR. Filter to `significant == True`; columns we use: `subject, electrode_idx, phoneme_pair, smin, smax, test_roc_auc, p_value, q_value, significant`. (`peaks` parquet is the authority — `trial_balance_index.py` reads it the same way; do not detour through `canonical_AS_sites.csv`.)
  - `outputs/causal46_joined/trial_balance_index.csv` — one row per (site × word_end × resampled): `n_class0, n_class1, n_total, min_class, meets_threshold_{4,5,10}`
  - `outputs/causal46_joined/trial_balance_summary.csv` — per (site, word_end), `qualifying_steps_{4,5,10}` as comma-joined strings + `n_qualifying_5`
- Reference plotter: `src.viz_provisional.provisional_star_plot` (3-panel: unambiguous step 1&6 / ambiguous controlled / decoder-view). Used in `notebooks/causal46_joined/as_reconciliation.py`. **Reuse, don't reinvent.**
- Reference loader: `src.viz_provisional.load_epochs_dict` and `load_ambig_steps`. Already handles `behavior_categorical → behavior_dummy_forced` aliasing and `add_metadata_features` enrichment.
- Recommended default threshold: **K=5** (per A2 plan rationale). Expose K as a notebook parameter; default to 5; trial-balance file already has `meets_threshold_{4,5,10}` so no recomputation needed.

### Definitions

- **B3 cell:** a (site, word_end, resampled) tuple from `trial_balance_index.csv` with `meets_threshold_K == True`. One panel per cell. Trace = mean HGA per `behavior_dummy_forced` class.
- **B4 cell:** a (site, word_end) tuple. Eligible steps = all `resampled` with `meets_threshold_K == True` for that (site, word_end). Per eligible step, subsample to `n_per_step = min over (steps × classes) of n_class{0,1}` trials per (step, class); pool the subsampled trials → trace per class.
- **Matched-N rationale:** if step 3 has 8 trials per class and step 4 has 25 per class, pooling raw biases the trace toward step 4's acoustics. Equal-N per step removes that bias inside a completion. Residual *step*-driven acoustic differences within the completion remain (e.g., step 3 vs step 4 spectra differ even at the same `word_end`) — flag in figure caption.

---

## File Structure

- **Create:** `notebooks/causal46_joined/star_plots.py` (main deliverable; `matched_n_star_plot` lives inline)
- **Create:** `outputs/causal46_joined/star_plots/single_step/per_site/*.pdf`
- **Create:** `outputs/causal46_joined/star_plots/single_step/star_plots_all.pdf`
- **Create:** `outputs/causal46_joined/star_plots/single_step/failures.csv`
- **Create:** `outputs/causal46_joined/star_plots/matched_n/per_site/*.pdf`
- **Create:** `outputs/causal46_joined/star_plots/matched_n/star_plots_all.pdf`
- **Create:** `outputs/causal46_joined/star_plots/matched_n/failures.csv`
- **Create:** `outputs/causal46_joined/star_plots/star_plot_keys.csv` — per-render manifest: site keys × what was plotted (step or matched-N step list) × n_per_class

---

## Inputs

### AS sites + trial balance
- `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet` — filter to `significant == True` (mirror the trial_balance_index.py pattern, including its `p_value < 0.05` fallback if `significant` is absent)
- `outputs/causal46_joined/trial_balance_index.csv`
- `outputs/causal46_joined/trial_balance_summary.csv` (used only for the manifest pre-flight; B3/B4 derive directly from `trial_balance_index.csv`)

### Per-subject epochs
- `outputs/epochs_preprocessed/{subject}_epo.fif`
- Load with `src.viz_provisional.load_epochs_dict(EPOCH_DIR)`. Override location via `BARAKEET_EPOCH_DIR` env var (as `as_reconciliation.py` does).

### Acoustic window highlight
- `smin`/`smax` from `phon_peaks_all.parquet` → top-panel `phon_smin_c6`/`phon_smax_c6` shade. Search-bound dashed lines from `config.yaml` `analysis.decoding.acoustic_peak_search_{smin,smax}` (same pattern as `as_reconciliation.py`). `test_roc_auc` → `acoustic_peak_auc` annotation in the figure suptitle.

---

## Outputs

### `star_plot_keys.csv` (manifest)
Columns: `subject, electrode_idx, phoneme_pair, word_end, mode, resampled_step, qualifying_steps, n_per_step, n_total, threshold_K, status`
- `mode` ∈ {`single_step`, `matched_n`}
- For `single_step`: `resampled_step` = the step; `qualifying_steps` = "" (empty)
- For `matched_n`: `resampled_step` = `null`; `qualifying_steps` = comma-joined; `n_per_step` = the min used for subsampling
- `status` ∈ {`rendered`, `skipped_no_qualifying`, `failed`}

### Per-site PDFs + combined PDF + failures CSV
Two parallel trees under `outputs/causal46_joined/star_plots/{single_step,matched_n}/`. Same packaging convention as `notebooks/causal4/star_plots.py`: title page + one page per site/cell + per-site PDFs in `per_site/`.

---

## Notes on TDD

Analysis notebook — no test suite. Verification is via:
1. Manifest counts (`star_plot_keys.csv`) against `trial_balance_summary.csv` qualifying-step counts at K=5.
2. Visual inspection of the combined PDFs.
3. Failures CSV must be empty for the typical case (a non-empty failures CSV is a bug, not a data property).

---

## Tasks

### Task 1: Scaffold notebook + load inputs

- [ ] **Step 1.1: Create the file**

Create `notebooks/causal46_joined/star_plots.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # causal46 AS-site star plots (B3 single-step + B4 matched-N across-step)
#
# Two HGA star-plot galleries driven by the JON-42 canonical-AS-site list
# and trial-balance index. Uses the `provisional_star_plot` helper from
# `as_reconciliation.py` for B3; matched-N variant for B4.
#
# See `docs/superpowers/plans/2026-05-20-causal46-star-plots.md` and Linear JON-43.

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
```

- [ ] **Step 1.2: Load AS sites (causal6 peaks) + trial balance**

```python
# %% [markdown]
# ## Load AS sites + JON-42 trial-balance outputs

# %%
CAUSAL6_PEAKS = REPO / "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"

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
```

- [ ] **Step 1.3: Load epochs**

```python
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
```

- [ ] **Step 1.4: Smoke-test as jupytext + commit**

```bash
./.venv/bin/jupytext --to notebook --output /dev/stdout notebooks/causal46_joined/star_plots.py | head -20
git add notebooks/causal46_joined/star_plots.py
git commit -m "scaffold causal46 star_plots notebook (JON-43)"
```

---

### Task 2: Build the single-step cell list (B3)

- [ ] **Step 2.1: Filter trial_balance to qualifying cells**

```python
# %% [markdown]
# ## B3 — single-step cells (meets_threshold_K)

# %%
b3_cells = (
    trial_balance
    .filter(pl.col(THRESHOLD_COL))
    .join(
        peaks.select(["subject", "electrode_idx", "phoneme_pair",
                      "smin", "smax", "test_roc_auc"])
             .rename({"smin": "phon_smin", "smax": "phon_smax",
                      "test_roc_auc": "acoustic_peak_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end", "resampled"])
)
print(f"B3 cells (K={K}): {b3_cells.height} "
      f"across {b3_cells.select(['subject','electrode_idx','phoneme_pair']).unique().height} sites")
print(b3_cells.group_by("resampled").len().sort("resampled"))
```

- [ ] **Step 2.2: Diagnostic — site coverage**

```python
# %%
sites_with_any_b3 = (
    b3_cells.select(["subject", "electrode_idx", "phoneme_pair"]).unique().height
)
print(f"AS sites with ≥1 B3 cell: {sites_with_any_b3}/{peaks.height}")
print(f"Sites with ZERO qualifying single-step cell at K={K}: "
      f"{peaks.height - sites_with_any_b3}")
```

If this is more than ~30% of AS sites, surface the diagnostic; the user may want to drop to K=4 before rendering. Don't auto-lower — let the user decide.

- [ ] **Step 2.3: Commit**

```bash
git add notebooks/causal46_joined/star_plots.py
git commit -m "compute B3 single-step cell list"
```

---

### Task 3: Render B3 single-step gallery

Reuse `provisional_star_plot` unchanged. For each B3 cell, build a single-entry `ambig_steps` dict that puts only that one step in the middle panel.

- [ ] **Step 3.1: Per-cell render loop**

```python
# %% [markdown]
# ## Render B3 single-step star plots

# %%
b3_failures: list[dict] = []
b3_manifest: list[dict] = []
combined_pdf = SINGLE_DIR / "star_plots_all.pdf"

with PdfPages(combined_pdf) as pdf:
    # Title page (always written; matplotlib >= 3.10 deletes empty PDFs).
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(0.5, 0.6, f"B3 single-step star plots\nK={K}\n"
                       f"{b3_cells.height} cells across {sites_with_any_b3} sites",
            ha="center", va="center", fontsize=18)
    ax.axis("off"); pdf.savefig(fig); plt.close(fig)

    for row in tqdm(b3_cells.iter_rows(named=True), total=b3_cells.height):
        subj = row["subject"]
        if subj not in epochs_dict:
            b3_failures.append({**row, "error": "no epochs for subject"})
            continue
        # Override ambig_steps for THIS cell only: a per-call dict containing
        # just the qualifying step at this (subject, pp, word_end).
        cell_ambig = {(subj, row["phoneme_pair"], row["word_end"]): [row["resampled"]]}
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
                "subject": subj, "electrode_idx": row["electrode_idx"],
                "phoneme_pair": row["phoneme_pair"], "word_end": row["word_end"],
                "mode": "single_step", "resampled_step": row["resampled"],
                "qualifying_steps": "",
                "n_per_step": int(row["min_class"]),
                "n_total": int(row["n_total"]),
                "threshold_K": K, "status": "rendered",
            })
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"FAILED: {subj} e{row['electrode_idx']} {row['phoneme_pair']} "
                  f"{row['word_end']} step={row['resampled']}\n{tb}")
            b3_failures.append({**row, "error": repr(exc), "traceback": tb})
            plt.close("all")

pl.DataFrame(b3_failures).write_csv(SINGLE_DIR / "failures.csv") if b3_failures \
    else (SINGLE_DIR / "failures.csv").write_text("")
print(f"B3 rendered: {len(b3_manifest)} cells  |  failed: {len(b3_failures)}")
```

- [ ] **Step 3.2: Commit**

```bash
git add notebooks/causal46_joined/star_plots.py outputs/causal46_joined/star_plots/single_step/
git commit -m "render B3 single-step star plots gallery"
```

---

### Task 4: Add `matched_n_star_plot` helper (inline in the notebook)

The helper takes a list of qualifying steps for one (subject, electrode_idx, phoneme_pair, word_end), an integer `n_per_step`, subsamples trials, and renders a two-panel (unambiguous top + matched-N bottom) HGA plot in the same visual style as `provisional_star_plot`'s top and middle panels. Defined inline in the notebook — keeps `src/` untouched.

- [ ] **Step 4.1: Define `matched_n_star_plot` in the notebook (right above the B4 cell list)**

```python
def matched_n_star_plot(
    subject: str,
    electrode_idx: int,
    phoneme_pair: str,
    word_end: str,
    qualifying_steps: list[int],
    *,
    epochs_dict: dict[str, "mne.Epochs"],
    n_per_step: int,
    phon_smin: int | None = None,
    phon_smax: int | None = None,
    phon_search_smin: int | None = None,
    phon_search_smax: int | None = None,
    textgrid_dir: str = "data/stimuli/textgrid",
    epoch_tmin: float = EPOCH_TMIN,
    epoch_sfreq: float = EPOCH_SFREQ,
    figsize: tuple[float, float] = (6.5, 5.5),
    acoustic_peak_auc: float | None = None,
    rng: np.random.Generator | None = None,
) -> "plt.Figure":
    """Two-panel matched-N star plot.

    Top: unambiguous trials at resampled 1 & 6 (identical to provisional_star_plot
         top panel) — visual anchor for acoustic selectivity.
    Bottom: within (word_end), for each step in qualifying_steps, subsample to
            `n_per_step` trials of each behavior_dummy_forced class; pool the
            subsamples; plot mean HGA per class with SEM.

    Subsampling uses `rng` (default: np.random.default_rng(0)) for reproducibility.
    Raises if any qualifying step lacks `n_per_step` trials of either class
    (caller is responsible for picking n_per_step ≤ min over steps × classes).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ep = epochs_dict[subject]
    md = ep.metadata
    bhv_col = ("behavior_dummy_forced"
               if "behavior_dummy_forced" in md.columns
               else "behavior_categorical")

    pp_mask = (md["phoneme_pair"] == phoneme_pair).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = (
        ep_pp.copy().apply_baseline((None, 0))
        .get_data(picks=[electrode_idx]).squeeze(1)
    )
    times = ep.times

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # — Top: unambiguous step 1 & 6 (same logic as provisional_star_plot top) —
    step_colors = {1: "#2166ac", 6: "#d73027"}
    for step, color in step_colors.items():
        mask = (md_pp["resampled"] == step).values
        if not mask.any():
            continue
        tr = hga[mask]
        m, se = tr.mean(0), tr.std(0) / np.sqrt(mask.sum())
        ax_top.plot(times, m, color=color, lw=1.5,
                    label=f"step {step}  (n={mask.sum()})")
        ax_top.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    if phon_search_smin is not None:
        for s in (phon_search_smin, phon_search_smax):
            ax_top.axvline(s / epoch_sfreq + epoch_tmin,
                           color="k", lw=0.6, ls="--", alpha=0.5)
    if phon_smin is not None:
        t_phon = np.array([phon_smin, phon_smax]) / epoch_sfreq + epoch_tmin
        ax_top.axvspan(*t_phon, color="#4dac26", alpha=0.20, label="acoustic peak")
    ax_top.axhline(0, color="k", lw=0.5, ls=":")
    ax_top.set_ylabel("HGA (z)")
    top_title = f"{subject} e{electrode_idx} {phoneme_pair} · {word_end} — unambiguous"
    if acoustic_peak_auc is not None:
        top_title += f"  (ac={acoustic_peak_auc:.3f})"
    ax_top.set_title(top_title, fontsize=9)
    ax_top.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # — Bottom: matched-N pooled across qualifying_steps within word_end —
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
        m, se = tr.mean(0), tr.std(0) / np.sqrt(tr.shape[0])
        color = bhv_colors[i % len(bhv_colors)]
        ax_bot.plot(times, m, color=color, lw=1.5,
                    label=f"resp={bhv}  (n={tr.shape[0]} = {n_per_step}×{len(qualifying_steps)} steps)")
        ax_bot.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    ax_bot.axhline(0, color="k", lw=0.5, ls=":")
    ax_bot.set_ylabel("HGA (z)")
    ax_bot.set_xlabel("Time (s, post word onset)")
    ax_bot.set_title(
        f"Matched-N pooled — steps {qualifying_steps}  "
        f"({n_per_step} per (step × class))", fontsize=9,
    )
    ax_bot.legend(fontsize=7, loc="upper left", framealpha=0.7)

    try:
        for ax in (ax_top, ax_bot):
            add_textgrid(ax, textgrid_dir=textgrid_dir,
                         textgrid_file=f"11_{word_end}_dn_002.TextGrid",
                         vline_extent=1.0)
    except Exception:
        pass

    xlim = OFFSET_DICT.get(word_end, 1.0) + 0.1
    ax_top.set_xlim(0.0, xlim)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4.2: Smoke-test the helper on one site**

Right after the helper definition:

```python
# %% [markdown]
# ## Sanity-check matched_n_star_plot on one site

# %%
# Pick a (site, word_end) with the largest qualifying step count, for the eyeball test.
_smoke = (
    trial_summary.filter(pl.col("n_qualifying_5") >= 2)
                 .sort("n_qualifying_5", descending=True).head(1)
).row(0, named=True)
_smoke_steps = [int(s) for s in _smoke[QUAL_COL].split(",")]
_smoke_row = peaks.filter(
    (pl.col("subject") == _smoke["subject"])
    & (pl.col("electrode_idx") == _smoke["electrode_idx"])
    & (pl.col("phoneme_pair") == _smoke["phoneme_pair"])
).row(0, named=True)

# n_per_step = min over (step × class) of min_class for those steps.
_smoke_n = int(
    trial_balance.filter(
        (pl.col("subject") == _smoke["subject"])
        & (pl.col("electrode_idx") == _smoke["electrode_idx"])
        & (pl.col("phoneme_pair") == _smoke["phoneme_pair"])
        & (pl.col("word_end") == _smoke["word_end"])
        & (pl.col("resampled").is_in(_smoke_steps))
    )["min_class"].min()
)
print(f"smoke: {_smoke['subject']} e{_smoke['electrode_idx']} "
      f"{_smoke['phoneme_pair']} · {_smoke['word_end']}  "
      f"steps={_smoke_steps}  n_per_step={_smoke_n}")
fig = matched_n_star_plot(
    subject=_smoke["subject"],
    electrode_idx=int(_smoke["electrode_idx"]),
    phoneme_pair=_smoke["phoneme_pair"],
    word_end=_smoke["word_end"],
    qualifying_steps=_smoke_steps,
    epochs_dict=epochs_dict,
    n_per_step=_smoke_n,
    phon_smin=int(_smoke_row["smin"]),
    phon_smax=int(_smoke_row["smax"]),
    phon_search_smin=AC_SEARCH_SMIN,
    phon_search_smax=AC_SEARCH_SMAX,
    acoustic_peak_auc=float(_smoke_row["test_roc_auc"]),
)
fig.savefig(MATCHED_DIR / "_smoke.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Wrote {MATCHED_DIR / '_smoke.pdf'} — eyeball this before the full gallery.")
```

- [ ] **Step 4.3: Commit**

```bash
git add notebooks/causal46_joined/star_plots.py outputs/causal46_joined/star_plots/matched_n/_smoke.pdf
git commit -m "add inline matched_n_star_plot helper + smoke test"
```

---

### Task 5: Render B4 matched-N gallery

- [ ] **Step 5.1: Build B4 cell list**

```python
# %% [markdown]
# ## B4 — matched-N across-step cells

# %%
# Per (site, word_end), gather qualifying steps + the per-step n_per_step.
# n_per_step = min over (qualifying steps × classes) of n_class{0,1}.
b4_per_step = (
    trial_balance
    .filter(pl.col(THRESHOLD_COL))
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.col("resampled").sort().alias("qualifying_steps"),
        pl.col("min_class").min().alias("n_per_step"),
        pl.len().alias("n_qualifying"),
    )
    .filter(pl.col("n_qualifying") >= 2)  # matched-N needs ≥2 steps to be meaningful
    .join(
        peaks.select(["subject", "electrode_idx", "phoneme_pair",
                      "smin", "smax", "test_roc_auc"])
             .rename({"smin": "phon_smin", "smax": "phon_smax",
                      "test_roc_auc": "acoustic_peak_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
)
print(f"B4 cells (K={K}, ≥2 qualifying steps): {b4_per_step.height}")
print(f"n_per_step distribution:")
print(b4_per_step.group_by("n_per_step").len().sort("n_per_step"))
```

- [ ] **Step 5.2: Render**

```python
# %%
b4_failures, b4_manifest = [], []
combined_pdf = MATCHED_DIR / "star_plots_all.pdf"

with PdfPages(combined_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.text(0.5, 0.6,
            f"B4 matched-N star plots\nK={K}  (≥2 qualifying steps)\n"
            f"{b4_per_step.height} (site × word_end) cells",
            ha="center", va="center", fontsize=18)
    ax.axis("off"); pdf.savefig(fig); plt.close(fig)

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
                f"steps={steps}  n_per_step={row['n_per_step']}  "
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
                "subject": subj, "electrode_idx": row["electrode_idx"],
                "phoneme_pair": row["phoneme_pair"], "word_end": row["word_end"],
                "mode": "matched_n", "resampled_step": None,
                "qualifying_steps": ",".join(str(s) for s in steps),
                "n_per_step": int(row["n_per_step"]),
                "n_total": int(row["n_per_step"] * len(steps) * 2),
                "threshold_K": K, "status": "rendered",
            })
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"FAILED: {subj} e{row['electrode_idx']} {row['phoneme_pair']} "
                  f"{row['word_end']}\n{tb}")
            b4_failures.append({**row, "error": repr(exc), "traceback": tb})
            plt.close("all")

(pl.DataFrame(b4_failures).write_csv(MATCHED_DIR / "failures.csv")
 if b4_failures else (MATCHED_DIR / "failures.csv").write_text(""))
print(f"B4 rendered: {len(b4_manifest)} cells  |  failed: {len(b4_failures)}")
```

- [ ] **Step 5.3: Write combined manifest**

```python
# %%
manifest = pl.DataFrame(b3_manifest + b4_manifest)
manifest.write_csv(STAR_DIR / "star_plot_keys.csv")
print(f"Wrote manifest: {STAR_DIR / 'star_plot_keys.csv'}  ({manifest.height} rows)")
print(manifest.group_by("mode").len().sort("mode"))
```

- [ ] **Step 5.4: Commit**

```bash
git add notebooks/causal46_joined/star_plots.py outputs/causal46_joined/star_plots/
git commit -m "render B4 matched-N star plots gallery + write manifest"
```

---

### Task 6: Reviewer summary + decision support

- [ ] **Step 6.1: Append summary block**

```python
# %% [markdown]
# ## Reviewer summary
#
# Use this section to decide whether B3 + B4 are sufficient evidence for the
# JON-41 Group B story, or whether to drop K to 4.

# %%
print(f"K={K} ({THRESHOLD_COL}) — production default")
print(f"\nAS sites (causal6 significant): {peaks.height}")
print(f"Sites with ≥1 B3 cell:   {sites_with_any_b3}")
print(
    "Sites with ≥1 B4 cell (≥2 qualifying steps): "
    f"{b4_per_step.select(['subject','electrode_idx','phoneme_pair']).unique().height}"
)
print(f"\nB3 cells rendered: {sum(1 for m in b3_manifest if m['status']=='rendered')}")
print(f"B4 cells rendered: {sum(1 for m in b4_manifest if m['status']=='rendered')}")
print(f"Failures: B3={len(b3_failures)}, B4={len(b4_failures)} (must be 0 — investigate any > 0)")
print("\nNext: read outputs/causal46_joined/star_plots/{single_step,matched_n}/star_plots_all.pdf.")
print("If B3 coverage looks too sparse, re-run with K=4 (set K=4 in Task 1.1).")
```

- [ ] **Step 6.2: End-to-end execution + acceptance checks**

```bash
./.venv/bin/jupytext --execute --to notebook --output - notebooks/causal46_joined/star_plots.py > /tmp/star_exec.ipynb 2>&1 || echo "EXECUTION FAILED"

ls -la outputs/causal46_joined/star_plots/single_step/star_plots_all.pdf \
       outputs/causal46_joined/star_plots/matched_n/star_plots_all.pdf \
       outputs/causal46_joined/star_plots/star_plot_keys.csv
```

- [ ] **Step 6.3: Commit**

```bash
git add notebooks/causal46_joined/star_plots.py
git commit -m "add reviewer summary to causal46 star_plots notebook"
```

---

## Acceptance criteria

1. `notebooks/causal46_joined/star_plots.py` runs end-to-end via `jupytext --execute` with no errors.
2. `outputs/causal46_joined/star_plots/` contains both `single_step/star_plots_all.pdf` and `matched_n/star_plots_all.pdf`, both non-empty (≥ 1 site rendered), plus per-site PDFs and an empty `failures.csv` in each.
3. `star_plot_keys.csv` row count = (B3 cells rendered) + (B4 cells rendered); `mode` column has both `single_step` and `matched_n` values.
4. B3 cell count = `trial_balance_index.csv` rows with `meets_threshold_5` minus any subjects missing epoch files.
5. B4 cell count = `trial_balance_summary.csv` rows with `n_qualifying_5 ≥ 2` minus any subjects missing epoch files.
6. Visual eyeball of the smoke `_smoke.pdf` and 5 random sites from each gallery: top panel shows clear step-1 vs step-6 separation in the acoustic window for high-`acoustic_peak_auc` sites; middle/bottom panels show response-split traces with non-zero N labelled in the legend.

## Out of scope

- Per-site t-tests on the within-completion contrast (separate sibling under JON-41).
- Picking the production K beyond exposing it as a notebook parameter (default 5).
- Anatomy / brain-position overlays.
- The bottom "decoder view" panel of `provisional_star_plot` is fine but redundant for B3; we keep it because it's already in the helper. Don't strip it.
- Re-running JON-42 for any subjects whose epoch files turn out to be missing — flag them and continue.
