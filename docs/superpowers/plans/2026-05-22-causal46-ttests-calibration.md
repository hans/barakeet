# causal46 Per-Site T-Tests + Global-N Calibration (JON-44) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Linear:** [JON-44](https://linear.app/jonlab/issue/JON-44/per-site-t-tests-population-summary-of-within-completion-behavior), Group B items 5 & 6 of [JON-41](https://linear.app/jonlab/issue/JON-41). Consumes the same canonical inputs as [JON-43](https://linear.app/jonlab/issue/JON-43): `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet` (filtered to `significant`) and `outputs/causal46_joined/trial_balance_index.csv` / `trial_balance_summary.csv` from [JON-42](https://linear.app/jonlab/issue/JON-42).

**Goal:** Produce the headline within-completion behavioral statistics for the JON-41 Group B decision branch:
- Per-AS-site sliding-window Welch's t-tests on HGA between behavior classes, in **B3 single-step** and **B4 matched-N across-step** modes, at a sample size chosen so the underlying acoustic contrast is reliably detectable.
- A **global calibration N** sweep on the acoustic contrast (step 1 vs step 6) that fixes the inclusion threshold for cells and the per-test subsample size.
- Population summaries (fraction significant, effect-size distribution, breakdown by `phoneme_pair` and ROI) and a filtered-gallery hook back into the JON-43 star plots.

**Architecture:** Three Jupytext percent notebooks plus one notebook-local shared module, all under `notebooks/causal46_joined/`. JON-43's existing `star_plots.py` is refactored to import the shared module so its trial-selection logic is the same code that the t-tests run against.

```
notebooks/causal46_joined/
  _within_completion.py   ← shared module (NEW)
  star_plots.py           ← refactored to use _within_completion
  calibration.py          ← global-N power sweep (NEW)
  t_tests.py              ← per-site t-tests + population summary (NEW)
```

**Tech Stack:** Python, polars, mne, numpy, scipy (`scipy.stats.ttest_ind`), matplotlib. Local execution via `./.venv` (`uv run` per project preference). All paths resolved from `Path(".").resolve()` (same convention as `as_reconciliation.py` and `star_plots.py`).

---

## Context

- **Searchlight, not single window.** The acoustic and perceptual responses are temporally distinct (acoustic ~150–250 ms; perceptual diffuse, often after POD). A single-window t-test undersells the perceptual signal when the chosen window misses it. We compute a sliding-window t-test on per-trial HGA window means using the same `(smin, smax)` grid as `phon_roc_auc_searchlight_df` (window_size=15, stride=15 — matches `find_site_windows`). Per-site output is a t-trace over time; population summaries are sites × time.
- **Power-equalized via global calibration N.** Sites differ in trial counts. Rather than report effects that may simply track N, we (a) sweep candidate Ns against the *known* acoustic contrast (step 1 vs step 6) and pick the smallest N where ≥X% of AS sites are recovered; (b) subsample every behavioral t-test to that same N per class. Cells with `min_class < N_cal` are excluded as underpowered. This separates "did the contrast emerge" from "did the site have enough trials."
- **B3 / B4 cell definitions identical to JON-43.** A B3 cell is one (site, word_end, resampled) with `meets_threshold_K`. A B4 cell is one (site, word_end) pooling all qualifying steps after equal-N per-step subsampling. The same `select_cell_trials` helper feeds both the gallery and the t-test, so the test runs on exactly the trials the figure displays.
- **Acoustic-calibration t-test mechanics.** For a given N, for each AS site: subsample resampled=1 and resampled=6 trials to N per group, compute the searchlight t-test in the same grid, mark significant if **any** window in the causal6 acoustic search range hits p < 0.05 (uncorrected). Repeat across `n_seeds=50` subsample draws; report mean fraction-significant. Pick the smallest N giving ≥80% recovery as `N_calibrated`.

### Definitions

- **searchlight grid:** `smin ∈ range(search_smin, search_smax - window_size + 1, stride)`, `smax = smin + window_size`. `window_size=15`, `stride=15` (15 samples = 150 ms at 100 Hz).
- **search range per analysis:**
  - **Acoustic calibration** (`calibration.py`): `search_smin/smax` = `analysis.decoding.acoustic_peak_search_smin/smax` from `config.yaml` — the range where the known acoustic contrast lives.
  - **Behavioral t-tests** (`t_tests.py`): follow causal4 `find_site_windows` convention. Per (site, word_end): `behav_search_smin = phon_smax_c6` (start of behavioral search at end of that site's acoustic peak window, since perceptual response by construction follows acoustic); `behav_search_smax = word_end_offset_sample + 20` (word offset + 200 ms). `word_end_offset_sample` derived from `src.stimuli.OFFSET_DICT[word_end] * epoch_sfreq - epoch_tmin * epoch_sfreq`. This is wider than the acoustic range — the perceptual response is temporally diffuse and may extend past word offset (CLAUDE.md).
- **per-window t:** Welch's t on `hga_trials[:, smin:smax].mean(axis=1)` per group. `scipy.stats.ttest_ind(equal_var=False)`. Effect size = Hedges' g (small-sample-corrected Cohen's d).
- **B3 t-test cell:** subsample `n_per_class = N_cal` from each behavior class within the single qualifying step; if `min_class < N_cal`, skip the cell (`status=underpowered`).
- **B4 t-test cell:** for each qualifying step, subsample `N_cal` per class; pool across steps. `n_per_test = N_cal × n_qualifying_steps` per class.
- **acoustic calibration cell:** for a site at sweep N, subsample N from resampled=1 and N from resampled=6 (pooled across word_ends; the causal6 peak is word_end-agnostic).

---

## File Structure

- **Create:** `notebooks/causal46_joined/_within_completion.py` (shared module)
- **Modify:** `notebooks/causal46_joined/star_plots.py` (replace inline trial-selection with imports)
- **Create:** `notebooks/causal46_joined/calibration.py`
- **Create:** `notebooks/causal46_joined/t_tests.py`
- **Create:** `outputs/causal46_joined/calibration/` — `calibration_curve.csv`, `N_calibrated.txt`, `calibration_curve.pdf`
- **Create:** `outputs/causal46_joined/t_tests/` — `b3_results.parquet`, `b4_results.parquet`, `population_summary.csv`, `population_summary.pdf`, `star_plots_filtered/` (B3 + B4 powered / powered+significant PDFs)

---

## Inputs

### Canonical AS sites + trial balance
- `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet` — filter to `significant == True` (same fallback to `p_value < 0.05` as JON-43 / JON-42 if `significant` is absent). Columns used: `subject, electrode_idx, phoneme_pair, smin, smax, test_roc_auc`.
- `outputs/causal46_joined/trial_balance_index.csv` — per (site, word_end, resampled) class counts.
- `outputs/causal46_joined/trial_balance_summary.csv` — qualifying-step aggregates (used for B4).

### Per-subject epochs
- `outputs/epochs_preprocessed/{subject}_epo.fif`, loaded via `src.viz_provisional.load_epochs_dict`. `BARAKEET_EPOCH_DIR` env override supported as in JON-43.

### Searchlight grid
- `config.yaml`: `analysis.decoding.acoustic_peak_search_smin/smax`. `window_size=15`, `stride=15`. (`find_site_windows` in `src/viz_paper.py` uses the same defaults — we are not reusing the function because it returns the argmax window only; see Task 1.)

### Optional electrode metadata for ROI breakdown
- `src.data.get_electrode_df(subject)` — used only in the population summary for the ROI breakdown.

---

## Outputs

### `outputs/causal46_joined/calibration/`
- `calibration_curve.csv` — one row per (N, seed); columns: `N, seed, n_sites_total, n_sites_significant, frac_significant`.
- `calibration_curve.pdf` — line plot, mean ± std across seeds, frac_significant vs N. Horizontal line at 0.80; vertical line at chosen `N_calibrated`.
- `N_calibrated.txt` — single integer, read by `t_tests.py`.

### `outputs/causal46_joined/t_tests/b3_results.parquet`
One row per (B3 cell × searchlight window × seed). Columns:
```
subject, electrode_idx, phoneme_pair, word_end, resampled (B3 step), mode='single_step',
seed, smin, smax, tmin, tmax,
t_stat, df, p_value, hedges_g,
n_per_class (= N_cal),
status ∈ {'ok', 'underpowered'}
```

### `outputs/causal46_joined/t_tests/b4_results.parquet`
Same schema with `resampled` = null and a `qualifying_steps` string column (comma-joined). `n_per_test = N_cal × n_qualifying_steps` per class.

### `outputs/causal46_joined/t_tests/population_summary.csv`
One row per (mode ∈ {b3, b4}, smin, smax, p_threshold ∈ {0.05, 0.01}, fdr ∈ {raw, bh}, phoneme_pair, roi). Columns: `n_cells, n_significant, frac_significant, median_abs_hedges_g, q25_hedges_g, q75_hedges_g`. Long form so the plotting cell can pivot.

Plus marginal cuts: by `phoneme_pair` only, by ROI only, and overall.

### `outputs/causal46_joined/t_tests/population_summary.pdf`
Multi-page: (a) fraction-significant heatmap (sites × time) per mode; (b) per-time fraction-significant curves with permutation null band if cheap to compute; (c) effect-size distributions; (d) ROI / phoneme_pair breakdowns. Decision-support block at the end with the JON-41 thresholds (≥40% greenlight / <15% revisit) called out.

### `outputs/causal46_joined/t_tests/star_plots_filtered/`
- `b3_powered.pdf` — concatenation of per-site B3 PDFs from `outputs/causal46_joined/star_plots/single_step/per_site/` for cells where `status='ok'`.
- `b3_powered_significant.pdf` — further filtered to cells with `min_p (over searchlight windows) < 0.05` after BH-FDR across windows within site (or raw — emit both manifests).
- `b4_powered.pdf` / `b4_powered_significant.pdf` — parallel for B4.
- `filtered_manifest.csv` — keys + status flags.

The filter is one-way downstream: JON-43's gallery PDFs and `star_plot_keys.csv` are not modified.

---

## Notes on TDD

Analysis notebooks — no test suite. Verification is via:
1. Cell counts in `b3_results.parquet` / `b4_results.parquet` matching `star_plot_keys.csv` minus underpowered cells.
2. Calibration curve monotonically non-decreasing in N (within seed noise).
3. Spot-check 5 random sites: t-trace peak window's HGA group means match what the corresponding star plot shows.
4. ROI breakdown matches anatomy distribution in `electrode_df` (no missing labels).

---

## Tasks

### Task 1: Build `_within_completion.py` shared module

Defines the primitives used by `star_plots.py`, `calibration.py`, and `t_tests.py`. Lives next to the notebooks (the same scope as `matched_n_star_plot` in JON-43 — `src/` stays untouched).

- [ ] **Step 1.1: Create the module skeleton**

```python
# notebooks/causal46_joined/_within_completion.py
"""Shared trial-selection / HGA-extraction / searchlight-t-test primitives
used by star_plots.py (JON-43), calibration.py and t_tests.py (JON-44).

Notebook-local on purpose: src/ stays untouched while JON-41 Group B is in
flux. Promote to src/ only if a third caller appears outside this directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def select_cell_trials(
    md_pp: pd.DataFrame,
    *,
    word_end: str | None,
    resampled_steps: Sequence[int],
    group_col: str,                       # behavior column, e.g. 'behavior_dummy_forced'
    n_per_group: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[int, np.ndarray]:
    """Return {group_value: trial_indices_into_md_pp} for one B3 or B4 cell.

    md_pp must already be filtered to a single phoneme_pair and have a clean
    integer index (0..len-1). `group_col` partitions trials within each step
    in `resampled_steps`; n_per_group (if set) subsamples per (step × group).

    - For B3: resampled_steps=[step], group_col='behavior_dummy_forced',
      n_per_group=N_cal (subsamples to N_cal per class).
    - For B4: resampled_steps=qualifying_steps, group_col='behavior_dummy_forced',
      n_per_group=N_cal — sampling done per (step × class), pooled across steps.

    For the acoustic-calibration case (resampled=1 vs 6) use the sibling
    helper `select_endpoint_trials` instead — overloading group_col='resampled'
    here would conflate the step axis with the group axis.

    Raises if any (step × group) has fewer than n_per_group trials. Caller is
    responsible for filtering cells where min_class >= n_per_group beforehand.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if word_end is not None:
        we_mask = (md_pp["word_end"] == word_end).values
    else:
        we_mask = np.ones(len(md_pp), dtype=bool)

    groups = sorted(md_pp.loc[we_mask, group_col].dropna().unique())
    out: dict[int, list[np.ndarray]] = {g: [] for g in groups}

    for step in resampled_steps:
        step_mask = we_mask & (md_pp["resampled"] == step).values
        for g in groups:
            cell_mask = step_mask & (md_pp[group_col] == g).values
            idxs = np.where(cell_mask)[0]
            if n_per_group is not None:
                if len(idxs) < n_per_group:
                    raise ValueError(
                        f"step={step} group={g}: only {len(idxs)} trials < "
                        f"n_per_group={n_per_group}"
                    )
                idxs = rng.choice(idxs, size=n_per_group, replace=False)
            out[g].append(idxs)

    return {g: np.concatenate(parts) for g, parts in out.items()}


def select_endpoint_trials(
    md_pp: pd.DataFrame,
    *,
    n_per_group: int,
    rng: np.random.Generator | None = None,
    endpoints: tuple[int, int] = (1, 6),
) -> dict[int, np.ndarray]:
    """Return {endpoint_step: trial_indices} for the acoustic-calibration test.

    Pools across word_ends (the causal6 acoustic peak is word_end-agnostic).
    Raises if either endpoint has fewer than n_per_group trials.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out: dict[int, np.ndarray] = {}
    for step in endpoints:
        idxs = np.where((md_pp["resampled"] == step).values)[0]
        if len(idxs) < n_per_group:
            raise ValueError(
                f"endpoint step={step}: only {len(idxs)} trials < "
                f"n_per_group={n_per_group}"
            )
        out[step] = rng.choice(idxs, size=n_per_group, replace=False)
    return out


def extract_hga(ep, electrode_idx: int) -> np.ndarray:
    """Trials × time HGA for one electrode, baseline-corrected.

    Returned array is indexed by ep.metadata's integer index. Callers slice
    with the indices from select_cell_trials.
    """
    return (
        ep.copy()
        .apply_baseline((None, 0))
        .get_data(picks=[electrode_idx])
        .squeeze(1)
    )


@dataclass
class SearchlightTResult:
    smin: int
    smax: int
    t_stat: float
    df: float
    p_value: float
    hedges_g: float
    n_group1: int
    n_group2: int


def searchlight_ttest(
    hga: np.ndarray,                       # (trials, time)
    g1_idx: np.ndarray,
    g2_idx: np.ndarray,
    *,
    search_smin: int,
    search_smax: int,
    window_size: int = 15,
    stride: int = 15,
) -> list[SearchlightTResult]:
    """Welch's t per window over [search_smin, search_smax) with window_size/stride."""
    results: list[SearchlightTResult] = []
    n1, n2 = len(g1_idx), len(g2_idx)
    for start in range(search_smin, search_smax - window_size + 1, stride):
        x1 = hga[g1_idx, start : start + window_size].mean(axis=1)
        x2 = hga[g2_idx, start : start + window_size].mean(axis=1)
        t, p = ttest_ind(x1, x2, equal_var=False)
        # Welch–Satterthwaite df
        v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
        # Hedges' g with small-sample correction
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = (x1.mean() - x2.mean()) / sp if sp > 0 else 0.0
        J = 1 - 3 / (4 * (n1 + n2) - 9)
        g = J * d
        results.append(SearchlightTResult(
            smin=start, smax=start + window_size,
            t_stat=float(t), df=float(df), p_value=float(p),
            hedges_g=float(g), n_group1=n1, n_group2=n2,
        ))
    return results
```

- [ ] **Step 1.2: Refactor `star_plots.py` to use `select_cell_trials` + `extract_hga`**

In `notebooks/causal46_joined/star_plots.py`:
- Import from `_within_completion` at the top: `from _within_completion import select_cell_trials, extract_hga`.
- Replace the B3 inline HGA computation in the `provisional_star_plot` call site to compute trials via `select_cell_trials(md_pp, word_end=..., resampled_steps=[row['resampled']], group_col='behavior_dummy_forced', n_per_group=None)`. (Note: `provisional_star_plot` is reused unchanged for visual styling; the goal of the refactor is that the *cell definition* lives in one place. If the call signature doesn't permit injecting indices, factor the trial-counting bookkeeping into the manifest write only — the priority is B4.)
- Replace `matched_n_star_plot`'s inline subsampling block with a call to `select_cell_trials(..., n_per_group=row['n_per_step'])` + `extract_hga`. The plotting code then just does mean+SEM on the returned indices.

Acceptance: re-running `star_plots.py` produces byte-identical (or visually identical modulo the same seed) outputs to the pre-refactor version.

- [ ] **Step 1.3: Commit**

```bash
git add notebooks/causal46_joined/_within_completion.py notebooks/causal46_joined/star_plots.py
git commit -m "factor within-completion trial selection into shared module (JON-44 prep)"
```

---

### Task 2: Calibration notebook (`calibration.py`)

Sweep candidate Ns against the acoustic contrast (step 1 vs step 6) to pick a global `N_calibrated`.

- [ ] **Step 2.1: Scaffold + load inputs**

```python
# notebooks/causal46_joined/calibration.py
# %% [markdown]
# # Global-N calibration: smallest N at which the known acoustic contrast is detectable
#
# For each candidate N: for each AS site, subsample resampled=1 and resampled=6
# to N per group, run the searchlight t-test in the causal6 acoustic search
# range, mark the site significant if any window's p < 0.05. Repeat across 50
# seeds, report mean fraction-significant. Pick the smallest N giving >=80%
# recovery as N_calibrated.

# %%
from __future__ import annotations
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from tqdm.auto import tqdm

from src.viz_provisional import load_epochs_dict
from _within_completion import (
    select_endpoint_trials, extract_hga, searchlight_ttest,
)

REPO = Path(".").resolve()
OUT_DIR = REPO / "outputs/causal46_joined/calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_DIR = Path(os.environ.get("BARAKEET_EPOCH_DIR",
                                str(REPO / "outputs/epochs_preprocessed")))
_cfg = yaml.safe_load((REPO / "config.yaml").read_text())
SMIN = int(_cfg["analysis"]["decoding"]["acoustic_peak_search_smin"])
SMAX = int(_cfg["analysis"]["decoding"]["acoustic_peak_search_smax"])

CANDIDATE_NS = [3, 5, 8, 10, 15]
N_SEEDS = 50
P_THRESH = 0.05
TARGET_FRAC = 0.80
```

- [ ] **Step 2.2: Load AS sites + epochs** (same pattern as `star_plots.py`)

- [ ] **Step 2.3: Per-(site, N, seed) sweep**

```python
# %%
peaks = pl.read_parquet(REPO / "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet")
peaks = peaks.filter(pl.col("significant")) if "significant" in peaks.columns \
        else peaks.filter(pl.col("p_value") < 0.05)

epochs_dict = load_epochs_dict(EPOCH_DIR)

rows = []
for row in tqdm(peaks.iter_rows(named=True), total=peaks.height):
    subj = row["subject"]
    if subj not in epochs_dict:
        continue
    ep = epochs_dict[subj]
    md = ep.metadata
    pp_mask = (md["phoneme_pair"] == row["phoneme_pair"]).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = extract_hga(ep_pp, int(row["electrode_idx"]))

    # min_class for step 1 vs 6 — limits the maximum useful N for this site
    n1 = int((md_pp["resampled"] == 1).sum())
    n6 = int((md_pp["resampled"] == 6).sum())
    site_min = min(n1, n6)

    for N in CANDIDATE_NS:
        if site_min < N:
            # site cannot supply N trials of one endpoint; record as miss
            for seed in range(N_SEEDS):
                rows.append({
                    "subject": subj, "electrode_idx": row["electrode_idx"],
                    "phoneme_pair": row["phoneme_pair"], "N": N, "seed": seed,
                    "significant": False, "skipped_insufficient": True,
                })
            continue
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            picks = select_endpoint_trials(md_pp, n_per_group=N, rng=rng)
            res = searchlight_ttest(
                hga, picks[1], picks[6],
                search_smin=SMIN, search_smax=SMAX,
            )
            sig = any(r.p_value < P_THRESH for r in res)
            rows.append({
                "subject": subj, "electrode_idx": row["electrode_idx"],
                "phoneme_pair": row["phoneme_pair"], "N": N, "seed": seed,
                "significant": sig, "skipped_insufficient": False,
            })

curve = pl.DataFrame(rows)
curve.write_csv(OUT_DIR / "calibration_curve_raw.csv")
```

- [ ] **Step 2.4: Aggregate + pick N_calibrated**

```python
# %%
agg = (curve.group_by(["N", "seed"])
            .agg(pl.col("significant").mean().alias("frac"))
            .group_by("N")
            .agg(pl.col("frac").mean().alias("frac_mean"),
                 pl.col("frac").std().alias("frac_std"))
            .sort("N"))
print(agg)

passing = agg.filter(pl.col("frac_mean") >= TARGET_FRAC).sort("N")
N_cal = int(passing["N"][0]) if passing.height else int(agg["N"][-1])
print(f"N_calibrated = {N_cal}")

agg.write_csv(OUT_DIR / "calibration_curve.csv")
(OUT_DIR / "N_calibrated.txt").write_text(str(N_cal))
```

- [ ] **Step 2.5: Plot calibration curve**

Line plot of `frac_mean ± frac_std` vs N, with `TARGET_FRAC` and `N_cal` annotated. Save to `calibration_curve.pdf`.

- [ ] **Step 2.6: Commit**

```bash
git add notebooks/causal46_joined/calibration.py outputs/causal46_joined/calibration/
git commit -m "global-N acoustic-contrast calibration sweep (JON-44 calibration)"
```

---

### Task 3: T-test notebook (`t_tests.py`) — B3 and B4 at N_calibrated

- [ ] **Step 3.1: Scaffold + read N_calibrated**

```python
# notebooks/causal46_joined/t_tests.py
# %% [markdown]
# # B3 + B4 within-completion behavior t-tests at N_calibrated
#
# Per-AS-site searchlight Welch's t-test between behavior_dummy_forced=0 and =1,
# subsampled to N_calibrated per class. B3 = one qualifying step at a time;
# B4 = pool across qualifying steps after equal-N subsampling per (step × class).
# All cells run at exactly N_cal per class so power is constant across sites.
# Multiple seeds; report median t-trace per cell.

# %%
from __future__ import annotations
from pathlib import Path
import os, numpy as np, polars as pl, yaml
from tqdm.auto import tqdm
from src.viz_provisional import load_epochs_dict
from _within_completion import select_cell_trials, extract_hga, searchlight_ttest

REPO = Path(".").resolve()
OUT_DIR = REPO / "outputs/causal46_joined/t_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_CAL = int((REPO / "outputs/causal46_joined/calibration/N_calibrated.txt").read_text().strip())
N_SEEDS = 50
print(f"N_calibrated = {N_CAL}; n_seeds = {N_SEEDS}")
```

- [ ] **Step 3.2: B3 t-tests**

For each B3 cell (same definition as JON-43's `b3_cells`), filter to `min_class >= N_CAL`. For each retained cell, for each seed, compute per-(site × word_end) behavioral search range as `(phon_smax_c6, word_end_offset_sample + 20)`; call `select_cell_trials(...n_per_group=N_CAL)` + `searchlight_ttest(..., search_smin=behav_smin, search_smax=behav_smax)`. Emit one row per (cell × window × seed) to `b3_results.parquet`.

Skipped cells (`min_class < N_CAL`) are written with `status='underpowered'` and no per-window rows. They appear in the manifest so coverage diagnostics are honest.

- [ ] **Step 3.3: B4 t-tests**

Same shape, parameterized over (site × word_end × qualifying_steps). Behavioral search range computed the same way.

**Critical: re-filter qualifying_steps at N_CAL.** JON-43's `b4_per_step` derives qualifying_steps from `trial_balance_index.csv` at `K=5`. If `N_CAL > 5`, some steps within a B4 cell will have `min_class < N_CAL` and `select_cell_trials` will raise. Re-compute per cell:

```python
b4_per_step_ncal = (
    trial_balance
    .filter(pl.col("min_class") >= N_CAL)      # NOT meets_threshold_5
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(pl.col("resampled").sort().alias("qualifying_steps_ncal"),
         pl.len().alias("n_qualifying_ncal"))
    .filter(pl.col("n_qualifying_ncal") >= 2)  # matched-N needs ≥2 steps
    .join(peaks.select(["subject", "electrode_idx", "phoneme_pair",
                        "smin", "smax"])
                .rename({"smin": "phon_smin", "smax": "phon_smax"}),
          on=["subject", "electrode_idx", "phoneme_pair"], how="inner")
)
```

`n_per_test = N_CAL × n_qualifying_ncal` per class. Cells dropped by this re-filter are reported in the manifest with `status='underpowered_b4_too_few_steps_at_ncal'` for symmetry with B3.

- [ ] **Step 3.4: Aggregate across seeds**

Per (cell × window): median `t_stat`, `hedges_g`, fraction-of-seeds with `p < 0.05`. Write to a `*_per_cell.parquet` companion file. This is the per-cell statistic the population summary aggregates over.

- [ ] **Step 3.5: Commit**

```bash
git add notebooks/causal46_joined/t_tests.py outputs/causal46_joined/t_tests/b3_results.parquet outputs/causal46_joined/t_tests/b4_results.parquet
git commit -m "B3 + B4 per-site searchlight t-tests at N_calibrated (JON-44)"
```

---

### Task 4: Population summary + filtered-gallery hook

- [ ] **Step 4.1: Compute population summary**

For each mode ∈ {b3, b4}, each searchlight window, and each (phoneme_pair, ROI) cut: fraction of cells significant (raw p < 0.05; BH-FDR over windows within cell; BH-FDR over cells within window — emit all three), median |Hedges' g|, IQR. Write to `population_summary.csv`.

ROI labels come from `src.data.get_electrode_df(subject)` joined on `(subject, electrode_idx)`.

- [ ] **Step 4.2: Plot population summary**

`population_summary.pdf` — multi-page:
1. Sites × time heatmap of t-stat (signed) per mode, sites sorted by acoustic-peak smin.
2. Per-time fraction-significant curves (raw + FDR), with the causal6 acoustic search range shaded.
3. Hedges' g distributions: histogram + boxplot per mode.
4. Phoneme_pair / ROI breakdown bars.
5. **Decision page**: callout box with JON-41 thresholds (≥40% greenlight / <15% revisit) and the actual numbers.

- [ ] **Step 4.3: Filtered-gallery hook**

Join the per-cell summary back to `outputs/causal46_joined/star_plots/star_plot_keys.csv`. For each mode, emit two filtered PDFs by concatenating per-site PDFs from `outputs/causal46_joined/star_plots/{single_step,matched_n}/per_site/`:
- `{mode}_powered.pdf` — `status='ok'` (cell had min_class ≥ N_cal).
- `{mode}_powered_significant.pdf` — additionally cell's best window has BH-FDR-corrected p < 0.05.

Use PyPDF2 or `PdfPages` to concatenate; do not re-render. Source PDFs are not modified.

`filtered_manifest.csv` records flags + p-values for every cell.

- [ ] **Step 4.4: Commit**

```bash
git add notebooks/causal46_joined/t_tests.py outputs/causal46_joined/t_tests/
git commit -m "population summary + filtered gallery hook (JON-44)"
```

---

### Task 5: End-to-end execution + acceptance

- [ ] **Step 5.1: Execute the three notebooks in order**

```bash
uv run jupytext --execute --to notebook --output - notebooks/causal46_joined/calibration.py > /tmp/calibration.ipynb
uv run jupytext --execute --to notebook --output - notebooks/causal46_joined/t_tests.py > /tmp/t_tests.ipynb
```

(`star_plots.py` was re-executed at the end of Task 1.)

- [ ] **Step 5.2: Acceptance checks**

```bash
test -s outputs/causal46_joined/calibration/N_calibrated.txt
test -s outputs/causal46_joined/calibration/calibration_curve.pdf
test -s outputs/causal46_joined/t_tests/b3_results.parquet
test -s outputs/causal46_joined/t_tests/b4_results.parquet
test -s outputs/causal46_joined/t_tests/population_summary.csv
test -s outputs/causal46_joined/t_tests/population_summary.pdf
ls outputs/causal46_joined/t_tests/star_plots_filtered/
```

- [ ] **Step 5.3: Spot-check 5 random sites**

For 5 (subject, electrode_idx, phoneme_pair, word_end) drawn from `b3_results.parquet` with peak |t| > 2: confirm that the t-trace's argmax window aligns with the visible HGA separation in the corresponding star plot PDF.

---

## Acceptance criteria

1. `calibration_curve.csv` shows monotonically non-decreasing `frac_mean` in N (within seed noise of ±2σ). `N_calibrated.txt` contains a single integer; the curve passes through (or above) `TARGET_FRAC=0.80` at that N.
2. `b3_results.parquet` row count = (B3 cells with `min_class ≥ N_cal`) × (searchlight windows) × `N_SEEDS`. `b4_results.parquet` row count is the analogous product. Both have a `status='ok'` value for all per-window rows.
3. `population_summary.csv` contains rows for both modes, at minimum the overall cuts plus per-phoneme_pair and per-ROI breakdowns.
4. `population_summary.pdf` exists, is multi-page, and contains the decision-page callout with actual numbers against the JON-41 thresholds.
5. `star_plots_filtered/{b3,b4}_powered.pdf` exist and are non-empty (assuming any cells passed the calibration filter); `filtered_manifest.csv` row count matches `star_plot_keys.csv` row count.
6. Re-running `star_plots.py` after the Task 1 refactor produces visually identical galleries.

---

## Out of scope

- Permutation null on the population fraction-significant curve. Mention in the population PDF if cheap, otherwise punt to a separate ticket.
- Anatomy chi-squared on significant survivors (lives in JON-41's "option 2 background track").
- Picking `TARGET_FRAC` beyond 0.80 (expose as a notebook variable; don't tune unless calibration recovers <80% at the highest candidate N).
- Re-doing causal4's `find_site_windows` — we deliberately roll a thin parallel implementation that returns the full searchlight trace instead of just the argmax window.
- Promoting `_within_completion.py` to `src/`. Reconsider only if a fourth caller appears.
- Cross-completion or TRF-residualized variants — the within-completion design IS the acoustic control here.
