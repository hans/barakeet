# causal6 Stage-3 Adaptive Null + ROI-Restricted FDR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Linear:** TODO — create as Group A sibling of JON-42 (parent JON-41). Title suggestion: "causal6 stage-3 null + ROI-restricted FDR for acoustic decoder."

**Goal:** Make BH-FDR feasible for the acoustic decoder by (a) restricting the multiple-testing family to a pre-registered set of speech-perception ROIs, and (b) adding a third permutation stage that boosts K for sites with current corrected-p in the BH rejection neighborhood. Companion fix: the `permutation_idx` collision between stage 1 and stage 2 that currently caps effective K at 9000 instead of 10000.

**Architecture:** All changes live in `src/models/causal6.py`, `src/models/causal6_aggregates.py`, `src/models/causal6_adaptive_null.py`, `notebooks/causal6/acoustic_decoding_null.py`, `notebooks/causal6/significance_aggregate.py`, and `workflows/causal6.Snakefile`. ROI list and stage-3 parameters live in `config.yaml`. Stage 3 runs in-process at the end of the existing null notebook (no new Snakemake rule).

**Tech Stack:** Python, polars, torch (GPU), Snakemake. GPU rules submitted via `submit_job.alt` per the project convention.

---

## Context

- N=2871 acoustic tests pooled across 10 subjects → BH-FDR at α=0.05 demands smallest `p ≤ 1.74e-5`, but K=10k caps achievable p at 1e-4. Result: **0 sites pass q<0.05** in prod despite top sites having AUC 0.78–0.92 (`outputs_prod/acoustic_decoding_peaks/phon_peaks_all.parquet`).
- Empirically the K-floor binds: 0 sites are at q<0.05; 35 are at q<0.10 with most p-values tied at the K-floor.
- Two complementary fixes are needed:
  1. **Reduce N** by restricting the BH-FDR family to ~7 a priori speech-perception ROIs. With STG+MTG+SMG+precentral+IFG, N_ROI ≈ 2007.
  2. **Increase K for the top candidates** via a third permutation stage gated at `p ≤ k × α / N_ROI` with **k=200** (≈ 2× the expected discovery count of ~100, matching causal4's headline 64 acoustic sites with a safety margin).
- A separate bug found during diagnosis: `_fit_batched_cv_permutations` (`causal6.py:560`) assigns `permutation_idx` as list position into the call's `permute_seeds`, not the seed value. Stage 1 uses range(0, 1000) and stage 2 uses range(1000, 10000), but both get perm_idx in [0, K_stage) — collisions at perm_idx [0, 1000). At merge time `null_standardized_peak_test` groups by `(site, perm_idx)` and `.max()`-pools collisions, dropping K1=1000 worth of perms for escalated sites. Must be fixed before stage 3 or seed [10000, …) will collide with stage 2's [0, 9000) the same way.

### Why the gate isn't circular

The gate is per-site (each site's own corrected p), not family-level (BH rejection set). Adding more permutations to a site refines its p-estimate but doesn't change the test statistic. After stage 3, BH is applied once to the final p-values — no iteration on observed BH outcomes.

### ROI list (pre-registered)

```yaml
analysis:
  fdr_rois:
    - superiortemporal      # STG — primary auditory belt
    - middletemporal        # MTG
    - supramarginal         # SMG
    - precentral            # premotor / motor planning
    - parsopercularis       # Broca's, posterior IFG
    - parstriangularis      # Broca's, anterior IFG
    - caudalmiddlefrontal   # dorsolateral PFC, speech-relevant
```

Empirical counts in current 10-subject prod: 669 SR electrodes × 3 phoneme_pairs = **N_ROI = 2007**.

**Excluded by design (with rationale):** postcentral (121 electrodes — somatomotor mouth area; include only if claim is motor-perceptual); lateraloccipital (53 — vision); inferiortemporal (8 — visual word form area, off-topic); rostralmiddlefrontal (6 — distal PFC); white matter / unknown / hippocampus / etc.

If the methodology decision later includes postcentral, N_ROI becomes 2370 and the gate threshold scales accordingly (k=200 still works).

---

## File Structure

- **Modify:** `src/models/causal6.py` — perm_idx collision fix.
- **Modify:** `src/models/causal6_aggregates.py` — add `restrict_to_roi` helper.
- **Modify:** `src/models/causal6_adaptive_null.py` — add `stage3_gate`, log helpers.
- **Modify:** `notebooks/causal6/acoustic_decoding_null.py` — append stage-3 section.
- **Modify:** `notebooks/causal6/significance_aggregate.py` — accept ROI list, restrict before BH.
- **Modify:** `workflows/causal6.Snakefile` — wire stage-3 params + ROI list into `acoustic_decoding_null` and `acoustic_decoding_peaks_aggregate` rules.
- **Modify:** `config.yaml` — add `analysis.fdr_rois`, `causal6.n_permutations_stage3`, `causal6.stage3_k_gate`.
- **Create:** `tests/test_perm_idx_seeds.py` — smoke test for the collision fix.

Stage 3 is implemented for **acoustic decoder only** in this plan. The same pattern generalizes to behavior_with_control, behavior_hga_only, ganong_full, ganong_hga_only — left for follow-up so this plan stays focused. The perm_idx collision fix benefits all five decoders immediately.

---

## Inputs / Outputs

### New config knobs

```yaml
causal6:
  n_permutations_stage3: 90000     # → total K=100k for stage-3 sites
  stage3_k_gate: 200               # 2× expected discovery count
  permutation_seed: 0              # existing — stage 3 uses [permutation_seed + 10000, +100000)
analysis:
  fdr_rois: [ ... ]                # list above
  fdr_alpha: 0.05                  # existing
```

### Pipeline I/O

`acoustic_decoding_null` rule:
- **Existing inputs:** real scores, electrodes CSV (for SR + ROI labels).
- **New inputs:** ROI list (from config; passed as parameter, not file).
- **Existing output:** `null_scores.parquet` (merged), `escalation_log.parquet`.
- **New output column on escalation_log:** `stage3_eligible` (bool), `stage3_refit` (bool).
- **Updated row counts:** stage-3 sites have up to 100k perms per (site, window).

`acoustic_decoding_peaks_aggregate` (consumes per-subject `phon_peaks*.parquet`):
- **New inputs:** electrodes CSVs from all subjects (for ROI labels), config `analysis.fdr_rois`.
- **Behavior:** filter the combined peaks table to ROI sites *before* BH-FDR. Sites outside the ROI family appear in the output with `q_value = NaN`, `significant = False`, and a new `in_fdr_family` boolean column.

---

## Notes on TDD

The perm_idx fix gets a unit test (small, well-scoped). The stage-3 gate logic is exercised by the equivalence check in Task 6.4 (re-run for one subject and compare expected K against escalation_log). Other changes are validated by end-to-end execution on EC243 prod data.

---

## Tasks

### Task 1: Fix the perm_idx collision

**Files:** `src/models/causal6.py`, new test.

- [ ] **Step 1.1: Read the current code** at `_fit_batched_cv_permutations` (`src/models/causal6.py` ~line 560) and confirm:
  - `permute_seeds` is the input parameter listing the seeds.
  - `perm_ids = np.repeat(np.arange(chunk_start, chunk_end, dtype=np.int64), B)` is the buggy line.

- [ ] **Step 1.2: Patch the line**

Replace:
```python
perm_ids = np.repeat(
    np.arange(chunk_start, chunk_end, dtype=np.int64), B
)
```
with:
```python
seeds_chunk = np.asarray(permute_seeds[chunk_start:chunk_end], dtype=np.int64)
perm_ids = np.repeat(seeds_chunk, B)
```

Now `permutation_idx` carries the *seed* (which the calling notebooks already chose to be non-overlapping across stages).

- [ ] **Step 1.3: Add unit test `tests/test_perm_idx_seeds.py`**

```python
"""Verify _fit_batched_cv_permutations writes permute_seeds as permutation_idx."""
import numpy as np
import polars as pl
import torch

from src.models.causal6 import _fit_batched_cv_permutations

def test_perm_idx_equals_seed():
    n_trials, B, d = 40, 2, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_trials, B, d))
    y = (rng.random(n_trials) > 0.5).astype(np.int64)
    problem_meta = pl.DataFrame({"problem_id": [0, 1]})

    seeds = [42, 100, 9999, 100000]   # non-contiguous, non-zero-based
    scores = _fit_batched_cv_permutations(
        X, y, problem_meta,
        permute_seeds=seeds, permutation_chunk_size=2,
        reg_lambda=1.0, n_folds=5, cv_random_state=42,
        device="cpu", dtype=torch.float32, tol=1e-6, max_iter=10,
    )
    got = sorted(scores["permutation_idx"].unique().to_list())
    assert got == sorted(seeds), f"expected perm_idx ∈ {seeds}, got {got}"
```

- [ ] **Step 1.4: Run the test**

```bash
./.venv/bin/python -m pytest tests/test_perm_idx_seeds.py -v
```

Expected: 1 passed.

- [ ] **Step 1.5: Commit**

```bash
git add src/models/causal6.py tests/test_perm_idx_seeds.py
git commit -m "fix permutation_idx collision in _fit_batched_cv_permutations"
```

---

### Task 2: Add ROI + stage-3 config knobs

**Files:** `config.yaml`.

- [ ] **Step 2.1: Inspect current config**

```bash
grep -nE "n_permutations|fdr|escalate" config.yaml
```

Note the existing causal6 + analysis subtree layout to mirror.

- [ ] **Step 2.2: Append new keys**

Under `causal6:`:
```yaml
  n_permutations_stage3: 90000     # stage 3 boost — see docs/superpowers/plans/2026-05-19-...
  stage3_k_gate: 200               # 2× expected discovery rate ~100
```

Under `analysis:`:
```yaml
  fdr_rois:
    - superiortemporal
    - middletemporal
    - supramarginal
    - precentral
    - parsopercularis
    - parstriangularis
    - caudalmiddlefrontal
```

Mirror in `config.smoke.yaml` with smaller K3 (e.g. `n_permutations_stage3: 100`) so smoke tests stay fast.

- [ ] **Step 2.3: Commit**

```bash
git add config.yaml config.smoke.yaml
git commit -m "add stage3 + ROI knobs to config"
```

---

### Task 3: ROI restriction helper

**Files:** `src/models/causal6_aggregates.py`.

The helper takes the combined peaks (or any per-site frame), joins with electrode metadata, and filters to ROI sites. Used by both the stage-3 gate and the significance_aggregate notebook.

- [ ] **Step 3.1: Add helper**

```python
def restrict_to_rois(
    df: pl.DataFrame,
    electrode_dfs: list[pl.DataFrame],
    rois: Sequence[str],
    *,
    site_keys: Sequence[str] = ("subject", "electrode_idx"),
) -> tuple[pl.DataFrame, int]:
    """Filter df to rows whose (subject, electrode_idx) lives in one of `rois`.

    Args:
        df: long-format with `site_keys` columns.
        electrode_dfs: per-subject electrode CSVs (cast to polars). Must have
            columns `subject, electrode_idx, roi`.
        rois: list of FreeSurfer aparc labels.

    Returns:
        (filtered_df, N_ROI). N_ROI is the count of rows in filtered_df —
        used as the family size for BH-FDR. Callers may multiply by
        phoneme_pair count if `df` is at electrode granularity.
    """
    elec = pl.concat(electrode_dfs).select(["subject", "electrode_idx", "roi"]).unique()
    roi_keys = elec.filter(pl.col("roi").is_in(list(rois)))
    filtered = df.join(roi_keys, on=list(site_keys), how="semi")
    return filtered, filtered.height
```

- [ ] **Step 3.2: Add a small unit test**

```python
# tests/test_restrict_to_rois.py
import polars as pl
from src.models.causal6_aggregates import restrict_to_rois


def test_restrict_basic():
    df = pl.DataFrame({
        "subject": ["A", "A", "B", "B"],
        "electrode_idx": [1, 2, 1, 2],
        "phoneme_pair": ["dn"] * 4,
        "p_value": [0.01, 0.02, 0.03, 0.04],
    })
    elec = [pl.DataFrame({
        "subject": ["A", "A", "B", "B"],
        "electrode_idx": [1, 2, 1, 2],
        "roi": ["superiortemporal", "lateraloccipital", "precentral", "fusiform"],
    })]
    out, n = restrict_to_rois(df, elec, ["superiortemporal", "precentral"])
    assert n == 2
    assert sorted(zip(out["subject"], out["electrode_idx"])) == [("A", 1), ("B", 1)]
```

```bash
./.venv/bin/python -m pytest tests/test_restrict_to_rois.py -v
```

- [ ] **Step 3.3: Commit**

```bash
git add src/models/causal6_aggregates.py tests/test_restrict_to_rois.py
git commit -m "add restrict_to_rois helper"
```

---

### Task 4: Stage-3 gate

**Files:** `src/models/causal6_adaptive_null.py`.

`stage1_gate` already implements the gate machinery for a *site-level* p-value threshold. Stage 3 reuses it with a different `p_max`. The thin wrapper here computes that `p_max` from `k_gate`, `alpha`, and `N_ROI`, and adds logging.

- [ ] **Step 4.1: Add stage3_gate**

```python
def stage3_gate(
    real_agg: pl.DataFrame,
    null_agg: pl.DataFrame,
    *,
    site_keys: Sequence[str],
    flavors: Sequence[FlavorSpec],
    k_gate: int,
    n_roi: int,
    alpha: float,
    window_keys: Sequence[str] = ("smin", "smax"),
    perm_key: str = "permutation_idx",
) -> tuple[set[tuple], pl.DataFrame]:
    """Gate the K-floored ROI sites for a stage-3 permutation refit.

    Threshold = k_gate * alpha / n_roi. With k_gate=200, alpha=0.05,
    n_roi=2007: threshold ≈ 5e-3. Catches sites whose current p, if true,
    could be FDR-significant at any of the top k_gate BH ranks.

    Returns (refit_keys, gate_log) in the same shape as stage1_gate.
    Caller must pre-restrict real_agg / null_agg to ROI sites — this
    function does not do the ROI filter itself.
    """
    p_max = k_gate * alpha / n_roi
    return stage1_gate(
        real_agg, null_agg,
        site_keys=site_keys, flavors=flavors,
        p_max=p_max, window_keys=window_keys, perm_key=perm_key,
    )
```

- [ ] **Step 4.2: Add `log_stage3_gate`** (parallel to existing `log_stage1_gate`)

```python
def log_stage3_gate(
    subject: str,
    *,
    n_permutations_total_pre_stage3: int,
    n_roi: int,
    k_gate: int,
    alpha: float,
    gate_log: pl.DataFrame,
    n_refit: int,
    print_top_k: int = 50,
) -> None:
    threshold = k_gate * alpha / n_roi
    print(
        f"[CAUSAL6-GATE/{subject}] stage3 K_pre={n_permutations_total_pre_stage3} "
        f"k_gate={k_gate} N_ROI={n_roi} threshold={threshold:.2e}: "
        f"{n_refit}/{gate_log.height} sites flagged",
        flush=True,
    )
    if n_refit > 0:
        print(
            gate_log.filter(pl.col("escalated"))
            .sort("min_corrected_p_global")
            .head(print_top_k)
            .to_pandas().to_string(index=False),
            flush=True,
        )
```

- [ ] **Step 4.3: Commit**

```bash
git add src/models/causal6_adaptive_null.py
git commit -m "add stage3_gate and log_stage3_gate"
```

---

### Task 5: Wire stage 3 into the acoustic null notebook

**Files:** `notebooks/causal6/acoustic_decoding_null.py`.

- [ ] **Step 5.1: Add new parameter cell**

In the `# %% tags=["parameters"]` cell, append:
```python
# Stage-3 boost — refit sites in the BH rejection neighborhood at K3 perms.
n_permutations_stage3 = 90000
stage3_k_gate = 200
fdr_alpha = 0.05
fdr_rois = []                       # populated from config.analysis.fdr_rois
electrode_dfs_paths = []            # all subjects' find_speech_responsive CSVs
```

- [ ] **Step 5.2: Append stage-3 section after the existing stage-1+stage-2 merge**

```python
# %% [markdown]
# ## Stage 3 — boost K for sites in the BH rejection neighborhood
#
# Restricts to ROI sites, computes corrected p at K1+K2 perms, gates at
# `p ≤ k_gate * alpha / N_ROI` (k_gate=200, threshold ≈ 5e-3 at N_ROI≈2007).
# Refits gated sites with K3 additional perms using seeds disjoint from
# stages 1 and 2.

# %%
import pandas as pd
from src.models.causal6_adaptive_null import stage3_gate, log_stage3_gate
from src.models.causal6_aggregates import restrict_to_rois

# Compute N_ROI = (count of ROI-eligible electrode rows) × n_phoneme_pairs.
# Use the same electrode CSVs the pipeline uses for SR filtering.
if fdr_rois and electrode_dfs_paths and n_permutations_stage3 > 0:
    electrode_dfs = [
        pl.from_pandas(pd.read_csv(p)) for p in electrode_dfs_paths
    ]
    elec_pool = pl.concat([
        e.filter(pl.col("speech_responsive"))
         .select(["subject", "electrode_idx", "roi"])
        for e in electrode_dfs
    ])
    n_roi_electrodes = elec_pool.filter(pl.col("roi").is_in(fdr_rois)).height
    n_phoneme_pairs = epochs.metadata.phoneme_pair.dropna().nunique()
    n_roi = n_roi_electrodes * n_phoneme_pairs

    real_agg_roi, _ = restrict_to_rois(real_agg, electrode_dfs, fdr_rois)
    null_agg_roi, _ = restrict_to_rois(null_scores, electrode_dfs, fdr_rois)

    refit_keys, stage3_log = stage3_gate(
        real_agg_roi, null_agg_roi,
        site_keys=SITE_KEYS_ACOUSTIC,
        flavors=FLAVORS_ACOUSTIC,
        k_gate=stage3_k_gate,
        n_roi=n_roi,
        alpha=fdr_alpha,
    )
    log_stage3_gate(
        subject,
        n_permutations_total_pre_stage3=n_permutations_stage1 + n_permutations_stage2,
        n_roi=n_roi,
        k_gate=stage3_k_gate,
        alpha=fdr_alpha,
        gate_log=stage3_log,
        n_refit=len(refit_keys),
    )

    if refit_keys:
        stage3_seeds = list(range(
            permutation_seed + n_permutations_stage1 + n_permutations_stage2,
            permutation_seed + n_permutations_stage1 + n_permutations_stage2
                + n_permutations_stage3,
        ))
        refit_electrode_idxs = sorted({
            k[SITE_KEYS_ACOUSTIC.index("electrode_idx")] for k in refit_keys
        })
        with stage2_spill_dir(outdir, name="_stage3_spill") as spill_dir:
            run_acoustic_searchlight_permutations(
                epochs, subject=subject,
                electrode_idxs=refit_electrode_idxs,
                windows=windows,
                reg_lambda=reg_lambda,
                permute_seeds=stage3_seeds,
                permutation_chunk_size=permutation_chunk_size,
                n_folds=n_folds, cv_random_state=cv_random_state,
                device=device, dtype=torch.float32,
                tol=tol, max_iter=max_iter,
                spill_dir=spill_dir,
            )
            null_stage3_raw = filter_null_to_borderline(
                pl.scan_parquet(spill_dir / "*.parquet"),
                refit_keys,
                site_keys=SITE_KEYS_ACOUSTIC,
            ).collect()

        null_stage3 = preagg_acoustic_null(null_stage3_raw, real_scores)
        null_scores = pl.concat([null_scores, null_stage3])
        del null_stage3_raw

    # Add stage3 columns to escalation_log
    stage3_flag = pl.DataFrame({
        **{sk: [k[i] for k in refit_keys]
           for i, sk in enumerate(SITE_KEYS_ACOUSTIC)},
        "stage3_refit": [True] * len(refit_keys),
    }) if refit_keys else pl.DataFrame(schema={
        sk: pl.Utf8 if sk == "phoneme_pair" or sk == "subject" else pl.Int64
        for sk in SITE_KEYS_ACOUSTIC
    } | {"stage3_refit": pl.Boolean})
    gate_log = gate_log.join(
        stage3_flag, on=SITE_KEYS_ACOUSTIC, how="left"
    ).with_columns(pl.col("stage3_refit").fill_null(False))
else:
    print(f"[{subject}] stage3 skipped (rois={bool(fdr_rois)}, K3={n_permutations_stage3})")
```

- [ ] **Step 5.3: Verify the notebook parses**

```bash
./.venv/bin/jupytext --to notebook --output /dev/null notebooks/causal6/acoustic_decoding_null.py
```

- [ ] **Step 5.4: Commit**

```bash
git add notebooks/causal6/acoustic_decoding_null.py
git commit -m "add stage3 boost to acoustic null notebook"
```

---

### Task 6: ROI-restricted FDR in significance_aggregate

**Files:** `notebooks/causal6/significance_aggregate.py`.

- [ ] **Step 6.1: Add ROI parameters**

In the parameters cell:
```python
fdr_rois = []
electrode_dfs_paths = []
```

- [ ] **Step 6.2: Restrict before BH**

Between the concat and `multipletests` call:

```python
# Apply ROI restriction if configured. Sites outside the family get q=NaN.
if fdr_rois and electrode_dfs_paths:
    import pandas as pd
    from src.models.causal6_aggregates import restrict_to_rois

    electrode_dfs = [pl.from_pandas(pd.read_csv(p)) for p in electrode_dfs_paths]
    combined_pl = pl.from_pandas(combined)
    in_family, n_roi = restrict_to_rois(
        combined_pl, electrode_dfs, fdr_rois,
        site_keys=("subject", "electrode_idx"),
    )
    in_family_keys = set(zip(in_family["subject"].to_list(),
                             in_family["electrode_idx"].to_list()))
    combined["in_fdr_family"] = [
        (s, e) in in_family_keys
        for s, e in zip(combined["subject"], combined["electrode_idx"])
    ]
    print(f"ROI restriction: {combined['in_fdr_family'].sum()} / {len(combined)} "
          f"rows in FDR family across {n_roi} sites")
else:
    combined["in_fdr_family"] = True

# BH on the in-family rows only.
mask = combined["in_fdr_family"].values
q_values = np.full(len(combined), np.nan)
_, q_in, _, _ = multipletests(
    combined.loc[mask, "p_value"].values, alpha=fdr_alpha, method="fdr_bh"
)
q_values[mask] = q_in
combined["q_value"] = q_values
combined["significant"] = (combined["q_value"] < fdr_alpha).fillna(False)
```

Add `import numpy as np` at the top of the notebook if not already present.

- [ ] **Step 6.3: Commit**

```bash
git add notebooks/causal6/significance_aggregate.py
git commit -m "apply ROI restriction in significance_aggregate"
```

---

### Task 7: Snakefile wiring

**Files:** `workflows/causal6.Snakefile`.

- [ ] **Step 7.1: Add stage-3 params to the acoustic_decoding_null rule**

In the `acoustic_decoding_null` rule's `run_notebook(... parameters=dict(...))` call, append:
```python
n_permutations_stage3=config["causal6"]["n_permutations_stage3"],
stage3_k_gate=config["causal6"]["stage3_k_gate"],
fdr_alpha=config["analysis"]["fdr_alpha"],
fdr_rois=config["analysis"]["fdr_rois"],
electrode_dfs_paths=expand(
    "outputs/causal6/find_speech_responsive/{subject}_results.csv",
    subject=config["data"]["subjects"],
),
```

And add the corresponding entries to the `input:` block of the rule so Snakemake sees the dependency on every subject's electrode CSV.

- [ ] **Step 7.2: Same params for `acoustic_decoding_peaks_aggregate` rule**

```python
fdr_rois=config["analysis"]["fdr_rois"],
electrode_dfs_paths=expand(
    "outputs/causal6/find_speech_responsive/{subject}_results.csv",
    subject=config["data"]["subjects"],
),
```

- [ ] **Step 7.3: Smoke test the workflow parses**

```bash
./.venv/bin/snakemake --configfile config.smoke.yaml --list-rules 2>&1 | head -20
```

- [ ] **Step 7.4: Commit**

```bash
git add workflows/causal6.Snakefile
git commit -m "wire stage3 + ROI params into acoustic null + aggregate rules"
```

---

### Task 8: Validate on one subject's prod data

**No files modified.** Run end-to-end for EC243 (one of the prod subjects with mid-range escalation count).

- [ ] **Step 8.1: Re-run acoustic null for EC243 with stage 3**

Trigger via SGE submit; use the `submit_job.alt` form per `feedback_submit_job_alt`:

```bash
submit_job.alt -- snakemake --configfile config.yaml \
    outputs/causal6/acoustic_decoding_null/EC243/null_scores.parquet \
    --forceall
```

- [ ] **Step 8.2: Inspect resulting null_scores**

```bash
./.venv/bin/python -c "
import polars as pl
ns = pl.read_parquet('outputs/causal6/acoustic_decoding_null/EC243/null_scores.parquet')
print('total null rows:', ns.height)
print('unique permutation_idx:', ns['permutation_idx'].n_unique())
print('perm_idx range:', ns['permutation_idx'].min(), ns['permutation_idx'].max())
print()
print('per-site unique perm_idx count:')
print(ns.group_by(['subject','electrode_idx','phoneme_pair'])
        .agg(pl.col('permutation_idx').n_unique().alias('K'))
        .group_by('K').len().sort('K'))
"
```

Expected: three K bands — K=1000 (non-escalated), K=10000 (escalated-not-stage3), K=100000 (stage3-refit). Critically, **no K=9000 band** — that would indicate the perm_idx collision fix didn't take.

- [ ] **Step 8.3: Re-run aggregate and inspect FDR pass count**

```bash
snakemake --configfile config.yaml outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet --forceall
./.venv/bin/python -c "
import polars as pl
df = pl.read_parquet('outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet')
print('rows total:', df.height)
print('rows in FDR family:', df['in_fdr_family'].sum())
print('q < 0.05:', (df['q_value'] < 0.05).sum())
print('q < 0.10:', (df['q_value'] < 0.10).sum())
print('min q in family:', df.filter(pl.col('in_fdr_family'))['q_value'].min())
print()
print('top 20 by q_value (in family):')
print(df.filter(pl.col('in_fdr_family')).sort('q_value').head(20)
        .select(['subject','electrode_idx','phoneme_pair','test_roc_auc','p_value','q_value']))
"
```

Expected: q_min substantially lower than 0.0686 (current pre-fix); ≥10 sites at q<0.05.

- [ ] **Step 8.4: Document the run**

Append a one-paragraph note to JON-42 (or its sibling) summarizing: how many sites got stage-3 refit, q_min before/after, number of FDR-significant sites. This is the empirical validation that the methodology change worked.

---

## Acceptance criteria

1. `tests/test_perm_idx_seeds.py` passes — permutation_idx carries seed values.
2. Smoke run produces a `null_scores.parquet` with NO unique `permutation_idx` count of 9000 (proves the collision is gone).
3. For at least one subject, stage 3 fires on ≥ 1 site and produces a null_scores parquet with ≥ 50k unique perm_idx for those sites.
4. `phon_peaks_all.parquet` has `in_fdr_family` and `q_value` columns; q-values are NaN for non-family sites.
5. After end-to-end re-run on all 10 subjects: q_min substantially below 0.05 (target: ≥ 20 sites at q<0.05; concrete reach goal is to recover causal4's headline ~64 acoustic sites).

## Out of scope

- Generalizing stage 3 to behavior_with_control, behavior_hga_only, ganong_full, ganong_hga_only. The perm_idx fix benefits them immediately, but the stage-3 mechanism + ROI wiring is per-decoder work. File as follow-up after acoustic validates.
- The with-control baseline-drop bug from the audit (a separate fix branch is already in flight).
- Westfall-Young joint max-stat across electrodes (more powerful but bigger compute; future work).
- Anatomy-driven ROI definitions beyond the FreeSurfer aparc labels listed in config.yaml.
- Per-subject FDR as a primary report (would be an additional CSV alongside the pooled BH).
