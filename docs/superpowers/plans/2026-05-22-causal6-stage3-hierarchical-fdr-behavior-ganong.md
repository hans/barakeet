# causal6 Stage-3 + Hierarchical FDR — Behavior & Ganong Decoders

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Linear:** TODO — create as Group A sibling of JON-50. Title suggestion: "causal6 stage-3 + hierarchical FDR for behavior + ganong decoders."

**Goal:** Bring the four behavior/ganong null pipelines up to parity with the acoustic null pipeline:
1. Stage-3 K-boost for sites in the BH rejection neighborhood.
2. ROI-restricted hierarchical Simes + BH + Holm FDR in the aggregate step.

**Factor-first**: extract a shared `stage3_boost(...)` orchestrator into `src/models/causal6_adaptive_null.py`, then refactor `acoustic_decoding_null.py` to use it (no-op-equivalent), then apply to the four behavior/ganong null notebooks. Each notebook becomes a ~20-line call instead of a copy-pasted ~100-line block.

**Architecture:** All notebook changes live in `notebooks/causal6/*_null.py`. The orchestrator lives in `src/models/causal6_adaptive_null.py`. No changes to `significance_aggregate.py` — its hierarchical-FDR branch already triggers on `fdr_rois + electrode_dfs_paths`; the existing `groupby(["subject", "electrode_idx"])` Simes collapse handles behavior (over phoneme_pair × word_end) and ganong (over phoneme_pair) without modification.

**Tech Stack:** Python, polars, torch (GPU), Snakemake.

---

## Context

The acoustic null pipeline has two features the behavior/ganong nulls don't have yet:

### Stage-3 K-boost (currently acoustic-only)
After stage 1 (K1=1000) and stage 2 (K2=9000), the acoustic null notebook re-aggregates the combined null, restricts to ROI sites, runs a third gate at threshold `k_gate * alpha / n_roi ≈ 5e-3`, and refits the flagged sites with K3=90000 disjoint-seed permutations. Implemented in `notebooks/causal6/acoustic_decoding_null.py:230-333` (~100 lines, mostly orchestration).

Helpers already in `src/models/causal6_adaptive_null.py` (decoder-agnostic):
- `stage3_gate(real_agg, null_agg, site_keys=..., flavors=..., k_gate=..., n_roi=..., alpha=...)` — wraps `stage1_gate` with `p_max = k_gate * alpha / n_roi`.
- `log_stage3_gate(subject, ..., gate_log, n_refit)` — emits CAUSAL6-GATE log line.
- `stage2_spill_dir(outdir, name="_stage3_spill")` — reusable spill context manager.
- `filter_null_to_borderline(...)` — already supports `baseline_site_keys` for with-control decoders (used by behavior_with_control's stage 2).

Helpers in `src/models/causal6_aggregates.py`:
- `restrict_to_rois(df, electrode_dfs, rois, site_keys=("subject", "electrode_idx"))`.

### Hierarchical Simes + BH + Holm FDR (currently acoustic-aggregate-only)
`notebooks/causal6/significance_aggregate.py:81-130` switches from flat BH to hierarchical FDR when `fdr_rois` and `electrode_dfs_paths` are both passed. The Simes step groups by `(subject, electrode_idx)`, collapsing K per-electrode p-values into one electrode-level p; BH runs across electrodes; Holm runs within significant electrodes.

For each decoder, "K per-electrode p-values" comes from the per-electrode site_keys structure:
- Acoustic: K=3 (phoneme_pair).
- Behavior_with_control / behavior_hga_only: K=6 (phoneme_pair × word_end).
- Ganong_with_control / ganong_hga_only: K=3 (phoneme_pair). word_end is not in site_keys — trials pool across completions.

The existing groupby is already `["subject", "electrode_idx"]`, so all four behavior/ganong decoders Just Work once `fdr_rois + electrode_dfs_paths` flow through the aggregate rules.

### ROI list (re-used from acoustic)

```yaml
analysis:
  fdr_rois:
    - superiortemporal
    - middletemporal
    - supramarginal
    - precentral
    - parsopercularis
    - parstriangularis
    - caudalmiddlefrontal
```

N_ROI ≈ 2007 electrodes across 10 subjects. Same gate threshold `k_gate * alpha / n_roi ≈ 5e-3` for all decoders, since the BH family is electrode-level after Simes collapse.

---

## File Structure

- **Modify:** `src/models/causal6_adaptive_null.py` — add `stage3_boost(...)` orchestrator.
- **Modify:** `notebooks/causal6/acoustic_decoding_null.py` — refactor stage-3 block to call the new orchestrator (no behavioral change; bit-identical output is the acceptance test).
- **Modify:** `notebooks/causal6/behavior_decoding_single_electrode_null.py` — add stage-3 block.
- **Modify:** `notebooks/causal6/behavior_decoding_single_electrode_hga_only_null.py` — add stage-3 block.
- **Modify:** `notebooks/causal6/ganong_decoding_null.py` — add stage-3 block.
- **Modify:** `notebooks/causal6/ganong_decoding_hga_only_null.py` — add stage-3 block.
- **Modify:** `workflows/causal6.Snakefile` — wire `fdr_rois + electrode_dfs_paths + stage3` params into 4 null rules and 6 aggregate rules.
- **Create:** `tests/test_stage3_boost.py` — smoke test the orchestrator's no-op equivalence on acoustic.

---

## Design: `stage3_boost` orchestrator

API (added to `src/models/causal6_adaptive_null.py`):

```python
from collections.abc import Callable, Sequence
from pathlib import Path
import polars as pl

def stage3_boost(
    *,
    subject: str,
    outdir: Path,
    # real & null state from stage 1+2
    real_scores: pl.DataFrame,                   # raw fold-level scores
    real_agg: pl.DataFrame,                      # pre-aggregated reals (reused from stage 1)
    null_scores: pl.DataFrame,                   # preagg null (post stage 1+2)
    gate_log: pl.DataFrame,                      # from stage1_gate
    # decoder seam
    site_keys: Sequence[str],
    flavors: Sequence[FlavorSpec],
    aggregate_fn: Callable[
        [pl.DataFrame, pl.DataFrame],
        tuple[pl.DataFrame, pl.DataFrame],
    ],                                            # bound partial: (real_scores, null) -> (real_agg, null_agg)
    preagg_fn: Callable[
        [pl.DataFrame, pl.DataFrame], pl.DataFrame,
    ],                                            # bound partial: (raw_null, real_scores) -> preagg
    run_permutations_fn: Callable[..., pl.DataFrame],
        # bound partial with epochs/windows/reg_lambda[/reg_lambda_baseline]/etc.
        # Caller passes only: electrode_idxs, permute_seeds, spill_dir.
    baseline_site_keys: Sequence[str] | None = None,
    # ROI + stage3 knobs
    electrode_dfs: list[pl.DataFrame],
    fdr_rois: Sequence[str],
    k_gate: int,
    fdr_alpha: float,
    permutation_seeds: list[int],                # caller computes disjoint seed range
    n_permutations_pre: int,                     # K1+K2, for log only
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stage-3 K-boost: refit ROI sites in the BH rejection neighborhood.

    Steps:
      1. Re-aggregate (real_scores, null_scores) via aggregate_fn so the
         gate sees the K1+K2 combined null in fold_mean / t_stat schema
         (not the preagg fold_mean_diff schema).
      2. restrict_to_rois on both real_agg and null_agg → ROI-only frames.
         n_roi = number of unique (subject, electrode_idx) tuples in the
         ROI electrode pool (matches BH family size after Simes collapse).
      3. stage3_gate → refit_keys + stage3_log.
      4. log_stage3_gate.
      5. If refit_keys: run_permutations_fn with permutation_seeds and a
         _stage3_spill scratch dir, lazily filter_null_to_borderline by
         refit_keys (with baseline_site_keys if with-control), preagg_fn,
         concat onto null_scores.
      6. Add stage3_refit boolean column to gate_log (true for keys in
         refit_keys, false otherwise).

    Returns (null_scores, gate_log) updated in place. Caller is responsible
    for writing final n_permutations column based on stage1/2/3 flags.
    """
```

Caller-side glue (template; same shape for all five null notebooks):

```python
from functools import partial

if fdr_rois and electrode_dfs_paths and n_permutations_stage3 > 0:
    electrode_dfs = [pl.from_pandas(pd.read_csv(p)) for p in electrode_dfs_paths]
    stage3_seeds = list(range(
        permutation_seed + n_permutations_stage1 + n_permutations_stage2,
        permutation_seed + n_permutations_stage1 + n_permutations_stage2
            + n_permutations_stage3,
    ))
    null_scores, gate_log = stage3_boost(
        subject=subject, outdir=outdir,
        real_scores=real_for_target,             # acoustic: target-filtered
        real_agg=real_agg, null_scores=null_scores, gate_log=gate_log,
        site_keys=SITE_KEYS_ACOUSTIC, flavors=FLAVORS_ACOUSTIC,
        aggregate_fn=partial(
            aggregate_acoustic,
            target=target,
            peak_search_smin=peak_search_smin,
            peak_search_smax=peak_search_smax,
        ),
        preagg_fn=partial(preagg_acoustic_null),
        run_permutations_fn=partial(
            run_acoustic_searchlight_permutations,
            epochs, subject=subject, windows=windows,
            reg_lambda=reg_lambda, target=target,
            permutation_chunk_size=permutation_chunk_size,
            n_folds=n_folds, cv_random_state=cv_random_state,
            device=device, dtype=torch.float32,
            tol=tol, max_iter=max_iter,
        ),
        baseline_site_keys=None,                 # acoustic: no paired baseline
        electrode_dfs=electrode_dfs, fdr_rois=fdr_rois,
        k_gate=stage3_k_gate, fdr_alpha=fdr_alpha,
        permutation_seeds=stage3_seeds,
        n_permutations_pre=n_permutations_stage1 + n_permutations_stage2,
    )
else:
    gate_log = gate_log.with_columns(pl.lit(False).alias("stage3_refit"))
```

Per-decoder seams in the partial:

| Decoder | `aggregate_fn` kwargs | `preagg_fn` | `run_permutations_fn` extras | `baseline_site_keys` |
|---|---|---|---|---|
| acoustic | `target, peak_search_smin, peak_search_smax` | `preagg_acoustic_null` | `target=target` | `None` |
| behavior_with_control | `epoch_tmin, epoch_sfreq, behav_peak_post_offset_s, peak_search_smin, peak_search_smax` | `preagg_behavior_with_control_null` | `reg_lambda_baseline=...` | `["subject", "phoneme_pair", "word_end"]` |
| behavior_hga_only | (same as with_control, minus baseline) | `preagg_behavior_hga_only_null` | (no baseline kwarg) | `None` |
| ganong_with_control | `epoch_tmin, epoch_sfreq, peak_search_smax` | `preagg_ganong_with_control_null` | `reg_lambda_baseline=...` | `["subject", "phoneme_pair"]` |
| ganong_hga_only | `epoch_tmin, epoch_sfreq, peak_search_smax` | `preagg_ganong_hga_only_null` | (no baseline kwarg) | `None` |

---

## Tasks

### A. Factor `stage3_boost` orchestrator
1. Add `stage3_boost` to `src/models/causal6_adaptive_null.py` with the API above. Internals mirror `acoustic_decoding_null.py:239-326` line-for-line, just routed through `aggregate_fn`/`preagg_fn`/`run_permutations_fn` partials.
2. Add `tests/test_stage3_boost.py`. Smoke test: construct tiny synthetic real_scores + preagg null with a known site that should be flagged at a low `k_gate * alpha / n_roi`, verify `refit_keys` matches, verify `gate_log["stage3_refit"]` column is added. No GPU.

### B. Refactor `acoustic_decoding_null.py` to call `stage3_boost`
1. Replace lines 230-333 with the caller-side glue (template above) using the `acoustic` row from the seam table.
2. Acceptance: run on EC282 before + after, verify bit-identical `null_scores.parquet` and `escalation_log.parquet` (modulo column order). Use `polars.DataFrame.equals(...)` or hash-compare parquet bytes after canonical sort.

### C-F. Add stage-3 block to four behavior/ganong null notebooks
For each of `behavior_decoding_single_electrode_null.py`, `behavior_decoding_single_electrode_hga_only_null.py`, `ganong_decoding_null.py`, `ganong_decoding_hga_only_null.py`:

1. Add notebook parameters: `n_permutations_stage3`, `stage3_k_gate`, `fdr_alpha`, `fdr_rois=[]`, `electrode_dfs_paths=[]`.
2. Import: `stage3_boost`, `restrict_to_rois`, the decoder's `aggregate_*`, `preagg_*`, `run_*_permutations` (already imported, just confirm).
3. After the stage-2 `else:` block, add the caller-side glue with the decoder's row from the seam table.
4. Update the final `gate_log.with_columns(... .alias("n_permutations"))` to encode three states (stage1, stage1+2, stage1+2+3) like `acoustic_decoding_null.py:339-346`.

### G. Snakefile null rules — wire stage-3 + ROI params
For each of `behavior_decoding_single_electrode_null`, `behavior_decoding_single_electrode_hga_only_null`, `ganong_decoding_null`, `ganong_decoding_hga_only_null`:

1. Add `all_electrode_dfs = expand("outputs/causal6/find_speech_responsive/{subject}_results.csv", subject=config["data"]["subjects"])` to `input:`.
2. Add to `parameters=dict(...)`:
   ```python
   n_permutations_stage3=C6["n_permutations_stage3"],
   stage3_k_gate=C6["stage3_k_gate"],
   fdr_alpha=config["analysis"]["fdr_alpha"],
   fdr_rois=config["analysis"]["fdr_rois"],
   electrode_dfs_paths=list(input.all_electrode_dfs),
   ```

### H. Snakefile aggregate rules — wire ROI params
For each of `behavior_decoding_single_electrode_summarize_aggregate`, `behavior_decoding_single_electrode_hga_only_summarize_aggregate`, `behavior_decoding_single_electrode_summarize_aggregate_tstat_maxstat`, `behavior_decoding_single_electrode_summarize_aggregate_foldmean_tfce`, `behavior_decoding_single_electrode_summarize_aggregate_tstat_tfce`, `behavior_decoding_single_electrode_hga_only_summarize_aggregate_tstat_maxstat`, `behavior_decoding_single_electrode_hga_only_summarize_aggregate_foldmean_tfce`, `behavior_decoding_single_electrode_hga_only_summarize_aggregate_tstat_tfce`, `ganong_decoding_summarize_aggregate`, `ganong_decoding_hga_only_summarize_aggregate`:

1. Add `all_electrode_dfs = expand(...)` to `input:` (as in task G).
2. Add to `parameters=dict(...)`:
   ```python
   fdr_rois=config["analysis"]["fdr_rois"],
   electrode_dfs_paths=list(input.all_electrode_dfs),
   ```

Note: that's **10 aggregate rules** total (2 v1 + 6 t-stat/TFCE flavors for behavior + 2 for ganong). Some aggregate rules currently emit a single output parquet; the hierarchical-FDR branch adds two new columns (`electrode_q_value`, `electrode_significant`) on top of the existing `q_value`, `significant`. Downstream consumers (`view_provisional_results.py`, `prepare_neurometrics.py`, etc.) should be audited for explicit references to `q_value`/`significant` and updated if they need the electrode-level columns — but they don't break if untouched.

### I. Sanity checks before full re-run
1. Run task A's test.
2. Run task B's bit-identical check on one subject (EC282 or similar fast subject).
3. Run the behavior_with_control null on EC282 with K1=100, K2=100, K3=100 (smoke values) to verify the orchestrator wiring end-to-end without burning GPU hours.
4. Run the behavior_with_control aggregate to confirm the hierarchical branch fires and produces `electrode_q_value` / `electrode_significant` columns.

### J. Re-run all four behavior/ganong nulls + aggregates at production K
After tasks A-I pass:
1. Wipe `outputs/causal6/{behavior_decoding_single_electrode,behavior_decoding_single_electrode_hga_only,ganong_decoding,ganong_decoding_hga_only}_null/` for affected subjects.
2. Re-run via Snakemake. Expect K_stage3 = 90000 on top of K_stage1+2 = 10000 for ROI sites in the rejection neighborhood — same runtime budget as the acoustic stage-3 (since behavior is paired full+baseline, ~2× the GPU work per permutation — budget for 1.5-2× the acoustic stage-3 wall time).
3. Re-run the 10 aggregate rules.

---

## Acceptance Criteria

- `stage3_boost` test passes.
- `acoustic_decoding_null` on one subject produces a `null_scores.parquet` bit-identical (canonical sort) to the pre-refactor output. Same for `escalation_log.parquet`.
- All four behavior/ganong null rules produce non-empty `null_scores.parquet` + `escalation_log.parquet` containing a `stage3_refit` boolean column.
- All 10 behavior/ganong aggregate rules emit `*_all.parquet` with both `electrode_q_value`/`electrode_significant` and `q_value`/`significant` columns populated.
- Spot-check: at least one site that was `significant=True` under the old flat-BH behavior_with_control aggregate is still `significant=True` under hierarchical FDR (or, if not, the reason is the smaller per-electrode family — check Holm thresholds).

---

## Followups / Out of scope

- Auditing `view_provisional_results.py`, `prepare_neurometrics.py`, and any other consumer of behavior aggregate parquets for awareness of the new `electrode_q_value`/`electrode_significant` columns. The columns are additive; consumers don't break if they ignore them, but star plots / pruning steps may want to start using `electrode_significant` as the primary gate.
- Choosing whether `peak_summary_foldmean_tfce` / `peak_summary_tstat_tfce` should use the same Simes family. They will under this plan because they pass through the same `significance_aggregate.py`. If we later decide TFCE flavors should be ranked under a separate family, that's a follow-up change in the aggregate notebook (probably gated by a new `fdr_family_name` param).
- Updating `docs/superpowers/plans/2026-05-19-causal6-stage3-null-roi-fdr.md` with a back-reference to this plan in its "out of scope" section.
- Considering whether ganong's smaller per-electrode family (K=3 vs behavior's K=6) warrants a different `k_gate` — probably not, since the threshold is `k_gate * alpha / n_roi` and `n_roi` is electrode-level after Simes regardless.
