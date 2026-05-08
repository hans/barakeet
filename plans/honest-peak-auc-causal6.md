# Plan: Honest (CV-debiased) peak AUC for causal6 decoders

## Status

**Possible future work.** Not started. This plan records a methodology
discussion from 2026-05-08 so it can be picked up later.

## Context / motivation

The causal6 peak-finding pipeline (acoustic, behavior, ganong) reports
the fold-mean test ROC-AUC at the selected peak window as the
per-site effect size. This number is **biased upward** because the same
fold-AUCs that select the peak window also define the value at that
window — a classic selection-on-the-statistic problem.

There are two distinct selection-bias problems; only one is currently
solved:

1. **Type I error from selecting across windows** — *already controlled*
   by `null_standardized_peak_test` in
   [src/models/significance.py](../src/models/significance.py). The null
   permutations are put through the same argmax-over-windows, so the
   corrected p-value is calibrated against the same selection bias that
   inflates the point estimate.

2. **Inflated point estimate at the chosen peak** — *not controlled.*
   The reported `real_statistic` (fold_mean AUC at the argmin-p window)
   is biased upward by selection. Reviewers asking "what's the actual
   decoding accuracy at this site?" deserve a debiased number.

This plan addresses #2 only. #1 stays as-is.

## Approach

**Nested window selection** (a.k.a. leave-one-fold-out window
selection) on the existing per-fold scores. Crucially, this is **not
nested CV** — no additional decoder fits are required. The decoders are
already K-fold CV'd at every window in
[acoustic_decoding_single_electrode.py](../notebooks/causal6/acoustic_decoding_single_electrode.py)
and equivalents; `scores.parquet` stores per-(site, window, fold) test
AUCs. "Honest peak AUC" is a re-aggregation of values you already have:

For each site:

1. For each held-out fold k:
   - Compute mean test AUC over folds {1..K} \ {k} per window.
   - Pick `w_k = argmax` of those means.
   - Read fold k's test AUC at window `w_k`.
2. `honest_roc_auc = mean_k(fold_k_test_AUC at w_k)`.

The independence guarantee comes from window selection and held-out
evaluation using disjoint folds. Since this requires K-fold (disjoint
test sets), it cannot be applied to the **legacy causal5 behavior
decoder** which uses ShuffleSplit (overlapping test sets across
splits). Causal6 uses StratifiedKFold for all three decoders (acoustic,
behavior, ganong) so all three are eligible — see
[project_behavior_decoder_imbalance.md](../../.claude/projects/-Users-jon-Projects-barakeet/memory/project_behavior_decoder_imbalance.md)
in user memory for the CV-strategy history.

## Three design decisions worth flagging

### 1. Decouple peak window from honest AUC

Different folds may select different `w_k`. Do **not** put a
fold-dependent (smin, smax) into `phon_peaks_df` /
`behav_hga_only_peaks_df` /
`ganong_hga_only_peaks_df` — `prepare_neurometrics.py` joins on these
keys to pull per-trial decoder predictions at peak windows
(see [prepare_neurometrics.py:671-728](../notebooks/causal6/prepare_neurometrics.py)).

Recommended: keep the current full-fold-mean argmax as the canonical
peak (smin, smax) for downstream joins and "when does it peak"
claims. Add `honest_roc_auc` as a **second column alongside** the
existing biased `real_statistic` / `test_roc_auc`. Two numbers, two
jobs:

- `real_statistic` / `test_roc_auc`: biased fold_mean at the canonical
  peak window. Used by all existing downstream consumers.
- `honest_roc_auc`: CV-debiased point estimate. New column, used in
  reporting / figure annotations / paper tables.

Additive change — avoids a sweep through downstream consumers.

### 2. Variance penalty

K=5 means each fold's AUC is on ~20% of trials. The nested estimate has
higher SE than the biased fold_mean. On low-trial electrodes the
de-biasing gain may be smaller than the added Monte Carlo noise. Worth
a quick empirical check (compute both, compare per-site SE) before
committing publicly.

### 3. Skip honest computation in null for v1

The pipeline is asymmetric in a way that makes "do it for null too" a
real refactor, not a regroup:

- **Real:** `scores.parquet` keeps per-(site, window, fold) test AUCs.
  Honest aggregation is polars regrouping. Cheap.
- **Null:** `null_scores.parquet` is **pre-aggregated to per-(site,
  window, perm) `fold_mean_diff`** by `_preagg_hga_only_null` /
  `_preagg_with_control_null` in
  [src/models/causal6_aggregates.py:541-628](../src/models/causal6_aggregates.py).
  Fold dimension is gone on disk. The paired-diff aggregation is the
  variance-reduction step that makes the existing test work.

To get an honest null you'd need to either retain per-fold null AUCs
on disk (5× storage on an already-large artifact) or change the perm
loop in
[acoustic_decoding_null.py](../notebooks/causal6/acoustic_decoding_null.py)
and the behavior/ganong null counterparts to compute honest_null_AUC
inside the loop (one scalar per site × perm).

For v1, **leave nulls untouched**. The existing max-stat handles
inference correctly; honest_AUC is a reporting statistic, not an
inference one. They're different jobs.

## Phasing

### Phase 1 — real-only honest AUC across all three decoders

Add a `honest_roc_auc` column to:

- `phon_peaks_foldmean_maxstat.parquet` and
  `phon_peaks_tstat_maxstat.parquet` (acoustic — see
  [acoustic_decoding_peaks.py](../notebooks/causal6/acoustic_decoding_peaks.py))
- `behav_hga_only_peaks_*` flavors
- `ganong_hga_only_peaks_*` flavors

Implementation: a helper in
[src/models/significance.py](../src/models/significance.py) (or a new
module if it grows) that takes `scores.parquet`-style per-fold AUCs and
returns per-site honest AUC. Call it from each `*_peaks.py` notebook
alongside the existing aggregation.

Update PaperData / `prepare_neurometrics.py` to thread the new column
through. Existing biased `real_statistic` / `test_roc_auc` stays as
the join key and primary effect-size column for legacy figures; new
column is opt-in for new figures.

### Phase 2 — sanity check

At sites the existing max-stat marks significant (p<0.05 corrected),
look at where `honest_roc_auc` falls.

- If it stays above chance with reasonable margin across the population
  → biased and honest agree on direction at significant sites; the
  existing inference is sufficient. Stop here.
- If a non-trivial fraction of "significant" sites have honest_AUC ≤
  0.5 → there's an interpretive problem regardless of inference
  machinery, and Phase 3 becomes worth doing.

### Phase 3 (conditional) — honest null + CV-honest p-value

Triggered only by an unfavorable Phase 2 result.

Right route: compute `honest_null_AUC` **inside the perm loop**, emit
one scalar per (site, perm). Same on-disk size as the current
pre-aggregate. Do **not** retain per-fold null AUCs on disk.

Inference becomes: standard per-site permutation p-value comparing
real `honest_roc_auc` against the null distribution of
`honest_null_AUC`. No max-stat correction needed because the nested
selection itself absorbs the multiple-comparisons cost (held-out
evaluation is independent of the selection).

Two p-values can coexist in the peaks parquets:

- `p_value` (existing): max-stat corrected on biased fold_mean.
- `honest_p_value` (new): per-site permutation against honest_null_AUC.

Disagreements between the two flag corner-case sites worth manual
inspection.

## Non-goals

- No change to the canonical peak (smin, smax) used for downstream
  joins.
- No change to causal5 (legacy, ShuffleSplit makes this approach
  inapplicable without modifications to outer CV).
- No change to existing FDR aggregate (`significance_aggregate.py`) — it
  consumes the existing `p_value` column, which stays as-is.

## References

- Discussion thread: 2026-05-08 conversation between user and assistant
  about de-biasing peak AUCs in causal6.
- Adjacent pipeline work:
  [tfce-tstat-causal6.md](tfce-tstat-causal6.md) (different angle on
  peak-finding — variance-normalized t-stat + TFCE, layered alongside
  fold_mean max-stat).
- Westfall & Young (1993) *Resampling-Based Multiple Testing* — for
  the relationship between max-stat and selection bias.
