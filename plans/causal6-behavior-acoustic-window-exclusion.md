# Per-electrode acoustic-window exclusion for behavior decoder peak search

## Context

Behavior in this task largely follows the acoustic cue. A behavior decoder
trained on HGA could rediscover the acoustic encoding rather than perceptual
processing — i.e., the "behavior peak" is just the acoustic peak relabeled.
The previous defense (require behavior decoder to beat a resampled-acoustics
regression baseline) is distrusted.

**Empirical motivation.** In the provisional results, 9 electrodes show
both an acoustic and a behavior response. For 8 of those 9, the acoustic and
behavior peak windows are essentially identical — at those sites we cannot
distinguish "behavior decoder finds genuine perceptual signal" from "behavior
decoder relabels the acoustic encoding". The constraint below directly
addresses this by forcing the behavior search past the acoustic window.

**Acoustic-only and behavior-only sites are largely disjoint**, so the
constraint should be gated on the acoustic decoder being meaningfully above
chance: at sites with no real acoustic signal there is no confound to
neutralise, and using the noisy argmax `smax` would arbitrarily clip the
behavior search.

We **gate on peak ROC-AUC**, not on p-value, because permutation results
aren't available for every subject yet — but per-site fold-mean AUCs are
already computed in §2 of the notebook for every subject with acoustic
scores.

## The rule

For each (subject, electrode, phoneme_pair) site, define a behavior search floor:

```
behav_smin_floor = acoustic_peak_smax + buffer        if peak_auc(acoustic) >= AUC_THR
                 = global _PEAK_SEARCH_SMIN           otherwise
```

Defaults:
- `buffer = 5 samples` (50 ms at 100 Hz, one window stride past the
  acoustic peak end)
- `AUC_THR = 0.60` (matches the threshold used in the existing §4 prints and
  §5 axvlines — `peak_auc > 0.6`)
- Acoustic window length is fixed at 15 samples (`config.yaml:35`), so
  `acoustic_peak_smax = acoustic_peak_smin + 15` when only `peak_smin` is
  available.

Replace the current global `smin >= _PEAK_SEARCH_SMIN` clause with
`smin >= behav_smin_floor`. The other two clauses (`smax <= word_end_offset`
and `smax <= _PEAK_SEARCH_SMAX`) are unchanged.

## Where the pieces live

- **Acoustic peak source for Phase 1**: §2 of
  `notebooks/causal6/view_provisional_results.py` already aggregates raw
  fold scores from
  `outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet`
  into per-site `(peak_smin, peak_auc)`. Phase 1 will reuse that
  computation and store the per-subject peaks in a list, then concat into a
  single `ac_peaks_df`. This works for every subject with acoustic
  decoding done, regardless of permutation status.
  - Window length 15 samples (`config.yaml:35`) → `peak_smax = peak_smin + 15`.
- For Phase 2, the same data is available in
  `outputs/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet`
  (significance-aware, but only where permutations have completed). We can
  switch to that source once it's available everywhere; the gating column
  is `test_roc_auc` either way.
- Current global behavior filter:
  - Notebook prototype: `notebooks/causal6/view_provisional_results.py:66-72`
    (used in sections 4, 5, 6 — raw peak AUC, fold-tstat brain export, and
    on-the-fly significance).
  - Production: `_filter_behavior_window` at
    `src/models/causal6_aggregates.py:226-245`.
- Snakemake rules that will gain a new dependency on `phon_peaks.parquet`:
  - `behavior_decoding_single_electrode_summarize`
    (`workflows/causal6.Snakefile:651-688`)
  - `behavior_decoding_single_electrode_hga_only_summarize`
    (`workflows/causal6.Snakefile:691-728`)

## Implementation, in two phases

### Phase 1 — prototype in `view_provisional_results.py` (this turn)

1. **Add constants near the top of the file:**

   ```python
   _AC_BEHAV_BUFFER_SAMPLES = 5     # 50 ms past acoustic peak smax
   _AC_AUC_THRESHOLD = 0.60         # gate constraint on acoustic decoder AUC
   _AC_WINDOW_LEN = 15              # samples; from config.yaml
   ```

2. **In §2's per-subject loop**, append the per-site `_peak` frames (already
   computed for the brain export) into a side list keyed by subject. After
   the loop, concatenate into a single `ac_peaks_df` and derive the floor
   table by gating on AUC:

   ```python
   ac_peaks_df = (
       pl.concat([
           p.with_columns(pl.lit(s).alias("subject"))
           for s, p in ac_peaks_per_subject.items()
       ])
       if ac_peaks_per_subject else pl.DataFrame()
   )
   ac_floor_df = (
       ac_peaks_df
       .filter(pl.col("peak_auc") >= _AC_AUC_THRESHOLD)
       .select(
           "subject", "electrode_idx", "phoneme_pair",
           (pl.col("peak_smin") + _AC_WINDOW_LEN + _AC_BEHAV_BUFFER_SAMPLES)
               .alias("_behav_smin_floor"),
       )
   )
   ```
   §2 currently builds `ac_brain_frames` (with x/y/z/roi joined). Splitting
   the bare `_peak` capture out adds a `subject` column on each frame and
   keeps the brain-plot path unchanged.

3. **Replace `_filter_window_expr` with a join-based helper** so the floor
   can vary per site:

   ```python
   def _apply_behavior_window_filter(df: pl.DataFrame | pl.LazyFrame):
       floor = ac_floor_df.lazy() if isinstance(df, pl.LazyFrame) else ac_floor_df
       return (
           df.join(floor, on=["subject", "electrode_idx", "phoneme_pair"], how="left")
             .with_columns(
                 pl.col("word_end")
                   .replace_strict(_OFFSET_SAMPLES, default=None)
                   .alias("_smax_limit"),
                 pl.col("_behav_smin_floor")
                   .fill_null(_PEAK_SEARCH_SMIN),
             )
             .filter(
                 (pl.col("smin") >= pl.col("_behav_smin_floor"))
                 & (pl.col("smax") <= pl.col("_smax_limit"))
                 & (pl.col("smax") <= _PEAK_SEARCH_SMAX)
             )
             .drop("_smax_limit", "_behav_smin_floor")
       )
   ```

4. **Update the three call sites** (sections 4, 5, 6) to call
   `_apply_behavior_window_filter(...)` instead of
   `.filter(_filter_window_expr())`. The eager paths in §4 and §5 use the
   DataFrame branch; §6's `pl.scan_parquet(...)` path uses the LazyFrame
   branch so streaming is preserved.

5. **Diagnostic block** right after `ac_floor_df` is built (printed, not
   plotted, to keep it light):
   - n sites with constraint (peak_auc ≥ AUC_THR) vs n falling back to
     `_PEAK_SEARCH_SMIN`, per subject.
   - Distribution of `_behav_smin_floor` (in ms post word onset) across
     constrained sites.
   - For each subject: count of behavior sites whose unconstrained peak
     `smin` was below the new floor (i.e., sites whose peak _moves_ under
     the constraint). The 8/9 "overlap" sites should show up here.

6. **Sanity check on the 9 overlap sites.** Print a small table: for each
   site that has `peak_auc(acoustic) ≥ AUC_THR` AND a behavior peak, show
   acoustic peak smin/smax, new floor, old behavior `peak_smin`, new
   behavior `peak_smin`, old/new `peak_auc`. We expect 8/9 to show a
   downward AUC shift after the constraint.

No edits outside this notebook in Phase 1.

### Phase 2 — productionise (separate turn, after Phase 1 looks good)

1. Extend `_filter_behavior_window` in `src/models/causal6_aggregates.py`
   with optional `acoustic_peaks_df`, `buffer_samples`, `auc_threshold`
   arguments. Default-off so existing callers don't change behaviour until
   wired in.
2. Add config keys:
   - `analysis.decoding.behavior_acoustic_exclusion_buffer_samples`
   - `analysis.decoding.behavior_acoustic_exclusion_auc_threshold`
3. Wire `phon_peaks.parquet` as an input to:
   - `behavior_decoding_single_electrode_summarize`
   - `behavior_decoding_single_electrode_hga_only_summarize`
   Pass the path through to the notebooks; load + filter inside.
4. Update both summarize notebooks
   (`behavior_decoding_single_electrode_summarize.py` and the HGA-only
   variant) to construct the floor table and call the extended filter.
5. Note in rule docstrings: behavior summarize now blocks on acoustic peaks.

## Verification (Phase 1)

1. Run notebook cell-by-cell. Confirm:
   - `ac_floor_df` has rows only for sites with `peak_auc ≥ AUC_THR` and no
     null `_behav_smin_floor`.
   - The diagnostic block reports a sane number of sites being constrained,
     roughly matching the count printed in §2's per-subject "AUC median /
     max" lines for high-AUC sites.
2. Cross-check the 9 overlap sites: for the 8 with co-located windows, the
   constrained behavior peak `smin >= acoustic peak smax + 5` and the new
   `peak_auc` should be measurably lower than the unconstrained value. For
   the 1 outlier, the new peak should be close to the old peak.
3. Population-level check: §5 brain-plot t-stat distribution and §6
   significance counts should drop modestly. The drop should concentrate at
   high-acoustic-AUC sites; behavior-only sites should be unaffected.
4. Output schema unchanged:
   `outputs/causal6/brain_plot_behav_tstats.parquet` should have the same
   columns as before.

## Cross-reference

Complements, does not replace, the baseline-control defense
(`aggregate_behavior_with_control`, `SITE_KEYS_BEHAVIOR_WITH_CONTROL`) — that
test asks "does HGA add anything beyond resampled acoustics?", whereas this
one asks "is the behavior peak temporally separable from the acoustic peak?".
Both can be reported.
