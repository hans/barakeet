# Plan: Early-window site-type classification

## Context

Within the **acoustic decoder's peak-selection search range** (the early window),
some acoustically-selective sites also show a within-completion behavioral
(report-dependent) HGA contrast on ambiguous trials. This is interesting because
that early window is *before* the lexical disambiguation point — so a
report-dependent split there means either the electrode is committing to a
percept early, or trial-by-trial acoustic variability is propagating to both
HGA and report, or there's an internal-state bias.

This analysis classifies each (subject × electrode × phoneme_pair) site into
one of five categories based on whether, within the early/acoustic-search
window, it shows:

- a clean acoustic contrast on unambiguous trials (`A`)
- a within-completion behavioral contrast on ambiguous -esolate-like trials (`B₁`)
- a within-completion behavioral contrast on ambiguous -ecessary-like trials (`B₂`)
- and whether the behavioral contrasts are *aligned* with the acoustic tuning.

## Conceptual ontology (5 types)

For sites with a clean (single-letter) acoustic tuning:

| Type | A | B₁ | B₂ | Interpretation |
|---|---|---|---|---|
| **Type 1** | present | absent | absent | Pure acoustic encoder. |
| **Type 2** | present | present, aligned | present, aligned | Early perceptual analyzer — same tuning across all three conditionals. |
| **Type 3** | present | exactly one of (B₁, B₂) present, aligned | other absent | Asymmetric across completion. Must rule out trivial cause (report imbalance / power). |
| **Grab-bag** | present | any anti-aligned B significant | (any) | Acoustic & behavioral tunings diverge in early window. |

Plus one more category retained from the manual annotation:

| Type | Definition |
|---|---|
| **Type C (complex)** | Site is annotated as `acoustic tuning ∈ {both, complex, two peaks}` in the manifest. Acoustic tuning is not a single-letter sign, so cannot be entered into the aligned/anti-aligned grid. Behavioral B₁/B₂ are still reported (raw, unsigned) but not classified into types 1–3 / grab-bag. |

Type C is reported alongside types 1–3/grab-bag but kept in a separate bucket;
no aligned-sign claim is made for these sites.

## Inputs

1. **Epochs**: `outputs/epochs_preprocessed/{subject}_epo.fif` — for trial-level
   HGA. Loaded via MNE; metadata enriched with `add_metadata_features()`.
2. **Filtered manifest**: `outputs/causal46_joined/filtered_manifest.csv` —
   per-cell manual annotation. Used for:
   - Restricting the analyzed site pool to those that appear in the manifest
     (already QC-filtered).
   - The `acoustic tuning` column drives the complex-vs-clean split: rows
     with `acoustic tuning ∈ {both, complex, two peaks}` → Type C; rows with
     a single-letter value → enter types 1–3/grab-bag classification.
   - Note: `acoustic tuning` is a **group property** of (electrode × phoneme_pair).
     Verify identical across both word_ends per site; if disagreement, log
     and treat as Type C conservatively.
3. **Phoneme-pair timing**: `OFFSET_DICT`, `WORD_END_TO_PHONEME_PAIR` from
   `src/stimuli.py` for the acoustic-search range upper bound.
4. **t_tests helpers**: `notebooks/causal46_joined/_within_completion.py` for
   `searchlight_mean_diff` (10-sample / 10-stride windowed contrasts).
5. **trial_balance** parquet (whatever path `t_tests.py` reads from) for the
   data-driven `is_ambiguous_step` qualification.

**Do NOT use** `phon_peaks_df` / decoder-peak windows for alignment sign.
The sign comes from the new A bootstrap (see below).

## Operational definitions

### Acoustic search range (per phoneme pair)

Matches the acoustic decoder's peak-selection range
(`notebooks/causal5/acoustic_decoding_peaks.py:118–151`):

- `smin_lo = phon_response_smin_min = int((0.0 - epoch_tmin) * epoch_sfreq)`
  → sample 40 with `epoch_tmin=-0.4, sfreq=100` (= word onset).
- `smax_hi = min(max_word_end_offset_sample(pair), int((1.3 - epoch_tmin) * epoch_sfreq))`
  where `max_word_end_offset_sample(pair) = max over word_ends in pair of
  (OFFSET_DICT[word_end] - epoch_tmin) * epoch_sfreq`.
- Searchlight windows: 10 samples wide, stride 10. Window `(smin, smax)`
  qualifies if `smin >= smin_lo` and `smax <= smax_hi`.

This is **the same range for A, B₁, and B₂** at a given phoneme pair, so the
three conditionals are directly comparable window-by-window.

### Ambiguous-trial qualification (for B₁, B₂)

Identical to `t_tests.py` (lines 178–186):

- A step is ambiguous if `(min_class > 2) & (~resampled.is_in([1, 6]))`
  (both classes appear with >2 trials at that step).
- A (subject × electrode × phoneme_pair × word_end) cell qualifies if the
  sum of `min_class` over qualifying steps is `>= K` (default `K = 4`).
- Bootstrap resamples balance trials within step per class (preserves per-step
  acoustic structure).

### Unambiguous-trial qualification (for A)

- Restrict to trials with `resampled ∈ {1, 6}`.
- Per phoneme pair, require some minimum n per endpoint (suggest `>= 4`, matching B).
- Bootstrap resamples balance trials within endpoint step (step 1 → /d/-like class,
  step 6 → /n/-like class for the dn pair; analogous for bm, pb).

### Bootstrap mechanics (A, B₁, B₂)

- `R = 1000` replicates.
- For each replicate × searchlight window:
  - Resample trials within each class with replacement, preserving per-step
    class counts (for B: per ambiguous step; for A: per endpoint step).
  - Compute `mean_diff_raw[r, w] = mean_pos - mean_neg`.
- Per-window summary: median, 2.5 / 97.5 percentile CI, two-sided empirical
  p, `ci_raw_excludes_zero = (ci_lo > 0) | (ci_hi < 0)`.

### Alignment sign (from A bootstrap)

For each (subject × electrode × phoneme_pair) site that has a significant A
window:

- Take the **best A window** = window maximizing `|median(mean_diff_raw_A)|`
  among A windows with `ci_raw_excludes_zero`.
- `acoustic_sign = sign(median(mean_diff_raw_A))` at the best A window.
  - `+1` → site is /n/-tuned (class 1 wins) on this pair.
  - `−1` → site is /d/-tuned.
- For B₁ and B₂: `mean_diff_aligned = acoustic_sign × mean_diff_raw_B`.
- Sites without a significant A window get `acoustic_sign = NaN` and are
  excluded from types 1–3 / grab-bag classification (reported as
  "A-unsigned", treated like Type C for the alignment-dependent splits).

### Per-cell aggregation

For each (subject × electrode × phoneme_pair × word_end) cell, declare
**B present** if **any window in the acoustic search range** has
`ci_aligned_excludes_zero` AND `median(mean_diff_aligned) > 0`.
**B anti-present** if any window has `ci_aligned_excludes_zero` AND
`median(mean_diff_aligned) < 0`.

(This matches the searchlight-cluster spirit: a contrast counts if it
appears anywhere in the restricted range; we are not constrained to the
exact A peak window.)

### Site-type assignment (per electrode × phoneme_pair)

Combine the two word_end cells (B₁ = -esolate-like, B₂ = -ecessary-like).
Both cells must satisfy `n_per_class >= K` to be classifiable; if one is
underpowered, mark the site `unclassifiable_B_power` and report what we have.

1. **If site is Type C** (manifest tuning ∈ {both, complex, two peaks}):
   → `site_type = "complex"`, no alignment classification.
2. **Else if A is not significant**:
   → `site_type = "A_unsigned"`, report raw B contrasts but no alignment.
3. **Else** (clean tuning, A significant, both B cells powered):
   - If neither B₁ nor B₂ has any significant window:
     → `site_type = "type1_acoustic_only"`.
   - If B₁ and B₂ both have at least one *aligned* significant window AND
     neither has an *anti-aligned* significant window:
     → `site_type = "type2_early_perceptual"`.
   - If exactly one of {B₁, B₂} is aligned-significant AND the other is
     fully null AND neither is anti-aligned-significant:
     → `site_type = "type3_asymmetric"`.
   - Else (any anti-aligned significant window, or mixed signals):
     → `site_type = "grab_bag"`.

## Implementation

### New files

- **Notebook**: `notebooks/causal46_joined/early_window_site_types.py`
  (Jupytext percent-format, single subject per invocation).
- **Helper additions**: extend `_within_completion.py` if needed to support
  the A-bootstrap (which lacks per-step within-class balancing — it
  resamples within each endpoint step). Prefer adding a new function
  `searchlight_mean_diff_endpoints` rather than overloading the existing
  one.
- **Snakemake rule**: add to `workflows/causal46_joined.Snakefile` a new rule
  `early_window_site_types` plus an aggregation rule that concats per-subject
  parquets into population summaries.

### Notebook structure (per subject)

1. Parameters: `subject`, `epochs_path`, `manifest_path`, `trial_balance_path`,
   `K` (default 4), `R` (default 1000), `outdir`.
2. Load epochs + metadata; load manifest filtered to this subject; load
   trial_balance.
3. Compute acoustic search range per pair.
4. For each (subject × electrode × phoneme_pair) site present in manifest:
   - Compute A bootstrap (unambiguous trials), per-window summary.
   - Compute B₁ and B₂ bootstraps (ambiguous, per word_end), per-window summary.
   - Apply alignment sign from A → aligned B summaries.
5. Save per-subject parquets (see Outputs below).

### Outputs (per subject, then aggregated)

| Parquet | Rows | Key columns |
|---|---|---|
| `A_per_window.parquet` | (subject × electrode × phoneme_pair × window) | smin, smax, tmin, tmax, mean_diff_raw_med, ci_lo, ci_hi, emp_p, ci_excludes_zero, n_per_class, **endpoint_steps_used** (always `[1, 6]` but recorded explicitly), **n_step1**, **n_step6** |
| `B_per_window.parquet` | (cell × window) | subject, electrode_idx, phoneme_pair, word_end, smin, smax, mean_diff_raw_med, ci_raw_lo/hi, mean_diff_aligned_med, ci_aligned_lo/hi, emp_p_aligned, ci_aligned_excludes_zero, n_per_class, **qualifying_steps** (list of resampled-step ints, e.g. `[3, 4, 5]`), **n_per_step_per_class** (dict-like: `min_class[s]` for each qualifying step) |
| `site_type_assignments.parquet` | (subject × electrode × phoneme_pair) | acoustic_sign, A_significant, B1_aligned_sig, B1_anti_sig, B2_aligned_sig, B2_anti_sig, manifest_tuning, site_type, status (ok / unclassifiable_B_power / A_unsigned / complex), **B1_qualifying_steps**, **B2_qualifying_steps**, **B1_n_per_class**, **B2_n_per_class** |

**Propagate `qualifying_steps` everywhere.** The ambiguous-step set is data-driven
per cell (depends on this subject's behavior at this phoneme_pair × word_end), so
it is essential metadata: any downstream reader needs to know which trial subsets
underlie each row. Match the existing convention in `b4_per_cell.parquet` from
`t_tests.py` (which already stores `qualifying_steps`). Per-step counts are also
recorded so power and balancing decisions are reproducible from the parquet alone.

### Aggregation rule

Concat per-subject parquets → population-level outputs in
`outputs/causal46_joined/early_window_site_types/`:

- `population_site_types.csv` — counts by `site_type × phoneme_pair × ROI`.
- All three per-window parquets aggregated.

### Snakemake rule sketch

```python
rule early_window_site_types:
    input:
        epochs        = "outputs/epochs_preprocessed/{subject}_epo.fif",
        manifest      = "outputs/causal46_joined/filtered_manifest.csv",
        trial_balance = "outputs/causal46_joined/trial_balance/{subject}.parquet",  # check actual path
        helper        = "notebooks/causal46_joined/_within_completion.py",
        notebook      = "notebooks/causal46_joined/early_window_site_types.py",
    output:
        notebook             = "outputs/causal46_joined/early_window_site_types/{subject}/notebook.ipynb",
        A_per_window         = "outputs/causal46_joined/early_window_site_types/{subject}/A_per_window.parquet",
        B_per_window         = "outputs/causal46_joined/early_window_site_types/{subject}/B_per_window.parquet",
        site_type_assignments = "outputs/causal46_joined/early_window_site_types/{subject}/site_type_assignments.parquet",
    run:
        # standard jupytext->execute pattern matching joined_acoustic_ax_discrimination
        ...

rule early_window_site_types_aggregate:
    input:
        site_types = expand("outputs/causal46_joined/early_window_site_types/{subject}/site_type_assignments.parquet",
                            subject=SUBJECTS),
        # plus per-window parquets if population analyses use them
    output:
        population = "outputs/causal46_joined/early_window_site_types/population_site_types.csv",
        figures    = directory("outputs/causal46_joined/early_window_site_types/figures"),
    run: ...
```

## Visualizations

All figures are produced by the aggregation step (population) or by the per-subject
notebook (per-site galleries). Output to
`outputs/causal46_joined/early_window_site_types/figures/`.

### 1. Per-site early-window star-plot gallery (`star_plots_early.pdf`)

PDF gallery, one page per (subject × electrode × phoneme_pair) site. Three stacked
panels per page, all sharing the same x-axis (time in seconds, with the acoustic
search range shaded gray):

- **Top panel — A (acoustic, unambiguous)**: trial-averaged HGA traces for steps 1
  vs 6. Color by endpoint phoneme (e.g. blue=/d/, orange=/n/ for `dn`). Bootstrap
  CI ribbons. Significant searchlight windows marked as colored bars along the top
  edge of the panel; the best A window highlighted with a thicker bar / different
  saturation.
- **Middle panel — B₁ (-esolate-like word_end)**: trial-averaged HGA traces for
  heard-/d/ vs heard-/n/ on this word_end's qualifying ambiguous steps. CI ribbons.
  Significant windows marked; sign of `mean_diff_aligned` shown as a glyph on each
  significant window (`▲` aligned, `▼` anti-aligned).
- **Bottom panel — B₂ (-ecessary-like word_end)**: same construction.

Page header (text):
- `subject / electrode / phoneme_pair / ROI / manifest_acoustic_tuning / site_type`
- `B₁ qualifying_steps = [...]`, `n_per_class = ...`
- `B₂ qualifying_steps = [...]`, `n_per_class = ...`
- `A endpoint counts: n_step1=..., n_step6=...`

Ordering: sort pages by `site_type` (type-2 first, then type-3, grab-bag,
type-1, complex, A-unsigned), then by phoneme_pair, then by subject. So
reviewers can flip straight to the interesting types.

This is the early-window analog of `b4_powered.pdf` but with three stacked
panels (A shared across word_ends, both B₁ and B₂ shown on one page per site
rather than per cell).

### 2. Population type-counts bar chart (`population_site_type_counts.pdf`)

Stacked horizontal bars, one bar per (phoneme_pair × ROI) combination,
segments colored by site_type. Segment widths = count of sites.

- Categories (ordered): type-1, type-2, type-3, grab-bag, complex, A-unsigned,
  unclassifiable_B_power.
- Color scheme: distinct categorical palette; type-2 (the headline category)
  gets the most salient color.
- Annotate each bar with total n.
- Companion CSV (`population_site_type_counts.csv`) with the same data tidy
  for re-plotting: columns `phoneme_pair, ROI, site_type, n`.

### 3. Per-type exemplar figures (`exemplars_type{1,2,3,grabbag,complex}.pdf`)

For each of the five named types, pick 3 exemplar sites (manually curated or
chosen by a deterministic rule, e.g., largest `|best_B_aligned_median|`
within type; for complex, by largest `|best_A_median|`). One figure file
per type, with 3 panel-rows (one per exemplar), each row having the same
three sub-panels as the star plot (A / B₁ / B₂) but laid out side-by-side
rather than stacked, with a per-row title showing site identity.

Designed for paper inclusion: thicker lines, larger fonts, no per-window
significance bars (just the best A and best B windows shaded). Captions in
the figure (not separate) describe what the type means.

Exemplar selection rule:
- Type 1: among sites with `site_type == "type1_acoustic_only"`, pick top 3
  by `|best_A_median|` (illustrates clean acoustic encoding with no behavioral
  leakage).
- Type 2: top 3 by `min(|B₁_aligned|, |B₂_aligned|)` (forces both word_ends
  to be strongly aligned, not just one).
- Type 3: top 3 by `|aligned_side_median| - |null_side_median|` (largest
  asymmetry across word_ends).
- Grab-bag: top 3 by `|anti_aligned_median|`.
- Complex: top 3 by `|best_A_median|` among complex-annotated sites.

Save the chosen exemplar IDs to `exemplar_sites.csv` so the figure is reproducible
and the choices can be reviewed.

### 4. A-vs-B scatter (`A_vs_B_scatter.pdf`)

Two-panel scatter (one panel per word_end, since B is per-cell). Each point is
one (subject × electrode × phoneme_pair × word_end) cell with a valid A best
window (i.e. excludes A-unsigned and complex).

- **x-axis**: `best_A_median × acoustic_sign` (always ≥ 0 — magnitude of acoustic
  contrast in tuning direction).
- **y-axis**: `best_B_aligned_median` (signed — positive = aligned with acoustic
  tuning, negative = anti-aligned).
- **Color**: `site_type` (type-1 / type-2 / type-3 / grab-bag).
- **Marker**: `phoneme_pair` (e.g. circle/square/triangle for bm/dn/pb).
- **Reference lines**: y = 0 (acoustic-only horizon), y = x diagonal (perfect
  proportional perceptual readout).
- **Error bars**: CI bars on both axes (from per-window bootstrap CIs at the best
  A and best B windows).

The expected geometry that the typology predicts:
- Type 1 → points on the y = 0 line, scattered along x.
- Type 2 → points above y = 0, ideally clustering near the y = x diagonal.
- Type 3 → mixed (one cell type-2-like, the other type-1-like at the same site).
- Grab-bag → points below y = 0.

If type-2 sites cluster off the diagonal (e.g. systematically below it),
that's a quantitative claim worth making about the magnitude relationship
between A and B in early-window perceptual analyzers.

## Decisions logged

1. **Ambiguous-only for B**: yes. Standard t_tests qualification (`is_ambiguous_step`).
2. **Bootstrap for A** (not the decoder ROC-AUC), to match B's statistical treatment.
3. **Searchlight range**: acoustic decoder's *peak-selection search range*
   `[word_onset_sample, max_word_end_offset_sample(pair)]`. NOT the per-site
   peak window. Same range for A, B₁, B₂.
4. **Alignment sign**: from the A bootstrap's best window. Not from
   `phon_peaks_df`.
5. **Cross-cell aggregation**: type-2 requires BOTH B₁ and B₂ significant
   and aligned with no anti-aligned signal; type-3 = exactly one;
   grab-bag = any anti-aligned.
6. **Complex acoustic responses retained**: sourced from
   `outputs/causal46_joined/filtered_manifest.csv` `acoustic tuning` column,
   values in `{both, complex, two peaks}`. Reported with raw (unsigned) B
   contrasts; not entered into types 1–3/grab-bag.
7. **Site pool**: only sites that appear in `filtered_manifest.csv` (already
   QC-filtered). Sites without manifest entries are out of scope for this
   analysis.

## Open issues for the implementing agent

1. **Trial-balance path**: confirm where `trial_balance` is written by the
   upstream rule and pass it as a notebook input. Check
   `workflows/causal46_joined.Snakefile` for the `trial_balance_index` rule.
2. **Manifest `acoustic tuning` per-WE consistency**: the manifest is per
   (cell × WE) but `acoustic tuning` should be a group property. Verify
   agreement across the two WE rows per site; on disagreement, conservatively
   tag the site complex and log the conflict.
3. **n per class for A**: pick a sensible minimum (suggest `>= 4` to match B).
   Endpoints typically have plenty of trials, so this should rarely bind.
4. **Aggregation of per-window CI for "B present"**: current spec uses
   "any window in restricted range with `ci_excludes_zero` and aligned median
   positive." If this proves too liberal (high false positive rate on
   nominally-null sites), consider a window-count threshold (e.g., ≥ 2
   contiguous significant windows) or cluster-based correction. Note this
   for later — start with the simple criterion.
5. **Smoke test**: pick one subject with known type-2 examples (consult
   filtered_manifest.csv `behav @ac` non-empty rows) and verify the new
   notebook reproduces the manual classification on those sites before
   running the full pipeline.
