# Infer behaviorally-discriminative windows (causal46_joined)

Status: design, ready to implement (2026-06-19)
Builds on: `t_tests.py` (B4 within-completion bootstrap), `strong_generator_demo.py`
Feeds (GIVEN, not built here): strong-generator analysis, transfer analysis

## Goal

For each **response** = B4 cell `(subject, electrode_idx, phoneme_pair, word_end)`
(the within-completion behavioral HGA contrast), infer the time window(s) **beyond
the early acoustic window** that carry a reliable behavioral (percept) difference,
and report the ambiguous-trial slope `β_ambig` over each. A response may yield
**multiple rows** when there are temporally non-adjacent discriminative windows.

This is the per-response window table that the downstream strong-generator and
transfer analyses consume.

## Decisions (locked via grilling + advisor)

1. **Union compute = average existing replicates (no epochs).** For the B4 grid
   (contiguous, equal-width, non-overlapping, every window computed from the *same*
   per-replicate draw), the union mean-diff is exactly the per-replicate average of
   the component windows:
   `mean_diff_union[r] = (1/K) · Σ_k mean_diff_raw_k[r]` — bit-identical to
   re-running the bootstrap with the same seeds. So this is a **pure
   post-processing rule** over `b4_bootstrap.parquet`. No epoch reload, no re-fit.
   - **MUST assert** `stride == window_size` and exact contiguity
     (`smax_i == smin_{i+1}`) read *dynamically from the parquet grid*, failing
     loudly otherwise. The pipeline currently runs window=5/stride=5
     (`config.yaml` `causal46_joined`), not the notebook's 10/10 default — never
     hardcode the width.
2. **Acoustic exclusion = causal6 peak, `smin ≥ phon_smax`.** Candidate windows are
   grid windows starting at or after the cell's acoustic peak `phon_smax` (no
   overlap with the acoustic window). The acoustic window is per-`(site, pair)`,
   shared across both word-ends.
3. **Raw reference for both detection and β.** Significance = bootstrap CI of
   `mean_diff_raw` excludes zero; `β_ambig` reported on `mean_diff_raw` (fixed
   /n/−/d/ reference, matching `strong_generator_demo.py`'s fixed-reference
   refactor). Not acoustic-polarity aligned.
4. **Multiple comparisons: inherit B4 — raw per-window CI, no FDR.** A window is
   significant iff its bootstrap CI excludes zero. Document the inflation risk.
5. **Union requires sign concordance.** Break a run at a sign flip; opposite-signed
   adjacent significant windows become separate rows (prevents +/− cancellation
   when averaging).
6. **Fallback (no significant window): max-|median|, then grow union.** Seed at the
   beyond-acoustic window with largest `|median(mean_diff_raw)|`; grow over adjacent
   *same-sign* windows regardless of their individual significance; stop at
   sign-flip / grid gap / acoustic boundary on each side. Re-test the union
   (averaging can tighten the CI → a fallback union can come back significant).
7. **Per-replicate β stored in a separate long parquet** (`b_windows_bootstrap.parquet`).
8. **Manual = automated + optional hook.** Rule runs without any manual file; an
   optional override CSV (existence-checked inside the notebook, NOT a snakemake
   `input:`) can post-fix windows. Schema defined now, merge implemented as a
   guarded stub.

## Inputs / outputs

New notebook: `notebooks/causal46_joined/behavioral_discriminative_windows.py`
Output dir: `outputs/causal46_joined/behavioral_discriminative_windows/`

Rule inputs (reference `outputs/...`; read prod state from `outputs_prod/...`):
- `outputs/causal46_joined/t_tests/b4_bootstrap.parquet` — per (cell × window × replicate) `mean_diff_raw`, `qualifying_steps`, `n_per_class`, `acoustic_peak_auc`
- `outputs/causal46_joined/t_tests/b4_per_cell.parquet` — source of `phon_smin/phon_smax` **(read the acoustic boundary from here, NOT from `phon_peaks_all`)** so it matches the exact window the B4 run joined (t_tests.py:546-553). Also carries `n_per_class`, `acoustic_peak_auc`.
- `notebooks/causal46_joined/behavioral_discriminative_windows.py`

No `epoch_fifs`, no `trial_balance` — this rule is pure post-processing.

Rule outputs:
- `notebook.ipynb`
- `b_windows.parquet` — summary, one row per inferred window
- `b_windows_bootstrap.parquet` — per-replicate β, keyed by `(cell, window_id, replicate)`
- (optional) `b_windows_summary.pdf` — light QC: count of rows/response, β histogram, timing scatter, fallback fraction

### `b_windows.parquet` schema
`subject, electrode_idx, phoneme_pair, word_end, window_id,
smin, smax, n_component_windows, component_smins (list[int]), sign,
beta_ambig_mean, beta_ambig_median, beta_ambig_ci_low, beta_ambig_ci_high,
ci_excludes_zero, is_fallback,
phon_smin, phon_smax, acoustic_peak_auc, n_per_class, R,
narrower_than_decoder, post_word_offset`

- `window_id`: per-cell index (0,1,2…) for the multiple-row case.
- `beta_ambig_mean` = spec headline (mean of union replicate array); `beta_ambig_median`
  kept for continuity (selection/ranking uses median like B4; reporting uses mean —
  store both to kill "which metric?" ambiguity downstream).
- `ci_excludes_zero` (final union CI) and `is_fallback` (no individually-significant
  window seeded this row) are **orthogonal** — store both.
- `narrower_than_decoder`: flag set when `smax-smin` < **15** (causal6 decoder window
  width, `config.yaml:50` `decoding.window_size`, stride 2). The b4 grid is 5 samples,
  so a union needs **≥3 contiguous component windows** to host even one decoder-width
  sub-window; single-window and 2-window (incl. most fallback) unions are narrower.
  For these, this step synthesizes a decoder-width window (see "Behavioral decoder
  window placement"). Transfer-facing only.
- `behav_decoder_smin`, `behav_decoder_smax`: precomputed decoder-width (15-sample)
  window for narrow unions, centered on the union and lower-clamped at `phon_smax`
  (null for `narrower_than_decoder=False` rows, which the transfer step places by
  max-separation). See placement section.
- `post_word_offset`: flag when the window extends past *this* word_end's own offset.
  B4 searches to `PAIR_SMAX` (necessary's offset) even for desolate cells; inherit that
  for consistency but flag the shorter word's post-offset region as an interpretive note.

### `b_windows_bootstrap.parquet` schema
`subject, electrode_idx, phoneme_pair, word_end, window_id, replicate, beta`
(`beta` = the union's per-replicate averaged `mean_diff_raw`.) Consumed by the
strong-generator difference CI (exact Monte-Carlo over β_amb replicates).

## Algorithm (per response / B4 cell)

1. Slice `b4_bootstrap` to the cell. Derive grid from unique `(smin,smax)`;
   **assert** `stride==window_size` and contiguity; else raise.
2. Look up `phon_smax` from `b4_per_cell`. Candidate windows = grid windows with
   `smin ≥ phon_smax`.
3. Per candidate window, summarize the `mean_diff_raw` replicate array with the
   **shared significance helper** (see below): `median`, `sign`,
   `ci_raw_excludes_zero` (raw CI, no FDR).
4. Significant windows → maximal runs of *adjacent + significant + same-sign*
   windows. Each run is a union (a single-window run is a **pass-through** — read
   its values straight from the per-window summary, no 1-element "average").
5. If no significant window → fallback (decision 6).
6. For each union, average `mean_diff_raw` **at the replicate level** across
   component windows → union β array. Summarize via the same helper:
   `beta_ambig_mean`, `beta_ambig_median`, CI = 2.5/97.5 percentiles of *that*
   array, `ci_excludes_zero`. **Never average per-window CIs.**
7. Emit one summary row + R per-replicate rows. Assign `window_id` per cell.
8. (Hook) if `manual_override_path` exists, merge per the schema below.

## Behavioral decoder window placement (transfer-facing, pure geometry)

Two regimes by union width (decoder width = 15 samples):
- **width ≥ 15** → leave placement to the transfer step (max-separation 15-sample
  sub-window within the union; needs epochs — b4's width-5/stride-5 grid cannot stand
  in for a width-15/stride-2 search). `behav_decoder_smin/smax` left null here.
- **width < 15** (`narrower_than_decoder=True`) → synthesize a centered decoder window
  here (no epochs):
  1. `center = (smin+smax)/2`; window `[center−7, center+8]` (15 samples).
  2. **Lower clamp only:** if `smin_dec < phon_smax`, shift later → `smin_dec = phon_smax`,
     `smax_dec = phon_smax+15` (never extend back into the acoustic decoding region).

No upper clamp needed: the b4 search bound `PAIR_SMAX` (= latest pair word-end offset
+ `WORD_END_TAIL_SAMPLES`=20, i.e. ~200ms; `t_tests.py:115,164-178`) already sits well
inside the real epoch, so every candidate window has `smax ≤ PAIR_SMAX` and a centered
15-sample window cannot run off the epoch end. (No `decoder_window_infeasible` case.)

## Shared-logic extraction (do this first)

The CI/`ci_raw_excludes_zero`/`emp_p` computation currently lives **inline** in
`t_tests.py:per_window_summary` (lines ~449-492). Extract it into a notebook-local
helper — `summarize_replicate_array(arr, ci_low=2.5, ci_high=97.5) -> dict` returning
`mean, median, ci_lo, ci_hi, emp_p, ci_excludes_zero` — placed in `_within_completion.py`
(or a new `_windows.py`), and **both** files import + use it.

**GATING: resolve interpolation mismatch before writing helper.** Polars `.quantile()`
and `np.percentile` differ in interpolation (nearest vs. linear by default). At a CI
bound near zero this can flip `ci_raw_excludes_zero`. The helper must use the same
method as B4's stored CIs — and `strong_generator_demo.py` already uses `np.percentile`
(linear). The gating check (once `outputs_prod/` is mounted): load prod
`b4_bootstrap.parquet` + `b4_per_window.parquet`, run `summarize_replicate_array` on
each window's replicate array, compare `ci_excludes_zero` against the stored
`ci_raw_excludes_zero`. If full agreement → canonicalize on `np.percentile` (linear)
and proceed. If any disagreement → surface to user (the two requirements — match
`strong_generator` vs. reproduce stored B4 CIs — may conflict). Until confirmed,
use `np.percentile` (matching `strong_generator_demo.py`) and document the choice.

## Manual override hook (schema now, guarded stub)

Optional CSV at `outputs/causal46_joined/manual_annotations/behavioral_windows_override.csv`.
Existence-checked inside the notebook; absent ⇒ automated-only. Schema:
`subject, electrode_idx, phoneme_pair, word_end, window_id, action, smin, smax`
- `action ∈ {add, drop, edit}`. `drop` removes the matching `window_id`; `edit`
  replaces its `smin/smax` (β recomputed by re-averaging that span from
  `b4_bootstrap` — same machinery); `add` introduces a new `window_id` (β computed
  the same way). Key = `(subject, electrode_idx, phoneme_pair, word_end, window_id)`.

## Snakefile wiring

New rule `joined_behavioral_discriminative_windows` in `causal46_joined.Snakefile`,
mirroring `joined_t_tests`'s `run:`/`run_notebook` style. Params: the two input
parquet paths, `outdir`, `ci_low`/`ci_high` (= 2.5/97.5 constants matching B4),
optional `manual_override_path`. Add `b_windows.parquet` to the
`causal46_joined_all` default target.

## Sequencing

0. **After `outputs_prod/` is mounted**: run the gating check (compare numpy helper
   CIs against stored `b4_per_window.parquet` `ci_raw_excludes_zero`) and resolve
   interpolation method. Then proceed:
1. Extract `summarize_replicate_array` helper into `_within_completion.py`; refactor
   `t_tests.py`'s `per_window_summary` to use it (replace inline polars quantile with
   `map_elements` or restructure to loop — or at minimum confirm the two computations
   agree on prod data before shipping the new notebook). Iterate B4 cells from
   `b4_per_cell` (already has `phon_smax`, `n_per_class`, `acoustic_peak_auc`, `R_replicates`);
   slice `b4_bootstrap` only for the replicate arrays.
2. Build `behavioral_discriminative_windows.py` (automated path) + outputs. Only touch
   `mean_diff_raw` (fixed /n/−/d/ reference); never `mean_diff_aligned`.
3. Wire the Snakefile rule; `uv run snakemake -n` to validate the DAG.
4. (Later) implement the manual merge body; build strong-generator + transfer rules
   on top of `b_windows*.parquet`.

## Open / verify

- **`outputs_prod/` must be mounted before running the gating check.** All downstream
  verification (interpolation agreement, CI counts, β histograms) requires prod parquets.
  Container restart required to mount — plan was updated 2026-06-19 pending that.
- causal6 decoder window width = **15 samples** (`config.yaml:50`), stride 2 — locked.
  Sets `narrower_than_decoder` (need ≥3 contiguous b4 grid windows for a transferable
  union).
- 16 both-WE pairs is small N — population aggregation choices deferred to the
  downstream analyses, not this table.
