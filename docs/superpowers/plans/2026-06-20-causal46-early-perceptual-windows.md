# Infer early perceptual windows (causal46_joined)

Status: design, ready to implement (2026-06-20)
Builds on: `behavioral_discriminative_windows.py` (2026-06-19) and `t_tests.py` (B4
within-completion bootstrap)
Feeds (GIVEN, not built here): a subsequent "perceptual onset vs. acoustic decoder
onset" comparison that takes the **leftmost edge** of each cell's earliest union and
compares it to the acoustic decoder's left edge.

## Goal

For each B4 cell `(subject, electrode_idx, phoneme_pair, word_end)` that has an
**at-acoustic behavioral response** (`behav @ac` non-empty in the manifest), find the
time window(s) **in `[t=0, phon_smax]`** carrying a reliable within-completion
behavioral (percept) contrast. This is the mirror image of the 2026-06-19 analysis:
that one searched *beyond* the acoustic peak (`smin ≥ phon_smax`); this one searches
*up to and including* the acoustic window, to locate where the early perceptual signal
begins relative to the acoustic response.

This step only **finds the unified windows**. The leftmost-edge selection and the
comparison to the acoustic decoder are explicitly downstream decisions (per user).

## Decisions (from user, 2026-06-20)

1. **Search region = `[t=0, phon_smax]`.** Candidate windows are grid windows with
   `smin ≥ SAMPLE_T0` and `smax ≤ phon_smax`, where
   `SAMPLE_T0 = round((0 − epoch_tmin) · epoch_sfreq) = round((0 − (−0.4)) · 100) = 40`
   and `phon_smax` is the per-cell acoustic-peak window end from `b4_per_cell`
   (same column the 2026-06-19 analysis used as its acoustic boundary).
   - Verified against prod `b4_per_window.parquet`: grid is 29 contiguous width-5
     windows, `smin ∈ [0,140]`, so `(40,45),(45,50),…` exist and `40` lands on a
     window edge (no mid-window truncation). `phon_smin ∈ [45,53]`, `phon_smax ∈
     [60,68]` → ~4–6 candidate windows per cell, covering both the pre-acoustic gap
     (40→phon_smin) and the acoustic window (phon_smin→phon_smax).
   - **Right-edge truncation at `phon_smax` is intentional** and does not affect the
     leftmost-edge goal. Unions are capped there by design.
2. **Cell filter = non-empty `behav @ac`.** 55 manifest rows qualify (vs. the
   `@ac slightly late`/`@late` set used 2026-06-19). Process the intersection with
   `b4_per_cell` (cells present in the bootstrap). `behav @ac` is "the bin where this
   matters" — the at-acoustic behavioral response is exactly the early signal of
   interest.
3. **Acoustic reference for the downstream comparison = deferred.** This notebook
   emits `phon_smin`/`phon_smax` (acoustic-peak window) only; the downstream analysis
   decides what "acoustic decoder left edge" resolves to.
4. **Bare-minimum machinery.** Keep union-finding (maximal runs of adjacent +
   significant + same-sign windows) and the union CI. **Drop** the fallback
   (max-|median| seed), the decoder-window placement, and the per-replicate
   `*_bootstrap.parquet`. A non-significant fallback would give a misleading "onset",
   so significance-only is required here.
   - Cells with **no** significant window in `[t=0, phon_smax]` produce **zero rows**
     (no fallback). Expect some empty cells; preserve the empty-output schema path.

Inherited unchanged from 2026-06-19:
- **Union compute = per-replicate mean of component `mean_diff_raw`** (bit-identical
  to re-running the bootstrap on the union for a contiguous equal-width grid).
- **Reference = `mean_diff_raw`** (fixed /n/−/d/), never `mean_diff_aligned`.
- **Significance = bootstrap CI excludes zero**, raw per-window, no FDR.
- **Sign concordance** breaks a run at a sign flip.
- **Grid validation:** assert `stride == window_size` and contiguity dynamically.

## Inputs / outputs

New notebook: `notebooks/causal46_joined/early_perceptual_windows.py`
Output dir: `outputs/causal46_joined/early_perceptual_windows/`

Rule inputs (reference `outputs/...`; read prod state from `outputs_prod/...`):
- `outputs/causal46_joined/t_tests/b4_bootstrap.parquet` — per (cell × window ×
  replicate) `mean_diff_raw`. Exists on prod; the local `outputs_prod/` mount is stale
  (shows only the Jun 17 `b4_per_window.parquet`), so the replicate-level array can't
  be inspected here, but `b4_per_window` confirms the grid structure used below.
- `outputs/causal46_joined/t_tests/b4_per_cell.parquet` — `phon_smin/phon_smax`,
  `n_per_class`, `acoustic_peak_auc`, `R_replicates`.
- `outputs/causal46_joined/manual_annotations/filtered_manifest.csv` — `behav @ac`.
- `notebooks/causal46_joined/early_perceptual_windows.py`

No epoch reload, no trial_balance — pure post-processing.

Rule outputs:
- `notebook.ipynb`
- `ep_windows.parquet` — summary, one row per inferred union window
- (optional) `ep_windows_summary.pdf` — light QC (windows/cell, β histogram, timing
  scatter)

### `ep_windows.parquet` schema (lean)
`subject, electrode_idx, phoneme_pair, word_end, window_id,
smin, smax, n_component_windows, component_smins (list[int]), sign,
beta_ambig_median, beta_ambig_ci_low, beta_ambig_ci_high, ci_excludes_zero,
phon_smin, phon_smax, n_per_class, acoustic_peak_auc, R, behav_ac_tuning`

- `window_id`: per-cell index for the multiple-union case.
- `beta_ambig_median` + CI: union-level, from `summarize_replicate_array` on the
  per-replicate averaged array (not averaged per-window CIs).
- `ci_excludes_zero`: union CI re-tested after averaging.
- `phon_smin/phon_smax`: acoustic-peak window, carried for the downstream comparison.
- `behav_ac_tuning`: the manifest `behav @ac` letter (selection provenance).
- No `beta_ambig_mean`, no `is_fallback`, no `narrower_than_decoder`,
  no `behav_decoder_smin/smax`, no `post_word_offset`, no per-replicate parquet.

## Algorithm (per qualifying cell)

1. Slice `b4_bootstrap` to the cell; derive grid from unique `(smin,smax)`; **assert**
   `stride==window_size` and contiguity.
2. Candidate windows = grid windows with `smin ≥ 40` and `smax ≤ phon_smax`
   (`phon_smax` from `b4_per_cell`).
3. Per candidate, `summarize_replicate_array(mean_diff_raw)` → `median`,
   `ci_excludes_zero`.
4. Significant windows → maximal runs of *adjacent + significant + same-sign* windows
   (each run = one union). **No fallback**: if no significant window, emit nothing.
5. Per union, average `mean_diff_raw` at the replicate level across component windows
   → union β array; summarize → `beta_ambig_median`, CI, `ci_excludes_zero`, `sign`.
6. Emit one summary row per union; assign `window_id` per cell.

## Shared-logic extraction (do first)

`_window_sign` and `_find_maximal_runs` currently live inline in
`behavioral_discriminative_windows.py`. Extract both into a new
`notebooks/causal46_joined/_windows.py`; have **both** notebooks import them
(`summarize_replicate_array` already lives in `_within_completion.py`).
`_fallback_run` stays with the 2026-06-19 notebook (only consumer).

**Re-validation note:** the 2026-06-19 notebook has never run on prod (no
`b4_bootstrap.parquet` yet), so there is no known-good output to regress the refactor
against. The functions are pure and copied verbatim; re-validation happens naturally
when t_tests is re-run and both notebooks first execute. Flag, don't block.

## Snakefile wiring

New rule `joined_early_perceptual_windows` in `causal46_joined.Snakefile`, mirroring
`joined_behavioral_discriminative_windows` (drop the `b_windows_boot` output and
`decoder_window_size`/`manual_override_path` params). Params: two input parquet paths,
`outdir`, `ci_low`/`ci_high` (2.5/97.5), `filtered_manifest_path`. Add
`ep_windows.parquet` to `causal46_joined_all`.

## Downstream (NOT built here)

A later analysis takes, per cell, the **leftmost** union's `smin` (earliest perceptual
onset), converts to time (`smin/sfreq + tmin`), and compares to the acoustic decoder's
left edge (reference TBD). The temporal-relationship question — does the perceptual
signal onset precede / coincide with / lag the acoustic decoder — is answered there.

## Open / verify

- `b4_bootstrap.parquet` exists on prod (the local `outputs_prod/` mount is stale and
  doesn't show it) — no t_tests re-run needed; the rule runs on prod against the real
  file. Local validation is limited to `py_compile` + `snakemake -n` + the grid checks
  already done from `b4_per_window.parquet`.
- Naming: `early_perceptual_windows.py` / `ep_windows.parquet` proposed; rename if a
  more parallel `..._discriminative_windows_early` is preferred.
