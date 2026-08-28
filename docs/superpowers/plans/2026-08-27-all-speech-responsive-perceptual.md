# Perceptual effects in all speech-responsive sites (unconditioned on acoustic response)

**Status: DELIVERED** (2026-08-27, branch `late-projection-manual`) — Steps 1–4
implemented as new notebooks under `notebooks/causal46_joined/all_sr/` (kept
out of the already-large flat `notebooks/causal46_joined/` directory) +
Snakefile rules (`workflows/causal46_joined_all_sr.Snakefile`, included from
`causal46_joined.Snakefile`; existing AS-restricted pipeline untouched).
Pure-Python core (`compute_sr_site_universe`) lives in
`src/causal46_joined.py`, unit-tested in `tests/test_all_sr_perceptual.py`
(no epoch/pipeline data required). Not yet run end-to-end against production
epoch data — epochs aren't available in this dev container; verified via
`py_compile`, `snakemake -n` DAG resolution, and offline unit tests on
synthetic data + real `phon_peaks_all.parquet` / `find_speech_responsive`
CSVs (see "Implementation notes" at the end of this doc). Someone with prod
access needs to run the Snakemake targets and inspect
`t_tests_all_sr_reconciliation`'s verdict before trusting Step 4's output.

Went through one round of `/code-review` (Standards + Spec sub-agents) after
the initial implementation, which drove a significance-methodology rewrite
(maxstat + BH-FDR) — **that rewrite was then itself reverted** after Jon
corrected the premise it rested on (see "Implementation notes," final entry):
`late_integration_maxstat_significance.py` is not wired into the Snakefile
and doesn't feed `plot_for_paper`, so it was never real precedent for how
this statistic should be treated. The headline is `mean_diff_raw` /
`ci_raw_excludes_zero` — raw, uncorrected — matching the design doc's
original Step 3 text and how `t_tests.py`'s output is actually consumed
everywhere else in the pipeline. This status line and the notes below
reflect that final, twice-corrected design.

**Goal.** Test for within-completion perceptual effects across *all* speech-responsive
sites, without first restricting to sites that have a significant early acoustic
response. Parallel pipeline, forking from `find_speech_responsive`. Do not touch the
existing (AS-restricted) pipeline.

## Where the acoustic conditioning actually lives

The paper's perceptual result feeding `plot_for_paper` is the **B4 within-completion
bootstrap** (`outputs/causal46_joined/t_tests/b4_*`), not behavior decoding. The only
acoustic conditioning is the **site set**, always derived from `phon_peaks_all.parquet`:

| Location | What it does |
|----------|--------------|
| `trial_balance_index.py:64` | `canonical = peaks.filter(p_value < 0.1)` — site universe. Trial-count/step-balance logic is subject-level, broadcast across electrodes (acoustic-independent); only the site **keys** are acoustic. |
| `t_tests.py:139` | `peaks = filter(p_value < 0.001)` — stricter AS set. |
| `t_tests.py:191` | `b4_qualified = trial_balance.filter(ambiguous).groupby(...).join(peaks, how="inner")` — **the gate.** Restricts cells to AS sites and attaches `phon_smin/phon_smax/acoustic_peak_auc`. |

Crucially, `phon_smin/phon_smax` are consumed **only** by aligned-polarity
(`acoustic_preferred_class` → `mean_diff_aligned`). The behavioral searchlight bound
`behav_search_range` ignores them (returns `0, PAIR_SMAX[pp]`), and `mean_diff_raw`
needs no acoustic window. The measurement machinery — per-step balanced
within-completion subsampling (`_within_completion.py`), searchlight bootstrap, null —
is entirely acoustic-agnostic.

**Therefore the fork is a site-set swap, not a re-derivation of the analysis.**

## Plan

New notebooks + Snakefile rules writing to a sibling tree
`outputs/causal46_joined/t_tests_all_sr/`, reusing `_within_completion.py` unchanged.

### Step 1 — all-SR site universe (new rule `sr_site_universe`)
Read every `outputs/causal6/find_speech_responsive/{subject}_results.csv`, keep
`speech_responsive` electrodes, cross with the phoneme_pairs each subject saw → a table
keyed by `(subject, electrode_idx, phoneme_pair)`. Left-join `phon_peaks` (p<0.001) to
attach an **`acoustic_significant` annotation column** — a label, never a filter. This
column is what enables the partition readout (Step 4). No manual-curation gate (unlike
`canonical_AS_sites.csv` / `as_reconciliation.py`).

### Step 2 — trial balance for all SR sites (`trial_balance_index_all_sr`)
Copy of `trial_balance_index.py` with the site universe = Step 1 table instead of the
`p_value < 0.1` peaks filter. Pure key-set swap (subject-level counts broadcast
unchanged). Output `trial_balance_index_all_sr.csv`.

### Step 3 — B4 bootstrap for all SR sites (`t_tests_all_sr`)
Copy of `t_tests.py` with two changes:
1. Line ~191 inner-join → **left join** against `phon_peaks`, so all SR cells survive;
   `phon_smin/phon_smax` are null for non-acoustic sites.
2. Carry `acoustic_significant` through to `b4_per_cell` / `population_summary`.

Emit `mean_diff_raw` as the headline contrast (the null + two-sided CI is polarity-free,
so `ci_excludes_zero` is unaffected). **Aligned / polarity labelling is out of scope** —
it's undefined for non-acoustic sites, adds nothing to per-cell significance (a constant
sign flip leaves a two-sided CI invariant), and is recomputed downstream only where a
signed/directional claim actually needs it. Leave `mean_diff_aligned` NaN (or drop the
column); don't gate anything on it here.

Searchlight bound and the within-completion draws are unchanged.

### Step 3b — AS-subset reconciliation (blocking gate on credibility)
Restrict the all-SR run to AS sites and diff against the existing `b4_per_cell` /
`population_summary`. Because `trial_balance` counts are subject-level and broadcast
across electrodes, and the left-join leaves AS rows unchanged, this should reconcile
**exactly** on `mean_diff_raw` / `ci_excludes_zero`. Any mismatch
means an upstream artifact leaked in (K-filter, join semantics, universe superset). The
partition in Step 4 is only meaningful once the AS cells reproduce the current test —
otherwise it's two different tests, not one test partitioned. (Same "pop reconciliation"
trap flagged in the #22 build.)

### Step 4 — partition readout (the scientifically load-bearing output)
Cross-tabulate **perceptual-significant × acoustic_significant** across all SR sites.
The new cell — *perceptually significant but NOT acoustically significant* — directly
bears on the CLAUDE.md claim that distal integration is "ruled out by co-localization
(91%)." **Either outcome is a result**: a large non-acoustic-but-perceptual cell
*revises* the co-localization claim; a near-empty one *confirms it more strongly than
the current design can* — because the AS-conditioned pipeline structurally cannot see
non-acoustic perceptual sites. Report N in each cell against the existing 64-site
acoustic denominator. AS ⊂ SR holds by construction (`phon_peaks` comes from acoustic
decoding that already ran on the SR set), so the universe is a clean superset.

## Caveats to decide before running (not blockers to the design)

- **Polarity / aligned labelling: out of scope** (per Jon). Recomputed downstream only
  where a signed/directional claim needs it. This fork produces raw contrasts + the
  `acoustic_significant` annotation; nothing here gates on polarity.
- **Copy drift.** `t_tests_all_sr.py` is a thin copy of `t_tests.py` (most logic lives in
  `_within_completion.py`); keep the two in sync, or keep the diff small enough to review.
- **Multiple comparisons.** A larger site universe inflates the test count; current B4
  has "No FDR for now." Broadening strengthens the need for FDR / count-vs-null (see
  the late-perceptual count-vs-null against the `b4_bootstrap` null).
- **Selection / circularity.** With no acoustic anchor, the searchlight selects its
  window on the perceptual contrast itself; the population count-vs-null must use the
  window-max null (the existing per-window null already searchlights — verify the
  population step consumes the max-window statistic).

## Implementation notes (2026-08-27)

- **Selection/circularity: promoted to blocker, "fixed" twice, then reverted — the
  headline is raw `ci_raw_excludes_zero` after all.** Removing the AS pre-selection
  removes its side effect of keeping the multiple-comparisons burden small, and this
  looked like it needed correcting. Round 1: a bespoke paired-replicate maxstat
  (`maxstat_replicate_test`) as the headline. Round 2, after `/code-review` flagged
  round 1 as reinventing machinery: rewrote to mirror
  `late_integration_maxstat_significance.py`'s max-|z| permutation + BH-FDR method
  (`cell_maxstat_fdr_test`), plus a permutation-floor adequacy check
  (`maxstat_floor_check`) added after a follow-up advisor pass caught that the mirror
  had dropped that method's own floor-censoring safeguard. **Both rounds were wrong at
  the root, per Jon's correction (2026-08-28):** `late_integration_maxstat_significance.py`
  has no Snakefile rule and does not feed `plot_for_paper` — it is a standalone
  diagnostic, not the pipeline's actual precedent for this statistic. Traced the real
  `plot_for_paper` dependency chain and confirmed: `t_tests.py`'s raw, uncorrected
  bootstrap CI (`ci_raw_excludes_zero`) is what the paper's B4-derived claims actually
  use directly, with no maxstat/BH-FDR/TFCE correction anywhere in that path. The one
  place real BH-FDR exists downstream (`late_perceptual_projection.py`) uses the raw CI
  purely as a candidate GATE, then applies its own separate correction to its own
  separate projection statistic — not relevant here since Step 4 isn't feeding a
  projection step. `cell_maxstat_fdr_test` and `maxstat_floor_check` were removed
  entirely from `src/causal46_joined.py` (confirmed with Jon: remove, don't keep as an
  optional column); `t_tests_all_sr.py` and `perceptual_acoustic_partition.py` are back
  to the plan's original literal design — `mean_diff_raw` / `ci_raw_excludes_zero` as
  the sole, unqualified headline, matching how `t_tests.py`'s output is treated
  everywhere else in the codebase. Lesson: "the codebase already has a method for this"
  needs verifying against the actual DAG/consumer graph, not just grepping for a
  notebook that operates on the same input file.
- **`acoustic_significant` is a single source of truth.** `sr_site_universe.py` (Step 1)
  is the only place `ac_p_value_threshold` gets applied (via left join against
  `phon_peaks_all` filtered to `p_value < threshold`, exactly mirroring `t_tests.py`'s
  own join — `phon_smin`/`phon_smax` are null for non-significant cells by the same
  mechanism). `t_tests_all_sr.py` never re-filters by p-value; it only reads the
  `acoustic_significant` column already attached in `sr_site_universe.parquet`. This
  makes the AS threshold used by the fork and by the reconciliation check
  (`t_tests_all_sr_reconciliation.py`) identical by construction, not by convention — the
  earlier risk (prod config's `ac_p_value_threshold: 0.01` diverging from the notebook's
  inline default `0.001`) can't recur because there's only one place the threshold is
  read. (Spec review independently confirmed both `joined_t_tests` and the new rules
  pull threshold/window/K from the same `config["causal46_joined"]` dict, not stale
  inline defaults.)
- **Reconciliation is bit-exact, not approximate, and it's why aligned could be dropped
  cleanly.** `per_cell_best` in `t_tests_all_sr.py` selects the best window by
  `|mean_diff_raw_med|` (no aligned column exists in this fork). In `t_tests.py`,
  `mean_diff_aligned = sign * mean_diff_raw` with `sign` constant *per cell* (fixed once
  from endpoint tuning, not per window), so `|aligned| == |raw|` at every window within a
  cell — the argmax-over-windows selection is identical either way, and the two-sided
  `ci_excludes_zero` test is sign-invariant. So the AS subset of the raw-only fork
  reconciles exactly against `t_tests.py`'s raw fields
  (`t_tests_all_sr_reconciliation.py` checks `mean_diff_raw_med` /
  `ci_raw_excludes_zero` with `atol=rtol=1e-9`, plus a cell-set diff). This is a blocking
  gate (raises `AssertionError` on any mismatch, not just a diagnostic) —
  `perceptual_acoustic_partition.py` (Step 4) reads `reconciliation_summary.csv` and
  refuses to run if it didn't pass. This check compares raw-field parity only, so it
  was unaffected by the significance-methodology churn above — the raw bootstrap
  computation itself is unchanged throughout and still a documented near-verbatim copy
  of `t_tests.py`'s (see the "Duplicated Code" note below).
- **Duplicated bootstrap code, deliberately not extracted.** Standards review flagged
  `t_tests_all_sr.py`'s `bootstrap_cell` as a near-verbatim copy of `t_tests.py`'s (same
  RNG call order, same scratch-variable names, minus the aligned/`preferred`-class
  block). Not extracted into a shared helper: `t_tests.py` is the frozen AS-restricted
  pipeline this fork must not touch, and reconciliation IS the sanctioned drift
  detector — if the two copies diverge on the shared raw computation, the reconciliation
  gate fails loudly the next time both pipelines run. `t_tests_all_sr.py` now says this
  explicitly next to the copy.
- **Rename to avoid confusion with an unrelated notebook.** `as_reconciliation_all_sr.py`
  was renamed to `t_tests_all_sr_reconciliation.py` — the original name collided
  (same "as_reconciliation" prefix) with the pre-existing, unrelated
  `notebooks/causal46_joined/as_reconciliation.py` (a causal4-vs-causal6 site
  reconciliation notebook with no Snakefile rule). Its two near-identical
  mismatch-finding loops (per-window, per-cell) were also collapsed into one
  `find_mismatches` helper.
- **Don't hardcode the paper-reported AS denominator.** The local dev snapshot of
  `phon_peaks_all.parquet` (a bind mount of prod state, not a fresh run) gives
  15/29/221 AS electrodes at `p<{0.05,0.001,0.05}` depending on threshold and
  FDR-correction choice — nowhere near CLAUDE.md's "64 electrodes" figure (presumably
  from a fuller production run under different config). `perceptual_acoustic_partition.py`
  computes and reports its own denominator from whatever
  `sr_site_universe_electrode_level.csv` actually contains, and prints the paper-reported
  figure alongside for reference only, never asserted against
  (`paper_reported_as_electrode_n` — updated to 44 by Jon directly in the notebook,
  2026-08-28, superseding the 64 placeholder used during initial implementation).
- **Local verification ceiling.** `outputs/epochs_preprocessed/` isn't available in this
  dev container (only `outputs_prod/causal6` and `outputs_prod/causal46_joined` are
  bind-mounted, and neither contains raw/preprocessed epochs) — every rule from
  `preprocess_epochs` onward is unreachable locally, same as the rest of this pipeline.
  Verification here: `py_compile` on all five new notebooks; `snakemake -n` DAG
  resolution through `perceptual_acoustic_partition` (fails only at the pre-existing
  `preprocess_epochs` missing-input wall, same as `joined_t_tests`); `compute_sr_site_universe`
  exercised directly against the real `outputs_prod/causal6/{acoustic_decoding_peaks,find_speech_responsive}`
  data (957 SR electrodes, 81 AS at p<0.01 — sane shape); the full
  per_window→per_cell→partition-crosstab chain (the notebook integration seam
  `py_compile` can't check — wrong-column-name bugs there wouldn't surface any other
  way) exercised end-to-end against synthetic bootstrap frames reproducing the shape
  `bootstrap_cell` emits, correctly separating the "new cell" (non-AS, significant)
  from an AS cell and a null one. 6 unit tests in `tests/test_all_sr_perceptual.py`
  (down from 23 after removing the maxstat/floor-check machinery and its tests — see
  the reverted "Selection/circularity" note above), all passing; full `tests/` suite
  passes except one pre-existing unrelated failure (`test_perm_idx_seeds.py`,
  confirmed pre-existing via `git stash`).
- **Reviewed via `/code-review since HEAD`** (Standards + Spec sub-agents, run against
  a scoped diff of only this fork's files — the branch itself is far ahead of `master`
  with unrelated prior work, so a `since master` diff would have reviewed hundreds of
  unrelated commits). Findings and how each was resolved are folded into the notes
  above rather than kept as a separate review log. Note the irony worth flagging
  explicitly: this review is what drove the maxstat/BH-FDR detour in the first place
  (citing `late_integration_maxstat_significance.py` as established precedent without
  checking whether it was wired into the DAG) — a lesson for weighting future
  sub-agent findings that cite "existing codebase precedent": verify the precedent is
  actually load-bearing, not just present.
