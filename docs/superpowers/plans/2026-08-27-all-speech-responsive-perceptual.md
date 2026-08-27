# Perceptual effects in all speech-responsive sites (unconditioned on acoustic response)

**Status: DELIVERED** (2026-08-27, branch `late-projection-manual`) — Steps 1–4
implemented as new notebooks under `notebooks/causal46_joined/all_sr/` (kept
out of the already-large flat `notebooks/causal46_joined/` directory) +
Snakefile rules (`workflows/causal46_joined_all_sr.Snakefile`, included from
`causal46_joined.Snakefile`; existing AS-restricted pipeline untouched).
Pure-Python cores (`compute_sr_site_universe`, `cell_maxstat_fdr_test`) live
in `src/causal46_joined.py`, unit-tested in `tests/test_all_sr_perceptual.py`
(no epoch/pipeline data required). Not yet run end-to-end against production
epoch data — epochs aren't available in this dev container; verified via
`py_compile`, `snakemake -n` DAG resolution, and offline unit tests on
synthetic data + real `phon_peaks_all.parquet` / `find_speech_responsive`
CSVs (see "Implementation notes" at the end of this doc). Someone with prod
access needs to run the Snakemake targets and inspect
`t_tests_all_sr_reconciliation`'s verdict before trusting Step 4's output.
Went through one round of `/code-review` (Standards + Spec sub-agents) after
the initial implementation; the significance-methodology finding it surfaced
drove a rewrite documented below — this status line and the notes reflect
the POST-review design, not the first draft.

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

- **Selection/circularity was promoted from caveat to blocker, then revised again after
  `/code-review`.** Removing the AS pre-selection removes its side effect of keeping the
  multiple-comparisons burden small — and Step 4's whole point is the
  *perceptual-but-not-acoustic* cell, exactly where an uncorrected best-window search
  would land false positives. A first pass added a bespoke paired-replicate maxstat
  (`maxstat_replicate_test`) as the headline. Both review sub-agents flagged this as
  reinventing machinery the codebase already has:
  `notebooks/causal46_joined/late_integration_maxstat_significance.py` already runs a
  max-|z| permutation correction per cell + BH-FDR across cells on this exact
  `t_tests/b4_bootstrap.parquet` structure, with its own note that a self-selected
  best-window count "must not be reported as a test." The fix (per advisor consult):
  `cell_maxstat_fdr_test` (`src/causal46_joined.py`) now MIRRORS that method — z_obs =
  |mean_r(mean_diff_raw)| / std_r(null) per window, obs_maxz = max over windows,
  null_maxz per replicate the same way, `p = (#{null≥obs}+1)/(R+1)` (unbiased, never
  exactly 0, unlike the first draft's plain fraction), then BH-FDR across all tested
  cells (`statsmodels.stats.multitest.multipletests`) — rather than importing from that
  notebook (which is part of the frozen AS-restricted pipeline this fork must not
  touch) or inventing a third lookalike. The one real difference: that notebook
  restricts windows to `smin >= phon_smax` (post-acoustic only), which doesn't apply
  here — non-AS cells have no `phon_smax` to anchor to, so this fork searches the full
  `behav_search_range` instead. `maxstat_reject` is the fork's headline per-cell call;
  `best_ci_raw_excludes_zero` (naive, self-selected window) is kept alongside labeled
  CIRCULAR, never as the headline, per that same notebook's own standard. This is still
  an implementer decision beyond the design doc's literal Step 3 text and was not
  confirmed with Jon synchronously before running — flagged as such in
  `t_tests_all_sr.py`'s "NOTE ON HEADLINE CHOICE," in `perceptual_acoustic_partition.py`,
  and here, not buried in a docstring only. BH-FDR is one family across all tested
  cells; it is NOT further corrected across the electrode-level collapse
  `perceptual_acoustic_partition.py` does downstream (an electrode with several tested
  cells gets several independent chances to pass) — a second deferred decision, same
  status.
- **Permutation-floor adequacy check (second advisor pass).** Mirroring
  `late_integration_maxstat_significance.py`'s method (above) but not its floor check
  left a gap: a permutation p floors at `1/(R+1)`, and BH-FDR rejects a rank-1 p only if
  `p <= alpha/n_cells`. Solving, a floor-pinned cell survives only if
  `n_cells <= (R+1)*alpha` — at R=1000, alpha=0.05, that's `n_cells <= 50`. The all-SR
  family (hundreds of qualifying cells, not late_integration's 187 AS-restricted ones)
  is plausibly over that line, and unlike that notebook (a null result either way) this
  fork is hunting a POSITIVE finding — the perceptual-not-acoustic cell. Undetected, a
  "0 in the new cell" result would read as confirmation of co-localization when it could
  be permutation censoring. `maxstat_floor_check` (`src/causal46_joined.py`) computes
  `floor`, `min_p`, `n_at_floor`, and `floor_limits_rejection` (`floor >
  alpha/n_cells`); `t_tests_all_sr.py` writes it to `maxstat_floor_check.csv` and prints
  a loud warning when triggered; `perceptual_acoustic_partition.py` reads it and
  attaches the same warning directly next to a "NEW CELL = 0" result, not just upstream.
  If triggered on a real run, the fix is a higher `n_bootstrap` (e.g. 10000 → floor
  1e-4, supports BH-FDR over a few-hundred-cell family), not a methodology change.
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
  refuses to run if it didn't pass. This check is orthogonal to the maxstat/BH-FDR
  rewrite above — it compares raw-field parity only, so changing the significance
  METHOD (but not the raw bootstrap computation, which is unchanged and still a
  documented near-verbatim copy of `t_tests.py`'s — see that file's "Duplicated Code"
  note below) can't break it.
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
- **Don't hardcode 64.** The local dev snapshot of `phon_peaks_all.parquet` (a bind
  mount of prod state, not a fresh run) gives 15/29/221 AS electrodes at
  `p<{0.05,0.001,0.05}` depending on threshold and FDR-correction choice — nowhere near
  CLAUDE.md's "64 electrodes" figure, which is presumably from a fuller
  production run under different config. `perceptual_acoustic_partition.py` computes
  and reports its own denominator from whatever `sr_site_universe_electrode_level.csv`
  actually contains, and prints the paper's "64" figure alongside for reference
  (`paper_reported_as_electrode_n`, never asserted against — spec review flagged the
  first draft for dropping even this non-asserting reference).
- **Local verification ceiling.** `outputs/epochs_preprocessed/` isn't available in this
  dev container (only `outputs_prod/causal6` and `outputs_prod/causal46_joined` are
  bind-mounted, and neither contains raw/preprocessed epochs) — every rule from
  `preprocess_epochs` onward is unreachable locally, same as the rest of this pipeline.
  Verification here: `py_compile` on all five new notebooks; `snakemake -n` DAG
  resolution through `perceptual_acoustic_partition` (fails only at the pre-existing
  `preprocess_epochs` missing-input wall, same as `joined_t_tests`); `compute_sr_site_universe`
  exercised directly against the real `outputs_prod/causal6/{acoustic_decoding_peaks,find_speech_responsive}`
  data (957 SR electrodes, 81 AS at p<0.01 — sane shape); the full
  per_window→per_cell→maxstat→floor_check→partition-crosstab chain (the notebook
  integration seam `py_compile` can't check — wrong-column-name bugs there wouldn't
  surface any other way) exercised end-to-end against synthetic bootstrap frames
  reproducing the shape `bootstrap_cell` emits, including at production scale (R=300,
  3 cells across the AS/non-AS/null split, 300 replicates × 3 windows — correctly
  separates the "new cell" from an AS cell and a null one, and separately at R=1000
  with 3 windows / 2 cells). 23 unit tests in `tests/test_all_sr_perceptual.py`, all
  passing; full `tests/` suite passes except one pre-existing unrelated failure
  (`test_perm_idx_seeds.py`, confirmed pre-existing via `git stash`).
- **Reviewed via `/code-review since HEAD`** (Standards + Spec sub-agents, run against
  a scoped diff of only this fork's files — the branch itself is far ahead of `master`
  with unrelated prior work, so a `since master` diff would have reviewed hundreds of
  unrelated commits). Findings and how each was resolved are folded into the notes
  above rather than kept as a separate review log.
