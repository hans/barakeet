# Perceptual effects in all speech-responsive sites (unconditioned on acoustic response)

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
