# causal46 late within-completion perceptual significance test — Implementation Plan

> **ℹ Built on the B4 bootstrap (2026-07-01).** The per-step class-balance
> subsampling this test consumes is defined canonically in
> `notebooks/causal46_joined/_within_completion.py` (module docstring); pointer at
> `docs/superpowers/plans/2026-07-01-causal46-within-completion-subsampling.md`.

> **This is the LATE-window port of the early-gate move.** ADR-0001 and ADR-0002
> replaced the manual entry gate for the *early* perceptual window with an
> automated projection (π) and explicitly left the *late* analysis on manual
> annotation. This plan does for `behav @late` what those ADRs did for `behav @ac`:
> replace a human review label with a documented, reproducible per-cell
> significance test — **only the gate moves**; the downstream window-finding is
> unchanged. A follow-up ADR-0003 should record the decision once implemented,
> matching how 0001/0002 recorded the early move.

## Motivation

`notebooks/causal46_joined/behavioral_discriminative_windows.py` currently enters
a cell into the late-window analysis iff a human tagged it `behav @late` in
`outputs/causal46_joined/manual_annotations/filtered_manifest.csv` (84 cells). It
then finds maximal same-sign CI-excludes-zero runs, but performs **no cluster-level
significance test** — the manual label *is* the significance decision. This is the
same non-reproducibility and non-circularity problem ADR-0002 identified for the
early window: the label is a judgement made while viewing the very star-plot
contrasts the analysis then consumes.

We replace the manual gate with a per-cell **TFCE permutation test** on the
within-completion percept contrast, computed as pure post-processing over the
already-persisted `b4_bootstrap.parquet` replicates. The gate becomes a documented
statistic with a permutation null; the late-window cell set regenerates from data.

## Goal

For each of the **187 powered B4 cells** `(subject, electrode_idx, phoneme_pair,
word_end)`, compute a two-tailed TFCE cluster statistic on the post-acoustic
`/n/−/d/` contrast and a within-step label-permutation p-value. Use uncorrected
per-cell p < 0.05 as the new entry gate for the late-window analysis (feeding the
downstream cascade), and report a separate population-level headline (count-vs-null
over the 187, BH-FDR floor).

**Out of scope:** cross-word-end pooling (dropped from this spec); any change to the
B4 trial-balance scheme or to the epoch-level bootstrap; describing cross-WE
matched/mirrored morphology.

---

## Settled design decisions

These were resolved in a grilling session before writing. Each is fixed; the
implementing agent should not reopen them.

### D1 — Polarity: two-tailed, signed `/n/−/d/` (raw)

The statistic operates on **`mean_diff_raw`** (positive = `/n/` > `/d/`), never
`mean_diff_aligned`. **No presupposed alignment** between the sign/tuning of the
early (acoustic-locked) response and the late response we are now detecting.

- Rationale: `behav @late` is two-tailed in character — the annotation schema records
  *both* tunings (multi-peak `"d, then n"`; matched/mirrored across WE), and ADR-0002
  states that mirrored (type4) responses "belong to the later/integration window
  studied separately." A one-tailed aligned test (π's design) would systematically
  drop the population the late window exists to capture.
- Consequence: because the statistic uses no acoustic sign, the test is independent
  of the acoustic response and of the manual labels **by construction** — the
  non-circularity concern is dissolved, not merely mitigated.

### D2 — Null source: post-processing of `b4_bootstrap.parquet`, no prod re-run

The persisted null is sufficient. In `bootstrap_cell` (t_tests.py:262–318) each
replicate `r`:
1. draws one bootstrap resample (`select_cell_trials_bootstrap`, per-step balanced);
2. builds **one** within-step label permutation (pool each step's two classes,
   shuffle once, re-split — t_tests.py:269–292);
3. evaluates that single shuffle at **every** searchlight window
   (`res_null = searchlight_mean_diff(...)`, line 293).

So the `mean_diff_aligned_null` values for a given replicate form **one coherent
across-window null curve** from a single consistent shuffle. Gathering replicate
`r`'s rows reconstructs a full time-course null; R=1000 replicates = 1000 coherent
permutations = exactly the raw material a max-TFCE null needs. **No epoch reload.**

- The permutation preserves per-step class balance → it is exactly the B4 stratum
  (`_within_completion.py`); stratification is already correct (within-step,
  within-completion — `word_end` is fixed per cell).
- **Signed raw null recovery.** `mean_diff_aligned_null = sign · null_raw` with
  `sign` a **per-cell constant** (`sign = 1 if raw_pos_key == preferred else -1`,
  t_tests.py:305). So `null_raw = sign · mean_diff_aligned_null`, and `sign` is
  recoverable per cell from any observed row as `mean_diff_aligned / mean_diff_raw`.
  For a two-tailed magnitude test this is not even needed:
  `|mean_diff_aligned_null| = |null_raw|` exactly.
- **Tied cells are the one gap.** When `preferred is None` (acoustic window cannot
  pick a preferred class), the code writes `mean_diff_aligned_null = NaN` and the raw
  null is not persisted (t_tests.py:301–303). These cells have no usable null in the
  parquet. **Handling: drop-and-document.** Plan step 1 confirms the count is
  negligible among the 187 (expected ~0); only if material do we revisit persisting a
  `mean_diff_raw_null` column (which would force one prod bootstrap re-run).

### D3 — Statistic (gate): TFCE

Per-cell gate statistic is **TFCE** (threshold-free cluster enhancement) over the
post-acoustic window axis, with a **max-TFCE permutation null** built from the R
coherent null curves. Two-tailed (enhance `|curve|`; report the signed peak).

- Params **E=0.5, H=2** (standard 1D defaults), **pre-registered**; a one-line
  param-sensitivity check (vary E∈{0.5,1.0}, H∈{1,2}) is reported, not tuned.
- **Rejected alternatives:** cluster-mass with a hard cluster-forming threshold
  (|z|>1.96) — discards the sub-threshold diffuse windows that dominate a ~0.5-SD
  signal; **integrated-window** mean over the fixed `[phon_smax, offset+0.1]` window
  — dilutes short effects against long dead time.
- **The integrated-window statistic is still computed and reported as a knob-free
  robustness column** (it has zero params, maximally satisfying the "documented,
  reproducible statistic" value), so the gate can be shown not to be an E/H artifact.
- **Empirical justification (run-length discriminator, computed 2026-07-20 over the
  84 manual cells' significant runs):** 65% are a single 50 ms window, 26% span
  2–3 windows (100–150 ms), 8% span ≥4 windows (up to 300 ms). This is the mixed
  short+diffuse regime TFCE is built for and where both rejected alternatives fail.
  Whether the single-window cells are truly short or diffuse-but-weak (only the peak
  window crosses the per-window CI), TFCE wins: tall-narrow enhancement for the
  former, low-wide shoulder aggregation for the latter.

### D4 — Window: per-WE `[phon_smax, offset+0.1]`

Candidate windows are post-acoustic (`smin ≥ phon_smax`, the acoustic boundary from
`b4_per_cell`) and within the per-WE bound
`smax ≤ round((OFFSET_DICT[we] + 0.1 − epoch_tmin) · epoch_sfreq)`. This matches the
window logic already in `behavioral_discriminative_windows.py:206–219`. No change.

### D5 — Altitude: two claims, two operating points (mirrors ADR-0002)

- **Gate (feeds the cascade).** Uncorrected per-cell TFCE p < 0.05. This regenerates
  `b_windows.parquet` and thereby feeds `strong_generator.py`, `contrast_plot.py`,
  `mismatch_regression.py`, `acoustic_endpoint_windows.py`, `acoustic_transfer.py`.
  The cascade's input set **changes** from the manual 84 to the TFCE-passing set
  (expected below the 84, since TFCE corrects across windows and the manual set
  includes 19 cells with no significant window at all). This is intended and matches
  ADR-0001's "the projection-passing set now *defines* the group."
- **Headline (population claim).** **Count-vs-null over the 187 manual-independent
  powered cells:** number with uncorrected TFCE p < 0.05 vs **Binomial(187, 0.05)** —
  the direct analogue of ADR-0002's CPO p=0.0012. **BH-FDR** across the 187 per-cell
  p-values is reported as the **conservative floor** (may be small, family-size-bound,
  as ADR-0002 noted for early). Same per-cell p-values feed both — one pass.
- **Family = 187, not 84.** The population test must not be conditioned on the manual
  labels it exists to replace.

### D6 — Manual-label agreement: calibration, not validation

The annotator tagged `behav @late` while viewing the same B4 star-plot contrasts the
TFCE test consumes → agreement is partly mechanical, **not** an independent
measurement. Report as **calibration**, mirroring ADR-0002's caveat verbatim:

- A 2×2 concordance table (TFCE-pass × manual `behav @late`) over the 187, with
  sensitivity/specificity-relative-to-manual framed explicitly as calibration.
- Disagreement cells listed **by name** (TFCE-pass ∩ not-manual = candidate manual
  misses; manual ∩ not-TFCE-pass = the 19 fallback + short cells that don't survive
  correction), for eyeball follow-up only.
- Manual labels are **not** ground truth and **never** tune the gate or operating
  point.

### D7 — Marginal cells: permutation p is the sole adjudicator

Cells between the harsh (~30) and lenient (~84) sets are decided by the declared
operating point (uncorrected TFCE p<0.05 for the cascade; BH-FDR for the headline) —
**no second criterion.** Split-half re-partitions the same bootstrap trials at half N
and is a lower-power restatement of the same permutation p; using it as a second gate
would reopen the operating-point question with no principled threshold. Split-half
survives only as an **optional descriptive reliability column** (sign agreement across
trial halves), reported, never gating.

---

## File structure

- **Create:** `notebooks/causal46_joined/late_perceptual_significance.py` — new
  notebook (Jupytext percent-format). Consumes `b4_bootstrap.parquet` +
  `b4_per_cell.parquet`; emits per-cell TFCE statistic + permutation p over the 187
  powered cells, the population headline, and the calibration table. (Separate
  notebook per the project convention: new analysis variant → new notebook, shared
  logic extracted first.)
- **Modify:** `notebooks/causal46_joined/_windows.py` — add TFCE enhancement + the
  max-statistic permutation-null helpers (extracted shared logic; `late_perceptual_
  significance.py` and `behavioral_discriminative_windows.py` both import them).
- **Modify:** `notebooks/causal46_joined/behavioral_discriminative_windows.py` —
  swap the entry gate from the manual manifest filter (lines 91–118) to the
  TFCE-passing set from the new parquet. Keep `_find_maximal_runs`, fallback, decoder
  placement, and all outputs unchanged.
- **Modify:** `workflows/causal46_joined.Snakefile` — add the
  `late_perceptual_significance` rule (inputs: `b4_bootstrap.parquet`,
  `b4_per_cell.parquet`; output: `late_perceptual_significance/site_results.parquet`
  + summary PDF); make it an input of the `behavioral_discriminative_windows` rule.
- **Create:** `outputs/causal46_joined/late_perceptual_significance/` — outputs
  (schema below).

## Outputs

- `late_perceptual_significance/site_results.parquet` — one row per powered B4 cell:
  - `subject, electrode_idx, phoneme_pair, word_end`
  - `phon_smin, phon_smax` — acoustic boundary (from `b4_per_cell`)
  - `search_smin, search_smax` — per-WE post-acoustic window bound (D4)
  - `n_windows` — candidate windows in-window
  - `tfce_peak` — signed TFCE-enhanced peak of the observed curve
  - `tfce_max_abs` — max |TFCE| (the gate statistic)
  - `tfce_emp_p` — two-tailed within-step label-permutation p (max-TFCE null)
  - `tfce_gate_pass` — `tfce_emp_p < 0.05` (uncorrected) — **the cascade gate**
  - `tfce_p_fdr`, `tfce_fdr_pass` — BH-FDR over the 187 (headline floor)
  - `integral_stat`, `integral_emp_p` — knob-free robustness column (D3)
  - `splithalf_sign_agree` — optional descriptive reliability (D7)
  - `is_tied` — `preferred is None` (dropped from gate; null unavailable, D2)
  - `manual_behav_late` — manual tag, for the calibration table only (D6)
- `late_perceptual_significance/population_summary.pdf` — count-vs-null figure,
  BH-FDR survivor count, per-window run-length histogram, param-sensitivity panel,
  2×2 calibration table + named disagreement cells.

---

## Tasks

### Step 0 — Guards / environment
- [ ] Confirm inputs exist on the `outputs_prod/` mount: `b4_bootstrap.parquet`
      (NB: not currently synced — verify it is present on prod before the run),
      `b4_per_cell.parquet` (187 rows), `filtered_manifest.csv`.
- [ ] All Python via `uv run`; render/verify locally with `py_compile` +
      `snakemake -n` (this notebook has no GL dependency, but follow the standard
      smoke path: `CONFIG_FILE=config.smoke.yaml uv run snakemake --configfile
      config.smoke.yaml -j1`).

### Step 1 — Tied-cell audit (gates D2's drop-and-document) — DONE (2026-07-20)
- [x] Count cells with `preferred is None` / `mean_diff_aligned` all-NaN among the
      187 (and among the 84 manual). Record the count in the plan checkbox.
      **Result: 0 / 187 powered B4 cells have `preferred_class is None`; 0 / 84
      manual `behav @late` cells (a subset of the 187, confirmed by key join).**
      Source: `outputs_prod/causal46_joined/t_tests/cell_manifest.parquet`,
      `preferred_class` column, filtered to `status == "ok"` (187 rows, zero
      nulls) — this is the direct per-cell output of `bootstrap_cell()`
      (t_tests.py:253, 301-321), not a proxy. Cross-checked against
      `b4_per_cell.parquet`'s `best_mean_diff_aligned_med` (also zero
      null/NaN across the 187), consistent with `preferred_class` being a
      per-cell constant so a tie NaNs the aligned value at every window.
- [x] If negligible (≤ a couple): proceed with drop-and-document. If material: STOP
      and escalate — D2 fallback (persist `mean_diff_raw_null`, one prod re-run).
      **Negligible (zero) — D2's drop-and-document strategy is confirmed.
      Downstream work (#9, #10) is unblocked; no `mean_diff_raw_null` column or
      prod re-run is needed.**

### Step 2 — TFCE helpers in `_windows.py` — DONE (2026-07-20)
- [x] `tfce_enhance(curve, dt, E=0.5, H=2)` — threshold-free enhancement over the 1D
      window axis on `|curve|`, preserving sign for reporting. Unit-test against a
      hand-worked small example (monotone bump → known enhancement ordering).
      Implemented as a thin wrapper around the MNE-validated one-tailed engine
      `src/models/significance.py::_tfce_1d` (enhance `|curve|`, restore sign).
- [x] `max_tfce_null(null_curves, ...)` — per-replicate max |TFCE| over the R coherent
      null curves → null vector; `emp_p = (1 + #{null ≥ obs}) / (1 + R)` (two-tailed on
      magnitude).
- [x] Assert each replicate contributes exactly one coherent across-window curve
      (row count == R × n_windows per cell), mirroring the union-β assertion in
      `behavioral_discriminative_windows.py:269`.
      `notebooks/causal46_joined/_windows.py`; unit tests in
      `tests/test_causal46_windows.py` (11 tests, `uv run pytest` green).

### Step 3 — `late_perceptual_significance.py` — DONE (2026-07-20)
- [x] Load `b4_bootstrap` + `b4_per_cell`; validate the grid (contiguous,
      stride==window_size) reusing the existing assertions.
      `validate_contiguous_grid` / `assert_coherent_null_replicates` (from #9's
      `_windows.py`), reused as planned.
- [x] Per cell over the **full 187**: build candidate post-acoustic windows (D4);
      form the observed curve (median over replicates of `mean_diff_raw` per window)
      and the R null curves (recovered signed `null_raw`, or `|·|` for two-tailed);
      compute `tfce_max_abs`, `tfce_emp_p`, integral stat/p, split-half column.
- [x] Drop tied cells from the gate; keep them in the parquet with `is_tied=True`.
- [x] BH-FDR over the family `tfce_emp_p`; count-vs-null vs Binomial(n_family, 0.05).
- [x] Join `manual_behav_late` for the calibration table (D6) — read-only.
- [x] Write `site_results.parquet` + `population_summary.pdf`.
      Committed `493c79e`. Verified end-to-end via `ploomber_engine` against
      synthetic fixtures (`tests/test_late_perceptual_significance.py`,
      `tests/test_causal46_windows.py` — 24/24 pass); the real
      `b4_bootstrap.parquet` is not synced to this container and is too large to
      fetch here, so the numeric population headline (count-vs-null, BH-FDR
      survivor count) has **not yet been produced on real data** — that happens
      the first time this rule runs on prod.

### Step 4 — Swap the gate in `behavioral_discriminative_windows.py` — NOT STARTED (this is #11)
- [ ] Replace the manifest filter (lines 91–118) with a join to
      `site_results.parquet` filtered on `tfce_gate_pass`. Add a param
      `late_significance_path` (must be declared in the Snakefile `run_notebook`
      parameters — notebook params must match Snakefile).
- [ ] Leave `_find_maximal_runs`, `_fallback_run`, decoder placement, and the
      `b_windows` / `b_windows_bootstrap` schema untouched.
- [ ] Keep the `manual_override_path` hook as-is.

### Step 5 — Snakefile wiring — PARTIALLY DONE (2026-07-20)
- [x] Add the `late_perceptual_significance` rule. Snakefile rule inputs reference
      `outputs/...` (not `outputs_prod/...`).
- [ ] Make its parquet an input of `behavioral_discriminative_windows`. **Deferred
      to Step 4 / #11 by design** — the rule currently runs as a leaf, listed
      directly in `causal46_joined_all` so it still executes end-to-end, but
      nothing downstream consumes `tfce_gate_pass` yet.
- [~] `snakemake -n` / `--list` confirms the rule parses and registers cleanly
      (`joined_late_perceptual_significance` appears in `snakemake --list`). A full
      dry-run against `causal46_joined_all` in this container fails upstream at
      `preprocess_epochs` (raw `epochs/{subject}_epochs.fif` not present locally)
      — a pre-existing container limitation unrelated to this rule, not yet
      re-verified as a full smoke run on prod.

### Step 6 — Report — NOT STARTED
- [ ] Population headline (count-vs-null p, BH-FDR survivors), run-length histogram,
      param-sensitivity panel, integral-vs-TFCE agreement, 2×2 calibration table +
      named disagreement cells.
- [ ] Draft **ADR-0003** recording the late-gate move (mirror ADR-0001 structure:
      what moved, why, consequences, the manual set retained as calibration only).

---

## Non-circularity ledger (why this is defensible)

- Statistic uses **no acoustic sign** → independent of the acoustic response (D1).
- Null is a **within-step label permutation** of the same trials → the correct null
  for "no percept difference", stratified to the B4 rule (D2).
- Gate is **manual-independent**; manual labels enter only a read-only calibration
  table (D6). Family for the population claim is the 187, not the 84 (D5).
- Train/test disjointness is **not** a concern here (unlike early π, which used an
  unambiguous-trial template): the late test is a single within-completion contrast
  on ambiguous trials with a permutation null — no template, no held-out split.
