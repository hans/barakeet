# Spec: Late-window perceptual projection — per-cell statistic

**Status:** ready to execute (design settled via grilling 2026-07-21, issue #20)
**Scope:** the **per-cell half** of the late-projection spec only — π_cell, its
window rule, normalization, and per-cell null. Does **not** decide the population
test, the pass threshold, or *which* â-population is claim-bearing — those are the
pre-run criterion ticket (#21). Map: #19.
**Pipeline:** `causal46_joined`. New notebook (`late_perceptual_projection` +
aggregate); reuses `_within_completion` primitives and the `b4_bootstrap`
searchlight windows. Prod-only inputs (see §7).

Companion decision record: `docs/adr/0003-late-window-perceptual-projection-percell.md`.

---

## 0. Purpose

Port the early perceptual projection (`early_perceptual_projection.py`, ADR-0002)
to the **late (integration) window** to produce a per-cell statistic
`π_cell = ⟨â, p⟩` measuring how much the within-completion **late** percept
contrast re-expresses **that same word-end's own** unambiguous /d/–/n/ tuning.
This is the evidence the map's go/no-go on *context-gated reactivation of the
perceptual code* will ride on.

The port is **not** mechanical. The late window differs from early in three ways
that force design choices, each settled below:

1. The late window is **word-end-specific** (offsets differ: desolate 0.498 s vs
   necessary 0.887 s; the response extends past offset), so there is no single
   global grid.
2. â–p alignment in the late window is **temporally narrow** — the unambiguous and
   ambiguous responses share only a short aligned band, with much idiosyncratic
   variation elsewhere. Whole-grid integration (early's reduction) would bury a
   narrow real alignment under non-aligned windows' variance.
3. Per-cell (strict same-word-end, no pooling) the endpoint-trial count is small,
   so a near-noise â can be normalized up — the "percept projected onto noise"
   problem the map's spine is about.

---

## 1. The cell

    cell = (subject, electrode_idx, phoneme_pair, word_end)

**Strict same-word-end, no pooling** — unlike early, which pooled â across
word_ends and pooled π across completions. Here â, p, and π are all computed
within a single word_end. Within-completion by construction (suffix acoustics
fixed), so the percept contrast controls acoustics.

Site pool: the same `A_significant` acoustic-responsive universe as the other
`causal46_joined` analyses (via `early_window_site_types` / `site_type_relabel.csv`),
matching the early projection's inclusion universe.

---

## 2. The window grid

Reuse the **`b4_bootstrap` late searchlight windows**: 10-sample windows, stride
10, spanning from the acoustic peak (`phon_smax_c6`) through
`word_offset + WORD_END_TAIL_SAMPLES`, **per word_end**. This grid is already
word-end-anchored (its `smax` is the word-end offset + tail), so it directly
resolves difference (1) above — no new grid construction needed.

â(w) and p(w) are **vectors over this grid** (per-window scalars), exactly as
early builds them per-window before reducing.

---

## 3. â — the late acoustic template (per window)

Per grid window w, within the cell's word_end:

    â_raw(w) = β_unamb(w) = mean HGA[step6] − mean HGA[step1]

on **unambiguous** trials of this same word_end, fixed /n/−/d/ reference (step6 =
high-numbered endpoint). Computed by
`_within_completion.bootstrap_endpoint_beta(word_end=…, smin=w.smin, smax=w.smax)`,
which returns the bootstrap distribution of β_unamb(w).

**Reliability, per window:** `beta_summary(β_unamb(w))["reliable"]` — the bootstrap
percentile CI (2.5–97.5) excludes zero. This is the existing, shared reliability
rule (`strong_generator_scan`, `strong_generator_demo` use the same).

â comes from **unambiguous** trials; p (§4) from **ambiguous** trials — disjoint
by construction, so selection and test are structurally independent (ADR-0002's
non-circularity guarantee, preserved). The sign is fixed by continuum polarity;
**no per-cell sign flip**.

---

## 4. p — the late percept contrast (per window)

Per grid window w, within word_end:

    p(w) = Σ_s [min_class[s] / N] · (mean HGA[heard-/n/, s, w] − mean HGA[heard-/d/, s, w])

the **deterministic** B4 min-class-weighted within-step percept contrast over the
cell's qualifying ambiguous steps — early's `compute_p_we`, restricted to the one
word_end. All ambiguous trials used; `min_class[s]` enters only as the per-step
weight (`min_class_k` gate, K=3). p is a **point statistic**, not a bootstrap
distribution.

`b4_bootstrap.parquet` (= β_amb replicates per cell × window) is a **cross-check on
the observed p only** — it is **not** the null engine and its bootstrap CI is
**not** the error model. The error model is the label-permutation null of §6.

---

## 5. Window rule + reduction (decision 1c) → π

The narrow-alignment fact (difference 2) rules out early's whole-grid integration.
We localize **on â** — independent of the permuted percept labels, so the null
stays a plain label-permutation with nothing to absorb — rather than on π (which
would select partly on p and force a search-absorbing null).

**Rule (â-anchored contiguous reliable run):**

1. Mark each grid window **reliable** if β_unamb(w)'s CI excludes 0 (§3).
2. **Anchor** at `w* = argmax_w |median β_unamb(w)|` (strongest tuning locus).
3. Take the **maximal contiguous run of reliable windows containing w\***. Call
   it `R`.
4. **Unit-L2-normalize** â over `R`: `â_unit = â_raw|_R / ‖â_raw|_R‖`.
5. **Primary statistic:**

       π_anchored = ⟨â_unit, p|_R⟩   (integrated over R)

   **NaN** if the cell has no reliable window (no anchor, no run) — you cannot
   project onto a tuning that isn't reliably present.

**Why unit-normalization is safe here:** the "near-noise â normalized up" worry
(difference 3) is defused by construction — every window in `R` has a
CI-excludes-0 â, so â_unit is a *reliable* direction. Normalization is not doing
the noise-rejection; the reliability gate is. Unit-normalization then makes
π_anchored "how much of the percept contrast lies along the tuning direction," in
p's units, comparable across cells.

### 5.1 Diagnostic + carried-forward descriptors (for #21)

π_anchored is NaN wherever â is unreliable, which mechanically restricts the
tested universe to â-reliable cells. To keep #21's "which â-population is
claim-bearing" decision *real* (the map's spine needs to show the
reliable-vs-all swing, stage-1's 9/77 vs 49/77), we also export, per cell:

- **π_peak** *(diagnostic, NOT the claim)* — π at the single window `w*`
  (`argmax|β_unamb|`), **reliability-ignored**; defined whenever â is computable.
- **â-reliability descriptors:** `n_reliable_windows`, `run_span` (smin/smax of
  `R`), `peak_beta_unamb_median` and its CI, `a_raw_norm` (‖â_raw‖ over `R`).

The population gate, threshold, and claim-bearing population choice all live in
#21; this ticket only *defines and exports* the statistic and the reliability
measure.

---

## 6. Per-cell null

Mirror `early_perceptual_projection.compute_permutation_null`, **restricted to one
word_end (not pooled):**

- Permute the **heard-/d/ vs heard-/n/ labels within each qualifying ambiguous
  step**, preserving that step's observed (n1, n0) split.
- **Held fixed** across replicates: `â_unit` and the window-set `R` — both derive
  from *unambiguous* trials, untouched by the permutation, so `R` is not
  re-derived per replicate.
- Recompute the **deterministic** p from the permuted labels (§4), form
  `π_null = ⟨â_unit, p_null|_R⟩`.
- Per cell = single (subject, electrode_idx, phoneme_pair, word_end). No pooling.

Bookkeeping ported as-is from early: `n_perms = 10000`, and the
`exhaustive` / `perm_space` / `min_p` accounting for small cells.

**Export both tails** per cell: one-tailed (`π ≥ π_obs`) and two-tailed
(`|π| ≥ |π_obs|`) permutation p-values. The **operating point (which tail,
threshold, FDR vs count-vs-null) is deferred to #21** — this ticket sets none.

Calibration note (from the map): the per-cell null is calibrated even at low ‖â‖
because â_unit is held fixed and only labels permute — so low-‖â‖ cells are not a
false-positive-inflation problem, they are an *interpretive* one (handled by the
reliability gate + descriptors, §5), which is why the fix is localization, not a
null correction.

---

## 7. Inputs / outputs / constraints

**Prod-only inputs** (not synced to dev):
- `outputs/epochs_preprocessed/{subject}_epo.fif` — HGA for â and deterministic p.
- `b4_bootstrap.parquet` — the searchlight window grid + observed-p cross-check.

**Outputs** (per subject + aggregate), mirroring the early projection layout:
- `site_results.csv` — per cell: `π_anchored`, `π_peak`, both-tail p-values,
  the §5.1 reliability descriptors, `n_reliable_windows`, `run_span`, cell N.
- `null_pi.npz` — per-cell null π arrays.
- `pi_dist.png` — per-subject diagnostics.

**Run constraints:** `uv run`; **serial** (never concurrent `uv run`); every run is
**Jon on prod** (the epochs + `b4_bootstrap` are prod-only). This is the #22 build.

---

## 8. Recorded alternative — searchlight π-search (1b), the fallback

If 1c under-detects (e.g. real alignments sit off the â peak, or the reliable-run
localization is too conservative), revert to a **max-over-windows π-search**:
per cell search `w` (or contiguous sub-bands) maximizing `π(w) = â(w)·p(w)`, with
the null **absorbing the search** — permute labels → recompute π over the whole
grid → re-take the max, every replicate. This is the `all_windows` path
`strong_generator_scan` already anticipated. It localizes to alignment *wherever*
it sits, at the cost of a heavier null and power spent on the search correction.
It selects partly on p, so it is **not** interchangeable with 1c's plain null —
adopting it is a re-opened design decision, recorded here so the revert is cheap.

---

## 9. Non-goals (this ticket)

- Population test / pass criterion / operating point → **#21**.
- Which â-population is claim-bearing → **#21** (this ticket exports the
  reliability measure; it does not gate on it beyond the mechanical NaN).
- The build, the prod run, and recording numbers → **#22**.
- The paper-claim go/no-go → **#23**.
- Downstream pipeline rewiring on a GO → out of scope (map #19).
