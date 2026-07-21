# Late-window perceptual projection: per-cell statistic by â-anchored contiguous reliable run

**Status:** proposed (2026-07-21) — flips to accepted on the build + prod run (#22)

Port the perceptual projection (π = ⟨â, p⟩, ADR-0002) from the early window to the
**late (integration) window**, to test whether the within-completion late percept
contrast re-expresses **that same word-end's own** unambiguous /d/–/n/ tuning. This
ADR records the **per-cell statistic** design decision; the population test and the
pass criterion are decided separately (issue #21). Map: #19. Full spec:
`docs/superpowers/plans/2026-07-21-late-perceptual-projection-percell-spec.md`.

## Context

The early projection (ADR-0002) integrates π = ⟨â, p⟩ over a *fixed, narrow*
pre-POD grid, pooling â across word_ends and π across completions. Three
properties of the **late** window make that reduction wrong to port mechanically:

1. **Word-end-specific timing.** Offsets differ (desolate 0.498 s, necessary
   0.887 s) and the response extends past offset — there is no single global grid.
2. **Narrow â–p alignment.** In the late window the unambiguous and ambiguous
   responses align only over a short band; elsewhere each varies idiosyncratically.
   Whole-grid integration would bury a narrow real alignment under non-aligned
   windows' variance.
3. **Small per-cell endpoint N.** Strict same-word-end (no pooling) leaves few
   endpoint trials, so a near-noise â gets **unit-normalized up** — the "percept
   projected onto noise" failure the map's spine is about.

The binding constraint throughout is ADR-0002's non-circularity guarantee: â
(unambiguous trials) and p (ambiguous trials) must stay structurally independent,
sign fixed by continuum polarity. Any window rule that selects on **p** breaks the
plain label-permutation null unless the null re-does the selection.

## Decision

**Cell** = (subject, electrode_idx, phoneme_pair, word_end); strict
same-word-end, no pooling. **Grid** = the `b4_bootstrap` late searchlight windows
(already word-end-anchored: acoustic-peak → offset + tail).

**Window rule (1c — â-anchored contiguous reliable run):**
mark each window reliable if its β_unamb bootstrap CI excludes 0
(`beta_summary`); **anchor** at `argmax|median β_unamb|`; take the **maximal
contiguous run of reliable windows containing the anchor**; **unit-L2-normalize**
â over that run; integrate

    π_anchored = ⟨â_unit, p⟩  over the run

as the **claim statistic** (NaN where no reliable window exists). p is the
**deterministic** min-class-weighted within-step percept contrast
(`compute_p_we`, one word_end) — **not** the `b4_bootstrap` replicate
distribution. The **null** is the within-step percept-label permutation with
â_unit and the window-set held fixed, recomputing deterministic p per replicate,
per cell, not pooled; both one- and two-tailed p exported.

Also exported for the population/claim ticket (#21): **π_peak** (single
argmax|â| window, reliability-ignored — diagnostic, not the claim) and the
â-reliability descriptors (`n_reliable_windows`, run span, peak β_unamb median +
CI, ‖â_raw‖).

**Population test + pre-registered pass criterion (#21, locked 2026-07-21 — a
confirmatory pre-registration fixed before the prod run #22).** Claim-bearing
population = the **â-reliable** cells (non-empty reliable run; `π_anchored`
non-NaN, ~34/187), claim statistic = **`π_anchored`**; the â-estimable/`π_peak`
set stays diagnostic. Aggregation unit = the **cell** (site × word_end), each word
end independent; per-site / word-end-asymmetry reported descriptively only.
Population statistic = **CPO count-vs-permutation-null** (early aggregate Test 2):
observed = cells passing the per-cell one-tailed p < 0.05 gate; matched
permutation null of that count; `p_cpo` = fraction of null counts ≥ observed.

**Pre-registered rule:** **GO** iff `p_cpo < 0.05`, one-tailed (π > 0) — no
minimum-count floor; Binomial(N, 0.05) and BH-FDR reported as references, neither
gates. **NO-GO** = `p_cpo ≥ 0.05` → integration section retreats to negative
claims only. A wrong-sign (π < 0) excess is never a GO. The population/statistic
pairing is a **mutable design choice**, locked for this run only.

**Claim licensing:** a positive π licenses *only* context-gated reactivation of
the **perceptual** code **along the acoustic-tuning direction** (mechanism-1,
tuning-direction sense) — via (1) within-completion p ⇒ percept-not-acoustic and
(2) ⟨â_unit, p⟩ ⇒ re-expresses the word-end's own tuning axis — **not** "same full
code" or "single population." Reconciliation with the mechanism framing is
deferred to the write-up (map #19), not decided here.

## Why

- **Localizes without breaking independence.** â-anchoring uses only unambiguous
  trials to place the window, so the percept-label permutation null stays plain —
  nothing to absorb, no optimism. Localization answers the narrow-alignment fact
  (context 2) at zero cost to ADR-0002's non-circularity.
- **Tests the tuning where the tuning lives.** "Reactivation of the acoustic
  *tuning*" is a claim about the response *at its locus* — the â-reliable band —
  not about coincident sign at an arbitrary bin.
- **Unit-normalization made safe.** Restricting to CI-excludes-0 windows means
  â_unit is a *reliable* direction; the reliability gate, not the normalization,
  does the noise-rejection — so early's unit-normalized statistic ports without
  reintroducing the "projection onto noise" failure.
- **Keeps #21's decision real.** The mechanical NaN restricts the tested universe
  to â-reliable cells; exporting π_peak + the reliability descriptors preserves
  the reliable-vs-all comparison (the map's spine) so #21 still chooses the
  claim-bearing population rather than inheriting it silently.

## Scope — what this decides and what it does not

Decides the **per-cell statistic** (window rule, â normalization, p error model,
per-cell null; #20) **and** the **population test + pre-registered pass criterion**
(claim-bearing population, aggregation unit, CPO statistic, go/no-go rule, claim
licensing; #21). Does **not** decide the build/prod run (#22), the mechanical
*application* of the criterion to the prod result (#23), or the write-up +
reconciliation with the mechanism framing (map #19, graduates on GO).

## Considered alternatives

- **Whole-grid integration (early's 1a reduction).** Rejected: over the wide late
  grid it dilutes a narrow alignment with non-aligned windows' variance.
- **Snap to `b_windows` / `peak_beta_amb`.** Rejected: selects the window on **p**,
  so the plain null is optimistic (`strong_generator_scan` already flags
  `peak_beta_amb` as circular); would need a search-absorbing null.
- **Bootstrap-CI error model on p (`strong_generator_scan`'s model, p from
  `b4_bootstrap`).** Rejected as the *inference* engine: it is not a
  label-permutation null and does not port ADR-0002's non-circular test. Retained
  only as a cross-check on the observed p.
- **Searchlight π-search with a search-absorbing null (1b).** Not adopted, but
  **recorded as the fallback** (spec §8): revert to it if 1c under-detects
  (alignments off the â peak, or the reliable-run localization too conservative).
  It localizes to alignment wherever it sits, at the cost of a heavier null and
  power spent on the search correction, and it re-selects partly on p — so
  adopting it is a re-opened design decision, not a drop-in.

## Consequences

- New notebook `late_perceptual_projection` (+ aggregate) in `causal46_joined`,
  reusing `_within_completion` primitives and the `b4_bootstrap` grid; prod-only
  inputs (epochs + `b4_bootstrap`), run serially under `uv run` on prod (#22).
- On a positive result, a follow-up ADR documents the *validated* late-projection
  method and its place in the mechanism framing (mechanism 1, reactivation of the
  **perceptual** code — distinct from transfer bimodality, which is code
  inconsistency *between windows*); that reconciliation is charted on GO in map
  #19, not here.
- Status flips to **accepted** once the build runs on prod and the statistic
  behaves as specified (#22).

Spec: `docs/superpowers/plans/2026-07-21-late-perceptual-projection-percell-spec.md`.
