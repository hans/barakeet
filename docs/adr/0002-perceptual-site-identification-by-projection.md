# Identify early perceptual sites by projection, not the early-window t-test

**Status:** accepted (2026-07-20)

We identify which acoustic sites carry an early within-completion perceptual
response using the **perceptual projection** (π, code `early_perceptual_projection`),
replacing the prior apparatus: an automated within-completion bootstrap t-test on
aligned percept splits per completion (the `B1`/`B2` aligned-split significance in
`early_window_site_types`), whose output a human then curated into the manual
`site_type_relabel` typology. The projection is now the canonical, reproducible
identifier of the aligned early perceptual response; the manual typology is
retained as validation ground-truth, not as the selection gate.

This is the method-level decision. Its first pipeline consequence — the
early-perceptual-window analysis swapping its entry gate to the projection — is
recorded in [ADR 0001](0001-early-perceptual-window-gate.md).

## Context

The early perceptual response was previously called by two layered steps:

1. An automated bootstrap t-test asking whether a site's within-completion
   percept split is significant *and aligned with the site's acoustic tuning*.
   The "aligned" criterion depends on a per-site `acoustic_sign` assignment — so
   the test's sign is set by the same acoustic response it is being compared to.
2. A manual relabel (`site_type_relabel`, type1–5), a neuroscientist reading the
   traces and overriding the t-test where intuition and statistics disagreed
   ("that should be an effect but the t-test misses it", and vice versa).

Two problems: step 1 is circular (induced sign), and step 2 is not reproducible
from data — it is a judgement made on the very traces the analysis then tests.

## Decision

Use π = ⟨â, p⟩: the report-driven percept contrast `p` (B4, within-completion,
pooled across completions) projected onto the site's own unit-normalized acoustic
template `â`, integrated over the early window `[50, 280] ms` (pre-POD). The
template comes from **unambiguous** trials and the test from **ambiguous** trials,
so selection and test are structurally independent; the sign falls out of the
fixed continuum polarity, so no per-site tuning assignment is needed. Significance
is a within-step label-permutation p-value. The adopted operating point is a
**type2-aligned detector**: one-tailed, π > 0 (the *projection gate*, uncorrected
p < 0.05).

## Why

- **Non-circularity.** Disjoint train/test trial sets and a polarity-fixed sign
  remove both circularities of the prior method (the induced `acoustic_sign` and
  the human-on-the-same-traces judgement).
- **Reproducibility.** A documented statistic with a permutation null regenerates
  the perceptual-site set from data; no review labels in the loop.
- **Validated against the manual labels as a detector.** At the projection gate,
  sensitivity is **7/8** manual `type2_early_perceptual` sites and the
  false-positive rate on `type1_acoustic_only` is **2/52 (~chance)**, with a
  significant population effect (CPO p = 0.0012). It reproduces the manual
  intuition for the aligned case without hand-labeling.
- **A stronger claim.** The window is entirely pre-POD, so a positive π asserts
  report-driven structure *before* the disambiguating completion arrives — a
  sharper claim than the prior whole-early-window t-test.

## Scope — what it is and is not

The projection is a **type2-aligned** detector *by design*. Because the headline
statistic pools the two completions, it reinforces splits aligned at both
completions (type2), dilutes one-sided splits (`type3_asymmetric`), and cancels
opposite-polarity splits (`type4_early_perceptual_mirrored`, whose per-completion
projections have opposite sign). For the goal — identifying the aligned early
perceptual response — this selectivity is intended, not a defect. `type4` sites
additionally carry no in-window (pre-POD) signal even per-completion; they belong
to the later/integration window studied separately. Recovering type3/type4 is
explicitly *not* a goal of this gate.

## Considered alternatives

- **Relax or modify the automated early-window t-test to reproduce the manual
  intuition.** Rejected: it re-introduces the per-site sign machinery and the
  hand-labeling the projection was built to avoid, is high-effort to render
  intuition as a defensible test, and remains circular.
- **Keep the manual `site_type_relabel` as the selection authority.** Rejected:
  not reproducible; a judgement on the tested traces. Retained instead as Test-3
  validation ground-truth and for the type1–5 taxonomy.
- **An acoustic prior gate (decoder-FDR / ‖â‖) to cut multiple-comparison
  dilution.** Rejected: type1 and type2 are acoustically indistinguishable, so an
  acoustic gate removes the targets, not the nulls — empirically the decoder-FDR
  gate keeps 17 type1 but drops 7 of 8 type2. (This is also why the inclusion
  universe is the broader `A_significant` acoustic-responsive set, not the
  narrower decoder-FDR set.)
- **Per-site BH-FDR over all acoustic sites as the headline (discovery-screen
  framing).** Deprioritized: it collapses to 3 survivors because the binding
  constraint is family size, not effect size (in an oracle 8-site type2 family,
  7/8 survive BH). The method is a *detector validated against labels*, so the
  primary report is operating characteristics (sensitivity / type1 false-positive
  rate) plus the population test; FDR is kept as the conservative population-level
  claim, not the site-identification rule.

## Consequences

- `early_perceptual_projection` (+ aggregate) is the canonical identifier of early
  perceptual sites; the projection gate is the downstream entry criterion (first
  consumer: `ep_windows`, ADR 0001).
- The manual `site_type_relabel` typology is retained for validation (Test 3) and
  for the type1–5 taxonomy, no longer as the selection gate. The automated
  `assign_site_type` classifier stays non-authoritative (cf. the retired
  `grab_bag`/`complex`/`unknown` labels).
- The **late** behaviorally-discriminative window analysis still uses the manual
  annotations; this ADR covers early perceptual identification only.
- Correctness dependency: Test-3 validation must join the manual
  `site_type_relabel` column from `manual_annotations/early_acoustic_window.csv`
  (fixed 2026-07-20; an earlier join read a stale computed vocabulary and made the
  projection appear to detect nothing relevant).
- Open: the population-headline operating point (strict FDR = 3 sites vs the
  uncorrected detector = 7/8 type2) is not yet fixed; the *gate* operating point
  (uncorrected one-tailed p < 0.05) is settled for the window consumer.

Spec: `docs/superpowers/plans/2026-07-16-early-perceptual-projection-spec.md`.
