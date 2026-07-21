# Define acoustic-only sites manual-free, by composing automated type1 with the projection gate

**Status:** accepted (2026-07-21)

The acoustic-only comparison group (previously the manual
`site_type_relabel == "type1_acoustic_only"`) is now derived without manual
annotation, by composing two automated statistics that already exist in the
pipeline:

```
early_response_class:
  type2_aligned  ← projection p_one_tailed < gate_alpha   (aligned early perceptual response)
  acoustic_only  ← automated site_type == "type1_acoustic_only"  AND NOT type2_aligned
  neither        ← everything else
```

Emitted as `site_class.parquet` by `early_perceptual_projection_aggregate`;
consumed by `early_perceptual_windows` as the acoustic-only group. Extends
[ADR 0002](0002-perceptual-site-identification-by-projection.md) (projection as
the aligned-perceptual identifier) to the *complementary* label.

## Context

The projection ([ADR 0002](0002-perceptual-site-identification-by-projection.md))
is a directional statistic: π = ⟨â, p⟩ projects the within-completion percept
contrast onto the acoustic template direction â. It is a good detector of the
**aligned** early perceptual response (type2), but its complement ("not aligned")
is a mixed bag — it cannot separate clean acoustic-only sites from one-sided
(type3) or mirrored/anti-aligned (type4) splits, because those live orthogonal to
â or cancel in the word-end pool. So "acoustic-only = projection complement" is
not clean.

The distinction that *is* clean — "no within-completion behavioral split on
either word end, in the early window" — is exactly what the automated
`assign_site_type` (`early_window_site_types`) already computes as
`type1_acoustic_only`, on the **same** `[45,68]`-sample window (its B1/B2
searchlight `B_SEARCH_SMIN/SMAX = ac_search_smin/smax`). The site types are fully
defined by early-window HGA; there is no window mismatch with the projection.

The automated type1 is not, on its own, a clean acoustic-only set: at these trial
counts its per-word-end bootstrap misses a few real splits, so ~2 genuine type2
sites leak in (validated against manual labels: computed type1 = 45, of which 2
are manual type2). But those are precisely the sites the projection *does* catch
as aligned. So the two automated statistics are complementary: automated type1
rejects the obviously-non-acoustic sites (`complex`/`unknown`/type3/…); the
projection subtracts the genuine perceptual (type2) sites that leaked into type1.

## Decision

`acoustic_only = (automated site_type == type1_acoustic_only) AND NOT (projection aligned)`,
with `type2_aligned` taking precedence and `neither` as the residual.

**Gate for the type2 subtraction: uncorrected `p_one_tailed < gate_alpha`
(default 0.05).** This matches the `early_perceptual_windows` operating point
([ADR 0001](0001-early-perceptual-window-gate.md)) and removes both leaked type2
sites. A `gate_mode="fdr"` option exists but is **not** a strict improvement: it
retains one borderline type1 (π p≈0.043) at the cost of re-admitting one leaked
type2 (π p≈0.0105, which does not survive cross-subject FDR) back into
acoustic-only. Uncorrected is preferred for acoustic-only purity.

Validated on prod (99 sites): `type2_aligned = 13`, `acoustic_only = 42`,
`neither = 44`. The acoustic-only set is 36 manual-type1 + a long tail (4 type3,
1 type4, 1 A_unsigned) that is accepted as acoustic-only-like; zero manual type2.

## Consequences

- No manual annotation in the acoustic-only path. `early_perceptual_windows`
  reads `site_class.parquet` instead of `manual_annotations/early_acoustic_window.csv`.
- The pipeline still depends on `early_window_site_types` (automated) for the
  type1 accept/reject and for the projection's A-significant site pool. Fully
  removing that dependency (a self-contained ‖â‖ acoustic-responsiveness gate +
  a non-directional per-word-end split test inside the projection) is deferred;
  it was not needed to make acoustic-only manual-free.
- Known limits carried by the composition: the projection's one-tailed-positive,
  A-significant-pool design does not label a *mirrored* type2 (negative π) or an
  out-of-pool type2 as `type2_aligned`; both land in `neither`, not
  `acoustic_only`, so acoustic-only purity is preserved. The automated B-power
  gate also parks ~8 genuinely-clean type1 sites in `neither` (`unknown`),
  making acoustic-only conservative rather than contaminated.
