# Early-perceptual-window gate: automated projection over manual annotation

**Status:** accepted (2026-07-20)

The early-perceptual-window analysis (`early_perceptual_windows.py`, code
`ep_windows`) previously entered cells that a human had tagged `behav @ac` in the
manual manifest. We switched the entry gate to the automated **perceptual
projection** (`early_perceptual_projection`): a site enters when its pooled
projection statistic π is valid and its uncorrected one-tailed permutation
p-value is below 0.05, and both of that site's completions then enter the window
search. The bootstrap-CI window-finding itself is unchanged — only the gate moved.

This is one consumer of the method-level decision to identify early perceptual
sites by projection rather than the early-window t-test; see
[ADR 0002](0002-perceptual-site-identification-by-projection.md) for that
decision and its evidence.

## Why

- **Non-circularity.** The projection's sign falls out of the fixed continuum
  polarity of the acoustic template, so it needs no per-site tuning assignment.
  This deliberately avoids the induced `acoustic_sign` alignment that made the
  prior sign-agreement criteria circular. The manual `behav @ac` tag is a human
  judgement made while looking at the same traces the analysis then tests.
- **Reproducibility.** The gate is now a documented statistic with a permutation
  null, not a review label, so the early-window cell set regenerates from data.
- **Adopted framing.** π at uncorrected one-tailed p<0.05 is the operating point
  already adopted as the type2 (aligned-both-completion) early-response detector
  (recovers 7/8 manual type2; ~chance type1 false positives).

## Considered alternatives

- **Two-tailed gate** — would additionally admit mirrored (type4, π<0) sites.
  Rejected: the analysis targets the aligned early perceptual response, for which
  one-tailed (π>0) is the intended selectivity.
- **FDR-significant (Test 1) gate** — stricter (~3 survivors). Rejected here: the
  gate is deliberately *uncorrected* (the aggregate applies BH-FDR separately for
  the population-level claim); this analysis wants the broader candidate set.

## Consequences

- `ep_windows.parquet` drops the `behav_ac_tuning` column and gains per-row
  `pi_pooled` and `p_one_tailed` (site-level gate values).
- The rule now depends on the per-subject `early_perceptual_projection`
  `site_results.csv` outputs rather than the manual manifest.
- Downstream, the projection-passing set now *defines* the "Perceptual" group for
  the acoustic-vs-perceptual timing comparison (no `site_type_relabel` re-filter).
- The *late* behaviorally-discriminative window analysis (`b_windows`) still uses
  the manual annotations; only the early set moved.
