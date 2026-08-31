# Late perceptual projection, gated on endpoint persistence

**Status:** prototype (scratch site-set done; full projection notebook ready for a prod run)
**Date:** 2026-08-31

## Motivation

The current late perceptual projection (`late_perceptual_projection.py`) has two
soft spots:

1. **Gate is at chance.** It gates each cell on a *behavioral* discriminative run
   (`b_windows`, ambiguous-trial /n/-/d/ contrast). That statistic sits at chance
   at the population level (`late_perceptual_significance`: 6/187 uncorrected ≈
   the 9.35 expected at 5%, 0 after BH-FDR, count-vs-null binom p = 0.91).
2. **Mild circularity.** It selects windows where the perceptual contrast `p` is
   large, then runs a test that holds that same `p` fixed.

Also, `â` is currently measured *in the late behavioral window*, where the
endpoint acoustic response has largely decayed (STG acoustic responses are
transient, peak 150–250 ms) — a weak, noisy template.

## Design: gate on late endpoint persistence

Make late **parallel to the early perceptual projection**: pool = acoustic-
reliable sites (`A_significant`), gate = **endpoint reliability** (not middle
trials), projection `π = ⟨p, â⟩` in the gated window, acoustic-label-shuffle null.

- **Gate:** maximal same-sign run of significant step6−step1 (endpoint,
  unambiguous) HGA windows, via the shared `_find_maximal_runs`. Post-POD the two
  endpoints are acoustically *identical* within a completion, so a late endpoint
  contrast is **neural persistence of the acoustic code**, not a stimulus
  difference. Endpoints have many trials → clean low-noise `â`.
- **Per-cell floor = disappearance of the initial acoustic response.** The run
  floor `smin` is set per cell by `find_early_offset_smin` (shared helper
  `_acoustic_offset.py`, extracted from `acoustic_late.py`): the first window
  at/after `phon_smax` where the endpoint contrast returns to non-significance. A
  late run must therefore be a **re-emergence** of the endpoint contrast after
  the initial response has diminished — not the sustained tail of a multi-phase
  early response leaking just past `phon_smax`. Cells whose contrast never
  returns to non-significance past `phon_smax` (no dissociable late region) are
  dropped and reported.
- **Runs built per word_end.** The endpoint bootstrap is read from
  `a_per_window_by_word_end_all` (per completion), so the two completions can
  carry **different** late endpoint windows — post-POD they diverge acoustically,
  and the within-completion projection is per word_end anyway. `phon_smax` stays
  per-site (pre-lexical); the offset is recomputed per word_end.
- **Geometry is clean:** `â` and `p` are computed over the *same* late run
  window, so `⟨p, â⟩` is well defined (no cross-window template-transfer needed).
- **Selection is honest:** the gate uses only endpoint trials, independent of the
  perceptual contrast `p` the projection tests. Removes the circularity above.
- **Only the gate changes.** The projection machinery (raw `⟨p, â⟩`, per-window
  null, max-over-window, BH-FDR) is byte-for-byte the current late one, so the
  gate source is the single manipulated variable.

Data source for late endpoint runs: `a_per_window_full_all.parquet` (endpoint
per-window bootstrap over the **full** epoch, not the `phon_smax`-capped
`acoustic_endpoint_windows`).

## Scratch site-set result (no epochs; gate only)

`scratchpad/late_endpoint_gate_search.py`, per-cell offset floor,
`ci_raw_excludes_zero`, run ≥ 2 windows:

| | current (middle-gated) | endpoint offset floor, pooled | endpoint offset floor, **per word_end** |
|---|---|---|---|
| Sites | 31 | 47 | **76** |
| Cells (site×word_end) | 38 | — | **114** |
| New sites vs current | — | 30 | **49** |
| Recovers current 3 FDR | — | 2/3 | **3/3** |
| Site classes | mostly type2 | 21 nei + 20 aco + 6 t2 | 32 nei + 35 aco + 9 t2 |

- 165 late endpoint runs; onsets span 0.26–1.04 s. 0 cells had a fully-sustained
  contrast (all dissociable).
- **38 sites are gated in both completions, and in all 38 the two late windows
  differ** — the reason for reading the endpoint bootstrap per word_end.
- Per word_end is more permissive at the site level (a run in *either* completion
  qualifies the cell), so it recovers all 3 current FDR sites where the pooled
  offset floor recovered 2/3.
- Reaches 32 `neither` + 35 `acoustic_only` sites the early perceptual detector
  ignores — a broader/different population.
- **Still endpoint-locked:** this does NOT address ambiguous-only generators
  (sites with no endpoint response at all). That remains Option 1 (bootstrap on
  all sites), a separate exploratory track.

## Open questions for the prod run

- How many of the 67 gated sites have qualifying ambiguous steps (min_class_k=3)
  → a testable `p`? (67 is an upper bound on the testable set.)
- How does the projection fare on the new/`neither`/`acoustic_only` sites vs the
  type2 ones — does endpoint persistence without an early perceptual response
  still carry a late perceptual projection?
- Knobs deliberately held at the current-late defaults (raw unnormalized `â`;
  `min_component_windows=2`). Normalizing `â` (as early does) is an available
  follow-up.

## Artifacts

- `notebooks/causal46_joined/late_endpoint_projection.py` — full projection
  notebook (parallel to `late_perceptual_projection.py`; **not yet wired into the
  Snakefile** — run manually on a machine with epochs).
- `scratchpad/late_endpoint_gate_search.py` — gate-only scratch (runs off
  bootstrap parquets, no epochs).
