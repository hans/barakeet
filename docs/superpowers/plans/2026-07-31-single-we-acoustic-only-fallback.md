# Spec: acoustic-only fallback for single-powered-word-end sites

## Problem
`assign_site_type` (`notebooks/causal46_joined/early_window_site_types.py:466-468`)
forces `"unknown"` (status `unclassifiable_B_power`) whenever *either* word-end
lacks a powered B-cell (insufficient qualifying ambiguous-step trials), even
when the other word-end is powered and shows a clean, non-significant-on-B
pattern (i.e. would independently satisfy the `type1_acoustic_only` criterion:
`not b1_any and not b2_any`). This drops such sites to `early_response_class
== "neither"` downstream, excluding them from analyses that key on
`acoustic_only`/`type2_aligned` (e.g. `acoustic_late.py`'s `to_study` pool).

Concrete case: EC250 electrode 215, phoneme_pair `dn`. `necessary` (we1) has
zero qualifying cells (`n_qualifying_cells_we1 = 0`). `desolate` (we0) is
powered, strongly acoustically tuned (`a_norm = 4.71`), and shows no
significant B-aligned/anti effect (`p_one_tailed_we0 = 0.4757`,
`p_two_tailed_we0 = 0.9487`). Currently labeled `"unknown"` → `"neither"`.

## Proposed rule
When exactly one of `B1_powered`/`B2_powered` is true, classify from the
powered side alone instead of forcing `"unknown"`:
- powered side shows no aligned/anti significance → `type1_acoustic_only`
  (site's principle: "not clearly type2 or other type on the observable
  word-end" ⇒ acoustic-only).
- powered side shows aligned/anti significance → fall through to the existing
  single-sided branch (`type3_asymmetric`), same as today's bilateral case.

Both-unpowered stays `"unknown"` (no evidence at all).

## Resolutions (grilled 2026-07-31)

**Scope of effect (measured against live prod output, 99 sites):** exactly 9
sites hit the old `unclassifiable_B_power` branch, and all 9 are
single-powered-word-end (there are currently *zero* both-unpowered sites, so the
"both-unpowered stays unknown" clause is vacuous today). All 9 have a
non-significant powered side → all become `type1_acoustic_only` (none go
type3). Identities: EC250 e185/191/195/206/207/215 (`dn`), EC279 e40/e170
(`pb`), EC282 e98 (`pb`) — **6/9 load onto EC250 `dn`**, 0 on `bm`.

1. **Distinct but included, via encoding (A):** keep the plain
   `type1_acoustic_only` label (matches the manual authority, which *already*
   annotates all 9 as `type1_acoustic_only`) and add a `single_we_fallback`
   boolean column for separability. No new `site_type` string — separability is
   also recoverable from `B1_n_per_class`/`B2_n_per_class`. This keeps the diff
   to one function and leaves the aggregate's exact-match
   `== "type1_acoustic_only"` (→ `early_response_class == "acoustic_only"`)
   untouched.
2. **Positive call is acceptable here:** a human already made it — the manual
   authority file (`early_acoustic_window.csv`) labels all 9
   `type1_acoustic_only`. The `single_we_fallback` flag preserves the
   epistemic distinction (one observed non-significance vs. two) for any later
   sensitivity check.
3. **No new manual review pass needed** — the manual annotation already exists
   and already agrees. Per [[project_site_type_manual_authority]], manual wins
   for the manual-path consumers, so they are unaffected by this change.
4. **Test 1/2/3 are unaffected.** Test 1/2 run on `valid_df` (non-null
   `pi_pooled`), not the computed `acoustic_only` pool; Test 3 keys on the
   manual `site_type_relabel` (already type1). The change touches *only* the
   manual-free `early_response_class == "acoustic_only"` pool feeding
   `acoustic_late` (`to_study`), `acoustic_bootstrap`,
   `early_perceptual_windows`, and `late_perceptual_projection` (`!= "neither"`).
5. **Confound noted, not instrumented (land-bare):** 6/9 fallback sites are one
   subject/pair (EC250 `dn`), reflecting a trial-starved completion cell rather
   than a random draw. Benign for an *acoustic* call, but the cluster could
   swing a late-projection population count. Chosen to watch manually rather
   than wire a with/without report.

## Implementation (landed)
`notebooks/causal46_joined/early_window_site_types.py`:
- `assign_site_type` B-power gate: `not B1_powered or not B2_powered` →
  `not B1_powered and not B2_powered`. Single-powered sites now resolve through
  the existing pattern logic (unpowered side contributes all-False flags).
- Added `single_we_fallback = bool(r0["powered"]) != bool(r1["powered"])` to the
  emitted site-type rows (and `False` in the two early-exit rows for column
  consistency).
