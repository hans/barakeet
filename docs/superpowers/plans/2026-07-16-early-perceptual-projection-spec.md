# Spec: Projection-Based Detection of Perceptual Early Responses

**Status:** ready to execute (design settled via grilling 2026-07-16)
**Scope:** early response window only. Integration-response detection and its
site typing are untouched.
**Pipeline:** `causal46_joined`. New notebook + Snakefile rule; reuses existing
per-window bootstrap parquets and `_within_completion` primitives.

---

## 0. Purpose

Replace the window-based bootstrap test for early perceptual sites with a
continuous, morphology-agnostic **projection** statistic. Per site, measure how
much the report-driven contrast on ambiguous trials resembles that site's
acoustic contrast on unambiguous trials, integrated over the whole early window
rather than searched for a peak.

Template estimation uses **unambiguous** trials; testing uses **ambiguous**
trials. Disjoint trial sets ⇒ selection and test are structurally independent.

This is a **candidate replacement**, not committed. Deliver the new procedure's
output *and* a comparison against current labels so the swap is judged on
evidence.

Critically, the projection **needs no per-site sign/tuning assignment**. Its
sign falls out of the fixed continuum polarity of `â`. This deliberately avoids
the induced `acoustic_sign` alignment (`early_window_site_types.py:38`,
`t_tests.py:304`) that makes the current sign-agreement criterion circular.

---

## 1. Design decisions (resolved)

| # | Decision | Choice |
|---|----------|--------|
| Q1 | π domain | **Window grid** (reuse sliding `(smin,smax)` tiles), not fresh per-timepoint. `window_size`/`stride` parameterized so the window can be shrunk later for finer resolution. |
| Q2 | Polarity | Positive pole = **`phoneme_pair[1]`** (/n/, /m/, /b/), identical across continua. No per-continuum mapping. One runtime assertion both parquets are char1-positive; null-calibration diagnostic is the empirical backstop. |
| Q3 | Perceptual estimator | **B4** (all qualifying ambiguous steps, min_class-weighted), not single-step B3. |
| Q4 | Normalization | Headline π uses **unit-L2 `â`**, raw `p`, retain `‖a‖`. Also carry **raw ⟨a,p⟩**. (Normalization does not change any p-value — it is a per-site fixed positive scalar; it only shapes the cross-site distribution + diagnostics.) |
| Q5 | Null | **Pure within-step label permutation of a deterministic estimator** (Option A). Not the bootstrap-permutation hybrid (miscalibrated: median-observed vs single-draw null → conservative, power loss). |
| Q6 | Inclusion | **Decoder-FDR-significant acoustic sites** (causal6 acoustic decoder gate), not the A-bootstrap p<0.01. `type5_behav_only` structurally excluded (no `â`). |
| Q7 | Test-3 labels | Cross-tab vs `site_type_relabel` (type1–5). Integration/late split omitted (separate window, out of scope). **"Detected" = two-tailed |π|-significant, sign-annotated** (see §7 Test 3 — a one-tailed-positive definition would wrongly score every `type4_early_perceptual_mirrored` site, whose π is negative by construction, as a "miss"). |
| Q8 | Early window | Global `[acoustic_peak_search_smin, acoustic_peak_search_smax]` = **[45, 68] samples = [50, 280] ms** (before POD). ~4 non-overlapping tiles at 5/5. |
| Q9 | Completion | **Pooled** across word_ends = headline (min_class-weighted avg of the two within-completion contrasts). **Per-word_end** computed as first-class secondary (own π/null/p) — some sites appear at only one completion. |
| Q10 | `â` source | **Recompute deterministically from raw HGA** (mean step6 − mean step1, pooled word_ends), same estimator family as `p`; cross-check against `a_per_window_full`. |
| Q11 | Population null | CPO permutation-count null **plus** Binomial(n_sites, 0.05) analytic reference **plus** leave-one-out percentiles (remove self-threshold circularity). All three should agree. |

---

## 2. Constants & grounding (verified)

- `epoch_sfreq = 100`, `epoch_tmin = -0.4` (`src/viz_paper.py:190-191`).
  Sample `s` → time `s/100 - 0.4` s. Word onset = sample 40.
- Early window: `config.yaml` `analysis.decoding.acoustic_peak_search_smin=45`,
  `acoustic_peak_search_smax=68` → [50 ms, 280 ms], entirely **before POD**
  (dn POD = 295 ms; `src/stimuli.py POD_dict`).
- Window grid: `causal46_joined.window_size=5, stride=5` (non-overlapping tiles;
  `config.yaml:91-93`; Snakefile rules pass `C46[...]`). Grid inner product =
  true Riemann-sum inner product (no overlap over-weighting).
- `causal46_joined.min_class_k=3`.
- Polarity (`src/data.py:add_metadata_features`): `resampled` low→char0,
  high→char1 (`:44`); `categorical_acoustic_cue=1`⇔char1 (`:92-93`);
  `behavior_dummy_forced=1`⇔reported char1 (`:177-178`). Both `mean_diff_raw`
  contrasts are char1-positive by construction.
- Site types (`early_window_site_types.py:20-30`, memory
  `project_site_type_manual_authority`): manual `site_type_relabel` (type1–5)
  overrides computed `assign_site_type`; grab_bag/complex retired.
  - type1_acoustic_only — acoustic, no early split
  - type2_early_perceptual — acoustic + aligned split, both word-ends
  - type3_asymmetric — acoustic + aligned split, one word-end
  - type4_early_perceptual_mirrored — acoustic + perceptual (mirrored)
  - type5_behav_only — perceptual only, **no acoustic response**

---

## 3. Definitions

Per site = **(subject, electrode_idx, phoneme_pair)**. Continuum sign pole =
`phoneme_pair[1]` positive.

Windows `w` = non-overlapping tiles with `smin ≥ 45 & smax ≤ 68` (params
`window_size`, `stride`; default 5/5).

**Acoustic template `a(w)`** (unambiguous, pooled word_ends, all endpoint
trials):
```
a(w) = mean HGA[resampled==6, w]  −  mean HGA[resampled==1, w]
```
Normalize over the early-window tiles: `â = a / ‖a‖₂`. Retain `‖a‖`.
Cross-check against `a_per_window_full_all.parquet` (`mean_diff_raw_med`).

**Perceptual contrast `p(w)`** (ambiguous, within-completion, deterministic
B4 min_class-weighted). For each cell `(word_end, step s)` in `qualifying_steps`
(`is_ambiguous_step: min_class>2`, endpoints excluded), weight
`w_s = min_class[s] / Σ min_class`:
```
p_we(w) = Σ_s w_s · [ mean HGA[class1, we, s, w] − mean HGA[class0, we, s, w] ]
```
- **Pooled (headline):** `p(w)` = min_class-weighted average of `p_we0`, `p_we1`
  (each term within-completion ⇒ average is within-completion-valid).
- **Per-word_end (secondary):** keep `p_we0`, `p_we1` separately.

class1 = reported `phoneme_pair[1]` (`behavior_dummy_forced`, via
`resolve_behavior_col`).

**Projection** `π = ⟨â, p⟩ = Σ_w â(w)·p(w)`. Positive ⇒ report-driven contrast
shares acoustic contrast's shape *and* sign. Compute for pooled `p` (headline)
and each `p_we` (secondary). Also record raw `⟨a, p⟩` (un-normalized).

---

## 4. Data sources

- **Raw:** `outputs/epochs_preprocessed/{subject}_epo.fif`, metadata via
  `src.data.add_metadata_features`. HGA extraction + `per_step_class_counts`
  from `notebooks/causal46_joined/_within_completion.py`; windowed contrasts via
  `searchlight_mean_diff`.
- **Inclusion gate (RESOLVED):** `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet`
  IS the FDR-aggregated table (rule `acoustic_decoding_peaks_aggregate` →
  `notebooks/causal6/significance_aggregate.py`). It carries a per-**(subject,
  electrode_idx, phoneme_pair)** boolean **`significant`** (+ `q_value`) plus the
  peak window. Inclusion = `phon_peaks_all.filter(significant == True)` — a
  single-column join, keys matching the projection's site unit.
  - Procedure behind `significant` (for the record): when `analysis.fdr_rois`
    is set, hierarchical — electrode-level Simes p then BH across the ROI family
    (`electrode_significant`), then Holm within each significant electrode across
    phoneme_pairs → final `significant` (`significance_aggregate.py:82–130`).
    When no ROI restriction: plain BH-FDR at `fdr_alpha` (`:134–141`).
- **Cross-check only:** `outputs/causal46_joined/acoustic_bootstrap/a_per_window_full_all.parquet`.
- **Test 3 labels:** `site_type_relabel` (from
  `early_window_site_types_aggregate_figures.py` → `site_type_relabel.csv`).

---

## 5. Site inclusion

Include every **decoder-FDR-significant acoustic site** —
`phon_peaks_all.filter(significant == True)` (§4) — (unambiguous-trial criterion;
independent of report/perceptual/sign) with a valid B4 cell. In practice types
1–4. Do **not** restrict to type1; do **not** filter on perceptual significance
or sign.

Record per site: n at step 1, step 6, and per qualifying (word_end, step) report
cell. Flag + exclude cells failing `min_class_k ≥ 3`.

Report **two exclusion tallies**: (a) no-acoustic-template / type5, (b)
`min_class_k < 3`.

---

## 6. Estimator + permutation null (per site, independent)

1. Compute `â` (fixed; unambiguous trials carry no shuffled labels).
2. Compute observed `p` (pooled) and `p_we0/p_we1` → observed `π`, `π_we*`.
3. **Null:** for each of N permutations, **shuffle report labels independently
   within each (word_end, step) cell**, preserving that cell's observed
   `(n1, n0)` split (permute, do **not** resample), recompute `p` (and `p_we*`)
   the *same deterministic way*, project against fixed `â` → `π_perm`.
   - Observed and null are the identical procedure ⇒ p uniform under H0.
4. **N:** target 10,000. Permutation space per cell = `Π_s C(n_s, n1_s)`; the
   product across cells is generally ≫ 10,000 under B4, so sample N. **Enumerate
   exhaustively** only when the site's total space < N; record achievable
   resolution (smallest possible p) per site and an exhaustive-vs-sampled flag.
5. **Seed:** fixed master seed; **independent per-site RNG streams** (master +
   site offset). Record seed.

Per-site p-values: one-tailed (positive direction) and two-tailed, from the
per-site null.

---

## 7. Tests

### Test 1 — per-site significance
- BH-FDR (q=0.05) across included sites on **one-tailed** p → surviving count +
  list = **candidate perceptual-site list**.
- Separately, BH-FDR on **two-tailed** p → report count of significantly
  **negative** π sites (report tracking opposite to acoustic tuning). Never
  silently dropped. The two-tailed-significant set (either sign) is what Test 3
  calls "detected."
- **Per-word_end p-values are descriptive** — reported alongside but corrected
  separately (their own BH pass), NOT folded into the primary per-site FDR
  family (which is the pooled statistic, one test per site).

### Test 2 — population count (CPO)
- Observed = #sites with one-tailed perm p < 0.05 (uncorrected).
- Null (matched): over permutation index i, count sites whose `π_perm[i]`
  exceeds that site's **leave-one-out** 95th null percentile → distribution over
  i → p.
- Analytic reference: observed vs **Binomial(n_sites, 0.05)**.
- Report observed, both nulls, both p. They should agree.
- Independence assumption noted: independent per-site shuffles (no shared
  cross-site permutation).

### Test 3 — comparison vs current labels
**"Detected" = two-tailed |π|-FDR-significant, annotated by sign** (positive =
report tracks acoustic tuning ≈ reactivation candidate; negative = mirrored,
the expected signature of `type4_early_perceptual_mirrored`). A one-tailed
positive definition would score every type4 site (π<0 by construction) as a
false "miss" — so detection must be two-tailed with sign as annotation, while
the one-tailed positive list (Test 1) remains the narrower reactivation
sub-question.

Cross-tab detected(±) × `site_type_relabel` (type1–5). Report:
- **type1_acoustic_only that the projection detects** (motivating case).
- **perceptual (type2/3/4) the projection misses** (see window caveat below).
- **type4 sites: expect detection with negative sign** — verify they land in the
  detected-negative cell, not the miss cell.
- Agreement rate.
- type5 reported separately (out of scope, no `â`).

**Window-mismatch caveat (interpretation).** Current labels come from a *wider*
window: `early_window_site_types.py` runs its A/B bootstraps over
`[word_onset, min(word_offset, 1.3 s)]`, whereas the projection sees only
`[45,68]` = [50,280] ms (the acoustic **peak-search** range, *not* the
perceptual-typing window). So a type2/3 the projection *misses* may reflect
perceptual signal after 280 ms (window truncation), not a genuine method
disagreement. State this wherever misses are interpreted.

For every disagreement site: trace plots (§9).

---

## 8. Diagnostics (weighted equally with the tests)

- **Null calibration:** per-site null π centered ~0; flag any site whose null
  mean departs from 0 (bug / label-shuffle confound).
- **Trial-count dependence:** per-site null SD vs qualifying-step trial count;
  expect ~1/√n. Deviation ⇒ wrong noise model.
- **Template magnitude:** observed π vs `‖a‖`. Flag if positive tail concentrates
  at low `‖a‖` (normalization artifact). Direct, since raw + normalized both
  carried.
- **Tail symmetry:** observed π across sites as **strip/histogram with visible
  points (NOT KDE)**; overlay pooled null *for display only* — code comment must
  state all inference uses per-site nulls.
- **Skew:** skewness of observed π vs skewness of pooled permuted. One-sided tail
  = signal; symmetric = noise.

**HGA-scale caveat.** HGA is baseline-corrected but **not z-scored across
electrodes** (`_within_completion.py:196,225`), so raw `p` (and hence the
cross-site π *magnitude*) carries per-electrode amplitude scale. This affects
only the cross-site distribution/skew *display* — every **per-site p-value is
unaffected** (the fixed-`â` scalar and the within-site null cancel scale). If
the cross-site distribution is to be read quantitatively, note the mixed scale
or add a scale-normalized companion (e.g. π / per-site HGA SD) for display only.

---

## 9. Plots

- π distribution strip/histogram (per-site nulls for inference).
- Diagnostic scatters (§8): null-SD vs n, π vs ‖a‖.
- **Disagreement traces:** per disagreement site, HGA traces (unambiguous
  step1/step6; ambiguous both reports) with `â` and `p(w)` overlaid. Reuse
  existing trace/contrast machinery (`contrast_plot.py` / `_star_gallery.py`).

---

## 10. Deliverables

1. Per-site results table (CSV) — cols per §6 + pooled and per-word_end
   π/null/p + ‖a‖ + raw ⟨a,p⟩ + window bounds + trial counts + n_perms +
   exhaustive flag.
2. Test 1, 2, 3 outputs (numbers above).
3. Diagnostic plots (§8).
4. π distribution plot (§9).
5. Disagreement-site trace plots (§9).
6. Short summary: does the projection show a population-level effect; how many
   sites survive; how the list differs from current labels.

---

## 11. Code structure

- New notebook `notebooks/causal46_joined/early_perceptual_projection.py`
  (Jupytext percent; plain `x = ...` params, no annotations — ploomber).
- Imports `_within_completion` primitives; new shared logic (deterministic
  weighted contrast, projection, per-site permutation) extracted to a helper if
  reused, else notebook-local.
- New Snakefile rule → `outputs/causal46_joined/early_perceptual_projection/`.
  Every `tags=["parameters"]` param mirrored in the rule's
  `run_notebook(parameters=dict(...))`.
- Verify with `CONFIG_FILE=config.smoke.yaml uv run snakemake --configfile
  config.smoke.yaml -j1` (smoke early window is [0,290]; expect coarse/looser).
- All Python via `uv run`; never concurrent `uv run`.

---

## 12. What not to do

- No window selection from ambiguous trials.
- No using observed π to define site types then testing those types on the same
  data.
- No KDE for π inference.
- No dropping sign-mismatched (negative-π) sites.
- No `mean_diff_aligned*` (induced per-site sign flip) anywhere.

---

## 13. Future (noted, not now)

- Shrink `window_size`/`stride` (< 5) for finer early-window resolution — one
  param change; the ~4-tile coarseness is the main current limitation.
