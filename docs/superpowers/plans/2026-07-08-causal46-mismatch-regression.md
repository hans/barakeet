# Spec: single-trial acoustic×percept mismatch regression (opponent / conflict coding)

**Date:** 2026-07-08
**Status:** spec — ready to implement
**Deliverable:** `notebooks/causal46_joined/mismatch_regression.py` (jupytext percent) + Snakefile rule `joined_mismatch_regression`
**Plan pointer:** this file

---

## 0. Figure set & scope (READ FIRST — this is the deliverable)

Two tiers. **Tier 1 is deliverable now from existing parquets — no new modeling.**
Tier 2 is the regression in this spec and may slip without breaking the story.

**Tier 1 — Opponent organization (headline; existing parquets only):**
- **F1 — Example sites.** `B(t)` and `A(t)` overlaid (raw /n/−/d/ convention) for
  2–3 sites (e102 bm mountains = clean/co-significant; e101 dn necessary = temporally
  offset; +1), significant windows shaded. Shows the mirror directly.
  *Source: `b4_per_window`, `b4_acoustic_per_window`.*
- **F2 — Population.** Histogram of per-cell **corr(B,A)** over significance-restricted
  windows (median −0.43, 66% negative), matched vs mirrored annotated. The "opponent
  is the dominant motif" claim. *Existing parquets.*
- **F3 — Where the acoustic effects live.** Timing: distribution of
  acoustic-significant window centers vs perceptual window centers vs the transient
  endpoint acoustic peak (≈0.13–0.28 s). Answers "where do acoustic effects live" and
  shows the mirroring signal is **late / post-transient**.
  *Source: `b4_acoustic_per_window`, `b4_per_window`, `phon_peaks_all`.*

**Tier 2 — Is it actually an interaction? (this regression; riskier):**
- **F4 — Example conflict profile.** (step × percept) HGA heatmap + within-step
  percept-contrast-vs-step line for e102; the significant negative interaction. One
  compelling single-cell figure.
- **F5 — Population interaction.** Histogram of `β_interaction` + sign test + count
  significant (FDR): of the opponent sites, how many are **additive** vs carry a
  **conflict interaction**.

**Scope / rabbit-hole guard.** F1–F3 are a complete, defensible deliverable on their
own ("AS sites show opponent organization; the ambiguous-trial acoustic-cue effect is
late and anti-correlated with the perceptual effect"). **Do F1–F3 first and ship.**
Timebox Tier 2 to F4+F5 only; if the interaction is null/heterogeneous, *that is the
result* and Tier 1 stands unchanged (the binary-vs-graded question simply stays open).
Do **not** expand Tier 2 into open-ended single-trial modeling (RT covariates,
per-trial priors, hierarchical models) for this delivery.

---

## 1. Motivation — how we got here

Recent work restored the **acoustic-step panel** in the combined star-plot gallery
(`acoustic_on_ambiguous/star_plots_both/*.pdf`; the panel had silently never
rendered — see `2026-07-07-causal46-acoustic-discriminative-windows.md` and the
`_star_gallery.py` fix). With that panel visible, an obvious and pervasive pattern
emerged: at many AS sites the **within-completion perceptual contrast** and the
**behavior-controlled acoustic-step contrast** run in **opposite directions**
("mirroring").

Two per-cell, time-resolved contrasts, both on **ambiguous trials**, expressed in a
common **raw /n/−/d/ (nasal−stop) convention**:

- **B — perceptual** (`b4_per_window`, middle star panel): `resp=1 − resp=0`
  (heard-nasal − heard-stop). Within-completion **and step-balanced**, so acoustics
  are matched → isolates the reported percept.
- **A — acoustic-step** (`b4_acoustic_per_window`, bottom star panel):
  `s_hi − s_lo` (more-nasal-like step − more-stop-like step), **behavior-balanced
  50/50 per step** → isolates the acoustic cue independent of percept.

**What we established (validation, already done):**

- The mirroring is real in raw-vs-raw convention. Examples: EC243 e102 bm mountains
  whole-trace corr(B,A) = −0.85 (with a window significant & opposite on *both*
  sides); EC243 e101 dn necessary corr = −0.61 (temporally offset).
- Population (97 cells with both contrasts): all-window mean corr −0.34 / 85%
  negative **overstates** it (whole-epoch co-drift); **significance-restricted:
  median −0.43, 66% negative** (n=32 cells). Opponent tendency is dominant but
  heterogeneous, not universal.
- The mirroring is **late** (0.4–0.8 s), well after the transient sensory acoustic
  peak (endpoint/causal6 window ≈ 0.13–0.28 s), so it is not a passive sensory tail.

**Why it is not a design artifact (important — this was the key worry):**

- B and A are the two **orthogonal main effects** of the (step × percept) layout.
  Their contrast-weight vectors are orthogonal (inner product 0, verified for the
  full qualifying-step set), and the bootstrap **equalizes cell counts**, so under
  noise the two estimators are *uncorrelated*, not anticorrelated. Anticorrelation
  requires the neural response to genuinely depend on step and percept with opposite
  signs.
- Every plausible sampling confound points the **other** way: on ambiguous trials
  percept and step are positively correlated, so imperfect balancing / shared gain
  would bias toward **matched (positive)**. The observed mirroring survives against
  the confound and is if anything underestimated.
- **Do not** orient by the endpoint acoustic response: at endpoints percept ≡
  acoustics (participants report the matching percept), so endpoint tuning conflates
  the two. (`contrast_plot.py` orients its acoustic panel by `acoustic_sign_endpoint`
  — fine for its out-of-sample display purpose, but not a clean reference for the
  mirroring question.)

**Interpretation forks (what the mirroring alone cannot decide):**
"Opponent organization" is produced *equally* by
1. a **single unit computing belief − evidence** (signed prediction-error /
   conflict), or
2. **two co-located, oppositely-tuned populations** (an acoustic-cue population +
   a perceptual population) summed at one contact — the SUPPORTED "local
   disambiguation" account.
A **signed** `(p − a)` unit is *additive* (opposite main effects, **no interaction**)
and is therefore indistinguishable from (2) by main effects alone.

**This analysis exists to add the one discriminating measurement**: does HGA carry a
**nonlinear step×percept interaction** — the signature of an *unsigned* conflict /
surprisal signal — beyond the additive opponent structure? This is the concrete,
partial handle on CLAUDE.md's open question (binary mechanism vs graded
belief-updating / surprisal), which the pooled contrasts cannot address.

**Proof-of-concept already run (single cells, exploratory):**
- e102 bm mountains (steps 3,4,5, window t=0.40–0.50, n=72): additive
  `β_step=+0.40 (p=.018)`, `β_percept=−0.63 (p=.022)` (opposite, both sig →
  single-trial mirroring confirmed); **`β_interaction=−0.80 (p=.015)`**, R²
  0.11→0.18. HGA peaks at *step5 + heard-stop* (max incongruence): a directional
  conflict profile. → candidate graded-conflict site.
- e101 dn necessary (steps 2,3,4, window t=0.70–0.80, n=72): `β_step=−0.41
  (p=.004)`, `β_percept=+0.17 (p=.45 ns)`, `β_interaction=−0.25 (p=.36 ns)`, R²
  0.13→0.14. Late window dominated by acoustics; percept effect and interaction
  absent here → its mirror is **temporal** (offset windows), more consistent with
  distinct processes than a single conflict unit.

The check **separates cells**. The population version below quantifies how many
sites carry the interaction signature.

---

## 2. The measurement

For each qualifying cell `(subject, electrode_idx, phoneme_pair, word_end)`, on
ambiguous trials with both percepts, fit single-trial windowed HGA:

```
additive:  HGA ~ step_c + percept_c
full:      HGA ~ step_c + percept_c + step_c:percept_c
```

- `step_c` = `resampled` centered on the mean of the included steps. Higher =
  toward the **step-6 endpoint** (dn: /n/, bm: /m/, pb: /b/).
- `percept_c` = `behavior_dummy_forced − 0.5`. Class **1 = the step-6-endpoint
  percept** (verify per pair: modal `behavior_dummy_forced` at step 6 must be 1;
  assert, else flip/skip).
- Report per cell: `β_step, β_percept, β_interaction` with SEs and p-values;
  `r2_add, r2_full, ΔR²`; nested-model F-test p (statsmodels `anova_lm`); n, and
  per-(step×percept) counts.

**Reading the coefficients:**
- `sign(β_step) ≠ sign(β_percept)` with both significant → single-trial mirroring
  (confirms the pooled result at trial level).
- **`β_interaction` is the diagnostic.** ≈0 → additive opponent (mechanisms 1-signed
  and 2 indistinguishable). Significant (expected **negative**: percept contrast
  declines as acoustics move toward the step-6 endpoint) → nonlinear conflict /
  surprisal beyond additive coding.

---

## 3. Cell & window selection (anti-double-dip — non-negotiable)

**Qualifying cells:** intersection of
- **acoustic-significant**: cell in `b4_acoustic_per_cell` with
  `best_ci_aligned_excludes_zero == True` (i.e. present in `powered_significant`);
- **behaviorally responsive**: cell has ≥1 significant behavioral window
  (`b4_per_window.ci_raw_excludes_zero`) OR appears in
  `behavioral_discriminative_windows/b_windows.parquet`;
- **trial structure**: ≥ **2** (prefer ≥3) ambiguous steps each with ≥ `K` (default
  **4**) trials in *both* percepts. Ambiguous steps from
  `trial_balance_index.csv` (`is_ambiguous_step`); require both percepts present.

**Analysis window — select on the ACOUSTIC contrast, not the percept:** use the
acoustic best window (`b4_acoustic_per_cell.best_smin/best_smax`) — or the
acoustic-significant window nearest the percept response. Justification: step
(acoustic) and percept main effects and their interaction are mutually **orthogonal**
under the balanced layout, so selecting on the acoustic main effect does **not**
bias `β_percept` or `β_interaction` (it *does* inflate `β_step` — do not interpret
`β_step` magnitude from selected windows). One window per cell (no window-level
multiple comparisons).

**Robustness variant (required):** re-run with an *a priori fixed* window
(e.g. `[POD, word_offset]` per `POD_dict`/`OFFSET_DICT`) and confirm the interaction
sign/rate is not an artifact of acoustic-window selection. Optional: time-resolved
(per grid window) interaction time course for the example cells.

---

## 4. Data sources & helpers

- Epochs: `outputs/epochs_preprocessed/{subject}_epo.fif` → `mne.read_epochs(...,
  preload=False)`, then `ep.metadata = add_metadata_features(ep.metadata.copy())`
  (`src.data`). Behavior column via `resolve_behavior_col(md)` (=
  `behavior_dummy_forced`).
- Behavioral windows/sig: `outputs/causal46_joined/t_tests/b4_per_window.parquet`
  (and `.../behavioral_discriminative_windows/b_windows.parquet`).
- Acoustic windows/sig: `outputs/causal46_joined/acoustic_on_ambiguous/
  b4_acoustic_per_window.parquet`, `b4_acoustic_per_cell.parquet`.
- Ambiguous steps / counts: `outputs/causal46_joined/trial_balance_index.csv`.
- Timing: `src.stimuli.POD_dict`, `OFFSET_DICT`; `epoch_tmin`, `epoch_sfreq` from
  `src.viz_paper` (tmin = −0.4, sfreq = 100 → **t=0 at sample 40**).
- Sign conventions authoritative refs: `_within_completion.py` (searchlight
  `mean_diff_raw`), `t_tests.py` (behavioral raw = class1−class0; line ~260),
  `_acoustic_step_bootstrap.py` (acoustic raw = s_hi−s_lo, aligned==raw).

---

## 5. N/A

## 6. Population statistics

Across qualifying cells (expect ~30–60):
- **Single-trial mirroring rate**: fraction with `sign(β_step) ≠ sign(β_percept)`
  (and fraction where both are individually significant).
- **Interaction**: distribution of `β_interaction`; one-sample sign test / Wilcoxon
  that it is systematically **negative**; count of cells with significant
  interaction at α=.05 and after **BH-FDR** across cells. Report the F-test p per
  cell and the FDR-adjusted q.
- Cross-tab: interaction-significant vs mirroring vs the conjunction_category /
  manifest annotations (matched/mirrored) for continuity with existing labels.
- Report per phoneme_pair (dn/bm/pb) to confirm coherence isn't a pooling artifact.

Optional robustness: per-cell **label-shuffle null** (permute percept within step)
→ empirical p for the interaction; confirms parametric p under small/unequal n.

---

## 7. Outputs

`outputs/causal46_joined/mismatch_regression/`
- `mismatch_per_cell.parquet` — one row per qualifying cell:
  `subject, electrode_idx, phoneme_pair, word_end, smin, smax, tmin, tmax, n,
  n_steps, min_per_step_per_class, beta_step, se_step, p_step, beta_percept,
  se_percept, p_percept, beta_int, se_int, p_int, F_int, p_int_ftest, r2_add,
  r2_full, delta_r2, mirrored_signs, both_main_sig, window_source (acoustic|fixed)`.
- `mismatch_cell_table.parquet` — long (step × percept) mean HGA + counts per cell
  (for plotting the incongruence profile).
- `mismatch_summary.{csv,pdf}` — population stats (§6) + QC panels (β_interaction
  histogram; example-cell (step×percept) heatmaps; interaction vs corr(B,A)).
- `notebook.ipynb`.

Preserve empty-but-typed schema when no qualifying cells (match the pattern in the
other causal46 notebooks).

---

## 8. Controls / caveats to state in the notebook

- **Selection bias**: acoustic-window selection inflates `β_step` only; percept and
  interaction are orthogonal to it — state and rely on this.
- **Signed vs unsigned**: a null interaction does **not** rule out a signed
  belief−evidence unit or two additive populations — it only fails to find the
  *unsigned conflict* signature. Say so explicitly.
- **RT / effort alternative**: if a reaction-time column exists in the metadata, add
  it as a covariate (and interaction) to test whether the step×percept interaction
  survives control for response difficulty. If RT is unavailable, flag this as an
  untested alternative driver.
- **Small n**: minority-percept cells can be n=4–7; prefer the F-test + FDR, and use
  the permutation null for cells below a count threshold. Report per-cell n so weak
  cells can be down-weighted.
- **One window per cell**; no window-level multiple comparisons; the fixed-window
  variant guards against window cherry-picking.

---

## 9. Validation / reproduction targets (self-check before scaling)

Implement the extractor + regression, then reproduce these exactly
(EC243, `behavior_dummy_forced`, manual baseline `[:41]`, both percepts in {0,1}):

- **e102 bm mountains, steps [3,4,5], window samples [80,90) (t 0.40–0.50):** n=72;
  cell means stop/nasal — step3 +0.65[19]/+0.85[5], step4 +1.39[11]/+0.80[13],
  step5 +2.17[7]/+0.77[17]; additive `β_step≈+0.40 (p≈.018)`,
  `β_percept≈−0.63 (p≈.022)`; full `β_int≈−0.80 (p≈.015)`, R² 0.105→0.180,
  interaction F≈6.2.
- **e101 dn necessary, steps [2,3,4], window samples [110,120) (t 0.70–0.80):** n=72;
  `β_step≈−0.41 (p≈.004)`, `β_percept≈+0.17 (p≈.45, ns)`, `β_int≈−0.25 (p≈.36, ns)`,
  R² 0.125→0.136.

If these don't reproduce, the baseline/window/sign wiring is wrong — fix before the
population run.

## 10. Deliverables

1. `extract_hga_trials` helper in `_within_completion.py` (+ keep `extract_hga`).
2. `notebooks/causal46_joined/mismatch_regression.py` implementing §2–§7.
3. Snakefile rule `joined_mismatch_regression` (inputs: epochs, b4_per_window,
   b4_acoustic_per_cell/per_window, trial_balance_index, the helper; outputs §7),
   mirroring the parameter-injection style of `joined_acoustic_discriminative_windows`.
4. Validate with §9, then `uv run` py_compile + jupytext + a single-subject smoke run.
