# Acoustic contrast controlling for behavior (ambiguous-trial acoustic effects)

Date: 2026-07-03
Status: spec, pre-implementation
Branch context: `causal6-speech-responsive-update`

## One-line

Add a second within-cell contrast — **acoustic step difference on ambiguous
trials, controlling for behavioral report** — as a sliding-window bootstrap
t-test and a new facet in the B4 star-plot gallery, alongside the existing
within-completion behavioral (perceptual) contrast.

## Motivation & the core symmetry

The existing B4 analysis (`notebooks/causal46_joined/t_tests.py`) contrasts the
two behavioral reports (heard /d/ vs /n/) on ambiguous, acoustically-matched
trials, *controlling for acoustic step* — the perceptual contrast. This spec adds
its mirror image: contrast **acoustic steps**, *controlling for behavioral
report* — the acoustic effect that survives within the ambiguous regime once
percept is held fixed.

The two contrasts run on the **identical bootstrap draw**. `per_step_class_counts`
(group_col = behavior) + `select_cell_trials_bootstrap` already draw, per step,
`n_s = min(#d, #n)` trials from *each* behavior — so every per-step chunk is
50/50 behavior. The perceptual contrast pools those chunks by behavior (matched
on step). The acoustic contrast re-slices the *same drawn indices* back into
per-step chunks and contrasts steps (each 50/50 behavior → matched on behavior).
One draw, two orthogonal, both-matched contrasts. See
`_within_completion.py` module docstring and the 2026-07-01 within-completion
subsampling plan for the canonical sampling rule.

## Decisions locked (with user)

1. **Contrast quantity (this phase): extreme qualifying-step pair.**
   `s_lo = min(qualifying_steps)`, `s_hi = max(qualifying_steps)`; signed diff
   `hi − lo`. (Future phase reuses the *same* per-step draws for a within-window
   sigmoid/gradient-vs-categorical shape test — see "Designed-for reuse" below.)
2. **Facet traces: all qualifying steps + diff overlay.** One bootstrap
   mean-HGA trace per qualifying step (color ramp), with the extreme-pair
   mean-diff + CI band + significance bars overlaid — mirrors the behavior
   facet's trace+overlay layout.
3. **Search window: broad common range** `[ac_search_smin .. behav_smax]`
   (acoustic-search start → perceptual-search end). Captures early acoustic-driven
   difference and any late persistence; keeps acoustic vs perceptual magnitudes
   comparable in one window family.
4. **Code layout: new notebook + shared engine.** Extract the bootstrap
   primitive into a shared helper; new notebook computes the acoustic rows;
   the combined gallery is produced **downstream** (avoids the dependency cycle,
   leaves the existing behavior-only gallery untouched).
5. **Panel order: append acoustic below behavior.** unambiguous(1&6) → behavior
   → acoustic → decoding. Behavior facet stays at `fig.axes[1]` so the existing
   `_ax_behav` / cross-WE-bar logic is undisturbed.

## Defaults (asserted; veto if wrong)

- **Scope: B4 across-step cells only** (`n_qualifying_steps ≥ 2`). B3 single-step
  cells cannot contrast steps → no acoustic facet.
- **Sign: raw signed diff, `hi − lo`.** No `acoustic_preferred_class` alignment
  (step order fixes polarity: higher continuum step = more /n/). The emitted
  `mean_diff_aligned` column *equals* `mean_diff_raw` for this contrast, so the
  contrast-agnostic summary functions reuse unchanged.
- **Control ratio: 50/50 behavior within each step** (what the existing draw
  gives) — the symmetric "acoustic effect with reports equally weighted"
  estimand.
- **`word_end` fixed (within-completion).** Isolates the continuum-step acoustic
  cue from the suffix; the acoustic contrast is run per completion exactly like
  the perceptual one.
- **Null: behavior-controlled step permutation.** Within each behavior, pool the
  s_lo + s_hi drawn trials and re-split across the two pseudo-step groups
  (preserving each group's 50/50 behavior balance). Mirrors the existing
  within-step behavior-pool null in `bootstrap_cell`.
- **Best-window + significance: reuse** `per_window_summary`, `per_cell_best`,
  `population_summary` verbatim (they key off `mean_diff_raw`/`mean_diff_aligned`).

## Naming (CONFIRMED 2026-07-03)
- Shared helper: **`_contrasts.py`** — per-step bootstrap draw primitive + the
  two contrast builders (behavior, acoustic-extreme).
- New notebook: **`acoustic_on_ambiguous.py`** (echoes legacy
  `acoustic_morphology_on_ambiguous.py`; "acoustic effect on ambiguous trials").
- Output parquets (schema-identical to `b4_*`): **`b4_acoustic_bootstrap.parquet`**,
  **`b4_acoustic_per_window.parquet`**, **`b4_acoustic_per_cell.parquet`**,
  **`acoustic_cell_manifest.parquet`**.
- Combined gallery dir: **`star_plots_both/`** →
  `{powered,powered_significant}.pdf`.

Alternates considered and rejected: `acoustic_within_behavior.py` (awb_ prefix) ·
`acoustic_contrast.py`.

## Implementation

### A. Shared sampling primitive (`_within_completion.py` or `_contrasts.py`)

Add `select_cell_trials_bootstrap_perstep(per_step, *, rng) -> {step: {class: idx}}`
that replicates the **exact RNG call sequence** of the existing
`select_cell_trials_bootstrap` (same `for step: n_s = min(...); for cls:
rng.choice(idxs, n_s, replace=True)` loop order) but returns the draw organized
by step instead of concatenated by class. Because the RNG sequence is identical,
with the same `base_seed` and `per_step` the acoustic and behavior facets are
provably built from the **same replicates**. (Optional, low-priority: refactor
the existing function to delegate to this — do only if the behavior outputs stay
bit-identical.)

Derived, all from one `perstep` draw `d = {step: {class: idx}}`:
- **Perceptual (existing):** `pos = concat_s d[s][b_hi]`, `neg = concat_s d[s][b_lo]`.
- **Acoustic extreme pair:** `pos = concat_b d[s_hi][b]`, `neg = concat_b d[s_lo][b]`.
- **Per-step profile (traces + future shape test):**
  `prof[s] = concat_b d[s][b]` → mean HGA per window.

### B. Acoustic bootstrap (shared engine, called by new notebook)

`bootstrap_cell_acoustic(...)` mirroring `bootstrap_cell` (t_tests.py:218):
- Build `per_step` once (group_col = behavior, qualifying_steps).
- `s_lo, s_hi = min/max(qualifying_steps)`.
- Search range `[ac_search_smin .. behav_smax]` (broad common); windows via
  `searchlight_mean_diff` with `WINDOW_SIZE`/`STRIDE`.
- Per replicate r (seed `base_seed + r`, R = 1000):
  - `d = select_cell_trials_bootstrap_perstep(per_step, rng)`
  - `pos = concat_b d[s_hi][b]`, `neg = concat_b d[s_lo][b]`
  - `res = searchlight_mean_diff(hga, pos, neg, search=broad)`
  - Null: per behavior b, pool `d[s_hi][b] ∪ d[s_lo][b]`, shuffle, split
    `n_shi → pseudo-hi`, `n_slo → pseudo-lo`; `null_pos/neg = concat_b ...`;
    `res_null = searchlight_mean_diff(...)`.
  - Emit rows with `mean_diff_raw = mean_diff_aligned = mean(pos) − mean(neg)`
    and `mean_diff_aligned_null = null diff`, plus `smin/smax/tmin/tmax`,
    `n_per_class`, `acoustic_peak_auc`, `s_lo`, `s_hi`. **Schema identical to
    `b4_bootstrap.parquet`** (so `per_window_summary` etc. reuse unchanged),
    with extra `s_lo`/`s_hi` columns.
- Summaries: `per_window_summary` → `b4_acoustic_per_window.parquet`;
  `per_cell_best` → `b4_acoustic_per_cell.parquet`; `population_summary` for the
  population panel.

### C. New notebook `acoustic_on_ambiguous.py`

Inputs: `b4_qualified` cell list + `cell_manifest.parquet` (from t_tests.py),
epochs, `phon_peaks_all.parquet`. Runs section B over B4 cells, writes the
`b4_acoustic_*` parquets + `acoustic_cell_manifest.parquet`. Then renders the
combined gallery (section E). Jupytext percent-format; `uv run` per project
convention.

### D. `matched_n_star_plot` — optional acoustic panel (`_within_completion.py:508`)

Add optional params: `acoustic_mean_diff_arrays`, `acoustic_sig_windows`,
`acoustic_extreme_steps=(s_lo, s_hi)`, `acoustic_R_plot`. When present:
- Insert `ax_acoustic` **below** `ax_bot` (behavior), decoding panel moves to
  bottom. `height_ratios` extend to `[1, 1, 1, 0.45]` (with decoding) /
  `[1, 1, 1]` (without).
- Draw per-step bootstrap mean-HGA traces for **all** qualifying steps (color
  ramp; emphasize s_lo/s_hi) using `select_cell_trials_bootstrap_perstep`
  aggregated per step over `acoustic_R_plot` reps.
- Overlay `acoustic_mean_diff_arrays` (dashed line + CI band) and
  `acoustic_sig_windows` (gray bars), identical style to the behavior overlay.
- `fig._ax_behav` stays = `ax_bot`; add `fig._ax_acoustic`.
- When acoustic params absent → **unchanged** 2/3-panel behavior figure, so
  t_tests.py's existing gallery is untouched.

### E. Combined gallery (in the new notebook)

Generalize `write_annotated_pdfs` (`_star_gallery.py:67`) with optional
`acoustic_per_window` + `acoustic_pair_lookup`. For each entry it extracts the
acoustic `mda`/`sig_wins` (same code path as behavior, since schema matches) and
passes them into `matched_n_star_plot`. Reads behavior `b4_per_window` (from
t_tests output) + acoustic `b4_acoustic_per_window`. Emits `star_plots_both/`.

### F. Snakefile (`workflows/causal46_joined.Snakefile`)

New rule `acoustic_on_ambiguous`: inputs = t_tests outputs (`b4_qualified` /
`cell_manifest`, `b4_per_window`) + epochs + phon_peaks; outputs = `b4_acoustic_*`
parquets + `star_plots_both/`. Downstream of the existing `t_tests` rule. No edge
into t_tests (no cycle).

## Reuse ledger (unchanged code)

`per_step_class_counts`, `select_cell_trials_bootstrap`, `searchlight_mean_diff`,
`per_window_summary`, `per_cell_best`, `population_summary`, `behav_search_range`,
`resolve_behavior_col`, `extract_hga`. New: one per-step sampler variant, one
acoustic bootstrap builder, optional acoustic panel in `matched_n_star_plot`,
optional acoustic args in `write_annotated_pdfs`, one notebook, one Snakefile rule.

## Designed-for reuse: future sigmoid/shape test

The per-step profile `prof[s]` (mean HGA per window, per replicate) is the exact
input a later within-window shape test needs: bootstrap the per-step mean-HGA
vector inside a chosen window and test competing forms (monotone/linear = gradient
vs single-boundary step = categorical vs sigmoid fit). No new sampling — same
`select_cell_trials_bootstrap_perstep` draws. Keep `s_lo/s_hi` and the full
per-step structure in the emitted rows so the shape test layers on without
re-running the bootstrap.

## Deferred (later discussions)

- **Relative scale** of acoustic vs perceptual contrasts per cell: `b4_acoustic_per_cell`
  and `b4_per_cell` are joinable on `(subject, electrode_idx, phoneme_pair,
  word_end)` → per-cell `|acoustic| / |perceptual|` in each best window, or paired
  in the broad common window. Schema is ready; analysis TBD.
- **Summary effect sizes** across the electrode population for acoustic vs
  perceptual responses.

## Open confirmations before coding

1. Naming — CONFIRMED (`acoustic_on_ambiguous.py` / `_contrasts.py` /
   `b4_acoustic_*` / `star_plots_both/`).
2. Broad-window lower bound: use `ac_search_smin` (default 45 in
   `write_annotated_pdfs`) as the acoustic-search start. Assumed yes unless changed.
3. Asserted defaults (B4-only, 50/50 control, behavior-controlled permutation
   null) stand unless vetoed.
