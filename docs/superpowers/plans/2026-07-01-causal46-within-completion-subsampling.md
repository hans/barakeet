# causal46 within-completion subsampling — canonical pointer & consumer map

Status: reference note (2026-07-01). **Not a design doc — a pointer.**

## Where the definition lives

The per-step class-balance subsampling rule (B3 single-step, B4 across-step)
that underlies every within-completion perceptual contrast in the causal46
evaluation pipeline is defined **authoritatively in the module docstring of**

    notebooks/causal46_joined/_within_completion.py

That module is imported by every consumer below, so it is the one location the
code forces contact with — read it for the mechanics (min_class[s] per class,
both classes bootstrapped with replacement, `n_per_class = Σ_s min_class[s]`,
the step-matched-gap property, and the shared-seeding equivalence between the
gallery figures and the t-tests). This note is deliberately thin: it does not
restate the mechanics, so there is nothing here to drift.

## Why the gap between the two behavior traces is a percept effect

One line summary (full argument in the module docstring): within a cell,
`word_end` is fixed (within-completion, suffix acoustics controlled) and both
report-behavior traces are built from the *identical* acoustic-step mixture
(min_class[s] trials per step). Any acoustic-step effect is therefore
common-mode and cancels in the class-0 vs class-1 difference; the residual gap
is a step-matched contrast of reported percept.

## Consumers (import `_within_completion`)

Trial selection / bootstrap:
- `_star_gallery.py` (→ `matched_n_star_plot`, `early_window_star_plot[_compact]`) — star-plot galleries
- `t_tests.py` (→ `bootstrap_cell`) — per-site bootstrap CIs, population summary, filtered gallery
- `t_tests_by_early_window.py`, `early_perceptual_windows.py`, `early_window_site_types.py`
- `behavioral_discriminative_windows.py`, `_contrast.py` / `contrast_plot_by_site_type.py`
- `strong_generator.py`, `strong_generator_demo.py`
- `type1_ambiguous_hga_coding.py`, `type1_early_decoder_on_ambiguous.py`
- `acoustic_bootstrap.py`, `acoustic_decoding_single_electrode_inspect.py`

Upstream producer of the cell/threshold inputs:
- `trial_balance_index.py` → `trial_balance_index.csv`, `trial_balance_summary.csv`
  (per-step `min_class`, `is_ambiguous_step`, `meets_threshold_K`)

## Historical design docs (superseded on the sampling rule)

These predate the current rule and carry a superseding header pointing here.
They remain as historical records; the code wins on any discrepancy.

- `2026-05-19-causal46-trial-balance-index.md` — cell/threshold definitions (still largely current)
- `2026-05-20-causal46-star-plots.md` — B3/B4 gallery (describes the retired `star_plots.py`, single-draw subsample)
- `2026-05-22-causal46-ttests-calibration.md` — **most stale**: global calibration-N (`N_cal`) + `select_cell_trials` scheme, both dropped in favor of per-step `min_class[s]` balance via `select_cell_trials_bootstrap`
- `2026-06-05-causal46-ganong-t-tests.md` — 2×2 percept/interaction built on the B4 bootstrap
- `2026-05-27-causal46-cross-we-pooled-test.md` — cross-WE pooled pair statistic (consumes the B4 bootstrap)
