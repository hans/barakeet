# Ganong t-tests: pooled-percept + interaction from the B4 bootstrap, and a factorized behavioral cross-reference

> **ℹ Built on the B4 bootstrap (2026-07-01).** The per-step class-balance
> subsampling this design consumes is defined canonically in
> `notebooks/causal46_joined/_within_completion.py` (module docstring); pointer
> at `docs/superpowers/plans/2026-07-01-causal46-within-completion-subsampling.md`.

Status: design sketch (2026-06-05)
Builds on: `2026-05-27-causal46-cross-we-pooled-test.md` (cross-WE pooled pair statistic)

## Goal

Apply the simple bootstrap-CI idiom of `notebooks/causal46_joined/t_tests.py` to the
Ganong effect, controlling for acoustic differences between the two lexical
completions, and cross-referencing (not filtering) the within-completion behavioral
results already produced by B4.

## Conceptual scaffold (the 2x2 at a matched ambiguous step)

For one electrode, one phoneme_pair, one fixed ambiguous step `s`, one time window,
let `M[we, p]` = mean windowed HGA over trials with completion `we` and reported
percept `p`. Model: `M[we,p] = a(s,we) + b(p) + c(we,p) + noise`.

- **Percept main effect** `= 1/2[(M[lex0,/n/]-M[lex0,/d/]) + (M[lex1,/n/]-M[lex1,/d/])]`.
  Suffix acoustics `a(s,we)` cancel inside each completion's percept difference ->
  **clean regardless of interaction**. This is just B4 pooled across completions.
- **Interaction** `= (M[lex0,/n/]-M[lex0,/d/]) - (M[lex1,/n/]-M[lex1,/d/])`.
  Difference of within-completion percept differences -> `a(s,we)` cancels again ->
  **clean**. Signed version of the existing `sign_concordance`. = "is the percept code
  the same in both completions?"
- **Completion main effect** = across-completion comparison. Carries `a(s,lex0)-a(s,lex1)`
  -> **confounded** (this is the literal neural Ganong boundary shift).

The 2x2 quarantines the suffix-acoustic confound into the completion main effect and
shows the two clean contrasts (percept coding + cross-completion code consistency) are
exactly the project's local-disambiguation question.

## Key reuse fact

`outputs/causal46_joined/t_tests/b4_bootstrap.parquet` already stores, per
`(subject, electrode_idx, phoneme_pair, word_end, smin, smax, replicate)`, the
sign-aligned per-replicate effect `mean_diff_aligned` (positive = acoustic-preferred
class higher), plus `mean_pos_raw`/`mean_neg_raw`. Both clean contrasts are linear
combinations of the two word_ends' `mean_diff_aligned` arrays -> no new bootstrap, no
re-fit. The window grid is identical across completions within a pair, so all windows
are shared.

Alignment is suffix-independent (the electrode's acoustic tuning is the same under
both completions; `acoustic_preferred_class` is computed from endpoints, which exist in
both), so `e_lex0` and `e_lex1` are on a common sign convention and can be averaged /
subtracted directly.

## Implementation constraints (from project conventions)

- **Separate notebook, do not rewrite `t_tests.py`.** New notebook
  `notebooks/causal46_joined/ganong_t_tests.py`. Extract any genuinely shared logic
  into a notebook-local helper `notebooks/causal46_joined/_ganong.py` and import it
  from the new notebook (mirror how `_within_completion.py` is shared).
- Prefix all runs with `uv run`; run notebooks serially (uv lock).
- Read prod state from `outputs_prod/...`; new rule inputs reference `outputs/...`.

---

## Part 1 -- Pooled-percept main effect + interaction (clean, ships)

New notebook `ganong_t_tests.py`. Parameters: `b4_bootstrap_path`, `b4_per_pair_path`,
`b4_per_cell_path`, `outdir`, `ci_low=2.5`, `ci_high=97.5`.

1. Load `b4_bootstrap.parquet`. Keep both-completion pairs only (16/29 in current run).
2. Order completions by `PHONEME_PAIR_TO_WORD_ENDS[pp]` -> `(lex0, lex1)` (NOT the
   alphabetical `sorted()` used in `cross_we_pair_summary`) so interaction sign is
   semantically fixed (lex0 effect minus lex1 effect).
3. Pivot per `(subject, electrode_idx, phoneme_pair, smin, smax, replicate)` to get
   `e_lex0`, `e_lex1` (from `mean_diff_aligned`). Compute per replicate:
   - `pooled_percept = (e_lex0 + e_lex1) / 2`
   - `interaction    =  e_lex0 - e_lex1`
4. Per-window summary (reuse the `per_window_summary` pattern in `t_tests.py:435`):
   median, [2.5, 97.5] CI, 2-sided bootstrap `emp_p = 2*min(frac<=0, frac>=0)`,
   `ci_excludes_zero` for each of `pooled_percept` and `interaction`.
5. Per-pair best window: argmax `|median|` of each contrast (each contrast at its own
   best window). Also emit the value at the magnitude-pool best window from
   `b4_per_pair` for continuity with the existing cross-WE statistic.
6. Outputs:
   - `ganong_per_window.parquet` -- per (pair x window): pooled + interaction CI/emp_p
   - `ganong_per_pair.parquet`   -- best-window pooled + interaction per pair, joined to
     existing `b4_per_pair` (`pair_statistic_med`, `sign_concordance`, `cells_individually_sig`)
   - `ganong_summary.pdf`:
     - signed histogram of best-window `pooled_percept` (should mirror B4 -- sanity)
     - signed histogram of best-window `interaction` centered at 0 (mass near 0 =
       consistent code across completions; tails = code divergence)
     - `e_lex0` vs `e_lex1` scatter (interaction = off-diagonal distance; reuse the
       existing D6a scatter style) colored by `interaction_ci_excludes_zero`
     - per-window population trace: fraction of pairs with `pooled_percept` /
       `interaction` CI excluding 0, vs window center

Null note: the CI/`emp_p` path above runs on the **existing** parquet. The
per-replicate **label-permutation** emp_p (analogous to `pair_emp_p`) needs
`mean_diff_aligned_null`, which the current prod `b4_bootstrap.parquet` lacks. If we
want it, re-run `t_tests.py` once (the code already writes the column) and recompute;
otherwise ship CI-based significance.

Validation: `interaction` mass appearing **pre-POD** is anticipation/prediction, not
Ganong -- label it as such (POD per pair from `POD_dict`).

---

## Part 2 -- Factorized cross-reference vs behavioral Ganong

The marginal (percept-collapsed) completion effect factorizes as
`acoustic_confound(s) + dP(/n/|s) * [within-completion percept-coding strength]`,
where `dP(/n/|s)` is the **behavioral** Ganong. So neural Ganong = behavioral
percept-mix shift x neural percept coding. This upgrades "do the same electrodes show
up" into a predicted relationship.

### 2a. Behavioral Ganong ingredient (clean, reuse)
Extract the behavioral-Ganong computation from
`notebooks/causal5/ganong_decoding_inspect_population.py` Analysis 5 into
`_ganong.py`: per `(subject, phoneme_pair)` compute `pse_shift` (sigmoid PSE_lex0 -
PSE_lex1 via `src.models.sigmoid.fit_sigmoid` + `_normalize_to_endpoints`) and
`simple_ganong` (mean `behavior_dummy_forced` at steps 3-4, lex1 - lex0). Source =
epoch metadata via `load_epochs_dict` + `add_metadata_features`.

### 2b. Cross-reference analyses (ordered by cleanliness)
1. **Tautology caveat, stated up front:** `pooled_percept` ~ B4, so overlap of
   "B4-significant" x "pooled-percept-significant" is near-trivial. Report it only as a
   consistency check.
2. **Co-localization (clean, no new estimator):** per `(subject, phoneme_pair)`,
   aggregate neural percept coding (mean `|pooled_percept|` or fraction of AS sites with
   `pooled_percept` CI excluding 0) vs `|behavioral Ganong|`. Scatter + correlation.
   Prediction: subjects/pairs with larger behavioral Ganong host stronger/more
   percept-coding sites.
3. **Interaction vs behavioral Ganong (clean):** does cross-completion code consistency
   (`|interaction|` or fraction with interaction CI excluding 0) relate to behavioral
   Ganong magnitude?
4. **Factorized regression (needs the confounded completion main effect -- optional):**
   regress the de-confounded completion effect on `pooled_percept * signed behavioral
   Ganong`. See 2c for the estimator; mark optional.

### 2c. (Optional) de-confounded completion main effect
This is the only piece that needs new compute and the one confounded ingredient.
Estimate it with an **endpoint difference-in-differences** rather than TRF:
- New bootstrap pass (reuse `select_cell_trials_bootstrap` + `searchlight_mean_diff`
  from `_within_completion.py`) with classes = **completion** (lex0 vs lex1), percept-
  collapsed, per-step balanced across completions.
- Ambiguous-step completion diff = `a(.,lex0)-a(.,lex1) + dP*percept_coding`.
- Endpoint (steps 1,6) completion diff estimates the pure suffix-acoustic term
  `a(.,lex0)-a(.,lex1)` -- valid because (i) behavioral Ganong ~ 0 at endpoints and
  (ii) suffix waveforms are step-invariant within a completion, so the post-POD suffix
  response is ~step-invariant.
- **completion_effect_deconf = ambiguous-step completion diff - endpoint completion diff.**
  Restrict to a post-POD window. Bootstrap CI as usual.
- Validation: the confounded ambiguous-step completion diff should be ~0 pre-POD and
  grow post-POD; the endpoint diff should track the suffix-acoustic component.

Outputs: `ganong_behavioral.parquet`, `ganong_cross_reference.parquet`, panels appended
to `ganong_summary.pdf` (co-localization scatter, interaction-vs-behavioral scatter,
and -- if 2c is built -- the factorized regression with the acoustic confound as a
nuisance/intercept term).

---

## Sequencing
1. Part 1 on existing `b4_bootstrap.parquet` (CI-based) -> ships immediately, 16 pairs.
2. Part 2a + 2b.1-2b.3 (behavioral ingredient + clean cross-references).
3. (Optional) re-run `t_tests.py` for `mean_diff_aligned_null` -> permutation emp_p.
4. (Optional) Part 2c completion main effect + factorized regression.

## Open questions
- Best-window selection per contrast vs a single shared window across all three
  contrasts (multiple-comparison vs interpretability).
- 16 both-WE pairs is a small N; decide whether to pool across phoneme_pairs for the
  population fractions or report per-pair.
