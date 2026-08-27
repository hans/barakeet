# Single-step late-perceptual projection gate

**Date:** 2026-08-27
**Status:** planned (grilled, decisions locked)

## Question

`plot_for_paper.ipynb` calls a site "late-perceptual" via the projection gate in
`late_perceptual_projection.py`, which computes the within-completion percept
contrast by **pooling all qualifying ambiguous steps** (min_class-weighted). How
many of those sites still pass the **same gate** if the contrast is computed from
just **one** acoustic step — specifically the **most ambiguous** step (report
proportion nearest 50/50)?

## One-line method

Re-run the existing projection gate **unchanged** except that the perceptual
contrast `compute_p` is restricted to a single step. Same windows, same acoustic
template, same permutation null, same uncorrected gate. Only the trial pool
feeding the percept contrast changes.

## Decisions locked (from grilling)

| # | Decision | Choice |
|---|----------|--------|
| Q1 | Framing | Independent single-step count as primary readout; overlap with existing B4 gate as secondary. (Realized via Path A, so it is inherently a paired comparison over the B4 cell pool — see Q6.) |
| Q2 | Power-matched arm | **Not** in this notebook. No confound discussion cluttering it. Just B3 counts vs B4 counts. |
| Q3 | Coverage | **Full landscape** — compute the gate for **every** qualifying ambiguous step at every cell; persist the whole table; filter to most-ambiguous for the headline. |
| Q4 | "Most ambiguous" grain | Per **(subject, phoneme_pair, word_end)**: the qualifying step whose report proportion is closest to 0.5 (max `p(1-p)`). Behavior-only selection — uses reports, never HGA (no double-dipping). |
| Q5 | Untestable most-ambiguous steps | Report **both**: of cells whose most-ambiguous step is testable, X/N pass; plus a separate count M of cells whose most-ambiguous step was untestable (fails `min_class_k` or degenerate null). Denominator for the fraction = testable cells. |
| Q6 | Reuse B4 windows/pool vs rediscover | **Path A** — reuse the exact `b_windows` and cell pool; the only change is restricting `compute_p` to one step. Denominator = the B4-perceptual cells ("these sites"). |
| Q7 | Shared logic | **Extract** `compute_p`/`compute_a_vector`/`compute_a_vector_null`/`get_qualifying_steps` into a helper; refactor `late_perceptual_projection.py` to import it; new notebook imports the same. Guarantees "same criteria" by running the same code. |
| Q8 | Headline grain | **Cell-level paired count** first (each B4 cell vs its own single-step version, same window), then the `plot_for_paper` **site-level** roll-up (`late_category` present/absent) for figure comparability. |
| gate | Which significance | `projection_significant_uncorrected` (p < 0.05) — the flag `plot_for_paper` sums into `late_category`. **No TFCE** (that is `late_perceptual_significance.py`, which `plot_for_paper` does not read). |

## What is reused vs. what changes

The gate (`late_perceptual_projection.py`) does, per cell
`(subject, electrode_idx, phoneme_pair, word_end)`:

1. Cell pool: `A_significant` sites, `early_response_class != "neither"` (type1/2),
   inner-joined to `b_windows` filtered `ci_excludes_zero & n_component_windows >= 2`.
2. For each significant window: `p = compute_p(...)` (min_class-weighted percept
   contrast **pooled over qualifying steps**), `a = compute_a_vector(...)`
   (step6−step1 acoustic template), `projection = p·a`, null by shuffling acoustic
   labels, max-over-window statistic, permutation p-value.
3. Across cells: BH-FDR → `projection_significant`; raw `p < 0.05` →
   `projection_significant_uncorrected`.

**Only step 2's `compute_p` changes**: restrict the weighted sum to one step. Steps
1 and 3 are byte-identical.

## False-negative accounting (why Path A is safe)

- **Window (reused from B4): protective.** Fixing the window a priori turns the
  single-step test into one powered test at a known latency — no searchlight, no
  multiple-comparison correction. B4's window was chosen from a pool that *includes*
  the most-ambiguous step's own trials, so the single-step test there is mildly
  circular → biases toward false *positives*, not negatives. The number is a mild
  over-count if anything; a low survival rate is therefore a conservative result.
  Only real false-negative channel: single-step response peaking at a different
  latency than the pooled response (modest — perceptual timing tracks POD, not step).
- **Cell pool (reused from B4): coverage limit, not misclassification.** Sites B4
  missed are never tested. A single-step-only site requires the percept contrast to
  flip sign across steps — theoretically unlikely for a percept-tracking response,
  so this blind spot is expected to be nearly empty.
- **Dominant loss is raw power** (one step ≈ ¼ the trials), intrinsic to the
  question and *minimized* by Path A.

## Implementation

### Step 1 — extract shared helper (small refactor)

Create `notebooks/causal46_joined/_late_projection.py` holding, lifted verbatim from
`late_perceptual_projection.py`:

- `get_qualifying_steps(md_pp, *, word_end, group_col, ambiguous_threshold=2)`
- `compute_a_vector(...)`, `compute_a_vector_null(...)`
- `compute_p(..., restrict_steps=None)` — **one new kwarg**. `None` ⇒ current
  behavior (all qualifying steps after the `min_class_k` filter). A list/int ⇒
  intersect `per_step_filtered` with those steps before the weighted sum; return the
  same `(p, min_classes, per_step_filtered, N, traces)` tuple. When restricted to a
  step that fails `min_class_k`, `per_step_filtered` is empty ⇒ returns
  `(None, ...)`, exactly the existing untestable path.

Refactor `late_perceptual_projection.py` to `from _late_projection import ...` and
delete its local copies. **Verify**: re-run it (or `git stash` compare) and confirm
`results.csv` is byte-identical. This is the guarantee behind "same criteria."

### Step 2 — the brief notebook

`notebooks/causal46_joined/single_step_perceptual_projection.py` (jupytext percent).
Structure mirrors `late_perceptual_projection.py`'s setup, then diverges only in the
loop. Cells:

1. **Params** (plain assignments, no annotations — ploomber): input paths identical
   to `late_perceptual_projection.py` (`b_windows_all`, `site_pool`, `early_window`,
   epochs), `outdir`, `min_class_k=3`, `window_size=2`, `stride=2`, `n_perms=50000`,
   `master_seed=42`, `fdr_alpha=0.05`.
2. **Load** cell pool + `b_windows` + epochs + `hga_dict` — copy from the original.
3. **Most-ambiguous step per cell.** For each `(subject, phoneme_pair, word_end)`,
   among `get_qualifying_steps(...)`, compute report proportion `mean(bhv_col==1)`
   per step; pick `argmax p(1-p)` (tie-break: larger `min_class` N). Build
   `most_ambiguous[(subject, pp, we)] = step`. Behavior only.
4. **Landscape loop.** Copy the original per-cell/per-window loop, wrapped in an
   extra loop over `qualifying_steps`, calling `compute_p(..., restrict_steps=[s])`.
   Emit one row per `(cell, step)` with `projection`, per-window max stat,
   permutation `p`, and `is_most_ambiguous = (s == most_ambiguous[cell])`. Rows where
   `compute_p` returns `None` → recorded with `testable=False`.
   Write `single_step_projection_landscape.parquet`.
5. **Gate + headline.** `sig_uncorrected = p < 0.05`. Filter to
   `is_most_ambiguous & testable`:
   - **Cell-level (headline):** print `n_pass / n_testable`, and `n_untestable`
     separately. Paired against B4: join to the existing `results.csv` on the cell
     key; print the 2×2 (B4 pass/fail × single-step pass/fail).
   - **Site-level roll-up:** replicate `plot_for_paper` — sum
     `sig_uncorrected` over `word_end` per `(subject, electrode_idx, phoneme_pair)`
     → `late_category ∈ {absent, one-sided, two-sided}`; print the present count and
     compare to the B4 site-level count (**11/31**).

### Outputs

- `outputs/causal46_joined/single_step_perceptual_projection/single_step_projection_landscape.parquet`
  — one row per (cell × qualifying step).
- Printed summary (headline counts). No figures required for v1.

## Reference B4 numbers (from `outputs_prod`, to compare against)

- Projected cells in pool: **38**
- Uncorrected gate-passers (cells): **12**
- FDR gate-passers (cells): 3
- Sites late-present (≥1 completion): **11 / 31**

The single-step headline is: of those same 38 cells (restricted to the ones whose
most-ambiguous step is testable), how many still clear `p < 0.05`, and how does the
site-level present count move from 11.

## How to run

```
# 1. refactor + verify identity
uv run jupytext --to notebook --execute notebooks/causal46_joined/late_perceptual_projection.py  # or via snakemake
#    diff results.csv against the pre-refactor copy — must be identical

# 2. the new notebook (interactive walk-through)
uv run jupytext --to notebook notebooks/causal46_joined/single_step_perceptual_projection.py
#    open and step through; or execute headless:
uv run jupytext --to notebook --execute notebooks/causal46_joined/single_step_perceptual_projection.py
```

Inputs are already built by `snakemake plot_for_paper_inputs`. Runtime ~4× the
original projection (full landscape × ~4 steps), still minutes — the null is
vectorized.

## Caveats (kept out of the notebook per Q2, recorded here)

- The single-step count conflates single-step localizability with per-step power;
  it is a **lower bound**. A power-matched B4-subsampled arm is the natural
  follow-up but is deliberately excluded here.
- Path A is conditional on B4 (windows + pool); it cannot surface single-step-only
  sites. See false-negative accounting above.
