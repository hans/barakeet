# causal46 cross-word-end pooled CI test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. There are explicit **design deliberation** points marked `🟡 DESIGN`; resolve each one before writing the corresponding code, briefly recording the chosen option and reasoning in the plan checkbox.

## Motivation

Manual annotation of `outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered.pdf` (annotated CSV at `~/freesurfer_subjects/barakeet/causal46_pipeline/filtered_manifest.csv`; schema at `notebooks/causal46_joined/manual_annotation_schema.md`) surfaced a robust pattern:

- For site×phoneme_pair groups where the annotator marked **matched/mirrored morphology across both word_ends** (`match_we=y`, n=23 pairs), the per-WE bootstrap effect sizes are tightly coupled in **magnitude** but their **signs are roughly independent**:
  - `|best_mean_diff_aligned_med|` across the two WEs: Spearman r = **0.72, p = 1e-4**
  - Same metric across all 88 two-WE pairs: r = 0.24, p = 0.027 (weaker but still real)
  - Sign concordance: 12/23 matched (52%, ns) — sign flips freely
- Of those 23 matched-WE pairs: 7 are individually significant on both WEs, 7 on one WE only, **9 on neither**. **16/23 (70%) lose evidence under the current per-WE CI test** despite the annotator (and the cross-WE magnitude statistic) judging the morphology consistent.

The current `t_tests.py` pipeline tests each (site × phoneme_pair × word_end) cell independently. It cannot see that two underpowered same-site/pair WEs that share magnitude form a consistent body of evidence. This is the lift this plan targets.

ROI breakdown (from the annotation analysis) suggests the lift will be largest in **supramarginal** (n=16 cells, 63% any-behav, 50% match_we, but only 19% per-cell sig). STG benefits less in relative terms (already 32% per-cell sig at 50% match-eligible).

## Goal

Add a **cross-word-end pooled bootstrap statistic** to `t_tests.py`, derived from the same per-replicate searchlight bootstrap that produces `b4_bootstrap.parquet`. Per (subject × electrode_idx × phoneme_pair), report a pair-level CI and empirical p over a magnitude-pooled effect that combines both word_ends' aligned mean-diffs. Augment the population summary and the filtered gallery so this statistic is reviewable alongside the per-cell results.

Out of scope: re-doing the bootstrap with a different trial-balance scheme, applying FDR, or refactoring `_within_completion.py`. We re-use the existing `b4_bootstrap.parquet` replicates.

## File Structure

- **Modify:** `notebooks/causal46_joined/t_tests.py` — add pair-level aggregation, augment per_cell parquet with pair-level columns, extend `population_summary.pdf` with a new section, optionally extend `star_plots_filtered/b4_powered.pdf` annotations.
- **Create:** `outputs/causal46_joined/t_tests/b4_per_pair.parquet` — one row per (subject, electrode_idx, phoneme_pair), pair-level statistic and CI.
- **🟡 DESIGN:** Either extend `population_summary.pdf` in place or create `outputs/causal46_joined/t_tests/cross_we_summary.pdf` as a standalone document. (See "Visualizations" below.)
- **🟡 DESIGN:** Optionally create `notebooks/causal46_joined/cross_we_inspection.py` as an auxiliary exploration notebook (per-pair scatter, ROI breakdown, comparison to annotation manifest). Resolve in design step 1 below.

## Inputs (already produced by current `t_tests.py`)

- `outputs/causal46_joined/t_tests/b4_bootstrap.parquet` — long-form per (cell × window × replicate); has `mean_diff_aligned` per replicate per window. Cell key: `(subject, electrode_idx, phoneme_pair, word_end)`. **Critical:** the existing bootstrap already aligns each cell's mean-diff to its own acoustic-preferred class. **We need to relate two WE cells with possibly different preferred classes** — see design step 2.
- `outputs/causal46_joined/t_tests/b4_per_window.parquet` — per (cell × window) aggregates with CI on aligned + raw.
- `outputs/causal46_joined/t_tests/b4_per_cell.parquet` — per cell best-window row.
- `outputs/causal46_joined/t_tests/cell_manifest.parquet` — manifest with `preferred_class`, `n_per_class`, etc.

## Outputs

- `outputs/causal46_joined/t_tests/b4_per_pair.parquet` — one row per (subject, electrode_idx, phoneme_pair). Schema (proposed; refine in design step 2):
  - `subject, electrode_idx, phoneme_pair`
  - `word_ends` — list (length 1 or 2) of the WEs that contributed
  - `n_we_contributing` — 1 or 2
  - `pair_smin, pair_smax, pair_tmin, pair_tmax` — best pair-level window (see design step 3)
  - `pair_statistic_med, pair_statistic_ci_lo, pair_statistic_ci_hi` — bootstrap CI on the pair-level statistic
  - `pair_emp_p` — empirical 2-sided bootstrap p
  - `pair_ci_excludes_zero` — boolean
  - `acoustic_peak_auc_max` — max across the two cells (or first if n=1)
  - `cells_individually_sig` — count (0, 1, or 2)
- Augment `b4_per_cell.parquet` with three additional columns joining the pair-level result back to the cell: `pair_statistic_med, pair_ci_excludes_zero, pair_n_we_contributing`. This lets the existing filtered-gallery code surface the pair verdict per cell.
- New section(s) of `population_summary.pdf` (or a standalone PDF — see Visualizations).

---

## Design deliberations (resolve BEFORE writing the corresponding code)

The implementing agent must work through these in order, briefly noting the chosen option and why in the checkbox. The default options in **bold** are recommended starting points but the agent should consider the alternatives and document the choice.

### 🟡 DESIGN 1: Where does the cross-WE plotting live?

- [ ] **Option A (recommended): extend `population_summary.pdf` in place.** Add a "Cross-WE pooled" page set after the existing per-cell pages. Pro: one document; reviewers don't have to discover a new PDF. Con: grows an already-large PDF.
- Option B: New `cross_we_summary.pdf` next to `population_summary.pdf`. Pro: clean separation; easier to iterate without touching existing pages. Con: bifurcates the review surface.
- Option C: Auxiliary exploration **notebook** `cross_we_inspection.py` for interactive review only; PDF is just the summary numbers. Pro: hosts richer per-pair scatter/per-WE comparisons; reviewer can drill into specific pairs.

Choose ONE. (A and C are not mutually exclusive — A handles "summary deliverable", C handles "exploration". If the agent picks C, still produce the summary in A.)

### 🟡 DESIGN 2: How is the pair-level statistic defined?

The bootstrap already produces `mean_diff_aligned` per (cell, window, replicate) where positive = acoustic-preferred class shows higher HGA. Two same-site/pair WE cells may have **different acoustic-preferred classes** (the within-completion contrast is "heard X vs heard Y" within one WE; the two WEs' "preferred class" can swap). The pair-level statistic needs to handle this.

- [ ] **Option A (recommended for hypothesis match): magnitude-pooled.** Define per-replicate pair statistic `S_r = (|mean_diff_aligned_we0_r| + |mean_diff_aligned_we1_r|) / 2`. The CI on `S_r` from R=1000 replicates is the pair-level CI. **This is the statistic that the annotation analysis directly validates** (the r=0.72 |effect|-correlation observation). Sign info is sacrificed. Empirical p tests against null `S_r ≤ 0` — but the magnitudes are non-negative, so this is one-sided by construction. **For inference, use a sign-flip / label-permutation null** instead of `S_r ≤ 0` (see design step 5).
- Option B: signed product. `S_r = sign(e0_r) · sign(e1_r) · sqrt(|e0_r·e1_r|)`. Captures "signs agree → positive S; signs disagree → negative S". Sensitive to coherent direction; loses cells where morphology matches but signs flip (which the annotation says is common). Likely WORSE for the observed data given sign concordance ≈ 50%.
- Option C: paired test on per-replicate aligned mean-diffs, treating WE as a paired factor. Equivalent to A in the magnitude-only case but more interpretable as a hypothesis test. Heavier machinery.

If the agent picks A: also **report the sign concordance** (`mean over r of sign(e0_r) == sign(e1_r)`) as a diagnostic column. If concordance is consistently high or low for a pair, that's interpretable.

### 🟡 DESIGN 3: Window selection for the pair

Each WE cell has its own per-window CI. To combine, we need a single (smin, smax) window per pair.

- [ ] **Option A (recommended): pair best-window over the pair-statistic.** For each window present in BOTH WEs, compute median pair-statistic across replicates; pick the window maximizing this. Use replicates only at that one window for the CI. Simplest; matches the per-cell `best-window` logic in `per_cell_best`.
- Option B: per-WE best-windows separately, combined into a pair-stat at whatever pair of windows each cell chose. Pro: each WE can find its own best timing. Con: cross-WE comparisons get muddied; window timing is unanchored.
- Option C: searchlight over the pair (report all windows × pair-stat, then take best). Effectively A but with a per-window pair output. Heavier but allows the visualization to show pair statistic over time.

If the agent picks A or C, the two WEs must share a window grid. They already do — both come from the same `WINDOW_SIZE/STRIDE` config.

### 🟡 DESIGN 4: Handling 1-WE pairs

About 27 of the 62 has-any-behav site+pair groups (from annotation) have only one WE with a behavioral response. Many more site+pair groups have only one WE present in `b4_bootstrap.parquet` at all (e.g. one WE was underpowered or had `search_range_too_narrow`).

- [ ] **Option A (recommended): include 1-WE pairs with `n_we_contributing=1`** and pair_statistic = `|mean_diff_aligned|` of the single WE. This is just |per-cell effect|; no gain, but no exclusion. Mark `n_we_contributing=1` so summaries can stratify.
- Option B: exclude 1-WE pairs entirely. Simpler but throws away ~30 site+pair groups that exist in the data.

### 🟡 DESIGN 5: Null distribution for the pair empirical p

`S_r = (|e0_r| + |e1_r|)/2` is non-negative. Comparing to zero is uninformative. Two reasonable approaches:

- [ ] **Option A (recommended): sign-flip null.** For each replicate, multiply each WE's per-trial labels by a random ±1 (within the existing bootstrap draws — re-use trial indices, swap class labels). Recompute `S_r`. Empirical pair p = fraction of null `S_r >= observed S_r`. Adds a second bootstrap pass — cost is ~2× current. Closest to "no real cross-WE consistency" null.
- Option B: build a permutation-based null entirely outside the existing bootstrap. More work; less re-use.
- Option C: report only the **observed pair CI** (lower bound > 0 is uninformative since S ≥ 0; use lower-bound > some threshold derived from the magnitude distribution under the existing per-cell raw mean-diff distribution as a heuristic). Cheaper but harder to defend.

If the agent picks A, **document that the existing bootstrap loop in `bootstrap_cell` needs to also yield label-shuffled replicates**, either as a parallel pass or as a sidecar parquet `b4_null_bootstrap.parquet`.

### 🟡 DESIGN 6: Visualizations

The implementing agent should propose 2–3 specific plots and pick the most informative subset. Strong candidates (do not need to include all):

- **Per-pair scatter**: x = `|effect_we0|`, y = `|effect_we1|`, one dot per site+pair, colored by `pair_ci_excludes_zero`, marker shape by `n_we_contributing`. Annotated with the cross-WE Spearman r. Make this **the headline plot** — it concretely shows the cross-WE magnitude coupling and how many pairs gain evidence from pooling. Strongly recommended.
- **Lift waterfall**: bar chart with one bar per matched-WE pair, height = `pair_emp_p` (or `-log10(pair_emp_p)`), colored by `cells_individually_sig` ∈ {0, 1, 2}. Reviewers can see at a glance how many pairs newly cross significance.
- **ROI / phoneme_pair breakdown**: peak fraction CI-pair-excludes-zero per ROI (mirrors the existing per-cell `frac_ci_aligned` ROI bars). Lets us quantify the predicted supramarginal lift.
- **Two-panel per-pair example pages**: for the top-N pairs by `pair_statistic_med`, show both WE bootstrap CI traces (the existing `site_effect_fig`) stacked, with a pair CI band overlaid. Goes into the filtered gallery, not the summary PDF.
- Sign-concordance histogram across pairs (diagnostic for design 2 choice).

🟡 The agent should pick the subset and justify in the checkbox.

---

## Tasks

### Setup

- [x] Read this entire plan. Read `notebooks/causal46_joined/t_tests.py` end-to-end. Read `manual_annotation_schema.md`. Verify `b4_bootstrap.parquet` exists locally OR confirm with the user where to get it. → outputs are on the remote compute host; user confirmed to update `t_tests.py` directly and run there.

### Design resolution

- [x] Resolve **DESIGN 1** → **Option A** (extend `population_summary.pdf` in place). One document keeps reviewers in one place; cross-WE pages are added after the Decision Callout page.
- [x] Resolve **DESIGN 2** → **Option A** (magnitude-pooled S_r). The r≈0.72 empirical validation was specified as a post-hoc check against the annotation CSV; user instructed to run the notebook end-to-end on the compute host. Sign concordance is reported as a diagnostic column.
- [x] Resolve **DESIGN 3** → **Option A** (best window = argmax median S_r over shared windows). Matches per-cell best-window logic; shared window grid guaranteed by PAIR_SMAX.
- [x] Resolve **DESIGN 4** → **Option A** (1-WE pairs included with n_we_contributing=1; S_r = |mean_diff_aligned_r|). No pairs discarded.
- [x] Resolve **DESIGN 5** → **Option A variant** (replicate-permutation null, not a re-run of bootstrap_cell). For each 2-WE pair, 999 independent shuffles of WE1's replicate order yield the null distribution; `pair_emp_p = P(null_median >= observed_median)`. No additional bootstrap pass needed — all null computation is post-hoc from existing `b4_bootstrap.parquet`. 1-WE pairs use per-cell `best_emp_p_aligned` from the existing 2-sided bootstrap p.
- [x] Resolve **DESIGN 6** → **Three plots**: (a) per-pair |effect| scatter (headline, 2-WE only, colored by pair_ci_excludes_zero, Spearman r annotated); (b) ROI breakdown bar chart (frac pair_ci_excludes_zero per ROI); (c) lift waterfall (-log10 pair_emp_p per 2-WE pair, colored by cells_individually_sig). Chosen because they directly map to the three key questions: coupling magnitude, anatomical distribution, lift over per-cell test.

### Implementation

- [x] In `t_tests.py`, after the per-cell augmentation section, add `cross_we_pair_summary(boot, per_cell)` function. Implemented at lines ~518–702.
- [x] DESIGN 5 variant used replicate-permutation null (post-hoc, no modification to `bootstrap_cell` needed). No `b4_null_bootstrap.parquet` written.
- [x] Write `b4_per_pair.parquet`. Join `pair_statistic_med`, `pair_ci_excludes_zero`, `pair_n_we_contributing` back into `b4_per_cell`; re-write `b4_per_cell.parquet` with pair columns.
- [x] Added cross-WE PDF pages (section header, scatter, ROI breakdown, lift waterfall) inside the existing `with PdfPages(pdf_path) as pdf:` block, after the Decision Callout page.
- [x] Augmented final print summary with n_we=2 count, n pair_ci_excludes_zero, lift breakdown (0-sig / 1-sig / 2-sig cells).

### Validation

- [ ] **Sanity check against manual annotation.** Load `~/freesurfer_subjects/barakeet/causal46_pipeline/filtered_manifest.csv`, restrict to `match_we_y = (matched/mirrored morphology across WE? == 'y')`. The 23 matched-WE pairs should have systematically tighter pair-level CIs than 23 random non-matched pairs. Print medians of `pair_ci_lo, pair_ci_hi, pair_emp_p` for both subsets; verify the matched subset's lower bound exceeds the random subset's. If not, surface this rather than burying it.
- [ ] **Sanity check for hypothesis lift.** Of the matched-WE pairs that were not individually significant on either WE under the per-cell CI (9 from the annotation), report how many become `pair_ci_excludes_zero=True`. The user expects roughly half. If it's 0 or all 9, surface this.
- [ ] **Confirm 1-WE pairs are handled.** Make sure `n_we_contributing=1` rows exist in the output and the per-cell augmentation handled missing-pair-partner cases without errors.
- [ ] Run the full notebook end-to-end via `uv run` (per project preference). Confirm no exceptions, all output files present, page counts look right.

### Handoff

- [ ] Print a short summary to the user: chosen design options with one-line reasons, headline numbers from validation, paths to new artifacts. Stop. Do not commit unless asked.

---

## Reference: relevant prior work

- Implementation backbone: `notebooks/causal46_joined/t_tests.py` (this plan modifies it in place).
- Annotation source for the cross-WE pattern: `notebooks/causal46_joined/manual_annotation_schema.md` + `~/freesurfer_subjects/barakeet/causal46_pipeline/filtered_manifest.csv`.
- Earlier t-tests spec (for format and conventions): `docs/superpowers/plans/2026-05-22-causal46-ttests-calibration.md`.
- Star plot gallery (consumes per-cell results, will need light tweak if filtered-gallery annotations change): `notebooks/causal46_joined/_within_completion.py:matched_n_star_plot`.

## Out of scope (do not do as part of this plan)

- Re-doing the bootstrap with a different per-step balancing.
- FDR / multiple-testing correction across pairs. The current pipeline doesn't correct across cells either; correction is a separate decision.
- Cross-phoneme-pair pooling at the same electrode. Annotation analysis (H9) found weak consistency across pairs at the same electrode; the cross-WE signal is stronger and more justified.
- Refactoring `_within_completion.py`.
