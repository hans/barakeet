# plot_for_paper.ipynb: timing/selection provenance map

**Why this doc exists.** The notebook uses two different statistical regimes for
acoustic vs. perceptual responses, and reports perceptual *timing* from two
different reductions of the same searchlight. Coming back cold, this reads as an
inconsistency. It isn't — but the double-sourcing of perceptual timing is a real
redundancy worth collapsing. This is the answer to the `# TODO is this the right
place to induce timings? verify sources of each` comment in the Timings section.

## The one-sentence reconciliation

**Acoustic responses SEPARATE detection from timing; perceptual responses FUSE
them.** The acoustic response location is known a priori (transient, ~150–250 ms
post-onset), so we pre-specify one window, run **one decoder + one permutation
test per site** for selection (cheap, no multiple-comparisons search to correct),
and do a **post-hoc bootstrap window search purely for description**. The
perceptual response timing is itself the discovery target (diffuse, per-site
variable, near/after POD), so the **window searchlight *is* the detector** — and
it pays for that search with the TFCE max-statistic gate. Both regimes control
FWER; they differ because the two responses differ in whether timing is known or
discovered.

## Provenance table (what feeds what)

| Response | SELECTION artifact | TIMING artifact | Regime |
|---|---|---|---|
| Early acoustic | `acoustic_early/acoustic_early_summary.csv` (`early_sig_df`) — 1 decoder over early window, 1 perm test/site | `acoustic_endpoint_windows/a_windows.parquet` (`early_acoustic_windows`) — post-hoc bootstrap per-window CI + `_find_maximal_runs` unification | separated |
| Late acoustic | `acoustic_late/acoustic_late_summary.csv` (`late_sig_df`) — 1 decoder over `[early-offset→offset]`, 1 perm test/site (5000 perms), FDR | post-hoc per-window bootstrap (companion to endpoint windows) | separated |
| Early perceptual | `early_perceptual_projection/all_sites.csv` (`epp`, `q_one_tailed`) gates; `early_perceptual_windows/ep_windows.parquet` (`early_perceptual_windows`) searchlight beyond acoustic window | same searchlight artifact (selection & timing fused) | fused |
| Late perceptual | `behavioral_discriminative_windows_all/b_windows.parquet` (`b_windows`) searchlight, TFCE-gated by `late_perceptual_significance.py`; projected by `late_perceptual_projection/results.csv` (`lpp`) | **two reductions of the same `b_windows`** — see below | fused |

## The actual tangle: late-perceptual timing is double-sourced

Both perceptual timing figures descend from the **same** `b_windows` TFCE-gated
searchlight. `late_perceptual_projection.py` consumes `b_windows` filtered to
`ci_excludes_zero & n_component_windows >= 2` and samples HGA within those
windows. Two different summaries then leave the notebook:

1. **`lpp.tmin`/`lpp.tmax`** → `integration_timing_from_pod` →
   `lpp_integration_timing_from_pod.pdf`. This is the **union span** of the
   projection's behavioral windows.
2. **`acoustic_cell_means.behav_tcenter`** (from `behav_smin`/`behav_smax` in the
   acoustic-transfer prep, subset to sig-`lpp` cells, `.sort_values("window_id")
   .last()` = **latest single window per cell**) →
   `acoustic_transfer_window_centers_hist.pdf` and the acoustic→behav delay hist.

Same searchlight, two reductions (**union span** vs **latest-window center**),
plus slightly different cell subsetting. That's why it feels like two methods —
it's one method reported twice. **Recommended cleanup:** pick one canonical
late-perceptual timing summary per figure (union span for "when does the
integration window sit relative to POD"; latest-window center only if you
specifically want the acoustic→integration *delay* against `phon_tcenter`), and
note the choice so future-you doesn't re-derive both.

## Two facts that make the FWER argument defensible in the paper

- **The late-acoustic window start is the early acoustic *offset*, not a
  pooled-mean trough.** `acoustic_late.py` (`find_early_offset_smin`) starts the
  window where the endpoint acoustic contrast (step6 − step1) has returned to
  non-significance — the end of the first significant *run* of length
  >= `early_offset_min_sig_run` (default 2, skips single-window onset blips) in
  `acoustic_bootstrap/a_per_window_full_all.parquet`. It anchors on the
  *contrast* (not pooled mean HGA, which cancels a differential late response by
  construction) and is pooled over word_end (the early response is pre-lexical).
  Still one decoder + one permutation test per site, so the FWER argument holds.
  **History:** superseded the earlier `smin_mode="trough"` mean-activation-trace
  detector, which anchored on the left edge of the min *bin* and fired on onset
  blips — placing the window start up to ~470 ms too early and re-including the
  early response (verified on the 13 late-acoustic sites, 2026-08-10).
- **The perceptual searchlight's multiple-comparisons cost is paid explicitly**
  by the TFCE max-|statistic| permutation gate in
  `notebooks/causal46_joined/_windows.py` (`tfce_enhance`, `max_tfce_null`,
  `late_cell_significance`), per
  `docs/superpowers/plans/2026-07-20-causal46-late-perceptual-significance.md`.

## If asked "should the two regimes be unified?"

No — the asymmetry is principled (known vs. discovered timing). What *should* be
unified is the late-perceptual timing reporting (item above). Keep the acoustic
single-test-selection + descriptive-windows split as-is.
