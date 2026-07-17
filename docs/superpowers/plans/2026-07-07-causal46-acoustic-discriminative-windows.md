# Acoustic discriminative windows (companion to behavioral_discriminative_windows)

**Date:** 2026-07-07
**Status:** draft
**Notebook:** `notebooks/causal46_joined/acoustic_discriminative_windows.py`
**Rule:** `joined_acoustic_discriminative_windows` (workflows/causal46_joined.Snakefile)

## Motivation

`behavioral_discriminative_windows.py` is a pure post-processing pass over the
**perceptual** within-completion bootstrap (`b4_bootstrap.parquet`): per B4 cell
it discovers union time-windows carrying a reliable behavioral (percept) HGA
contrast. There has been **no companion** that runs the same window-discovery
machinery over the **acoustic-step** bootstrap.

Note there are *two* acoustic bootstraps in this pipeline; this companion targets
the second:

| bootstrap | source | contrast | trials | key |
|---|---|---|---|---|
| `a_bootstrap.parquet` | `acoustic_bootstrap.py` (`bootstrap_A_site`) | step6 − step1 | **endpoint / unambiguous** | per **site** |
| `b4_acoustic_bootstrap.parquet` | `acoustic_on_ambiguous.py` (`bootstrap_cell_acoustic`) | s_hi − s_lo (extreme qualifying steps), behavior-controlled | **ambiguous** | per **cell (word_end)** |

The first already gets union-run treatment inside `early_perceptual_windows.py`
("Type1" section). This plan adds the missing pass over the second.

## Decisions (confirmed with analyst 2026-07-07)

1. **Mode = independent discovery.** Run the full `_find_maximal_runs` machinery
   on `b4_acoustic_bootstrap` to discover acoustic union windows per cell — a
   structural mirror of `behavioral_discriminative_windows`, *not* an
   evaluate-the-acoustic-contrast-inside-the-behavioral-windows pass.
2. **Candidate region = full range `[onset, PAIR_SMAX]`.** No `smin >= phon_smax`
   lower bound (unlike the behavioral notebook) — the acoustic peak itself is
   *included*. Lower bound is `SAMPLE_T0 = 40` (t=0, word onset), matching
   `early_perceptual_windows.py`; the bootstrap grid actually starts at sample 0
   (= t−0.4 s, pre-onset baseline), which we exclude to avoid spurious pre-onset
   candidates. Upper bound is left at the pair-level `PAIR_SMAX` (all windows);
   a per-window `post_word_offset` flag is emitted for reference.
3. **Significance = bootstrap CI excludes zero** on `mean_diff_raw`, mirroring the
   behavioral notebook. (The acoustic bootstrap also carries a behavior-controlled
   null `mean_diff_aligned_null` → an `emp_p` is *available* but not the primary
   criterion here.)
4. **Descriptive only — no decoder-window placement.** The acoustic decoder window
   already exists (causal6 `phon_smin`/`phon_smax`), so there is no
   `acoustic_transfer`-analog to feed. Output drops the `behav_decoder_smin/smax`
   / `narrower_than_decoder` fields.

## Disanalogies vs `behavioral_discriminative_windows.py`

| aspect | behavioral | acoustic companion |
|---|---|---|
| input bootstrap | `b4_bootstrap.parquet` (perceptual) | `b4_acoustic_bootstrap.parquet` (step) |
| driver / per-cell table | `b4_per_cell.parquet` | `b4_acoustic_per_cell.parquet` |
| cell filter | manifest `behav @late` cells only | **all** acoustic-ok cells (no manifest filter) |
| candidate lower bound | `smin >= phon_smax` (post-acoustic) | `smin >= SAMPLE_T0` (onset; **includes** acoustic peak) |
| value / reference | `mean_diff_raw`, /n/−/d/ | `mean_diff_raw` (= aligned; s_hi−s_lo, step order fixes polarity) |
| fallback | yes (annotation asserts presence) | **no** by default (`use_fallback=False`) — unfiltered discovery, matches early_perceptual / Type1 precedent |
| decoder placement | yes | no |
| extra columns | — | `s_lo`, `s_hi` (extreme steps) |

**Open judgment call for review:** the `use_fallback` default. Behavioral uses a
fallback because every `behav @late` cell was hand-annotated as *having* a late
response, so a window should always be emitted. The acoustic pass has no such
per-cell prior (it processes every acoustic-ok cell), so a fallback would emit a
window for cells with no reliable acoustic contrast. Defaulting **off** matches
`early_perceptual_windows.py` and its Type1 acoustic section. Parameterized so it
can be flipped to a faithful behavioral mirror.

## Shared-logic extraction

`_fallback_run` moved from `behavioral_discriminative_windows.py` into `_windows.py`
(alongside `_find_maximal_runs`, `_window_sign`) so all three window-discovery
notebooks import it. Behavioral behavior is bit-identical (pure move).

## Outputs

- `outputs/causal46_joined/acoustic_discriminative_windows/ad_windows.parquet`
- `outputs/causal46_joined/acoustic_discriminative_windows/ad_windows_bootstrap.parquet`
- `.../notebook.ipynb`, `.../ad_windows_summary.pdf` (QC)

`ad_windows.parquet` schema mirrors `b_windows` minus decoder fields, plus
`s_lo`/`s_hi`; β columns named `beta_acoustic_*`.
