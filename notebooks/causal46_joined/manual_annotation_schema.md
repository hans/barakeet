# Manual annotation schema — `b4_powered.pdf` star plots

Schema for hand-coded review of the per-site star-plot gallery produced by
`notebooks/causal46_joined/t_tests.py`
(`outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered.pdf`).

The annotated CSV lives at
`~/freesurfer_subjects/barakeet/causal46_pipeline/filtered_manifest.csv`
(one row per `subject × electrode × phoneme_pair × word_end` cell, matching the
auto-generated `filtered_manifest.csv` written by `t_tests.py`). Hand-added
columns are listed below; auto columns (e.g. `best_smin`, `best_emp_p_aligned`)
come from the script.

## What each star-plot panel shows

Each page of `b4_powered.pdf` shows one cell — `subject × electrode ×
phoneme_pair × word_end` — split into two stacked panels:

- **Upper panel — acoustic contrast.** Trial-averaged HGA traces for the two
  unambiguous endpoint sounds of the phoneme pair (e.g. clear /b/ vs clear
  /m/ for the `bm` pair). The "acoustic window" is the time region where the
  decoder peaks; this is the reference window for several annotation
  questions.
- **Lower panel — within-completion behavioral contrast.** Trial-averaged HGA
  for "heard X" vs "heard Y" trials, restricted to ambiguous steps and a
  single lexical completion (the row's `word_end`). Acoustics are matched
  across the two report categories within `word_end`, so any HGA difference
  reflects perceptual report, not acoustic input.

## Manual annotation columns

Empty cell = implicit "no" (e.g. blank in `behave response late` means no
late behavioral response) UNLESS otherwise noted in the row below.

| Column | Values | Meaning |
|---|---|---|
| `behav response at ac` | `y` / empty | Lower-panel behavioral contrast is temporally aligned with the acoustic contrast (best-window) in the upper panel. |
| `behav response at ac slightly late` | `y` / empty | Behavioral contrast within roughly one window-stride past the acoustic best-window — still close to the acoustic event. |
| `behave response late` | `y` / empty | Behavioral contrast clearly removed from the acoustic best-window (well after it). *(NB: column name has a typo — `behave` not `behav`.)* |
| `polarity coefficient` | `1`, `-1`, `?`, `both`/`BOTH`, empty | For cells with single-peaked acoustic AND behavioral contrasts where one trace is clearly above the other in both panels: does behavioral tuning match acoustic tuning? `1` = same tuning; `-1` = opposite tuning; `?` = single-peak assumption violated in one or both panels; `both`/`BOTH` = behavioral response is biphasic with peaks of opposing polarity. *(NB: column name has a trailing space.)* |
| `later contrast also present in unambiguous?` | `y` / empty | Filled ONLY when `behave response late`='y'. Is the same morphology as the late behavioral contrast (timing/shape; tuning may match or mirror) also visible in the unambiguous (upper) panel? Targets the broader question: is the late behavioral response unique to ambiguous trials, or does it appear on all trials? |
| `acoustic tuning` | phoneme letter (`b`/`m`/`d`/`n`/`p`), `?`, empty | Which end of the acoustic spectrum elicits the higher HGA response in the upper panel. `?` = there is an acoustic response but no clean single-peak structure (multi-peaked or mixed). Empty = not annotated because there was no behavioral response in this row, so the annotator didn't bother judging acoustic tuning — empty here does NOT mean "no acoustic peak". |
| `sus — behav at ac but polarity differs` | `y` / empty | Flag for "suspicious": the behavioral response IS aligned with the acoustic window AND its polarity does not match the acoustic tuning. Verified: only fires when `behav response at ac`='y' (does not extend to `slightly late`). |
| `matched/mirrored morphology across WE?` | `y` / `n` / empty | Cross-row property (within `subject × electrode × phoneme_pair`): when a behavioral response emerges at BOTH word-ends for the same site/pair, does the morphology of the behavioral response (timing, shape) look similar across the two `word_end`s? Polarity may match or flip; what's being judged is curve shape, not sign. Empty = no behavioral response on one (or both) sides, so there's nothing to compare. |
| `comments` | free text | Free-text observations. Recurring phrase: *"this is a good example of leaning on the other WE — the signal is weak here but it's nicely mirrored in the other WE"* — flags cells whose statistical case rests on the cross-WE consistency, not the per-cell effect. |

### Caveats / loose conventions

- **Timing bins are not mutually exclusive.** A row can have `y` in multiple
  timing columns. Case-by-case interpretation: sometimes two distinct
  behavioral peaks (one in each window), sometimes one broad/sustained
  response spanning multiple bins. The flags alone don't tell you which.
- **Polarity coefficient with multiple timing bins flagged: no firm rule.**
  A signed `1`/`-1` does not strictly guarantee that all flagged peaks
  share that polarity. `both`/`BOTH` is used when bipolar behavior is
  visually obvious; otherwise the annotator picked a representative
  polarity. Programmatic users should not assume single-polarity means all
  peaks agree.

## Motivating intuitions behind the schema

- The acoustic vs perceptual distinction is the central project question (see
  `CLAUDE.md`); the timing annotations (`at ac` / `slightly late` / `late`)
  break the behavioral contrast into bins relative to the acoustic response,
  which is the natural axis for asking "is the percept code locked to the
  acoustic code, or temporally distinct from it?"
- `polarity coefficient` operationalizes the "do the two responses share a
  population code" question at a single electrode — same tuning suggests
  shared population; opposite tuning suggests something more complex.
- `later contrast also present in unambiguous?` is the ambiguity-dependence
  test on a per-site basis: a late response that's present on unambiguous
  trials too is probably a general processing signal, not specifically a
  disambiguation signal.
- `matched/mirrored morphology across WE` was added late, after noticing that
  many behavioral effects fail to reach per-cell significance individually
  but show clearly correlated morphology across both word-ends of the same
  site/pair. This is informal evidence that a cross-WE-aware statistical
  test (rather than per-cell testing) might be more sensitive.

## Suggested additional annotations to consider

Ordered roughly by likely value for downstream analysis.

1. **Behavioral response magnitude bin** (`weak` / `medium` / `strong`). The
   three timing columns are binary; they collapse a huge range of effect
   sizes. A bin would let downstream analysis weight cells by visual
   confidence independent of the statistical pipeline's effect size.
2. **Acoustic response complexity** (`single` / `double` / `sustained` /
   `none`). The current `polarity coefficient` schema assumes single-peaked
   acoustic AND behavioral responses; many cells violate this (`?` is the
   dump). Naming the violation type would let us partition the `?` cells
   into substantively different categories.
3. **Behavioral response shape** (`transient` / `sustained`). Distinguishes
   short-lived percept signals from sustained tonic responses.
4. **Behavioral latency relative to acoustic peak**, in coarse bins (e.g.
   `precedes` / `coincides` / `0–100 ms after` / `100–300 ms after` /
   `≥300 ms after`). Finer than the current 3 timing flags and recoverable
   to time-locked questions ("does latency cluster around POD?").
5. **Number of distinct behavioral peaks** (1, 2, ≥3). Useful for
   identifying biphasic responses (`both`/`BOTH` in polarity) and
   sequential responses (acoustic-aligned + late).
6. **Cross-phoneme-pair generalization** at the same electrode. Some
   electrodes have multiple phoneme pairs annotated (e.g. EC253 e21 has
   `bm`, `dn`, `pb`). One flag per electrode: does the same morphological
   pattern appear across pairs?
7. **"Morphological-mirroring" sub-type for matched-WE cells**:
   `polarity-preserved` (same sign across WEs) vs `polarity-flipped`
   (opposite sign, same curve shape). The current `matched/mirrored` flag
   pools both; separating them connects to the analysis hypothesis below
   (signs are roughly random across WEs even when |effects| co-vary).
8. **Per-step consistency within ambiguous trials.** Is the behavioral
   contrast consistent across the qualifying ambiguous steps (3, 4, …) or
   driven by one step? The star plot shows each step; this could be a
   visual quality call.
9. **Eyeball significance**: independent of any statistics, would you call
   this effect real? (`yes` / `borderline` / `no`). Lets us calibrate the
   statistical pipeline against expert intuition.
10. **Trial-balance artifact flags** (`y` / empty). Cells where one class
    looks driven by a small number of trials, or with very different
    pre-onset baselines.

## Patterns the annotation is trying to surface

From the annotator's own running notes:

1. **Mirrored-tuning sites.** Many sites have `polarity=1` on one `word_end`
   and `polarity=-1` on the other (same `subject × electrode ×
   phoneme_pair`). Worth understanding the spatial distribution of these
   sites, and whether the WE that gets `polarity=1` is predicted by the
   acoustic tuning of the site.
2. **Cross-WE pooling for significance.** When deciding informally whether
   to count an effect as real, the annotator combines evidence across the
   two word-ends. The statistical pipeline currently does not. There may be
   a way to gain real sensitivity by adding a cross-WE-aware test.
3. **Hypothesis.** When acoustic tuning and word-end "agree" in some sense,
   behavioral responses are visible on unambiguous trials too — i.e.
   ambiguity-independence may be predicted by the acoustic-tuning /
   word-end relationship.
