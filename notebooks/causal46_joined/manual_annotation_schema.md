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

> **Schema rev:** This document was rewritten 2026-05-28 to match the current
> CSV. The earlier scheme used a binary `behav response at ac` flag + a
> separate `polarity coefficient` column; the current scheme collapses these
> two into a single tuning letter in each timing column (see below).

## What each star-plot panel shows

Each page of `b4_powered.pdf` shows one cell — `subject × electrode ×
phoneme_pair × word_end` — split into two stacked panels:

- **Upper panel — acoustic contrast.** Trial-averaged HGA traces for the two
  unambiguous endpoint sounds of the phoneme pair (e.g. clear /b/ vs clear
  /m/ for the `bm` pair). The "acoustic window" is the time region where the
  decoder peaks; this is the reference window for the timing annotations.
- **Lower panel — within-completion behavioral contrast.** Trial-averaged HGA
  for "heard X" vs "heard Y" trials, restricted to ambiguous steps and a
  single lexical completion (the row's `word_end`). Acoustics are matched
  across the two report categories within `word_end`, so any HGA difference
  reflects perceptual report, not acoustic input.

## Manual annotation columns

Empty cell = implicit "no" (e.g. blank in `behav @late` means no late
behavioral response) UNLESS otherwise noted in the row below.

| Column | Values | Meaning |
|---|---|---|
| `acoustic tuning` | phoneme letter (`b`/`m`/`d`/`n`/`p`), `both`/`complex`/`two peaks`, empty | Which end of the acoustic spectrum elicits higher HGA in the **upper panel**. Single-letter = clean single-peaked acoustic response. `both` / `complex` / `two peaks` are functionally synonymous: an acoustic response exists but doesn't reduce to one dominant tuning (two distinct peaks, mixed morphology, or too noisy to tell). Empty = not annotated; in 75/77 empty rows this co-occurs with no behavioral response (the annotator didn't bother judging acoustic tuning when there was nothing to compare against). Empty here does NOT strictly mean "no acoustic peak." |
| `behav @ac` | phoneme letter, empty | Lower-panel behavioral contrast is temporally aligned with the acoustic best-window in the upper panel. **The letter encodes which behavioral category (`heard X`) shows higher HGA**, not just presence. Empty = true negative (no behavioral peak in the acoustic window). |
| `behav @ac slightly late` | phoneme letter, empty | Behavioral contrast within roughly one window-stride past the acoustic best-window — still close to the acoustic event. Same letter convention. |
| `behav @late` | phoneme letter, multi-peak notation, empty | Behavioral contrast clearly removed from the acoustic best-window (well after it). Multi-peak notation `"X, then Y"` (comma optional) means two distinct behavioral peaks within the late window, listed earliest → latest, each token being the tuning of that peak. Multi-peak notation is currently observed only in this column (the @ac and @ac-slightly-late windows are too short to host two distinguishable peaks). |
| `later contrast also present in unambiguous?` | phoneme letter, `partial X`, `second, X`, `NA, then X`, empty | Only meaningful when `behav @late` is non-empty. Asks: is the same morphology as the late behavioral contrast (timing/shape; tuning may match or flip) also visible in the **unambiguous (upper) panel**? Letter = tuning of the matching unambiguous contrast. `partial X` = only one of the behavioral responses in the row has a corresponding unambiguous contrast (tuning X). `second, X` / `NA, then X` = multi-peak parallel to `behav @late`'s multi-peak notation; identifies which behavioral peak has the match (`NA` = no match for that position) and its unambiguous tuning. Empty when `behav @late` is non-empty = true negative (looked, no match). When `behav @late` is also empty, this column should be ignored. |
| `sus — behav at ac but tuning differs` | `y`, empty | Flag for "suspicious": fires only when `behav @ac` is non-empty. `y` = the behavioral tuning at the acoustic window differs from `acoustic tuning` — i.e. the cell has a behavioral and an acoustic response in the same window but with opposite polarities. Conceptually shouldn't happen often if behavioral and acoustic responses share a population code. **Cross-check:** the column is filled by hand and one mismatch case (EC287 e109 pb penecillin: `acoustic=b`, `behav @ac=p`) was missed; programmatic users should derive this column from the two tuning columns rather than trust the manual flag. |
| `matched/mirrored morphology across WE?` | see lexicon below | **Group-level property** (identical for both WE rows of a given `subject × electrode × phoneme_pair`). Asks: when a behavioral response emerges at BOTH word-ends for the same site/pair, does the shape (timing, envelope) of the behavioral response look similar across the two word-ends? Polarity is allowed to flip — the `matched` vs `mirrored` distinction captures the polarity relationship; the `@bin` suffix captures the timing locus. Empty = no behavioral response on one (or both) WEs, so there's nothing to compare. Verified pair-consistency: all 88 site×pair groups have identical values across their two WE rows. |
| `comments` | free text | Free-text observations. Recurring tags worth treating as enums when present: `viz` = "good candidate to visualize in the paper"; `"this is a good example of leaning on the other WE — the signal is weak here but it's nicely mirrored in the other WE"` = flags cells whose case rests on cross-WE consistency. |

### `matched/mirrored` value lexicon

Two orthogonal axes encoded into one string:

**Polarity axis** — does the behavioral tuning at the matched morphology have the same or opposite sign across word-ends?
- `matched…` family → same tuning across WEs
- `mirrored…` family → opposite tuning across WEs

**Timing axis** — where does the matched morphology live?
- `@ac` → at the acoustic window
- `@slightly late` → at the slightly-late window
- `@late` → at the late window
- bare (no suffix) or `all` → matching applies across all bins where a behavioral response exists
- `sustain` → matching is a long sustained chunk, not bin-locked

Distinct values observed (counts):

| value | n | normalized meaning |
|---|---|---|
| `n` | 20 | no match |
| `matched @late` | 14 | same-polarity match, late bin |
| `matched @ac` | 8 | same-polarity match, ac bin |
| `mirrored @late` | 8 | opposite-polarity match, late bin |
| `mirrored @ac` | 4 | opposite-polarity match, ac bin |
| `matched all` | 4 | same-polarity match across all bins |
| `matched @slightly late` | 4 | same-polarity match, slightly-late bin |
| `mirrored` | 4 | opposite-polarity match across bins |
| `y` | 4 | yes; sign axis not specified (mix of matched and mirrored — see normalization note) |
| `mirrored @slightly late` | 2 | opposite-polarity match, slightly-late bin |
| `matched` | 2 | same-polarity match (equivalent to `matched all`) |
| `matched @ac, matched @slightly late` | 2 | same-polarity match at two specific bins |
| `roughly matched` | 2 | same-polarity match, weaker / qualified |
| `matched sustain` | 2 | same-polarity match in a sustained chunk |
| `yes, long sustained contrast in both` | 2 | synonym of `matched sustain` |

**Normalization for programmatic use** (recommended):
- Treat `matched`, `matched all`, `matched sustain`, `roughly matched`, `yes, long sustained contrast in both` as `match_polarity = "matched"`, `bin = "all"` (with `roughly` and `sustain` retained as flags).
- Treat `y` as `match_polarity = "unspecified"` — resolve to `matched` / `mirrored` by inspecting the per-WE tunings (EC278 e105 → mirrored, EC278 e121 → matched, etc.).
- Parse `mode @bin` strings with regex; collapse comma-joined entries to a set.

**Implicit-negative rule for multi-peak rows**: when a row has multiple
behavioral responses but `matched/mirrored` only specifies a match for one
of them (e.g. `matched @late` for a row that has both `@ac` and `@late`
behav responses), the unspecified bin is an implicit negative — the
behavioral response at that bin does NOT have a matching counterpart at the
other WE.

### Caveats / loose conventions

- **Timing bins are not mutually exclusive.** A row can have non-empty
  values in multiple timing columns. Case-by-case interpretation: sometimes
  two distinct behavioral peaks (one in each window), sometimes one broad /
  sustained response spanning multiple bins. The columns alone don't say
  which.
- **Multi-peak notation is positional, not statistical.** `"d, then n"`
  means the annotator saw two distinguishable peaks visually; this is not a
  bootstrapped or thresholded judgment. Comma is optional.
- **`sus` should be derived, not trusted.** It's a manual flag and at least
  one true-positive was missed. For analysis, compute `sus_derived = (behav
  @ac is single-letter) AND (acoustic tuning is single-letter) AND (behav
  @ac != acoustic tuning)`.
- **`acoustic tuning` empty does not strictly mean "no acoustic peak."** In
  ~97% of empty cases there's also no behavioral response, suggesting the
  annotator skipped judging acoustic tuning when there was nothing to
  compare it against. Don't read these as confirmed acoustic-null.
- **`later contrast also present in unambiguous?` is meaningless when
  `behav @late` is empty.** Five rows have a non-empty entry here while
  their own `behav @late` is empty; the annotator was filling it based on
  the paired WE's late behavioral response. Filter these out unless that
  semantics is what you want.

## Motivating intuitions behind the schema

- The acoustic vs perceptual distinction is the central project question
  (see `CLAUDE.md`); the timing annotations (`@ac` / `@ac slightly late` /
  `@late`) break the behavioral contrast into bins relative to the acoustic
  response, which is the natural axis for asking "is the percept code
  locked to the acoustic code, or temporally distinct from it?"
- Embedding the tuning letter directly into each timing column (rather than
  a separate `polarity coefficient`) lets a single row carry **different
  tunings at different times**: e.g. `behav @ac = d, behav @late = n`
  flags a sign-flip across time at the same site/word-end.
- `later contrast also present in unambiguous?` is the ambiguity-dependence
  test on a per-site basis: a late response that's present on unambiguous
  trials too is probably a general processing signal, not specifically a
  disambiguation signal.
- `matched/mirrored morphology across WE` was added late, after noticing
  that many behavioral effects fail to reach per-cell significance
  individually but show clearly correlated morphology across both
  word-ends of the same site/pair. This is informal evidence that a
  cross-WE-aware statistical test (rather than per-cell testing) might be
  more sensitive — and motivated
  `docs/superpowers/plans/2026-05-27-causal46-cross-we-pooled-test.md`.

## Suggested additional annotations to consider

Ordered roughly by likely value for downstream analysis. (Carried over from
prior schema rev and updated.)

1. **Behavioral response magnitude bin** (`weak` / `medium` / `strong`).
   The timing columns now carry tuning but not magnitude. A bin would let
   downstream analysis weight cells by visual confidence independent of the
   statistical pipeline's effect size.
2. **Acoustic response complexity** as an explicit enum (`single` /
   `double` / `sustained` / `none`). The current `both` / `complex` /
   `two peaks` trio is functionally one bucket; splitting it would let us
   partition the non-single-peaked cells substantively. Worth promoting
   `two peaks` to a real category if its current 2-row footprint is just
   the user not yet annotating it widely.
3. **Behavioral response shape** (`transient` / `sustained`). Distinguishes
   short-lived percept signals from sustained tonic responses.
4. **Behavioral latency relative to acoustic peak**, in coarse bins (e.g.
   `precedes` / `coincides` / `0–100 ms after` / `100–300 ms after` /
   `≥300 ms after`). Finer than the current three timing flags and
   recoverable to time-locked questions ("does latency cluster around
   POD?").
5. **Number of distinct behavioral peaks** in each timing bin (1, 2, ≥3).
   The current multi-peak notation in `behav @late` already implicitly
   carries this; promoting to an explicit count would make it grep-able.
6. **Cross-phoneme-pair generalization** at the same electrode. Some
   electrodes have multiple phoneme pairs annotated (e.g. EC253 e21 has
   `bm`, `dn`, `pb`). One flag per electrode: does the same morphological
   pattern appear across pairs?
7. **Per-step consistency within ambiguous trials.** Is the behavioral
   contrast consistent across the qualifying ambiguous steps (3, 4, …) or
   driven by one step? The star plot shows each step; this could be a
   visual quality call.
8. **Eyeball significance** independent of any statistics (`yes` /
   `borderline` / `no`). Lets us calibrate the statistical pipeline against
   expert intuition. The implicit version is "did the annotator bother
   filling any behav column" but an explicit grade would be cleaner.
9. **Trial-balance artifact flags** (`y` / empty). Cells where one class
   looks driven by a small number of trials, or with very different
   pre-onset baselines.
10. **`viz` worthiness as a real column** rather than a comment, so the
    annotator can flag candidates uniformly.

## Suggested CSV cleanups (for easier programmatic processing)

Low-effort changes that would simplify parsing without losing information:

- Rename columns to be regex-friendly: replace ` `, `@`, `/`, `—`, `?` with
  `_`. Suggested target names: `behav_at_ac`, `behav_at_ac_slightly_late`,
  `behav_at_late`, `unambig_match_late`, `sus_behav_ac_tuning_mismatch`,
  `matched_mirrored_across_we`, `acoustic_tuning`.
- Normalize `matched/mirrored` lexicon to two columns: `match_polarity ∈
  {matched, mirrored, unspecified, n}` and `match_bin` as a set-valued
  field (`{ac, slightly_late, late, all, sustain}`).
- Multi-peak notation: split `behav @late` and `later contrast also
  present in unambiguous?` into a long-form companion table when more than
  one peak is annotated, with one row per peak. Keep the wide
  representation for human review; derive the long form for analysis.
- Backfill the missed `sus` entry (EC287 e109 pb penecillin) or replace
  the manual column with a derived one in analysis code.
- Promote recurring `comments` tags (`viz`, the cross-WE-leaning phrase)
  to dedicated boolean columns.

## Patterns the annotation is trying to surface

From the annotator's own running notes:

1. **Mirrored-tuning sites.** Many sites have opposite-tuning behavioral
   responses at the two word-ends (same `subject × electrode ×
   phoneme_pair`). Worth understanding the spatial distribution of these
   sites, and whether the WE that gets the "positive" sign is predicted by
   the site's acoustic tuning.
2. **Cross-WE pooling for significance.** When deciding informally whether
   to count an effect as real, the annotator combines evidence across the
   two word-ends. The statistical pipeline currently does not. There may
   be a way to gain real sensitivity by adding a cross-WE-aware test —
   already in motion: see
   `docs/superpowers/plans/2026-05-27-causal46-cross-we-pooled-test.md`.
3. **Ambiguity-independence prediction.** When acoustic tuning and
   word-end "agree" in some sense, behavioral responses may be visible on
   unambiguous trials too — i.e. ambiguity-independence may be predicted
   by the acoustic-tuning / word-end relationship.
