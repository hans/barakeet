# Context: causal46_joined analysis pipeline

Ubiquitous language for the `causal46_joined` pipeline — the joined acoustic +
within-completion perceptual analyses that run downstream of `causal6`
(`workflows/causal46_joined.Snakefile`, `causal46_joined_all` target). This
glossary **extends** the domain description in `CLAUDE.md`; it does not restate
the acoustic/perceptual response distinction, ASI/PSI, or the four candidate
mechanisms. Where a term already lives in `CLAUDE.md`, the entry here only
sharpens the pipeline-specific usage.

## Language

### Electrode selection

**AS site** _(acoustic-significant site)_:
An electrode with a statistically reliable acoustic response. The pipeline
restricts every within-completion perceptual test to AS sites so the
per-electrode multiple-comparison family is small enough to survive FDR
correction. Significance is collapsed to the electrode level (an electrode is AS
if it is significant for *any* phoneme pair).
_Avoid_: acoustic electrode, significant-acoustic electrode.

**Site**:
An (electrode × phoneme pair) unit. One electrode can be several sites. Distinct
from a *cell*, which additionally fixes the completion.

### Trial grouping

**Cell** _(B4 cell)_:
The finest unit of a within-completion perceptual contrast: (subject, electrode,
phoneme pair, completion). Because the completion is fixed within a cell, the
contrast is within-completion by construction (suffix acoustics are held
constant). Canonical trial-selection rule lives in
`notebooks/causal46_joined/_within_completion.py`.
_Avoid_: unit, group.

**B3 cell**:
A cell further restricted to a single ambiguous step. The cleanest acoustic
control (no cross-step mixing) at the cost of fewer trials; B4 is the default.

**Completion** _(word end)_:
The lexical continuation that disambiguates the initial phoneme and fixes the
suffix acoustics — e.g. `-esolate` → "desolate" (/d/) vs `-ecessary` →
"necessary" (/n/). Fixed within a cell.
_Avoid_: word_end (code spelling; prose uses "completion"), suffix, lexical
frame.

**Qualifying steps**:
The ambiguous steps of a cell that carry both reported percepts with enough
trials to contrast. Endpoints (steps 1 and 6) are never qualifying steps.

**Per-step class balance**:
The rule that makes a within-completion percept contrast acoustically clean:
within each qualifying step, both percept classes contribute the same number of
trials, so any step-acoustic effect is common-mode and cancels in the contrast.
The authoritative statement is the `_within_completion.py` module docstring.

**Trial balance index**:
The site-by-step table of per-class trial counts (the tabulated substrate of
per-step class balance) that determines which cells and steps are qualifying.
Consumed as a shared input by the perceptual, acoustic-on-ambiguous, and
mismatch analyses.
_Avoid_: trial counts table, balance table.

### Decoders

**Behavior decoding**:
A single-electrode decoder of the reported percept computed *within completion*.
The perceptual-side complement to acoustic decoding.
_Avoid_: percept decoding, choice decoding.

**Ganong decoding**:
The same reported-percept target as behavior decoding, but *pooled across both
completions* rather than split by completion. Pooling exposes the Ganong lexical
boundary shift riding on top of the acoustic step. Contrast with behavior
decoding, which never pools completions.

**Full decoder** _(with resampled control)_:
A decoder whose predictors include a control for the acoustic step, so its
readout reflects neural signal beyond the step itself.
_Avoid_: with-control decoder, baseline+HGA decoder.

**HGA-only decoder**:
The variant that decodes from high-gamma activity alone, with no acoustic-step
control predictor. Tuned independently from the full decoder.

### Windows — decoder-peak family

Windows located as the *peak of a univariate decoder*. These are the windows
`CLAUDE.md` already names; the entries below only fix vocabulary.

**Acoustic window**:
The per-electrode peak *acoustic-decoding* window (as in `CLAUDE.md`).
_Avoid_: phonemic peak window, phon-peak window — these are the same object;
prefer "acoustic window" in prose.

**Perceptual window**:
The per-electrode peak *perceptual-decoding* window (as in `CLAUDE.md`). This is
a decoder-peak window and is **not** the same object as a bootstrap-contrast
window (below), even when both are "perceptual".

### Windows — bootstrap-contrast family

A separate family of windows discovered from a per-cell HGA bootstrap
confidence interval (not from a decoder peak). Time regions where the
within-cell HGA contrast is reliably non-zero. Always name the specific member;
never call one a "perceptual window" or "acoustic window".

**Endpoint-acoustic window** _(code: `a_windows`)_:
Bootstrap-contrast window for the endpoint acoustic contrast (step 6 − step 1,
unambiguous trials).

**Acoustic-discriminative window** _(code: `ad_windows`)_:
Bootstrap-contrast window for the acoustic-step contrast measured *on ambiguous
trials* with percept held balanced (see s_hi vs s_lo).

**Behaviorally-discriminative window** _(code: `b_windows`)_:
Bootstrap-contrast window for the within-completion percept contrast, searched
*beyond* the acoustic peak.
_Avoid_: behavioral window (ambiguous), late perceptual window.

**Early-perceptual window** _(code: `ep_windows`)_:
Bootstrap-contrast window for the within-completion percept contrast, searched
in the *early* region up to and including the acoustic peak. The early
counterpart of the behaviorally-discriminative window.

### Contrasts and tests

**β_ambig**:
The within-completion percept contrast for one cell in a given window — the
cell-and-window-level, bootstrap-estimated analog of the population PSI.
_Avoid_: perceptual beta, ambiguous slope.

**β_unamb**:
The endpoint acoustic contrast (step 6 − step 1, unambiguous trials) measured in
the *same* window as a β_ambig, used as its reference.

**Strong-generator test**:
A per-window comparison of β_ambig against β_unamb. It probes the open question
(see `CLAUDE.md`) of whether one belief-driven generator accounts for both
ambiguous and unambiguous responses; it does not presuppose an answer.
_Avoid_: single-generator test, reactivation test.

**s_lo / s_hi**:
The lowest and highest acoustic steps among a cell's qualifying (ambiguous)
steps. The **s_hi vs s_lo contrast** is the acoustic-step effect measured on
ambiguous trials with the reported percept held balanced — the acoustic mirror
of the within-completion percept contrast.

**Acoustic-on-ambiguous**:
The behavior-controlled acoustic-step analysis (the s_hi vs s_lo contrast run as
the structural mirror of the perceptual `t_tests`). Asks what the acoustic cue
does when the percept is balanced.

**Mismatch regression** _(opponent / conflict)_:
A single-trial model of HGA that separates the additive contributions of
acoustic step and reported percept from their *interaction*. A reliable
interaction ("conflict"/"opponent" coding) means the two do not simply add.
_Avoid_: interaction regression, surprise regression.

**Acoustic transfer**:
A test of whether an acoustic decoder trained at the acoustic window still reads
out the acoustic cue when applied in a perceptual window — i.e. whether the
acoustic *code* persists into the later window. The pipeline complement to the
transfer analysis described in `CLAUDE.md`.

### Site typology

**Site type**:
A **hand-assigned** early-window classification of a site (the `site_type_relabel`
label). Five typed categories, by whether the site shows an early acoustic
response and/or an aligned within-completion percept split: `type1_acoustic_only`
("Acoustic only"), `type2_early_perceptual` ("Acoustic+perceptual" — aligned
split at both completions), `type3_asymmetric` ("Acoustic+perceptual, one-sided"
— split at one completion only), `type4_early_perceptual_mirrored`
("Acoustic+perceptual, mirrored" — splits of opposite polarity across the two
completions), and `type5_behav_only` ("Perceptual only" — a percept split with no
acoustic response). Everything else (`A_unsigned`, `problematic`, `interesting`)
collapses to "Other". Canonical labels, ordering, and colors live in
`src/causal46_joined.py`.
_Avoid_: response type, category; and the **retired automated labels**
`grab_bag` / `complex` / `unknown` — the computed `assign_site_type` classifier
is no longer authoritative for this typology (the manual relabel is).

**Aligned / anti-aligned**:
Whether a cell's within-completion percept split runs in the same direction as
the site's acoustic tuning (aligned) or the opposite (anti-aligned). A site whose
two completions carry opposite-polarity early splits is the *mirrored* type
(`type4_early_perceptual_mirrored`).

**Early type × late type**:
The cross-classification (early site type against presence/sidedness of a late
behavioral contrast) used to organize galleries and stratify contrast plots.

### Figures and review artifacts

**Star plot**:
A per-cell figure pairing the acoustic endpoint traces with the
within-completion percept traces, so one cell's acoustic and perceptual
responses are read together.

**Gallery**:
A multi-page PDF with one star plot per page, the substrate for manual review.

**Powered / powered-significant**:
A cell is *powered* when it has enough trials per class to enter the gallery;
*powered-significant* when its bootstrap contrast additionally excludes zero.
_Avoid_: passing, qualifying (reserved for steps).

**Contrast plot**:
A population time-course overlay of the acoustic contrast against the aligned
perceptual contrast. The **by-site-type** variant draws one such plot per early
site type.

**Manual annotations**:
Human review labels attached to sites/cells (acoustic tuning, and behavioral
tags such as `behav @ac` for a split at the acoustic window vs `behav @late` for
a much later one). They gate which cells enter the bootstrap-contrast window
analyses. Canonical column meanings live in
`notebooks/causal46_joined/manual_annotation_schema.md`.

**Peak-summary flavor**:
One of the four decoder significance summaries formed by crossing the statistic
(fold-mean vs t-stat) with the correction (max-stat vs TFCE). `foldmean_maxstat`
is the v1 default that downstream rules consume.
_Avoid_: peak variant, summary type.
