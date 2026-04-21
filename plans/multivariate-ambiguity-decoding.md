# Sliding-window ambiguity decoding

## Objective

Characterize the temporal profile of the population-level ambiguity signal in acoustically-selective STG sites. Univariate ambiguity selectivity is null or near-null; this analysis tests when ambiguity is decodable from the population and whether it re-emerges around the perceptual response.

## Method

Sliding-window multivariate decoder contrasting behaviorally ambiguous vs. unambiguous trials. Mirror the existing multivariate temporal dissociation pipeline (acoustic and perceptual decoders) for all shared components.

### Reuse from existing pipeline

- Feature construction at each window endpoint.
- Window size, stride, time range, classifier, CV scheme.
- Group-level inference (cluster-based permutation across time, across subjects).

Do not re-specify these here. Reference the existing decoder scripts.

### Ambiguity labels

- Reuse `PaperData.get_ambiguous_steps`.
- If the step → trial-label mapping is not already a standalone helper, factor it out. Do not duplicate the logic.

### Electrode set

- Acoustically-selective sites only.
- Held fixed across all windows, including those past the acoustic epoch.

### Time range

- Acoustic onset through the post-POD range used by the perceptual decoder. The ambiguity and perceptual decoders must cover the same windows so their time courses are directly comparable.

## Design decisions

### Phoneme-pair handling

- **Primary:** fit per phoneme pair, average AUC curves within subject, group stats across subjects.
- **Fallback:** subject-level pooling across pairs with pair as a nuisance covariate. Build this as an available mode from the start; switch to it only if per-pair curves are floor-bound on trial count.

### Class imbalance

- Stratified CV with class weights set to inverse frequency.
- Report AUC. Do not report accuracy.
- Class balance varies across subjects by construction: temporal profile and within-subject effects are interpretable, cross-subject magnitude differences are not.

## Planned temporal measurements

Pre-specified; compute all three regardless of which cluster structure emerges.

1. **Onset latency.** First cluster-significant window above chance. Report alongside acoustic and perceptual onset latencies from the temporal dissociation analysis.
2. **Peak time and above-chance duration.**
3. **Late re-emergence.** Planned test for a second above-chance cluster near or after POD. Treat as a planned contrast, not post-hoc.

## Acoustic-confound control

- **Primary analysis:** binary ambiguous vs. unambiguous decoder.
- **Confirmatory control:** decoder predicting report-distribution entropy per step, residualized on step-mean HGA. Supports the discussion; not a primary result.

## Figures

Three-panel figure making the univariate/multivariate dissociation explicit:

- (a) Univariate ambiguity selectivity map across sites (null result).
- (b) Sliding-window multivariate AUC with cluster-significant regions.
- (c) Best-single-site AUC overlaid on population AUC.

## Out of scope

- Feature set extension to perceptually-selective sites (separate analysis, separate electrode set).
- Cross-temporal generalization of the ambiguity decoder (possible follow-up; not this analysis).