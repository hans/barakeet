## Acoustic response morphology on ambiguous inputs

**Updated finding:** Most single-electrode acoustic responses are categorical — electrodes
commit even on ambiguous input. The question is no longer "categorical vs. graded at single
electrodes" but rather: can the system detect ambiguity, and at what level of organization?

## Scientific questions

### Q1: Single electrodes have limited continuum coverage
A categorical single electrode can only discriminate adjacent steps that straddle its
own PSE. Most electrode PSEs cluster at steps 3–4, so on average, single-electrode
adjacent-step discrimination (AX) peaks at 3v4 and drops off at the edges (1v2, 5v6 —
both steps fall on the same side of most electrodes' boundaries).

This establishes the negative result: single electrodes cannot uniformly resolve
fine-grained acoustic differences across the full continuum. The positive counterpart
(multivariate decoder achieves uniform AX) belongs in the multivariate gradient
perception analysis.

### Q2: Gradient for graded perceptual resolution
Even if single electrodes are categorical, does the population preserve enough
within-ambiguous gradient to support graded perceptual resolution (e.g., more /d/
choices at step 2 than step 3)? This connects to the multivariate gradient perception
analysis: if the acoustic population code is graded (via heterogeneous categorical
responses), that provides the input substrate for graded perceptual coding downstream.

## Implementation approach

### Adjacent-step discrimination (AX)
Use `all_outcomes.parquet` from `acoustic_decoding_single_electrode` (endpoint-trained
acoustic decoder applied to all trials). For each adjacent step pair (1v2, 2v3, ...,
5v6), compute ROC-AUC of decoder_proba distinguishing the two steps:

- **Single electrode:** AX per electrode, then average across electrodes per subject ×
  phoneme pair. Expect peaked curve (best at 3v4, poor at edges).

Report the average single-electrode AX curve (discrimination vs. continuum position).

### Confidence analysis (secondary)
Decoder confidence (`abs(decoder_proba - 0.5)`) on ambiguous trials at single-electrode
level should remain high (categorical commitment). At the population level,
inter-electrode disagreement (variance of decoder predictions across electrodes on the
same trial) indexes ambiguity.

## Coordination with multivariate gradient perception

These two analyses must be narrated together. If individual acoustic responses are
categorical but the population is graded, that's an emergence story: categorical
single-site responses + heterogeneous PSEs → graded population code. If individual
responses were already graded, the population result would be less surprising.
