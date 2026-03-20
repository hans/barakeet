Design sliding-window multivariate decoders of both acoustic and perceptual targets.
How to include electrodes at each time window? This could be noisy given diffuseness over time of perceptual response.
Window width? Here the diffuseness of the perceptual response also matters.

## Analysis 1: Sliding-window temporal dissociation
Goal: reproduce the univariate acoustic/perceptual timing dissociation and cross-decoding
results at the multivariate level, and characterize **how the population geometry transforms**
from acoustic to perceptual coding — not just *when*.

### Decoding design
100ms windows as a default, with 50ms and 150ms as robustness checks — with only 5-10
electrodes per subject/phoneme pair, per-feature SNR matters more than maximizing temporal
resolution. Electrode inclusion via overall responsiveness to the relevant contrast
(collapsed across time) to reduce the feature set without time-specific circularity;
regularized decoder (L1/elastic net) handles irrelevant electrodes at any given window.
Key predictions: peak acoustic decoding and peak perceptual decoding occur at different
latencies; at each peak time, the other target should not decode well, replicating the
double dissociation from the univariate cross-decoding analysis.
Report that the dissociation holds across window widths for a stronger claim.

### Population geometry: rotation trajectory
The value-add over the univariate analysis is characterizing whether the population code
*rotates* (acoustic and perceptual codes occupy different subspaces) or merely *scales*
(same direction, different magnitude). This is the multivariate analogue of the univariate
code-inconsistency finding.

**Anchor:** v_acoustic = decoder weight vector at the peak acoustic window. This is stable
because acoustic responses are temporally synchronous across electrodes.

**Trajectory:** At every sliding window, extract the decoder weight vector and compute its
angle relative to v_acoustic (cosine similarity). The perceptual response is temporally
diffuse, so rather than picking a single v_perceptual, track the full rotation trajectory.
Expected pattern: small angles early (acoustic-like geometry), gradual rotation as the
perceptual response emerges, plateau at maximum angle near/after POD.

**Key outcomes:**
- If angles stay near zero but magnitude changes → scaling (same code re-expressed). Would
  support reactivation, which univariate code-inconsistency already argues against.
- If angles increase over time → rotation (genuinely distinct subspaces). Strengthens
  local-disambiguation: co-localized but geometrically separable populations.
- Gradual vs. abrupt rotation is itself informative: gradual suggests a continuous
  transformation; abrupt suggests a discrete switch between coding regimes.

**Summary statistic:** For reporting, extract a consensus v_perceptual via SVD on stacked
weight vectors from the perceptual epoch (POD ± 200ms), and report the single angle between
v_acoustic and this consensus axis. This collapses the trajectory to one number per
subject × phoneme pair.

## Analysis 2: Population-level gradient perception
Goal: ask whether graded perceptual behavior (cf. McMurray et al. 2008) is recoverable from the population of perceptually selective sites, even though individual electrodes appear categorical.
Design: use a single wider time window encompassing the acoustic response epoch. Features are the full single-trial HGA time course within this window at each electrode, concatenated across electrodes to form a spatiotemporal feature vector (electrodes × time points). Apply multivariate decoding or dimensionality reduction to this representation and ask whether the population spatiotemporal pattern varies continuously with stimulus step on ambiguous trials. Regularization handles the high dimensionality relative to trial count.
If a population readout reconstructs the psychometric function's gradient shape from an ensemble of individually categorical responses, that's evidence for a distributed code where categorical responses occur at single sites and gradience emerges at the population level.
