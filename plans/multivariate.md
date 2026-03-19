Design sliding-window multivariate decoders of both acoustic and perceptual targets.
How to include electrodes at each time window? This could be noisy given diffuseness over time of perceptual response.
Window width? Here the diffuseness of the perceptual response also matters.

## Analysis 1: Sliding-window temporal dissociation
Goal: reproduce the univariate acoustic/perceptual timing dissociation and cross-decoding results at the multivariate level. Decode both acoustic identity and perceptual report using a sliding-window multivariate decoder across time.
Design: 100ms windows as a default, with 50ms and 150ms as robustness checks — with only 5-10 electrodes per subject/phoneme pair, per-feature SNR matters more than maximizing temporal resolution. Electrode inclusion via overall responsiveness to the relevant contrast (collapsed across time) to reduce the feature set without time-specific circularity; regularized decoder (L1/elastic net) handles irrelevant electrodes at any given window. Key predictions: peak acoustic decoding and peak perceptual decoding occur at different latencies; at each peak time, the other target should not decode well, replicating the double dissociation from the univariate cross-decoding analysis.
Report that the dissociation holds across window widths for a stronger claim.

## Analysis 2: Population-level gradient perception
Goal: ask whether graded perceptual behavior (cf. McMurray et al. 2008) is recoverable from the population of perceptually selective sites, even though individual electrodes appear categorical.
Design: use a single wider time window encompassing the acoustic response epoch. Features are the full single-trial HGA time course within this window at each electrode, concatenated across electrodes to form a spatiotemporal feature vector (electrodes × time points). Apply multivariate decoding or dimensionality reduction to this representation and ask whether the population spatiotemporal pattern varies continuously with stimulus step on ambiguous trials. Regularization handles the high dimensionality relative to trial count.
If a population readout reconstructs the psychometric function's gradient shape from an ensemble of individually categorical responses, that's evidence for a distributed code where categorical responses occur at single sites and gradience emerges at the population level.
