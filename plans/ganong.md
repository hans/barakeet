## **Targeting the Ganong effect**

Current within-completion analyses maximize internal validity (acoustics are perfectly matched on all decoder inputs) but leave minimal behavioral variance to explain — completion \+ step already predict behavior near-ceiling in most subjects.

Dropping the within-completion constraint lets us decode the Ganong effect itself: the shift in perceptual boundary across completions. Much more variance, much more power, but opens the door to acoustic confounds since completions differ acoustically.

Possible mitigations:

* TRF residualization  
* temporal dissociation (perceptual effects at different latencies than acoustic encoding)  
* restricting to ambiguous steps where target acoustics are matched and the Ganong effect is largest  
  * we already know that some responses are specific to ambiguous steps, so there’s a good chance we’d find an interaction effect — Ganong only on certain acoustic steps, where completion is balanced  
  * look for completion \* step interactions that track behavioral sigmoid rather than tracking raw acoustic differences

Conservative strategy to explore:

* Keep current pipeline as electrode selection \+ first wave of results  
* Given that filtered electrode set showing perceptual response, ask whether they also show a Ganong effect

Story framing: within-completion results establish that STG carries perceptual codes dissociated from acoustics; across-completion Ganong analyses show these same sites carry signatures of lexical context biasing perception at a larger scale. First validates the interpretation of the second.

Open questions: TRF implementation details given electrode coverage, whether this is a new analysis section or reshapes the narrative, and whether the acoustic controls are convincing enough for reviewers.

---

## Acoustic controls

The Ganong decoder pools across completions, so post-POD acoustic differences between
"-esolate" and "-ecessary" are a direct confound. These arrive at the same latency as
the perceptual response, so simple temporal dissociation (early vs. late window) does
not help — the confound and the signal of interest overlap in time.

### Control 1: Ambiguity dependence
Post-POD acoustic differences between completions are constant across acoustic steps:
step 2 of "-esolate" has exactly the same "-esolate" completion as step 5 of
"-esolate." So if the Ganong decoder's accuracy varies systematically with step —
largest at ambiguous steps (3–4), weak or absent at endpoints (1, 6) — that
interaction cannot be explained by completion acoustics alone.

Operationalize: fit the Ganong decoder (or evaluate a single decoder) per step, and
test whether decoding accuracy tracks the behavioral Ganong magnitude. A completion ×
step interaction matching the behavioral sigmoid shape is the key signature.

### Control 2: Ambiguity-specific sites
We already know that many perceptual responses are specific to ambiguous trials (80/110
emerge only on ambiguous steps). If Ganong decoding is driven preferentially by these
ambiguity-specific sites, that is hard to explain acoustically — pure acoustic
responses to completion differences should not depend on whether the initial consonant
was ambiguous.

Operationalize: split the electrode set into ambiguity-dependent perceptual sites
(significant PSI only on ambiguous trials) and ambiguity-independent sites. Test
whether the Ganong effect is concentrated in the ambiguity-dependent set.

### Control 3: TRF residualization (stretch goal)
TRF-residualize HGA against the acoustic envelope / spectrogram before running the
Ganong decoder. If the Ganong effect survives removal of stimulus-driven acoustic
variance, completion acoustics are unlikely to explain it. Treat as confirmatory rather
than primary evidence, given TRF implementation complexity.

---

## Operationalizing "same signals" as within-completion perceptual responses

The key claim is not just that the same *electrodes* show both within-completion
perceptual effects and Ganong effects, but that the same *neural code* underlies both.
Electrode overlap is necessary but not sufficient — the same site could host two
independent signals.

### Test 1: Weight-vector similarity
Compare the decoder weight vectors (or LDA axes) from the within-completion behavioral
decoder and the Ganong decoder at matched time windows. Compute cosine similarity. If
the same code drives both, the weight vectors should be correlated (same electrodes
contribute with same sign and relative magnitude). Permutation test: shuffle electrode
labels and recompute cosine similarity to build a null.

### Test 2: Cross-decoder transfer
Train the within-completion decoder, apply it to across-completion trials (without
retraining), and ask whether its predictions track the Ganong boundary shift. If the
within-completion code generalizes to the Ganong contrast, the signals share a common
code. Conversely, train the Ganong decoder and evaluate within-completion: does it
predict trial-level perceptual variance on acoustically matched trials?

### Test 3: Selectivity correlation
For each electrode, compute the within-completion PSI and the across-completion Ganong
selectivity (e.g., AUC or d' for decoding behavior pooled across completions, minus
the within-completion component). Correlate across electrodes. If the same population
drives both, electrodes with stronger within-completion perceptual selectivity should
also show stronger Ganong effects.

### Minimum viable claim
At minimum, report Test 1 (weight similarity) and Test 2 (cross-decoder transfer).
Test 3 is supplementary. If weight vectors are uncorrelated or cross-transfer fails,
the claim must be weakened to "co-localized but potentially distinct signals."