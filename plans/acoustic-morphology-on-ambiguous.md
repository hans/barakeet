## Acoustic response morphology on ambiguous inputs
At acoustically selective electrodes (responding at amplitude X for /d/ and Y for /n/, X > Y), what happens on ambiguous trials? Three competing accounts: (1) responses are categorical — each trial elicits X or Y as if the input were unambiguous; (2) responses are intermediate — ambiguous items elicit something like (X+Y)/2, a fixed blend regardless of exact step; (3) responses are graded and track within-ambiguous acoustics — if steps 3 and 4 are both ambiguous, the response to step 3 is still larger than to step 4, reflecting the finer acoustic gradient.

Adjudicate via model comparison on the acoustically selective sites. This is a question about how the auditory cortex encodes inputs that fall between learned categories — does it commit, hedge uniformly, or faithfully track sub-categorical acoustic detail?

## Implementation approach

The primary analysis uses `all_outcomes.parquet` from `acoustic_decoding_single_electrode`,
which applies the endpoint-trained acoustic decoder to ALL trials including ambiguous steps.
Decoder confidence (`abs(decoder_proba - 0.5)`) on ambiguous trials distinguishes:
- **Categorical**: confidence maintained on ambiguous steps (decoder commits)
- **Intermediate or graded**: confidence collapses toward chance (decoder confused)

A secondary measure — ROC-AUC of acoustic decoder predicting `behavior_categorical_forced`
on ambiguous trials — tests whether the representation aligns with or dissociates from percept.

## Follow-up: graded vs. intermediate

The primary analysis does not distinguish graded from intermediate (both predict confidence
collapse). To distinguish: inspect step-wise mean `decoder_proba` trajectory on ambiguous
trials in the collapsed-confidence regime. Graded predicts a monotonic gradient (step 2 > 3
> 4 > 5 in confidence); intermediate predicts all ambiguous steps collapse equally. Readable
from the step-wise confidence figure (Fig 1 of the analysis) with no additional computation.
