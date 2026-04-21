## ambig-only vs. all analysis

Core question
Within each electrode, does simple-decoder AUC differ between ambiguous and unambiguous trials? How does that difference distribute across electrodes?

Key design choice: the "simple decoder"
Our main decoder includes stimulus step as a control predictor to isolate perceptual from acoustic variance. For this analysis, we want a simple decoder that predicts behavior from HGA alone, without the stimulus-step control.
Rationale: including stimulus step makes decoding on unambiguous trials trivial (step ≈ behavior there), saturating AUC at ceiling for mechanical reasons. Dropping the control gives us a uniform measurement — "how well does HGA predict behavioral response" — that ranges freely on both trial types and permits within-electrode comparison across trial types. We accept that simple-decoder AUC on unambig trials conflates acoustic and perceptual coding; the window-level analyses already establish that the late window is temporally and functionally distinct from early acoustic responses, which carries the interpretive load here.

Measurements per electrode

Using the simple decoder architecture, compute held-out AUC separately in two conditions:

AUC_a: decoder evaluated on ambiguous trials
AUC_u: decoder evaluated on unambiguous trials

Estimate these values within CV folds; we’ll compute their difference across folds

Sanity checks / tips

“Ambiguous” is defined behaviorally; see the stuff in `PaperData.get_ambig_steps`
On ambiguous trials, confirm simple-decoder AUC is similar to the controlled-decoder AUC at the per-electrode level
Use class-balanced AUC or downsampling as appropriate.
Note that ambig trials are typically fewer than unambig trials — AUC_a will have wider CIs than AUC_u by default. Bootstrap CIs on the difference should reflect this honestly.

Outputs

A per-electrode table with AUC_a, AUC_u, AUC_a − AUC_u, per CV fold, and trial counts per condition
A summary of how AUC_a − AUC_u distributes across electrodes: central tendency, spread, and how many electrodes show CIs on the difference that exclude zero in either direction
A scatter of (AUC_u, AUC_a) across electrodes with error bars, with the unity line marked

Scope notes

Use the electrodes and time windows selected by the perceptual decoding analysis (output from prepare_neurometrics)
Keep the existing decoders untouched; this analysis is additive, not a replacement
Do not attempt per-electrode categorization into discrete bins (ambig_only / both / etc.); report the continuous difference