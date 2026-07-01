# Project: Neural mechanisms of perceptual disambiguation in speech cortex

## Motivation
When listeners hear ambiguous speech sounds, they revise their interpretation as later
context arrives. This project investigates how the superior temporal gyrus (STG)
implements this revision process using intracranial ECoG recordings from human
participants.

## Core experimental design
- Single-word perception task with a /d/-to-/n/ acoustic continuum (6 steps)
- Step 1 = clear /d/, step 6 = clear /n/, steps 2-5 = progressively ambiguous
- Each continuum token is followed by one of two lexical completions:
  "-esolate" (making "desolate") or "-ecessary" (making "necessary")
- Lexical completion disambiguates the initial sound for ambiguous steps
- Participants report via button press whether they heard /d/ or /n/
- Key behavioral phenomenon: endpoints elicit consistent reports; middle steps
  (especially 3-4) elicit variable reports across trials (the Ganong effect)

## Recording and signal
- High-density electrocorticography (ECoG) from epilepsy surgery patients
- 10 participants total
- Primary signal: high-gamma activity (HGA), power in 70-150 Hz band
- Analysis focused on speech-responsive electrodes in STG and surrounding cortex

## Two core neural responses
The project distinguishes two responses observable at single electrodes:

1. **Acoustic response**: Tracks the physical acoustic cue (/d/ vs /n/).
   - Measured on unambiguous trials (steps 1 and 6)
   - Transient, peaks ~150-250ms post word onset
   - 64 electrodes across 10 participants show significant acoustic selectivity

2. **Perceptual response**: Tracks the participant's reported percept.
   - Measured on ambiguous trials (steps 3-4), where matched stimuli elicit
     different reports across trials
   - CRITICAL: computed within-completion (e.g., only -ecessary trials) so that
     neural differences reflect percept, not acoustic differences between completions
   - Temporally diffuse, peaks near or after point of disambiguation (POD),
     can extend beyond word offset
   - 58 of 64 acoustic sites (91%) also show perceptual selectivity

## Key indices
- **ASI (Acoustic Selectivity Index)**: HGA difference between clear /d/ and /n/
  at each electrode and timepoint. Positive = stronger response to one sound.
- **PSI (Perceptual Selectivity Index)**: HGA difference between "heard /d/" and
  "heard /n/" trials on acoustically matched ambiguous stimuli, computed
  within-completion. Positive = stronger response to one reported percept.

## Decoding framework
- Univariate single-electrode decoders using windowed HGA
- Each electrode gets an **acoustic window** (peak acoustic decoding) and a
  **perceptual window** (peak perceptual decoding)
- **Cross-decoding**: Hold prediction target constant, swap time window.
  Tests whether the two windows encode the same information. (They don't.)
- **Transfer analysis**: Train decoder on one target in one window, apply to the
  other window, evaluate on the other target. Tests whether the neural CODE is
  consistent. (It often isn't — transfer is bimodal across electrodes.)

## Acoustic tuning
- Each electrode is assigned an acoustic tuning (/d/ or /n/) based on which
  sound elicits stronger HGA in its acoustic window
- Similarly assigned a perceptual tuning from its perceptual window
- There is coarse co-location: acoustic /n/ sites tend to show perceptual
  responses to -ecessary; acoustic tuning predicts perceptual tuning direction
- But transfer analysis shows the detailed codes often diverge

## Theoretical framing
The project evaluates four candidate mechanisms:
1. **Reactivation**: Same code re-expressed later. Tentatively disfavored by code
   inconsistency (transfer bimodality). The "ambiguity dependence" argument that
   previously also counted against it rested on a claim now retracted as unreliable
   (perceptual responses emerging only for ambiguous trials); whether a single
   belief-driven generator accounts for both ambiguous and unambiguous responses is
   an OPEN question under active test, not ruled out.
2. **Interactive processing**: Top-down feedback overwrites acoustic representation.
   RULED OUT by code inconsistency (if same population, codes should match).
3. **Distal integration**: Perceptual resolution in a different brain region.
   RULED OUT by co-localization (91% overlap).
4. **Local disambiguation**: Functionally distinct populations, co-located in STG,
   one for acoustic encoding and one for perceptual resolution. SUPPORTED.

Open question: whether local disambiguation reflects a binary mechanism or graded
belief-updating (surprisal/prediction error). Not distinguishable in this design.

## Analysis conventions
- All perceptual analyses use within-completion contrasts to control acoustics
- Ambiguous steps = steps where participant gives both /d/ and /n/ reports
  (typically steps 2-5, most variable at 3-4)
- Unambiguous steps = endpoints (steps 1 and 6)
- Statistical tests on electrode populations use chi-squared for categorical
  comparisons, permutation tests for decoding significance
- When designing new analyses, always consider whether acoustic confounds between
  the two completions (-ecessary vs -esolate) could drive effects. The
  within-completion constraint is the primary defense; TRF residualization is
  available as an additional control.

## Key design principles for new analyses
- Prefer analyses that can dissociate acoustic from perceptual contributions
- Always check: could this effect be driven by acoustic differences between
  completions rather than perceptual state?
- The within-completion constraint is non-negotiable for perceptual claims
- When comparing across completions, use TRF residualization or demonstrate
  that timing/selectivity patterns rule out acoustic confounds
- Individual electrode examples are important for grounding population results
- Maintain distinction between what the data show (code inconsistency) and
  what we infer (distinct populations) — the latter is speculative

---

## Code structure

### Standard pipeline: causal6 + causal46_joined
`workflows/causal6.Snakefile` (acoustic decoding + speech-responsive selection;
produces `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet`) and
`workflows/causal46_joined.Snakefile` (the joined acoustic + within-completion
perceptual analyses) are the live pipeline. Notebooks live in
`notebooks/causal46_joined/` (Jupytext percent-format .py). **causal5 and causal4
are defunct** — kept only for reference; the causal5 run-order table below is
retained as a schema/protocol reference where the two pipelines still share
`src/` code.

**Within-completion subsampling (canonical).** The per-step class-balance rule
(B3 single-step / B4 across-step) that underlies every within-completion
perceptual contrast — star-plot galleries, bootstrap t-tests, early-window and
strong-generator analyses — is defined **authoritatively in the module docstring
of `notebooks/causal46_joined/_within_completion.py`** (imported by 14+
notebooks). Read it before touching any B3/B4 trial-selection code. Pointer +
consumer map: `docs/superpowers/plans/2026-07-01-causal46-within-completion-subsampling.md`.

**Legacy causal5 run order** (`notebooks/causal5/` — Jupytext percent-format .py files; retained for reference):

| Rule | Notebook | Key outputs |
|------|----------|-------------|
| `find_speech_responsive` | `find_speech_responsive.py` | `outputs/causal5/find_speech_responsive/{subject}_results.csv` |
| `behavior_decoding_single_electrode` | `behavior_decoding_single_electrode.py` | `{subject}/results.joblib` → keys `decoding_results`, `decoders` |
| `behavior_decoding_single_electrode_summarize` | `behavior_decoding_single_electrode_summarize.py` | `A-predictions.parquet` (late/perceptual window), `A_early-predictions.parquet` (early/acoustic), `A_results.csv`, `A_final_summary.csv` |
| `acoustic_decoding_single_electrode` | `acoustic_decoding_single_electrode.py` | `all_outcomes.parquet`, `outcomes.parquet`, `decoding_models.joblib` |
| `acoustic_decoding_peaks` | `acoustic_decoding_peaks.py` | `phon_peaks_df.parquet`, `phon_roc_auc_searchlight_df.parquet` (peak acoustic window per site; loads `all_outcomes.parquet` per subject and filters to `categorical_acoustic_cue` on the fly) |
| `acoustic_morphology_on_ambiguous` | `acoustic_morphology_on_ambiguous.py` | `trial_df.parquet`, `site_stats.parquet` (decoder confidence on ambiguous trials) |
| `prepare_neurometrics` | `prepare_neurometrics.py` | 13+ parquets in `outputs/causal5/prepare_neurometrics/` (see below) |
| `A_neurometrics` | `A_neurometrics.py` | Figures; `hga_zoomin_search_keys.csv` |

All causal5 outputs live under `outputs/causal5/`.

### Where to find structure documentation before reading source

Check these docstrings/constants **before** tracing through notebook code — they are
the canonical reference for data schemas and protocols:

- **HGA extraction**: `src/viz_paper.py:extract_hga_windows_df()` docstring; `src/data.py` module docstring
- **hga_df schema**: `src/viz_paper.py` `PaperData` docstring (~lines 124–155) — site/trial identifiers, `hga_early`/`hga_late`, window metadata
- **Epoch metadata columns**: `src/data.py:add_metadata_features()` docstring — `resampled`, `behavior_categorical_forced`, `ambiguity`, `categorical_acoustic_cue`, etc.
- **Decoder checkpoint formats**: `src/models/decoding.py` module docstring
- **Timing constants**: `src/stimuli.py` — `POD_dict`, `OFFSET_DICT`, `WORD_PHASES`
- **Within-completion B3/B4 subsampling**: `notebooks/causal46_joined/_within_completion.py` module docstring — canonical per-step class-balance rule (both classes bootstrapped with replacement; gallery and t-test share draws)
- **all_outcomes.parquet schema**: columns `subject, electrode_idx, phoneme_pair, smin, smax, measure, epoch_idx, fold, decoder_target, decoder_proba, decoder_prediction`; `measure` ∈ {`categorical_acoustic_cue`, `subject_specific_acoustics`}; predictions on ALL trials including ambiguous steps

### Key source files

**`src/models/decoding.py`** — Core decoding logic. See module docstring for
checkpoint formats. Key functions:
- `run_decoding_model_comparison_population()` — behavioral decoders (baseline vs full model)
- `run_decoding_searchlight_single_electrode()` — acoustic searchlight
- `fit_train_test()` — core CV loop with stratification

**`src/data.py`** — Data loading and feature engineering.
- `get_electrode_df(subject)` — electrode positions/anatomy from `.mat` file
- `add_metadata_features(md)` — adds all derived columns to epoch metadata:
  `categorical_acoustic_cue`, `behavior_categorical`, `ambiguity`, `mismatch`,
  `belief_update`, `label_acoustic/lexical/behavior`, POD per phoneme pair

**`src/stimuli.py`** — Critical constants:
- `POD_dict` — point of disambiguation in seconds: `{'bm': 0.28, 'dn': 0.295, 'pb': 0.21}`
- `OFFSET_DICT` — word end times (e.g. `'desolate': 0.498`, `'necessary': 0.887`)
- `WORD_PHASES` — named time windows (acoustic, POD, offset) per word

**`src/viz_paper.py`** — `PaperData` dataclass + all paper figure functions.
`PaperData` holds all precomputed parquets; loaded at the top of `A_neurometrics`.

**`src/data_cleaning.py`** — Post-hoc cleaning of decoding results (ROI relabeling,
polarity, TRF scores). Used primarily in causal4 legacy code.

**`src/figure_builder.py`** — Incremental matplotlib figure staging for slide builds.

**`src/analysis/`, `src/encoding/`** — Symlinks to the `big-trf` project (TRF
residualization and encoding model utilities).

### Config
`config.yaml` at root; these are passed to various processing notebooks

### prepare_neurometrics outputs (PaperData parquets)
- `electrode_df` — electrode metadata + ROI
- `plot_{phon,behav}_{phon,behav}_df` — 4 cross-window prediction DataFrames
- `{phon,behav}_peaks_df` — peak decoding windows per site
- `{phon,behav}_roc_auc_searchlight_df` — fold-level ROC-AUC
- `early_polarity`, `late_polarity` — response polarities
- `hga_df` — extracted HGA amplitudes per site/window
- `all_md` — combined epoch metadata across subjects
- `zoomin_keys` — sentinel file (also used as input to A_neurometrics rule)

### Epochs
Raw preprocessed epochs at `outputs/epochs_preprocessed/{subject}_epo.fif`.
Loaded via MNE; metadata enriched with `add_metadata_features()`.
Phoneme pairs: `bm` (/b/-/m/), `dn` (/d/-/n/), `pb` (/p/-/b/).

### Environment
Conda environment: `/scratch/jgauthier/transformers3`
Activate before running any notebooks or scripts: `conda activate /scratch/jgauthier/transformers3`
Or run directly: `conda run -p /scratch/jgauthier/transformers3 <command>` (use `-p`, not `-n`)