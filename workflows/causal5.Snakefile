# =============================================================================
# causal5 — Distilled single-electrode behavioral decoding pipeline
# =============================================================================
#
# Pipeline overview:
#
#   find_speech_responsive
#       │  Identifies speech-responsive electrodes per subject.
#       │  (outputs in outputs/causal5/find_speech_responsive/)
#       │
#       ├─► behavior_decoding_single_electrode
#       │       Fits behavioral (choice) decoders on ALL speech-responsive
#       │       electrodes × phoneme pairs.
#       │       → results.joblib (A_decoding_results, A_decoders)
#       │
#       ├─► behavior_decoding_single_electrode_summarize
#       │       Partitions decoding results into early (acoustic) and late
#       │       (perceptual) windows; finds peak window per site.
#       │       → A-predictions.parquet, A_early-predictions.parquet
#       │
#       └─► acoustic_decoding_single_electrode
#               Acoustic-category searchlight on all speech-responsive electrodes.
#               → all_outcomes.parquet (used directly by acoustic_decoding_peaks
#                  and prepare_neurometrics; filtered to categorical_acoustic_cue
#                  on the fly — no intermediate A_predictions step)
#
#   prepare_neurometrics
#       Computes all DataFrames for neurometric visualizations (slow HGA extraction).
#       → parquet files for electrode_df, phon/behav peaks, HGA windows, polarity, etc.
#
#   A_neurometrics
#       Produces all neurometric figures from precomputed prepare_neurometrics outputs.
#
# Key simplifications vs causal4:
#   - Removed: find_As, unify_As, find_Bs, find_Cs (electrode pre-selection steps)
#   - Removed: behavior_decoding (multi-electrode population decoders)
#   - Removed: behavior_decoding_super, behavior_decoding_single_electrode_transfer
#   - Removed: neurometrics hyperparameter sweep (prepare_neurometrics_sweep,
#              A_neurometrics sweep, prepare_neurometrics_ttest, A_neurometrics_ttest)
#   - Removed: behavior_decoding_single_electrode_permutation tests
# =============================================================================

from pathlib import Path

from ploomber_engine import execute_notebook


def run_notebook(input_path: str, output_path: str, parameters, **kwargs):
    """Convert a jupytext .py notebook to .ipynb and execute it via ploomber_engine.

    Validates parameters against the notebook's declared parameters cell before
    running, so mismatches are caught early.
    """
    import tempfile

    import jupytext

    input_path = Path(input_path)

    if input_path.suffix == ".py":
        nb = jupytext.read(input_path)
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            tmp_path = Path(f.name)
        jupytext.write(nb, tmp_path)
    else:
        tmp_path = None

    try:
        actual_input = tmp_path if tmp_path is not None else input_path

        from ploomber import DAG
        from ploomber.products import File
        from ploomber.tasks import NotebookRunner

        dag = DAG(name="temp_dag")
        NotebookRunner(
            actual_input,
            File(output_path),
            dag=dag,
            params=parameters,
            static_analysis="strict",
        )
        dag.render(force=True)

        return execute_notebook(actual_input, Path(output_path), parameters=parameters)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


# =============================================================================
# Rules
# =============================================================================

rule all:
    """Default target: run the full causal5 pipeline end-to-end."""
    input:
        "outputs/causal5/A_neurometrics/notebook.ipynb",


rule find_speech_responsive:
    """Identify speech-responsive electrodes for a single subject.

    Uses a simple t-test on baseline-corrected epochs.
    """
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        notebook = "notebooks/causal5/find_speech_responsive.ipynb",

    output:
        notebook = "outputs/causal5/find_speech_responsive/{subject}.ipynb",
        results  = "outputs/causal5/find_speech_responsive/{subject}_results.csv",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=input.epochs,
                outdir=str(outdir),
            ),
        )


rule behavior_decoding_single_electrode:
    """Behavioral (choice) decoding from single electrodes — all speech-responsive sites.

    Fits a two-model comparison per (electrode × phoneme_pair):
      - baseline: logistic regression on morph-step feature only
      - full:     logistic regression on baseline + windowed HGA

    Unlike causal4, no upstream find_As step is required. Decoders are fit on
    ALL speech-responsive electrodes for every phoneme pair present in the epoch
    metadata.

    Runtime: ~hours per subject (parallelised within each site via n_jobs=5).
    """
    input:
        epochs    = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        notebook  = "notebooks/causal5/behavior_decoding_single_electrode.py",

    output:
        notebook = "outputs/causal5/behavior_decoding_single_electrode/{subject}/notebook.ipynb",
        results  = "outputs/causal5/behavior_decoding_single_electrode/{subject}/results.joblib",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],
                n_jobs=config["analysis"]["decoding"]["n_jobs"],
            ),
        )


rule behavior_decoding_single_electrode_summarize:
    """Summarise single-electrode behavioral decoding results for one subject.

    Partitions the decoded time axis into:
      - early / acoustic window  (smin <= A_max_decoding_sample)
      - late  / perceptual window (smin >= A_max_decoding_sample)

    Finds the peak decoding window per (electrode × phoneme_pair × word_end) site
    and saves trial-level predictions as parquet for prepare_neurometrics.

    Key outputs:
      A-predictions.parquet       → prepare_neurometrics A_behav_predictions
      A_early-predictions.parquet → prepare_neurometrics A_early_behav_predictions
    """
    input:
        epochs    = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        result    = "outputs/causal5/behavior_decoding_single_electrode/{subject}/results.joblib",
        notebook  = "notebooks/causal5/behavior_decoding_single_electrode_summarize.py",

    output:
        notebook         = "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/notebook.ipynb",
        A_results        = "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A_results.csv",
        A_final_summary  = "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A_final_summary.csv",
        A_predictions    = "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A-predictions.parquet",
        A_trial_analysis = "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A-trial_analysis-ensembled.csv",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                result_path=str(input.result),
                outdir=str(outdir),
                groupby=["word_end"],

                min_decoding_sample=0,
                max_decoding_sample=290,
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
            ),
        )


rule ganong_decoding_single_electrode:
    """Ganong-effect decoding from single electrodes — across-completion behavioral decoder.

    Drops the within-completion constraint of `behavior_decoding_single_electrode`:
    decodes `behavior_categorical_forced` pooled across both lexical completions.
    This exposes the systematic Ganong boundary shift alongside trial-level perceptual
    variance.

    Electrode set is restricted to sites that showed significant behavioral decoding
    in `behavior_decoding_single_electrode_summarize` (full_roc_auc >= threshold).

    Stratification on (resampled, lexical_evidence) balances acoustic steps and
    completions across CV folds without trial filtering.

    Runtime: ~hours per subject (parallelised within each site via n_jobs=5).
    """
    input:
        epochs           = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes       = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        behav_summary    = "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A_final_summary.csv",
        notebook         = "notebooks/causal5/ganong_decoding_single_electrode.py",

    output:
        notebook = "outputs/causal5/ganong_decoding/{subject}/notebook.ipynb",
        results  = "outputs/causal5/ganong_decoding/{subject}/results.joblib",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                behav_summary_path=str(input.behav_summary),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],
                n_jobs=config["analysis"]["decoding"]["n_jobs"],
                behav_peak_threshold=config["analysis"]["behav_response_peak_threshold"],
            ),
        )


rule ganong_decoding_summarize:
    """Summarise Ganong-effect decoding results across all subjects.

    Loads `results.joblib` from every subject's `ganong_decoding_single_electrode` run,
    finds the peak decoding window per (subject, electrode, phoneme_pair) site,
    and saves combined trial-level predictions and peak-window summaries.

    Key outputs:
      ganong_peaks.parquet       → per-site peak Ganong window (consumed by prepare_neurometrics)
      ganong_predictions.parquet → all trial-level predictions for transfer analyses
    """
    input:
        notebook     = "notebooks/causal5/ganong_decoding_summarize.py",
        result_paths = expand(
            "outputs/causal5/ganong_decoding/{subject}/results.joblib",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook             = "outputs/causal5/ganong_decoding/notebook.ipynb",
        ganong_peaks         = "outputs/causal5/ganong_decoding/ganong_peaks.parquet",
        ganong_predictions   = "outputs/causal5/ganong_decoding/ganong_predictions.parquet",

    run:
        outdir = "outputs/causal5/ganong_decoding"
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                outdir=outdir,
            ),
        )


rule ganong_decoding_all:
    """Run Ganong decoding + summarise for all subjects."""
    input:
        "outputs/causal5/ganong_decoding/ganong_peaks.parquet",


rule ganong_decoding_inspect_population:
    """Population-level inspection of Ganong decoding results.

    Compares Ganong decoding (across-completion) with within-completion behavioral
    decoding. Analyses: peak Δ distributions, lexical-evidence symmetry, timing
    vs. performance, and Ganong Δ vs. behavioral Δ.
    """
    input:
        notebook           = "notebooks/causal5/ganong_decoding_inspect_population.py",
        ganong_peaks       = "outputs/causal5/ganong_decoding/ganong_peaks.parquet",
        ganong_predictions = "outputs/causal5/ganong_decoding/ganong_predictions.parquet",
        behav_summaries    = expand(
            "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A_final_summary.csv",
            subject=config["data"]["subjects"],
        ),
        all_epochs         = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook = "outputs/causal5/ganong_decoding_inspect_population/notebook.ipynb",

    run:
        outdir = "outputs/causal5/ganong_decoding_inspect_population"
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                ganong_peaks_path=str(input.ganong_peaks),
                ganong_predictions_path=str(input.ganong_predictions),
                behav_summary_paths=list(input.behav_summaries),
                all_epochs=list(input.all_epochs),
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                window_size=config["analysis"]["decoding"]["window_size"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                outdir=outdir,
            ),
        )


# --------------------------------------------------------------------------- #
# HGA-only variants of the behavior and ganong decoders.
#
# These mirror the paired-comparison pipelines above but drop `resampled` as a
# control predictor (`baseline_features=[]`). Outputs live under `_hga_only`
# suffixed directories so the originals stay intact.
# --------------------------------------------------------------------------- #


rule behavior_decoding_single_electrode_hga_only:
    """Behavioral decoding — HGA-only variant (no stimulus-step control).

    Single-decoder version of `behavior_decoding_single_electrode`: trained on
    windowed HGA alone. Used downstream for ambig-vs-unambig AUC comparisons
    where the full decoder's stimulus-step prior would saturate at ceiling.
    """
    input:
        epochs    = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        notebook  = "notebooks/causal5/behavior_decoding_single_electrode_hga_only.py",

    output:
        notebook = "outputs/causal5/behavior_decoding_single_electrode_hga_only/{subject}/notebook.ipynb",
        results  = "outputs/causal5/behavior_decoding_single_electrode_hga_only/{subject}/results.joblib",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],
                n_jobs=config["analysis"]["decoding"]["n_jobs"],
            ),
        )


rule behavior_decoding_single_electrode_hga_only_summarize:
    """Summarise HGA-only behavior decoding results for one subject.

    Peak-finding uses max `full_roc_auc` (no baseline). Emits a clean-schema
    predictions parquet (`decoder_proba`/`decoder_prediction`) and an
    `A_final_summary.csv` consumed by the HGA-only ganong decoder as the site
    filter.
    """
    input:
        epochs    = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        result    = "outputs/causal5/behavior_decoding_single_electrode_hga_only/{subject}/results.joblib",
        notebook  = "notebooks/causal5/behavior_decoding_single_electrode_hga_only_summarize.py",

    output:
        notebook        = "outputs/causal5/behavior_decoding_single_electrode_hga_only_summarize/{subject}/notebook.ipynb",
        A_results       = "outputs/causal5/behavior_decoding_single_electrode_hga_only_summarize/{subject}/A_results.csv",
        A_final_summary = "outputs/causal5/behavior_decoding_single_electrode_hga_only_summarize/{subject}/A_final_summary.csv",
        A_predictions   = "outputs/causal5/behavior_decoding_single_electrode_hga_only_summarize/{subject}/A-predictions.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                result_path=str(input.result),
                outdir=str(outdir),
                groupby=["word_end"],

                min_decoding_sample=0,
                max_decoding_sample=290,
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
            ),
        )


rule ganong_decoding_hga_only:
    """Ganong-effect decoding — HGA-only variant (no stimulus-step control).

    Pools trials across both lexical completions (`groupby=None`). Filters sites
    using the HGA-only behavior summary (not the full-comparison one), keeping
    the HGA-only chain internally consistent.
    """
    input:
        epochs           = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes       = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        behav_summary    = "outputs/causal5/behavior_decoding_single_electrode_hga_only_summarize/{subject}/A_final_summary.csv",
        notebook         = "notebooks/causal5/ganong_decoding_single_electrode_hga_only.py",

    output:
        notebook = "outputs/causal5/ganong_decoding_hga_only/{subject}/notebook.ipynb",
        results  = "outputs/causal5/ganong_decoding_hga_only/{subject}/results.joblib",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.electrodes),
                behav_summary_path=str(input.behav_summary),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],
                n_jobs=config["analysis"]["decoding"]["n_jobs"],
                behav_peak_threshold=config["analysis"]["behav_response_peak_threshold"],
            ),
        )


rule ganong_decoding_hga_only_summarize:
    """Summarise HGA-only Ganong decoding results across all subjects.

    Peak-finding uses max `full_roc_auc` (no baseline). Emits clean-schema
    `ganong_predictions.parquet` and `ganong_peaks.parquet`.
    """
    input:
        notebook     = "notebooks/causal5/ganong_decoding_hga_only_summarize.py",
        result_paths = expand(
            "outputs/causal5/ganong_decoding_hga_only/{subject}/results.joblib",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook           = "outputs/causal5/ganong_decoding_hga_only/notebook.ipynb",
        ganong_peaks       = "outputs/causal5/ganong_decoding_hga_only/ganong_peaks.parquet",
        ganong_predictions = "outputs/causal5/ganong_decoding_hga_only/ganong_predictions.parquet",

    run:
        outdir = "outputs/causal5/ganong_decoding_hga_only"
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                result_paths=list(input.result_paths),
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],
                outdir=outdir,
            ),
        )


rule ganong_decoding_hga_only_all:
    """Run HGA-only Ganong decoding + summarise for all subjects."""
    input:
        "outputs/causal5/ganong_decoding_hga_only/ganong_peaks.parquet",


rule acoustic_decoding_single_electrode:
    """Acoustic-category searchlight decoding on all speech-responsive electrodes.

    For each (electrode × phoneme_pair × time window), fits an acoustic-category
    decoder on morph-step endpoint epochs (resampled ∈ {1, 6}) and evaluates
    predictions on all relevant epochs.

    Key outputs:
      all_outcomes.parquet → acoustic_decoding_peaks, prepare_neurometrics, A_neurometrics
    """
    input:
        epochs           = "outputs/epochs_preprocessed/{subject}_epo.fif",
        speech_responsive = "outputs/causal5/find_speech_responsive/{subject}_results.csv",
        notebook         = "notebooks/causal5/acoustic_decoding_single_electrode.py",

    output:
        notebook     = "outputs/causal5/acoustic_decoding_single_electrode/{subject}/notebook.ipynb",
        outcomes     = "outputs/causal5/acoustic_decoding_single_electrode/{subject}/outcomes.parquet",
        all_outcomes = "outputs/causal5/acoustic_decoding_single_electrode/{subject}/all_outcomes.parquet",
        train_scores = "outputs/causal5/acoustic_decoding_single_electrode/{subject}/train_scores.parquet",
        test_scores  = "outputs/causal5/acoustic_decoding_single_electrode/{subject}/test_scores.parquet",
        avg_scores   = "outputs/causal5/acoustic_decoding_single_electrode/{subject}/avg_test_scores.csv",
        models       = "outputs/causal5/acoustic_decoding_single_electrode/{subject}/decoding_models.joblib",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            dict(
                epochs_path=str(input.epochs),
                electrodes_path=str(input.speech_responsive),
                outdir=str(outdir),

                min_sample=config["analysis"]["decoding"]["min_sample"],
                window_size=config["analysis"]["decoding"]["window_size"],
                stride=config["analysis"]["decoding"]["stride"],
                n_jobs=config["analysis"]["decoding"]["n_jobs"],
            ),
        )


rule acoustic_decoding_peaks:
    """Compute phonetic decoder peak windows from the acoustic searchlight.

    Extracted from prepare_neurometrics so that downstream analyses (e.g.
    acoustic_morphology_on_ambiguous) can run immediately without waiting for
    the full prepare_neurometrics pipeline.

    Outputs consumed by:
      - acoustic_morphology_on_ambiguous (phon_peaks_df.parquet)
      - prepare_neurometrics (both outputs, passed as inputs)
    """
    input:
        all_outcomes = expand(
            "outputs/causal5/acoustic_decoding_single_electrode/{subject}/all_outcomes.parquet",
            subject=config["data"]["subjects"],
        ),
        notebook         = "notebooks/causal5/acoustic_decoding_peaks.py",

    output:
        notebook                 = "outputs/causal5/acoustic_decoding_peaks/notebook.ipynb",
        phon_peaks               = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet",
        phon_roc_auc_searchlight = "outputs/causal5/acoustic_decoding_peaks/phon_roc_auc_searchlight_df.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_outcomes_paths=list(input.all_outcomes),
                outdir=str(outdir),
                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                phon_response_tmin_min=0.0,
                all_response_tmax_max=1.3,
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
            ),
        )


rule acoustic_ax_discrimination:
    """Adjacent-step AX discrimination decoders at acoustically selective sites.

    For each site and each adjacent step pair (1v2, 2v3, ..., 5v6), trains a
    fresh binary decoder on the raw HGA at the site's peak acoustic window.
    Tests whether the neural signal can distinguish adjacent morph steps — the
    hallmark of categorical perception is high discrimination at the category
    boundary and low discrimination within categories.

    Separated from acoustic_morphology_on_ambiguous because decoding is slow.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        phon_peaks = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet",
        notebook   = "notebooks/causal5/acoustic_ax_discrimination.py",

    output:
        notebook             = "outputs/causal5/acoustic_ax_discrimination/notebook.ipynb",
        ax_discrimination_df = "outputs/causal5/acoustic_ax_discrimination/ax_discrimination_df.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                phon_peaks_path=str(input.phon_peaks),
                outdir=str(outdir),
                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
            ),
        )


rule acoustic_morphology_on_ambiguous:
    """Acoustic response morphology on ambiguous inputs at acoustically selective sites.

    Extracts raw HGA in each site's peak acoustic window and asks whether the
    neural response remains categorically committed on ambiguous trials (steps 2–5)
    or collapses toward the midpoint (graded account).

    Key measures per site:
      - HGA confidence on ambiguous vs. endpoint trials (|hga_norm - 0.5|)
      - ROC-AUC of raw HGA predicting behavior on ambiguous trials
        (AUC ≈ 0.5 = dissociation; AUC >> 0.5 = acoustic drives percept)
      - Sigmoid steepness k: small = categorical, large = graded

    Runs after acoustic_decoding_peaks + acoustic_ax_discrimination;
    does NOT require acoustic_decoding_single_electrode or prepare_neurometrics.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        phon_peaks = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet",
        ax_discrimination_df = "outputs/causal5/acoustic_ax_discrimination/ax_discrimination_df.parquet",
        notebook   = "notebooks/causal5/acoustic_morphology_on_ambiguous.py",

    output:
        notebook              = "outputs/causal5/acoustic_morphology_on_ambiguous/notebook.ipynb",
        site_stats            = "outputs/causal5/acoustic_morphology_on_ambiguous/site_stats.parquet",
        trial_df              = "outputs/causal5/acoustic_morphology_on_ambiguous/trial_df.parquet",
        neurometrics_clusters = "outputs/causal5/acoustic_morphology_on_ambiguous/neurometrics_clusters.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                phon_peaks_path=str(input.phon_peaks),
                ax_discrimination_path=str(input.ax_discrimination_df),
                outdir=str(outdir),
                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
            ),
        )


rule multivariate_temporal_dissociation:
    """Sliding-window multivariate temporal dissociation for one subject.

    Decodes acoustic identity and perceptual report using a sliding-window
    multivariate decoder across time. Tests whether the double dissociation
    (peak acoustic decoding early, peak perceptual decoding late) holds at
    the population level.

    NOTE: Electrode selection is a double-dip (overall responsiveness collapsed
    across time). TODO: revisit with held-out selection.
    """
    input:
        epochs        = "outputs/epochs_preprocessed/{subject}_epo.fif",
        phon_peaks    = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet",
        behav_summary = "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A_final_summary.csv",
        notebook      = "notebooks/causal5/multivariate_temporal_dissociation.py",

    output:
        notebook         = "outputs/causal5/multivariate_temporal_dissociation/{subject}/notebook.ipynb",
        acoustic_scores  = "outputs/causal5/multivariate_temporal_dissociation/{subject}/acoustic_scores.parquet",
        perceptual_scores = "outputs/causal5/multivariate_temporal_dissociation/{subject}/perceptual_scores.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                phon_peaks_path=str(input.phon_peaks),
                behav_summary_path=str(input.behav_summary),
                outdir=str(outdir),

                window_sizes=config["analysis"]["multivariate"]["window_sizes"],
                stride=config["analysis"]["multivariate"]["stride"],
                pca_num_components=config["analysis"]["multivariate"]["pca_num_components"],
                n_jobs=config["analysis"]["multivariate"]["n_jobs"],

                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                behav_response_peak_threshold=config["analysis"]["behav_response_peak_threshold"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
            ),
        )


rule multivariate_temporal_dissociation_all:
    """Run multivariate temporal dissociation for all subjects."""
    input:
        expand(
            "outputs/causal5/multivariate_temporal_dissociation/{subject}/notebook.ipynb",
            subject=config["data"]["subjects"],
        ),


rule multivariate_ambiguity_decoding:
    """Sliding-window multivariate ambiguity decoder for one subject.

    Decodes trial-level ambiguity (behaviorally ambiguous vs. unambiguous) from
    acoustically-selective STG sites per phoneme pair. Stratifies CV on the
    (resampled, word_end) composite so step identity and completion are balanced
    across folds. See plans/multivariate-ambiguity-decoding.md.
    """
    input:
        epochs     = "outputs/epochs_preprocessed/{subject}_epo.fif",
        phon_peaks = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet",
        all_md     = "outputs/causal5/prepare_neurometrics/all_md.parquet",
        notebook   = "notebooks/causal5/multivariate_ambiguity_decoding.py",

    output:
        notebook         = "outputs/causal5/multivariate_ambiguity_decoding/{subject}/notebook.ipynb",
        scores           = "outputs/causal5/multivariate_ambiguity_decoding/{subject}/scores.parquet",
        outcomes         = "outputs/causal5/multivariate_ambiguity_decoding/{subject}/outcomes.parquet",
        fold_balance     = "outputs/causal5/multivariate_ambiguity_decoding/{subject}/fold_balance.parquet",
        ambiguity_labels = "outputs/causal5/multivariate_ambiguity_decoding/{subject}/ambiguity_labels.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_path=str(input.epochs),
                phon_peaks_path=str(input.phon_peaks),
                all_md_path=str(input.all_md),
                outdir=str(outdir),

                window_sizes=config["analysis"]["multivariate"]["window_sizes"],
                stride=config["analysis"]["multivariate"]["stride"],
                pca_num_components=config["analysis"]["multivariate"]["pca_num_components"],
                n_jobs=config["analysis"]["multivariate"]["n_jobs"],

                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
                stratum_min_trials=config["analysis"]["multivariate_ambiguity"]["stratum_min_trials"],

                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
            ),
        )


rule multivariate_ambiguity_decoding_all:
    """Run ambiguity decoder for all subjects."""
    input:
        expand(
            "outputs/causal5/multivariate_ambiguity_decoding/{subject}/notebook.ipynb",
            subject=config["data"]["subjects"],
        ),


rule multivariate_ambiguity_decoding_group:
    """Group-level aggregation: subject-averaged AUC, FDR-windowed significance,
    temporal measurements, fold-balance report, two-panel figure. No cluster perm."""
    input:
        scores           = expand(
            "outputs/causal5/multivariate_ambiguity_decoding/{subject}/scores.parquet",
            subject=config["data"]["subjects"],
        ),
        fold_balance     = expand(
            "outputs/causal5/multivariate_ambiguity_decoding/{subject}/fold_balance.parquet",
            subject=config["data"]["subjects"],
        ),
        ambiguity_labels = expand(
            "outputs/causal5/multivariate_ambiguity_decoding/{subject}/ambiguity_labels.parquet",
            subject=config["data"]["subjects"],
        ),
        hga_df   = "outputs/causal5/prepare_neurometrics/hga_df.parquet",
        notebook = "notebooks/causal5/multivariate_ambiguity_decoding_group.py",

    output:
        notebook              = "outputs/causal5/multivariate_ambiguity_decoding_group/notebook.ipynb",
        group_auc             = "outputs/causal5/multivariate_ambiguity_decoding_group/group_auc.parquet",
        temporal_measurements = "outputs/causal5/multivariate_ambiguity_decoding_group/temporal_measurements.csv",
        fold_balance_report   = "outputs/causal5/multivariate_ambiguity_decoding_group/fold_balance_report.csv",
        figure                = "outputs/causal5/multivariate_ambiguity_decoding_group/figure_ambiguity_decoding.pdf",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                scores_paths=list(input.scores),
                fold_balance_paths=list(input.fold_balance),
                ambiguity_labels_paths=list(input.ambiguity_labels),
                hga_df_path=str(input.hga_df),
                outdir=str(outdir),

                behavior_imbalance_threshold=config["analysis"]["multivariate_ambiguity"]["behavior_imbalance_threshold"],
                fdr_alpha=config["analysis"]["fdr_alpha"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
            ),
        )


rule multivariate_gradient_perception:
    """Population-level gradient perception from acoustically selective sites.

    Trains logistic regression (with PCA) on endpoint trials to classify acoustic
    cue, then applies to ambiguous trials. If decoder probability tracks morph step
    continuously, that's evidence for graded acoustic population coding despite
    individually categorical electrodes.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        phon_peaks = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet",
        notebook   = "notebooks/causal5/multivariate_gradient_perception.py",

    output:
        notebook                = "outputs/causal5/multivariate_gradient_perception/notebook.ipynb",
        regression_predictions  = "outputs/causal5/multivariate_gradient_perception/regression_predictions.parquet",
        endpoint_predictions    = "outputs/causal5/multivariate_gradient_perception/endpoint_predictions.parquet",
        gradient_stats          = "outputs/causal5/multivariate_gradient_perception/gradient_stats.parquet",
        multivariate_ax          = "outputs/causal5/multivariate_gradient_perception/multivariate_ax_discrimination_df.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                epochs_paths=list(input.all_epochs),
                phon_peaks_path=str(input.phon_peaks),
                outdir=str(outdir),

                pca_num_components=config["analysis"]["multivariate"]["pca_num_components"],
                n_jobs=config["analysis"]["multivariate"]["n_jobs"],

                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
            ),
        )


rule gradient_perception_report:
    """PDF report for the multivariate gradient perception analysis."""
    input:
        regression_predictions = "outputs/causal5/multivariate_gradient_perception/regression_predictions.parquet",
        endpoint_predictions   = "outputs/causal5/multivariate_gradient_perception/endpoint_predictions.parquet",
        gradient_stats         = "outputs/causal5/multivariate_gradient_perception/gradient_stats.parquet",
        all_md                 = "outputs/causal5/prepare_neurometrics/all_md.parquet",
        script                 = "scripts/gradient_perception_report.py",

    output:
        report = "outputs/causal5/multivariate_gradient_perception/report.pdf",

    shell:
        "PYTHONPATH=. python {input.script} "
        "--data-dir outputs/causal5/multivariate_gradient_perception "
        "--all-md {input.all_md} "
        "--output {output.report}"


rule prepare_neurometrics:
    """Pre-compute PaperData for A_neurometrics visualisations.

    Runs the slow extract_hga_windows_df step and derives early_polarity /
    late_polarity so that A_neurometrics can load precomputed parquets instead
    of rerunning them.

    Inputs come from:
      - behavior_decoding_single_electrode_summarize (A-predictions.parquet)
      - acoustic_decoding_single_electrode (all_outcomes.parquet, filtered on the fly)
      - acoustic_decoding_peaks (phon_peaks_df, phon_roc_auc_searchlight_df)
      - find_speech_responsive (electrode_df)
      - raw epochs (for HGA extraction)
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        A_behav_predictions = expand(
            "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/A-predictions.parquet",
            subject=config["data"]["subjects"],
        ),
        all_outcomes = expand(
            "outputs/causal5/acoustic_decoding_single_electrode/{subject}/all_outcomes.parquet",
            subject=config["data"]["subjects"],
        ),
        phon_peaks               = "outputs/causal5/acoustic_decoding_peaks/phon_peaks_df.parquet",
        phon_roc_auc_searchlight = "outputs/causal5/acoustic_decoding_peaks/phon_roc_auc_searchlight_df.parquet",
        electrode_paths = expand(
            "outputs/causal5/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),
        ganong_peaks = "outputs/causal5/ganong_decoding/ganong_peaks.parquet",
        ganong_predictions = "outputs/causal5/ganong_decoding/ganong_predictions.parquet",
        notebook = "notebooks/causal5/prepare_neurometrics.py",

    output:
        notebook                    = "outputs/causal5/prepare_neurometrics/notebook.ipynb",
        electrode_df                = "outputs/causal5/prepare_neurometrics/electrode_df.parquet",
        plot_phon_phon_df           = "outputs/causal5/prepare_neurometrics/plot_phon_phon_df.parquet",
        plot_behav_phon_df          = "outputs/causal5/prepare_neurometrics/plot_behav_phon_df.parquet",
        plot_behav_behav_df         = "outputs/causal5/prepare_neurometrics/plot_behav_behav_df.parquet",
        plot_phon_behav_df          = "outputs/causal5/prepare_neurometrics/plot_phon_behav_df.parquet",
        behav_roc_auc_searchlight_df = "outputs/causal5/prepare_neurometrics/behav_roc_auc_searchlight_df.parquet",
        phon_roc_auc_searchlight_df  = "outputs/causal5/prepare_neurometrics/phon_roc_auc_searchlight_df.parquet",
        all_md                      = "outputs/causal5/prepare_neurometrics/all_md.parquet",
        word_end_df                 = "outputs/causal5/prepare_neurometrics/word_end_df.parquet",
        phon_peaks_df               = "outputs/causal5/prepare_neurometrics/phon_peaks_df.parquet",
        behav_peaks_df              = "outputs/causal5/prepare_neurometrics/behav_peaks_df.parquet",
        behav_peaks_df_unfiltered   = "outputs/causal5/prepare_neurometrics/behav_peaks_df_unfiltered.parquet",
        behav_baseline_df           = "outputs/causal5/prepare_neurometrics/behav_baseline_df.parquet",
        zoomin_keys                 = "outputs/causal5/prepare_neurometrics/zoomin_keys.parquet",
        early_polarity              = "outputs/causal5/prepare_neurometrics/early_polarity.parquet",
        late_polarity               = "outputs/causal5/prepare_neurometrics/late_polarity.parquet",
        hga_df                      = "outputs/causal5/prepare_neurometrics/hga_df.parquet",
        reg_df                      = "outputs/causal5/prepare_neurometrics/reg_df.parquet",
        ganong_peaks                = "outputs/causal5/prepare_neurometrics/ganong_peaks.parquet",
        ganong_predictions          = "outputs/causal5/prepare_neurometrics/ganong_predictions.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                A_behav_predictions=list(input.A_behav_predictions),
                all_outcomes_paths=list(input.all_outcomes),
                phon_peaks_path=str(input.phon_peaks),
                phon_roc_auc_searchlight_path=str(input.phon_roc_auc_searchlight),
                electrode_paths=list(input.electrode_paths),
                ganong_peaks_path=str(input.ganong_peaks),
                ganong_predictions_path=str(input.ganong_predictions),

                phon_response_tmin_min=0.0,
                all_response_tmax_max=1.3,

                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                epoch_tmin=config["analysis"]["epoch_tmin"],

                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                behav_response_peak_threshold=config["analysis"]["behav_response_peak_threshold"],
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
                hga_window_source=config["analysis"]["hga_window_source"],
                behav_peak_post_offset_s=config["analysis"]["behav_peak_post_offset_s"],

                outdir=str(outdir),
            ),
        )


rule A_neurometrics:
    """Run neurometric visualisations from precomputed prepare_neurometrics outputs.

    Loads all PaperData parquets and produces the suite of figures for the paper
    (decoding timing, cross-window transfer, HGA zoomin plots, etc.).

    Behavioral decoder checkpoints (results.joblib) and acoustic decoder outputs
    (decoding_models.joblib + parquets) are passed as parameterized lists so no
    paths are hardcoded in the notebook.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"],
        ),
        # Use zoomin_keys as the sentinel confirming prepare_neurometrics is done
        neurometrics_sentinel = "outputs/causal5/prepare_neurometrics/zoomin_keys.parquet",
        behavioral_decoder_paths = expand(
            "outputs/causal5/behavior_decoding_single_electrode/{subject}/results.joblib",
            subject=config["data"]["subjects"],
        ),
        acoustic_decoder_models = expand(
            "outputs/causal5/acoustic_decoding_single_electrode/{subject}/decoding_models.joblib",
            subject=config["data"]["subjects"],
        ),
        neurometrics_clusters = "outputs/causal5/acoustic_morphology_on_ambiguous/neurometrics_clusters.parquet",
        notebook = "notebooks/causal5/A_neurometrics.py",

    output:
        notebook                  = "outputs/causal5/A_neurometrics/notebook.ipynb",
        hga_zoomin_search_keys    = "outputs/causal5/A_neurometrics/hga_zoomin_search_keys.csv",
        phonetic_transfer_results = "outputs/causal5/A_neurometrics/phonetic_transfer_extreme_results_mean.csv",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                neurometrics_dir=str(Path(input.neurometrics_sentinel).parent),
                behavioral_decoder_paths=list(input.behavioral_decoder_paths),
                acoustic_decoder_dirs=[
                    str(Path(p).parent) for p in input.acoustic_decoder_models
                ],
                acoustic_morphology_dir=str(Path(input.neurometrics_clusters).parent),
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                textgrid_dir="textgrids",
                outdir=str(outdir),
            ),
        )


# =============================================================================
# Aggregate targets
# =============================================================================

rule behavior_decoding_single_electrode_all:
    """Run behavioral decoding + summarise for all subjects."""
    input:
        expand(
            "outputs/causal5/behavior_decoding_single_electrode_summarize/{subject}/notebook.ipynb",
            subject=config["data"]["subjects"],
        ),


rule acoustic_decoding_single_electrode_all:
    """Run acoustic searchlight for all subjects."""
    input:
        expand(
            "outputs/causal5/acoustic_decoding_single_electrode/{subject}/notebook.ipynb",
            subject=config["data"]["subjects"],
        ),


rule acoustic_morphology_on_ambiguous_all:
    """Run acoustic morphology analysis (after acoustic decoding + peak extraction)."""
    input:
        "outputs/causal5/acoustic_morphology_on_ambiguous/site_stats.parquet",


rule causal5_all:
    """Run the full causal5 pipeline end-to-end."""
    input:
        "outputs/causal5/A_neurometrics/notebook.ipynb",
