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
#               → all_outcomes.parquet (used by A_predictions and A_neurometrics)
#
#   A_predictions
#       Extracts fine-grained phonetic predictions from the acoustic searchlight.
#       → behavior_to_phonetic_decoding.parquet (consumed by prepare_neurometrics)
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
            ),
        )


rule acoustic_decoding_single_electrode:
    """Acoustic-category searchlight decoding on all speech-responsive electrodes.

    For each (electrode × phoneme_pair × time window), fits an acoustic-category
    decoder on morph-step endpoint epochs (resampled ∈ {1, 6}) and evaluates
    predictions on all relevant epochs.

    Key outputs:
      all_outcomes.parquet → A_predictions (behavior_to_phonetic_decoding.parquet)
                             A_neurometrics (phonetic_decoder_checkpoints)
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
            ),
        )


rule A_predictions:
    """Collect fine-grained phonetic predictions from the acoustic searchlight.

    Loads `all_outcomes.parquet` from acoustic_decoding_single_electrode
    for all subjects, filters to the `categorical_acoustic_cue` measure, and saves
    a combined parquet used by prepare_neurometrics as `phon_predictions_path`.

    Output:
      behavior_to_phonetic_decoding.parquet → prepare_neurometrics phon_predictions_path
    """
    input:
        notebook         = "notebooks/causal5/A_predictions.py",
        behav_acoustic_paths = expand(
            "outputs/causal5/acoustic_decoding_single_electrode/{subject}/all_outcomes.parquet",
            subject=config["data"]["subjects"],
        ),

    output:
        notebook                     = "outputs/causal5/A_predictions/notebook.ipynb",
        behavior_to_phonetic_decoding = "outputs/causal5/A_predictions/behavior_to_phonetic_decoding.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            dict(
                behav_p_searchlight_paths=list(input.behav_acoustic_paths),
                outdir=str(outdir),
            ),
        )


rule prepare_neurometrics:
    """Pre-compute PaperData for A_neurometrics visualisations.

    Runs the slow extract_hga_windows_df step and derives early_polarity /
    late_polarity so that A_neurometrics can load precomputed parquets instead
    of rerunning them.

    Inputs come from:
      - behavior_decoding_single_electrode_summarize (A-predictions.parquet)
      - A_predictions (behavior_to_phonetic_decoding.parquet)
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
        phon_predictions = "outputs/causal5/A_predictions/behavior_to_phonetic_decoding.parquet",
        electrode_paths = expand(
            "outputs/causal5/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"],
        ),
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

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                A_behav_predictions=list(input.A_behav_predictions),
                phon_predictions_path=str(input.phon_predictions),
                electrode_paths=list(input.electrode_paths),

                phon_response_tmin_min=0.0,
                all_response_tmax_max=1.3,

                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                epoch_tmin=config["analysis"]["epoch_tmin"],

                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                behav_response_peak_threshold=config["analysis"]["behav_response_peak_threshold"],
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
                hga_window_source=config["analysis"]["hga_window_source"],

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


rule causal5_all:
    """Run the full causal5 pipeline end-to-end."""
    input:
        "outputs/causal5/A_neurometrics/notebook.ipynb",
