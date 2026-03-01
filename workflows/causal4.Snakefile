from itertools import product as _product
from pathlib import Path

from ploomber_engine import execute_notebook


def run_notebook(input_path: str, output_path: str, parameters, **kwargs):
    import tempfile
    import jupytext

    input_path = Path(input_path)

    # jupytext .py percent-format files must be converted to .ipynb before
    # ploomber_engine can execute them.
    if input_path.suffix == ".py":
        nb = jupytext.read(input_path)
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            tmp_path = Path(f.name)
        jupytext.write(nb, tmp_path)
    else:
        tmp_path = None

    try:
        actual_input = tmp_path if tmp_path is not None else input_path

        # First hack into the Ploomber API. Build a fake DAG so that we can validate parameters.
        from ploomber import DAG
        from ploomber.products import File
        from ploomber.tasks import NotebookRunner

        dag = DAG(name="temp_dag")
        runner = NotebookRunner(
            actual_input,
            File(output_path),
            dag=dag,
            params=parameters,
            static_analysis="strict",
        )
        # This will throw an exception if there are parameter issues (e.g. missing parameters)
        dag.render(force=True)

        # Now ditch that and run directly with `ploomber_engine`
        return execute_notebook(
            actual_input,
            Path(output_path),
            parameters=parameters,
        )
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Neurometrics hyperparameter sweep helpers
# ---------------------------------------------------------------------------

def _params_tag(phon, behav, ambig):
    """Encode threshold triple as a filesystem-safe directory name."""
    return f"p{int(phon * 100)}_b{int(behav * 1000)}_a{ambig}"


def _ttest_params_tag(phon, ambig, pval):
    """Encode ttest-based threshold triple as a filesystem-safe directory name."""
    return f"p{int(phon * 100)}_a{ambig}_bp{pval:g}"


NEUROMETRICS_TTEST_PARAMS_TAG = _ttest_params_tag(
    config["analysis"]["phon_response_peak_threshold"],
    config["analysis"]["ambiguous_response_threshold"],
    config["analysis"]["behav_ttest_pvalue_threshold"],
)

NEUROMETRICS_PARAMS_GRID = {
    _params_tag(p, b, a): dict(
        phon_response_peak_threshold=p,
        behav_response_peak_threshold=b,
        ambiguous_response_threshold=a,
    )
    for p, b, a in _product(
        config["analysis"]["sweep"]["phon_response_peak_thresholds"],
        config["analysis"]["sweep"]["behav_response_peak_thresholds"],
        config["analysis"]["sweep"]["ambiguous_response_thresholds"],
    )
}
ALL_NEUROMETRICS_PARAMS = list(NEUROMETRICS_PARAMS_GRID.keys())

# ---------------------------------------------------------------------------

# params: power threshold
rule find_speech_responsive:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        notebook = "notebooks/causal4/find_speech_responsive.ipynb"

    output:
        notebook = "outputs/causal4/find_speech_responsive/{subject}.ipynb",
        results = "outputs/causal4/find_speech_responsive/{subject}_results.csv"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            outdir=str(outdir)),
        )

# store graded output, eventually do a ttest
# TODO merge this into the pipeline and re-run everything..
rule find_speech_responsive_graded:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        notebook = "notebooks/causal4/find_speech_responsive_graded.ipynb"

    output:
        notebook = "outputs/causal4/find_speech_responsive_graded/{subject}.ipynb",
        results = "outputs/causal4/find_speech_responsive_graded/{subject}_results.csv"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            outdir=str(outdir)),
        )


# params: decoding window, stride
# decode spectrum extremes vs. all
rule find_As:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        speech_responsive = "outputs/causal4/find_speech_responsive/{subject}_results.csv",
        notebook = "notebooks/causal4/find_As.ipynb"

    output:
        notebook = "outputs/causal4/find_As/{subject}.ipynb",
        results = "outputs/causal4/find_As/{subject}_results.csv",
        decoders = "outputs/causal4/find_As/{subject}_decoders.pt"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            speech_responsive=input.speech_responsive,
                            outdir=str(outdir)),
        )


rule find_As_all:
    input:
        expand("outputs/causal4/find_As/{subject}.ipynb", subject=config["data"]["subjects"])


rule A_stepwise_trf:
    input:
        epochs = "outputs/epochs_preprocessed_{subject}_epo.fif",
        A_results = "outputs/causal4/find_As/{subject}_results.csv",
        notebook = "notebooks/causal4/A_stepwise_trf.ipynb"
    
    output:
        outdir = directory("outputs/causal4/A_stepwise_trf/{subject}"),
        notebook = "outputs/causal4/A_stepwise_trf/{subject}/notebook.ipynb",
        results_csv = "outputs/causal4/A_stepwise_trf/{subject}/results.csv",
        results_pkl = "outputs/causal4/A_stepwise_trf/{subject}/results.pkl"

    run:
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(subject=wildcards.subject,
                            epochs_path=input.epochs,
                            A_results_path=input.A_results,
                            outdir=str(output.outdir)),
        )


rule run_A_intrinsics:
    input:
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        all_results = expand("outputs/causal4/find_As/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_decoders = expand("outputs/causal4/find_As/{subject}_decoders.pt", subject=config["data"]["subjects"]),
        all_electrode_dfs = expand("outputs/causal4/find_speech_responsive/{subject}_results.csv", subject=config["data"]["subjects"]),
        notebook = "notebooks/causal4/run_A_intrinsics.ipynb"

    output:
        notebook = "outputs/causal4/run_A_intrinsics/run_A_intrinsics.ipynb",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(all_epochs=input.all_epochs,
                            all_results=input.all_results,
                            all_decoders=input.all_decoders,
                            all_electrode_dfs=input.all_electrode_dfs,
                            outdir=str(outdir)),
        )


rule compute_A_stimulus_correlations:
    input:
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        all_results = expand("outputs/causal4/find_As/{subject}_results.csv",
                            subject=config["data"]["subjects"]),
        all_decoders = expand("outputs/causal4/find_As/{subject}_decoders.pt",
                              subject=config["data"]["subjects"]),
        notebook = "notebooks/causal4/compute_A_stimulus_correlations.ipynb"

    output:
        notebook = "outputs/causal4/compute_A_stimulus_correlations/compute_A_stimulus_correlations.ipynb",
        results = "outputs/causal4/compute_A_stimulus_correlations/results.csv"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(all_epochs=input.all_epochs,
                            all_results=input.all_results,
                            all_decoders=input.all_decoders,
                            outdir=str(outdir)),
        )


# Form unions of As within-subject based on similarity in space
# and/or function.
rule unify_As:
    input:
        all_results = expand("outputs/causal4/find_As/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_decoders = expand("outputs/causal4/find_As/{subject}_decoders.pt", subject=config["data"]["subjects"]),
        all_electrode_dfs = expand("outputs/causal4/find_speech_responsive/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        notebook = "notebooks/causal4/unify_As.ipynb"

    output:
        notebook = "outputs/causal4/unify_As/unify_As.ipynb",
        results = "outputs/causal4/unify_As/results.csv",
        decoders = "outputs/causal4/unify_As/unified_decoders.pt"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(all_results=input.all_results,
                            all_decoders=input.all_decoders,
                            all_electrode_dfs=input.all_electrode_dfs,
                            all_epochs=input.all_epochs,
                            outdir=str(outdir)),
        )


rule find_Bs:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        speech_responsive = "outputs/causal4/find_speech_responsive/{subject}_results.csv",
        unified_As = "outputs/causal4/unify_As/results.csv",
        unified_A_decoders = "outputs/causal4/unify_As/unified_decoders.pt",
        notebook = "notebooks/causal4/find_Bs.ipynb"

    output:
        notebook = "outputs/causal4/find_Bs/{subject}.ipynb",
        results = "outputs/causal4/find_Bs/{subject}_results.csv"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            speech_responsive_path=input.speech_responsive,
                            As_path=input.unified_As,
                            A_decoders_path=input.unified_A_decoders,
                            outdir=str(outdir)),
        )


rule find_Bs_all:
    input:
        expand("outputs/causal4/find_Bs/{subject}.ipynb", subject=config["data"]["subjects"])


rule run_B_intrinsics:
    input:
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        all_results = expand("outputs/causal4/find_Bs/{subject}_results.csv", subject=config["data"]["subjects"]),
        notebook = "notebooks/causal4/run_B_intrinsics.ipynb"

    output:
        notebook = "outputs/causal4/run_B_intrinsics/run_B_intrinsics.ipynb",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(all_epochs=input.all_epochs,
                            all_results=input.all_results,
                            outdir=str(outdir)),
        )


rule plot_B_study:
    input:
        notebook = "notebooks/causal4/plot_B_study.ipynb",
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        textgrids = "textgrids",
        all_speech_responsive = expand("outputs/causal4/find_speech_responsive/{subject}_results.csv", subject=config["data"]["subjects"]),
        unified_As = "outputs/causal4/unify_As/results.csv",
        unified_A_decoders = "outputs/causal4/unify_As/unified_decoders.pt",
        all_results = expand("outputs/causal4/find_Bs/{subject}_results.csv", subject=config["data"]["subjects"])

    output:
        notebook = "outputs/causal4/plot_B_study/plot_B_study.ipynb",
        results = "outputs/causal4/plot_B_study/B_study_results.csv",
        pdf = "outputs/causal4/plot_B_study/B_study.pdf"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_paths=input.all_epochs,
                            tg_dir=input.textgrids,
                            electrodes_paths=input.all_speech_responsive,
                            A_result_path=input.unified_As,
                            A_decoders_path=input.unified_A_decoders,
                            all_B_result_paths=input.all_results,
                            outdir=str(outdir)),
        )


rule find_Cs:
    input:
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        all_speech_responsive = expand("outputs/causal4/find_speech_responsive/{subject}_results.csv", subject=config["data"]["subjects"]),
        unified_As = "outputs/causal4/unify_As/results.csv",
        unified_A_decoders = "outputs/causal4/unify_As/unified_decoders.pt",
        all_B_results = expand("outputs/causal4/find_Bs/{subject}_results.csv", subject=config["data"]["subjects"]),
        notebook = "notebooks/causal4/find_Cs.ipynb"

    output:
        notebook = "outputs/causal4/find_Cs/find_Cs.ipynb",
        study_results = "outputs/causal4/find_Cs/C_study_results.csv"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_paths=input.all_epochs,
                            electrodes_paths=input.all_speech_responsive,
                            A_result_path=input.unified_As,
                            A_decoders_path=input.unified_A_decoders,
                            all_B_result_paths=input.all_B_results,
                            outdir=str(outdir)),
        )


rule analyze:
    input:
        all_A_results = expand("outputs/causal4/find_As/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_A_decoders = expand("outputs/causal4/find_As/{subject}_decoders.pt", subject=config["data"]["subjects"]),
        all_B_results = expand("outputs/causal4/find_Bs/{subject}_results.csv", subject=config["data"]["subjects"]),
        annotated_B_results = "outputs/causal4/annotated_B_results.csv",
        single_electrode_decoding_results = "outputs/single_electrode_decoding/30/acoustic/scores.csv",

        trf_scores = "/userdata/jgauthier/projects/big-trf/outputs/encoder_summary/timit-no_repeats/word.csv",

        notebook = "notebooks/causal4/analyze.ipynb"

    output:
        notebook = "outputs/causal4/analyze/analyze.ipynb",
        results = "outputs/causal4/analyze/analysis_results.csv"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(all_results=input.all_results,
                            outdir=str(outdir)),
        )


rule behavior_decoding:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal4/find_speech_responsive/{subject}_results.csv",
        annotated_B_results = "outputs/causal4/annotated_B_results.csv",
        annotated_C_results = "outputs/causal4/annotated_C_results.csv",
        unified_A_results = "outputs/causal4/unify_As/results.csv",
        unified_A_decoders = "outputs/causal4/unify_As/unified_decoders.pt",
        all_A_results = "outputs/causal4/find_As/{subject}_results.csv",
        all_A_decoders = "outputs/causal4/find_As/{subject}_decoders.pt",
        notebook = "notebooks/causal4/behavior_decoding.ipynb"

    output:
        notebook = "outputs/causal4/behavior_decoding/{subject}/behavior_decoding.ipynb",
        results = "outputs/causal4/behavior_decoding/{subject}/results.pt",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            electrodes_paths=input.electrodes,
                            B_annotated_path=input.annotated_B_results,
                            C_annotated_path=input.annotated_C_results,
                            A_result_path=input.unified_A_results,
                            A_decoders_path=input.unified_A_decoders,
                            all_A_result_path=input.all_A_results,
                            all_A_decoders_path=input.all_A_decoders,
                            outdir=str(outdir)),
        )


rule behavior_decoding_single_electrode:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal4/find_speech_responsive/{subject}_results.csv",
        annotated_B_results = "outputs/causal4/annotated_B_results.csv",
        annotated_C_results = "outputs/causal4/annotated_C_results.csv",
        unified_A_results = "outputs/causal4/unify_As/results.csv",
        unified_A_decoders = "outputs/causal4/unify_As/unified_decoders.pt",
        all_A_results = "outputs/causal4/find_As/{subject}_results.csv",
        all_A_decoders = "outputs/causal4/find_As/{subject}_decoders.pt",
        notebook = "notebooks/causal4/behavior_decoding_single_electrode.ipynb"

    output:
        notebook = "outputs/causal4/behavior_decoding_single_electrode/{subject}/behavior_decoding.ipynb",
        results = "outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            electrodes_paths=input.electrodes,
                            B_annotated_path=input.annotated_B_results,
                            C_annotated_path=input.annotated_C_results,
                            A_result_path=input.unified_A_results,
                            A_decoders_path=input.unified_A_decoders,
                            all_A_result_path=input.all_A_results,
                            all_A_decoders_path=input.all_A_decoders,
                            outdir=str(outdir)),
        )

rule behavior_decoding_single_electrode_summarize:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        speech_reponsive = "outputs/causal4/find_speech_responsive/{subject}_results.csv",
        result = "outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt",

        individual_stimulus_decoder_results = "outputs/causal4/find_As/{subject}_results.csv",
        individual_stimulus_decoders = "outputs/causal4/find_As/{subject}_decoders.pt",

        unified_stimulus_decoder_results = "outputs/causal4/unify_As/results.csv",
        unified_stimulus_decoders = "outputs/causal4/unify_As/unified_decoders.pt",

        annotated_B_results = "outputs/causal4/annotated_B_results.csv",
        annotated_C_results = "outputs/causal4/annotated_C_results.csv",

        trf_results = "/userdata/jgauthier/projects/big-trf/outputs/encoder_summary/timit-no_repeats/vanilla_aud.csv",
        acoustic_decoding_scores = "outputs/single_electrode_decoding/30/acoustic/scores.csv",

        notebook = "notebooks/causal4/behavior_decoding_single_electrode_summarize.ipynb"

    output:
        notebook = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/behavior_decoding_single_electrode_summarize.ipynb",

        A_results = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_results.csv",
        A_early_results = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early_results.csv",
        B_results = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/B_results.csv",
        C_results = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/C_results.csv",

        A_final_summary = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_final_summary.csv",
        A_early_final_summary = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early_final_summary.csv",
        B_final_summary = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/B_final_summary.csv",
        C_final_summary = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/C_final_summary.csv",
        all_summary = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/all_summary.csv",

        A_predictions = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A-predictions.parquet",
        A_early_predictions = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early-predictions.parquet",
        B_predictions = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/B-predictions.parquet",
        C_predictions = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/C-predictions.parquet",

        A_trial_analysis_ensembled = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A-trial_analysis-ensembled.csv",
        A_early_trial_analysis_ensembled = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early-trial_analysis-ensembled.csv",
        B_trial_analysis_ensembled = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/B-trial_analysis-ensembled.csv",
        C_trial_analysis_ensembled = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/C-trial_analysis-ensembled.csv",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            electrodes_path=input.speech_reponsive,
                            result_path=input.result,
                            A_individual_results_path=input.individual_stimulus_decoder_results,
                            A_individual_decoder_path=input.individual_stimulus_decoders,
                            A_results_path=input.unified_stimulus_decoder_results,
                            A_stimulus_decoders_path=input.unified_stimulus_decoders,
                            B_results_path=input.annotated_B_results,
                            C_results_path=input.annotated_C_results,
                            trf_results_path=input.trf_results,
                            acoustic_decoding_scores_path=input.acoustic_decoding_scores,
                            outdir=str(outdir)),
        )

# acoustic decoding analysis on those electrodes that show behavioral response
# ACTUALLY this has been expanded to be a general searchlight; TODO rename
rule behavior_decoding_single_electrode_acoustic:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        speech_reponsive = "outputs/causal4/find_speech_responsive/{subject}_results.csv",

        notebook = "notebooks/causal4/behavior_decoding_single_electrode_acoustic.ipynb"

    output:
        notebook = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/behavior_decoding_single_electrode_acoustic.ipynb",

        outcomes = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/outcomes.parquet",
        all_outcomes = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/all_outcomes.parquet",
        train_scores = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/train_scores.parquet",
        test_scores = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/test_scores.parquet",
        avg_scores = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/avg_test_scores.csv",
        models = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/decoding_models.joblib",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            input.notebook,
            output.notebook,
            dict(epochs_path=input.epochs,
                 electrodes_path=input.speech_reponsive,
                 outdir=str(outdir)),
        )

rule behavior_decoding_single_electrode_super:
    input:
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        all_speech_responsive = expand("outputs/causal4/find_speech_responsive/{subject}_results.csv", subject=config["data"]["subjects"]),
        
        all_A_results = expand("outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_results.csv", subject=config["data"]["subjects"]),
        all_A_early_results = expand("outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early_results.csv", subject=config["data"]["subjects"]),

        notebook = "notebooks/causal4/behavior_decoding_single_electrode_super.ipynb"

    output:
        notebook = "outputs/causal4/behavior_decoding_single_electrode_super/behavior_decoding_single_electrode_super.ipynb",
        results = "outputs/causal4/behavior_decoding_single_electrode_super/results.pt",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_paths=input.all_epochs,
                            electrodes_paths=input.all_speech_responsive,
                            result_paths={"A": input.all_A_results,
                                          "A_early": input.all_A_early_results},
                            outdir=str(outdir)),
        )

# Attempt transferring behavior decoders estimated on different temporal windows to one another.
rule behavior_decoding_single_electrode_transfer:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        result = "outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt",

        individual_stimulus_decoder_results = "outputs/causal4/find_As/{subject}_results.csv",

        A_early_final_summary = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early_final_summary.csv",
        A_final_summary = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_final_summary.csv",
        
        trf_results = "/userdata/jgauthier/projects/big-trf/outputs/encoder_summary/timit-no_repeats/vanilla_aud.csv",
        
        notebook = "notebooks/causal4/behavior_decoding_single_electrode_transfer.ipynb"

    output:
        notebook = "outputs/causal4/behavior_decoding_single_electrode_transfer/{subject}/behavior_decoding_single_electrode_transfer.ipynb",
        transfer_results = "outputs/causal4/behavior_decoding_single_electrode_transfer/{subject}/transfer_results.csv",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            result_path=input.result,
                            A_individual_results_path=input.individual_stimulus_decoder_results,
                            trf_results_path=input.trf_results,
                            A_early_final_summary_path=input.A_early_final_summary,
                            A_final_summary_path=input.A_final_summary,
                            outdir=str(outdir)),
        )


rule behavior_decoding_super:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        electrodes = "outputs/causal4/find_speech_responsive/{subject}_results.csv",
        annotated_B_results = "outputs/causal4/annotated_B_results.csv",
        annotated_C_results = "outputs/causal4/annotated_C_results.csv",
        unified_A_results = "outputs/causal4/unify_As/results.csv",
        unified_A_decoders = "outputs/causal4/unify_As/unified_decoders.pt",
        all_A_results = "outputs/causal4/find_As/{subject}_results.csv",
        all_A_decoders = "outputs/causal4/find_As/{subject}_decoders.pt",
        notebook = "notebooks/causal4/behavior_decoding_super.ipynb"

    output:
        notebook = "outputs/causal4/behavior_decoding_super/{subject}/behavior_decoding_super.ipynb",
        results = "outputs/causal4/behavior_decoding_super/{subject}/results.pt",

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            electrodes_paths=input.electrodes,
                            B_annotated_path=input.annotated_B_results,
                            C_annotated_path=input.annotated_C_results,
                            A_result_path=input.unified_A_results,
                            A_decoders_path=input.unified_A_decoders,
                            all_A_result_path=input.all_A_results,
                            all_A_decoders_path=input.all_A_decoders,
                            outdir=str(outdir)),
        )


rule behavior_decoding_all:
    input:
        expand("outputs/causal4/behavior_decoding/{subject}/behavior_decoding.ipynb",
               subject=config["data"]["subjects"])


rule behavior_decoding_single_electrode_all:
    input:
        expand("outputs/causal4/behavior_decoding_single_electrode/{subject}/behavior_decoding.ipynb",
               subject=config["data"]["subjects"]),
        expand("outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/behavior_decoding_single_electrode_summarize.ipynb",
               subject=config["data"]["subjects"]),
        expand("outputs/causal4/behavior_decoding_single_electrode_transfer/{subject}/behavior_decoding_single_electrode_transfer.ipynb",
               subject=config["data"]["subjects"]),
        expand("outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/behavior_decoding_single_electrode_acoustic.ipynb",
               subject=config["data"]["subjects"])


rule behavior_decoding_super_all:
    input:
        expand("outputs/causal4/behavior_decoding_super/{subject}/behavior_decoding_super.ipynb",
               subject=config["data"]["subjects"])


# Permutation-based null distribution for behavior_decoding_single_electrode.
# Re-runs decoding K times per subject with shuffled labels, using hyperparameters
# fixed from the true model fit to avoid re-running the inner grid search.
rule behavior_decoding_single_electrode_permutation:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        true_results = "outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt",
        behav_peaks = "outputs/causal4/prepare_neurometrics/behav_peaks_df.parquet",
        notebook = "notebooks/causal4/behavior_decoding_single_electrode_permutation.py"

    output:
        notebook = "outputs/causal4/behavior_decoding_single_electrode_permutation/{subject}/notebook.ipynb",
        permutation_results = "outputs/causal4/behavior_decoding_single_electrode_permutation/{subject}/permutation_results.parquet"

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            true_results_path=input.true_results,
                            behav_peaks_path=input.behav_peaks,
                            n_permutations=1000,
                            outdir=str(outdir)),
        )


rule behavior_decoding_single_electrode_permutation_all:
    input:
        expand("outputs/causal4/behavior_decoding_single_electrode_permutation/{subject}/permutation_results.parquet",
               subject=config["data"]["subjects"])


# NHST across all subjects: compare true vs permuted Δ ROC-AUC per decoder,
# apply Benjamini-Hochberg FDR correction.
rule behavior_decoding_single_electrode_permutation_test:
    input:
        all_true_results = expand("outputs/causal4/behavior_decoding_single_electrode/{subject}/results.pt",
                                  subject=config["data"]["subjects"]),
        all_permutation_results = expand("outputs/causal4/behavior_decoding_single_electrode_permutation/{subject}/permutation_results.parquet",
                                         subject=config["data"]["subjects"]),
        notebook = "notebooks/causal4/behavior_decoding_single_electrode_permutation_test.py"

    output:
        notebook = "outputs/causal4/behavior_decoding_single_electrode_permutation_test/notebook.ipynb",
        results = "outputs/causal4/behavior_decoding_single_electrode_permutation_test/results.csv"

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(all_true_results=list(input.all_true_results),
                            all_permutation_results=list(input.all_permutation_results),
                            fdr_alpha=config["analysis"]["fdr_alpha"],
                            outdir=str(outdir)),
        )


# Compute A predictions on both phonetic and behavior targets
rule A_predictions:
    input:
        notebook = "notebooks/causal4/A_predictions.ipynb",
        all_results = expand("outputs/causal4/find_As/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_decoders = expand("outputs/causal4/find_As/{subject}_decoders.pt", subject=config["data"]["subjects"]),

        behav_acoustic_paths = expand("outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/all_outcomes.parquet",
                                      subject=config["data"]["subjects"]),

    output:
        notebook = "outputs/causal4/A_predictions/A_predictions.ipynb",
        phonetic_decoding = "outputs/causal4/A_predictions/phonetic_decoding.parquet",
        phonetic_summary = "outputs/causal4/A_predictions/phonetic_summary.parquet",
        behavior_to_phonetic_decoding = "outputs/causal4/A_predictions/behavior_to_phonetic_decoding.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            input.notebook,
            output.notebook,
            dict(all_results=input.all_results,
                 all_decoders=input.all_decoders,
                 behav_p_searchlight_paths=input.behav_acoustic_paths,
                 outdir=str(outdir)),
        )


rule prepare_neurometrics:
    """
    Pre-compute PaperData for A_neurometrics visualizations.
    The slow step (extract_hga_windows_df) and polarity derivations are done here
    so that A_neurometrics.py can load precomputed parquets instead of rerunning them.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"]
        ),
        A_behav_predictions = expand(
            "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A-predictions.parquet",
            subject=config["data"]["subjects"]
        ),
        A_early_behav_predictions = expand(
            "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early-predictions.parquet",
            subject=config["data"]["subjects"]
        ),
        phon_predictions = "outputs/causal4/A_predictions/behavior_to_phonetic_decoding.parquet",
        electrode_paths = expand(
            "outputs/causal4/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"]
        ),
        notebook = "notebooks/causal4/prepare_neurometrics.py",

    output:
        notebook = "outputs/causal4/prepare_neurometrics/prepare_neurometrics.ipynb",
        electrode_df = "outputs/causal4/prepare_neurometrics/electrode_df.parquet",
        plot_phon_phon_df = "outputs/causal4/prepare_neurometrics/plot_phon_phon_df.parquet",
        plot_behav_phon_df = "outputs/causal4/prepare_neurometrics/plot_behav_phon_df.parquet",
        plot_behav_behav_df = "outputs/causal4/prepare_neurometrics/plot_behav_behav_df.parquet",
        plot_phon_behav_df = "outputs/causal4/prepare_neurometrics/plot_phon_behav_df.parquet",
        behav_roc_auc_searchlight_df = "outputs/causal4/prepare_neurometrics/behav_roc_auc_searchlight_df.parquet",
        phon_roc_auc_searchlight_df = "outputs/causal4/prepare_neurometrics/phon_roc_auc_searchlight_df.parquet",
        all_md = "outputs/causal4/prepare_neurometrics/all_md.parquet",
        word_end_df = "outputs/causal4/prepare_neurometrics/word_end_df.parquet",
        phon_peaks_df = "outputs/causal4/prepare_neurometrics/phon_peaks_df.parquet",
        behav_peaks_df = "outputs/causal4/prepare_neurometrics/behav_peaks_df.parquet",
        behav_peaks_df_unfiltered = "outputs/causal4/prepare_neurometrics/behav_peaks_df_unfiltered.parquet",
        behav_baseline_df = "outputs/causal4/prepare_neurometrics/behav_baseline_df.parquet",
        zoomin_keys = "outputs/causal4/prepare_neurometrics/zoomin_keys.parquet",
        early_polarity = "outputs/causal4/prepare_neurometrics/early_polarity.parquet",
        late_polarity = "outputs/causal4/prepare_neurometrics/late_polarity.parquet",
        hga_df = "outputs/causal4/prepare_neurometrics/hga_df.parquet",
        reg_df = "outputs/causal4/prepare_neurometrics/reg_df.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                A_behav_predictions=list(input.A_behav_predictions),
                A_early_behav_predictions=list(input.A_early_behav_predictions),
                phon_predictions_path=str(input.phon_predictions),
                electrode_paths=list(input.electrode_paths),

                phon_response_tmin_min=0.0,
                all_response_tmax_max=1.3,

                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                epoch_tmin=config["analysis"]["epoch_tmin"],

                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                behav_response_peak_threshold=config["analysis"]["behav_response_peak_threshold"],
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],

                outdir=str(outdir),
            ),
        )


rule prepare_neurometrics_sweep:
    """
    Parameterized variant of prepare_neurometrics.
    Runs for every combination in NEUROMETRICS_PARAMS_GRID, writing outputs to
    outputs/causal4/prepare_neurometrics/{params_id}/.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"]
        ),
        A_behav_predictions = expand(
            "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A-predictions.parquet",
            subject=config["data"]["subjects"]
        ),
        A_early_behav_predictions = expand(
            "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early-predictions.parquet",
            subject=config["data"]["subjects"]
        ),
        phon_predictions = "outputs/causal4/A_predictions/behavior_to_phonetic_decoding.parquet",
        electrode_paths = expand(
            "outputs/causal4/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"]
        ),
        notebook = "notebooks/causal4/prepare_neurometrics.py",

    output:
        notebook = "outputs/causal4/prepare_neurometrics/{params_id}/notebook.ipynb",
        electrode_df = "outputs/causal4/prepare_neurometrics/{params_id}/electrode_df.parquet",
        plot_phon_phon_df = "outputs/causal4/prepare_neurometrics/{params_id}/plot_phon_phon_df.parquet",
        plot_behav_phon_df = "outputs/causal4/prepare_neurometrics/{params_id}/plot_behav_phon_df.parquet",
        plot_behav_behav_df = "outputs/causal4/prepare_neurometrics/{params_id}/plot_behav_behav_df.parquet",
        plot_phon_behav_df = "outputs/causal4/prepare_neurometrics/{params_id}/plot_phon_behav_df.parquet",
        behav_roc_auc_searchlight_df = "outputs/causal4/prepare_neurometrics/{params_id}/behav_roc_auc_searchlight_df.parquet",
        phon_roc_auc_searchlight_df = "outputs/causal4/prepare_neurometrics/{params_id}/phon_roc_auc_searchlight_df.parquet",
        all_md = "outputs/causal4/prepare_neurometrics/{params_id}/all_md.parquet",
        word_end_df = "outputs/causal4/prepare_neurometrics/{params_id}/word_end_df.parquet",
        phon_peaks_df = "outputs/causal4/prepare_neurometrics/{params_id}/phon_peaks_df.parquet",
        behav_peaks_df = "outputs/causal4/prepare_neurometrics/{params_id}/behav_peaks_df.parquet",
        behav_peaks_df_unfiltered = "outputs/causal4/prepare_neurometrics/{params_id}/behav_peaks_df_unfiltered.parquet",
        behav_baseline_df = "outputs/causal4/prepare_neurometrics/{params_id}/behav_baseline_df.parquet",
        zoomin_keys = "outputs/causal4/prepare_neurometrics/{params_id}/zoomin_keys.parquet",
        early_polarity = "outputs/causal4/prepare_neurometrics/{params_id}/early_polarity.parquet",
        late_polarity = "outputs/causal4/prepare_neurometrics/{params_id}/late_polarity.parquet",
        hga_df = "outputs/causal4/prepare_neurometrics/{params_id}/hga_df.parquet",
        reg_df = "outputs/causal4/prepare_neurometrics/{params_id}/reg_df.parquet",

    wildcard_constraints:
        params_id = "|".join(ALL_NEUROMETRICS_PARAMS)

    run:
        params = NEUROMETRICS_PARAMS_GRID[wildcards.params_id]
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                A_behav_predictions=list(input.A_behav_predictions),
                A_early_behav_predictions=list(input.A_early_behav_predictions),
                phon_predictions_path=str(input.phon_predictions),
                electrode_paths=list(input.electrode_paths),

                phon_response_tmin_min=0.0,
                all_response_tmax_max=1.3,

                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                epoch_tmin=config["analysis"]["epoch_tmin"],

                phon_response_peak_threshold=params["phon_response_peak_threshold"],
                behav_response_peak_threshold=params["behav_response_peak_threshold"],
                ambiguous_response_threshold=params["ambiguous_response_threshold"],

                outdir=str(outdir),
            ),
        )


rule A_neurometrics:
    """
    Run A_neurometrics visualizations for a single params_id.
    Depends on the corresponding prepare_neurometrics_sweep outputs.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"]
        ),
        phonetic_searchlight_paths = expand(
            "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/results.pt",
            subject=config["data"]["subjects"]
        ),
        # Use zoomin_keys as the sentinel that prepare_neurometrics_sweep is done
        neurometrics_sentinel = "outputs/causal4/prepare_neurometrics/{params_id}/zoomin_keys.parquet",
        notebook = "notebooks/causal4/A_neurometrics.py",

    output:
        notebook = "outputs/causal4/A_neurometrics/{params_id}/notebook.ipynb",
        hga_zoomin_search_keys = "outputs/causal4/A_neurometrics/{params_id}/hga_zoomin_search_keys.csv",
        phonetic_transfer_results = "outputs/causal4/A_neurometrics/{params_id}/phonetic_transfer_extreme_results_mean.csv",

    wildcard_constraints:
        params_id = "|".join(ALL_NEUROMETRICS_PARAMS)

    run:
        params = NEUROMETRICS_PARAMS_GRID[wildcards.params_id]
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                phonetic_searchlight_paths=list(input.phonetic_searchlight_paths),
                neurometrics_dir=str(Path(input.neurometrics_sentinel).parent),
                ambiguous_response_threshold=params["ambiguous_response_threshold"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                textgrid_dir="textgrids",
                outdir=str(outdir),
            ),
        )


rule neurometrics_sweep_all:
    """Run the full prepare_neurometrics + A_neurometrics sweep over all param combinations."""
    input:
        expand(
            "outputs/causal4/A_neurometrics/{params_id}/hga_zoomin_search_keys.csv",
            params_id=ALL_NEUROMETRICS_PARAMS,
        )


rule prepare_neurometrics_ttest:
    """
    T-test-based companion to prepare_neurometrics.
    Behavioral site selection uses a sliding Welch's t-test (find_site_windows)
    rather than behavioral decoder ROC-AUC improvement.
    The slow window search is run here; results are saved pre-threshold so that
    the p-value cut-off can be adjusted in a notebook without re-running.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"]
        ),
        A_behav_predictions = expand(
            "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A-predictions.parquet",
            subject=config["data"]["subjects"]
        ),
        A_early_behav_predictions = expand(
            "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early-predictions.parquet",
            subject=config["data"]["subjects"]
        ),
        phon_predictions = "outputs/causal4/A_predictions/behavior_to_phonetic_decoding.parquet",
        electrode_paths = expand(
            "outputs/causal4/find_speech_responsive/{subject}_results.csv",
            subject=config["data"]["subjects"]
        ),
        notebook = "notebooks/causal4/prepare_neurometrics_ttest.py",

    output:
        notebook = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/notebook.ipynb",
        electrode_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/electrode_df.parquet",
        plot_phon_phon_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/plot_phon_phon_df.parquet",
        plot_behav_phon_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/plot_behav_phon_df.parquet",
        plot_behav_behav_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/plot_behav_behav_df.parquet",
        plot_phon_behav_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/plot_phon_behav_df.parquet",
        behav_roc_auc_searchlight_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/behav_roc_auc_searchlight_df.parquet",
        phon_roc_auc_searchlight_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/phon_roc_auc_searchlight_df.parquet",
        all_md = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/all_md.parquet",
        word_end_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/word_end_df.parquet",
        phon_peaks_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/phon_peaks_df.parquet",
        # ttest-specific outputs
        ttest_behav_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/ttest_behav_df.parquet",
        behav_peaks_ttest_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/behav_peaks_ttest_df.parquet",
        ttest_snap_report = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/ttest_snap_report.parquet",
        ttest_behav_df_full = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/ttest_behav_df_full.parquet",
        behav_baseline_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/behav_baseline_df.parquet",
        zoomin_keys = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/zoomin_keys.parquet",
        early_polarity = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/early_polarity.parquet",
        late_polarity = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/late_polarity.parquet",
        hga_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/hga_df.parquet",
        reg_df = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/reg_df.parquet",

    wildcard_constraints:
        params_id = NEUROMETRICS_TTEST_PARAMS_TAG

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                A_behav_predictions=list(input.A_behav_predictions),
                A_early_behav_predictions=list(input.A_early_behav_predictions),
                phon_predictions_path=str(input.phon_predictions),
                electrode_paths=list(input.electrode_paths),

                phon_response_tmin_min=0.0,
                all_response_tmax_max=1.3,

                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                epoch_tmin=config["analysis"]["epoch_tmin"],

                phon_response_peak_threshold=config["analysis"]["phon_response_peak_threshold"],
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
                behav_ttest_pvalue_threshold=config["analysis"]["behav_ttest_pvalue_threshold"],

                outdir=str(outdir),
            ),
        )


rule A_neurometrics_ttest:
    """
    T-test-based companion to A_neurometrics.
    Loads PaperData from prepare_neurometrics_ttest outputs and runs the same
    suite of analyses with t-test-selected behavioral sites.
    """
    input:
        all_epochs = expand(
            "outputs/epochs_preprocessed/{subject}_epo.fif",
            subject=config["data"]["subjects"]
        ),
        phonetic_searchlight_paths = expand(
            "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/results.pt",
            subject=config["data"]["subjects"]
        ),
        # Use zoomin_keys as the sentinel that prepare_neurometrics_ttest is done
        neurometrics_sentinel = "outputs/causal4/prepare_neurometrics_ttest/{params_id}/zoomin_keys.parquet",
        notebook = "notebooks/causal4/A_neurometrics_ttest.py",

    output:
        notebook = "outputs/causal4/A_neurometrics_ttest/{params_id}/notebook.ipynb",
        hga_zoomin_search_keys = "outputs/causal4/A_neurometrics_ttest/{params_id}/hga_zoomin_search_keys.csv",
        phonetic_transfer_results = "outputs/causal4/A_neurometrics_ttest/{params_id}/phonetic_transfer_extreme_results_mean.csv",

    wildcard_constraints:
        params_id = NEUROMETRICS_TTEST_PARAMS_TAG

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                all_epochs=list(input.all_epochs),
                phonetic_searchlight_paths=list(input.phonetic_searchlight_paths),
                neurometrics_dir=str(Path(input.neurometrics_sentinel).parent),
                ambiguous_response_threshold=config["analysis"]["ambiguous_response_threshold"],
                epoch_tmin=config["analysis"]["epoch_tmin"],
                epoch_sfreq=config["analysis"]["epoch_sfreq"],
                textgrid_dir="textgrids",
                outdir=str(outdir),
            ),
        )