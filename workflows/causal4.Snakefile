from pathlib import Path

from ploomber_engine import execute_notebook


def run_notebook(input_path: str, output_path: str, parameters, **kwargs):
    # First hack into the Ploomber API. Build a fake DAG so that we can validate parameters.
    from ploomber import DAG
    from ploomber.products import File
    from ploomber.tasks import NotebookRunner

    dag = DAG(name="temp_dag")
    runner = NotebookRunner(
        Path(input_path),
        File(output_path),
        dag=dag,
        params=parameters,
        static_analysis="strict",
    )
    # This will throw an exception if there are parameter issues (e.g. missing parameters)
    dag.render(force=True)

    # Now ditch that and run directly with `ploomber_engine`
    return execute_notebook(
        Path(input_path),
        Path(output_path),
        parameters=parameters,
    )


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
rule behavior_decoding_single_electrode_acoustic:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        speech_reponsive = "outputs/causal4/find_speech_responsive/{subject}_results.csv",
        behavior_A_early = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_early_final_summary.csv",
        behavior_A = "outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/A_final_summary.csv",

        notebook = "notebooks/causal4/behavior_decoding_single_electrode_acoustic.ipynb"

    output:
        notebook = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/behavior_decoding_single_electrode_acoustic.ipynb",
        results = "outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/results.pt",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            input.notebook,
            output.notebook,
            dict(epochs_path=input.epochs,
                 electrodes_path=input.speech_reponsive,
                 summary_path={"A_early": input.behavior_A_early,
                               "A": input.behavior_A},
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


# Compute A predictions on both phonetic and behavior targets
rule A_predictions:
    input:
        notebook = "notebooks/causal4/A_predictions.ipynb",
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        all_speech_responsive = expand("outputs/causal4/find_speech_responsive/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_results = expand("outputs/causal4/find_As/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_decoders = expand("outputs/causal4/find_As/{subject}_decoders.pt", subject=config["data"]["subjects"]),

        behav_summary_paths = expand("outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/{kind}_final_summary.csv",
                                    subject=config["data"]["subjects"], kind=["A", "A_early"]),
        behav_trial_paths = expand("outputs/causal4/behavior_decoding_single_electrode_summarize/{subject}/{kind}_trial_analysis.csv",
                                subject=config["data"]["subjects"], kind=["A", "A_early"]),

        behav_acoustic_paths = expand("outputs/causal4/behavior_decoding_single_electrode_acoustic/{subject}/results.pt",
                                      subject=config["data"]["subjects"]),

    output:
        notebook = "outputs/causal4/A_predictions/A_predictions.ipynb",
        phonetic_decoding = "outputs/causal4/A_predictions/phonetic_decoding.parquet",
        behavior_decoding = "outputs/causal4/A_predictions/behavior_decoding.parquet",
        phonetic_summary = "outputs/causal4/A_predictions/phonetic_summary.parquet",
        behavior_summary = "outputs/causal4/A_predictions/behavior_summary.parquet",
        behavior_to_phonetic_decoding = "outputs/causal4/A_predictions/behavior_to_phonetic_decoding.parquet",

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            input.notebook,
            output.notebook,
            dict(all_epochs=input.all_epochs,
                 all_electrode_dfs=input.all_speech_responsive,
                 all_results=input.all_results,
                 all_decoders=input.all_decoders,
                 behav_summary_paths=input.behav_summary_paths,
                 behav_decoder_trial_paths=input.behav_trial_paths,
                 behav_p_searchlight_paths=input.behav_acoustic_paths,
                 outdir=str(outdir)),
        )