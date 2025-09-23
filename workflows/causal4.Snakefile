from pathlib import Path

from ploomber_engine import execute_notebook


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


# TODO unified A intrinsics
# - neurometric response functions


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


rule behavior_decoding_all:
    input:
        expand("outputs/causal4/behavior_decoding/{subject}/behavior_decoding.ipynb",
               subject=config["data"]["subjects"])