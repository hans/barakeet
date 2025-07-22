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
                            outdir=str(outdir)),
        )


rule find_As_all:
    input:
        expand("outputs/causal4/find_As/{subject}.ipynb", subject=config["data"]["subjects"])


rule run_A_intrinsics:
    input:
        all_epochs = expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"]),
        all_results = expand("outputs/causal4/find_As/{subject}_results.csv", subject=config["data"]["subjects"]),
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
                            outdir=str(outdir)),
        )


# Form unions of As within-subject based on similarity in space
# and/or function.
rule unify_As:
    input:
        all_results = expand("outputs/causal4/find_As/{subject}_results.csv", subject=config["data"]["subjects"]),
        notebook = "notebooks/causal4/unify_As.ipynb"

    output:
        notebook = "outputs/causal4/unify_As/unify_As.ipynb",
        results = "outputs/causal4/unify_As/unified_results.csv"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(all_results=input.all_results,
                            outdir=str(outdir)),
        )


rule find_Bs:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        speech_responsive = "outputs/causal4/find_speech_responsive/{subject}_results.csv",
        unified_As = "outputs/causal4/unify_As/unified_results.csv",
        notebook = "notebooks/causal4/find_Bs.ipynb"

    output:
        notebook = "outputs/causal4/find_Bs/{subject}.ipynb",
        results = "outputs/causal4/find_Bs/{subject}_results.csv",
        decoders = "outputs/causal4/find_Bs/{subject}_decoders.pt"

    run:
        outdir = Path(output.notebook).parent
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            speech_responsive=input.speech_responsive,
                            unified_As=input.unified_As,
                            outdir=str(outdir)),
        )


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


rule analyze:
    input:
        all_A_results = expand("outputs/causal4/find_As/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_A_decoders = expand("outputs/causal4/find_As/{subject}_decoders.pt", subject=config["data"]["subjects"]),
        all_B_results = expand("outputs/causal4/find_Bs/{subject}_results.csv", subject=config["data"]["subjects"]),
        all_B_decoders = expand("outputs/causal4/find_Bs/{subject}_decoders.pt", subject=config["data"]["subjects"]),
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