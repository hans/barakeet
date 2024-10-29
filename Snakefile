from pathlib import Path
import yaml

from papermill import execute_notebook

configfile: "config.yaml"

DEFAULT_NOTEBOOKS = {run_name: run_dict if run_dict else {"notebook": run_name}
                     for run_name, run_dict in config["run_notebooks"].items()}


rule preprocess_epochs:
    input:
        epochs = "epochs/{subject}_epochs.fif"

    output:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif"

    script: "scripts/preprocess_epochs.py"


rule preprocess_all_epochs:
    input:
        expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"])


rule run_notebook:
    input:
        notebook = "notebooks/{notebook}.ipynb",

    output:
        out_notebook = "outputs/{run_name}/{notebook}.ipynb"

    run:
        params = DEFAULT_NOTEBOOKS[wildcards.run_name].get("params", {})
        params["outdir"] = str(Path(output.out_notebook).parent)
        params_str = yaml.dump(params, default_flow_style=True)
        shell("""
        papermill --log-output {input.notebook} {output.out_notebook} \
            -y "{params_str}"
        """)


rule run_all_notebooks:
    input:
        [f"outputs/{run_name}/{run_dict['notebook']}.ipynb" for run_name, run_dict in DEFAULT_NOTEBOOKS.items()]


rule fit_trf:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        notebook = "notebooks/electrodes_of_interest.ipynb"

    output:
        trace = directory("outputs/trfs/{subject}"),
        notebook = "outputs/trfs/{subject}/electrodes_of_interest.ipynb",
        result = "outputs/trfs/{subject}/results.pkl"

    shell:
        """
        papermill --log-output {input.notebook} \
            {output.notebook} \
            -p epochs_path {input.epochs} \
            -p out_path {output.result}
        """


rule fit_windowed:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        notebook = "notebooks/electrodes_of_interest-windowed.ipynb"

    output:
        trace = directory("outputs/windowed_regression/{subject}"),
        notebook = "outputs/windowed_regression/{subject}/electrodes_of_interest-windowed.ipynb",
        result = "outputs/windowed_regression/{subject}/results.pkl"

    shell:
        """
        papermill --log-output {input.notebook} \
            {output.notebook} \
            -p epochs_path {input.epochs} \
            -p out_path {output.result}
        """


rule fit_all_trfs:
    input:
        expand("outputs/trfs/{subject}/results.pkl", subject=config["data"]["subjects"])

rule fit_all_windowed:
    input:
        expand("outputs/windowed_regression/{subject}/results.pkl", subject=config["data"]["subjects"])


rule trf_eoi:
    input:
        trf_paths = expand("outputs/trfs/{subject}/results.pkl", subject=config["data"]["subjects"]),
        notebook = "notebooks/trf_eois.ipynb",

    output:
        notebook = "outputs/trf_eois/notebook.ipynb",
        outdir = directory("outputs/trf_eois"),
        eois = "outputs/trf_eois/eois.csv",
        coefs = "outputs/trf_eois/coefs.csv",

    run:
        execute_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(trf_paths=input.trf_paths,
                            outdir=str(output.outdir)),
        )