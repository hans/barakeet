from pathlib import Path
import yaml

from papermill import execute_notebook as execute_notebook_papermill
from ploomber_engine import execute_notebook

configfile: "config.yaml"

# include: "workflows/causal4.Snakefile"
# include: "workflows/causal5.Snakefile"
include: "workflows/causal6.Snakefile"

# DEFAULT_NOTEBOOKS = {run_name: run_dict if run_dict else {"notebook": run_name}
#                      for run_name, run_dict in config["run_notebooks"].items()}
DEFAULT_NOTEBOOKS = {}


wildcard_constraints:
    window = r"\d+",
    target = r"[\w_]+",


rule preprocess_epochs:
    input:
        epochs = "epochs/{subject}_epochs.fif"

    output:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif"

    script: "scripts/preprocess_epochs.py"


rule preprocess_all_epochs:
    input:
        expand("outputs/epochs_preprocessed/{subject}_epo.fif", subject=config["data"]["subjects"])


# rule run_notebook:
#     input:
#         notebook = "notebooks/{notebook}.ipynb",

#     output:
#         out_notebook = "outputs/{run_name}/{notebook}.ipynb"

#     run:
#         params = DEFAULT_NOTEBOOKS[wildcards.run_name].get("params", {})
#         params["outdir"] = str(Path(output.out_notebook).parent)
#         params_str = yaml.dump(params, default_flow_style=True)
#         shell("""
#         papermill --log-output {input.notebook} {output.out_notebook} \
#             -y "{params_str}"
#         """)


# rule run_all_notebooks:
#     input:
#         [f"outputs/{run_name}/{run_dict['notebook']}.ipynb" for run_name, run_dict in DEFAULT_NOTEBOOKS.items()]


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


rule fit_trf_with_behavior:
    input:
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        notebook = "notebooks/electrodes_of_interest-behavior.ipynb"

    output:
        trace = directory("outputs/trfs_behavior/{subject}"),
        notebook = "outputs/trfs_behavior/{subject}/electrodes_of_interest-behavior.ipynb",
        result = "outputs/trfs_behavior/{subject}/results.pkl"

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

rule fit_all_trfs_behavior:
    input:
        expand("outputs/trfs_behavior/{subject}/results.pkl", subject=config["data"]["subjects"])

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
        execute_notebook_papermill(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(trf_paths=input.trf_paths,
                            outdir=str(output.outdir)),
        )

rule trf_behavior_eoi:
    input:
        trf_paths = expand("outputs/trfs_behavior/{subject}/results.pkl", subject=config["data"]["subjects"]),
        notebook = "notebooks/trf_eois.ipynb",

    output:
        notebook = "outputs/trf_eois_behavior/notebook.ipynb",
        outdir = directory("outputs/trf_eois_behavior"),
        eois = "outputs/trf_eois_behavior/eois.csv",
        coefs = "outputs/trf_eois_behavior/coefs.csv",

    run:
        execute_notebook_papermill(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(trf_paths=input.trf_paths,
                            outdir=str(output.outdir)),
        )

rule trf_stepwise:
    input:
        trf_path = "outputs/trfs/{subject}/results.pkl",
        trf_eois = "outputs/trf_eois/eois.csv",
        epochs = "outputs/epochs_preprocessed/{subject}_epo.fif",
        notebook = "notebooks/electrodes_of_interest-stepwise.ipynb",

    output:
        outdir = directory("outputs/trf_stepwise/{subject}"),
        notebook = "outputs/trf_stepwise/{subject}/notebook.ipynb",
        results = "outputs/trf_stepwise/{subject}/results.pkl",
    
    run:
        execute_notebook_papermill(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(trf_path=input.trf_path,
                            eois_path=input.trf_eois,
                            epochs_path=input.epochs,
                            outdir=str(output.outdir)),
        )

all_stepwise_results = expand("outputs/trf_stepwise/{subject}/results.pkl", subject=config["data"]["subjects"])
rule all_trf_stepwise:
    input:
        all_stepwise_results

rule analyze_stepwise:
    input:
        stepwise_results = all_stepwise_results,
        notebook = "notebooks/analyze_stepwise.ipynb",

    output:
        outdir = directory("outputs/analyze_stepwise"),
        notebook = "outputs/analyze_stepwise/notebook.ipynb",
        eois = "outputs/analyze_stepwise/eois.csv",

    run:
        execute_notebook_papermill(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(stepwise_results=input.stepwise_results,
                            outdir=str(output.outdir)),
        )


rule trf_epoched_plots:
    input:
        trf_eois = "outputs/trf_eois/eois.csv",
        trf_behavior_eois = "outputs/trf_eois_behavior/eois.csv",
        trf_stepwise_eois = "outputs/analyze_stepwise/eois.csv",
        textgrids = "textgrids",
        notebook = "notebooks/epoched_plots.ipynb",
        epochs = lambda _: expand("outputs/epochs_preprocessed/{subject}_epo.fif",
                                  subject=glob_wildcards("outputs/epochs_preprocessed/{subject}_epo.fif")[0]),

    output:
        outdir = directory("outputs/epoch_plots"),
        notebook = "outputs/epoch_plots/epoched_plots.ipynb",

    run:
        execute_notebook_papermill(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(trf_eois=input.trf_eois,
                            trf_stepwise_eois=input.trf_stepwise_eois,
                            epoch_paths=input.epochs,
                            textgrids_path=input.textgrids,
                            outdir=str(output.outdir)),
        )


# maximum window size = 120 samples
max_decoding_window_size = 120
# minimum window size = 10 samples (100 ms)
min_decoding_window_size = 10
# decoding stride = 50 ms
decoding_stride = 5
rule single_electrode_decoding_full_window:
    input:
        epochs = "outputs/epochs_preprocessed",
        notebook = "notebooks/single_electrode_decoding.ipynb",
    
    output:
        outdir = directory("outputs/single_electrode_decoding/full_window/{target}"),
        notebook = "outputs/single_electrode_decoding/full_window/{target}/notebook.ipynb",
        scores = "outputs/single_electrode_decoding/full_window/{target}/scores.csv",
        outcomes = "outputs/single_electrode_decoding/full_window/{target}/outcomes.pt",

    run:
        execute_notebook(
            str(input.notebook),
            output_path=str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            outdir=str(output.outdir),
                            prediction_target=wildcards.target,
                            window_size=max_decoding_window_size,
                            stride=max_decoding_window_size,
                            save_outcomes=False),
            log_output=True,
            progress_bar=True,
        )

rule single_electrode_decoding_specific_window:
    input:
        epochs = "outputs/epochs_preprocessed",
        notebook = "notebooks/single_electrode_decoding.ipynb",
    
    output:
        outdir = directory("outputs/single_electrode_decoding/{window}/{target}"),
        notebook = "outputs/single_electrode_decoding/{window}/{target}/notebook.ipynb",
        scores = "outputs/single_electrode_decoding/{window}/{target}/scores.csv",
        outcomes = "outputs/single_electrode_decoding/{window}/{target}/outcomes.pt",

    run:
        window_size = int(wildcards.window)
        if window_size < min_decoding_window_size or window_size > max_decoding_window_size:
            raise ValueError(f"Window size {window_size} is not in the range [{min_decoding_window_size}, {max_decoding_window_size}]")

        execute_notebook(
            str(input.notebook),
            output_path=str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            outdir=str(output.outdir),
                            prediction_target=wildcards.target,
                            window_size=window_size,
                            stride=decoding_stride,
                            save_outcomes=False),
            log_output=True,
            progress_bar=True,
        )


rule single_electrode_decoding_full_window_random:
    input:
        epochs = "outputs/epochs_preprocessed",
        notebook = "notebooks/single_electrode_decoding.ipynb",
    
    output:
        outdir = directory("outputs/single_electrode_decoding-random_{run}/full_window/{target}"),
        notebook = "outputs/single_electrode_decoding-random_{run}/full_window/{target}/notebook.ipynb",
        scores = "outputs/single_electrode_decoding-random_{run}/full_window/{target}/scores.csv",
        outcomes = "outputs/single_electrode_decoding-random_{run}/full_window/{target}/outcomes.pt",

    run:
        execute_notebook(
            str(input.notebook),
            output_path=str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            outdir=str(output.outdir),
                            prediction_target=wildcards.target,
                            window_size=max_decoding_window_size,
                            stride=max_decoding_window_size,
                            randomize=True,
                            save_outcomes=False),
            log_output=True,
            progress_bar=True,
        )


rule single_electrode_decoding_specific_window_random:
    input:
        epochs = "outputs/epochs_preprocessed",
        notebook = "notebooks/single_electrode_decoding.ipynb",
    
    output:
        outdir = directory("outputs/single_electrode_decoding-random_{run}/{window}/{target}"),
        notebook = "outputs/single_electrode_decoding-random_{run}/{window}/{target}/notebook.ipynb",
        scores = "outputs/single_electrode_decoding-random_{run}/{window}/{target}/scores.csv",
        outcomes = "outputs/single_electrode_decoding-random_{run}/{window}/{target}/outcomes.pt",

    run:
        window_size = int(wildcards.window)
        if window_size < min_decoding_window_size or window_size > max_decoding_window_size:
            raise ValueError(f"Window size {window_size} is not in the range [{min_decoding_window_size}, {max_decoding_window_size}]")

        execute_notebook(
            str(input.notebook),
            output_path=str(output.notebook),
            parameters=dict(epochs_path=input.epochs,
                            outdir=str(output.outdir),
                            prediction_target=wildcards.target,
                            window_size=window_size,
                            stride=decoding_stride,
                            randomize=True,
                            save_outcomes=False),
            log_output=True,
            progress_bar=True,
        )


# rule single_electrode_decoding_all_results:
#     input:
#         expand("outputs/single_electrode_decoding/full_window/{target}/scores.csv",
#                target=config["decoding"]["targets"]),
#         expand("outputs/single_electrode_decoding/{window}/{target}/scores.csv",
#                window=config["decoding"]["window_sizes"],
#                target=config["decoding"]["targets"]),
#         # expand("outputs/single_electrode_decoding-random_{run}/{window}/{target}/scores.csv",
#         #         window=config["decoding"]["window_sizes"] + ["full_window"],
#         #         target=config["decoding"]["targets"],
#         #         run=list(range(config["decoding"]["num_random_runs"]))),
