from pathlib import Path
import yaml

configfile: "config.yaml"

DEFAULT_NOTEBOOKS = {run_name: run_dict if run_dict else {"notebook": run_name}
                     for run_name, run_dict in config["run_notebooks"].items()}


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