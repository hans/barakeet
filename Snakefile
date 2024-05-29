
configfile: "config.yaml"

DEFAULT_NOTEBOOKS = list(config["notebooks"].keys())


rule run_notebook:
    input:
        notebook = "notebooks/{notebook}.ipynb",

    output:
        outdir = directory("outputs/{notebook}"),
        out_notebook = "outputs/{notebook}/{notebook}.ipynb"

    shell:
        """
        papermill {input.notebook} {output.out_notebook} \
            -p outdir {output.outdir}
        """


rule run_all_notebooks:
    input:
        expand("outputs/{notebook}/{notebook}.ipynb", notebook=DEFAULT_NOTEBOOKS)