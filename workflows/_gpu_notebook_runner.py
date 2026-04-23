#!/usr/bin/env python
"""Run a jupytext `.py` / `.ipynb` notebook in a subprocess.

Invoked by GPU-heavy Snakemake rules so `CUDA_VISIBLE_DEVICES` can be
scoped per-process. Torch reads that env var at CUDA-init time and
caches it; mutating `os.environ` *after* torch is imported (e.g. from
inside a Snakemake `run:` block) has no effect. Spawning a fresh Python
process per rule is the only way to route different rules to different
GPUs.

Mirrors `run_notebook()` in `workflows/causal6.Snakefile` — same
jupytext conversion + ploomber strict static analysis + executor.

CLI: `python _gpu_notebook_runner.py INPUT OUTPUT PARAMS_JSON`
where `PARAMS_JSON` is a path to a JSON file containing the notebook
parameters dict.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} INPUT OUTPUT PARAMS_JSON", file=sys.stderr)
        sys.exit(2)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    params_path = Path(sys.argv[3])

    with open(params_path) as f:
        parameters = json.load(f)

    import jupytext
    from ploomber import DAG
    from ploomber.products import File
    from ploomber.tasks import NotebookRunner
    from ploomber_engine import execute_notebook

    if input_path.suffix == ".py":
        nb = jupytext.read(input_path)
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            tmp_path = Path(f.name)
        jupytext.write(nb, tmp_path)
    else:
        tmp_path = None

    try:
        actual_input = tmp_path if tmp_path is not None else input_path
        dag = DAG(name="temp_dag")
        NotebookRunner(
            actual_input,
            File(str(output_path)),
            dag=dag,
            params=parameters,
            static_analysis="strict",
        )
        dag.render(force=True)
        execute_notebook(actual_input, str(output_path), parameters=parameters)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
