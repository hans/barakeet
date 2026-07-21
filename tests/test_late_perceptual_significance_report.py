"""End-to-end integration test for the late_perceptual_significance_report
notebook (plan Step 6: docs/superpowers/plans/2026-07-20-causal46-late-perceptual-significance.md).

Chains the same synthetic fixtures used by
test_late_perceptual_significance.py through both notebooks — #10's gate
notebook produces site_results.parquet, then the report notebook consumes it
plus the raw bootstrap curves for the sensitivity panel — mirroring how the
two rules will chain in the Snakefile.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parent.parent
REPORT_NOTEBOOK = REPO / "notebooks" / "causal46_joined" / "late_perceptual_significance_report.py"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_late_perceptual_significance import _build_fixtures, _run_notebook  # noqa: E402


def _run_report_notebook(fixtures: Path, site_results_path: Path, outdir: Path, work: Path) -> None:
    jupytext = pytest.importorskip("jupytext")
    from ploomber_engine import execute_notebook

    nb = jupytext.read(REPORT_NOTEBOOK)
    input_ipynb = work / "report_input.ipynb"
    jupytext.write(nb, input_ipynb)

    params = dict(
        site_results_path=str(site_results_path),
        b4_bootstrap_path=str(fixtures / "b4_bootstrap.parquet"),
        b4_per_cell_path=str(fixtures / "b4_per_cell.parquet"),
        outdir=str(outdir),
    )
    output_ipynb = work / "report_output.ipynb"
    execute_notebook(
        str(input_ipynb), str(output_ipynb),
        parameters=params, cwd=str(REPO), progress_bar=False,
    )


@pytest.fixture(autouse=True)
def _mpl_cache(tmp_path, monkeypatch):
    cache = tmp_path / "mplcache"
    cache.mkdir(exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(cache))


def test_late_perceptual_significance_report_end_to_end(tmp_path):
    fixtures = _build_fixtures(tmp_path)
    gate_outdir = tmp_path / "gate_results"
    gate_work = tmp_path / "gate_work"
    gate_work.mkdir()
    _run_notebook(fixtures, gate_outdir, gate_work)  # writes site_results.parquet

    report_outdir = tmp_path / "report_results"
    report_work = tmp_path / "report_work"
    report_work.mkdir()
    _run_report_notebook(
        fixtures, gate_outdir / "site_results.parquet", report_outdir, report_work
    )

    assert (report_outdir / "sensitivity_grid.csv").exists()
    assert (report_outdir / "calibration_disagreements.csv").exists()
    assert (report_outdir / "report_summary.pdf").exists()

    sensitivity = pl.read_csv(report_outdir / "sensitivity_grid.csv")
    assert sensitivity.height == 4  # E in {0.5, 1.0} x H in {1.0, 2.0}
    assert set(sensitivity.columns) >= {
        "E", "H", "n_family", "n_gate_pass", "binom_p", "n_fdr_pass", "is_preregistered"
    }
    assert int(sensitivity["is_preregistered"].sum()) == 1
    prereg = sensitivity.filter(pl.col("is_preregistered")).row(0, named=True)
    assert prereg["E"] == 0.5 and prereg["H"] == 2.0
    # Only SYNTH1/1 and SYNTH1/2 have usable curves (SYNTH2/1 tied, SYNTH2/2
    # no-candidates, SYNTH3/1 missing — see test_late_perceptual_significance).
    assert (sensitivity["n_family"] == 2).all()

    disagreements = pl.read_csv(report_outdir / "calibration_disagreements.csv")
    assert set(disagreements.columns) >= {
        "subject", "electrode_idx", "phoneme_pair", "word_end",
        "tfce_emp_p", "disagreement_kind",
    }
    assert set(disagreements["disagreement_kind"].unique().to_list()) <= {
        "manual_only", "tfce_only"
    }
