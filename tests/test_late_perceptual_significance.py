"""End-to-end integration test for the late_perceptual_significance notebook
(plan Step 3: docs/superpowers/plans/2026-07-20-causal46-late-perceptual-significance.md).

Executes the actual jupytext notebook (via ploomber_engine, papermill-style
parameter injection) against synthetic b4_bootstrap / b4_per_cell / manual
manifest fixtures — the real fixtures live only on the prod mount
(`outputs_prod/causal46_joined/t_tests/`, and `b4_bootstrap.parquet` itself
is excluded from the prod sync entirely; see `sync.sh`), so this is the only
way to exercise the notebook's per-cell loop, BH-FDR, and count-vs-null
headline logic in this environment. `late_cell_significance`'s own numeric
correctness is covered separately in `tests/test_causal46_windows.py`; this
test focuses on the notebook's data plumbing (schema handling, per-cell
edge cases, joins, output writing).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "causal46_joined" / "late_perceptual_significance.py"

GRID = [(s, s + 10) for s in range(0, 130, 10)]  # (0,10) ... (120,130)
R = 100


def _add_cell(boot_rows, per_cell_rows, manifest_rows, rng, *,
              subject, eidx, pp, we, phon_smax, kind):
    per_cell_rows.append({
        "subject": subject, "electrode_idx": eidx, "phoneme_pair": pp, "word_end": we,
        "phon_smin": 0, "phon_smax": phon_smax,
        "n_per_class": 8, "acoustic_peak_auc": 0.7, "R_replicates": R,
    })
    manifest_rows.append({
        "subject": subject, "electrode_idx": eidx, "phoneme_pair": pp, "word_end": we,
        "behav @late": "n" if kind == "signal" else None,
    })
    if kind == "missing":
        return
    for r in range(R):
        for smin, smax in GRID:
            if kind == "signal":
                base = 2.5 if 40 <= smin < 70 else 0.0
                mean_diff_raw = base + rng.normal(0, 0.3)
                mean_diff_aligned_null = rng.normal(0, 0.3)
            elif kind == "noise":
                mean_diff_raw = rng.normal(0, 0.3)
                mean_diff_aligned_null = rng.normal(0, 0.3)
            elif kind == "tied":
                mean_diff_raw = rng.normal(0, 0.3)
                mean_diff_aligned_null = float("nan")
            else:
                raise ValueError(kind)
            boot_rows.append({
                "subject": subject, "electrode_idx": eidx, "phoneme_pair": pp, "word_end": we,
                "replicate": r, "smin": smin, "smax": smax,
                "mean_diff_raw": mean_diff_raw,
                "mean_diff_aligned": mean_diff_raw,
                "mean_diff_aligned_null": mean_diff_aligned_null,
                "n_per_class": 8, "acoustic_peak_auc": 0.7,
            })


def _build_fixtures(tmp_path: Path) -> Path:
    """Five cells covering every branch of the per-cell loop:

    - SYNTH1/1/dn/desolate: real signal bump -> should clear the gate.
    - SYNTH1/2/dn/necessary: pure noise -> should not clear the gate.
    - SYNTH2/1/dn/desolate: tied (mean_diff_aligned_null all-NaN, D2) -> dropped from gate.
    - SYNTH2/2/dn/necessary: phon_smax beyond the grid -> no candidate windows.
    - SYNTH3/1/dn/desolate: absent from b4_bootstrap entirely -> missing data.
    """
    rng = np.random.default_rng(42)
    boot_rows: list[dict] = []
    per_cell_rows: list[dict] = []
    manifest_rows: list[dict] = []

    _add_cell(boot_rows, per_cell_rows, manifest_rows, rng,
              subject="SYNTH1", eidx=1, pp="dn", we="desolate", phon_smax=20, kind="signal")
    _add_cell(boot_rows, per_cell_rows, manifest_rows, rng,
              subject="SYNTH1", eidx=2, pp="dn", we="necessary", phon_smax=20, kind="noise")
    _add_cell(boot_rows, per_cell_rows, manifest_rows, rng,
              subject="SYNTH2", eidx=1, pp="dn", we="desolate", phon_smax=20, kind="tied")
    _add_cell(boot_rows, per_cell_rows, manifest_rows, rng,
              subject="SYNTH2", eidx=2, pp="dn", we="necessary", phon_smax=999, kind="noise")
    _add_cell(boot_rows, per_cell_rows, manifest_rows, rng,
              subject="SYNTH3", eidx=1, pp="dn", we="desolate", phon_smax=20, kind="missing")

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    pl.DataFrame(boot_rows).write_parquet(fixtures / "b4_bootstrap.parquet")
    pl.DataFrame(per_cell_rows).write_parquet(fixtures / "b4_per_cell.parquet")
    pl.DataFrame(manifest_rows).write_csv(fixtures / "filtered_manifest.csv")
    return fixtures


def _run_notebook(fixtures: Path, outdir: Path, work: Path) -> pl.DataFrame:
    jupytext = pytest.importorskip("jupytext")
    from ploomber_engine import execute_notebook

    nb = jupytext.read(NOTEBOOK)
    input_ipynb = work / "input.ipynb"
    jupytext.write(nb, input_ipynb)

    params = dict(
        b4_bootstrap_path=str(fixtures / "b4_bootstrap.parquet"),
        b4_per_cell_path=str(fixtures / "b4_per_cell.parquet"),
        filtered_manifest_path=str(fixtures / "filtered_manifest.csv"),
        outdir=str(outdir),
    )
    output_ipynb = work / "output.ipynb"
    execute_notebook(
        str(input_ipynb), str(output_ipynb),
        parameters=params, cwd=str(REPO), progress_bar=False,
    )
    return pl.read_parquet(outdir / "site_results.parquet")


@pytest.fixture(autouse=True)
def _mpl_cache(tmp_path, monkeypatch):
    cache = tmp_path / "mplcache"
    cache.mkdir(exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(cache))


def _row(df: pl.DataFrame, subject: str, eidx: int) -> dict:
    sub = df.filter((pl.col("subject") == subject) & (pl.col("electrode_idx") == eidx))
    assert sub.height == 1, f"expected exactly one row for {subject}/{eidx}, got {sub.height}"
    return sub.row(0, named=True)


def test_late_perceptual_significance_notebook_end_to_end(tmp_path):
    fixtures = _build_fixtures(tmp_path)
    outdir = tmp_path / "results"
    work = tmp_path / "work"
    work.mkdir()

    site_results = _run_notebook(fixtures, outdir, work)

    assert (outdir / "population_summary.pdf").exists()
    assert site_results.height == 5, "one row per powered B4 cell, including edge cases"

    signal = _row(site_results, "SYNTH1", 1)
    assert signal["is_tied"] is False
    assert signal["n_windows"] == 8  # desolate: smin in [20, 100) at stride 10
    assert signal["tfce_gate_pass"] is True
    assert signal["tfce_emp_p"] < 0.05
    assert signal["manual_behav_late"] == "n"

    noise = _row(site_results, "SYNTH1", 2)
    assert noise["is_tied"] is False
    assert noise["tfce_gate_pass"] is False
    assert noise["manual_behav_late"] is None

    tied = _row(site_results, "SYNTH2", 1)
    assert tied["is_tied"] is True
    assert tied["tfce_emp_p"] is None
    assert tied["tfce_gate_pass"] is False
    assert tied["tfce_fdr_pass"] is False

    no_candidates = _row(site_results, "SYNTH2", 2)
    assert no_candidates["n_windows"] == 0
    assert no_candidates["tfce_emp_p"] is None
    assert no_candidates["tfce_gate_pass"] is False

    missing = _row(site_results, "SYNTH3", 1)
    assert missing["tfce_emp_p"] is None
    assert missing["tfce_gate_pass"] is False

    # Family excludes tied / no-candidate / missing-data cells (D5: "family, not 84").
    family = site_results.filter(pl.col("tfce_emp_p").is_not_null())
    assert family.height == 2
    assert int(family["tfce_gate_pass"].sum()) == 1

    # BH-FDR over the family only, tied/missing/no-candidate rows excluded (NaN q).
    non_family = site_results.filter(pl.col("tfce_emp_p").is_null())
    assert non_family["tfce_fdr_pass"].to_list() == [False, False, False]
    assert all(np.isnan(v) for v in non_family["tfce_p_fdr"].to_list())
