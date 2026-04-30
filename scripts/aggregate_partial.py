"""Aggregate per-subject peak/significance parquets over the subjects that
currently have outputs, mirroring `notebooks/causal6/significance_aggregate.py`
without going through Snakemake.

Use this when the upstream pipeline is still running (or some subjects haven't
converged) and you want a partial aggregate over whatever's finished. The
output overwrites the canonical aggregate path used by Snakemake; re-run the
proper Snakemake aggregator once upstream is fully fresh to restore canonical
FDR over the full subject set.

Examples:
    # Acoustic peaks, foldmean_maxstat (the v1 phon_peaks.parquet)
    uv run python scripts/aggregate_partial.py acoustic

    # Acoustic peaks, t-stat maxstat
    uv run python scripts/aggregate_partial.py acoustic_tstat_maxstat

    # Behavior HGA-only — runs all four flavors
    uv run python scripts/aggregate_partial.py behav_hga_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path("outputs/causal6")

# Maps job-name -> (per-subject glob, aggregate output path).
JOBS = {
    "acoustic": (
        ROOT / "acoustic_decoding_peaks/*/phon_peaks.parquet",
        ROOT / "acoustic_decoding_peaks/phon_peaks_all.parquet",
    ),
    "acoustic_tstat_maxstat": (
        ROOT / "acoustic_decoding_peaks/*/phon_peaks_tstat_maxstat.parquet",
        ROOT / "acoustic_decoding_peaks/phon_peaks_tstat_maxstat_all.parquet",
    ),
    "behav_hga_only_foldmean_maxstat": (
        ROOT / "behavior_decoding_single_electrode_hga_only_summarize/*/peak_summary.parquet",
        ROOT / "behavior_decoding_single_electrode_hga_only_summarize/peak_summary_all.parquet",
    ),
    "behav_hga_only_tstat_maxstat": (
        ROOT / "behavior_decoding_single_electrode_hga_only_summarize/*/peak_summary_tstat_maxstat.parquet",
        ROOT / "behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_maxstat_all.parquet",
    ),
    "behav_hga_only_foldmean_tfce": (
        ROOT / "behavior_decoding_single_electrode_hga_only_summarize/*/peak_summary_foldmean_tfce.parquet",
        ROOT / "behavior_decoding_single_electrode_hga_only_summarize/peak_summary_foldmean_tfce_all.parquet",
    ),
    "behav_hga_only_tstat_tfce": (
        ROOT / "behavior_decoding_single_electrode_hga_only_summarize/*/peak_summary_tstat_tfce.parquet",
        ROOT / "behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_tfce_all.parquet",
    ),
}

# Convenience aliases that fan out to all flavors of a decoder.
GROUPS = {
    "acoustic_all_flavors": ["acoustic", "acoustic_tstat_maxstat"],
    "behav_hga_only": [
        "behav_hga_only_foldmean_maxstat",
        "behav_hga_only_tstat_maxstat",
        "behav_hga_only_foldmean_tfce",
        "behav_hga_only_tstat_tfce",
    ],
}


def aggregate_one(per_subject_glob: Path, output_path: Path, *, fdr_alpha: float) -> None:
    paths = sorted(Path(".").glob(str(per_subject_glob)))
    if not paths:
        print(f"[skip] no inputs match {per_subject_glob}")
        return

    dfs = [pd.read_parquet(p) for p in paths]
    combined = pd.concat(dfs, ignore_index=True)

    _, q_values, _, _ = multipletests(
        combined["p_value"].values, alpha=fdr_alpha, method="fdr_bh"
    )
    combined["q_value"] = q_values
    combined["significant"] = combined["q_value"] < fdr_alpha

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)

    subjects_in = sorted(combined["subject"].unique().tolist())
    n_sig = int(combined["significant"].sum())
    print(
        f"[ok]   {output_path.name}: {len(combined)} rows from {len(paths)} subjects "
        f"({subjects_in}); {n_sig} significant at q<{fdr_alpha}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=list(JOBS) + list(GROUPS))
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    args = ap.parse_args()

    job_names = GROUPS.get(args.job, [args.job])
    for name in job_names:
        glob, out = JOBS[name]
        aggregate_one(glob, out, fdr_alpha=args.fdr_alpha)


if __name__ == "__main__":
    main()
