"""Selectively mark causal6 outputs as up-to-date for snakemake.

After source-code changes (e.g. the stage-2 spill refactor),
``--rerun-triggers code`` would re-run all downstream rules. Outputs
that are *semantically still valid* should be touched so snakemake
skips them. Outputs that genuinely need re-running are left alone.

Per-category validation tests (output is safe to touch iff test passes):

  decoder rules (acoustic / behavior_full / behavior_hga_only /
                 ganong_full / ganong_hga_only)
      pass: ``coefficients.parquet`` exists in the per-subject decoder
            dir. Pre-coefficients runs only emitted scores +
            predictions and must be re-run.

  null rules (one per decoder)
      pass: ``escalation_log.parquet`` exists in the per-subject null
            dir AND has a per-row ``n_permutations`` column (the
            adaptive-K marker). Pre-adaptive-K runs only emitted
            ``null_scores.parquet``. Logs whose max(n_permutations)
            never reaches K1+K2 are a soft pass — warned but kept
            (no site escalated, which is legitimate).

  summarize rules (behavior_full, behavior_hga_only)
      pass: ``peak_summary_tstat_maxstat.parquet`` AND
            ``peak_summary_tstat_tfce.parquet`` exist. Pre-flavor runs
            only emitted ``peak_summary.parquet``.

  acoustic peaks
      pass: ``phon_peaks_tstat_maxstat.parquet`` exists in the
            per-subject acoustic_decoding_peaks dir.

Ganong summarize rules don't ship TFCE flavors at all in the current
Snakefile; this script SKIPS them — neither touches nor flags. Pass
``--include-ganong-summarize`` to touch them whenever all declared
rule outputs exist.

Usage::

    # Preview (default): list every (rule, subject) and its decision.
    python scripts/touch_safe_causal6_outputs.py

    # Actually run snakemake --touch on the safe set.
    python scripts/touch_safe_causal6_outputs.py --execute

The preview groups paths by directory and prints skip reasons so you
can spot smoke-test artifacts (delete by hand) before executing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

OUT_ROOT = Path("outputs/causal6")


@dataclass
class RuleSpec:
    """One rule's outputs + its per-subject validity test."""
    name: str  # snakemake rule name (for logging only)
    out_dir: str  # outputs/causal6/{out_dir}/{subject}/
    output_files: list[str]  # filenames declared as `output:` in the rule
    test_files: list[str] = field(default_factory=list)  # files whose existence proves the run is current
    test_columns: dict[str, list[str]] = field(default_factory=dict)  # parquet → required columns

    def subject_dirs(self, root: Path) -> list[Path]:
        d = root / self.out_dir
        if not d.exists():
            return []
        return sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("EC"))

    def validate(self, subject_dir: Path) -> tuple[bool, str]:
        for f in self.output_files:
            if not (subject_dir / f).exists():
                return False, f"missing declared output {f!r}"
        for f in self.test_files:
            if not (subject_dir / f).exists():
                return False, f"missing test file {f!r}"
        for fname, cols in self.test_columns.items():
            df = pl.read_parquet(subject_dir / fname)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                return False, f"{fname} missing column(s) {missing}"
        return True, ""

    def warnings(self, subject_dir: Path) -> list[str]:
        # Soft-pass diagnostics. Currently: warn if escalation_log shows no escalation.
        warns = []
        log = subject_dir / "escalation_log.parquet"
        if log.exists() and "n_permutations" in pl.read_parquet(log).columns:
            df = pl.read_parquet(log)
            n_max = int(df["n_permutations"].max())
            if "escalated" in df.columns and not df["escalated"].any():
                warns.append(f"no site escalated; n_permutations capped at {n_max}")
        return warns

    def output_paths(self, subject_dir: Path) -> list[Path]:
        return [subject_dir / f for f in self.output_files]


def build_rule_specs(*, include_ganong_summarize: bool) -> list[RuleSpec]:
    decoder_outputs = [
        "notebook.ipynb",
        "scores.parquet",
        "predictions.parquet",
        "coefficients.parquet",
    ]
    null_outputs = [
        "notebook.ipynb",
        "null_scores.parquet",
        "escalation_log.parquet",
    ]
    behavior_summarize_outputs = [
        "notebook.ipynb",
        "peak_summary.parquet",
        "peak_summary_tstat_maxstat.parquet",
        "peak_summary_foldmean_tfce.parquet",
        "peak_summary_tstat_tfce.parquet",
        "peak_predictions.parquet",
    ]
    ganong_summarize_outputs = [
        "notebook.ipynb",
        "peak_summary.parquet",
        "peak_predictions.parquet",
    ]
    acoustic_peaks_outputs = [
        "notebook.ipynb",
        "phon_peaks.parquet",
        "phon_peaks_tstat_maxstat.parquet",
        "phon_roc_auc_searchlight.parquet",
    ]

    specs = [
        # Decoders
        RuleSpec("acoustic_decoding_single_electrode",
                 "acoustic_decoding_single_electrode",
                 decoder_outputs,
                 test_files=["coefficients.parquet"]),
        RuleSpec("behavior_decoding_single_electrode",
                 "behavior_decoding_single_electrode",
                 decoder_outputs,
                 test_files=["coefficients.parquet"]),
        RuleSpec("behavior_decoding_single_electrode_hga_only",
                 "behavior_decoding_single_electrode_hga_only",
                 decoder_outputs,
                 test_files=["coefficients.parquet"]),
        RuleSpec("ganong_decoding_single_electrode",
                 "ganong_decoding_single_electrode",
                 decoder_outputs,
                 test_files=["coefficients.parquet"]),
        RuleSpec("ganong_decoding_single_electrode_hga_only",
                 "ganong_decoding_single_electrode_hga_only",
                 decoder_outputs,
                 test_files=["coefficients.parquet"]),
        # Nulls
        RuleSpec("acoustic_decoding_null",
                 "acoustic_decoding_null",
                 null_outputs,
                 test_files=["escalation_log.parquet"],
                 test_columns={"escalation_log.parquet": ["n_permutations"]}),
        RuleSpec("behavior_decoding_single_electrode_null",
                 "behavior_decoding_single_electrode_null",
                 null_outputs,
                 test_files=["escalation_log.parquet"],
                 test_columns={"escalation_log.parquet": ["n_permutations"]}),
        RuleSpec("behavior_decoding_single_electrode_hga_only_null",
                 "behavior_decoding_single_electrode_hga_only_null",
                 null_outputs,
                 test_files=["escalation_log.parquet"],
                 test_columns={"escalation_log.parquet": ["n_permutations"]}),
        RuleSpec("ganong_decoding_null",
                 "ganong_decoding_null",
                 null_outputs,
                 test_files=["escalation_log.parquet"],
                 test_columns={"escalation_log.parquet": ["n_permutations"]}),
        RuleSpec("ganong_decoding_hga_only_null",
                 "ganong_decoding_hga_only_null",
                 null_outputs,
                 test_files=["escalation_log.parquet"],
                 test_columns={"escalation_log.parquet": ["n_permutations"]}),
        # Behavior summarize (TFCE-flavor check)
        RuleSpec("behavior_decoding_single_electrode_summarize",
                 "behavior_decoding_single_electrode_summarize",
                 behavior_summarize_outputs,
                 test_files=[
                     "peak_summary_tstat_maxstat.parquet",
                     "peak_summary_tstat_tfce.parquet",
                 ]),
        RuleSpec("behavior_decoding_single_electrode_hga_only_summarize",
                 "behavior_decoding_single_electrode_hga_only_summarize",
                 behavior_summarize_outputs,
                 test_files=[
                     "peak_summary_tstat_maxstat.parquet",
                     "peak_summary_tstat_tfce.parquet",
                 ]),
        # Acoustic peaks (no TFCE — different filename)
        RuleSpec("acoustic_decoding_peaks",
                 "acoustic_decoding_peaks",
                 acoustic_peaks_outputs,
                 test_files=["phon_peaks_tstat_maxstat.parquet"]),
    ]
    if include_ganong_summarize:
        specs.extend([
            RuleSpec("ganong_decoding_summarize",
                     "ganong_decoding_summarize",
                     ganong_summarize_outputs),
            RuleSpec("ganong_decoding_hga_only_summarize",
                     "ganong_decoding_hga_only_summarize",
                     ganong_summarize_outputs),
        ])
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="Run `snakemake --touch` on the safe set. Default: dry-run preview.")
    ap.add_argument("--snakefile", default="workflows/causal6.Snakefile")
    ap.add_argument("--configfile", default="config.yaml")
    ap.add_argument("--cores", default="1", help="--cores forwarded to snakemake")
    ap.add_argument("--root", default=str(OUT_ROOT), type=Path,
                    help="outputs/causal6 root (default: outputs/causal6)")
    ap.add_argument("--include-ganong-summarize", action="store_true",
                    help="Touch ganong_*_summarize whenever all declared outputs exist "
                         "(no TFCE check — those rules don't ship TFCE).")
    args = ap.parse_args()

    specs = build_rule_specs(include_ganong_summarize=args.include_ganong_summarize)
    safe_paths: list[Path] = []
    skipped: list[tuple[str, str, str]] = []  # (rule, subject, reason)
    warned: list[tuple[str, str, str]] = []   # (rule, subject, warning)

    for spec in specs:
        for sd in spec.subject_dirs(args.root):
            ok, reason = spec.validate(sd)
            if ok:
                safe_paths.extend(spec.output_paths(sd))
                for w in spec.warnings(sd):
                    warned.append((spec.name, sd.name, w))
            else:
                skipped.append((spec.name, sd.name, reason))

    # --- preview ---
    by_dir: dict[str, list[str]] = {}
    for p in safe_paths:
        by_dir.setdefault(str(p.parent), []).append(p.name)

    print(f"=== {len(safe_paths)} files to touch across {len(by_dir)} directories ===")
    for d in sorted(by_dir):
        print(f"\n  {d}/")
        for f in sorted(by_dir[d]):
            print(f"    {f}")

    if warned:
        print(f"\n=== {len(warned)} soft-pass warnings (still touching) ===")
        for rule, subj, w in sorted(warned):
            print(f"  {rule}/{subj}: {w}")

    if skipped:
        print(f"\n=== {len(skipped)} (rule, subject) skipped — will be re-run ===")
        for rule, subj, reason in sorted(skipped):
            print(f"  {rule}/{subj}: {reason}")

    if not args.execute:
        print("\n(dry-run; pass --execute to run `snakemake --touch` on the safe set)")
        return 0

    if not safe_paths:
        print("\nNothing to touch.")
        return 0

    cmd = [
        "snakemake",
        "--snakefile", args.snakefile,
        "--configfile", args.configfile,
        "--cores", args.cores,
        "--touch",
        *(str(p) for p in safe_paths),
    ]
    print(f"\n+ {' '.join(cmd[:8])} ... ({len(safe_paths)} paths)")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
