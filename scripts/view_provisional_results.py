"""Provisional results viewer for causal6 pipeline.

Prints a formatted summary to stdout. No files written.
Usage: ./.venv/bin/python scripts/view_provisional_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path("outputs/causal6")
EPOCH_TMIN = -0.4
EPOCH_SFREQ = 100.0

SITE_KEYS_BEHAV = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
SITE_KEYS_ACOUSTIC = ["subject", "electrode_idx", "phoneme_pair"]


def _smin_to_ms(smin: int | float) -> float:
    return (smin / EPOCH_SFREQ + EPOCH_TMIN) * 1000


def _section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def _auc_summary(arr: np.ndarray, label: str = "peak AUC") -> str:
    return (
        f"{label}: median={np.median(arr):.3f}"
        f"  p75={np.percentile(arr, 75):.3f}"
        f"  max={np.max(arr):.3f}"
    )


# ---------------------------------------------------------------------------
# Section 1: Acoustic — read existing phon_peaks.parquet (already computed)
# ---------------------------------------------------------------------------

def acoustic_significance() -> None:
    _section("ACOUSTIC DECODING  —  significance  (peaks already computed)")

    paths = sorted(ROOT.glob("acoustic_decoding_peaks/*/phon_peaks.parquet"))
    if not paths:
        print("  No acoustic peaks found.")
        return

    for p in paths:
        subject = p.parent.name
        df = pl.read_parquet(p)
        n_total = len(df)
        n_sig = int((df["p_value"] < 0.05).sum())

        aucs = df["test_roc_auc"].to_numpy()
        peak_ms = np.array([_smin_to_ms(s) for s in df["smin"].to_list()])

        n_perm = int(df["n_permutations"].max())
        min_achievable_p = 1.0 / (n_perm + 1) if n_perm > 0 else float("nan")

        print(f"\n  {subject}  ({n_total} sites = electrodes × phoneme_pairs)")
        print(f"    permutations completed (max across sites): {n_perm}"
              f"  →  min achievable p = {min_achievable_p:.4f}")
        print(f"    significant (p < 0.05, maxstat-corrected):  {n_sig}/{n_total}"
              f"  ({100 * n_sig / n_total:.0f}%)")
        print(f"    {_auc_summary(aucs)}")
        print(f"    peak timing (ms post onset): median={np.median(peak_ms):.0f}"
              f"  p25={np.percentile(peak_ms, 25):.0f}"
              f"  p75={np.percentile(peak_ms, 75):.0f}")

        for pp in sorted(df["phoneme_pair"].unique().to_list()):
            sub = df.filter(pl.col("phoneme_pair") == pp)
            n_pp_sig = int((sub["p_value"] < 0.05).sum())
            print(f"    {pp}: {n_pp_sig}/{len(sub)} significant")


# ---------------------------------------------------------------------------
# Section 2: Behavior HGA-only — raw peak AUC (all subjects with scores)
# ---------------------------------------------------------------------------

def behavior_raw() -> None:
    _section("BEHAVIOR HGA-ONLY  —  raw peak ROC-AUC  (all subjects with scores)")

    paths = sorted(ROOT.glob("behavior_decoding_single_electrode_hga_only/*/scores.parquet"))
    if not paths:
        print("  No behavior scores found.")
        return

    window_keys = SITE_KEYS_BEHAV + ["smin", "smax"]

    for p in paths:
        subject = p.parent.name
        df = pl.read_parquet(p).filter(pl.col("model") == "full")

        fold_mean = (
            df.group_by(window_keys)
            .agg(pl.col("test_roc_auc").mean().alias("fold_mean"))
        )
        peak_per_site = (
            fold_mean.group_by(SITE_KEYS_BEHAV)
            .agg(pl.col("fold_mean").max().alias("peak_auc"))
        )

        aucs = peak_per_site["peak_auc"].to_numpy()
        n_sites = len(aucs)
        n_above_06 = int((peak_per_site["peak_auc"] > 0.6).sum())

        has_null = (
            ROOT / f"behavior_decoding_single_electrode_hga_only_null/{subject}/null_scores.parquet"
        ).exists()
        null_tag = "(null done)" if has_null else "(null pending)"

        print(f"\n  {subject}  {null_tag}  ({n_sites} sites = electrodes × pp × word_end)")
        print(f"    {_auc_summary(aucs)}")
        print(f"    sites > 0.60 AUC:  {n_above_06}/{n_sites}  ({100 * n_above_06 / n_sites:.0f}%)")

        # Per-phoneme-pair breakdown
        for pp in sorted(peak_per_site["phoneme_pair"].unique().to_list()):
            sub = peak_per_site.filter(pl.col("phoneme_pair") == pp)
            sub_aucs = sub["peak_auc"].to_numpy()
            n_sub_above = int((sub["peak_auc"] > 0.6).sum())
            print(f"    {pp} ({len(sub)} sites): median={np.median(sub_aucs):.3f}"
                  f"  max={np.max(sub_aucs):.3f}"
                  f"  >0.60: {n_sub_above}/{len(sub)}")


# ---------------------------------------------------------------------------
# Section 3: Behavior HGA-only — significance (subjects with nulls)
# ---------------------------------------------------------------------------

def behavior_significance() -> None:
    _section("BEHAVIOR HGA-ONLY  —  significance  (subjects with null tests)")
    print(
        "  NOTE: no BH-FDR applied — partial data. Run\n"
        "  `uv run python scripts/aggregate_partial.py behav_hga_only`\n"
        "  after the summarize step finishes for all available subjects."
    )

    null_dir = ROOT / "behavior_decoding_single_electrode_hga_only_null"
    score_dir = ROOT / "behavior_decoding_single_electrode_hga_only"

    null_paths = sorted(null_dir.glob("*/null_scores.parquet"))
    if not null_paths:
        print("\n  No null scores found.")
        return

    from src.models.causal6_aggregates import aggregate_behavior_hga_only
    from src.models.significance import null_standardized_peak_test

    for null_path in null_paths:
        subject = null_path.parent.name
        scores_path = score_dir / subject / "scores.parquet"
        if not scores_path.exists():
            print(f"\n  {subject}: null exists but no real scores — skipping.")
            continue

        print(f"\n  {subject}: computing significance on the fly …", flush=True)

        real_agg, null_agg = aggregate_behavior_hga_only(
            real_scores=pl.read_parquet(scores_path),
            null_scores=pl.read_parquet(null_path),
            epoch_tmin=EPOCH_TMIN,
            epoch_sfreq=EPOCH_SFREQ,
            behav_peak_post_offset_s=0.2,
            peak_search_smin=0,
            peak_search_smax=290,
        )

        peaks, _ = null_standardized_peak_test(
            real_agg,
            null_agg,
            site_keys=SITE_KEYS_BEHAV,
            window_keys=["smin", "smax"],
            stat_col="fold_mean",
        )

        n_total = len(peaks)
        n_sig = int((peaks["p_value"] < 0.05).sum())

        # Report permutation count — minimum achievable p = 1/(K+1)
        n_perm = int(peaks["n_permutations"].max())
        min_achievable_p = 1.0 / (n_perm + 1) if n_perm > 0 else float("nan")

        aucs = peaks["real_statistic"].to_numpy()
        peak_ms = np.array([_smin_to_ms(s) for s in peaks["peak_smin"].to_list()])
        pv = peaks["p_value"].drop_nulls().to_numpy()

        print(f"    permutations completed (max across sites): {n_perm}"
              f"  →  min achievable p = {min_achievable_p:.4f}")
        print(f"    significant (p < 0.05, maxstat-corrected, uncorrected for FDR):  "
              f"{n_sig}/{n_total}  ({100 * n_sig / n_total:.0f}%)")
        print(f"    {_auc_summary(aucs, label='peak fold-mean AUC')}")
        print(f"    peak timing (ms post onset): median={np.median(peak_ms):.0f}"
              f"  p25={np.percentile(peak_ms, 25):.0f}"
              f"  p75={np.percentile(peak_ms, 75):.0f}")
        print(f"    p-value distribution: min={np.min(pv):.4f}"
              f"  median={np.median(pv):.4f}"
              f"  max={np.max(pv):.4f}")

        for pp in sorted(peaks["phoneme_pair"].unique().to_list()):
            sub = peaks.filter(pl.col("phoneme_pair") == pp)
            n_pp_sig = int((sub["p_value"] < 0.05).sum())
            print(f"    {pp}: {n_pp_sig}/{len(sub)} significant")


if __name__ == "__main__":
    acoustic_significance()
    behavior_raw()
    behavior_significance()
    print()
