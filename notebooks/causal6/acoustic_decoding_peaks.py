# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# causal6: per-subject acoustic-decoding peak-finding with null-standardized
# significance.
#
# Emits three flavors per subject (each fed into a separate aggregate+FDR rule
# downstream):
#   * foldmean_maxstat  — statistic = fold-mean AUC; peak selected by argmin
#     pointwise p + max-stat correction. Existing v1 contract.
#   * tstat_maxstat     — statistic = (fold_mean - 0.5) / (fold_std / sqrt(n_folds));
#     variance-normalized. Same peak selection + max-stat correction.
#   * foldmean_tfce     — TFCE-enhanced fold-mean AUC (threshold=0.5, above-chance
#     windows only), then max-stat correction. Included for reconciliation: tests
#     whether cluster credit changes site counts despite the narrow window (W ≈ 3-4).
#
# Inputs:
#   scores.parquet         — real fold-wise test AUC per (site, window)
#   null_scores.parquet    — permutation-null fold-wise AUC per (site, window, perm)
#
# Outputs:
#   phon_peaks.parquet                 — foldmean_maxstat (unchanged schema).
#   phon_peaks_tstat_maxstat.parquet   — tstat_maxstat, same schema.
#   phon_peaks_foldmean_tfce.parquet   — foldmean_tfce, same schema.
#   phon_roc_auc_searchlight.parquet   — fold-mean AUC per (site, window) for
#                                         diagnostic plots (unchanged).

# %%
from pathlib import Path

import polars as pl

from src.models.causal6_aggregates import (
    SITE_KEYS_ACOUSTIC as site_keys,
    FlavorSpec,
    aggregate_acoustic,
    tfce_enhanced_peak_test,
)
from src.models.significance import null_standardized_peak_test

# %% tags=["parameters"]
subject = "EC282"
scores_path = f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet"
null_scores_path = f"outputs/causal6/acoustic_decoding_null/{subject}/null_scores.parquet"
outdir = "."

target = "categorical_acoustic_cue"
peak_search_smin = 0
peak_search_smax = 290

# %%
window_keys = site_keys + ["smin", "smax"]

# %% [markdown]
# Aggregate each fold into (fold_mean, fold_std, t_stat) per (site, window)
# for real and per (site, window, perm) for null. Both flavors share the
# same aggregation; only the statistic column fed to the peak test differs.

# %%
real_agg, null_agg = aggregate_acoustic(
    pl.read_parquet(scores_path),
    pl.read_parquet(null_scores_path),
    target=target,
    peak_search_smin=peak_search_smin,
    peak_search_smax=peak_search_smax,
)


# %%
def _run_maxstat(stat_col: str, rename_real_to: str) -> pl.DataFrame:
    real_in = real_agg.select(window_keys + [stat_col]).rename({stat_col: "statistic"})
    null_in = null_agg.select(window_keys + ["permutation_idx", stat_col]).rename(
        {stat_col: "statistic"}
    )
    peaks, _ = null_standardized_peak_test(
        real_in, null_in,
        site_keys=site_keys, window_keys=["smin", "smax"], stat_col="statistic",
    )
    return peaks.rename({
        "peak_smin": "smin", "peak_smax": "smax", "real_statistic": rename_real_to,
    })


# foldmean_maxstat — backward-compatible phon_peaks.parquet
phon_peaks = _run_maxstat("fold_mean", rename_real_to="test_roc_auc")
print(
    f"[foldmean_maxstat] {phon_peaks.height} sites: "
    f"{(phon_peaks['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# tstat_maxstat
phon_peaks_tstat = _run_maxstat("t_stat", rename_real_to="t_stat")
print(
    f"[tstat_maxstat]    {phon_peaks_tstat.height} sites: "
    f"{(phon_peaks_tstat['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# foldmean_tfce — TFCE on fold-mean AUC (threshold=0.5: above-chance windows only)
_FOLDMEAN_TFCE = FlavorSpec("fold_mean", apply_tfce=True, tfce_threshold=0.5)
phon_peaks_foldmean_tfce = tfce_enhanced_peak_test(
    real_agg, null_agg,
    site_keys=site_keys,
    flavor=_FOLDMEAN_TFCE,
).rename({"peak_smin": "smin", "peak_smax": "smax", "real_statistic": "test_roc_auc"})
print(
    f"[foldmean_tfce]    {phon_peaks_foldmean_tfce.height} sites: "
    f"{(phon_peaks_foldmean_tfce['p_value'] < 0.05).sum()} with p<0.05 (uncorrected)"
)

# Diagnostic searchlight: per-(site, window) fold-mean AUC, same as before.
phon_roc_auc_searchlight = real_agg.select(
    window_keys + ["fold_mean"]
).rename({"fold_mean": "test_roc_auc"})

# %%
outdir = Path(outdir)
phon_peaks.write_parquet(outdir / "phon_peaks.parquet")
phon_peaks_tstat.write_parquet(outdir / "phon_peaks_tstat_maxstat.parquet")
phon_peaks_foldmean_tfce.write_parquet(outdir / "phon_peaks_foldmean_tfce.parquet")
phon_roc_auc_searchlight.write_parquet(outdir / "phon_roc_auc_searchlight.parquet")
print(
    f"Wrote phon_peaks.parquet ({phon_peaks.height} rows), "
    f"phon_peaks_tstat_maxstat.parquet ({phon_peaks_tstat.height} rows), "
    f"phon_peaks_foldmean_tfce.parquet ({phon_peaks_foldmean_tfce.height} rows), "
    f"phon_roc_auc_searchlight.parquet ({phon_roc_auc_searchlight.height} rows) to {outdir}"
)
