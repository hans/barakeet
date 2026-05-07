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
# # causal6: provisional results viewer
#
# Reads whatever outputs exist in `outputs/causal6/` and shows:
# 1. **Acoustic** — significance from `acoustic_decoding_peaks/*/phon_peaks.parquet`
# 2. **Behavior HGA-only (raw)** — peak fold-mean AUC across all subjects with scores
# 3. **Behavior HGA-only (significance)** — on-the-fly significance for subjects
#    that have both real scores and null permutations
#
# No files are written. Re-run the cell block you care about at any time.

# %%
import os
import yaml
# Cap threading so this doesn't monopolize the node.
os.environ.setdefault("POLARS_MAX_THREADS", "8")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pathlib import Path

from src.data import get_electrode_df
from src.models.causal6_aggregates import SITE_KEYS_BEHAVIOR_HGA_ONLY, _behavior_offset_samples
from src.models.significance import null_standardized_peak_test

with open("config.yaml") as _f:
    _config = yaml.safe_load(_f)

ROOT = Path("outputs/causal6")
EPOCH_TMIN = -0.4
EPOCH_SFREQ = 100.0
SITE_KEYS_BEHAV = SITE_KEYS_BEHAVIOR_HGA_ONLY  # ["subject", "electrode_idx", "phoneme_pair", "word_end"]
WINDOW_KEYS_BEHAV = SITE_KEYS_BEHAV + ["smin", "smax"]

# Behavior decoder window search constraints (shared across sections)
_PEAK_SEARCH_SMIN = 0
_PEAK_SEARCH_SMAX = 290
_BEHAV_POST_OFFSET_S = 0.2  # max time post word-end to include
_OFFSET_SAMPLES = _behavior_offset_samples(EPOCH_TMIN, EPOCH_SFREQ, _BEHAV_POST_OFFSET_S)

# Acoustic decoder window search constraints
_AC_PEAK_SEARCH_SMIN = _config["analysis"]["decoding"]["acoustic_peak_search_smin"]
_AC_PEAK_SEARCH_SMAX = _config["analysis"]["decoding"]["acoustic_peak_search_smax"]
_AC_TARGET = "categorical_acoustic_cue"
_AC_SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]
_AC_WINDOW_KEYS = _AC_SITE_KEYS + ["smin", "smax"]


def _filter_window_expr() -> pl.Expr:
    """Keep only windows within the causal search range for each word_end."""
    return (
        (pl.col("smin") >= _AC_PEAK_SEARCH_SMIN)
        & (pl.col("smax") <= pl.col("_smax_limit"))
        & (pl.col("smax") <= _PEAK_SEARCH_SMAX)
    )


def smin_to_ms(s) -> np.ndarray:
    return (np.asarray(s) / EPOCH_SFREQ + EPOCH_TMIN) * 1000


# %% [markdown]
# ## 1  Acoustic decoding — significance
#
# `acoustic_decoding_peaks/*/phon_peaks.parquet` is produced by the
# `acoustic_decoding_peaks` Snakemake rule and already contains maxstat-corrected
# p-values.  No recomputation needed here.

# %%
ac_peak_paths = sorted(ROOT.glob("acoustic_decoding_peaks/*/phon_peaks.parquet"))
ac_frames = {p.parent.name: pl.read_parquet(p) for p in ac_peak_paths}

for subject, df in ac_frames.items():
    n_total = len(df)
    n_sig = int((df["p_value"] < 0.05).sum())
    n_perm = int(df["n_permutations"].max())
    min_p = 1.0 / (n_perm + 1)
    print(
        f"{subject}  {n_total} sites  |  "
        f"permutations: {n_perm} (min achievable p = {min_p:.4f})  |  "
        f"significant (p<0.05): {n_sig}/{n_total}"
    )
    for pp in sorted(df["phoneme_pair"].unique().to_list()):
        sub = df.filter(pl.col("phoneme_pair") == pp)
        print(
            f"  {pp}:  {int((sub['p_value'] < 0.05).sum())}/{len(sub)} sig  "
            f"  peak-AUC median={sub['test_roc_auc'].median():.3f}  "
            f"max={sub['test_roc_auc'].max():.3f}"
        )

# %%
if ac_frames:
    ac_all = pl.concat(list(ac_frames.values()))
    subjects = sorted(ac_frames)
    n_cols = max(len(ac_frames), 1)

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)

    for col, subject in enumerate(subjects):
        df = ac_frames[subject]
        aucs = df["test_roc_auc"].to_numpy()
        peak_ms = smin_to_ms(df["smin"].to_numpy())
        pv = df["p_value"].to_numpy()
        n_perm = int(df["n_permutations"].max())
        min_p = 1.0 / (n_perm + 1)

        # Peak AUC distribution, coloured by significance
        sig_mask = pv < 0.05
        ax = axes[0, col]
        ax.hist(aucs[~sig_mask], bins=20, color="steelblue", alpha=0.7, label="n.s.")
        ax.hist(aucs[sig_mask], bins=20, color="tomato", alpha=0.8, label="p<0.05")
        ax.axvline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xlabel("peak ROC-AUC")
        ax.set_ylabel("sites")
        ax.set_title(f"{subject} — acoustic peak AUC\n({n_perm} perms, min p={min_p:.3f})")
        ax.legend(fontsize=8)

        # Peak timing distribution
        ax2 = axes[1, col]
        ax2.hist(peak_ms[~sig_mask], bins=20, color="steelblue", alpha=0.7, label="n.s.")
        ax2.hist(peak_ms[sig_mask], bins=20, color="tomato", alpha=0.8, label="p<0.05")
        ax2.axvline(0, color="k", lw=0.8, ls="--", label="word onset")
        ax2.set_xlabel("peak window onset (ms post word onset)")
        ax2.set_ylabel("sites")
        ax2.set_title(f"{subject} — acoustic peak timing")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 2  Acoustic — fold t-stats for brain plotting
#
# Same fold t-stat approach as the behavior section below, but for acoustic
# decoding.  Uses `acoustic_decoding_single_electrode/*/scores.parquet`
# (fold-level AUC per site × window) with the same search window as the peaks
# rule (`smin` 0–290 samples, `target == "categorical_acoustic_cue"`).
#
# **Output**: `outputs/causal6/brain_plot_acoustic_tstats.parquet`
# Columns: subject, electrode\_idx, phoneme\_pair,
#          peak\_smin, peak\_auc, fold\_tstat, n\_folds, x, y, z, roi

# %%
ac_brain_frames: list[pl.DataFrame] = []

for _p in sorted(ROOT.glob("acoustic_decoding_single_electrode/*/scores.parquet")):
    _subject = _p.parent.name
    _df = (
        pl.read_parquet(_p)
        .filter(
            (pl.col("target") == _AC_TARGET)
            & (pl.col("smin") >= _AC_PEAK_SEARCH_SMIN)
            & (pl.col("smax") <= _AC_PEAK_SEARCH_SMAX)
        )
    )

    _win_stats = (
        _df.group_by(_AC_WINDOW_KEYS)
        .agg(
            pl.col("test_roc_auc").mean().alias("fold_mean"),
            pl.col("test_roc_auc").std().alias("fold_std"),
            pl.col("test_roc_auc").len().alias("n_folds"),
        )
    )

    _peak = (
        _win_stats.group_by(_AC_SITE_KEYS)
        .agg(
            pl.col("fold_mean").max().alias("peak_auc"),
            pl.col("smin").get(pl.col("fold_mean").arg_max()).alias("peak_smin"),
            pl.col("smax").get(pl.col("fold_mean").arg_max()).alias("peak_smax"),
            pl.col("fold_std").get(pl.col("fold_mean").arg_max()).alias("peak_fold_std"),
            pl.col("n_folds").get(pl.col("fold_mean").arg_max()).alias("n_folds"),
        )
        .with_columns(
            (
                (pl.col("peak_auc") - 0.5)
                / (
                    pl.max_horizontal(pl.col("peak_fold_std"), pl.lit(1e-6))
                    / pl.col("n_folds").cast(pl.Float64).sqrt()
                )
            ).alias("fold_tstat")
        )
    )

    _elec_df = get_electrode_df(_subject)
    _elec_tmp = _elec_df.reset_index()[["electrode_idx", "x", "y", "z", "roi"]]
    _elec_tmp["roi"] = _elec_tmp["roi"].astype(str)
    _elec_pl = pl.from_pandas(_elec_tmp)
    ac_brain_frames.append(_peak.join(_elec_pl, on="electrode_idx", how="left"))

    _n = len(_peak)
    _t = _peak["fold_tstat"].to_numpy()
    print(
        f"{_subject}: {_n} sites  |  "
        f"fold_tstat  median={np.median(_t):.2f}  "
        f"max={_t.max():.2f}  "
        f">2.13 (≈p<0.05, df=4): {int((_t > 2.13).sum())}/{_n}"
    )
    for _pp in sorted(_peak["phoneme_pair"].unique().to_list()):
        _sub = _peak.filter(pl.col("phoneme_pair") == _pp)
        _st = _sub["fold_tstat"].to_numpy()
        print(
            f"  {_pp} ({len(_sub)} sites):  "
            f"t median={np.median(_st):.2f}  max={_st.max():.2f}  "
            f"AUC median={_sub['peak_auc'].median():.3f}"
        )

# %%
if ac_brain_frames:
    ac_brain_df = pl.concat(ac_brain_frames)
    _ac_out = ROOT / "brain_plot_acoustic_tstats.parquet"
    ac_brain_df.write_parquet(_ac_out)
    print(f"Written: {_ac_out}  ({len(ac_brain_df)} rows × {ac_brain_df.width} cols)")

# %% [markdown]
# ## 4  Behavior HGA-only — raw peak ROC-AUC
#
# For every subject with a `scores.parquet`, compute the per-site peak
# fold-mean AUC across all windows (no null needed).

# %%
beh_score_paths = sorted(
    ROOT.glob("behavior_decoding_single_electrode_hga_only/*/scores.parquet")
)
beh_peak_frames: dict[str, pl.DataFrame] = {}

for p in beh_score_paths:
    subject = p.parent.name
    df = (
        pl.read_parquet(p)
        .filter(pl.col("model") == "full")
        .with_columns(
            pl.col("word_end")
            .replace_strict(_OFFSET_SAMPLES, default=None)
            .alias("_smax_limit")
        )
        .filter(_filter_window_expr())
        .drop("_smax_limit")
    )
    fold_mean = (
        df.group_by(WINDOW_KEYS_BEHAV)
        .agg(pl.col("test_roc_auc").mean().alias("fold_mean"))
    )
    peak_per_site = (
        fold_mean.group_by(SITE_KEYS_BEHAV)
        .agg(
            pl.col("fold_mean").max().alias("peak_auc"),
            pl.col("smin").get(pl.col("fold_mean").arg_max()).alias("peak_smin"),
        )
    )
    has_null = (
        ROOT / f"behavior_decoding_single_electrode_hga_only_null/{subject}/null_scores.parquet"
    ).exists()
    beh_peak_frames[subject] = peak_per_site.with_columns(
        pl.lit(has_null).alias("has_null")
    )
    n = len(peak_per_site)
    n_above = int((peak_per_site["peak_auc"] > 0.6).sum())
    null_tag = "(null done)" if has_null else "(null pending)"
    print(
        f"{subject} {null_tag}  {n} sites  |  "
        f"peak-AUC median={peak_per_site['peak_auc'].median():.3f}  "
        f"p75={np.percentile(peak_per_site['peak_auc'].to_numpy(), 75):.3f}  "
        f"max={peak_per_site['peak_auc'].max():.3f}  |  "
        f">0.60: {n_above}/{n}"
    )
    for pp in sorted(peak_per_site["phoneme_pair"].unique().to_list()):
        sub = peak_per_site.filter(pl.col("phoneme_pair") == pp)
        print(
            f"  {pp} ({len(sub)} sites):  "
            f"median={sub['peak_auc'].median():.3f}  "
            f"max={sub['peak_auc'].max():.3f}  "
            f">0.60: {int((sub['peak_auc'] > 0.6).sum())}/{len(sub)}"
        )

# %%
if beh_peak_frames:
    subjects = sorted(beh_peak_frames)
    phoneme_pairs = sorted(
        pl.concat(list(beh_peak_frames.values()))["phoneme_pair"].unique().to_list()
    )
    colors = {"bm": "#4C8BE2", "dn": "#E26B4C", "pb": "#4CE28B"}

    fig, axes = plt.subplots(
        len(subjects), 2, figsize=(12, 4 * len(subjects)), squeeze=False
    )

    for row, subject in enumerate(subjects):
        df = beh_peak_frames[subject]

        # Left: peak AUC distributions by phoneme pair
        ax = axes[row, 0]
        for pp in phoneme_pairs:
            aucs = df.filter(pl.col("phoneme_pair") == pp)["peak_auc"].to_numpy()
            ax.hist(aucs, bins=15, alpha=0.6, label=pp, color=colors.get(pp))
        ax.axvline(0.5, color="k", lw=0.8, ls="--")
        ax.axvline(0.6, color="k", lw=0.8, ls=":", label="0.60 threshold")
        ax.set_xlabel("peak fold-mean ROC-AUC")
        ax.set_ylabel("sites")
        ax.set_title(f"{subject} — behavior raw peak AUC")
        ax.legend(fontsize=8)

        # Right: peak timing by phoneme pair
        ax2 = axes[row, 1]
        for pp in phoneme_pairs:
            sub = df.filter(pl.col("phoneme_pair") == pp)
            peak_ms = smin_to_ms(sub["peak_smin"].to_numpy())
            ax2.hist(peak_ms, bins=15, alpha=0.6, label=pp, color=colors.get(pp))
        ax2.axvline(0, color="k", lw=0.8, ls="--", label="word onset")
        ax2.set_xlabel("peak window onset (ms post word onset)")
        ax2.set_ylabel("sites")
        ax2.set_title(f"{subject} — behavior peak timing")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ### Interactive: filter by peak AUC → inspect timing distribution
#
# Suspicious pre-onset peaks show up clearly when you raise the AUC threshold:
# if they survive a high bar they deserve investigation; if they drop out they
# were low-accuracy noise leaking through.

# %%
import ipywidgets as widgets
from IPython.display import display

_colors = {"bm": "#4C8BE2", "dn": "#E26B4C", "pb": "#4CE28B"}


def _make_timing_draw(pp_arr, auc_arr, ms_arr, phoneme_pairs, subject, out):
    def draw(auc_min):
        mask = auc_arr >= auc_min
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 3))
            for pp in phoneme_pairs:
                sel = mask & (pp_arr == pp)
                if sel.any():
                    ax.hist(
                        ms_arr[sel], bins=25, alpha=0.65,
                        label=f"{pp} (n={sel.sum()})", color=_colors.get(pp),
                    )
            ax.axvline(0, color="k", lw=1.0, ls="--", label="word onset")
            ax.set_xlabel("peak window onset (ms post word onset)")
            ax.set_ylabel("sites")
            ax.set_title(
                f"{subject} — peak timing  "
                f"({mask.sum()}/{len(mask)} sites,  AUC ≥ {auc_min:.2f})"
            )
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.show()
    return draw


for _subject in sorted(beh_peak_frames):
    _df = beh_peak_frames[_subject]
    _pp_arr  = _df["phoneme_pair"].to_numpy()
    _auc_arr = _df["peak_auc"].to_numpy()
    _ms_arr  = smin_to_ms(_df["peak_smin"].to_numpy())
    _pps     = sorted(_df["phoneme_pair"].unique().to_list())

    _slider = widgets.FloatSlider(
        value=0.5, min=0.5, max=round(float(_auc_arr.max()), 2), step=0.01,
        description="min AUC:",
        continuous_update=True,
        style={"description_width": "80px"},
        layout=widgets.Layout(width="500px"),
    )
    _out = widgets.Output()
    _draw = _make_timing_draw(_pp_arr, _auc_arr, _ms_arr, _pps, _subject, _out)

    _slider.observe(lambda change, d=_draw: d(change["new"]), names="value")
    _draw(0.5)  # initial render

    display(widgets.VBox([
        widgets.HTML(f"<b style='font-size:14px'>{_subject}</b>"),
        _slider,
        _out,
    ]))

# %% [markdown]
# ## 5  Behavior HGA-only — fold t-stats for brain plotting
#
# Without permutation results, we can still rank sites by the CV fold t-statistic:
#
#   **t = (mean\_fold\_AUC − 0.5) / (std\_fold\_AUC / √n\_folds)**
#
# This is a one-sample t-test of the fold distribution against chance.  It
# accounts for subject/ROI SNR differences: noisy folds inflate the denominator,
# so a consistent AUC=0.55 can outscore a noisy AUC=0.62.
# With typical 5-fold CV: df=4, t>2.13 ≈ p<0.05 (uncorrected, single site).
#
# **Output**: `outputs/causal6/brain_plot_behav_tstats.parquet`
# Columns: subject, electrode\_idx, phoneme\_pair, word\_end,
#          peak\_smin, peak\_auc, fold\_tstat, n\_folds, x, y, z, roi

# %%
brain_frames: list[pl.DataFrame] = []

for _p in sorted(ROOT.glob("behavior_decoding_single_electrode_hga_only/*/scores.parquet")):
    _subject = _p.parent.name
    _df = (
        pl.read_parquet(_p)
        .filter(pl.col("model") == "full")
        .with_columns(
            pl.col("word_end")
            .replace_strict(_OFFSET_SAMPLES, default=None)
            .alias("_smax_limit")
        )
        .filter(_filter_window_expr())
        .drop("_smax_limit")
    )

    # Per-(site, window): fold statistics
    _win_stats = (
        _df.group_by(WINDOW_KEYS_BEHAV)
        .agg(
            pl.col("test_roc_auc").mean().alias("fold_mean"),
            pl.col("test_roc_auc").std().alias("fold_std"),
            pl.col("test_roc_auc").len().alias("n_folds"),
        )
    )

    # Peak window per site = argmax of fold_mean; collect fold stats at that window
    _peak = (
        _win_stats.group_by(SITE_KEYS_BEHAV)
        .agg(
            pl.col("fold_mean").max().alias("peak_auc"),
            pl.col("smin").get(pl.col("fold_mean").arg_max()).alias("peak_smin"),
            pl.col("smax").get(pl.col("fold_mean").arg_max()).alias("peak_smax"),
            pl.col("fold_std").get(pl.col("fold_mean").arg_max()).alias("peak_fold_std"),
            pl.col("n_folds").get(pl.col("fold_mean").arg_max()).alias("n_folds"),
        )
        .with_columns(
            (
                (pl.col("peak_auc") - 0.5)
                / (
                    pl.max_horizontal(pl.col("peak_fold_std"), pl.lit(1e-6))
                    / pl.col("n_folds").cast(pl.Float64).sqrt()
                )
            ).alias("fold_tstat")
        )
    )

    # Merge with MNI electrode positions (warped coords)
    _elec_df = get_electrode_df(_subject)
    _elec_tmp = _elec_df.reset_index()[["electrode_idx", "x", "y", "z", "roi"]]
    _elec_tmp["roi"] = _elec_tmp["roi"].astype(str)
    _elec_pl = pl.from_pandas(_elec_tmp)
    _peak_pos = _peak.join(_elec_pl, on="electrode_idx", how="left")

    brain_frames.append(_peak_pos.with_columns(pl.lit(_subject).alias("subject")))

    _n = len(_peak)
    _t = _peak["fold_tstat"].to_numpy()
    print(
        f"{_subject}: {_n} sites  |  "
        f"fold_tstat  median={np.median(_t):.2f}  "
        f"max={_t.max():.2f}  "
        f">2.13 (≈p<0.05, df=4): {int((_t > 2.13).sum())}/{_n}"
    )
    for _pp in sorted(_peak["phoneme_pair"].unique().to_list()):
        _sub = _peak.filter(pl.col("phoneme_pair") == _pp)
        _st = _sub["fold_tstat"].to_numpy()
        print(
            f"  {_pp} ({len(_sub)} sites):  "
            f"t median={np.median(_st):.2f}  max={_st.max():.2f}  "
            f"AUC median={_sub['peak_auc'].median():.3f}"
        )

# %%
if brain_frames:
    brain_df = pl.concat(brain_frames)
    _brain_out = ROOT / "brain_plot_behav_tstats.parquet"
    brain_df.write_parquet(_brain_out)
    print(f"Written: {_brain_out}  ({len(brain_df)} rows × {brain_df.width} cols)")
    print(brain_df.schema)

# %%
# t-stat distribution across all sites/subjects
if brain_frames:
    _t_all = brain_df["fold_tstat"].to_numpy()
    _auc_all = brain_df["peak_auc"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 3))

    axes[0].hist(_t_all[np.isfinite(_t_all)], bins=40, color="steelblue", alpha=0.8)
    for _thresh, _ls in [(2.13, "--"), (3.0, ":")]:
        axes[0].axvline(_thresh, color="tomato", lw=1.2, ls=_ls,
                        label=f"t={_thresh}  n={int((_t_all > _thresh).sum())}")
    axes[0].set_xlabel("fold t-stat")
    axes[0].set_ylabel("sites")
    axes[0].set_title("All subjects — fold t-stat distribution")
    axes[0].legend(fontsize=8)

    axes[1].scatter(_auc_all, _t_all, s=6, alpha=0.4, color="steelblue")
    axes[1].axhline(2.13, color="tomato", lw=0.8, ls="--", label="t=2.13")
    axes[1].axvline(0.6, color="k", lw=0.8, ls=":", label="AUC=0.60")
    axes[1].set_xlabel("peak fold-mean AUC")
    axes[1].set_ylabel("fold t-stat")
    axes[1].set_title("AUC vs t-stat (thresholds are not equivalent)")
    axes[1].legend(fontsize=8)

    # Threshold sweep: how many sites survive each t threshold?
    _thresholds = np.arange(0, 5.05, 0.25)
    _n_survive = [int((_t_all > th).sum()) for th in _thresholds]
    axes[2].plot(_thresholds, _n_survive, "o-", color="steelblue", ms=4)
    axes[2].axvline(2.13, color="tomato", lw=0.8, ls="--", label="t=2.13 (≈p<0.05, df=4)")
    axes[2].set_xlabel("fold t-stat threshold")
    axes[2].set_ylabel("sites surviving")
    axes[2].set_title("Threshold sweep")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 6  Behavior HGA-only — on-the-fly significance
#
# Runs `aggregate_behavior_hga_only` + `null_standardized_peak_test` for every
# subject that has both `scores.parquet` and `null_scores.parquet`.
#
# **No BH-FDR applied** — partial data.  Once the pipeline finishes, run:
# ```
# uv run python scripts/aggregate_partial.py behav_hga_only
# ```
# to get FDR-corrected aggregates.

# %%
# Null aggregation uses scan_parquet + streaming collect so the raw null frame
# (potentially 100s of GB) never fully materialises in RAM.  The fold-collapse
# happens chunk-by-chunk and only the per-(site, window, perm) means land in memory.
import gc

from src.models.significance import fold_tstat_aggregate


null_dir = ROOT / "behavior_decoding_single_electrode_hga_only_null"
score_dir = ROOT / "behavior_decoding_single_electrode_hga_only"

beh_sig_frames: dict[str, pl.DataFrame] = {}

for null_path in sorted(null_dir.glob("*/null_scores.parquet")):
    subject = null_path.parent.name
    scores_path = score_dir / subject / "scores.parquet"
    if not scores_path.exists():
        print(f"{subject}: null exists but no real scores — skipping.")
        continue

    print(f"{subject}: aggregating …", end=" ", flush=True)

    # Real scores: small, load eagerly
    real_agg = fold_tstat_aggregate(
        pl.read_parquet(scores_path)
        .with_columns(
            pl.col("word_end")
            .replace_strict(_OFFSET_SAMPLES, default=None)
            .alias("_smax_limit")
        )
        .filter(_filter_window_expr())
        .drop("_smax_limit"),
        group_keys=WINDOW_KEYS_BEHAV,
        stat_col="test_roc_auc",
        center=0.5,
    )

    # Null scores: scan lazily, stream the fold-collapse to avoid materialising
    # the full file (can be 100+ GB for large subjects with many permutations).
    null_agg = (
        pl.scan_parquet(null_path)
        .with_columns(
            pl.col("word_end")
            .replace_strict(_OFFSET_SAMPLES, default=None)
            .alias("_smax_limit")
        )
        .filter(_filter_window_expr())
        .drop("_smax_limit")
        .group_by(WINDOW_KEYS_BEHAV + ["permutation_idx"])
        .agg(
            pl.col("test_roc_auc").mean().alias("fold_mean"),
            pl.col("test_roc_auc").std().alias("fold_std"),
            pl.col("test_roc_auc").len().alias("n_folds"),
        )
        .with_columns(
            (
                (pl.col("fold_mean") - 0.5)
                / (
                    pl.max_horizontal(pl.col("fold_std"), pl.lit(0.01))
                    / pl.col("n_folds").cast(pl.Float64).sqrt()
                )
            ).alias("t_stat")
        )
        .collect(streaming=True)
    )

    peaks, _ = null_standardized_peak_test(
        real_agg, null_agg,
        site_keys=SITE_KEYS_BEHAV,
        window_keys=["smin", "smax"],
        stat_col="fold_mean",
    )
    del null_agg
    gc.collect()

    beh_sig_frames[subject] = peaks
    n_total = len(peaks)
    n_sig = int((peaks["p_value"] < 0.05).sum())
    n_perm = int(peaks["n_permutations"].max())
    min_p = 1.0 / (n_perm + 1)
    print(
        f"{n_perm} perms (min p={min_p:.4f})  |  "
        f"significant (p<0.05): {n_sig}/{n_total}"
    )
    for pp in sorted(peaks["phoneme_pair"].unique().to_list()):
        sub = peaks.filter(pl.col("phoneme_pair") == pp)
        print(
            f"  {pp}:  {int((sub['p_value'] < 0.05).sum())}/{len(sub)} sig  "
            f"  peak fold-mean median={sub['real_statistic'].median():.3f}  "
            f"max={sub['real_statistic'].max():.3f}"
        )

# %%
if beh_sig_frames:
    subjects = sorted(beh_sig_frames)

    fig, axes = plt.subplots(
        len(subjects), 3, figsize=(15, 4 * len(subjects)), squeeze=False
    )

    for row, subject in enumerate(subjects):
        peaks = beh_sig_frames[subject]
        n_perm = int(peaks["n_permutations"].max())
        min_p = 1.0 / (n_perm + 1)

        aucs = peaks["real_statistic"].to_numpy()
        peak_ms = smin_to_ms(peaks["peak_smin"].to_numpy())
        pv = peaks["p_value"].to_numpy()
        sig_mask = pv < 0.05

        # AUC distribution coloured by significance
        ax = axes[row, 0]
        ax.hist(aucs[~sig_mask], bins=20, color="steelblue", alpha=0.7, label="n.s.")
        ax.hist(aucs[sig_mask], bins=20, color="tomato", alpha=0.8, label="p<0.05")
        ax.axvline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xlabel("peak fold-mean ROC-AUC")
        ax.set_ylabel("sites")
        ax.set_title(
            f"{subject} — behavior peak AUC\n({n_perm} perms, min p={min_p:.4f})"
        )
        ax.legend(fontsize=8)

        # Peak timing coloured by significance
        ax2 = axes[row, 1]
        ax2.hist(peak_ms[~sig_mask], bins=20, color="steelblue", alpha=0.7, label="n.s.")
        ax2.hist(peak_ms[sig_mask], bins=20, color="tomato", alpha=0.8, label="p<0.05")
        ax2.axvline(0, color="k", lw=0.8, ls="--", label="word onset")
        ax2.set_xlabel("peak window onset (ms post word onset)")
        ax2.set_ylabel("sites")
        ax2.set_title(f"{subject} — behavior peak timing")
        ax2.legend(fontsize=8)

        # p-value histogram
        ax3 = axes[row, 2]
        ax3.hist(pv[np.isfinite(pv)], bins=20, color="slategray", alpha=0.8)
        ax3.axvline(0.05, color="tomato", lw=1.2, ls="--", label="p=0.05")
        ax3.axvline(min_p, color="goldenrod", lw=1.2, ls=":", label=f"min achievable ({min_p:.3f})")
        ax3.set_xlabel("p-value (maxstat-corrected, uncorrected for FDR)")
        ax3.set_ylabel("sites")
        ax3.set_title(f"{subject} — p-value distribution")
        ax3.legend(fontsize=8)

    fig.tight_layout()
    plt.show()

# %% [markdown]
# ----
# ## 7  Contingency + provisional star plots — sites passing both thresholds
#
# Cross-filter `brain_plot_acoustic_tstats.parquet` (section 2) and
# `brain_plot_behav_tstats.parquet` (section 5) by AUC thresholds, identify
# electrodes that pass **both** filters, and render two-panel HGA traces.
#
# **Requires sections 2 and 5 to have been run** (parquets must exist on disk).
#
# Star plot panels:
#   - Top: unambiguous trials (resampled 1 & 6), acoustic peak window shaded.
#   - Bottom: within-completion controlled ambiguous trials (behaviorally-defined
#     ambiguous steps via `get_ambiguous_resampled_steps`), behavioral peak
#     window shaded, split by button-press response.
#
# No polarity correction is applied (requires `prepare_neurometrics`).
#
# **Output**: `outputs/causal6/provisional_star_plots.pdf`

# %%
import mne as _mne
import pandas as pd
import re as _re_s7
from matplotlib.backends.backend_pdf import PdfPages
from src.data import add_metadata_features, get_ambiguous_resampled_steps as _get_ambig_steps
from src.viz_paper import add_textgrid

_TEXTGRID_DIR = "data/stimuli/textgrid"

# %%
# Load epochs once per session — guard to avoid reloading on re-run.
if "epochs_dict" not in dir():
    _epo_paths = sorted(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))
    epochs_dict: dict = {}
    for _p in _epo_paths:
        _m = _re_s7.search(r"(EC\d+)_epo", str(_p))
        if not _m:
            continue
        _subj = _m.group(1)
        _ep = _mne.read_epochs(str(_p), preload=False, verbose="WARNING")
        _ep.metadata = add_metadata_features(_ep.metadata.copy())
        epochs_dict[_subj] = _ep
    print(f"Loaded epochs for: {sorted(epochs_dict)}")

# %%
acoustic_auc_threshold = 0.75
behav_auc_threshold = 0.8

_ac_path = ROOT / "brain_plot_acoustic_tstats.parquet"
_beh_path = ROOT / "brain_plot_behav_tstats.parquet"
if not (_ac_path.exists() and _beh_path.exists()):
    raise RuntimeError(
        "brain_plot parquets not found — run sections 2 and 5 first."
    )

_ac_pd = pl.read_parquet(_ac_path).to_pandas()
_beh_pd = pl.read_parquet(_beh_path).to_pandas()

acoustic_passes = _ac_pd[_ac_pd.peak_auc >= acoustic_auc_threshold].copy()
behav_passes    = _beh_pd[_beh_pd.peak_auc >= behav_auc_threshold].copy()

behav_plot = (
    behav_passes
    .sort_values("peak_auc", ascending=False)
    .groupby(["subject", "electrode_idx"], observed=True, as_index=False)
    .first()
)
A_phonetic_plot = (
    acoustic_passes
    .sort_values("peak_auc", ascending=False)
    .groupby(["subject", "electrode_idx"], observed=True, as_index=False)
    .first()
)

contingency_df = pd.merge(
    A_phonetic_plot.drop_duplicates(["subject", "electrode_idx"]),
    behav_plot.drop_duplicates(["subject", "electrode_idx"]),
    on=["subject", "electrode_idx"],
    how="outer",
    suffixes=("_phon", "_behav"),
)
contingency_df["outcome"] = "all"
contingency_df.loc[contingency_df["peak_auc_behav"].isna(), "outcome"] = "phonetic"
contingency_df.loc[contingency_df["peak_auc_phon"].isna(),  "outcome"] = "behav"
contingency_df = (
    contingency_df
    .sort_values(["subject", "electrode_idx", "outcome"])
    .drop_duplicates(["subject", "electrode_idx"])
    .reset_index(drop=True)
)

print(contingency_df.outcome.value_counts(normalize=False))
print(contingency_df.outcome.value_counts(normalize=True).round(3))

_both = contingency_df[contingency_df.outcome == "all"].copy()
if len(_both):
    _both["same_phoneme_pair"] = _both["phoneme_pair_phon"] == _both["phoneme_pair_behav"]
    print(f"\n{len(_both)} sites with BOTH responses:")
    print(f"  same phoneme pair (acoustic vs behavioral): "
          f"{_both['same_phoneme_pair'].sum()}/{len(_both)}")
    print(_both[["subject", "electrode_idx",
                 "phoneme_pair_phon", "phoneme_pair_behav", "word_end_behav",
                 "peak_auc_phon", "peak_auc_behav",
                 "peak_smin_phon", "peak_smin_behav"]].to_string(index=False))

# %%
# Build behaviorally-defined ambiguous steps dict.
# get_ambiguous_resampled_steps expects behavior_dummy_forced; alias if absent.
_all_md_frames = []
for _s, _ep in epochs_dict.items():
    _md = _ep.metadata.copy()
    _md["subject"] = _s
    _all_md_frames.append(_md)

_all_md_pd = pd.concat(_all_md_frames, ignore_index=True)
if "behavior_dummy_forced" not in _all_md_pd.columns:
    _all_md_pd = _all_md_pd.rename(
        columns={"behavior_categorical": "behavior_dummy_forced"}
    )

ambig_steps = _get_ambig_steps(
    pl.from_pandas(
        _all_md_pd[["subject", "phoneme_pair", "word_end",
                    "resampled", "behavior_dummy_forced"]]
    ),
    ambiguous_response_threshold=2,
)
print(f"ambig_steps: {len(ambig_steps)} (subject, phoneme_pair, word_end) keys")

# %%
def _provisional_star_plot(
    subject,
    electrode_idx,
    phoneme_pair,
    word_end,
    epochs_dict,
    phon_smin,
    phon_smax,
    behav_smin,
    behav_smax,
    textgrid_dir,
    ambig_steps,
    epoch_tmin=EPOCH_TMIN,
    epoch_sfreq=EPOCH_SFREQ,
    figsize=(6.5, 5.0),
):
    """Two-panel provisional HGA star plot (no prepare_neurometrics required).

    Top: unambiguous trials (resampled 1 & 6), acoustic peak window shaded.
    Bottom: within-completion controlled ambiguous trials (behaviorally-defined
    steps from ambig_steps), behavioral peak window shaded, split by response.
    """
    ep = epochs_dict[subject]
    times = ep.times
    md = ep.metadata

    pp_mask = md["phoneme_pair"] == phoneme_pair
    ep_pp = ep[pp_mask.values]
    md_pp = md[pp_mask].reset_index(drop=True)

    hga = (
        ep_pp.copy()
        .apply_baseline((None, 0))
        .get_data(picks=[electrode_idx])
        .squeeze(1)
    )

    t_phon  = np.array([phon_smin,  phon_smax])  / epoch_sfreq + epoch_tmin
    t_behav = np.array([behav_smin, behav_smax]) / epoch_sfreq + epoch_tmin

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ── Top: unambiguous ───────────────────────────────────────────────
    _step_colors = {1: "#2166ac", 6: "#d73027"}
    for step, color in _step_colors.items():
        mask = md_pp["resampled"] == step
        if not mask.any():
            continue
        tr = hga[mask.values]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        ax_top.plot(times, m, color=color, lw=1.5, label=f"step {step}  (n={mask.sum()})")
        ax_top.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    ax_top.axvspan(*t_phon, color="#4dac26", alpha=0.14, label="acoustic window")
    ax_top.axhline(0, color="k", lw=0.5, ls=":")
    ax_top.set_ylabel("HGA (z)")
    ax_top.set_title(
        f"{subject}  e{electrode_idx}  {phoneme_pair} — unambiguous", fontsize=9
    )
    ax_top.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # ── Bottom: controlled ambiguous (within-completion) ───────────────
    amb = ambig_steps.get((subject, phoneme_pair, word_end), [3, 4])
    we_amb_mask = (md_pp["word_end"] == word_end) & md_pp["resampled"].isin(amb)

    _bhv_col = (
        "behavior_dummy_forced"
        if "behavior_dummy_forced" in md_pp.columns
        else "behavior_categorical"
    )
    _bhv_colors = ["#762a83", "#1b7837"]
    for _i, _bhv_val in enumerate(
        sorted(md_pp.loc[we_amb_mask, _bhv_col].dropna().unique())
    ):
        mask = we_amb_mask & (md_pp[_bhv_col] == _bhv_val)
        if not mask.any():
            continue
        tr = hga[mask.values]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        color = _bhv_colors[_i % len(_bhv_colors)]
        ax_bot.plot(times, m, color=color, lw=1.5,
                    label=f"resp={_bhv_val}  (n={mask.sum()})")
        ax_bot.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    ax_bot.axvspan(*t_behav, color="#f4a582", alpha=0.25, label="behavioral window")
    ax_bot.axhline(0, color="k", lw=0.5, ls=":")
    ax_bot.set_ylabel("HGA (z)")
    ax_bot.set_xlabel("Time (s, post word onset)")
    ax_bot.set_title(
        f"Controlled ambiguous — {word_end}  (steps {amb})", fontsize=9
    )
    ax_bot.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # ── TextGrid ────────────────────────────────────────────────────────
    for _ax in (ax_top, ax_bot):
        try:
            add_textgrid(_ax, textgrid_dir=textgrid_dir,
                         textgrid_file=f"11_{word_end}_dn_002.TextGrid",
                         vline_extent=1.0)
        except Exception:
            pass

    fig.tight_layout()
    return fig

# %%
_star_out = ROOT / "provisional_star_plots.pdf"
if len(_both) == 0:
    print("No sites pass both thresholds — no star plots to render.")
else:
    _missing = [s for s in _both["subject"].unique() if s not in epochs_dict]
    if _missing:
        print(f"Warning: epochs not loaded for {_missing} — those sites will be skipped.")

    with PdfPages(_star_out) as _pdf:
        for _, _row in _both.iterrows():
            if _row["subject"] not in epochs_dict:
                continue
            try:
                _fig = _provisional_star_plot(
                    subject=_row["subject"],
                    electrode_idx=int(_row["electrode_idx"]),
                    phoneme_pair=_row["phoneme_pair_behav"],
                    word_end=_row["word_end_behav"],
                    epochs_dict=epochs_dict,
                    phon_smin=int(_row["peak_smin_phon"]),
                    phon_smax=int(_row["peak_smax_phon"]),
                    behav_smin=int(_row["peak_smin_behav"]),
                    behav_smax=int(_row["peak_smax_behav"]),
                    textgrid_dir=_TEXTGRID_DIR,
                    ambig_steps=ambig_steps,
                )
                _pdf.savefig(_fig)
                plt.close(_fig)
            except Exception as _e:
                print(f"  skipped {_row['subject']} e{int(_row['electrode_idx'])}: {_e}")

    print(f"Written {len(_both)} pages → {_star_out}")
