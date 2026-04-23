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
# # Cross-check decoding results: causal4 / causal5 / causal6
#
# Answers two questions, per decoder type:
#   1. How much do the per-electrode decoding results agree across pipelines?
#   2. For sites where they disagree, what explains the drift?
#
# Coverage rule:
#   - Acoustic                — causal4 vs causal5 vs causal6
#   - Behavior-with-control   — causal4 vs causal5 vs causal6
#   - Behavior-HGA-only       — causal5 vs causal6 only (causal4 never ran this)
#
# Comparison axes per decoder: per-site peak ROC-AUC, peak-window (smin, smax),
# and the full searchlight AUC map. Debug section shows top-disagreement sites
# and side-by-side searchlight heatmaps.
#
# The notebook loads each pipeline from its `outputs/<pipeline>` root so it runs
# wherever the data live (HPC, etc.). Missing roots degrade gracefully — loaders
# return empty frames and the comparison code skips empty pairs.

# %%
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# %% tags=["parameters"]
causal4_root = "outputs/causal4"
causal5_root = "outputs/causal5"
causal6_root = "outputs/causal6"

subjects = [
    "EC183", "EC195", "EC212", "EC235", "EC237",
    "EC243", "EC260", "EC266", "EC282", "EC296",
]

top_n_disagreements = 20
behav_peak_post_offset_s = 0.2
epoch_tmin = -0.4
epoch_sfreq = 100

# %%
ROOTS = {
    "causal4": Path(causal4_root),
    "causal5": Path(causal5_root),
    "causal6": Path(causal6_root),
}

for name, root in ROOTS.items():
    print(f"{name}: {root}  (exists={root.exists()})")

# %% [markdown]
# ## 1. Loaders
#
# Each loader returns a normalized `pl.DataFrame` with a consistent schema so
# downstream comparison code is pipeline-agnostic.
#
# **Acoustic searchlight**: `subject, electrode_idx, phoneme_pair, smin, smax, fold, roc_auc`
# **Behavior-with-control searchlight**: adds `word_end`; columns `roc_auc_full, roc_auc_baseline, roc_auc_improvement`
# **Behavior-HGA-only searchlight**: adds `word_end`; column `roc_auc`
#
# For each pipeline we pick the cheapest source of per-fold AUC:
# - causal5 / causal6 have cached fold-level parquets — read directly.
# - causal4 requires deriving AUC from trial-level `all_outcomes.parquet` /
#   `A-predictions.parquet` via sklearn.

# %%
def _compute_roc_auc_per_group(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    proba_col: str,
    roc_auc_name: str,
) -> pl.DataFrame:
    """Compute ROC-AUC per group from trial-level predictions.

    Rows where a group has only one class get NaN (undefined AUC).
    """
    records = []
    for keys, sub in df.groupby(group_cols, sort=False):
        y = sub[target_col].to_numpy()
        if np.unique(y).size < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y, sub[proba_col].to_numpy())
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row[roc_auc_name] = auc
        records.append(row)
    return pl.from_pandas(pd.DataFrame(records))


def _empty(schema: list[tuple[str, pl.DataType]]) -> pl.DataFrame:
    return pl.DataFrame(schema={k: v for k, v in schema})


_UTF8_COLS = ("subject", "phoneme_pair", "word_end")


def _normalize_types(df: pl.DataFrame) -> pl.DataFrame:
    """Cast Enum/Categorical string-like columns to Utf8 so polars joins across
    pipelines don't fail on schema mismatch (causal5 writes Enum-typed columns
    via `viz_paper` while causal4 and causal6 use plain strings)."""
    if df.is_empty():
        return df
    casts = [pl.col(c).cast(pl.Utf8) for c in _UTF8_COLS if c in df.columns]
    return df.with_columns(casts) if casts else df


# %% [markdown]
# ### Acoustic searchlight loaders

# %%
def load_acoustic_searchlight_causal4(root: Path, subjects: list[str]) -> pl.DataFrame:
    """Derive per-fold AUC from causal4's trial-level `all_outcomes.parquet`."""
    frames = []
    for subj in subjects:
        p = root / "behavior_decoding_single_electrode_acoustic" / subj / "all_outcomes.parquet"
        if not p.exists():
            continue
        pdf = pd.read_parquet(p)
        if len(pdf) == 0:
            continue
        # binary target encoding (causal5 uses `== 1`); match that
        pdf["decoder_target"] = (pdf["decoder_target"] == 1).astype(np.int8)
        auc = _compute_roc_auc_per_group(
            pdf,
            group_cols=["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold"],
            target_col="decoder_target",
            proba_col="decoder_proba",
            roc_auc_name="roc_auc",
        )
        frames.append(auc)
    if not frames:
        return _empty([
            ("subject", pl.Utf8), ("electrode_idx", pl.Int64), ("phoneme_pair", pl.Utf8),
            ("smin", pl.Int64), ("smax", pl.Int64), ("fold", pl.Int64), ("roc_auc", pl.Float64),
        ])
    return pl.concat(frames)


def load_acoustic_searchlight_causal5(root: Path) -> pl.DataFrame:
    """causal5 acoustic fold-level AUC — prewritten by acoustic_decoding_peaks."""
    p = root / "acoustic_decoding_peaks" / "phon_roc_auc_searchlight_df.parquet"
    if not p.exists():
        return _empty([("subject", pl.Utf8)])
    return (
        pl.read_parquet(p)
        .rename({"phon_roc_auc": "roc_auc"})
        .select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold", "roc_auc"])
    )


def load_acoustic_searchlight_causal6(root: Path, subjects: list[str]) -> pl.DataFrame:
    """causal6 acoustic fold-level AUC — `scores.parquet` already stores it."""
    frames = []
    for subj in subjects:
        p = root / "acoustic_decoding_single_electrode" / subj / "scores.parquet"
        if not p.exists():
            continue
        df = (
            pl.read_parquet(p)
            .filter(pl.col("target") == "categorical_acoustic_cue")
            .rename({"test_roc_auc": "roc_auc"})
            .select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold", "roc_auc"])
        )
        frames.append(df)
    if not frames:
        return _empty([("subject", pl.Utf8)])
    return pl.concat(frames)


# %% [markdown]
# ### Behavior-with-control searchlight loaders

# %%
def load_behavior_ctrl_searchlight_from_predictions(
    paths: list[Path],
) -> pl.DataFrame:
    """Derive per-fold baseline/full AUC from trial-level `A-predictions.parquet`.

    Used for causal4 (no cached searchlight) and as a fallback for causal5.
    Concatenates late and early prediction files if both exist.
    """
    frames = []
    for p in paths:
        if not p.exists():
            continue
        pdf = pd.read_parquet(p)
        if len(pdf) == 0:
            continue
        pdf["decoder_target"] = pdf["decoder_target"].astype(np.int8)
        group_cols = [
            "subject", "electrode_idx", "phoneme_pair", "word_end",
            "smin", "smax", "fold",
        ]
        full = _compute_roc_auc_per_group(
            pdf, group_cols, "decoder_target", "full_decoder_proba", "roc_auc_full"
        )
        base = _compute_roc_auc_per_group(
            pdf, group_cols, "decoder_target", "baseline_decoder_proba", "roc_auc_baseline"
        )
        frames.append(full.join(base, on=group_cols, how="left"))
    if not frames:
        return _empty([("subject", pl.Utf8)])
    return pl.concat(frames).with_columns(
        (pl.col("roc_auc_full") - pl.col("roc_auc_baseline")).alias("roc_auc_improvement")
    )


def load_behavior_ctrl_searchlight_causal4(root: Path, subjects: list[str]) -> pl.DataFrame:
    paths = []
    for subj in subjects:
        base = root / "behavior_decoding_single_electrode_summarize" / subj
        paths += [base / "A-predictions.parquet", base / "A_early-predictions.parquet"]
    return load_behavior_ctrl_searchlight_from_predictions(paths)


def load_behavior_ctrl_searchlight_causal5(root: Path, subjects: list[str]) -> pl.DataFrame:
    """Prefer the cached fold-level parquet from prepare_neurometrics."""
    cached = root / "prepare_neurometrics" / "behav_roc_auc_searchlight_df.parquet"
    if cached.exists():
        df = pl.read_parquet(cached)
        # Column names differ: behav_roc_auc (full), behav_roc_auc_baseline
        renames = {}
        if "behav_roc_auc" in df.columns:
            renames["behav_roc_auc"] = "roc_auc_full"
        if "behav_roc_auc_baseline" in df.columns:
            renames["behav_roc_auc_baseline"] = "roc_auc_baseline"
        if "behav_roc_auc_improvement" in df.columns:
            renames["behav_roc_auc_improvement"] = "roc_auc_improvement"
        df = df.rename(renames)
        if "roc_auc_improvement" not in df.columns and {"roc_auc_full", "roc_auc_baseline"} <= set(df.columns):
            df = df.with_columns(
                (pl.col("roc_auc_full") - pl.col("roc_auc_baseline")).alias("roc_auc_improvement")
            )
        keep = [
            "subject", "electrode_idx", "phoneme_pair", "word_end",
            "smin", "smax", "fold",
            "roc_auc_full", "roc_auc_baseline", "roc_auc_improvement",
        ]
        return df.select([c for c in keep if c in df.columns])
    # Fallback: derive from trial-level predictions
    paths = []
    for subj in subjects:
        base = root / "behavior_decoding_single_electrode_summarize" / subj
        paths += [base / "A-predictions.parquet", base / "A_early-predictions.parquet"]
    return load_behavior_ctrl_searchlight_from_predictions(paths)


def load_behavior_ctrl_searchlight_causal6(root: Path, subjects: list[str]) -> pl.DataFrame:
    frames = []
    for subj in subjects:
        p = root / "behavior_decoding_single_electrode" / subj / "scores.parquet"
        if not p.exists():
            continue
        scores = pl.read_parquet(p)
        full = (
            scores.filter(pl.col("model") == "full")
            .rename({"test_roc_auc": "roc_auc_full"})
            .drop("model")
        )
        base = (
            scores.filter(pl.col("model") == "baseline")
            .select(["subject", "phoneme_pair", "word_end", "fold", "test_roc_auc"])
            .rename({"test_roc_auc": "roc_auc_baseline"})
        )
        df = full.join(base, on=["subject", "phoneme_pair", "word_end", "fold"], how="left")
        df = df.with_columns(
            (pl.col("roc_auc_full") - pl.col("roc_auc_baseline")).alias("roc_auc_improvement")
        )
        keep = [
            "subject", "electrode_idx", "phoneme_pair", "word_end",
            "smin", "smax", "fold",
            "roc_auc_full", "roc_auc_baseline", "roc_auc_improvement",
        ]
        frames.append(df.select([c for c in keep if c in df.columns]))
    if not frames:
        return _empty([("subject", pl.Utf8)])
    return pl.concat(frames)


# %% [markdown]
# ### Behavior-HGA-only searchlight loaders

# %%
def load_behavior_hga_searchlight_causal5(root: Path, subjects: list[str]) -> pl.DataFrame:
    """causal5 HGA-only writes A_results.csv per subject with per-fold full_roc_auc.

    The CSV is written via pandas `to_csv` without `index=False`, so the first
    column is an unnamed pandas index — strip it via `index_col=0`.
    """
    frames = []
    for subj in subjects:
        p = root / "behavior_decoding_single_electrode_hga_only_summarize" / subj / "A_results.csv"
        if not p.exists():
            continue
        pdf = pd.read_csv(p, index_col=0)
        # causal5 HGA-only uses `population` as a stringified electrode_idx
        if "population" in pdf.columns and "electrode_idx" not in pdf.columns:
            pdf["electrode_idx"] = pdf["population"].astype(int)
        pdf = pdf.rename(columns={"full_roc_auc": "roc_auc"})
        keep = [
            "subject", "electrode_idx", "phoneme_pair", "word_end",
            "smin", "smax", "fold", "roc_auc",
        ]
        frames.append(pl.from_pandas(pdf[[c for c in keep if c in pdf.columns]]))
    if not frames:
        return _empty([("subject", pl.Utf8)])
    return pl.concat(frames)


def load_behavior_hga_searchlight_causal6(root: Path, subjects: list[str]) -> pl.DataFrame:
    frames = []
    for subj in subjects:
        p = root / "behavior_decoding_single_electrode_hga_only" / subj / "scores.parquet"
        if not p.exists():
            continue
        df = (
            pl.read_parquet(p)
            .rename({"test_roc_auc": "roc_auc"})
            .select([
                "subject", "electrode_idx", "phoneme_pair", "word_end",
                "smin", "smax", "fold", "roc_auc",
            ])
        )
        frames.append(df)
    if not frames:
        return _empty([("subject", pl.Utf8)])
    return pl.concat(frames)


# %% [markdown]
# ### Peak derivation — consistent across pipelines
#
# Rather than load each pipeline's own peak parquet (they apply pipeline-specific
# thresholds and filters), we derive peaks from the normalized searchlight using
# a single policy: fold-mean, then argmax of the per-decoder-type criterion.

# %%
def derive_peaks(searchlight: pl.DataFrame, criterion: str, site_cols: list[str]) -> pl.DataFrame:
    """Fold-mean, then argmax `criterion` per site."""
    if searchlight.is_empty():
        return searchlight.clone()
    window_cols = site_cols + ["smin", "smax"]
    # fold-mean across all numeric columns that look like AUC metrics
    agg_cols = [c for c in searchlight.columns if c.startswith("roc_auc")]
    fold_mean = searchlight.group_by(window_cols).agg([pl.col(c).mean() for c in agg_cols])
    peaks = (
        fold_mean.sort(criterion, descending=True, nulls_last=True)
        .group_by(site_cols, maintain_order=True)
        .agg(pl.all().first())
    )
    return peaks


SITE_COLS = {
    "acoustic": ["subject", "electrode_idx", "phoneme_pair"],
    "behavior_ctrl": ["subject", "electrode_idx", "phoneme_pair", "word_end"],
    "behavior_hga": ["subject", "electrode_idx", "phoneme_pair", "word_end"],
}
PEAK_CRITERION = {
    "acoustic": "roc_auc",
    "behavior_ctrl": "roc_auc_improvement",
    "behavior_hga": "roc_auc",
}

# %% [markdown]
# ### Build all tables

# %%
ACOUSTIC = {
    "causal4": _normalize_types(load_acoustic_searchlight_causal4(ROOTS["causal4"], subjects)),
    "causal5": _normalize_types(load_acoustic_searchlight_causal5(ROOTS["causal5"])),
    "causal6": _normalize_types(load_acoustic_searchlight_causal6(ROOTS["causal6"], subjects)),
}
BEHAV_CTRL = {
    "causal4": _normalize_types(load_behavior_ctrl_searchlight_causal4(ROOTS["causal4"], subjects)),
    "causal5": _normalize_types(load_behavior_ctrl_searchlight_causal5(ROOTS["causal5"], subjects)),
    "causal6": _normalize_types(load_behavior_ctrl_searchlight_causal6(ROOTS["causal6"], subjects)),
}
BEHAV_HGA = {
    "causal5": _normalize_types(load_behavior_hga_searchlight_causal5(ROOTS["causal5"], subjects)),
    "causal6": _normalize_types(load_behavior_hga_searchlight_causal6(ROOTS["causal6"], subjects)),
}

SEARCHLIGHTS = {"acoustic": ACOUSTIC, "behavior_ctrl": BEHAV_CTRL, "behavior_hga": BEHAV_HGA}
PEAKS = {
    kind: {p: derive_peaks(df, PEAK_CRITERION[kind], SITE_COLS[kind])
           for p, df in pipelines.items()}
    for kind, pipelines in SEARCHLIGHTS.items()
}

for kind in SEARCHLIGHTS:
    print(f"\n{kind}:")
    for p, df in SEARCHLIGHTS[kind].items():
        print(f"  {p}: {df.height} searchlight rows, {PEAKS[kind][p].height} sites")

# %% [markdown]
# ## 2. Coverage summary
#
# For each decoder, show per-pipeline site counts and the pairwise overlap.
# Useful before reading any scatter plot — if one pipeline covers many fewer
# sites than another (e.g., causal4's pre-selection vs causal5 dropping it),
# that alone drives apparent disagreement.

# %%
def site_coverage(peaks: dict[str, pl.DataFrame], site_cols: list[str]) -> pd.DataFrame:
    """Wide coverage table: rows = sites, columns = pipelines, values = present."""
    per_pipe = []
    for pipe, df in peaks.items():
        if df.is_empty():
            continue
        pdf = df.select(site_cols).unique().to_pandas()
        pdf[pipe] = True
        per_pipe.append(pdf)
    if not per_pipe:
        return pd.DataFrame()
    out = per_pipe[0]
    for other in per_pipe[1:]:
        out = out.merge(other, on=site_cols, how="outer")
    for pipe in peaks:
        if pipe in out.columns:
            out[pipe] = out[pipe].fillna(False).astype(bool)
        else:
            out[pipe] = False
    return out


for kind in SEARCHLIGHTS:
    cov = site_coverage(PEAKS[kind], SITE_COLS[kind])
    print(f"\n=== {kind} coverage ===")
    if cov.empty:
        print("  (no pipelines loaded)")
        continue
    pipes = [p for p in PEAKS[kind] if p in cov.columns]
    for pipe in pipes:
        print(f"  {pipe:>8}: {int(cov[pipe].sum())} sites")
    if len(pipes) >= 2:
        print("  pairwise intersections:")
        for a, b in combinations(pipes, 2):
            both = int((cov[a] & cov[b]).sum())
            only_a = int((cov[a] & ~cov[b]).sum())
            only_b = int((~cov[a] & cov[b]).sum())
            print(f"    {a} ∩ {b} = {both} | only {a} = {only_a} | only {b} = {only_b}")

# %% [markdown]
# ## 3. Per-decoder comparison
#
# For each of {acoustic, behavior-with-control, behavior-HGA-only}, produce the
# four agreement diagnostics plus a drill-in table and per-site searchlight
# heatmaps for the top-N disagreement sites.

# %%
def pair_peaks(
    peaks_a: pl.DataFrame, peaks_b: pl.DataFrame,
    site_cols: list[str], criterion: str,
) -> pd.DataFrame:
    """Inner-join two pipelines' peak tables; return suffixed pandas frame."""
    if peaks_a.is_empty() or peaks_b.is_empty():
        return pd.DataFrame()
    a = peaks_a.select(site_cols + ["smin", "smax", criterion]).rename(
        {"smin": "smin_a", "smax": "smax_a", criterion: f"{criterion}_a"}
    )
    b = peaks_b.select(site_cols + ["smin", "smax", criterion]).rename(
        {"smin": "smin_b", "smax": "smax_b", criterion: f"{criterion}_b"}
    )
    return a.join(b, on=site_cols, how="inner").to_pandas()


def scatter_with_unity(ax, x, y, xlabel, ylabel, title):
    ax.scatter(x, y, s=10, alpha=0.5)
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.75, alpha=0.5)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() >= 3:
        r, _ = pearsonr(x[mask], y[mask])
        ax.set_title(f"{title}\nr={r:.3f}, n={int(mask.sum())}")
    else:
        ax.set_title(f"{title}\n(n={int(mask.sum())})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_peak_auc_scatter(peaks: dict[str, pl.DataFrame], site_cols: list[str], criterion: str, kind: str):
    pipes = [p for p, df in peaks.items() if not df.is_empty()]
    pairs = list(combinations(pipes, 2))
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        joined = pair_peaks(peaks[a], peaks[b], site_cols, criterion)
        if joined.empty:
            ax.set_title(f"{a} vs {b}: no shared sites")
            continue
        scatter_with_unity(
            ax,
            joined[f"{criterion}_a"].to_numpy(),
            joined[f"{criterion}_b"].to_numpy(),
            f"{a} {criterion}",
            f"{b} {criterion}",
            f"{kind} peak AUC",
        )
    fig.suptitle(f"{kind}: peak AUC agreement")
    fig.tight_layout()
    plt.show()


def plot_peak_window_scatter(peaks: dict[str, pl.DataFrame], site_cols: list[str], criterion: str, kind: str):
    pipes = [p for p, df in peaks.items() if not df.is_empty()]
    pairs = list(combinations(pipes, 2))
    if not pairs:
        return
    fig, axes = plt.subplots(2, len(pairs), figsize=(5 * len(pairs), 8), squeeze=False)
    for col, (a, b) in enumerate(pairs):
        joined = pair_peaks(peaks[a], peaks[b], site_cols, criterion)
        if joined.empty:
            for row in (0, 1):
                axes[row, col].set_title(f"{a} vs {b}: no shared sites")
            continue
        scatter_with_unity(
            axes[0, col], joined["smin_a"].to_numpy(), joined["smin_b"].to_numpy(),
            f"{a} smin", f"{b} smin", f"{kind} peak smin",
        )
        scatter_with_unity(
            axes[1, col], joined["smax_a"].to_numpy(), joined["smax_b"].to_numpy(),
            f"{a} smax", f"{b} smax", f"{kind} peak smax",
        )
    fig.suptitle(f"{kind}: peak window agreement")
    fig.tight_layout()
    plt.show()


def searchlight_per_site_correlation(
    searchlights: dict[str, pl.DataFrame], site_cols: list[str], criterion: str,
) -> dict[tuple[str, str], pd.DataFrame]:
    """For each pipeline pair, return a DataFrame of per-site Pearson r across windows."""
    out = {}
    pipes = [p for p, df in searchlights.items() if not df.is_empty()]
    for a, b in combinations(pipes, 2):
        fold_mean_a = searchlights[a].group_by(site_cols + ["smin", "smax"]).agg(pl.col(criterion).mean())
        fold_mean_b = searchlights[b].group_by(site_cols + ["smin", "smax"]).agg(pl.col(criterion).mean())
        joined = fold_mean_a.join(
            fold_mean_b, on=site_cols + ["smin", "smax"], how="inner", suffix="_b"
        ).rename({criterion: f"{criterion}_a"})
        if joined.is_empty():
            out[(a, b)] = pd.DataFrame(columns=site_cols + ["r", "n_windows"])
            continue
        pdf = joined.to_pandas()
        records = []
        for keys, sub in pdf.groupby(site_cols, sort=False):
            x = sub[f"{criterion}_a"].to_numpy()
            y = sub[f"{criterion}_b"].to_numpy()
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 3 and x[mask].std() > 0 and y[mask].std() > 0:
                r, _ = pearsonr(x[mask], y[mask])
            else:
                r = np.nan
            row = dict(zip(site_cols, keys if isinstance(keys, tuple) else (keys,)))
            row["r"] = r
            row["n_windows"] = int(mask.sum())
            records.append(row)
        out[(a, b)] = pd.DataFrame(records)
    return out


def plot_searchlight_correlation_histograms(corrs: dict[tuple[str, str], pd.DataFrame], kind: str):
    if not corrs:
        return
    fig, axes = plt.subplots(1, len(corrs), figsize=(5 * len(corrs), 4), squeeze=False)
    for ax, ((a, b), df) in zip(axes[0], corrs.items()):
        r = df["r"].dropna().to_numpy()
        ax.hist(r, bins=30, range=(-1, 1), edgecolor="black", alpha=0.7)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"{kind}: searchlight r  {a} vs {b}\nmedian={np.nanmedian(r):.3f}, n={len(r)}")
        ax.set_xlabel("per-site Pearson r across (smin,smax)")
        ax.set_ylabel("count")
    fig.tight_layout()
    plt.show()


def disagreement_table(
    peaks: dict[str, pl.DataFrame],
    corrs: dict[tuple[str, str], pd.DataFrame],
    site_cols: list[str],
    criterion: str,
    top_n: int,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Rank sites per pipeline pair by |Δpeak criterion|; attach searchlight r."""
    out = {}
    pipes = [p for p, df in peaks.items() if not df.is_empty()]
    for a, b in combinations(pipes, 2):
        joined = pair_peaks(peaks[a], peaks[b], site_cols, criterion)
        if joined.empty:
            out[(a, b)] = joined
            continue
        joined["delta"] = joined[f"{criterion}_a"] - joined[f"{criterion}_b"]
        joined["abs_delta"] = joined["delta"].abs()
        if (a, b) in corrs and not corrs[(a, b)].empty:
            joined = joined.merge(corrs[(a, b)][site_cols + ["r", "n_windows"]], on=site_cols, how="left")
        out[(a, b)] = joined.sort_values("abs_delta", ascending=False).head(top_n).reset_index(drop=True)
    return out


def plot_searchlight_heatmap(
    ax, searchlight: pl.DataFrame, site_values: dict, site_cols: list[str],
    criterion: str, title: str,
):
    """Plot a single site's fold-mean AUC as a heatmap in (smin, smax) space."""
    flt = searchlight
    for col, val in site_values.items():
        flt = flt.filter(pl.col(col) == val)
    if flt.is_empty():
        ax.set_title(f"{title}\n(no data)")
        ax.set_axis_off()
        return
    fold_mean = (
        flt.group_by(["smin", "smax"])
        .agg(pl.col(criterion).mean().alias("auc"))
        .to_pandas()
    )
    pivot = fold_mean.pivot(index="smax", columns="smin", values="auc").sort_index()
    pivot = pivot[sorted(pivot.columns)]
    im = ax.imshow(
        pivot.values, origin="lower", aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        vmin=0.4, vmax=1.0, cmap="viridis",
    )
    ax.set_xlabel("smin")
    ax.set_ylabel("smax")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)


def plot_top_disagreement_heatmaps(
    disagree: dict[tuple[str, str], pd.DataFrame],
    searchlights: dict[str, pl.DataFrame],
    site_cols: list[str],
    criterion: str,
    n_show: int = 5,
):
    for (a, b), df in disagree.items():
        if df.empty:
            continue
        for _, row in df.head(n_show).iterrows():
            site_values = {c: row[c] for c in site_cols}
            peak_a = (int(row["smin_a"]), int(row["smax_a"]), row[f"{criterion}_a"])
            peak_b = (int(row["smin_b"]), int(row["smax_b"]), row[f"{criterion}_b"])
            title = " / ".join(f"{c}={row[c]}" for c in site_cols)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            plot_searchlight_heatmap(
                axes[0], searchlights[a], site_values, site_cols, criterion,
                f"{a}\npeak=({peak_a[0]},{peak_a[1]}) auc={peak_a[2]:.3f}",
            )
            plot_searchlight_heatmap(
                axes[1], searchlights[b], site_values, site_cols, criterion,
                f"{b}\npeak=({peak_b[0]},{peak_b[1]}) auc={peak_b[2]:.3f}",
            )
            axes[0].plot(peak_a[0], peak_a[1], "r*", markersize=14)
            axes[1].plot(peak_b[0], peak_b[1], "r*", markersize=14)
            fig.suptitle(f"{a} vs {b} — {title}\nΔ={row['delta']:+.3f}  r={row.get('r', float('nan')):.3f}")
            fig.tight_layout()
            plt.show()


# %%
def run_comparison(kind: str, n_show_heatmaps: int = 5):
    print(f"\n====================  {kind}  ====================")
    site_cols = SITE_COLS[kind]
    criterion = PEAK_CRITERION[kind]
    searchlights = SEARCHLIGHTS[kind]
    peaks = PEAKS[kind]

    plot_peak_auc_scatter(peaks, site_cols, criterion, kind)
    plot_peak_window_scatter(peaks, site_cols, criterion, kind)

    corrs = searchlight_per_site_correlation(searchlights, site_cols, criterion)
    plot_searchlight_correlation_histograms(corrs, kind)

    disagree = disagreement_table(peaks, corrs, site_cols, criterion, top_n_disagreements)
    for (a, b), df in disagree.items():
        print(f"\nTop {len(df)} disagreements {a} vs {b}:")
        if df.empty:
            print("  (no shared sites)")
        else:
            display_cols = site_cols + [f"{criterion}_a", f"{criterion}_b", "delta",
                                        "smin_a", "smax_a", "smin_b", "smax_b", "r"]
            display_cols = [c for c in display_cols if c in df.columns]
            print(df[display_cols].to_string(index=False))

    plot_top_disagreement_heatmaps(disagree, searchlights, site_cols, criterion, n_show=n_show_heatmaps)


# %% [markdown]
# ### Acoustic

# %%
run_comparison("acoustic")

# %% [markdown]
# ### Behavior-with-control
#
# Peak window is chosen per-site by max `roc_auc_improvement = full − baseline`,
# matching the convention used inside causal5's `prepare_neurometrics` and
# causal6's `behavior_decoding_single_electrode_summarize`.

# %%
run_comparison("behavior_ctrl")

# %% [markdown]
# ### Behavior-HGA-only
#
# causal4 never ran this variant, so the comparison is causal5 vs causal6 only.

# %%
run_comparison("behavior_hga")

# %% [markdown]
# ## 4. Notes on expected differences
#
# Before chasing down every disagreement, keep in mind these structural
# differences between pipelines that will produce real, non-bug AUC drift:
#
# - **Window grids**: `min_sample`, `window_size`, and `stride` come from
#   `config.yaml` in each pipeline. If they differ, searchlight grids don't
#   line up and the inner-join in `searchlight_per_site_correlation`
#   silently drops non-shared windows. Check with
#   `searchlight.select(['smin','smax']).unique()` per pipeline.
# - **Trial filters**: causal5 acoustic restricts to
#   `measure='categorical_acoustic_cue'` (endpoints only); causal4's acoustic
#   decoder uses a different filter. Verify in the subject-level notebooks
#   before comparing.
# - **CV scheme**: causal5 behavior uses ShuffleSplit (minority-class reason
#   documented in memory); causal6 uses StratifiedKFold(5). This can shift
#   per-fold AUC while leaving fold-mean AUC roughly intact.
# - **Regularization**: causal6 uses a tuned reg_lambda from `reg_lambda_sweep`
#   (see `reg_lambda_winners.json`); causal5 uses an inner CV grid. Different
#   regularization → different AUC at high-noise sites.
# - **Peak thresholds**: the pipelines' own peak parquets apply thresholds
#   (causal5 acoustic uses 0.65, behavior uses an improvement cutoff). This
#   notebook bypasses those and takes raw argmax — so sites that would be
#   filtered out of `phon_peaks_df.parquet` still appear in these scatters.
# - **Significance**: only causal6 writes per-site permutation p/q-values
#   (`significance_all.parquet`). Not used here; see
#   `notebooks/causal6/significance_aggregate.py`.
