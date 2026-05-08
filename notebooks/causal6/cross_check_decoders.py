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
# # Cross-check decoding results: causal4 vs causal6
#
# Answers two questions, per decoder type:
#   1. How much do the per-electrode decoding results agree across pipelines?
#   2. For sites where they disagree, what explains the drift?
#
# Pipelines compared: causal4 (legacy) vs causal6 (current). Decoder types:
# acoustic and behavior-with-control. (causal4 never ran HGA-only.) causal5
# loaders were dropped because we never plot it; bring them back from git
# history if you need a three-way comparison again.
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

from src.viz_paper import pl_roc_auc

# %% tags=["parameters"]
causal4_root = "outputs/causal4"
causal6_root = "outputs/causal6"

top_n_disagreements = 20
behav_peak_post_offset_s = 0.2
epoch_tmin = -0.4
epoch_sfreq = 100

# Pipeline pairs to plot. We treat causal6 as the trusted reference (better CV
# scheme + tuned regularization → lower-variance per-fold estimates) and use it
# to validate causal4's small claimed effect sizes.
PIPELINE_PAIRS = [("causal4", "causal6")]

# %%
ROOTS = {
    "causal4": Path(causal4_root),
    "causal6": Path(causal6_root),
}

for name, root in ROOTS.items():
    print(f"{name}: {root}  (exists={root.exists()})")

# %% [markdown]
# ### Subject discovery
#
# Discover subjects from each pipeline's per-subject directories rather than
# hard-coding a list. Then verify that the relevant pipelines (those in
# `PIPELINE_PAIRS`) produce the same subject set — divergence is surfaced
# explicitly so we don't silently drop subjects from a comparison.

# %%
GLOB_TARGETS = {
    "causal4": [
        "behavior_decoding_single_electrode_acoustic/*",
        "behavior_decoding_single_electrode_summarize/*",
    ],
    "causal6": [
        "acoustic_decoding_single_electrode/*",
        "behavior_decoding_single_electrode/*",
        "behavior_decoding_single_electrode_hga_only/*",
    ],
}


def _discover_subjects(root: Path, patterns: list[str]) -> set[str]:
    out: set[str] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_dir():
                out.add(p.name)
    return out


DISCOVERED = {
    p: sorted(_discover_subjects(ROOTS[p], pats))
    for p, pats in GLOB_TARGETS.items() if p in ROOTS
}

print("Discovered subjects per pipeline:")
for pipe, ss in DISCOVERED.items():
    print(f"  {pipe}: {len(ss)} → {ss}")

relevant = sorted({p for pair in PIPELINE_PAIRS for p in pair})
relevant_sets = {p: set(DISCOVERED.get(p, [])) for p in relevant}
inter = sorted(set.intersection(*relevant_sets.values())) if relevant_sets else []
union = sorted(set.union(*relevant_sets.values())) if relevant_sets else []
print(f"\nRelevant pipelines {relevant}:")
print(f"  intersection ({len(inter)}): {inter}")
print(f"  union        ({len(union)}): {union}")
for p, s in relevant_sets.items():
    only = sorted(s - set(inter))
    if only:
        print(f"  only in {p}: {only}")
if relevant and not all(s == relevant_sets[relevant[0]] for s in relevant_sets.values()):
    print("\n⚠ relevant pipelines disagree on subject coverage — comparisons will inner-join, "
          "so divergent subjects are silently dropped at the peak/scatter level.")

subjects = union  # loaders skip per-subject paths that don't exist, so union is safe

# %% [markdown]
# ## 0. Speech-responsive screen comparison
#
# The decoding coverage asymmetry (~700 sites differ in each direction) traces
# back here: causal4 uses an amplitude-threshold criterion (max|epoch| > 0.3
# after baselining, averaged evoked), while causal5/causal6 uses a paired
# t-test across all epochs (pre- vs post-onset mean, t > 7) and overrides the
# amplitude flag with that result. Both are per-subject; causal6 imports its
# electrode lists directly from `outputs/causal5/find_speech_responsive/`.
#
# This section loads both screens, shows their per-subject counts, and
# breaks down the 4-way agreement (both / only-causal4 / only-causal5 / neither).

# %%
causal5_speech_resp_root = Path("outputs/causal5/find_speech_responsive")
causal4_speech_resp_root = ROOTS["causal4"] / "find_speech_responsive"


def load_speech_responsive(root: Path, subjects: list[str]) -> pl.DataFrame:
    frames = []
    for subj in subjects:
        p = root / f"{subj}_results.csv"
        if not p.exists():
            continue
        df = pl.read_csv(p)
        keep = ["subject", "electrode_idx", "speech_responsive"]
        optional = ["speech_responsive_test_value", "speech_responsive_tval",
                    "speech_responsive_ttest"]
        keep += [c for c in optional if c in df.columns]
        frames.append(df.select(keep).with_columns(
            pl.col("speech_responsive").cast(pl.Boolean)
        ))
    if not frames:
        return pl.DataFrame({"subject": pl.Series([], dtype=pl.Utf8),
                             "electrode_idx": pl.Series([], dtype=pl.Int64),
                             "speech_responsive": pl.Series([], dtype=pl.Boolean)})
    return pl.concat(frames, how="diagonal")


sr_c4 = load_speech_responsive(causal4_speech_resp_root, subjects)
sr_c5 = load_speech_responsive(causal5_speech_resp_root, subjects)

# %%
print("Speech-responsive electrode counts (all electrodes loaded):")
for label, df in [("causal4", sr_c4), ("causal5/6", sr_c5)]:
    if df.is_empty():
        print(f"  {label}: (no data)")
        continue
    n_resp = df.filter(pl.col("speech_responsive")).height
    n_total = df.height
    print(f"  {label}: {n_resp} / {n_total} = {n_resp/n_total:.1%}")

# %%
if not sr_c4.is_empty() and not sr_c5.is_empty():
    site_cols_sr = ["subject", "electrode_idx"]
    joined_sr = (
        sr_c4.select(site_cols_sr + ["speech_responsive"])
        .rename({"speech_responsive": "sr_c4"})
        .join(
            sr_c5.select(site_cols_sr + ["speech_responsive"])
            .rename({"speech_responsive": "sr_c5"}),
            on=site_cols_sr, how="outer",
        )
        .with_columns([
            pl.col("sr_c4").fill_null(False),
            pl.col("sr_c5").fill_null(False),
        ])
    )
    both    = joined_sr.filter( pl.col("sr_c4") &  pl.col("sr_c5")).height
    only_c4 = joined_sr.filter( pl.col("sr_c4") & ~pl.col("sr_c5")).height
    only_c5 = joined_sr.filter(~pl.col("sr_c4") &  pl.col("sr_c5")).height
    neither = joined_sr.filter(~pl.col("sr_c4") & ~pl.col("sr_c5")).height
    total   = joined_sr.height
    print("\n4-way agreement (all shared electrode slots):")
    print(f"  both responsive  : {both:5d}  ({both/total:.1%})")
    print(f"  only causal4     : {only_c4:5d}  ({only_c4/total:.1%})")
    print(f"  only causal5/6   : {only_c5:5d}  ({only_c5/total:.1%})")
    print(f"  neither          : {neither:5d}  ({neither/total:.1%})")

    # Per-subject breakdown
    print("\nPer-subject breakdown (both / only-c4 / only-c5 / neither):")
    for subj, grp in joined_sr.group_by("subject", maintain_order=True):
        b  = grp.filter( pl.col("sr_c4") &  pl.col("sr_c5")).height
        c4 = grp.filter( pl.col("sr_c4") & ~pl.col("sr_c5")).height
        c5 = grp.filter(~pl.col("sr_c4") &  pl.col("sr_c5")).height
        n  = grp.filter(~pl.col("sr_c4") & ~pl.col("sr_c5")).height
        print(f"  {subj[0]:>6}:  both={b:3d}  only_c4={c4:3d}  only_c5={c5:3d}  neither={n:3d}")

    # If causal5 wrote test values, show the distribution at disagreement sites
    if "speech_responsive_test_value" in sr_c5.columns:
        disagreement = joined_sr.filter(pl.col("sr_c4") != pl.col("sr_c5"))
        if not disagreement.is_empty():
            with_vals = disagreement.join(
                sr_c5.select(["subject", "electrode_idx", "speech_responsive_test_value"]),
                on=["subject", "electrode_idx"], how="left",
            )
            print("\nAmplitude test values at disagreement sites (causal5 criterion):")
            for group_label, filt in [
                ("only_causal4 (c4=T, c5=F)", with_vals.filter( pl.col("sr_c4") & ~pl.col("sr_c5"))),
                ("only_causal5 (c4=F, c5=T)", with_vals.filter(~pl.col("sr_c4") &  pl.col("sr_c5"))),
            ]:
                vals = filt["speech_responsive_test_value"].drop_nulls().to_numpy()
                if len(vals):
                    print(f"  {group_label}: n={len(vals)}, "
                          f"med={np.nanmedian(vals):.3f}, "
                          f"min={vals.min():.3f}, max={vals.max():.3f}")

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
# - causal6 has cached fold-level parquets — read directly.
# - causal4 requires deriving AUC from trial-level `all_outcomes.parquet` /
#   `A-predictions.parquet` via sklearn.

# %%
# Per-fold AUC is computed via the canonical rank-based polars helper
# `pl_roc_auc` (src/viz_paper.py:1645). Same Mann–Whitney U identity, same
# tie-handling as sklearn — but as a single polars query per call, so the
# tens-of-thousands of (subject, electrode, smin, smax, fold) groups are
# scored without a Python-per-group loop.


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
        df = pl.read_parquet(p)
        if df.is_empty():
            continue
        # binary target encoding (causal5 uses `== 1`); match that
        df = df.with_columns((pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"))
        auc = pl_roc_auc(
            df=df,
            target_col="decoder_target",
            proba_col="decoder_proba",
            group_cols=["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold"],
            roc_auc_name="roc_auc",
        )
        frames.append(auc)
    if not frames:
        return _empty([
            ("subject", pl.Utf8), ("electrode_idx", pl.Int64), ("phoneme_pair", pl.Utf8),
            ("smin", pl.Int64), ("smax", pl.Int64), ("fold", pl.Int64), ("roc_auc", pl.Float64),
        ])
    return pl.concat(frames)


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

    Used for causal4 (no cached searchlight). Concatenates late and early
    prediction files if both exist. Mirrors the full+baseline join pattern
    used in `notebooks/causal4/A_neurometrics.py` (~line 531).
    """
    group_cols = [
        "subject", "electrode_idx", "phoneme_pair", "word_end",
        "smin", "smax", "fold",
    ]
    frames = []
    for p in paths:
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        if df.is_empty():
            continue
        df = df.with_columns(pl.col("decoder_target").cast(pl.Int8))
        full = pl_roc_auc(
            df=df, target_col="decoder_target", proba_col="full_decoder_proba",
            group_cols=group_cols, roc_auc_name="roc_auc_full",
        )
        base = pl_roc_auc(
            df=df, target_col="decoder_target", proba_col="baseline_decoder_proba",
            group_cols=group_cols, roc_auc_name="roc_auc_baseline",
        )
        frames.append(full.join(base, on=group_cols, how="inner"))
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
load_spec = [
    ("acoustic", load_acoustic_searchlight_causal4, load_acoustic_searchlight_causal6),
    ("behavior_ctrl", load_behavior_ctrl_searchlight_causal4, load_behavior_ctrl_searchlight_causal6),
    ("behavior_hga", None, load_behavior_hga_searchlight_causal6),
]
SEARCHLIGHTS = {}
for target, loader4, loader6 in tqdm(load_spec):
    causal4_result = _normalize_types(loader4(ROOTS["causal4"], subjects)) if loader4 else None
    causal6_result = _normalize_types(loader6(ROOTS["causal6"], subjects)) if loader6 else None
    SEARCHLIGHTS[target] = {
        "causal4": causal4_result,
        "causal6": causal6_result,
    }

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
    """Peak-AUC scatter; points colored by subject so single-subject drift is visible."""
    pipes = [p for p, df in peaks.items() if not df.is_empty()]
    pairs = [(a, b) for a, b in PIPELINE_PAIRS if a in pipes and b in pipes]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        joined = pair_peaks(peaks[a], peaks[b], site_cols, criterion)
        if joined.empty:
            ax.set_title(f"{a} vs {b}: no shared sites")
            continue
        x = joined[f"{criterion}_a"].to_numpy()
        y = joined[f"{criterion}_b"].to_numpy()
        for subj, sub in joined.groupby("subject"):
            ax.scatter(sub[f"{criterion}_a"], sub[f"{criterion}_b"], s=14, alpha=0.65, label=subj)
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.75, alpha=0.5)
        mask = ~(np.isnan(x) | np.isnan(y))
        r = pearsonr(x[mask], y[mask])[0] if mask.sum() >= 3 else float("nan")
        ax.set_title(f"{kind} peak AUC\nr={r:.3f}, n={int(mask.sum())}")
        ax.set_xlabel(f"{a} {criterion}")
        ax.set_ylabel(f"{b} {criterion}")
        ax.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle(f"{kind}: peak AUC agreement (color = subject)")
    fig.tight_layout()
    plt.show()


def plot_peak_window_scatter(peaks: dict[str, pl.DataFrame], site_cols: list[str], criterion: str, kind: str):
    pipes = [p for p, df in peaks.items() if not df.is_empty()]
    pairs = [(a, b) for a, b in PIPELINE_PAIRS if a in pipes and b in pipes]
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
    for a, b in [(x, y) for x, y in PIPELINE_PAIRS if x in pipes and y in pipes]:
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
    for a, b in [(x, y) for x, y in PIPELINE_PAIRS if x in pipes and y in pipes]:
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


# %% [markdown]
# ### Extra diagnostics for the causal4 vs causal6 question
#
# The four functions below answer questions the headline scatter can't:
# - **window-grid summary**: silent grid mismatch makes the inner-join drop
#   non-shared windows; we print it explicitly so we know what fraction of
#   the search space is even comparable.
# - **selection-bias check**: causal4 only ran behavior decoding on
#   acoustically selected sites, so the inner-join already restricts to
#   that subset. This block reports how many "high-AUC in causal6" sites
#   were never tested in causal4 at all.
# - **Bland–Altman & paired histograms**: a scatter with r≈0.7 hides
#   systematic bias and regression-to-mean. These two views show whether
#   causal6's distribution is shifted (and by how much) at matched sites.
# - **peak fold-std**: causal6 should produce tighter per-fold estimates;
#   if so, that's empirical evidence the methodological upgrade is real
#   and lends weight to causal6 where the two pipelines disagree.

# %%
def print_window_grid_summary(searchlights: dict[str, pl.DataFrame], kind: str) -> None:
    pipes = [p for p, df in searchlights.items() if not df.is_empty()]
    grids = {p: set(map(tuple, searchlights[p].select(["smin", "smax"]).unique().rows())) for p in pipes}
    print(f"\n--- {kind}: window-grid ---")
    for p, g in grids.items():
        print(f"  {p}: {len(g)} unique (smin,smax)")
    pairs = [(a, b) for a, b in PIPELINE_PAIRS if a in pipes and b in pipes]
    for a, b in pairs:
        shared = grids[a] & grids[b]
        union = grids[a] | grids[b]
        print(f"  {a} ∩ {b} = {len(shared)} / union {len(union)} "
              f"(only-{a}={len(grids[a] - grids[b])}, only-{b}={len(grids[b] - grids[a])})")


def print_selection_bias(
    peaks: dict[str, pl.DataFrame], site_cols: list[str], criterion: str, kind: str,
    thresholds: tuple[float, ...] = (0.55, 0.6, 0.65),
) -> None:
    pipes = [p for p, df in peaks.items() if not df.is_empty()]
    pairs = [(a, b) for a, b in PIPELINE_PAIRS if a in pipes and b in pipes]
    if not pairs:
        return
    print(f"\n--- {kind}: selection-bias check ---")
    for a, b in pairs:
        a_sites = peaks[a].select(site_cols)
        b_sites = peaks[b].select(site_cols)
        for thr in thresholds:
            a_hi = peaks[a].filter(pl.col(criterion) >= thr)
            b_hi = peaks[b].filter(pl.col(criterion) >= thr)
            a_in_b = a_hi.select(site_cols).join(b_sites, on=site_cols, how="inner").height
            b_in_a = b_hi.select(site_cols).join(a_sites, on=site_cols, how="inner").height
            print(
                f"  thr ≥{thr:.2f}: {a}={a_hi.height} (of which {a_in_b} in {b}'s coverage); "
                f"{b}={b_hi.height} (of which {b_in_a} in {a}'s coverage; "
                f"so {b_hi.height - b_in_a} {b}-high sites were never tested in {a})"
            )


def plot_bland_altman_and_paired_hist(
    peaks: dict[str, pl.DataFrame], site_cols: list[str], criterion: str, kind: str,
) -> None:
    """Two rows: top = Bland–Altman (mean vs Δ); bottom = paired histograms."""
    pipes = [p for p, df in peaks.items() if not df.is_empty()]
    pairs = [(a, b) for a, b in PIPELINE_PAIRS if a in pipes and b in pipes]
    if not pairs:
        return
    fig, axes = plt.subplots(2, len(pairs), figsize=(5 * len(pairs), 8), squeeze=False)
    for col, (a, b) in enumerate(pairs):
        joined = pair_peaks(peaks[a], peaks[b], site_cols, criterion)
        ax_ba, ax_h = axes[0, col], axes[1, col]
        if joined.empty:
            ax_ba.set_title("(no shared sites)"); ax_h.set_title("(no shared sites)")
            continue
        x = joined[f"{criterion}_a"].to_numpy()
        y = joined[f"{criterion}_b"].to_numpy()
        delta = x - y
        mean_xy = (x + y) / 2.0
        med = float(np.nanmedian(delta)); sd = float(np.nanstd(delta))
        ax_ba.scatter(mean_xy, delta, s=10, alpha=0.5)
        ax_ba.axhline(0, color="k", lw=0.5)
        ax_ba.axhline(med, color="tomato", lw=1.0, label=f"median Δ = {med:+.3f}")
        ax_ba.axhline(med + 1.96 * sd, color="tomato", lw=0.6, ls="--")
        ax_ba.axhline(med - 1.96 * sd, color="tomato", lw=0.6, ls="--",
                      label=f"±1.96 sd ({sd:.3f})")
        ax_ba.set_xlabel(f"mean {criterion}")
        ax_ba.set_ylabel(f"{a} − {b}")
        ax_ba.set_title(f"{kind} Bland–Altman  n={len(delta)}")
        ax_ba.legend(fontsize=8)

        bins = 30
        ax_h.hist(x, bins=bins, alpha=0.55,
                  label=f"{a}  med={np.nanmedian(x):.3f}")
        ax_h.hist(y, bins=bins, alpha=0.55,
                  label=f"{b}  med={np.nanmedian(y):.3f}")
        chance = 0.5 if criterion == "roc_auc" else 0.0
        ax_h.axvline(chance, color="k", lw=0.5, ls="--")
        ax_h.set_xlabel(criterion)
        ax_h.set_ylabel("matched sites")
        ax_h.set_title(f"{kind} matched-site peak {criterion}")
        ax_h.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_peak_fold_std(
    searchlights: dict[str, pl.DataFrame], peaks: dict[str, pl.DataFrame],
    site_cols: list[str], criterion: str, kind: str,
) -> None:
    """At each pipeline's own peak window, fold-std distribution. Lower = tighter."""
    pipes = [p for p, df in searchlights.items() if not df.is_empty()]
    pairs = [(a, b) for a, b in PIPELINE_PAIRS if a in pipes and b in pipes]
    if not pairs:
        return

    def _fold_std_at_peak(name: str) -> pl.DataFrame:
        pk = peaks[name].select(site_cols + ["smin", "smax"])
        sl = searchlights[name].join(pk, on=site_cols + ["smin", "smax"], how="inner")
        return (
            sl.group_by(site_cols)
              .agg(pl.col(criterion).std().alias("fold_std"))
        )

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4), squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        fa = _fold_std_at_peak(a).rename({"fold_std": "fold_std_a"})
        fb = _fold_std_at_peak(b).rename({"fold_std": "fold_std_b"})
        joined = fa.join(fb, on=site_cols, how="inner").to_pandas()
        if joined.empty:
            ax.set_title(f"{a} vs {b}: no shared sites"); continue
        hi = float(np.nanmax([joined["fold_std_a"].max(), joined["fold_std_b"].max()]))
        bins = np.linspace(0, hi, 30)
        ax.hist(joined["fold_std_a"], bins=bins, alpha=0.55,
                label=f"{a}  med={joined['fold_std_a'].median():.3f}")
        ax.hist(joined["fold_std_b"], bins=bins, alpha=0.55,
                label=f"{b}  med={joined['fold_std_b'].median():.3f}")
        ax.set_xlabel("fold-std at peak window")
        ax.set_ylabel("matched sites")
        ax.set_title(f"{kind}: per-fold variability at each pipeline's own peak")
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


# %%
def run_comparison(kind: str, n_show_heatmaps: int = 5):
    print(f"\n====================  {kind}  ====================")
    site_cols = SITE_COLS[kind]
    criterion = PEAK_CRITERION[kind]
    searchlights = SEARCHLIGHTS[kind]
    peaks = PEAKS[kind]

    print_window_grid_summary(searchlights, kind)
    print_selection_bias(peaks, site_cols, criterion, kind)

    plot_peak_auc_scatter(peaks, site_cols, criterion, kind)
    plot_peak_window_scatter(peaks, site_cols, criterion, kind)
    plot_bland_altman_and_paired_hist(peaks, site_cols, criterion, kind)
    plot_peak_fold_std(searchlights, peaks, site_cols, criterion, kind)

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
# Skipped: causal4 never ran HGA-only behavior decoding, and we've restricted
# `PIPELINE_PAIRS` to causal4 vs causal6. `BEHAV_HGA` is still populated for
# causal6 if a single-pipeline view is ever useful.

# %% [markdown]
# ## 4. Highlight-electrode drill-in
#
# The most direct test of "are causal4's small effects real?" is a fold-level
# look at the electrodes that causal4's `A_neurometrics` highlighted: if their
# claimed peak AUC comes from one anomalous fold, that's the smoking gun. For
# each named site, we plot:
# - side-by-side (smin, smax) searchlight heatmap with each pipeline's peak
#   starred,
# - per-fold AUC at each pipeline's own peak window (so you can see whether the
#   peak is consistent across folds or driven by a single outlier),
# - a small text summary of (smin, smax, fold-mean, fold-std).

# %%
def inspect_site(
    kind: str, subject: str, electrode_idx: int, phoneme_pair: str,
    word_end: str | None = None,
):
    site_cols = SITE_COLS[kind]
    criterion = PEAK_CRITERION[kind]
    site_values: dict[str, object] = {
        "subject": subject, "electrode_idx": int(electrode_idx),
        "phoneme_pair": phoneme_pair,
    }
    if "word_end" in site_cols:
        if word_end is None:
            print(f"word_end required for {kind}"); return
        site_values["word_end"] = word_end

    label = " ".join(f"{k}={v}" for k, v in site_values.items())
    print(f"\n=== {kind}  {label} ===")

    sls = SEARCHLIGHTS[kind]; peaks = PEAKS[kind]
    pipes = [p for p, df in sls.items() if not df.is_empty()]
    pairs = [(a, b) for a, b in PIPELINE_PAIRS if a in pipes and b in pipes]

    def _site_filter(df):
        for k, v in site_values.items():
            df = df.filter(pl.col(k) == v)
        return df

    for a, b in pairs:
        # Print per-pipeline peak summaries
        for name in (a, b):
            row = _site_filter(peaks[name])
            if row.is_empty():
                print(f"  {name}: site not present"); continue
            r = row.to_pandas().iloc[0]
            sl_at_peak = _site_filter(sls[name]).filter(
                (pl.col("smin") == int(r["smin"])) & (pl.col("smax") == int(r["smax"]))
            )
            folds = sl_at_peak.select(criterion).to_numpy().flatten()
            print(
                f"  {name}: peak (smin={int(r['smin'])}, smax={int(r['smax'])})  "
                f"{criterion}={r[criterion]:.3f}  "
                f"folds n={len(folds)}, mean={np.nanmean(folds):.3f}, std={np.nanstd(folds):.3f}, "
                f"min={np.nanmin(folds):.3f}, max={np.nanmax(folds):.3f}"
            )

        # Side-by-side heatmaps + per-fold strip
        fig = plt.figure(figsize=(13, 5.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.45, wspace=0.25)
        for col, name in enumerate((a, b)):
            ax_h = fig.add_subplot(gs[0, col])
            plot_searchlight_heatmap(ax_h, sls[name], site_values, site_cols, criterion, name)
            row = _site_filter(peaks[name])
            if not row.is_empty():
                r = row.to_pandas().iloc[0]
                ax_h.plot(int(r["smin"]), int(r["smax"]), "r*", markersize=14)

            ax_f = fig.add_subplot(gs[1, col])
            if not row.is_empty():
                sl_at_peak = (
                    _site_filter(sls[name]).filter(
                        (pl.col("smin") == int(r["smin"])) & (pl.col("smax") == int(r["smax"]))
                    ).sort("fold")
                )
                folds = sl_at_peak.select("fold").to_numpy().flatten()
                aucs = sl_at_peak.select(criterion).to_numpy().flatten()
                ax_f.bar(folds, aucs, color="steelblue")
                chance = 0.5 if criterion == "roc_auc" else 0.0
                ax_f.axhline(chance, color="k", lw=0.5, ls="--")
                ax_f.set_xlabel("fold")
                ax_f.set_ylabel(criterion)
                ax_f.set_title(f"{name}: per-fold AUC at peak")
        fig.suptitle(label)
        plt.show()


# %% [markdown]
# ### causal4-highlighted electrodes
# Sourced from `notebooks/causal4/A_neurometrics.py`. If causal6 reproduces
# the peak window and per-fold consistency, the causal4 effect survives;
# if causal6's fold-mean drops or its peak shifts, treat the causal4 result
# as overstated.

# %%
inspect_site("acoustic",      subject="EC250", electrode_idx=185, phoneme_pair="dn")
inspect_site("behavior_ctrl", subject="EC250", electrode_idx=185, phoneme_pair="dn", word_end="desolate")
inspect_site("acoustic",      subject="EC278", electrode_idx=38,  phoneme_pair="dn")
inspect_site("behavior_ctrl", subject="EC278", electrode_idx=38,  phoneme_pair="dn", word_end="necessary")

# %% [markdown]
# ## 5. Notes on expected differences
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
