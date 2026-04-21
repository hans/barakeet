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
# Ambig-vs-unambig simple-decoder AUC per site.
#
# For each site (subject, electrode_idx, phoneme_pair, word_end) in
# `behav_peaks_df`, the HGA-only decoder produces held-out predictions across
# CV folds. Here we split those predictions by trial ambiguity and compute
# AUC_a (ambig trials) and AUC_u (unambig trials) per fold, then aggregate to
# mean ± SEM across folds per site.
#
# Ambiguity is defined behaviorally per (subject, phoneme_pair, word_end) via
# `src.data.get_ambiguous_resampled_steps`: steps that elicit both /d/ and
# /n/ reports across repeats. Endpoints (resampled=1, 6) and mid-continuum
# steps not flagged for that site are treated as unambiguous.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from src.data import get_ambiguous_resampled_steps
from src.viz_paper import (
    pl_roc_auc,
    phoneme_pair_enum,
    subject_enum,
    word_end_enum,
)

# %% tags=["parameters"]
all_predictions = []
all_md_path = "outputs/causal4/prepare_neurometrics/all_md.parquet"
behav_peaks_path = "outputs/causal4/prepare_neurometrics/behav_peaks_df.parquet"
behav_searchlight_path = (
    "outputs/causal4/prepare_neurometrics/behav_roc_auc_searchlight_df.parquet"
)
ambiguous_response_threshold = 2
outdir = "."

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load predictions and metadata

# %%
predictions_df = pl.concat(
    [pl.read_parquet(p) for p in all_predictions]
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
    (pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"),
)

# %%
all_md = pl.read_parquet(all_md_path)
behav_peaks_df = pl.read_parquet(behav_peaks_path)

# %% [markdown]
# ## Derive per-site ambiguity labels
#
# `get_ambiguous_resampled_steps(all_md, ambiguous_response_threshold=...)`
# returns `{(subject, phoneme_pair, word_end): [resampled_steps]}`. We flatten
# this to a long-form frame and left-join to the predictions. Trials that do
# not match fall to `is_ambiguous=False`.

# %%
ambig_map = get_ambiguous_resampled_steps(
    all_md, ambiguous_response_threshold=ambiguous_response_threshold
)

ambig_rows = []
for (subject, phoneme_pair, word_end), steps in ambig_map.items():
    for step in steps:
        ambig_rows.append(
            {
                "subject": subject,
                "phoneme_pair": phoneme_pair,
                "word_end": word_end,
                "resampled": float(step),
                "is_ambiguous": True,
            }
        )

ambig_df = pl.DataFrame(
    ambig_rows,
    schema={
        "subject": subject_enum,
        "phoneme_pair": phoneme_pair_enum,
        "word_end": word_end_enum,
        "resampled": pl.Float64,
        "is_ambiguous": pl.Boolean,
    },
)

# %% [markdown]
# ## Attach `resampled` to each trial-prediction row and tag ambiguity

# %%
md_for_join = all_md.select(
    ["subject", "epoch_idx", "phoneme_pair", "resampled"]
)

predictions_labeled = (
    predictions_df.join(
        md_for_join, on=["subject", "epoch_idx", "phoneme_pair"], how="left"
    )
    .join(
        ambig_df,
        on=["subject", "phoneme_pair", "word_end", "resampled"],
        how="left",
    )
    .with_columns(pl.col("is_ambiguous").fill_null(False))
)

# %% [markdown]
# ## Per-fold AUC split by ambiguity

# %%
site_cols = ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]

per_fold_auc = pl_roc_auc(
    df=predictions_labeled,
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=site_cols + ["fold", "is_ambiguous"],
    roc_auc_name="roc_auc",
)

trial_counts = (
    predictions_labeled.group_by(site_cols + ["fold", "is_ambiguous"])
    .len()
    .rename({"len": "n_trials"})
)

per_fold = per_fold_auc.join(
    trial_counts, on=site_cols + ["fold", "is_ambiguous"], how="inner"
)

# %%
# Pivot long → wide on is_ambiguous so each fold becomes a single row with
# auc_a, auc_u, n_ambig, n_unambig columns.
per_fold_wide = (
    per_fold.with_columns(
        pl.when(pl.col("is_ambiguous")).then(pl.lit("a")).otherwise(pl.lit("u")).alias("cond")
    )
    .pivot(
        on="cond",
        index=site_cols + ["fold"],
        values=["roc_auc", "n_trials"],
    )
)

# Column names after pivot: "roc_auc_a", "roc_auc_u", "n_trials_a", "n_trials_u"
per_fold_wide = per_fold_wide.rename(
    {
        "roc_auc_a": "auc_a",
        "roc_auc_u": "auc_u",
        "n_trials_a": "n_ambig",
        "n_trials_u": "n_unambig",
    }
).with_columns((pl.col("auc_a") - pl.col("auc_u")).alias("diff"))

# %% [markdown]
# ## Aggregate across folds per site: mean ± SEM

# %%
def _sem(col):
    return pl.col(col).std() / pl.col(col).count().sqrt()


per_site = per_fold_wide.group_by(site_cols).agg(
    auc_a_mean=pl.col("auc_a").mean(),
    auc_a_sem=_sem("auc_a"),
    auc_u_mean=pl.col("auc_u").mean(),
    auc_u_sem=_sem("auc_u"),
    diff_mean=pl.col("diff").mean(),
    diff_sem=_sem("diff"),
    n_folds=pl.col("fold").count(),
    n_ambig_mean=pl.col("n_ambig").mean(),
    n_unambig_mean=pl.col("n_unambig").mean(),
)

# %% [markdown]
# ## Save per-fold and per-site tables

# %%
per_fold_wide.write_parquet(outdir / "per_fold.parquet")
per_site.write_parquet(outdir / "per_site.parquet")

# %% [markdown]
# ## Sanity check: simple-decoder AUC_a vs. controlled-decoder AUC on ambig trials
#
# The `behav_roc_auc_searchlight_df` from `prepare_neurometrics` already carries
# the controlled (resampled + HGA) decoder's per-fold AUC at every window. For
# the peak windows, the controlled-decoder AUC is dominated by the acoustic
# prior on unambig trials but is informative on ambig trials — we expect it to
# correlate well with the simple decoder's AUC_a across sites.

# %%
behav_searchlight = pl.read_parquet(behav_searchlight_path)

controlled_at_peaks = (
    behav_peaks_df.select(site_cols)
    .join(behav_searchlight, on=site_cols, how="inner")
    .group_by(site_cols)
    .agg(controlled_auc_mean=pl.col("behav_roc_auc").mean())
)

sanity = per_site.select(site_cols + ["auc_a_mean"]).join(
    controlled_at_peaks, on=site_cols, how="inner"
)
sanity.write_csv(outdir / "sanity_controlled_vs_simple.csv")

# %% [markdown]
# ## Scatter: (auc_u, auc_a) per site with unity line and SEM error bars

# %%
per_site_pd = per_site.to_pandas()

fig, ax = plt.subplots(figsize=(6, 6))
ax.errorbar(
    per_site_pd["auc_u_mean"],
    per_site_pd["auc_a_mean"],
    xerr=per_site_pd["auc_u_sem"],
    yerr=per_site_pd["auc_a_sem"],
    fmt="o",
    alpha=0.5,
    ms=4,
)
lim = [0.3, 1.0]
ax.plot(lim, lim, "k--", alpha=0.5, label="unity")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("AUC (unambiguous trials)")
ax.set_ylabel("AUC (ambiguous trials)")
ax.set_title("Simple (HGA-only) decoder, per-site AUC")
ax.legend()
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(outdir / "scatter.pdf")

# %% [markdown]
# ## Histogram of AUC_a − AUC_u across sites

# %%
diffs = per_site_pd["diff_mean"].values
diff_sems = per_site_pd["diff_sem"].values

n_pos = int(np.sum((diffs - diff_sems) > 0))
n_neg = int(np.sum((diffs + diff_sems) < 0))

fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(diffs, bins=30, ax=ax)
ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.set_xlabel("AUC_a − AUC_u (per site)")
ax.set_ylabel("Count")
ax.set_title(
    f"AUC_a − AUC_u across {len(diffs)} sites "
    f"(mean={np.nanmean(diffs):.3f}, median={np.nanmedian(diffs):.3f})\n"
    f"sites with mean±SEM excluding zero: {n_pos} positive, {n_neg} negative"
)
fig.tight_layout()
fig.savefig(outdir / "diff_histogram.pdf")
