# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Within-completion trial-balance index — all speech-responsive sites
#
# Step 2 of the all-SR perceptual fork
# (`docs/superpowers/plans/2026-08-27-all-speech-responsive-perceptual.md`).
# A pure key-set swap of `trial_balance_index.py`: the site universe is
# `sr_site_universe.parquet` (every speech-responsive electrode x
# phoneme_pair, Step 1) instead of the `phon_peaks p_value < 0.1` candidate
# pool. The per-step class-balance logic itself is unchanged — those counts
# are subject-level and broadcast across electrodes regardless of which site
# list is crossed in.
#
# See `docs/superpowers/plans/2026-05-19-causal46-trial-balance-index.md`
# for the canonical trial-balance design (this notebook is a sibling fork,
# not a revision of it) and `notebooks/causal46_joined/_within_completion.py`
# for the canonical B3/B4 subsampling rule.

# %%
from __future__ import annotations

from pathlib import Path

import mne
import polars as pl

from src.data import add_metadata_features, get_ambiguous_resampled_steps

# %% tags=["parameters"]
sr_site_universe_path = "outputs/causal46_joined/sr_site_universe/sr_site_universe.parquet"
epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/trial_balance_index_all_sr"

# %%
SR_SITE_UNIVERSE = Path(sr_site_universe_path)
EPOCH_DIR = Path(epoch_dir)
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# K=5 is the recommended default; K=4 is the permissive companion for
# borderline subjects; K=10 is retained as a strict-tail sanity column.
THRESHOLDS = (3, 4, 5, 10)

# %% [markdown]
# ## Load the all-SR site universe and discover needed subjects

# %%
canonical = pl.read_parquet(SR_SITE_UNIVERSE)

needed_subjects = sorted(canonical["subject"].unique().to_list())
print(f"All-SR sites: {canonical.height}  across {len(needed_subjects)} subjects")
print(f"Subjects: {needed_subjects}")

# %% [markdown]
# ## Read epoch metadata for each subject (metadata only — no data load)

# %%
md_frames: list[pl.DataFrame] = []
for subject in needed_subjects:
    path = EPOCH_DIR / f"{subject}_epo.fif"
    if not path.exists():
        print(f"  ⚠ {subject}: {path} missing — skipping")
        continue
    epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
    md = add_metadata_features(epochs.metadata).reset_index(drop=True)
    md_pl = pl.from_pandas(
        md[["phoneme_pair", "word_end", "resampled", "behavior_dummy_forced"]]
    ).with_columns(pl.lit(subject).alias("subject"))
    md_frames.append(md_pl)

all_md = pl.concat(md_frames)
print(f"Total trial rows loaded: {all_md.height}")
print("Per-subject row counts:")
print(all_md.group_by("subject").len().sort("subject"))

# %%
# Sanity: assert types are clean, no nulls in the columns we depend on.
# A null word_end (e.g. catch trials) would silently inflate counts by
# broadcasting across both word_ends in the cross-join below.
assert all_md["behavior_dummy_forced"].is_in([0, 1]).all(), "expected behavior_dummy_forced ∈ {0, 1}"
assert all_md["resampled"].is_in([1, 2, 3, 4, 5, 6]).all(), "expected resampled ∈ {1..6}"
assert all_md.filter(pl.col("word_end").is_null()).height == 0, "found nulls in word_end"

# %% [markdown]
# ## Per-(subject, pp, word_end, resampled) class counts
#
# Independent of electrode_idx — these counts are subject-level and broadcast
# across all electrodes of the same subject in the cross-join below.

# %%
# Per-step `is_ambiguous_step` mirrors `src.data.get_ambiguous_resampled_steps`
# (`min_class > 2`, endpoints excluded). It says "this step elicited both
# responses non-trivially", nothing about sample size — sample-size gating
# happens at the cell level via `meets_threshold_K` on the pooled balanced N.
counts = (
    all_md
    .group_by(["subject", "phoneme_pair", "word_end", "resampled"])
    .agg(
        (pl.col("behavior_dummy_forced") == 0).sum().alias("n_class0"),
        (pl.col("behavior_dummy_forced") == 1).sum().alias("n_class1"),
        pl.len().alias("n_total"),
    )
    .with_columns(
        pl.min_horizontal("n_class0", "n_class1").alias("min_class"),
    )
    .with_columns(
        (
            (pl.col("min_class") > 2)
            & (~pl.col("resampled").is_in([1, 6]))
        ).alias("is_ambiguous_step"),
    )
    .sort(["subject", "phoneme_pair", "word_end", "resampled"])
)

print(f"counts table: {counts.height} rows")
print(counts.head(12))

counts.write_csv(OUT_DIR / "trial_counts_by_subject.csv")
print(f"Written: {OUT_DIR / 'trial_counts_by_subject.csv'}")

# %%
# How many ambiguous steps (M-passing) per (subject, pp, word_end)?
amb_per_we = (
    counts.filter(pl.col("is_ambiguous_step"))
    .group_by(["subject", "phoneme_pair", "word_end"])
    .agg(
        pl.col("resampled").sort().alias("ambiguous_steps"),
        pl.len().alias("n_ambiguous"),
    )
    .sort(["subject", "phoneme_pair", "word_end"])
)
print(f"(subject, pp, word_end) with ≥1 ambiguous step: {amb_per_we.height}")
print("distribution of n_ambiguous:")
print(amb_per_we.group_by("n_ambiguous").len().sort("n_ambiguous"))

# %% [markdown]
# ## Cross with the all-SR site universe
#
# sr_site_universe.parquet is keyed by (subject, electrode_idx,
# phoneme_pair) — no word_end. Join with the (word_end, resampled) tuples
# observed in metadata for each (subject, phoneme_pair); the natural join
# automatically enumerates both word_ends per phoneme_pair plus every
# resampled step.

# %%
# Reduce canonical to the keys we need; drop columns that conflict downstream
# (acoustic_significant / phon_smin / phon_smax are annotations, not part of
# the trial-balance key set — consumers re-attach them from
# sr_site_universe.parquet directly when needed).
canonical_keys = canonical.select(["subject", "electrode_idx", "phoneme_pair"]).unique()

# Join key is (subject, phoneme_pair). counts has the (word_end, resampled)
# dimensions we need; broadcasts across electrodes.
trial_balance = (
    canonical_keys
    .join(counts, on=["subject", "phoneme_pair"], how="inner")
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end", "resampled"])
)

print(f"trial_balance: {trial_balance.height} rows "
      f"(canonical {canonical_keys.height} × ~12 per site = ~{canonical_keys.height * 12})")

trial_balance.write_csv(OUT_DIR / "trial_balance_index.csv")
print(f"Written: {OUT_DIR / 'trial_balance_index.csv'}")

# %%
def _step_str(steps_list):
    return ",".join(str(s) for s in steps_list)

# Start from all (site, word_end) combinations so cells with zero ambiguous steps still appear.
all_site_we = trial_balance.select(
    ["subject", "electrode_idx", "phoneme_pair", "word_end"]
).unique()

# Pool aggregates across ambiguous steps, per (site, word_end). At each
# ambiguous step `s`, the per-step class-balanced contribution is
# `min_class[s]` per class (downstream consumers draw both classes WITH
# REPLACEMENT to that size — see the canonical rule in `_within_completion.py`).
# The cell pool sums those contributions: `n_per_class = sum_s min_class[s]`. Both
# classes have the same step composition by construction, so the
# class0–class1 contrast is free of within-class step-acoustic confounds.
# `meets_threshold_K = n_per_class >= K` gates cell inclusion downstream.
pool = (
    trial_balance.filter(pl.col("is_ambiguous_step"))
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.col("resampled").sort().alias("ambiguous_steps"),
        pl.col("min_class").sum().alias("n_per_class"),
        pl.col("n_class0").sum().alias("n_class0_pool"),
        pl.col("n_class1").sum().alias("n_class1_pool"),
        pl.len().alias("n_ambiguous"),
    )
)

summary = (
    all_site_we
    .join(
        pool,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="left",
    )
    .with_columns(
        pl.col("n_class0_pool").fill_null(0),
        pl.col("n_class1_pool").fill_null(0),
        pl.col("n_per_class").fill_null(0),
        pl.col("n_ambiguous").fill_null(0),
    )
    .with_columns(
        pl.col("ambiguous_steps")
        .map_elements(lambda lst: _step_str(lst) if lst is not None else "", return_dtype=pl.Utf8)
        .alias("ambiguous_steps"),
        *[
            (pl.col("n_per_class") >= k).alias(f"meets_threshold_{k}")
            for k in THRESHOLDS
        ],
    )
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
)

summary.write_csv(OUT_DIR / "trial_balance_summary.csv")
print(f"Written: {OUT_DIR / 'trial_balance_summary.csv'}")
print(summary.head(10))

# %%
# Sanity: row arithmetic.
# Sites × (word_ends × resampled_steps observed per subject/pp) = trial_balance.height
expected_per_site = (
    counts.group_by(["subject", "phoneme_pair"])
    .len()
    .rename({"len": "_n_we_resampled"})
)
expected_total = (
    canonical_keys.join(expected_per_site, on=["subject", "phoneme_pair"], how="inner")
    ["_n_we_resampled"].sum()
)
assert trial_balance.height == expected_total, (
    f"row count mismatch: trial_balance has {trial_balance.height}, expected {expected_total}"
)
print(f"✓ row count check passes: {trial_balance.height}")

# %% [markdown]
# ## Cross-check against src.data.get_ambiguous_resampled_steps
#
# That helper computes the same qualifying-step set at threshold=2 and
# excludes endpoints 1 and 6. The oracle uses `min_count > threshold`
# (strictly greater), so at threshold=2 a step qualifies when min_class >= 3.
# Anything other than exact agreement is a counting bug.

# %%
ref_dict = get_ambiguous_resampled_steps(all_md, ambiguous_response_threshold=2)

# Convert dict to polars DataFrame for comparison.
ref_df = pl.DataFrame(
    [(s, pp, we, sorted(xs)) for (s, pp, we), xs in ref_dict.items()],
    schema={"subject": pl.Utf8, "phoneme_pair": pl.Utf8, "word_end": pl.Utf8, "_ref": pl.List(pl.Int64)},
    orient="row",
)

# `is_ambiguous_step` is defined as `min_class > 2 AND resampled ∉ {1, 6}` —
# i.e., exactly the oracle's condition.
ours_df = (
    counts.filter(pl.col("is_ambiguous_step"))
    .group_by(["subject", "phoneme_pair", "word_end"])
    .agg(pl.col("resampled").sort().alias("_ours"))
)

merged = ref_df.join(
    ours_df,
    on=["subject", "phoneme_pair", "word_end"],
    how="full",
)

mismatches = merged.filter(
    pl.col("_ref").is_null()
    | pl.col("_ours").is_null()
    | (pl.col("_ref") != pl.col("_ours"))
)
assert mismatches.height == 0, (
    f"counts disagree with get_ambiguous_resampled_steps on "
    f"{mismatches.height} (subject, pp, word_end) tuples:\n{mismatches}"
)
print(f"✓ matches get_ambiguous_resampled_steps on all "
      f"{ref_df.height} (subject, pp, word_end) tuples at threshold=2")

# %% [markdown]
# ## Summary for downstream consumers
#
# - `trial_balance_index.csv` — long format, one row per
#   (all_sr_site, word_end, resampled). Carries per-step `is_ambiguous_step`
#   (min_class > 2, endpoints excluded). No K threshold lives at this level.
# - `trial_balance_summary.csv` — per (site, word_end). Columns:
#   `ambiguous_steps` (comma-joined), `n_ambiguous`, `n_class0_pool`,
#   `n_class1_pool`, `n_per_class` (= max class-balanced subsample size over
#   the union of ambiguous steps), and `meets_threshold_K` for K ∈ {3, 4, 5, 10}.
# - `trial_counts_by_subject.csv` — electrode-agnostic raw counts.
#   Useful for sanity checks but not for plotting.
#
# Cell inclusion rule downstream (B4): a (site, word_end) cell qualifies iff
# `meets_threshold_K`, `n_ambiguous ≥ 1`. `acoustic_significant` is NOT a
# column here by design — it lives in `sr_site_universe.parquet` and is
# re-attached downstream (`t_tests_all_sr.py`) as an annotation.

# %%
for k in THRESHOLDS:
    col = f"meets_threshold_{k}"
    n_any = summary.filter(pl.col(col)).height
    n_two_plus = summary.filter(
        pl.col(col) & (pl.col("n_ambiguous") >= 1)
    ).height
    print(
        f"K={k:2d}: (site×word_end) with n_per_class ≥ K: {n_any:4d}/{summary.height}; "
        f"and ≥1 ambiguous step: {n_two_plus:4d}/{summary.height}"
    )
