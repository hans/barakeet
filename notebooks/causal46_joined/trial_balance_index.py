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
# # Within-completion trial-balance index
#
# For every (canonical AS site, word_end), enumerate which `resampled` steps
# have enough trials of each `behavior_dummy_forced` class for within-step
# perceptual contrasts. Drives Group B plotting (JON-43).
#
# See `docs/superpowers/plans/2026-05-19-causal46-trial-balance-index.md`
# and Linear JON-42.

# %%
from __future__ import annotations

from pathlib import Path

import mne
import polars as pl

from src.data import add_metadata_features, get_ambiguous_resampled_steps

# %%
REPO = Path(".").resolve()
EPOCH_DIR = REPO / "outputs/epochs_preprocessed"
OUT_DIR = REPO / "outputs/causal46_joined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical acoustic sites: causal6 foldmean-maxstat peaks, FDR-significant.
# Produced by acoustic_decoding_peaks_aggregate rule in causal6.Snakefile.
CAUSAL6_PEAKS = REPO / "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"

# K=5 is the recommended default; K=4 is the permissive companion for
# borderline subjects; K=10 is retained as a strict-tail sanity column.
THRESHOLDS = (4, 5, 10)

# %% [markdown]
# ## Load canonical sites and discover needed subjects

# %%
_peaks_raw = pl.read_parquet(CAUSAL6_PEAKS)
if "significant" in _peaks_raw.columns:
    canonical = _peaks_raw.filter(pl.col("significant"))
else:
    # Fallback for parquets produced before significance_aggregate added the column.
    canonical = _peaks_raw.filter(pl.col("p_value") < 0.05)
    print("⚠ no `significant` column — falling back to p_value < 0.05 (uncorrected)")

needed_subjects = sorted(canonical["subject"].unique().to_list())
print(f"Canonical sites: {canonical.height}  across {len(needed_subjects)} subjects")
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
)
for k in THRESHOLDS:
    counts = counts.with_columns(
        (
            (pl.col("min_class") >= k)
            & (~pl.col("resampled").is_in([1, 6]))
        ).alias(f"meets_threshold_{k}")
    )
counts = counts.sort(["subject", "phoneme_pair", "word_end", "resampled"])

print(f"counts table: {counts.height} rows")
print(counts.head(12))

counts.write_csv(OUT_DIR / "trial_counts_by_subject.csv")
print(f"Written: {OUT_DIR / 'trial_counts_by_subject.csv'}")

# %%
# How does qualifying-step count vary by threshold?
for k in (4, 5, 10):
    qual = (
        counts.filter(pl.col(f"meets_threshold_{k}"))
        .group_by(["subject", "phoneme_pair", "word_end"])
        .agg(
            pl.col("resampled").sort().alias("qualifying_steps"),
            pl.len().alias("n_qualifying"),
        )
        .sort(["subject", "phoneme_pair", "word_end"])
    )
    print(f"\n=== threshold K={k} ===")
    print(f"(subject, pp, word_end) tuples with ≥1 qualifying step: {qual.height}")
    print(f"distribution of n_qualifying:")
    print(qual.group_by("n_qualifying").len().sort("n_qualifying"))

# %% [markdown]
# ## Cross with canonical AS sites
#
# canonical_AS_sites.csv is keyed by (subject, electrode_idx, phoneme_pair) —
# no word_end. Join with the (word_end, resampled) tuples observed in
# metadata for each (subject, phoneme_pair); the natural join automatically
# enumerates both word_ends per phoneme_pair plus every resampled step.

# %%
# Reduce canonical to the keys we need; drop columns that conflict downstream.
canonical_keys = canonical.select(["subject", "electrode_idx", "phoneme_pair"])

# Join key is (subject, phoneme_pair). counts has the (word_end, resampled)
# dimensions we need; broadcasts across electrodes.
trial_balance = (
    canonical_keys
    .join(counts, on=["subject", "phoneme_pair"], how="inner")
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end", "resampled"])
)

print(f"trial_balance: {trial_balance.height} rows "
      f"(canonical {canonical.height} × ~12 per site = ~{canonical.height * 12})")

trial_balance.write_csv(OUT_DIR / "trial_balance_index.csv")
print(f"Written: {OUT_DIR / 'trial_balance_index.csv'}")

# %%
def _step_str(steps_list):
    return ",".join(str(s) for s in steps_list)

# Start from all (site, word_end) combinations so empty-qualifying ones still appear.
all_site_we = trial_balance.select(
    ["subject", "electrode_idx", "phoneme_pair", "word_end"]
).unique()

summary = all_site_we
for k in THRESHOLDS:
    grouped = (
        trial_balance.filter(pl.col(f"meets_threshold_{k}"))
        .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
        .agg(pl.col("resampled").sort().alias(f"qualifying_steps_{k}"))
    )
    summary = summary.join(
        grouped,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="left",
    )

summary = summary.with_columns(
    pl.col("qualifying_steps_5")
    .list.len().fill_null(0).cast(pl.Int64)
    .alias("n_qualifying_5"),
    *[
        pl.col(f"qualifying_steps_{k}")
        .map_elements(lambda lst: _step_str(lst) if lst is not None else "", return_dtype=pl.Utf8)
        .alias(f"qualifying_steps_{k}")
        for k in THRESHOLDS
    ],
).sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])

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

# Replicate the oracle's logic: min_class > 2, exclude endpoints.
ours_df = (
    counts.filter(
        (pl.col("min_class") > 2) & (~pl.col("resampled").is_in([1, 6]))
    )
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
#   (canonical_site, word_end, resampled). Most plotting code wants this:
#   filter by `meets_threshold_K` for whatever K the plot tolerates.
# - `trial_balance_summary.csv` — compact per-(site, word_end) view.
#   Faster to skim by eye.
# - `trial_counts_by_subject.csv` — electrode-agnostic raw counts.
#   Useful for sanity checks but not for plotting.
#
# Recommended default for Group B (JON-43): use `meets_threshold_5`.
# For real perceptual responses K=4–5 is usually sufficient. Drop to K=4 if
# K=5 leaves too many (site, word_end) tuples with zero qualifying steps;
# raise to K=10 only if trace SEMs at K=5 look implausibly tight.

# %%
n_sites = canonical.height
for k in (4, 5, 10):
    col = f"qualifying_steps_{k}"
    n_any = summary.filter(pl.col(col).str.len_chars() > 0).height
    n_two_plus = summary.filter(
        pl.col(col).str.count_matches(",") >= 1
    ).height
    print(
        f"K={k:2d}: (site×word_end) with ≥1 qualifying step: {n_any:4d}/{summary.height}; "
        f"with ≥2 qualifying steps: {n_two_plus:4d}/{summary.height}"
    )
