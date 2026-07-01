# causal4/causal6 Within-Completion Trial-Balance Index — Implementation Plan

> **ℹ Sampling rule now canonical elsewhere (2026-07-01).** This plan's
> cell/threshold definitions (`min_class`, `meets_threshold_K`,
> `is_ambiguous_step`) are still current. The downstream B3/B4 bootstrap that
> consumes them is defined canonically in
> `notebooks/causal46_joined/_within_completion.py` (module docstring); pointer
> + consumer map at
> `docs/superpowers/plans/2026-07-01-causal46-within-completion-subsampling.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Linear:** [JON-42](https://linear.app/jonlab/issue/JON-42/canonical-as-site-list-within-completion-trial-balance-index), sub-task **A2**. Depends on A1's `outputs/causal46_joined/canonical_AS_sites.csv`.

**Goal:** For every (canonical AS site, word_end) pair, enumerate which `resampled` steps have ≥ K trials of each `behavior_dummy_forced` class, so downstream Group B plotting (JON-43) can pick the within-completion controlled steps that aren't sample-size starved.

**Architecture:** One Jupytext percent notebook at `notebooks/causal46_joined/trial_balance_index.py`. Loads canonical CSV + per-subject epoch metadata, computes per-(subject, phoneme_pair, word_end, resampled) class counts once, broadcasts those counts across the canonical sites of each subject, and writes two CSVs (long + summary).

**Tech Stack:** Python, polars, mne (metadata only, no data load), jupytext. Local execution via `./.venv` (no GPU). All paths assume repo root `/Users/jon/Projects/barakeet`.

---

## Context

- A1 produced `canonical_AS_sites.csv` with columns `subject, electrode_idx, phoneme_pair, smin, smax, peak_auc, p_value, bucket`. **No `word_end` column** — acoustic decoding pools across completions and only uses resampled ∈ {1, 6}.
- Group B perceptual analyses operate **within word_end** on ambiguous resampled steps where behavior varies trial-to-trial. Plotting two HGA traces (heard /d/ vs /n/) for a given (site, word_end, resampled) needs enough trials of each class — for real responses K ≈ 4–5 is usually sufficient (trace SEMs are coarse but interpretable); push to K ≈ 10 only if SEMs at K=5 look implausibly tight.
- `src/data.py:get_ambiguous_resampled_steps` already implements this exact check with `ambiguous_response_threshold=2`. A2 reuses the same counting logic but exposes the per-step counts (not just the qualifying set) so downstream plots can drop in different thresholds without re-running this notebook.

### Definitions

For each (subject, phoneme_pair, word_end, resampled) tuple:

- `n_class0` = count of trials with `behavior_dummy_forced == 0`
- `n_class1` = count of trials with `behavior_dummy_forced == 1`
- `min_class` = min(n_class0, n_class1)
- `meets_threshold_K` = (min_class >= K), for K ∈ {4, 5, 10}

`min_class` is the relevant statistic — a step with 50 of class 0 and 3 of class 1 is unusable for a within-step contrast.

`resampled ∈ {1, 6}` (endpoints) typically have `min_class == 0` (one class only). They are included in the long output with `meets_threshold_*` = False so the schema is uniform; downstream code can filter as needed.

---

## File Structure

- **Create:** `notebooks/causal46_joined/trial_balance_index.py` (main deliverable)
- **Create:** `outputs/causal46_joined/trial_counts_by_subject.csv` (intermediate; one row per (subject, pp, word_end, resampled))
- **Create:** `outputs/causal46_joined/trial_balance_index.csv` (final long-format; one row per (canonical site, word_end, resampled))
- **Create:** `outputs/causal46_joined/trial_balance_summary.csv` (per (canonical site, word_end), qualifying-step lists as comma-joined strings)

---

## Inputs

### A1 canonical sites
- **Path:** `outputs/causal46_joined/canonical_AS_sites.csv`
- **Schema:** `subject, electrode_idx, phoneme_pair, smin, smax, peak_auc, p_value, bucket`

### Per-subject epoch metadata
- **Path:** `outputs/epochs_preprocessed/{subject}_epo.fif`
- Load with `mne.read_epochs(path, preload=False)` — metadata is parsed without loading raw data, so this is fast.
- Apply `src.data.add_metadata_features` to get `behavior_dummy_forced`, `phoneme_pair`, `word_end`, `resampled`.

### Existing reference function
- `src.data.get_ambiguous_resampled_steps(all_md, ambiguous_response_threshold=...)` — used as a verification oracle in Task 5.

---

## Outputs

### `trial_counts_by_subject.csv` (intermediate, electrode-agnostic)

Columns: `subject (str), phoneme_pair (str), word_end (str), resampled (i64), n_class0 (i64), n_class1 (i64), n_total (i64), min_class (i64), meets_threshold_4 (bool), meets_threshold_5 (bool), meets_threshold_10 (bool)`

One row per actual (subject, pp, word_end, resampled) tuple in metadata. Restricted to subjects present in canonical CSV (no point computing for unused subjects).

### `trial_balance_index.csv` (final long-format)

Columns: canonical-site keys + `word_end (str), resampled (i64), n_class0, n_class1, n_total, min_class, meets_threshold_4/5/10`.

One row per (canonical site × word_end × resampled). Sites where `phoneme_pair`'s two word_ends are e.g. {`desolate`, `necessary`} produce 2 word_ends × 6 resampled = 12 rows per site. ~N_canonical × 12 rows total.

### `trial_balance_summary.csv` (per (site, word_end) compact view)

Columns: canonical-site keys + `word_end, qualifying_steps_4 (str), qualifying_steps_5 (str), qualifying_steps_10 (str), n_qualifying_5 (i64)`.

`qualifying_steps_*` is comma-joined list of resampled steps where `meets_threshold_*` is True (e.g. `"2,3,4,5"`). Easier for downstream notebooks than re-aggregating the long file.

---

## Notes on TDD

Analysis notebook — no test suite. Verification is via (a) sanity counts printed at each section, (b) cross-check against `get_ambiguous_resampled_steps` at threshold=2 (Task 5).

---

## Tasks

### Task 1: Scaffold notebook

- [ ] **Step 1.1: Create the file**

Create `notebooks/causal46_joined/trial_balance_index.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Within-completion trial-balance index
#
# For every (canonical AS site, word_end), enumerate which `resampled` steps
# have enough trials of each `behavior_dummy_forced` class for within-step
# perceptual contrasts. Drives Group B plotting (JON-43).

# %%
from __future__ import annotations

from pathlib import Path

import mne
import polars as pl

from src.data import add_metadata_features, get_ambiguous_resampled_steps

# %%
REPO = Path("/Users/jon/Projects/barakeet")
EPOCH_DIR = REPO / "outputs/epochs_preprocessed"
OUT_DIR = REPO / "outputs/causal46_joined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CSV = OUT_DIR / "canonical_AS_sites.csv"

# K=5 is the recommended default; K=4 is the permissive companion for
# borderline subjects; K=10 is retained as a strict-tail sanity column.
THRESHOLDS = (4, 5, 10)
```

- [ ] **Step 1.2: Smoke-test as jupytext**

```bash
./.venv/bin/jupytext --to notebook --output /dev/stdout notebooks/causal46_joined/trial_balance_index.py | head -20
```

Expected: prints valid `.ipynb` JSON header.

- [ ] **Step 1.3: Commit**

```bash
git add notebooks/causal46_joined/trial_balance_index.py
git commit -m "scaffold trial_balance_index notebook"
```

---

### Task 2: Build per-subject metadata frame

- [ ] **Step 2.1: Append load section**

```python
# %% [markdown]
# ## Load canonical sites and discover needed subjects

# %%
canonical = pl.read_csv(CANONICAL_CSV)
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
```

- [ ] **Step 2.2: Sanity-check schema**

```python
# %%
# Quick sanity: assert types are clean, no nulls in the columns we depend on.
assert all_md["behavior_dummy_forced"].is_in([0, 1]).all(), "expected behavior_dummy_forced ∈ {0, 1}"
assert all_md["resampled"].is_in([1, 2, 3, 4, 5, 6]).all(), "expected resampled ∈ {1..6}"
assert all_md.filter(pl.col("word_end").is_null()).height == 0, "found nulls in word_end"
```

Why the asserts: A2's whole correctness depends on the metadata being clean at this stage. A null `word_end` (e.g. catch trials, if any exist) would silently broadcast across both word_ends in Task 4 and inflate counts. Fail loud here.

- [ ] **Step 2.3: Commit**

```bash
git add notebooks/causal46_joined/trial_balance_index.py
git commit -m "load per-subject metadata for trial-balance index"
```

---

### Task 3: Compute per-(subject, pp, word_end, resampled) class counts

- [ ] **Step 3.1: Append counts section**

```python
# %% [markdown]
# ## Per-(subject, pp, word_end, resampled) class counts
#
# Independent of electrode_idx — these counts are subject-level and broadcast
# across all electrodes of the same subject in Task 4.

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
        (pl.col("min_class") >= k).alias(f"meets_threshold_{k}")
    )
counts = counts.sort(["subject", "phoneme_pair", "word_end", "resampled"])

print(f"counts table: {counts.height} rows")
print(counts.head(12))

counts.write_csv(OUT_DIR / "trial_counts_by_subject.csv")
print(f"Written: {OUT_DIR / 'trial_counts_by_subject.csv'}")
```

- [ ] **Step 3.2: Diagnostic — qualifying-step summary at K=5 and K=10**

```python
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
```

Expected shape at K=5: many `n_qualifying ∈ {2, 3, 4}` (steps 3, 4 and sometimes 2, 5 qualify; 1 and 6 almost never qualify). If most tuples show 0 qualifying steps even at K=4, the subject's behavioral split may be too one-sided to support within-step contrasts — surface this diagnostic before downstream plotting decisions.

- [ ] **Step 3.3: Commit**

```bash
git add notebooks/causal46_joined/trial_balance_index.py outputs/causal46_joined/trial_counts_by_subject.csv
git commit -m "compute per-subject trial-balance counts"
```

---

### Task 4: Cross with canonical sites → long-format trial_balance_index.csv

- [ ] **Step 4.1: Append cross section**

```python
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
```

- [ ] **Step 4.2: Per-(site, word_end) summary**

```python
# %%
def _step_str(steps_list):
    return ",".join(str(s) for s in steps_list)


# Start from all (site, word_end) combinations (so empty-qualifying ones still appear)
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
    *[
        pl.col(f"qualifying_steps_{k}")
        .map_elements(lambda lst: _step_str(lst) if lst is not None else "", return_dtype=pl.Utf8)
        .alias(f"qualifying_steps_{k}")
        for k in THRESHOLDS
    ],
    pl.col("qualifying_steps_5")
    .map_elements(lambda s: 0 if not s else len(s.split(",")), return_dtype=pl.Int64)
    .alias("n_qualifying_5"),
).sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])

summary.write_csv(OUT_DIR / "trial_balance_summary.csv")
print(f"Written: {OUT_DIR / 'trial_balance_summary.csv'}")
print(summary.head(10))
```

- [ ] **Step 4.3: Sanity check — row arithmetic**

```python
# %%
# Sites × word_ends_per_pp × resampled_steps_observed should equal trial_balance.height
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
```

- [ ] **Step 4.4: Commit**

```bash
git add notebooks/causal46_joined/trial_balance_index.py outputs/causal46_joined/trial_balance_index.csv outputs/causal46_joined/trial_balance_summary.csv
git commit -m "build trial-balance index across canonical sites"
```

---

### Task 5: Verify against `get_ambiguous_resampled_steps`

The existing helper does the same count at threshold=2 and returns the qualifying set per (subject, pp, word_end). Our threshold=2 (recomputed for verification) must agree exactly.

- [ ] **Step 5.1: Append verification section**

```python
# %% [markdown]
# ## Cross-check against src.data.get_ambiguous_resampled_steps
#
# That helper computes the same qualifying-step set at threshold=2 and
# excludes endpoints 1 and 6. We don't emit a `meets_threshold_2` column in
# the final outputs, but we recompute it here for the equivalence check —
# anything other than exact agreement is a counting bug.

# %%
ref = get_ambiguous_resampled_steps(all_md, ambiguous_response_threshold=2)
# ref schema: subject, phoneme_pair, word_end, resampled (list[int])

ours = (
    counts.filter(
        (pl.col("min_class") >= 2) & (~pl.col("resampled").is_in([1, 6]))
    )
    .group_by(["subject", "phoneme_pair", "word_end"])
    .agg(pl.col("resampled").sort().alias("resampled"))
)

merged = ref.rename({"resampled": "_ref"}).join(
    ours.rename({"resampled": "_ours"}),
    on=["subject", "phoneme_pair", "word_end"], how="full",
)

mismatches = merged.filter(
    (pl.col("_ref") != pl.col("_ours"))
    | pl.col("_ref").is_null()
    | pl.col("_ours").is_null()
)
assert mismatches.height == 0, (
    f"counts disagree with get_ambiguous_resampled_steps on "
    f"{mismatches.height} (subject, pp, word_end) tuples:\n{mismatches}"
)
print(f"✓ matches get_ambiguous_resampled_steps on all "
      f"{ref.height} (subject, pp, word_end) tuples at threshold=2")
```

This is a hard equivalence assert — if it fails, the bug is in this notebook (or in `get_ambiguous_resampled_steps`), not in the data. Don't silently relax it.

- [ ] **Step 5.2: Commit**

```bash
git add notebooks/causal46_joined/trial_balance_index.py
git commit -m "verify trial-balance index matches get_ambiguous_resampled_steps"
```

---

### Task 6: Reviewer summary

- [ ] **Step 6.1: Append summary section**

```python
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
```

- [ ] **Step 6.2: Final end-to-end run**

```bash
./.venv/bin/jupytext --execute --to notebook --output - notebooks/causal46_joined/trial_balance_index.py > /tmp/tb_exec.ipynb 2>&1 || echo "EXECUTION FAILED"
```

Verify outputs exist:

```bash
ls -la outputs/causal46_joined/trial_counts_by_subject.csv outputs/causal46_joined/trial_balance_index.csv outputs/causal46_joined/trial_balance_summary.csv
```

- [ ] **Step 6.3: Commit**

```bash
git add notebooks/causal46_joined/trial_balance_index.py
git commit -m "add reviewer summary to trial-balance index notebook"
```

---

## Acceptance criteria

1. `notebooks/causal46_joined/trial_balance_index.py` runs end-to-end via `jupytext --execute` with no errors.
2. `outputs/causal46_joined/` contains: `trial_counts_by_subject.csv`, `trial_balance_index.csv`, `trial_balance_summary.csv`.
3. Row-arithmetic check in Step 4.3 passes.
4. Equivalence check against `get_ambiguous_resampled_steps` at threshold=2 (Step 5.1) passes exactly — zero mismatching tuples.
5. `trial_balance_index.csv` row count = `canonical_AS_sites.csv` row count × (word_ends × resampled_steps_observed_per_subject_pp). For 6 resampled steps × 2 word_ends and N canonical sites: ~12N.

## Out of scope

- Choosing the production threshold. We emit `meets_threshold_{4,5,10}` so Group B notebooks can pick at consumption time; the recommended default is K=5.
- Re-running A1 for the 3 absent subjects (EC248, EC250, EC253). They were excluded from `canonical_AS_sites.csv` by A1; they're absent here for the same reason. When causal6 prod re-syncs them, re-run A1 first and then this notebook.
- Anything beyond `behavior_dummy_forced`. If a downstream analysis needs balance on `behavior_categorical` (which has a third "unsure" class), this notebook needs a new column, not a new threshold.
