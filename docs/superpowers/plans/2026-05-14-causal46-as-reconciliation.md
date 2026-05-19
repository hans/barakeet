# causal4/causal6 AS-Site Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Linear:** [JON-42](https://linear.app/jonlab/issue/JON-42/canonical-as-site-list-within-completion-trial-balance-index) — covers the AS-site half of that issue only. The "within-completion trial-balance index" task remains in JON-42 and consumes `canonical_AS_sites.csv` produced here as its input.

**Goal:** Build a single notebook that compares causal4's acoustic-selectivity (AS) screen against causal6's NHST-based acoustic-significance calls, classifies every evaluated electrode-site into one of four buckets (`both`, `causal4_only`, `causal6_only_eligible`, `causal6_only_newly_eligible`), produces summary statistics and three star-plot PDFs for visual inspection of gains and losses, and writes a canonical AS-site CSV that downstream Group B/C analyses will read.

**Architecture:** A single Jupytext percent-format notebook at `notebooks/causal46_joined/as_reconciliation.py`. Loads two source parquets, joins on `(subject, electrode_idx, phoneme_pair)`, classifies sites, prints/plots summaries, then renders three PDFs by reusing the existing star-plot helper from `notebooks/causal6/view_provisional_results.py` (extracted to `src/viz_provisional.py` as a side task so it can be imported cleanly).

**Tech Stack:** Python, polars, pandas (only where required for legacy interop), matplotlib, MNE-Python, jupytext. Local execution via `./.venv` (no GPU needed). All paths assume the repo root `/Users/jon/Projects/barakeet`.

---

## Context

The project is pivoting back to causal4's STG-anchored "AS electrodes show behavioral predictivity" story (see strategy summary in conversation history). The first step is to define the canonical AS-site list. The user has decided causal6's acoustic NHST is strictly better than causal4's AUC≥0.65 threshold and wants to adopt it, but needs to visually inspect what is *lost* (causal4-AS sites that fail causal6 NHST) and *gained* (causal6-significant sites missing from causal4's list) before signing off.

### Bucket definitions

For every `(subject, electrode_idx, phoneme_pair)` tuple ever evaluated by either pipeline:

| Bucket | causal4 evaluated? | causal4_AS (AUC≥0.65)? | causal6 p<0.05? | Interpretation |
|---|---|---|---|---|
| `both` | yes | **yes** | **yes** | Both methods agree → safest |
| `causal4_only` | yes | **yes** | no | LOSS — causal4 called AS, causal6 NHST rejects |
| `causal6_only_eligible` | yes | no | **yes** | GAIN — causal4 evaluated and missed |
| `causal6_only_newly_eligible` | no | (n/a) | **yes** | GAIN — outside causal4's pre-screened pool |

`causal4_eligible` is determined by membership in `phon_roc_auc_searchlight_df.parquet` (causal4 evaluated 411 site-tuples in this file; the 257-row `phon_peaks_df.parquet` is the AUC≥0.65 subset).

### Subject coverage caveat

causal4 phon_peaks_df has 10 subjects: `EC243, EC248, EC250, EC253, EC260, EC270, EC278, EC279, EC282, EC287`. causal6 `causal6_speech_responsive_pipeline` prod currently has 7: `EC243, EC260, EC270, EC278, EC279, EC282, EC287`. The 3 missing subjects (`EC248, EC250, EC253`) cannot be reconciled and must be excluded with a printed warning. Do not silently drop them.

---

## File Structure

- **Create:** `notebooks/causal46_joined/as_reconciliation.py` (Jupytext percent-format notebook — main deliverable)
- **Create:** `src/viz_provisional.py` (extracted star-plot helper module)
- **Modify:** `notebooks/causal6/view_provisional_results.py` (replace inline `_provisional_star_plot` with import from `src.viz_provisional`)
- **Create:** `outputs/causal46_joined/reconciliation.parquet` (full classification table — written by the notebook)
- **Create:** `outputs/causal46_joined/canonical_AS_sites.csv` (final sign-off list — written by the notebook)
- **Create:** `outputs/causal46_joined/losses.pdf`, `gains_eligible.pdf`, `gains_newly_eligible.pdf` (three star-plot galleries)
- **Create:** `outputs/causal46_joined/both.pdf` (a small sanity-check sample of agreement sites)

---

## Inputs (concrete paths and schemas)

### causal4 AS list — sites that passed AUC≥0.65 filter
- **Path:** `~/freesurfer_subjects/barakeet/causal4_pipeline/prepare_neurometrics/phon_peaks_df.parquet`
- **Schema:** `subject (Enum), electrode_idx (i64), phoneme_pair (Enum), smin (i64), smax (i64), phon_roc_auc (f64), word_end_offset_sample (f64)`
- **Shape:** 257 rows, pre-filtered to `phon_roc_auc >= 0.65`
- **Subjects:** EC243, EC248, EC250, EC253, EC260, EC270, EC278, EC279, EC282, EC287

### causal4 eligibility pool — every site causal4 attempted to decode
- **Path:** `~/freesurfer_subjects/barakeet/causal4_pipeline/prepare_neurometrics/phon_roc_auc_searchlight_df.parquet`
- **Schema:** `subject, electrode_idx, phoneme_pair, smin, smax, fold (i64), phon_roc_auc (f64)` — fold-level
- **Shape:** 119190 rows; 411 unique `(subject, electrode_idx, phoneme_pair)` tuples
- **Use:** Get the set of evaluated tuples — `df.select(['subject','electrode_idx','phoneme_pair']).unique()`

### causal6 acoustic NHST — per-subject phon_peaks.parquet
- **Path glob:** `outputs_prod/acoustic_decoding_peaks/{subject}/phon_peaks.parquet`
  (`outputs_prod` is a symlink to `~/freesurfer_subjects/barakeet/causal6_speech_responsive_pipeline/`)
- **Schema:** `subject (str), electrode_idx (i64), phoneme_pair (str), smin (i64), smax (i64), test_roc_auc (f32), pointwise_p (f64), T_obs (f64), p_value (f64), n_permutations (u32), null_q05/q50/q95/q99 (f64)`
- **Criterion:** `p_value < 0.05` (already maxstat-corrected within subject; do not re-correct)
- **Subjects present:** EC243, EC260, EC270, EC278, EC279, EC282, EC287 (7 of 10)

### Epochs (for star plots)
- **Path:** `outputs/epochs_preprocessed/{subject}_epo.fif`
- Loaded via MNE; metadata enriched via `src.data.add_metadata_features`
- Loading pattern: see `notebooks/causal6/view_provisional_results.py` around the `epochs_dict` construction near line 1167+

### Ambiguous-step lookup (for star plot middle panel)
- Use `src.data.get_ambiguous_resampled_steps` as imported in `view_provisional_results.py:1171`

---

## Outputs

All under `outputs/causal46_joined/`:

- `reconciliation.parquet` — one row per `(subject, electrode_idx, phoneme_pair)` ever seen by either pipeline. Columns:
  - Keys: `subject (str), electrode_idx (i64), phoneme_pair (str)`
  - causal4 fields: `causal4_eligible (bool), causal4_AS (bool), causal4_peak_auc (f64 | null), causal4_smin (i64 | null), causal4_smax (i64 | null)`
  - causal6 fields: `causal6_AS (bool), causal6_test_roc_auc (f32 | null), causal6_p_value (f64 | null), causal6_n_perm (u32 | null), causal6_smin (i64 | null), causal6_smax (i64 | null)`
  - Derived: `bucket (str)` ∈ {`both`, `causal4_only`, `causal6_only_eligible`, `causal6_only_newly_eligible`, `neither_evaluated_pos`}
- `canonical_AS_sites.csv` — final selected sites for downstream use. Initially populated as `bucket in {'both', 'causal6_only_eligible', 'causal6_only_newly_eligible'}` (i.e., the causal6_AS set). Columns: `subject, electrode_idx, phoneme_pair, smin, smax, peak_auc, p_value`. The user may overwrite this manually after inspection.
- `losses.pdf` — star plots for `bucket == 'causal4_only'`, sorted by `causal4_peak_auc` descending.
- `gains_eligible.pdf` — star plots for `bucket == 'causal6_only_eligible'`, sorted by `causal6_test_roc_auc` descending.
- `gains_newly_eligible.pdf` — star plots for `bucket == 'causal6_only_newly_eligible'`, sorted by `causal6_test_roc_auc` descending.
- `both.pdf` — random 10-site sample from `bucket == 'both'` for sanity-checking.

---

## Notes on TDD

This is an analysis notebook, not production code. Follow the project's notebook conventions (no test suite for analysis notebooks). Verification is via running the notebook end-to-end and visually inspecting outputs at each checkpoint. The one piece that gets a unit-style test is the bucket-classification function — small, pure, easy to test, and a wrong classification poisons every downstream output.

---

## Tasks

### Task 1: Extract `_provisional_star_plot` to `src/viz_provisional.py`

**Why:** The helper currently lives inside `notebooks/causal6/view_provisional_results.py` as `_provisional_star_plot`. Importing from a Jupytext notebook with top-level data-loading code is fragile. Extract to a real module so both notebooks can import cleanly. (Per user feedback: extract shared helpers, don't copy.)

**Files:**
- Create: `src/viz_provisional.py`
- Modify: `notebooks/causal6/view_provisional_results.py` (replace inline def + adjust imports)

- [ ] **Step 1.1: Inspect the function**

Read `notebooks/causal6/view_provisional_results.py:1359-1517`. Note the function signature, what it imports (`add_textgrid` from `src.viz_paper`, `OFFSET_DICT` from `src.stimuli`, etc.), and what module-level constants it depends on (`EPOCH_TMIN`, `EPOCH_SFREQ`, `_TEXTGRID_DIR` defined nearby).

- [ ] **Step 1.2: Create `src/viz_provisional.py`**

Move the function verbatim, but with these adjustments:
- Top-level imports (no inline imports). Pull `add_textgrid` from `src.viz_paper`, `OFFSET_DICT` from `src.stimuli` (use real name, not the `_OFFSET_DICT` alias).
- Take `EPOCH_TMIN`, `EPOCH_SFREQ`, `textgrid_dir` as parameters with sensible defaults (`EPOCH_TMIN=-0.4`, `EPOCH_SFREQ=100.0`, `textgrid_dir="data/stimuli/textgrid"`).
- Rename to `provisional_star_plot` (drop leading underscore — it's a public helper now).

Write the file:

```python
"""Three-panel HGA star plot for site-level inspection.

Top: unambiguous trials (resampled 1 vs 6), with acoustic window shaded.
Middle: within-completion controlled ambiguous trials, split by response.
Bottom: all trials within word_end split by response (decoder view).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.stimuli import OFFSET_DICT
from src.viz_paper import add_textgrid


def provisional_star_plot(
    subject,
    electrode_idx,
    phoneme_pair,
    word_end,
    epochs_dict,
    ambig_steps,
    phon_smin=None,
    phon_smax=None,
    behav_smin=None,
    behav_smax=None,
    textgrid_dir="data/stimuli/textgrid",
    epoch_tmin=-0.4,
    epoch_sfreq=100.0,
    figsize=(6.5, 7.5),
    acoustic_peak_auc=None,
    acoustic_peak_auc_pct=None,
    behav_full_peak_diff=None,
    behav_full_peak_diff_pct=None,
    behav_hga_peak_auc=None,
    behav_hga_peak_auc_pct=None,
):
    # [paste body from view_provisional_results.py:1391-1517 verbatim,
    #  except replace _OFFSET_DICT with OFFSET_DICT and use the
    #  epoch_tmin / epoch_sfreq parameters in place of module constants]
    ...
```

Concretely: copy lines 1391-1517 of the original, then in the pasted body change `_OFFSET_DICT` → `OFFSET_DICT` (one site at the bottom, `ax_top.set_xlim` call).

- [ ] **Step 1.3: Update `view_provisional_results.py` to import from the new module**

Around line 1359 in `notebooks/causal6/view_provisional_results.py`, delete the inline `def _provisional_star_plot(...)` (lines 1359-1517). Add at the top of section 8 (near line 1167's other imports):

```python
from src.viz_provisional import provisional_star_plot as _provisional_star_plot
```

(Alias preserves the existing call sites without rewriting them.)

- [ ] **Step 1.4: Smoke-test the import**

Run from the repo root:

```bash
./.venv/bin/python -c "from src.viz_provisional import provisional_star_plot; print(provisional_star_plot)"
```

Expected: prints the function repr, no ImportError. If it fails, fix the import path and re-run.

- [ ] **Step 1.5: Commit**

```bash
git add src/viz_provisional.py notebooks/causal6/view_provisional_results.py
git commit -m "extract provisional_star_plot to src/viz_provisional"
```

---

### Task 2: Scaffold the reconciliation notebook

**Files:**
- Create: `notebooks/causal46_joined/__init__.py` (empty, optional — only if needed)
- Create: `notebooks/causal46_joined/as_reconciliation.py`

- [ ] **Step 2.1: Create directory**

```bash
mkdir -p notebooks/causal46_joined outputs/causal46_joined
```

- [ ] **Step 2.2: Write notebook header**

Create `notebooks/causal46_joined/as_reconciliation.py` with:

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
# # causal4/causal6 AS-site reconciliation
#
# Classifies every (subject, electrode_idx, phoneme_pair) tuple evaluated by
# either pipeline into one of four buckets, then renders summary stats and
# three star-plot PDFs (losses, gains-eligible, gains-newly-eligible) for
# visual inspection. Final canonical AS-site list is written to
# `outputs/causal46_joined/canonical_AS_sites.csv`.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

# %%
HOME = Path(os.path.expanduser("~"))
REPO = Path("/Users/jon/Projects/barakeet")
CAUSAL4_DIR = HOME / "freesurfer_subjects/barakeet/causal4_pipeline/prepare_neurometrics"
CAUSAL6_DIR = REPO / "outputs_prod/acoustic_decoding_peaks"
OUT_DIR = REPO / "outputs/causal46_joined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSAL4_AUC_THRESHOLD = 0.65
CAUSAL6_P_THRESHOLD = 0.05
```

- [ ] **Step 2.3: Verify the notebook opens as a jupytext file**

```bash
./.venv/bin/jupytext --to notebook --output /dev/stdout notebooks/causal46_joined/as_reconciliation.py | head -20
```

Expected: prints valid `.ipynb` JSON header (a `"cells"` key, etc.) with no errors.

- [ ] **Step 2.4: Commit**

```bash
git add notebooks/causal46_joined/as_reconciliation.py
git commit -m "scaffold causal46_joined/as_reconciliation notebook"
```

---

### Task 3: Load + classify — build the reconciliation table

**Files:**
- Modify: `notebooks/causal46_joined/as_reconciliation.py` (append sections)
- Create: `outputs/causal46_joined/reconciliation.parquet` (written by notebook)

- [ ] **Step 3.1: Append load + classify section**

Add to the notebook:

```python
# %% [markdown]
# ## Load causal4 outputs

# %%
c4_AS = pl.read_parquet(CAUSAL4_DIR / "phon_peaks_df.parquet")
# Cast Enums to Utf8 for clean joins with causal6 (which uses str)
c4_AS = c4_AS.with_columns(
    pl.col("subject").cast(pl.Utf8),
    pl.col("phoneme_pair").cast(pl.Utf8),
).rename({
    "phon_roc_auc": "causal4_peak_auc",
    "smin": "causal4_smin",
    "smax": "causal4_smax",
}).drop("word_end_offset_sample")
assert (c4_AS["causal4_peak_auc"] >= CAUSAL4_AUC_THRESHOLD).all(), \
    "phon_peaks_df is supposed to be pre-filtered to AUC>=0.65"

c4_eligible = (
    pl.read_parquet(CAUSAL4_DIR / "phon_roc_auc_searchlight_df.parquet")
    .with_columns(
        pl.col("subject").cast(pl.Utf8),
        pl.col("phoneme_pair").cast(pl.Utf8),
    )
    .select(["subject", "electrode_idx", "phoneme_pair"])
    .unique()
)

print(f"causal4 AS sites: {c4_AS.shape[0]}")
print(f"causal4 evaluated tuples: {c4_eligible.shape[0]}")
print(f"causal4 subjects (in AS): {sorted(c4_AS['subject'].unique().to_list())}")
```

```python
# %% [markdown]
# ## Load causal6 outputs

# %%
c6_paths = sorted(CAUSAL6_DIR.glob("*/phon_peaks.parquet"))
c6_subjects_present = [p.parent.name for p in c6_paths]
print(f"causal6 subjects in prod: {c6_subjects_present}")

c6_all = pl.concat([pl.read_parquet(p) for p in c6_paths])
c6_all = c6_all.rename({
    "test_roc_auc": "causal6_test_roc_auc",
    "p_value": "causal6_p_value",
    "n_permutations": "causal6_n_perm",
    "smin": "causal6_smin",
    "smax": "causal6_smax",
}).select([
    "subject", "electrode_idx", "phoneme_pair",
    "causal6_test_roc_auc", "causal6_p_value", "causal6_n_perm",
    "causal6_smin", "causal6_smax",
])
print(f"causal6 evaluated tuples: {c6_all.shape[0]}")
print(f"causal6 significant (p<0.05): {int((c6_all['causal6_p_value'] < CAUSAL6_P_THRESHOLD).sum())}")
```

```python
# %% [markdown]
# ## Subject coverage warning

# %%
c4_subj = set(c4_AS["subject"].unique().to_list())
c6_subj = set(c6_subjects_present)
missing_in_c6 = sorted(c4_subj - c6_subj)
if missing_in_c6:
    print(
        f"⚠ {len(missing_in_c6)} causal4 subjects absent from causal6 prod: "
        f"{missing_in_c6}. Their sites are excluded from reconciliation."
    )
    c4_AS = c4_AS.filter(~pl.col("subject").is_in(missing_in_c6))
    c4_eligible = c4_eligible.filter(~pl.col("subject").is_in(missing_in_c6))
```

```python
# %% [markdown]
# ## Build the reconciliation table

# %%
KEYS = ["subject", "electrode_idx", "phoneme_pair"]

# Universe = union of every tuple either pipeline evaluated (within reconcilable subjects).
universe = pl.concat([
    c4_eligible.select(KEYS),
    c6_all.select(KEYS),
]).unique()

recon = (
    universe
    .join(
        c4_eligible.with_columns(pl.lit(True).alias("causal4_eligible")),
        on=KEYS, how="left",
    )
    .with_columns(pl.col("causal4_eligible").fill_null(False))
    .join(
        c4_AS.with_columns(pl.lit(True).alias("causal4_AS")),
        on=KEYS, how="left",
    )
    .with_columns(pl.col("causal4_AS").fill_null(False))
    .join(c6_all, on=KEYS, how="left")
    .with_columns(
        (pl.col("causal6_p_value") < CAUSAL6_P_THRESHOLD)
            .fill_null(False)
            .alias("causal6_AS"),
    )
)


def assign_bucket(c4_elig: bool, c4_AS_: bool, c6_AS_: bool) -> str:
    if c4_AS_ and c6_AS_:
        return "both"
    if c4_AS_ and not c6_AS_:
        return "causal4_only"
    if c6_AS_ and c4_elig:
        return "causal6_only_eligible"
    if c6_AS_ and not c4_elig:
        return "causal6_only_newly_eligible"
    return "neither_AS"

recon = recon.with_columns(
    pl.struct(["causal4_eligible", "causal4_AS", "causal6_AS"])
      .map_elements(
          lambda s: assign_bucket(s["causal4_eligible"], s["causal4_AS"], s["causal6_AS"]),
          return_dtype=pl.Utf8,
      )
      .alias("bucket")
)

print("Bucket counts:")
print(recon.group_by("bucket").len().sort("len", descending=True))

recon.write_parquet(OUT_DIR / "reconciliation.parquet")
print(f"Written: {OUT_DIR / 'reconciliation.parquet'}  ({recon.shape[0]} rows)")
```

- [ ] **Step 3.2: Write a small classification unit test**

Create `tests/test_as_reconciliation.py`:

```python
"""Smoke test for the bucket-classification logic."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/causal46_joined"))

# Re-define the function here rather than executing the notebook.
def assign_bucket(c4_elig, c4_AS, c6_AS):
    if c4_AS and c6_AS:
        return "both"
    if c4_AS and not c6_AS:
        return "causal4_only"
    if c6_AS and c4_elig:
        return "causal6_only_eligible"
    if c6_AS and not c4_elig:
        return "causal6_only_newly_eligible"
    return "neither_AS"


def test_buckets():
    # Both flagged
    assert assign_bucket(True, True, True) == "both"
    # causal4_AS, causal6 rejects → loss
    assert assign_bucket(True, True, False) == "causal4_only"
    # causal4 evaluated and missed, causal6 picks up
    assert assign_bucket(True, False, True) == "causal6_only_eligible"
    # causal4 never evaluated, causal6 picks up
    assert assign_bucket(False, False, True) == "causal6_only_newly_eligible"
    # Neither flagged
    assert assign_bucket(True, False, False) == "neither_AS"
    # Sanity: c4_AS cannot be True if c4_eligible is False, but the function
    # should still classify deterministically if that ever happened.
    assert assign_bucket(False, True, True) == "both"


if __name__ == "__main__":
    test_buckets()
    print("OK")
```

- [ ] **Step 3.3: Run the test**

```bash
./.venv/bin/python tests/test_as_reconciliation.py
```

Expected output: `OK`. If `AssertionError`, fix `assign_bucket` and re-run.

- [ ] **Step 3.4: Execute the notebook through this section**

```bash
./.venv/bin/jupytext --execute --to notebook --output - notebooks/causal46_joined/as_reconciliation.py > /tmp/recon_exec.ipynb 2>&1 || echo "EXECUTION FAILED"
```

Then check `outputs/causal46_joined/reconciliation.parquet` exists and inspect:

```bash
./.venv/bin/python -c "
import polars as pl
df = pl.read_parquet('outputs/causal46_joined/reconciliation.parquet')
print('shape:', df.shape)
print('schema:', df.schema)
print()
print(df.group_by('bucket').len().sort('len', descending=True))
"
```

Expected: prints bucket counts. `both + causal4_only` should sum to (causal4 AS sites in reconcilable subjects). `both + causal6_only_eligible + causal6_only_newly_eligible` should sum to (causal6 significant sites in reconcilable subjects). Sanity-check these against the raw counts printed earlier in the notebook; if they don't add up, debug the joins before proceeding.

- [ ] **Step 3.5: Commit**

```bash
git add notebooks/causal46_joined/as_reconciliation.py tests/test_as_reconciliation.py
git commit -m "build causal46 reconciliation table"
```

---

### Task 4: Summary statistics and plots

**Files:** Modify `notebooks/causal46_joined/as_reconciliation.py`

- [ ] **Step 4.1: Append summary panels section**

```python
# %% [markdown]
# ## Summary panels
#
# - Bucket counts per subject and phoneme_pair
# - Loss audit: distribution of causal4 peak AUC for `causal4_only` sites
#   (high-AUC losses are the worrying ones)
# - Gain audit: distribution of causal6 corrected p-values for gain buckets
#   (borderline p-values are weaker gains)
# - Joint scatter: causal4 peak AUC × causal6 p-value, coloured by bucket

# %%
# Bucket × subject × phoneme_pair table
breakdown = (
    recon
    .group_by(["bucket", "subject", "phoneme_pair"])
    .len()
    .pivot(values="len", index=["subject", "phoneme_pair"], on="bucket")
    .fill_null(0)
    .sort(["subject", "phoneme_pair"])
)
print("Per-subject / phoneme_pair bucket breakdown:")
print(breakdown)
breakdown.write_csv(OUT_DIR / "bucket_breakdown.csv")
```

```python
# %%
# Loss audit: causal4 peak AUC distribution for causal4_only sites
losses = recon.filter(pl.col("bucket") == "causal4_only")
print(f"Losses: {losses.shape[0]} sites")
print(f"  causal4_peak_auc:  min={losses['causal4_peak_auc'].min():.3f}  "
      f"median={losses['causal4_peak_auc'].median():.3f}  "
      f"max={losses['causal4_peak_auc'].max():.3f}")
print(f"  count >= 0.70: {int((losses['causal4_peak_auc'] >= 0.70).sum())}  "
      f">= 0.75: {int((losses['causal4_peak_auc'] >= 0.75).sum())}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(
    losses["causal4_peak_auc"].to_numpy(),
    bins=20, color="tomato", alpha=0.8, edgecolor="k",
)
axes[0].axvline(0.65, color="k", lw=0.8, ls="--", label="causal4 threshold")
axes[0].set_xlabel("causal4 peak AUC")
axes[0].set_ylabel("losses (count)")
axes[0].set_title(f"Loss audit — {losses.shape[0]} sites")
axes[0].legend()

# Gain audit: causal6 p_value distribution for gain buckets
gains = recon.filter(pl.col("bucket").is_in(
    ["causal6_only_eligible", "causal6_only_newly_eligible"]
))
print(f"Gains: {gains.shape[0]} sites")
print(f"  eligible:        {(gains['bucket'] == 'causal6_only_eligible').sum()}")
print(f"  newly_eligible:  {(gains['bucket'] == 'causal6_only_newly_eligible').sum()}")

for bucket, color in [
    ("causal6_only_eligible", "steelblue"),
    ("causal6_only_newly_eligible", "seagreen"),
]:
    vals = gains.filter(pl.col("bucket") == bucket)["causal6_p_value"].to_numpy()
    axes[1].hist(vals, bins=np.linspace(0, 0.05, 21), alpha=0.6,
                 color=color, edgecolor="k", label=bucket)
axes[1].set_xlabel("causal6 corrected p-value")
axes[1].set_ylabel("gains (count)")
axes[1].set_title(f"Gain audit — {gains.shape[0]} sites")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "summary_audit.png", dpi=150)
plt.show()
```

```python
# %%
# Joint scatter: causal4 peak AUC × causal6 p-value
plot_df = recon.filter(
    pl.col("causal4_peak_auc").is_not_null()
    | pl.col("causal6_p_value").is_not_null()
)

fig, ax = plt.subplots(figsize=(7, 5))
bucket_colors = {
    "both": "#3a823a",
    "causal4_only": "#c44e4e",
    "causal6_only_eligible": "#4a78b8",
    "causal6_only_newly_eligible": "#2d8b8b",
    "neither_AS": "#999999",
}
for bucket, color in bucket_colors.items():
    sub = plot_df.filter(pl.col("bucket") == bucket)
    if sub.shape[0] == 0:
        continue
    # Missing values: use NaN sentinels for plotting on the axes
    x = sub["causal4_peak_auc"].fill_null(np.nan).to_numpy()
    y = sub["causal6_p_value"].fill_null(np.nan).to_numpy()
    ax.scatter(x, y, c=color, s=18, alpha=0.55, label=f"{bucket} (n={sub.shape[0]})",
               edgecolors="none")

ax.axvline(0.65, color="k", lw=0.6, ls="--", label="causal4 AUC threshold")
ax.axhline(0.05, color="k", lw=0.6, ls=":", label="causal6 p threshold")
ax.set_xlabel("causal4 peak AUC")
ax.set_ylabel("causal6 corrected p-value")
ax.set_yscale("log")
ax.set_title("causal4 vs causal6 — agreement & disagreement")
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(OUT_DIR / "summary_scatter.png", dpi=150)
plt.show()
```

- [ ] **Step 4.2: Execute the notebook through this section**

```bash
./.venv/bin/jupytext --execute --to notebook --output - notebooks/causal46_joined/as_reconciliation.py > /tmp/recon_exec.ipynb 2>&1 || echo "EXECUTION FAILED"
```

Verify:

```bash
ls -la outputs/causal46_joined/summary_audit.png outputs/causal46_joined/summary_scatter.png outputs/causal46_joined/bucket_breakdown.csv
```

Expected: all three files exist, non-zero size.

- [ ] **Step 4.3: Commit**

```bash
git add notebooks/causal46_joined/as_reconciliation.py
git commit -m "add reconciliation summary panels"
```

---

### Task 5: Star plot galleries

**Files:** Modify `notebooks/causal46_joined/as_reconciliation.py`

- [ ] **Step 5.1: Append epoch-loading section**

```python
# %% [markdown]
# ## Star plot galleries — visual inspection
#
# Three PDFs:
#   - losses.pdf:                bucket == "causal4_only", sort by causal4 peak AUC desc
#   - gains_eligible.pdf:        bucket == "causal6_only_eligible", sort by causal6 AUC desc
#   - gains_newly_eligible.pdf:  bucket == "causal6_only_newly_eligible", sort by causal6 AUC desc
#   - both.pdf:                  random sample of 10 from "both" for sanity
#
# Star plot helper imported from src.viz_provisional.

# %%
import mne
from src.data import add_metadata_features, get_ambiguous_resampled_steps
from src.viz_provisional import provisional_star_plot

EPOCH_DIR = REPO / "outputs/epochs_preprocessed"


def load_epochs_dict(subjects):
    out = {}
    for s in subjects:
        path = EPOCH_DIR / f"{s}_epo.fif"
        if not path.exists():
            print(f"  (skip {s}: {path} missing)")
            continue
        ep = mne.read_epochs(path, preload=True, verbose="ERROR")
        ep.metadata = add_metadata_features(ep.metadata)
        out[s] = ep
    return out

needed_subjects = sorted(
    recon.filter(pl.col("bucket").is_in([
        "causal4_only", "causal6_only_eligible",
        "causal6_only_newly_eligible", "both",
    ]))["subject"].unique().to_list()
)
print(f"Loading epochs for {len(needed_subjects)} subjects: {needed_subjects}")
epochs_dict = load_epochs_dict(needed_subjects)
ambig_steps = get_ambiguous_resampled_steps()
```

- [ ] **Step 5.2: Append star-plot rendering helper**

```python
# %%
def _word_ends_for_pp(pp):
    # dn → desolate, necessary; bm → bemoan, mosaic; pb → ... (verify)
    return {
        "dn": ["desolate", "necessary"],
        "bm": ["bemoan", "mosaic"],
        "pb": ["puppet", "bishop"],
    }.get(pp, [])


def render_gallery(rows: pl.DataFrame, out_path: Path, title_prefix: str):
    """Render one PDF per site, one page per word_end.

    `rows` should already be sorted; each row produces 2 pages (one per word_end).
    """
    if rows.shape[0] == 0:
        print(f"  (no sites for {out_path.name})")
        return
    n_pages_rendered = 0
    with PdfPages(out_path) as pdf:
        for row in rows.iter_rows(named=True):
            if row["subject"] not in epochs_dict:
                continue
            for we in _word_ends_for_pp(row["phoneme_pair"]):
                try:
                    fig = provisional_star_plot(
                        subject=row["subject"],
                        electrode_idx=int(row["electrode_idx"]),
                        phoneme_pair=row["phoneme_pair"],
                        word_end=we,
                        epochs_dict=epochs_dict,
                        ambig_steps=ambig_steps,
                        phon_smin=int(row["causal6_smin"]) if row["causal6_smin"] is not None else
                                  (int(row["causal4_smin"]) if row["causal4_smin"] is not None else None),
                        phon_smax=int(row["causal6_smax"]) if row["causal6_smax"] is not None else
                                  (int(row["causal4_smax"]) if row["causal4_smax"] is not None else None),
                        acoustic_peak_auc=(
                            float(row["causal6_test_roc_auc"]) if row["causal6_test_roc_auc"] is not None
                            else (float(row["causal4_peak_auc"]) if row["causal4_peak_auc"] is not None else None)
                        ),
                    )
                    fig.suptitle(
                        f"{title_prefix}  |  {row['subject']} e{row['electrode_idx']} "
                        f"{row['phoneme_pair']} → {we}",
                        y=1.02, fontsize=10,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    n_pages_rendered += 1
                except Exception as ex:
                    print(f"  star_plot failed for {row['subject']} e{row['electrode_idx']} "
                          f"{row['phoneme_pair']} {we}: {ex}")
                    plt.close("all")
    print(f"Wrote {out_path.name}: {n_pages_rendered} pages")
```

- [ ] **Step 5.3: Append rendering invocations**

```python
# %%
# Losses — sort by causal4 peak AUC descending
losses_sorted = (
    recon.filter(pl.col("bucket") == "causal4_only")
         .sort("causal4_peak_auc", descending=True)
)
render_gallery(losses_sorted, OUT_DIR / "losses.pdf", title_prefix="LOSS")

# %%
# Gains, formerly eligible
ge_sorted = (
    recon.filter(pl.col("bucket") == "causal6_only_eligible")
         .sort("causal6_test_roc_auc", descending=True)
)
render_gallery(ge_sorted, OUT_DIR / "gains_eligible.pdf", title_prefix="GAIN(elig)")

# %%
# Gains, newly eligible
gne_sorted = (
    recon.filter(pl.col("bucket") == "causal6_only_newly_eligible")
         .sort("causal6_test_roc_auc", descending=True)
)
render_gallery(gne_sorted, OUT_DIR / "gains_newly_eligible.pdf", title_prefix="GAIN(new)")

# %%
# Sanity sample from both
both_sample = (
    recon.filter(pl.col("bucket") == "both")
         .sample(min(10, recon.filter(pl.col("bucket") == "both").shape[0]),
                 seed=0)
         .sort("causal6_test_roc_auc", descending=True)
)
render_gallery(both_sample, OUT_DIR / "both.pdf", title_prefix="BOTH")
```

- [ ] **Step 5.4: Verify word_end mapping**

Before running, confirm the `_word_ends_for_pp` mapping is correct by checking `src.stimuli.OFFSET_DICT` keys and which word_ends pair with which phoneme_pair. Look at how `view_provisional_results.py` constructs word_end lists (search for `word_end`).

If `bm` / `pb` word_ends are wrong, fix the dict before running. (Default for `dn` is known-good from the existing notebook.)

```bash
./.venv/bin/python -c "from src.stimuli import OFFSET_DICT; print(sorted(OFFSET_DICT.keys()))"
```

Cross-reference with `notebooks/causal6/view_provisional_results.py` to see how it iterates `word_end` per `phoneme_pair`.

- [ ] **Step 5.5: Execute the notebook end-to-end**

```bash
./.venv/bin/jupytext --execute --to notebook --output - notebooks/causal46_joined/as_reconciliation.py > /tmp/recon_full.ipynb 2>&1 || echo "EXECUTION FAILED"
```

Verify:

```bash
ls -la outputs/causal46_joined/*.pdf
```

Expected: four PDFs, all non-zero size. Spot-check by opening `outputs/causal46_joined/losses.pdf` — first page should show a `causal4_only` site sorted as highest-AUC loss.

- [ ] **Step 5.6: Commit**

```bash
git add notebooks/causal46_joined/as_reconciliation.py
git commit -m "render reconciliation star-plot galleries"
```

---

### Task 6: Canonical AS-site CSV

**Files:** Modify `notebooks/causal46_joined/as_reconciliation.py`

- [ ] **Step 6.1: Append canonical-list section**

```python
# %% [markdown]
# ## Canonical AS-site list
#
# Initial canonical list = every site with causal6_AS == True (union of `both`,
# `causal6_only_eligible`, `causal6_only_newly_eligible`).
#
# The user may overwrite `canonical_AS_sites.csv` manually after reviewing the
# PDFs (e.g., to add back high-AUC `causal4_only` losses, or remove borderline
# gains). Downstream notebooks (Group B/C) MUST read from this CSV.

# %%
canonical = (
    recon.filter(pl.col("causal6_AS"))
         .select([
             "subject", "electrode_idx", "phoneme_pair",
             pl.col("causal6_smin").alias("smin"),
             pl.col("causal6_smax").alias("smax"),
             pl.col("causal6_test_roc_auc").alias("peak_auc"),
             pl.col("causal6_p_value").alias("p_value"),
             "bucket",
         ])
         .sort(["subject", "electrode_idx", "phoneme_pair"])
)
canonical.write_csv(OUT_DIR / "canonical_AS_sites.csv")
print(f"Canonical AS sites: {canonical.shape[0]}")
print(f"Written: {OUT_DIR / 'canonical_AS_sites.csv'}")
print(canonical.group_by("bucket").len().sort("len", descending=True))
```

- [ ] **Step 6.2: Execute the notebook end-to-end one more time**

Same as Step 5.5. Verify the CSV exists and has the expected row count (= `both` + `causal6_only_eligible` + `causal6_only_newly_eligible`).

```bash
./.venv/bin/python -c "
import polars as pl
df = pl.read_csv('outputs/causal46_joined/canonical_AS_sites.csv')
print('rows:', df.shape[0])
print('columns:', df.columns)
print(df.head(5))
"
```

- [ ] **Step 6.3: Commit**

```bash
git add notebooks/causal46_joined/as_reconciliation.py outputs/causal46_joined/canonical_AS_sites.csv
git commit -m "write canonical AS-site CSV for downstream use"
```

---

### Task 7: Final review checklist (no code)

This is for the user, not the agent. Agent should print a summary at the end of the notebook reminding the user to:

- [ ] **Step 7.1: Append review reminder section**

```python
# %% [markdown]
# ## Review checklist for the user
#
# 1. Open `outputs/causal46_joined/losses.pdf` — are any of the highest-AUC
#    losses visually compelling (clear divergence between step 1 and step 6
#    HGA in the top panel within the shaded acoustic window)? If yes, causal6
#    NHST may be over-conservative; consider relaxing the p threshold or
#    keeping selected sites manually.
# 2. Open `outputs/causal46_joined/gains_eligible.pdf` — do the gains look
#    real? If most are noisy, causal6 NHST may have inflated power (e.g.,
#    insufficient permutations).
# 3. Open `outputs/causal46_joined/gains_newly_eligible.pdf` — these sites
#    were rejected by causal4's speech-responsive pre-screen. Validating
#    these supports dropping the pre-screen.
# 4. Edit `outputs/causal46_joined/canonical_AS_sites.csv` manually if you
#    want to override the default (causal6_AS = True) selection.
# 5. The 3 absent subjects (EC248, EC250, EC253) are NOT in the canonical
#    list. When causal6 prod is re-synced with them, re-run this notebook.
```

- [ ] **Step 7.2: Commit and done**

```bash
git add notebooks/causal46_joined/as_reconciliation.py
git commit -m "add reviewer checklist to reconciliation notebook"
```

---

## Acceptance criteria

The reconciliation work is complete when:

1. `notebooks/causal46_joined/as_reconciliation.py` runs end-to-end via `jupytext --execute` without errors (modulo the 3 missing-subject warning, which is expected).
2. `outputs/causal46_joined/` contains: `reconciliation.parquet`, `canonical_AS_sites.csv`, `bucket_breakdown.csv`, `summary_audit.png`, `summary_scatter.png`, `losses.pdf`, `gains_eligible.pdf`, `gains_newly_eligible.pdf`, `both.pdf`.
3. `bucket` counts are internally consistent:
   - `both + causal4_only` == number of causal4 AS sites in the 7 reconcilable subjects.
   - `both + causal6_only_eligible + causal6_only_newly_eligible` == number of causal6-significant sites in those subjects.
4. `canonical_AS_sites.csv` row count == count of sites with `causal6_AS == True`.
5. `src/viz_provisional.py` is importable from anywhere and `view_provisional_results.py` still runs after the refactor.
6. `tests/test_as_reconciliation.py` passes.

## Out of scope

- Permutation-null correction on causal6 behavior decoder (Group B/C territory).
- Anatomy chi-squared or ROI-level analysis on the canonical list (downstream).
- Re-running causal6 for the missing 3 subjects — that's a pipeline ops task, not part of this notebook.
- Modifying causal4 outputs in any way.
