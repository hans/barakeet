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
# # t_tests_all_sr AS-subset reconciliation — blocking gate on credibility
#
# (Not to be confused with `as_reconciliation.py`, an unrelated
# causal4-vs-causal6 site reconciliation notebook.)
#
# Step 3b of the all-SR perceptual fork
# (`docs/superpowers/plans/2026-08-27-all-speech-responsive-perceptual.md`).
#
# The partition readout (Step 4) is only meaningful if the all-SR pipeline's
# acoustic-significant (AS) cells reproduce the existing AS-restricted
# `t_tests.py` output EXACTLY — otherwise Step 4 is comparing two different
# tests, not partitioning one test. Because:
#
# - trial-balance counts are subject-level and broadcast across electrodes
#   (`trial_balance_index_all_sr.py` is a pure key-set swap of
#   `trial_balance_index.py`),
# - the B4 bootstrap uses the same seeding (`base_seed + r`) on the same
#   per-step trial pools, so an AS cell's draws are bit-identical between the
#   two pipelines,
# - and best-window selection by |mean_diff_raw| agrees with the original's
#   selection by |mean_diff_aligned| (aligned = a constant per-cell sign flip
#   of raw, so the argmax over windows is unchanged, and the two-sided
#   `ci_excludes_zero` test is sign-invariant),
#
# the raw fields (`mean_diff_raw_med`, `ci_raw_excludes_zero`, ...) should
# reconcile to floating-point precision. Any mismatch beyond tolerance, or
# any AS cell present in one output but not the other, means an upstream
# artifact leaked in (K mismatch, window/stride mismatch, threshold
# mismatch, join semantics) — not real divergence. This notebook hard-fails
# (raises) when that happens; it is a gate, not a diagnostic.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# %% tags=["parameters"]
all_sr_per_window_path = "outputs/causal46_joined/t_tests_all_sr/b4_per_window.parquet"
all_sr_per_cell_path = "outputs/causal46_joined/t_tests_all_sr/b4_per_cell.parquet"
original_per_window_path = "outputs/causal46_joined/t_tests/b4_per_window.parquet"
original_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
outdir = "outputs/causal46_joined/t_tests_all_sr_reconciliation"

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
atol = 1e-9
rtol = 1e-9

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

# %% [markdown]
# ## Load and restrict all-SR output to the acoustic_significant subset

# %%
all_sr_window = pl.read_parquet(all_sr_per_window_path)
all_sr_cell = pl.read_parquet(all_sr_per_cell_path)
orig_window = pl.read_parquet(original_per_window_path)
orig_cell = pl.read_parquet(original_per_cell_path)

as_window = all_sr_window.filter(pl.col("acoustic_significant"))
as_cell = all_sr_cell.filter(pl.col("acoustic_significant"))

print(f"all-SR per_window rows: {all_sr_window.height}  (AS subset: {as_window.height})")
print(f"all-SR per_cell rows:   {all_sr_cell.height}  (AS subset: {as_cell.height})")
print(f"original per_window rows: {orig_window.height}")
print(f"original per_cell rows:   {orig_cell.height}")

# %% [markdown]
# ## Set reconciliation — same AS cells on both sides
#
# AS ⊂ SR by construction (`phon_peaks` comes from acoustic decoding that
# already ran on the SR set), so cell-set membership should match exactly.

# %%
as_window_keys = as_window.select(CELL_KEYS + ["smin", "smax"]).unique()
orig_window_keys = orig_window.select(CELL_KEYS + ["smin", "smax"]).unique()

only_in_all_sr_window = as_window_keys.join(
    orig_window_keys, on=CELL_KEYS + ["smin", "smax"], how="anti",
)
only_in_orig_window = orig_window_keys.join(
    as_window_keys, on=CELL_KEYS + ["smin", "smax"], how="anti",
)

print(f"(cell x window) only in all-SR AS subset: {only_in_all_sr_window.height}")
print(f"(cell x window) only in original:         {only_in_orig_window.height}")

as_cell_keys = as_cell.select(CELL_KEYS).unique()
orig_cell_keys = orig_cell.select(CELL_KEYS).unique()
only_in_all_sr_cell = as_cell_keys.join(orig_cell_keys, on=CELL_KEYS, how="anti")
only_in_orig_cell = orig_cell_keys.join(as_cell_keys, on=CELL_KEYS, how="anti")

print(f"cells only in all-SR AS subset: {only_in_all_sr_cell.height}")
print(f"cells only in original:         {only_in_orig_cell.height}")

set_mismatch = (
    only_in_all_sr_window.height or only_in_orig_window.height
    or only_in_all_sr_cell.height or only_in_orig_cell.height
)
if only_in_all_sr_window.height:
    only_in_all_sr_window.write_csv(OUT_DIR / "only_in_all_sr_window.csv")
if only_in_orig_window.height:
    only_in_orig_window.write_csv(OUT_DIR / "only_in_orig_window.csv")
if only_in_all_sr_cell.height:
    only_in_all_sr_cell.write_csv(OUT_DIR / "only_in_all_sr_cell.csv")
if only_in_orig_cell.height:
    only_in_orig_cell.write_csv(OUT_DIR / "only_in_orig_cell.csv")

# %% [markdown]
# ## Numeric reconciliation — raw fields, per-window and per-cell
#
# Same shape both times (join AS-subset to original, diff a numeric column
# list + a boolean column, report mismatches) — one helper, two call sites.

# %%
def find_mismatches(
    joined: pl.DataFrame,
    *,
    id_cols: list[str],
    numeric_cols: list[str],
    bool_col: str,
    extra_bad_cols: list[str] | None = None,
) -> list[dict]:
    """Row-wise diff of `joined` (an inner join with `_orig`-suffixed columns).

    A row is a mismatch if any `numeric_cols` entry differs beyond
    (atol, rtol), `bool_col` differs, or any `extra_bad_cols` entry differs
    (used for window bounds, which must match exactly, not approximately).
    """
    extra_bad_cols = extra_bad_cols or []
    mismatches: list[dict] = []
    for row in joined.iter_rows(named=True):
        bad_numeric = any(
            not np.isclose(row[col], row[f"{col}_orig"], atol=atol, rtol=rtol, equal_nan=True)
            for col in numeric_cols
        )
        bad_bool = row[bool_col] != row[f"{bool_col}_orig"]
        bad_extra = any(row[col] != row[f"{col}_orig"] for col in extra_bad_cols)
        if bad_numeric or bad_bool or bad_extra:
            mismatches.append({
                **{k: row[k] for k in id_cols},
                **{col: row[col] for col in extra_bad_cols},
                **{f"{col}_orig": row[f"{col}_orig"] for col in extra_bad_cols},
                bool_col: row[bool_col],
                f"{bool_col}_orig": row[f"{bool_col}_orig"],
                **{f"{col}_diff": row[col] - row[f"{col}_orig"] for col in numeric_cols},
            })
    return mismatches


# %% [markdown]
# ## Numeric reconciliation — per-window raw fields

# %%
joined_window = as_window.join(
    orig_window, on=CELL_KEYS + ["smin", "smax"], how="inner", suffix="_orig",
)
print(f"joined per_window rows (AS ∩ original): {joined_window.height}")

window_mismatches = find_mismatches(
    joined_window,
    id_cols=CELL_KEYS + ["smin", "smax"],
    numeric_cols=["mean_diff_raw_med", "mean_diff_raw_ci_lo", "mean_diff_raw_ci_hi", "emp_p_raw"],
    bool_col="ci_raw_excludes_zero",
)
window_mismatch_df = pl.DataFrame(window_mismatches) if window_mismatches else pl.DataFrame()
if window_mismatch_df.height:
    window_mismatch_df.write_csv(OUT_DIR / "window_mismatches.csv")
print(f"per_window numeric/boolean mismatches: {len(window_mismatches)} / {joined_window.height}")

# %% [markdown]
# ## Numeric reconciliation — per-cell best-window raw fields

# %%
joined_cell = as_cell.join(orig_cell, on=CELL_KEYS, how="inner", suffix="_orig")
print(f"joined per_cell rows (AS ∩ original): {joined_cell.height}")

cell_mismatches = find_mismatches(
    joined_cell,
    id_cols=CELL_KEYS,
    numeric_cols=[
        "best_mean_diff_raw_med", "best_mean_diff_raw_ci_lo", "best_mean_diff_raw_ci_hi",
        "best_emp_p_raw",
    ],
    bool_col="best_ci_raw_excludes_zero",
    extra_bad_cols=["best_smin", "best_smax"],
)
cell_mismatch_df = pl.DataFrame(cell_mismatches) if cell_mismatches else pl.DataFrame()
if cell_mismatch_df.height:
    cell_mismatch_df.write_csv(OUT_DIR / "cell_mismatches.csv")
print(f"per_cell numeric/boolean/window mismatches: {len(cell_mismatches)} / {joined_cell.height}")

# %% [markdown]
# ## Verdict — blocking gate

# %%
n_numeric_mismatch = len(window_mismatches) + len(cell_mismatches)
passed = (not set_mismatch) and (n_numeric_mismatch == 0)

summary = pl.DataFrame({
    "only_in_all_sr_window": [only_in_all_sr_window.height],
    "only_in_orig_window": [only_in_orig_window.height],
    "only_in_all_sr_cell": [only_in_all_sr_cell.height],
    "only_in_orig_cell": [only_in_orig_cell.height],
    "window_mismatches": [len(window_mismatches)],
    "cell_mismatches": [len(cell_mismatches)],
    "joined_window_rows": [joined_window.height],
    "joined_cell_rows": [joined_cell.height],
    "passed": [passed],
})
summary.write_csv(OUT_DIR / "reconciliation_summary.csv")
print(summary)

if passed:
    print("=" * 70)
    print("RECONCILIATION PASSED — all-SR AS subset reproduces t_tests.py exactly.")
    print("=" * 70)
else:
    print("=" * 70)
    print("RECONCILIATION FAILED — see mismatches CSVs under", OUT_DIR)
    print("=" * 70)
    raise AssertionError(
        "t_tests_all_sr_reconciliation: AS subset of the all-SR pipeline does NOT "
        f"reconcile with t_tests.py. set_mismatch={bool(set_mismatch)}, "
        f"numeric/boolean mismatches={n_numeric_mismatch}. This blocks the "
        "Step 4 partition readout (perceptual_acoustic_partition.py) — the "
        "two pipelines are testing different things, not one test "
        "partitioned. See docs/superpowers/plans/"
        "2026-08-27-all-speech-responsive-perceptual.md caveat: "
        "'the partition is only meaningful once the AS cells reproduce the "
        "current test'. Check: ac_p_value_threshold consistency between "
        "sr_site_universe.py and t_tests.py's config, K / window_size / "
        "stride config parity, and qualifying_steps ordering."
    )
