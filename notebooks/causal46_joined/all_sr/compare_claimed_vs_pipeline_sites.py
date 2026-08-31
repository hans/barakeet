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
# # Compare all-SR "claimed" perceptual sites vs main-pipeline sites
#
# Interactive scaffold (NOT a Snakefile rule) for eyeballing whether the
# non-acoustic perceptual sites surfaced by the all-SR fork
# (`t_tests_all_sr` → `perceptual_acoustic_partition`, the "NEW CELL":
# perceptual-significant AND NOT acoustic-significant) look like real
# sustained within-completion responses or like noise picked up by a
# self-selected best window with a raw, uncorrected bootstrap CI.
#
# **Why this notebook exists.** `t_tests_all_sr.py` deliberately emits no
# star-plot gallery (it dropped the aligned-polarity columns the gallery
# needs). So there is currently *no* per-site visualization for the ~2857
# claimed cells. This is the informal version of the missing window-max
# count-vs-null test: look at the traces before deciding whether the 10 GB
# formal test is worth running.
#
# **What it does**
# 1. Loads the all-SR per-cell test, defines the CLAIMED set (new cell),
#    ranks by |mean_diff| with an `n_per_class` floor, and flags cells whose
#    self-selected window is implausibly early (before the point of
#    disambiguation) — the tell-tale of a baseline/noise blip.
# 2. Builds a REFERENCE set from the trusted main pipeline (default: the 187
#    reconciled AS cells; swappable).
# 3. Joins both to electrode anatomy (ROI) and cross-tabulates — answers
#    "where are these sites" and is itself a signal/noise diagnostic (STG
#    near AS sites = interesting; scattered across non-auditory cortex = the
#    circularity showing).
# 4. Renders the canonical within-completion star plots (full HGA contrast
#    time course, selected window marked) for the top claimed cells and the
#    reference cells, via the same `write_annotated_pdfs` machinery the AS
#    gallery uses.
#
# **Runs on prod** (needs `outputs/epochs_preprocessed/*_epo.fif` and the
# electrode `.mat` files, neither available in the dev container).

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src._star_gallery import write_annotated_pdfs
from src.data import get_electrode_df
from src.stimuli import POD_dict
from src.viz_provisional import load_epochs_dict

# %% tags=["parameters"]
# --- all-SR (claimed) inputs ---
b4_per_cell_path = "outputs/causal46_joined/t_tests_all_sr/b4_per_cell.parquet"
b4_per_window_path = "outputs/causal46_joined/t_tests_all_sr/b4_per_window.parquet"
cell_manifest_path = "outputs/causal46_joined/t_tests_all_sr/cell_manifest.parquet"
reconciliation_summary_path = "outputs/causal46_joined/t_tests_all_sr_reconciliation/reconciliation_summary.csv"

# --- reference (main-pipeline) selector ---
# "as_reconciled"  : the 187 AS cells that reconcile bit-exact vs t_tests.py (default)
# "type2_early"    : curated type2 early-perceptual example sites
#                    (early_perceptual_projection/test3_detail.csv)
# "late_projection": late_perceptual_projection sig cells (projection_significant)
reference_set = "as_reconciled"
type2_detail_path = "outputs/causal46_joined/early_perceptual_projection/test3_detail.csv"
late_projection_results_path = "outputs/causal46_joined/late_perceptual_projection/results.csv"

# --- selection knobs ---
n_per_class_min = 15    # drop underpowered cells (small n fakes large |mean_diff|)
top_n_claimed = 16      # how many claimed cells to render (ranked by |mean_diff|)

# --- environment ---
EPOCH_DIR = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/compare_claimed_vs_pipeline_sites"

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Gate: reconciliation must have passed
# The claimed/reference comparison is only meaningful if the all-SR fork
# reproduces `t_tests.py` bit-exact on the AS subset.

# %%
recon = pl.read_csv(reconciliation_summary_path)
assert bool(recon["passed"][0]), (
    "t_tests_all_sr_reconciliation did NOT pass — the all-SR fork does not "
    "reproduce t_tests.py on AS cells; the partition/claimed set is untrustworthy."
)
print("✓ reconciliation passed")

# %% [markdown]
# ## Load all-SR per-cell test; define the CLAIMED (new-cell) set

# %%
per_cell = pl.read_parquet(b4_per_cell_path).with_columns([
    pl.col("acoustic_significant").fill_null(False),
    pl.col("best_ci_raw_excludes_zero").fill_null(False),
    pl.col("best_mean_diff_raw_med").abs().alias("absmd"),
])

# POD (point of disambiguation) per phoneme pair, seconds. A "perceptual"
# window whose whole extent is before POD cannot reflect a percept — it is a
# baseline/onset blip. This flag is a first-pass plausibility screen, not a
# significance test.
pod = pl.DataFrame(
    {"phoneme_pair": list(POD_dict), "pod_s": [POD_dict[k] for k in POD_dict]}
)
per_cell = per_cell.join(pod, on="phoneme_pair", how="left").with_columns(
    (pl.col("best_tmax") < pl.col("pod_s")).alias("window_before_pod")
)

claimed = (
    per_cell
    .filter((~pl.col("acoustic_significant")) & pl.col("best_ci_raw_excludes_zero"))
    .filter(pl.col("n_per_class") >= n_per_class_min)
    .sort("absmd", descending=True)
)
print(f"CLAIMED new-cell (perc-sig, not acoustic, n_per_class>={n_per_class_min}): "
      f"{claimed.height} cells across "
      f"{claimed.select(['subject','electrode_idx']).unique().height} electrodes, "
      f"{claimed.select('subject').unique().height} subjects")
print(f"  of which self-selected window is BEFORE POD (implausible): "
      f"{int(claimed['window_before_pod'].sum())} / {claimed.height}")
print(f"  subject spread of top {top_n_claimed}:")
print(claimed.head(top_n_claimed).group_by("subject").len().sort("len", descending=True))

# %% [markdown]
# ## Build the REFERENCE (main-pipeline) set

# %%
if reference_set == "as_reconciled":
    reference = per_cell.filter(
        pl.col("acoustic_significant") & pl.col("best_ci_raw_excludes_zero")
    ).sort("absmd", descending=True)
elif reference_set == "type2_early":
    t2 = pl.read_csv(type2_detail_path)
    # keep the columns needed to key back into per_cell
    keycols = ["subject", "electrode_idx", "phoneme_pair"]
    reference = per_cell.join(t2.select(keycols).unique(), on=keycols, how="inner")
elif reference_set == "late_projection":
    lp = pl.read_csv(late_projection_results_path).filter(
        pl.col("projection_significant")
    )
    keycols = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
    reference = per_cell.join(lp.select(keycols).unique(), on=keycols, how="inner")
else:
    raise ValueError(f"unknown reference_set={reference_set!r}")

print(f"REFERENCE ({reference_set}): {reference.height} cells across "
      f"{reference.select(['subject','electrode_idx']).unique().height} electrodes")

# %% [markdown]
# ## Where are these sites? — anatomy (ROI) cross-tab
# `get_electrode_df(subject)` returns anatomy rows in electrode order; we map
# `electrode_idx` positionally. If the claimed sites cluster in STG alongside
# the AS sites, that is corroborating; if they scatter across non-auditory
# cortex, that is the selection artifact showing.

# %%
def attach_roi(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for subj in df["subject"].unique():
        edf = get_electrode_df(subj)  # pandas; roi in column "roi"
        for eidx in df.filter(pl.col("subject") == subj)["electrode_idx"].unique():
            roi = edf["roi"].iloc[int(eidx)] if int(eidx) < len(edf) else "OOB"
            rows.append({"subject": subj, "electrode_idx": int(eidx), "roi": roi})
    roi_map = pl.DataFrame(rows)
    return df.join(roi_map, on=["subject", "electrode_idx"], how="left")


claimed_elec = claimed.select(["subject", "electrode_idx"]).unique()
ref_elec = reference.select(["subject", "electrode_idx"]).unique()
try:
    claimed_roi = attach_roi(claimed_elec)
    ref_roi = attach_roi(ref_elec)
    print("CLAIMED electrodes by ROI:")
    print(claimed_roi.group_by("roi").len().sort("len", descending=True))
    print("\nREFERENCE electrodes by ROI:")
    print(ref_roi.group_by("roi").len().sort("len", descending=True))
    claimed_roi.write_csv(OUT_DIR / "claimed_electrodes_roi.csv")
    ref_roi.write_csv(OUT_DIR / "reference_electrodes_roi.csv")
except Exception as e:  # electrode .mat files absent (e.g. dev container)
    print(f"[skipped ROI join: {e!r}]")

# %% [markdown]
# ## Render within-completion star plots (full HGA contrast time course)
#
# Reuses the AS gallery machinery (`write_annotated_pdfs` → `matched_n_star_plot`).
# That code path reads the *aligned*-polarity columns, which the all-SR fork
# dropped; we alias raw → aligned here (polarity is a per-cell sign flip and
# does not change the trace's shape or the CI-excludes-zero verdict).

# %%
b4_per_window = pl.read_parquet(b4_per_window_path)
cell_manifest = pl.read_parquet(cell_manifest_path)
epochs_dict = load_epochs_dict(Path(EPOCH_DIR))

RAW_TO_ALIGNED_CELL = {
    "best_mean_diff_raw_med": "best_mean_diff_aligned_med",
    "best_emp_p_raw": "best_emp_p_aligned",
    "best_ci_raw_excludes_zero": "best_ci_aligned_excludes_zero",
}


def alias_raw_to_aligned(df: pl.DataFrame, mapping: dict) -> pl.DataFrame:
    exprs = [pl.col(src).alias(dst) for src, dst in mapping.items() if src in df.columns]
    return df.with_columns(exprs)


# per_window needs ci_aligned_excludes_zero for the sig-bar overlay
pw = b4_per_window
if "ci_raw_excludes_zero" in pw.columns and "ci_aligned_excludes_zero" not in pw.columns:
    pw = pw.with_columns(pl.col("ci_raw_excludes_zero").alias("ci_aligned_excludes_zero"))


def build_entries(cells: pl.DataFrame) -> list[dict]:
    cells = alias_raw_to_aligned(cells, RAW_TO_ALIGNED_CELL)
    entries = []
    for row in cells.iter_rows(named=True):
        mrow = cell_manifest.filter(
            (pl.col("subject") == row["subject"])
            & (pl.col("electrode_idx") == row["electrode_idx"])
            & (pl.col("phoneme_pair") == row["phoneme_pair"])
            & (pl.col("word_end") == row["word_end"])
        )
        qs = mrow["qualifying_steps"][0] if mrow.height else ""
        entries.append({
            "mode": "matched_n",
            "subject": row["subject"], "electrode_idx": row["electrode_idx"],
            "phoneme_pair": row["phoneme_pair"], "word_end": row["word_end"],
            "resampled": None, "qualifying_steps": qs,
            "best_smin": row["best_smin"], "best_smax": row["best_smax"],
            "best_mean_diff_aligned_med": row.get("best_mean_diff_aligned_med"),
            "best_emp_p_aligned": row.get("best_emp_p_aligned"),
            "best_ci_aligned_excludes_zero": row.get("best_ci_aligned_excludes_zero", False),
            "powered": True, "significant": bool(row.get("best_ci_aligned_excludes_zero")),
            "pdf_path": "", "status": "ok",
        })
    return entries


cell_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

claimed_top = claimed.head(top_n_claimed)
n_c = write_annotated_pdfs(
    build_entries(claimed_top), pw, cell_keys,
    OUT_DIR / "claimed_gallery.pdf", epochs_dict=epochs_dict,
)
n_r = write_annotated_pdfs(
    build_entries(reference), pw, cell_keys,
    OUT_DIR / f"reference_{reference_set}_gallery.pdf", epochs_dict=epochs_dict,
)
print(f"claimed_gallery.pdf: {n_c} cells")
print(f"reference_{reference_set}_gallery.pdf: {n_r} cells")

# %% [markdown]
# ## Save the claimed-site table for reference

# %%
claimed_top.select([
    "subject", "electrode_idx", "phoneme_pair", "word_end", "n_per_class",
    "best_mean_diff_raw_med", "best_emp_p_raw",
    "best_smin", "best_smax", "best_tmin", "best_tmax", "window_before_pod",
]).write_csv(OUT_DIR / "claimed_top.csv")
print(f"wrote {OUT_DIR / 'claimed_top.csv'}")
print("\nOpen the two gallery PDFs side by side. Read the CLAIMED traces for:")
print("  - a sustained within-completion separation vs a single-window blip")
print("  - window timing near/after POD vs implausibly early (window_before_pod)")
print("  - whether the ROI table places them in STG near the AS sites")
