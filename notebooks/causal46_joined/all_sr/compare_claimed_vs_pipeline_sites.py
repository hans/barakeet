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
#     display_name: barakeet (3.12.13)
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
from matplotlib.backends.backend_pdf import PdfPages

from src._star_gallery import matched_n_star_plot_paper
from src.data import get_electrode_df
from src.stimuli import POD_dict
from src.viz_paper import resampled_cmap
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
top_n_claimed = 30      # how many claimed cells to render (ranked by |mean_diff|)
top_n_reference = 24    # cap on reference cells rendered (ranked by |mean_diff|)

# --- environment ---
EPOCH_DIR = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/compare_claimed_vs_pipeline_sites"

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
phon_peaks = pl.read_parquet("outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet")

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

reference = reference.sort("absmd", descending=True)
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
            if int(eidx) < len(edf):
                roi = edf["roi"].iloc[int(eidx)]
                if isinstance(roi, str):
                    roi = roi.strip()
                else:
                    roi = "NA"
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
# ## Render within-completion star plots (`matched_n_star_plot_paper`)
#
# The paper renderer recomputes the bottom-panel per-step class-balanced
# bootstrap contrast internally from the epochs (full HGA contrast time
# course). For each cell we mark:
#
# - **acoustic window** (`phon_smin`/`phon_smax`, from the manifest — null for
#   the non-acoustic claimed cells) on the top (acoustic) panel;
# - the **self-selected perceptual window** (`best_smin`/`best_smax`) on the
#   bottom (behavioral) panel, routed to the **EARLY** slot (green) when it
#   falls before POD or the **LATE** slot (purple) otherwise — so the figure
#   itself encodes the early/late plausibility split;
# - **POD** (dashed) and the windows where the bootstrap CI excludes zero
#   (`sig_windows`).

# %%
b4_per_window = pl.read_parquet(b4_per_window_path)
cell_manifest = pl.read_parquet(cell_manifest_path)
epochs_dict = load_epochs_dict(Path(EPOCH_DIR))

manifest_lut = {
    (r["subject"], r["electrode_idx"], r["phoneme_pair"], r["word_end"]): r
    for r in cell_manifest.iter_rows(named=True)
}


def _sig_windows_for(cell) -> list[tuple[float, float]]:
    pw = b4_per_window.filter(
        (pl.col("subject") == cell["subject"])
        & (pl.col("electrode_idx") == cell["electrode_idx"])
        & (pl.col("phoneme_pair") == cell["phoneme_pair"])
        & (pl.col("word_end") == cell["word_end"])
        & pl.col("ci_raw_excludes_zero")
    )
    return [(float(r["tmin"]), float(r["tmax"])) for r in pw.iter_rows(named=True)]


def render_gallery(cells: pl.DataFrame, out_pdf: Path) -> int:
    n = 0
    with PdfPages(out_pdf) as pdf:
        for cell in cells.iter_rows(named=True):
            key = (cell["subject"], cell["electrode_idx"],
                   cell["phoneme_pair"], cell["word_end"])
            mrow = manifest_lut.get(key)
            if mrow is None or not mrow["qualifying_steps"]:
                continue
            qs = [int(s) for s in str(mrow["qualifying_steps"]).split(",") if s != ""]

            # acoustic window (top panel) — null for non-acoustic claimed cells
            phon_smin, phon_smax = mrow["phon_smin"], mrow["phon_smax"]
            top_early = ((phon_smin, phon_smax)
                         if phon_smin is not None and phon_smax is not None else None)

            # self-selected perceptual window (bottom panel), routed early/late vs POD
            best_win = (cell["best_smin"], cell["best_smax"])
            before_pod = bool(cell.get("window_before_pod"))
            bottom_early = best_win if before_pod else None
            bottom_late = None if before_pod else best_win

            fig = matched_n_star_plot_paper(
                subject=cell["subject"],
                electrode_idx=int(cell["electrode_idx"]),
                phoneme_pair=cell["phoneme_pair"],
                word_end=cell["word_end"],
                qualifying_steps=qs,
                epochs_dict=epochs_dict,
                phon_smin=phon_smin, phon_smax=phon_smax,
                top_early_window=top_early,
                bottom_early_window=bottom_early,
                bottom_late_window=bottom_late,
                sig_windows=_sig_windows_for(cell),
                plot_first_sound=False,
                plot_pod=True,
                resampled_cmap=resampled_cmap,
            )
            md = cell["best_mean_diff_raw_med"]
            fig.suptitle(
                f"{cell['subject']} e{cell['electrode_idx']} {cell['phoneme_pair']} "
                f"{cell['word_end']}  |md|={abs(md):.2f} p={cell['best_emp_p_raw']:.3f} "
                f"dec={cell['test_roc_auc']:.3f} "
                f"{'  [window<POD]' if before_pod else ''}",
                fontsize=8,
            )
            pdf.savefig(fig)
            plt.close(fig)
            n += 1
    return n


plot_claimed = (
    claimed.head(top_n_claimed)
    .join(phon_peaks.select(["subject", "electrode_idx", "phoneme_pair", "test_roc_auc", "p_value"]),
          on=["subject", "electrode_idx", "phoneme_pair"])
)
plot_reference = (
    reference.head(top_n_reference)
    .join(phon_peaks.select(["subject", "electrode_idx", "phoneme_pair", "test_roc_auc", "p_value"]),
          on=["subject", "electrode_idx", "phoneme_pair"])
)
n_c = render_gallery(plot_claimed, OUT_DIR / "claimed_gallery.pdf")
n_r = render_gallery(plot_reference,
                     OUT_DIR / f"reference_{reference_set}_gallery.pdf")
print(f"claimed_gallery.pdf: {n_c} cells")
print(f"reference_{reference_set}_gallery.pdf: {n_r} cells")

# %%
render_gallery(
    (claimed.join(phon_peaks.select(["subject", "electrode_idx", "phoneme_pair", "test_roc_auc", "p_value"]),
                 on=["subject", "electrode_idx", "phoneme_pair"])
    .filter(pl.col("test_roc_auc") < 0.6)
    .sort("best_emp_p_raw")
    .head(20)),
    OUT_DIR / "claimed_gallery_only_bad_acoustic.pdf"
)

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
