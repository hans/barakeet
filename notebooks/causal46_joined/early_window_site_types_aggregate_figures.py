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
# # Early-window site types — aggregate figures
#
# Population-level figures from the per-subject early_window_site_types outputs:
#
# 1. **`population_site_type_counts.pdf`** — stacked horizontal bar chart of
#    site-type counts by (phoneme_pair × ROI).
# 2. **`A_vs_B_scatter.pdf`** — best-A acoustic contrast vs best-B aligned
#    contrast, colored by site_type.
# 3. **`star_plots_all.pdf`** — full gallery concatenated from per-subject
#    `star_plots_early.pdf` files (subject-major; within each subject sorted
#    by site_type).

# %%
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

try:
    from pypdf import PdfReader, PdfWriter
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False
    print("⚠ pypdf not installed — star_plots_all.pdf will be skipped")

sys.path.insert(0, str(Path(".").resolve() / "src"))
from data import get_electrode_df  # noqa: E402

# %% tags=["parameters"]
site_types_path    = "outputs/causal46_joined/early_window_site_types/site_type_assignments_all.parquet"
A_per_window_path  = "outputs/causal46_joined/early_window_site_types/A_per_window_all.parquet"
B_per_window_path  = "outputs/causal46_joined/early_window_site_types/B_per_window_all.parquet"
phon_peaks_path    = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
star_plots_dir     = "outputs/causal46_joined/early_window_site_types"
outdir             = "outputs/causal46_joined/early_window_site_types"

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

_cfg = yaml.safe_load((Path(".").resolve() / "config.yaml").read_text())
SUBJECTS = _cfg["data"]["subjects"]

# %% [markdown]
# ## Load data

# %%
site_types  = pl.read_parquet(site_types_path)
A_per_window = pl.read_parquet(A_per_window_path)
B_per_window = pl.read_parquet(B_per_window_path)
phon_peaks  = pl.read_parquet(phon_peaks_path)

print(f"site_types:   {site_types.height} rows")
print(f"A_per_window: {A_per_window.height} rows")
print(f"B_per_window: {B_per_window.height} rows")
print(f"phon_peaks:   {phon_peaks.height} rows")
if site_types.height > 0 and "site_type" in site_types.columns:
    print(site_types.group_by("site_type").len().sort("site_type"))

# %% [markdown]
# ## ROI lookup

# %%
roi_frames = []
for subj in SUBJECTS:
    try:
        edf = get_electrode_df(subj)
    except Exception as exc:
        print(f"  ⚠ no electrode_df for {subj}: {exc}")
        continue
    roi_col = "roi" if "roi" in edf.columns else ("anat" if "anat" in edf.columns else None)
    if roi_col is None:
        continue
    edf2 = edf.reset_index().rename(columns={"index": "electrode_idx"}) \
        if "electrode_idx" not in edf.columns else edf
    roi_frames.append(pl.from_pandas(
        edf2[["electrode_idx", roi_col]].assign(subject=subj)
                                        .rename(columns={roi_col: "roi"})
                                        .astype({"roi": str})
    ))
electrode_roi = (
    pl.concat(roi_frames, how="diagonal_relaxed")
    if roi_frames else
    pl.DataFrame(schema={"subject": pl.Utf8, "electrode_idx": pl.Int64, "roi": pl.Utf8})
)
print(f"ROI rows: {electrode_roi.height}")

# Add ROI to site_types
site_types_roi = (
    site_types
    .join(electrode_roi, on=["subject", "electrode_idx"], how="left")
    .with_columns(pl.col("roi").fill_null("unknown"))
)

# %% [markdown]
# ## Site-type relabeling manifest
#
# One row per (subject × electrode × phoneme_pair).  Fill `site_type_override`
# to override the auto-assigned `site_type`; leave blank to keep the original.

# %%
_relabel_cols = [
    "subject", "electrode_idx", "phoneme_pair", "roi",
    "site_type", "status",
    "manifest_tuning", "tuning_conflict",
    "acoustic_sign", "A_significant",
    "B1_word_end", "B1_aligned_sig", "B1_anti_sig", "B1_n_per_class",
    "B2_word_end", "B2_aligned_sig", "B2_anti_sig", "B2_n_per_class",
    "A_n_step1", "A_n_step6",
]
_avail_cols = [c for c in _relabel_cols if c in site_types_roi.columns]
relabel_df = (
    site_types_roi
    .select(_avail_cols)
    .with_columns(pl.lit("").alias("site_type_override"))
    .sort(["subject", "electrode_idx", "phoneme_pair"])
)
_relabel_path = OUT_DIR / "site_type_relabel.csv"
relabel_df.write_csv(_relabel_path)
print(f"wrote {_relabel_path}  ({relabel_df.height} rows)")

# %% [markdown]
# ## 1. Population site-type counts bar chart

# %%
_TYPE_ORDER = [
    "type2_early_perceptual",
    "type3_asymmetric",
    "grab_bag",
    "type1_acoustic_only",
    "complex",
    "A_unsigned",
    "unknown",
    "unclassifiable_B_power",
]
_TYPE_COLORS = {
    "type2_early_perceptual":   "#1a9850",
    "type3_asymmetric":         "#91cf60",
    "grab_bag":                 "#d73027",
    "type1_acoustic_only":      "#4393c3",
    "complex":                  "#762a83",
    "A_unsigned":               "#b2b2b2",
    "unknown":                  "#d9d9d9",
    "unclassifiable_B_power":   "#f5f5f5",
}

counts = (
    site_types_roi
    .filter(pl.col("site_type").is_not_null())
    .group_by(["phoneme_pair", "roi", "site_type"])
    .len()
    .rename({"len": "n"})
    .sort(["phoneme_pair", "roi", "site_type"])
)

# One bar per (phoneme_pair × roi)
bar_groups = (
    counts
    .select(["phoneme_pair", "roi"])
    .unique(maintain_order=True)
    .sort(["phoneme_pair", "roi"])
)

if bar_groups.height > 0:
    fig_h = max(4.0, bar_groups.height * 0.5)
    fig_bar, ax_bar = plt.subplots(figsize=(10, fig_h))

    lefts = np.zeros(bar_groups.height)
    for stype in _TYPE_ORDER:
        bar_vals = []
        for bgrow in bar_groups.iter_rows(named=True):
            match = counts.filter(
                (pl.col("phoneme_pair") == bgrow["phoneme_pair"])
                & (pl.col("roi") == bgrow["roi"])
                & (pl.col("site_type") == stype)
            )
            bar_vals.append(int(match["n"][0]) if match.height > 0 else 0)
        bar_vals_arr = np.array(bar_vals, dtype=float)
        y_labels = [f"{r['phoneme_pair']} / {r['roi']}" for r in bar_groups.iter_rows(named=True)]
        ax_bar.barh(
            y_labels, bar_vals_arr, left=lefts,
            color=_TYPE_COLORS.get(stype, "#aaaaaa"),
            label=stype, edgecolor="none", height=0.7,
        )
        for i, (v, l) in enumerate(zip(bar_vals_arr, lefts)):
            if v >= 1:
                ax_bar.text(l + v / 2, i, str(int(v)),
                            ha="center", va="center", fontsize=7, color="w")
        lefts += bar_vals_arr

    ax_bar.set_xlabel("Number of sites")
    ax_bar.set_title("Early-window site-type counts by phoneme pair × ROI")
    ax_bar.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax_bar.set_xlim(0, lefts.max() * 1.08)
    fig_bar.tight_layout()
    fig_bar.savefig(OUT_DIR / "population_site_type_counts.pdf", bbox_inches="tight")
    plt.close(fig_bar)
    print(f"wrote population_site_type_counts.pdf  ({bar_groups.height} bars)")
else:
    fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
    ax_bar.text(0.5, 0.5, "No data", ha="center", va="center")
    fig_bar.savefig(OUT_DIR / "population_site_type_counts.pdf")
    plt.close(fig_bar)
    print("population_site_type_counts.pdf: no data")

# %% [markdown]
# ## 2. A-vs-B scatter

# %%
# Best-A window per (subject × electrode × phoneme_pair): max |mean_diff_raw_med|
# among ci_excludes_zero windows; fall back to max regardless.
SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]

def _best_window(df, sig_col, val_col, keys):
    """Return best-window row per group: prefer sig rows, then abs-max."""
    if df.height == 0:
        return pl.DataFrame()
    sig = df.filter(pl.col(sig_col))
    base = sig if sig.height > 0 else df
    return (
        base
        .with_columns(pl.col(val_col).abs().alias("__abs"))
        .sort(keys + ["__abs"], descending=[False] * len(keys) + [True])
        .group_by(keys, maintain_order=True)
        .head(1)
        .drop("__abs")
    )

best_A = _best_window(A_per_window, "ci_excludes_zero", "mean_diff_raw_med", SITE_KEYS)
best_B = _best_window(
    B_per_window, "ci_aligned_excludes_zero", "mean_diff_aligned_med",
    SITE_KEYS + ["word_end"],
)

# Join A → site_types to get acoustic_sign, site_type
best_A_typed = best_A.join(
    site_types.select(SITE_KEYS + ["acoustic_sign", "site_type"]),
    on=SITE_KEYS, how="left",
)

# Compute signed A: acoustic_sign × mean_diff_raw_med (always ≥ 0 by construction)
best_A_typed = best_A_typed.with_columns([
    (pl.col("acoustic_sign") * pl.col("mean_diff_raw_med")).alias("A_signed"),
])

# Two panels — one per word_end; each point is (site × word_end)
CELL_KEYS = SITE_KEYS + ["word_end"]
scatter_df = (
    best_B
    .join(best_A_typed.select(SITE_KEYS + ["A_signed", "acoustic_sign", "site_type"]),
          on=SITE_KEYS, how="left")
    .with_columns(pl.col("site_type").fill_null("unknown"))
)

word_ends = sorted(scatter_df["word_end"].drop_nulls().unique().to_list())
n_we = len(word_ends)

if n_we > 0 and scatter_df.height > 0:
    fig_sc, axes_sc = plt.subplots(1, max(n_we, 1), figsize=(6 * max(n_we, 1), 5),
                                   sharey=True)
    if n_we == 1:
        axes_sc = [axes_sc]
    for ax_sc, we in zip(axes_sc, word_ends):
        sub = scatter_df.filter(pl.col("word_end") == we)
        for stype, color in _TYPE_COLORS.items():
            pts = sub.filter(pl.col("site_type") == stype)
            if pts.height == 0:
                continue
            x_arr = pts["A_signed"].to_numpy().astype(float)
            y_arr = pts["mean_diff_aligned_med"].to_numpy().astype(float)
            valid = np.isfinite(x_arr) & np.isfinite(y_arr)
            if valid.sum() == 0:
                continue
            ax_sc.scatter(x_arr[valid], y_arr[valid], color=color, s=40,
                          alpha=0.8, edgecolors="k", lw=0.4, label=stype)
        ax_sc.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax_sc.axvline(0, color="k", lw=0.4, ls=":", alpha=0.4)
        lim = max(
            np.nanmax(np.abs(scatter_df["A_signed"].to_numpy().astype(float))),
            np.nanmax(np.abs(scatter_df["mean_diff_aligned_med"].to_numpy().astype(float))),
            0.01,
        )
        ax_sc.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.25)
        ax_sc.set_xlim(-lim * 1.05, lim * 1.05)
        ax_sc.set_xlabel("best A effect (acoustic_sign × mean_diff_raw_med)")
        ax_sc.set_title(f"word_end: {we}  (n={sub.height})")
    axes_sc[0].set_ylabel("best B aligned effect (mean_diff_aligned_med)")
    handles = [plt.scatter([], [], color=c, s=40, edgecolors="k", lw=0.4, label=t)
               for t, c in _TYPE_COLORS.items()]
    axes_sc[-1].legend(handles=handles, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig_sc.suptitle("A-vs-B scatter — best window per cell", fontsize=11)
    fig_sc.tight_layout()
    fig_sc.savefig(OUT_DIR / "A_vs_B_scatter.pdf", bbox_inches="tight")
    plt.close(fig_sc)
    print(f"wrote A_vs_B_scatter.pdf  ({scatter_df.height} points)")
else:
    fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
    ax_sc.text(0.5, 0.5, "No data", ha="center", va="center")
    fig_sc.savefig(OUT_DIR / "A_vs_B_scatter.pdf")
    plt.close(fig_sc)
    print("A_vs_B_scatter.pdf: no data")

# %% [markdown]
# ## 3. Full star-plot gallery (concatenate per-subject PDFs)
#
# Subject-major ordering; within each subject sites are sorted by site_type
# (type-2 first) as produced by the per-subject notebook.

# %%
_star_plots_dir = Path(star_plots_dir)
_per_subj_pdfs = sorted(
    _star_plots_dir.glob("*/star_plots_early.pdf"),
    key=lambda p: p.parent.name,
)
print(f"Per-subject star_plots_early.pdf files found: {len(_per_subj_pdfs)}")
for p in _per_subj_pdfs:
    print(f"  {p.parent.name}: {p}")

if _per_subj_pdfs and _HAS_PYPDF:
    _writer = PdfWriter()
    _n_pages = 0
    for _pdf_path in tqdm(_per_subj_pdfs, desc="concatenating"):
        try:
            _reader = PdfReader(str(_pdf_path))
            for _page in _reader.pages:
                _writer.add_page(_page)
            _n_pages += len(_reader.pages)
        except Exception as exc:
            print(f"  ⚠ failed to read {_pdf_path}: {exc}")
    _out_path = OUT_DIR / "star_plots_all.pdf"
    with _out_path.open("wb") as fh:
        _writer.write(fh)
    print(f"wrote star_plots_all.pdf  ({_n_pages} pages from {len(_per_subj_pdfs)} subjects)")
elif not _HAS_PYPDF:
    print("⚠ pypdf not installed — star_plots_all.pdf skipped")
else:
    print("⚠ no per-subject PDFs found — star_plots_all.pdf skipped")
    (OUT_DIR / "star_plots_all.pdf").write_bytes(b"")

# %% [markdown]
# ## Done

# %%
print("=" * 70)
print(f"Output dir: {OUT_DIR}")
print(f"site_types: {site_types.height} rows across {site_types['subject'].n_unique()} subjects")
print("Files written:")
for _f in ["site_type_relabel.csv",
           "population_site_type_counts.pdf", "A_vs_B_scatter.pdf", "star_plots_all.pdf"]:
    _p = OUT_DIR / _f
    print(f"  {_f}: {'ok' if _p.exists() and _p.stat().st_size > 0 else 'MISSING/EMPTY'}")
print("=" * 70)
