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
# # Acoustic step contrast on ambiguous trials (behavior-controlled)
#
# Mirror of the B4 perceptual bootstrap (`t_tests.py`): contrasts the extreme
# qualifying steps (s_hi vs s_lo) on ambiguous trials while holding behavioral
# report fixed at 50/50 per step. The two contrasts share the same bootstrap
# draw (`select_cell_trials_bootstrap_perstep` makes identical RNG calls to
# `select_cell_trials_bootstrap`), so perceptual and acoustic facets are
# orthogonal matched estimands on the same replicates.
#
# Scope: B4 cells with n_qualifying_steps ≥ 2 only.
#
# Outputs (schema-identical to b4_*.parquet plus s_lo/s_hi columns):
# - `b4_acoustic_bootstrap.parquet`
# - `b4_acoustic_per_window.parquet`
# - `b4_acoustic_per_cell.parquet`
# - `acoustic_cell_manifest.parquet`
# - `star_plots_both/{powered,powered_significant}.pdf`

# %%
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from tqdm.auto import tqdm

from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin
from src.viz_provisional import load_epochs_dict

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    extract_hga,
    per_step_class_counts,
    resolve_behavior_col,
)
from _contrasts import (  # noqa: E402
    bootstrap_cell_acoustic,
    per_cell_best,
    per_window_summary,
)
from _star_gallery import HAS_PYPDF, write_annotated_pdfs  # noqa: E402

# %% tags=["parameters"]
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
epoch_dir = "outputs/epochs_preprocessed"
trial_balance_path = "outputs/causal46_joined/trial_balance_index.csv"
b4_per_window_path = "outputs/causal46_joined/t_tests/b4_per_window.parquet"
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
outdir = "outputs/causal46_joined/acoustic_on_ambiguous"
min_class_k = 4
window_size = 10
stride = 10
ac_p_value_threshold = 0.001
n_bootstrap = 1000

# %%
REPO = Path(".").resolve()
OUT_DIR = Path(outdir)
GALLERY_DIR = OUT_DIR / "star_plots_both"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_DIR = Path(epoch_dir)
CAUSAL6_PEAKS = Path(phon_peaks_path)

_cfg = yaml.safe_load((REPO / "config.yaml").read_text())
WINDOW_SIZE = window_size
STRIDE = stride
WORD_END_TAIL_SAMPLES = 20  # +200 ms past word offset (sfreq=100)

AC_P_VALUE_THRESHOLD = ac_p_value_threshold

K = min_class_k
R = n_bootstrap
CI_LOW, CI_HIGH = 2.5, 97.5

print(f"REPO:      {REPO}")
print(f"EPOCH_DIR: {EPOCH_DIR}  (exists: {EPOCH_DIR.exists()})")
print(f"K = {K}   R = {R}   window={WINDOW_SIZE}  stride={STRIDE}")

# %% [markdown]
# ## Load AS sites, trial balance, and epochs

# %%
_peaks_raw = pl.read_parquet(CAUSAL6_PEAKS)
peaks = _peaks_raw.filter(pl.col("p_value") < AC_P_VALUE_THRESHOLD)
print(f"AS sites: {peaks.height}")

trial_balance = pl.read_csv(trial_balance_path)
print(f"trial_balance: {trial_balance.height} rows")

epochs_dict = load_epochs_dict(EPOCH_DIR)
print(f"epochs loaded: {sorted(epochs_dict)}")

# %% [markdown]
# ## Search range (matches behav_search_range in t_tests.py)

# %%
def word_end_search_smax(word_end):
    offset_s = OFFSET_DICT[word_end]
    sample = int(round((offset_s - epoch_tmin) * epoch_sfreq))
    return sample + WORD_END_TAIL_SAMPLES


WE_SMAX = {we: word_end_search_smax(we) for we in OFFSET_DICT.keys()}
PAIR_SMAX = {
    pp: max(WE_SMAX[we] for we in wes)
    for pp, wes in PHONEME_PAIR_TO_WORD_ENDS.items()
}
print(f"pair search_smax (samples): {PAIR_SMAX}")


def acoustic_search_range(phoneme_pair):
    """Search range from onset to pair-level word-end maximum.

    Matches behav_search_range(pp, phon_smax) → (0, PAIR_SMAX[pp]) exactly,
    so acoustic and perceptual per-window grids share identical (smin, smax) keys.
    Note: acoustic_peak_search_smin in config (45) applies only to causal6 peak
    detection, not to this bootstrap search.
    """
    return 0, int(PAIR_SMAX[phoneme_pair])


# %% [markdown]
# ## B4 acoustic-qualified cell list (n_qualifying_steps ≥ 2)

# %%
b4_acoustic_qualified = (
    trial_balance
    .filter(pl.col("is_ambiguous_step"))
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.col("resampled").sort().alias("qualifying_steps"),
        pl.col("min_class").sum().alias("n_per_class"),
        pl.len().alias("n_qualifying"),
    )
    .filter((pl.col("n_qualifying") >= 2) & (pl.col("n_per_class") >= K))
    .join(
        peaks.select(["subject", "electrode_idx", "phoneme_pair",
                      "smin", "smax", "test_roc_auc"])
             .rename({"smin": "phon_smin", "smax": "phon_smax",
                      "test_roc_auc": "acoustic_peak_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
)
print(f"Acoustic-qualified B4 cells (n_qualifying ≥ 2, n_per_class ≥ {K}): "
      f"{b4_acoustic_qualified.height}")

# %% [markdown]
# ## Acoustic bootstrap loop

# %%
ac_boot_rows: list[dict] = []
ac_cell_manifest: list[dict] = []
ac_failures: list[dict] = []

for row in tqdm(b4_acoustic_qualified.iter_rows(named=True),
                total=b4_acoustic_qualified.height, desc="acoustic bootstrap"):
    subj = row["subject"]
    if subj not in epochs_dict:
        ac_failures.append({**row, "error": "no epochs for subject"})
        continue
    ep = epochs_dict[subj]
    md = ep.metadata
    bhv_col = resolve_behavior_col(md)
    pp_mask = (md["phoneme_pair"] == row["phoneme_pair"]).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = extract_hga(ep_pp, int(row["electrode_idx"]))
    search_smin, search_smax = acoustic_search_range(row["phoneme_pair"])
    steps = [int(s) for s in row["qualifying_steps"]]
    if search_smax - search_smin < WINDOW_SIZE:
        ac_cell_manifest.append({
            "subject": subj,
            "electrode_idx": int(row["electrode_idx"]),
            "phoneme_pair": row["phoneme_pair"],
            "word_end": row["word_end"],
            "qualifying_steps": ",".join(str(s) for s in steps),
            "n_qualifying_steps": len(steps),
            "n_per_class": int(row["n_per_class"]),
            "s_lo": None, "s_hi": None,
            "phon_smin": None, "phon_smax": None, "acoustic_peak_auc": None,
            "status": "search_range_too_narrow",
        })
        continue
    try:
        result = bootstrap_cell_acoustic(
            md_pp=md_pp, hga=hga, bhv_col=bhv_col,
            word_end=row["word_end"], qualifying_steps=steps,
            acoustic_peak_auc=float(row["acoustic_peak_auc"]),
            search_smin=search_smin, search_smax=search_smax,
            window_size=WINDOW_SIZE, stride=STRIDE,
            R=R,
        )
        if result is None:
            ac_cell_manifest.append({
                "subject": subj,
                "electrode_idx": int(row["electrode_idx"]),
                "phoneme_pair": row["phoneme_pair"],
                "word_end": row["word_end"],
                "qualifying_steps": ",".join(str(s) for s in steps),
                "n_qualifying_steps": len(steps),
                "n_per_class": int(row["n_per_class"]),
                "s_lo": None, "s_hi": None,
                "phon_smin": None, "phon_smax": None, "acoustic_peak_auc": None,
                "status": "insufficient_extreme_steps",
            })
            continue
        rows, s_lo, s_hi = result
        for r in rows:
            ac_boot_rows.append({
                "subject": subj,
                "electrode_idx": int(row["electrode_idx"]),
                "phoneme_pair": row["phoneme_pair"],
                "word_end": row["word_end"],
                "qualifying_steps": ",".join(str(s) for s in steps),
                "n_qualifying_steps": len(steps),
                "acoustic_peak_auc": float(row["acoustic_peak_auc"]),
                **r,
            })
        ac_cell_manifest.append({
            "subject": subj,
            "electrode_idx": int(row["electrode_idx"]),
            "phoneme_pair": row["phoneme_pair"],
            "word_end": row["word_end"],
            "qualifying_steps": ",".join(str(s) for s in steps),
            "n_qualifying_steps": len(steps),
            "n_per_class": int(rows[0]["n_per_class"]) if rows else int(row["n_per_class"]),
            "s_lo": s_lo, "s_hi": s_hi,
            "status": "ok",
            "phon_smin": int(row["phon_smin"]),
            "phon_smax": int(row["phon_smax"]),
            "acoustic_peak_auc": float(row["acoustic_peak_auc"]),
        })
    except Exception as exc:
        tb = traceback.format_exc()
        ac_failures.append({
            **{k: row[k] for k in
               ("subject", "electrode_idx", "phoneme_pair", "word_end")},
            "qualifying_steps": ",".join(str(s) for s in steps),
            "error": repr(exc), "traceback": tb,
        })
        print(f"FAILED: {subj} e{row['electrode_idx']} {row['phoneme_pair']} "
              f"{row['word_end']}\n{tb}")

ac_boot = pl.DataFrame(ac_boot_rows) if ac_boot_rows else pl.DataFrame()
if ac_boot.height:
    ac_boot.write_parquet(OUT_DIR / "b4_acoustic_bootstrap.parquet")
print(f"acoustic bootstrap rows: {ac_boot.height}  (failures: {len(ac_failures)})")

acoustic_cell_manifest = pl.DataFrame(ac_cell_manifest)
acoustic_cell_manifest.write_parquet(OUT_DIR / "acoustic_cell_manifest.parquet")
print(f"acoustic_cell_manifest: {acoustic_cell_manifest.height} rows")
print(acoustic_cell_manifest.group_by(["status"]).len().sort(["status"]))

# %% [markdown]
# ## Per-window aggregation

# %%
CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

ac_per_window = per_window_summary(ac_boot, CELL_KEYS)
if ac_per_window.height:
    ac_per_window.write_parquet(OUT_DIR / "b4_acoustic_per_window.parquet")
print(f"acoustic per_window rows: {ac_per_window.height}")

# %% [markdown]
# ## Per-cell best window

# %%
ac_per_cell = per_cell_best(ac_per_window, CELL_KEYS)
if ac_per_cell.height:
    # Augment with fields needed for gallery regeneration.
    manifest_ok = acoustic_cell_manifest.filter(pl.col("status") == "ok")
    ac_per_cell = ac_per_cell.join(
        manifest_ok.select([
            "subject", "electrode_idx", "phoneme_pair", "word_end",
            "qualifying_steps", "s_lo", "s_hi", "phon_smin", "phon_smax",
        ]),
        on=CELL_KEYS, how="left",
    )
    ac_per_cell.write_parquet(OUT_DIR / "b4_acoustic_per_cell.parquet")
print(f"acoustic per_cell rows: {ac_per_cell.height}")

if ac_per_cell.height:
    n_sig = ac_per_cell.filter(pl.col("best_ci_aligned_excludes_zero")).height
    print(f"  cells with CI excludes 0: {n_sig} / {ac_per_cell.height}")

# %% [markdown]
# ## Combined gallery (behavior + acoustic facets)

# %%
b4_per_window = pl.read_parquet(b4_per_window_path) if Path(b4_per_window_path).exists() else pl.DataFrame()
b4_per_cell = pl.read_parquet(b4_per_cell_path) if Path(b4_per_cell_path).exists() else pl.DataFrame()

if not HAS_PYPDF:
    print("pypdf not available — skipping gallery PDF generation")
elif b4_per_cell.height == 0:
    print("b4_per_cell empty — no behavior data for combined gallery")
elif ac_per_cell.height == 0:
    print("ac_per_cell empty — no acoustic data for combined gallery")
else:
    # Build gallery entry list: acoustic-ok cells that also appear in b4_per_cell.
    # (inner join on CELL_KEYS only — we use ac_ok's columns for the plot params)
    ac_ok = acoustic_cell_manifest.filter(pl.col("status") == "ok")
    gallery_cells = ac_ok.join(
        b4_per_cell.select(CELL_KEYS),
        on=CELL_KEYS, how="inner",
    )
    print(f"combined gallery cells: {gallery_cells.height}")

    # Powered: CI excludes zero in acoustic best window.
    ac_sig_set = set(
        (r["subject"], r["electrode_idx"], r["phoneme_pair"], r["word_end"])
        for r in ac_per_cell.filter(pl.col("best_ci_aligned_excludes_zero")).iter_rows(named=True)
    )

    all_entries = gallery_cells.iter_rows(named=True)

    powered_entries = list(gallery_cells.iter_rows(named=True))
    sig_entries = [
        r for r in powered_entries
        if (r["subject"], r["electrode_idx"], r["phoneme_pair"], r["word_end"]) in ac_sig_set
    ]

    for label, entries in [("powered", powered_entries), ("powered_significant", sig_entries)]:
        out_pdf = GALLERY_DIR / f"{label}.pdf"
        n_written = write_annotated_pdfs(
            entries=entries,
            per_window=b4_per_window,
            cell_keys=CELL_KEYS,
            out_path=out_pdf,
            epochs_dict=epochs_dict,
            acoustic_per_window=ac_per_window,
            acoustic_R_plot=200,
        )
        print(f"  {label}.pdf: {n_written} pages written → {out_pdf}")

print("done.")
