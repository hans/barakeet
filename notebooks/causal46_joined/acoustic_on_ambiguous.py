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
# Also computes a step-tuning curve: the windowed mean HGA (behavior-
# controlled) at EVERY qualifying step, not just s_lo/s_hi. Two window
# variants are computed per cell, both from `b4_acoustic_per_window.parquet`
# (already a full searchlight, no extra bootstrap needed to pick a window):
#   - `global_best`     — best_smin/best_smax from the unrestricted s_lo/s_hi
#                          contrast (per_cell_best over the full search range).
#   - `late_excl_phon`   — best window excluding any overlap with the site's
#                          acoustic-peak window (phon_smin/phon_smax), for
#                          isolating the late acoustic/perceptual effect from
#                          cells whose global-best window sits on/near the
#                          transient acoustic response.
# Rows for both variants are concatenated into one parquet, tagged by
# `window_kind`. The gallery draws both as stacked tuning panels alongside
# the full timecourse per-step ramp (ax_acoustic).
#
# Outputs (schema-identical to b4_*.parquet plus s_lo/s_hi columns):
# - `b4_acoustic_bootstrap.parquet`
# - `b4_acoustic_per_window.parquet`
# - `b4_acoustic_per_cell.parquet`
# - `b4_acoustic_per_cell_late.parquet`
# - `acoustic_cell_manifest.parquet`
# - `b4_step_tuning.parquet`  (window_kind ∈ {global_best, late_excl_phon})
# - `star_plots_both/{powered,powered_significant}.pdf`

# %%
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as scipy_stats
from tqdm.auto import tqdm

from src._star_gallery import HAS_PYPDF, write_annotated_pdfs
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin, resampled_cmap
from src.viz_provisional import load_epochs_dict

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _acoustic_step_bootstrap import (  # noqa: E402
    bootstrap_cell_acoustic,
    exclude_overlapping_windows,
    per_cell_best,
    per_window_summary,
    step_tuning_pass,
    step_tuning_timecourse_pass,
)
from _within_completion import (  # noqa: E402
    extract_hga,
    extract_hga_trials,
    resolve_behavior_col,
)

# %% tags=["parameters"]
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
epoch_dir = "outputs/epochs_preprocessed"
trial_balance_path = "outputs/causal46_joined/trial_balance_index.csv"
b4_per_window_path = "outputs/causal46_joined/t_tests/b4_per_window.parquet"
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
late_acoustic_summary_path = "outputs/causal46_joined/acoustic_late/acoustic_late_summary.csv"
late_acoustic_results_path = "outputs/causal46_joined/acoustic_late/acoustic_late_results.csv"
a_per_window_by_word_end_path = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_by_word_end_all.parquet"
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


def late_window_smax_bound(phoneme_pair, word_end):
    """Late-window smax bound: word offset + 100ms, whichever word-end in the
    pair runs later — same convention as `dec_smax` in acoustic_late.py's
    prepare_decoder_bounds(). Deliberately NOT acoustic_search_range's
    PAIR_SMAX (200ms tail): that bound is shared with the unrestricted
    searchlight and is "briefly past word end" only for the pipeline's
    ordinary acoustic contrast, not for the late-window variant, which should
    match the acoustic_late rule's own notion of "briefly past word end".
    """
    other_word_end = next(iter(set(PHONEME_PAIR_TO_WORD_ENDS[phoneme_pair]) - {word_end}))
    return max(
        int(round((OFFSET_DICT[word_end] + 0.1 - epoch_tmin) * epoch_sfreq)),
        int(round((OFFSET_DICT[other_word_end] + 0.1 - epoch_tmin) * epoch_sfreq)),
    )


LATE_WINDOW_SMAX_BOUNDS = pl.DataFrame([
    {"phoneme_pair": pp, "word_end": we, "late_smax_bound": late_window_smax_bound(pp, we)}
    for pp, wes in PHONEME_PAIR_TO_WORD_ENDS.items() for we in wes
])
print(f"late-window smax bounds (samples): {LATE_WINDOW_SMAX_BOUNDS.to_dicts()}")


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
manifest_ok = acoustic_cell_manifest.filter(pl.col("status") == "ok")

ac_per_cell = per_cell_best(ac_per_window, CELL_KEYS)
if ac_per_cell.height:
    # Augment with fields needed for gallery regeneration.
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
# ## Per-cell best window, late variant (excludes acoustic-peak overlap)
#
# Same rank criterion as per_cell_best, but restricted to windows that don't
# overlap the site's acoustic-peak window (phon_smin/phon_smax — the ~150-
# 250ms transient acoustic response). We're especially interested in the
# late acoustic/perceptual effect, and a cell's unrestricted best window can
# land on/near the acoustic peak, masking a distinct later effect.
#
# Also caps smax at late_window_smax_bound (word offset + 100ms, matching
# acoustic_late.py's prepare_decoder_bounds dec_smax) rather than inheriting
# acoustic_search_range's PAIR_SMAX (200ms tail). PAIR_SMAX is a shared
# search bound for the unrestricted searchlight; this variant is specifically
# meant to land "briefly past word end" in the same sense as the acoustic_late
# rule, not just anywhere short of the wider unrestricted bound.

# %%
ac_per_window_late = (
    exclude_overlapping_windows(
        ac_per_window, manifest_ok, CELL_KEYS,
        excl_smin_col="phon_smin", excl_smax_col="phon_smax",
    )
    .join(LATE_WINDOW_SMAX_BOUNDS, on=["phoneme_pair", "word_end"], how="left")
    .filter(pl.col("smax") <= pl.col("late_smax_bound"))
    .drop("late_smax_bound")
)
ac_per_cell_late = per_cell_best(ac_per_window_late, CELL_KEYS)
if ac_per_cell_late.height:
    ac_per_cell_late = ac_per_cell_late.join(
        manifest_ok.select([
            "subject", "electrode_idx", "phoneme_pair", "word_end",
            "qualifying_steps", "s_lo", "s_hi", "phon_smin", "phon_smax",
        ]),
        on=CELL_KEYS, how="left",
    )
    ac_per_cell_late.write_parquet(OUT_DIR / "b4_acoustic_per_cell_late.parquet")
print(f"acoustic per_cell (late, excl phon overlap) rows: {ac_per_cell_late.height}"
      f"  (of {ac_per_cell.height} global-best cells)")

# %% [markdown]
# ## Step tuning curve (best-window-per-cell, all qualifying steps)
#
# Second pass over "ok" cells: now that best_smin/best_smax is known — both
# the unrestricted global-best window and the late/excl-phon-overlap
# variant — re-extract HGA for each cell and run step_tuning_curve in that
# one fixed window across ALL qualifying steps, not just the extremes.
# Cheap relative to the main loop (one window, not a searchlight). Both
# passes are concatenated into one parquet, disambiguated by `window_kind`.

# %%
tuning_rows: list[dict] = []
if ac_per_cell.height:
    tuning_rows += step_tuning_pass(
        ac_per_cell, epochs_dict, cell_keys=CELL_KEYS, R=R,
        window_kind="global_best", desc="step tuning (global best)",
    )
if ac_per_cell_late.height:
    tuning_rows += step_tuning_pass(
        ac_per_cell_late, epochs_dict, cell_keys=CELL_KEYS, R=R,
        window_kind="late_excl_phon", desc="step tuning (late, excl phon overlap)",
    )

step_tuning_df = pl.DataFrame(tuning_rows)
if step_tuning_df.height:
    step_tuning_df.write_parquet(OUT_DIR / "b4_step_tuning.parquet")
print(f"step tuning rows: {step_tuning_df.height}")
if step_tuning_df.height:
    print(step_tuning_df.group_by("window_kind").agg(
        pl.col("subject").n_unique().alias("n_cells_x_steps_rows")
    ))

# %% [markdown]
# ## Step tuning, full timecourse (no window collapse)
#
# `step_tuning_df` above evaluates every qualifying step inside ONE fixed
# window per cell (best_smin/best_smax). This section runs the same
# per-step bootstrap draw but keeps the whole per-sample timecourse instead
# of collapsing to a window, over the same B4 acoustic-qualified population
# ("ok" cells in acoustic_cell_manifest) — so the step-tuning gradient (or
# lack thereof) can be inspected across the entire epoch rather than at one
# pre-selected slice.
#
# Outputs:
# - `b4_step_tuning_timecourse.parquet`
# - `step_tuning_timecourse.pdf` (one page per cell; mean +/- CI per step,
#   colored by step, full epoch timecourse)

# %%
timecourse_rows = step_tuning_timecourse_pass(
    manifest_ok, epochs_dict, cell_keys=CELL_KEYS, R=R,
)
step_tuning_timecourse_df = pl.DataFrame(timecourse_rows)
if step_tuning_timecourse_df.height:
    step_tuning_timecourse_df = step_tuning_timecourse_df.with_columns(
        (pl.col("sample") / epoch_sfreq + epoch_tmin).alias("t")
    )
    step_tuning_timecourse_df.write_parquet(OUT_DIR / "b4_step_tuning_timecourse.parquet")
print(f"step tuning (full timecourse) rows: {step_tuning_timecourse_df.height}"
      f"  (cells: {manifest_ok.height})")

# %% [markdown]
# ### Plot: one page per cell, mean +/- CI HGA(t) per step

# %%
if step_tuning_timecourse_df.height:
    _tc_pd = step_tuning_timecourse_df.to_pandas()
    with PdfPages(GALLERY_DIR.parent / "step_tuning_timecourse.pdf") as pdf:
        for (subj, eidx, pp, we), cell_df in tqdm(
            _tc_pd.groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"]),
            desc="step tuning timecourse plots",
        ):
            f, ax = plt.subplots(figsize=(5, 3.5))
            for step in sorted(cell_df["step"].unique()):
                step_df = cell_df.query("step == @step").sort_values("t")
                color = resampled_cmap.get(int(step), "#999999")
                ax.plot(step_df["t"], step_df["mean"], color=color, lw=1.5,
                        label=f"step {int(step)}")
                ax.fill_between(step_df["t"], step_df["ci_lo"], step_df["ci_hi"],
                                 color=color, alpha=0.2, linewidth=0)
            ax.axvline(0, color="k", lw=0.8, linestyle=":")
            we_offset = OFFSET_DICT.get(we)
            if we_offset is not None:
                ax.axvline(we_offset, color="k", lw=0.8, linestyle="--")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("HGA (behavior-matched)")
            ax.set_title(f"{subj} e{eidx} {pp} {we}", fontsize=9)
            ax.legend(fontsize=7, frameon=False)
            f.tight_layout()
            pdf.savefig(f)
            plt.close(f)
    print(f"wrote step_tuning_timecourse.pdf ({_tc_pd.groupby(CELL_KEYS).ngroups} pages)")
else:
    print("no full-timecourse rows -- skipping plot")

# %% [markdown]
# ## Late-region acoustic gradient regression (whole-window, behavior-controlled)
#
# `b4_step_tuning` above evaluates every step at ONE fixed narrow window per
# cell (`window_size` samples, `causal46_joined.window_size` in config.yaml —
# 20ms at 100Hz — the best_smin/best_smax picked by the s_lo/s_hi searchlight).
# This section instead asks whether the late response scales gradedly with
# acoustic step across a whole late-response REGION, mirroring the "late
# acoustic regression" in plot_for_paper.ipynb:
#
# 1. Define each site's late-response region independently of the ambiguous
#    trials: union the significant per-window bootstrap windows
#    (`a_per_window_by_word_end`) that fall inside the site's significant
#    late-acoustic decoder window (`acoustic_late`, fit on unambiguous
#    step-1/step-6 trials only — see `run_acoustic_searchlight`'s
#    `resampled_steps` default). Adjacent/same-sign windows (gap <= 2
#    samples) are merged into one interval per site, exactly as in
#    plot_for_paper's `late_acoustic_windows` construction. This region is
#    defined entirely from UNAMBIGUOUS trials, so evaluating it on ambiguous
#    trials below is not circular.
# 2. Restrict to the same B4 acoustic-qualified cells used above
#    (n_qualifying_steps >= 2, n_per_class >= K) — the "highly variable
#    reports across multiple steps" subset.
# 3. Extract per-trial mean HGA over the union region on each cell's
#    qualifying ambiguous steps, sign-aligned by the union window's tuning
#    direction (mean_diff_raw_med).
# 4. Fit a mixed-effects regression of aligned HGA on acoustic step (centered,
#    continuous) controlling for reported percept (behavior_dummy_forced),
#    random intercept per site, and a likelihood-ratio test for the
#    acoustic-step coefficient — a population-level test of graded acoustic
#    tuning in the late region, independent of percept.
#
# Outputs:
# - `late_acoustic_gradient_trial_df.parquet`
# - `late_acoustic_gradient_lme_results.json`

# %%
late_results = pd.read_csv(late_acoustic_results_path)
late_summary = pd.read_csv(late_acoustic_summary_path)

late_sig_df = pd.merge(
    late_summary,
    late_results[["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]]
        .drop_duplicates(),
    how="left", on=["subject", "electrode_idx", "phoneme_pair", "word_end"], validate="m:1",
).query("significant_target")
print(f"late-acoustic significant sites (unambiguous step1/step6 decoder): {late_sig_df.shape[0]}")

a_per_window_by_we = pl.read_parquet(a_per_window_by_word_end_path).to_pandas()

# %% [markdown]
# ### Union significant per-window bootstrap results into one late-region interval per site

# %%
_GRAD_GROUP_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
_MAX_GAP = 2  # samples of separation tolerated within a union

_late_windows = (
    pd.merge(
        late_sig_df, a_per_window_by_we,
        how="left", on=_GRAD_GROUP_KEYS,
        suffixes=("_decoder", "_bootstrap"),
    )
    .query("smin_bootstrap >= smin_decoder and smax_bootstrap <= smax_decoder")
)
_late_windows["mean_diff_raw_med_abs"] = _late_windows["mean_diff_raw_med"].abs()
_late_windows = _late_windows.sort_values(_GRAD_GROUP_KEYS + ["smin_bootstrap"]).reset_index(drop=True)
_late_windows["contrast_sign"] = np.sign(_late_windows["mean_diff_raw_med"])

_run_max = _late_windows.groupby(_GRAD_GROUP_KEYS, sort=False)["smax_bootstrap"].cummax()
_new_group = (_late_windows[_GRAD_GROUP_KEYS] != _late_windows[_GRAD_GROUP_KEYS].shift(1)).any(axis=1)
_gap = _late_windows["smin_bootstrap"] > _run_max.shift(1) + _MAX_GAP
_sign_flip = _late_windows["contrast_sign"] != _late_windows["contrast_sign"].shift(1)
_starts_new = _new_group | _gap | _sign_flip
_starts_new.iloc[0] = True
_late_windows["union_id"] = _starts_new.cumsum()

late_acoustic_windows = (
    _late_windows
    .groupby(_GRAD_GROUP_KEYS + ["union_id"])
    .agg(
        smin_bootstrap=("smin_bootstrap", "min"),
        smax_bootstrap=("smax_bootstrap", "max"),
        mean_diff_raw_med_abs=("mean_diff_raw_med_abs", "mean"),
        mean_diff_raw_med=("mean_diff_raw_med", "mean"),
        n_windows=("smin_bootstrap", "size"),
    )
    .reset_index()
    # retain, per site, the union interval with the largest mean contrast
    .sort_values("mean_diff_raw_med_abs")
    .groupby(_GRAD_GROUP_KEYS).last().reset_index()
)
late_acoustic_windows["contrast_sign"] = np.sign(late_acoustic_windows["mean_diff_raw_med"])
print(f"late-acoustic union windows (one per site): {late_acoustic_windows.shape[0]}")

# %% [markdown]
# ### Restrict to B4 acoustic-qualified cells, extract per-trial late-region HGA

# %%
manifest_ok_pd = acoustic_cell_manifest.filter(pl.col("status") == "ok").to_pandas()
gradient_cells = pd.merge(
    late_acoustic_windows,
    manifest_ok_pd[_GRAD_GROUP_KEYS + ["qualifying_steps"]],
    how="inner", on=_GRAD_GROUP_KEYS,
)
print(f"gradient-eligible cells (late-acoustic-significant ∩ B4-acoustic-qualified): "
      f"{gradient_cells.shape[0]} / {late_acoustic_windows.shape[0]} late-acoustic union windows")

gradient_trial_rows: list[pd.DataFrame] = []
for row in tqdm(gradient_cells.itertuples(), total=gradient_cells.shape[0],
                 desc="late-region gradient trials"):
    subj = row.subject
    if subj not in epochs_dict:
        continue
    ep = epochs_dict[subj]
    hga_cell, md_cell = extract_hga_trials(ep, int(row.electrode_idx), row.phoneme_pair, row.word_end)
    steps = [int(s) for s in row.qualifying_steps.split(",") if s]
    step_mask = md_cell["resampled"].isin(steps).values
    if not step_mask.any():
        continue
    bhv_col = resolve_behavior_col(md_cell)
    smin_i, smax_i = int(row.smin_bootstrap), int(row.smax_bootstrap)
    hga_region = hga_cell[:, smin_i:smax_i + 1].mean(axis=1)  # inclusive upper bound

    gradient_trial_rows.append(pd.DataFrame({
        "subject": subj,
        "electrode_idx": int(row.electrode_idx),
        "phoneme_pair": row.phoneme_pair,
        "word_end": row.word_end,
        "resampled": md_cell.loc[step_mask, "resampled"].values,
        "behavior_dummy_forced": md_cell.loc[step_mask, bhv_col].values,
        "hga_region": hga_region[step_mask],
        # sign-flipped by the union window's tuning direction, so a pooled
        # regression slope is interpretable across sites tuned in opposite
        # directions (same convention as mean_diff_aligned elsewhere in this
        # codebase).
        "hga_region_aligned": hga_region[step_mask] * row.contrast_sign,
        "smin_bootstrap": smin_i,
        "smax_bootstrap": smax_i,
    }))

late_acoustic_gradient_trial_df = (
    pd.concat(gradient_trial_rows, ignore_index=True) if gradient_trial_rows else pd.DataFrame()
)
if late_acoustic_gradient_trial_df.shape[0]:
    late_acoustic_gradient_trial_df.to_parquet(OUT_DIR / "late_acoustic_gradient_trial_df.parquet")
print(f"late-region gradient trial rows: {late_acoustic_gradient_trial_df.shape[0]}")

# %% [markdown]
# ### Population-level LME: graded acoustic-step effect, percept controlled
#
# Ambiguous steps only by construction (qualifying_steps never includes the
# endpoints 1/6). `resampled_centered` is step minus the continuum midpoint
# (3.5), so the coefficient is directly comparable to plot_for_paper's
# late-acoustic regression.

# %%
gradient_lme_results: dict = {}
if late_acoustic_gradient_trial_df.shape[0]:
    reg_df = late_acoustic_gradient_trial_df.copy()
    reg_df["resampled_centered"] = reg_df["resampled"] - 3.5
    reg_df["site"] = (
        reg_df["subject"].astype(str) + ":" + reg_df["electrode_idx"].astype(str) + ":"
        + reg_df["phoneme_pair"] + ":" + reg_df["word_end"]
    )
    model_base = smf.mixedlm(
        "hga_region_aligned ~ behavior_dummy_forced",
        data=reg_df, groups=reg_df["site"],
    ).fit(reml=False)  # ML, to match the LRT comparison
    model_full = smf.mixedlm(
        "hga_region_aligned ~ resampled_centered + behavior_dummy_forced",
        data=reg_df, groups=reg_df["site"],
    ).fit(reml=False)
    lr_stat = 2 * (model_full.llf - model_base.llf)
    lr_p = scipy_stats.chi2.sf(lr_stat, df=1)

    gradient_lme_results = {
        "n_trials": int(reg_df.shape[0]),
        "n_sites": int(reg_df["site"].nunique()),
        "acoustic_step_coef": float(model_full.params["resampled_centered"]),
        "acoustic_step_se": float(model_full.bse["resampled_centered"]),
        "acoustic_step_wald_p": float(model_full.pvalues["resampled_centered"]),
        "lr_stat": float(lr_stat),
        "lr_p": float(lr_p),
    }
    print(model_full.summary())
    print(f"Likelihood ratio test for graded acoustic-step effect: "
          f"χ²(1) = {lr_stat:.3f}, p = {lr_p:.3g}")
else:
    print("no gradient trial rows -- skipping LME")

with (OUT_DIR / "late_acoustic_gradient_lme_results.json").open("w") as f:
    json.dump(gradient_lme_results, f, indent=2)
print(f"wrote late_acoustic_gradient_lme_results.json: {gradient_lme_results}")

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
            step_tuning_df=step_tuning_df,
        )
        print(f"  {label}.pdf: {n_written} pages written → {out_pdf}")

print("done.")
