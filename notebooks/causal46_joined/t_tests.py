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
# # B4 within-completion behavior bootstrap CIs (JON-44)
#
# Per-AS-site searchlight bootstrap of the within-completion HGA contrast
# under **per-step class balance** (the same trial-selection rule as the
# refactored star_plots gallery on `causal6-speech-responsive-update`):
#
# - For each (site × word_end × qualifying step `s`), pick `min_class[s]`
#   trials per class. Both classes are bootstrapped with replacement to
#   that size. Concat across qualifying steps → `sum_s min_class[s]`
#   trials per class per replicate (constant within cell, varies between
#   cells).
# - For each bootstrap replicate, compute the searchlight mean-HGA
#   difference per window. Two signed variants:
#     - `mean_diff_raw`     = mean(HGA[class==1]) − mean(HGA[class==0])
#     - `mean_diff_aligned` = mean(HGA[acoustic-preferred class])
#                             − mean(HGA[non-preferred class])
#   where the acoustic-preferred class is determined per cell by which
#   behavior label shows higher mean HGA in the cell's acoustic window
#   (causal6 peak smin/smax). If the two class means are tied,
#   `mean_diff_aligned = NaN` for the cell and the aligned summary skips it.
# - R = 1000 bootstrap replicates per cell. Per (cell × window) we report
#   median, 2.5/97.5 percentile, std, and a bootstrap 2-sided empirical
#   p-value. `ci_excludes_zero` is the headline boolean for cell-window
#   significance. No FDR for now (TBD).
#
# Cells with `n_per_class < K` (= K from star_plots.py) are flagged
# underpowered and excluded from the bootstrap loop; they appear in the
# manifest with status='underpowered'.
#
# Outputs:
# - `outputs/causal46_joined/t_tests/b4_bootstrap.parquet` — per (cell × window × replicate)
# - `outputs/causal46_joined/t_tests/b4_per_window.parquet` — aggregates
# - `outputs/causal46_joined/t_tests/b4_per_cell.parquet` — best-window per cell (augmented with pair columns)
# - `outputs/causal46_joined/t_tests/b4_per_pair.parquet` — pair-level CI per (subject × electrode × phoneme_pair)
# - `outputs/causal46_joined/t_tests/cell_manifest.parquet`
# - `outputs/causal46_joined/t_tests/population_summary.csv`
# - `outputs/causal46_joined/t_tests/population_summary.pdf`
# - `outputs/causal46_joined/t_tests/star_plots_filtered/{b4}_{powered,powered_significant}.pdf`
# - `outputs/causal46_joined/t_tests/star_plots_filtered/filtered_manifest.csv`

# %%
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from src.data import get_electrode_df
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin
from src.viz_provisional import load_epochs_dict

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    acoustic_preferred_class,
    extract_hga,
    matched_n_star_plot,
    n_per_class_from_per_step,
    per_step_class_counts,
    resolve_behavior_col,
    searchlight_mean_diff,
    select_cell_trials_bootstrap,
)

# %% tags=["parameters"]
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
epoch_dir = "outputs/epochs_preprocessed"
trial_balance_path = "outputs/causal46_joined/trial_balance_index.csv"
outdir = "outputs/causal46_joined/t_tests"
min_class_k = 4
window_size = 10
stride = 10
ac_p_value_threshold = 0.001

# %%
REPO = Path(".").resolve()
OUT_DIR = Path(outdir)
FILT_DIR = OUT_DIR / "star_plots_filtered"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FILT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_DIR = Path(epoch_dir)
CAUSAL6_PEAKS = Path(phon_peaks_path)

_cfg = yaml.safe_load((REPO / "config.yaml").read_text())
WINDOW_SIZE = window_size
STRIDE = stride
WORD_END_TAIL_SAMPLES = 20  # +200 ms past word offset (sfreq=100)

AC_P_VALUE_THRESHOLD = ac_p_value_threshold
AC_SEARCH_SMIN = int(_cfg["analysis"]["decoding"].get("acoustic_peak_search_smin", 0))
AC_SEARCH_SMAX = int(_cfg["analysis"]["decoding"].get("acoustic_peak_search_smax", 50))

K = min_class_k
R = 1000              # bootstrap replicates per cell
CI_LOW, CI_HIGH = 2.5, 97.5

print(f"REPO:      {REPO}")
print(f"EPOCH_DIR: {EPOCH_DIR}  (exists: {EPOCH_DIR.exists()})")
print(f"K = {K}   R = {R}   CI = [{CI_LOW}, {CI_HIGH}]")
print(f"behavioral search range = (phon_smax_c6,  word_offset_sample + {WORD_END_TAIL_SAMPLES})")
print(f"window={WINDOW_SIZE}  stride={STRIDE}")

# %% [markdown]
# ## Load AS sites, trial balance, and epochs

# %%
_peaks_raw = pl.read_parquet(CAUSAL6_PEAKS)
# if "significant" in _peaks_raw.columns:
#     peaks = _peaks_raw.filter(pl.col("significant"))
# else:
peaks = _peaks_raw.filter(pl.col("p_value") < AC_P_VALUE_THRESHOLD)
print(f"using p_value < {AC_P_VALUE_THRESHOLD} (uncorrected)")
print(f"AS sites: {peaks.height}")

trial_balance = pl.read_csv(trial_balance_path)
print(f"trial_balance: {trial_balance.height} rows")

epochs_dict = load_epochs_dict(EPOCH_DIR)
print(f"epochs loaded: {sorted(epochs_dict)}")

# %% [markdown]
# ## Word-end behavioral search bound (samples)

# %%
def word_end_search_smax(word_end: str) -> int:
    offset_s = OFFSET_DICT[word_end]
    sample = int(round((offset_s - epoch_tmin) * epoch_sfreq))
    return sample + WORD_END_TAIL_SAMPLES


WE_SMAX = {we: word_end_search_smax(we) for we in OFFSET_DICT.keys()}
print(f"word-end search_smax (samples): {WE_SMAX}")

# Match the star-plot x-axis (shared across WEs in a site+pair).
PAIR_SMAX = {
    pp: max(WE_SMAX[we] for we in wes)
    for pp, wes in PHONEME_PAIR_TO_WORD_ENDS.items()
}
print(f"pair search_smax (samples): {PAIR_SMAX}")


def behav_search_range(phoneme_pair: str, phon_smax_c6: int) -> tuple[int, int]:
    # return int(phon_smax_c6), int(WE_SMAX[word_end])
    # DEV: just do search from onset onward, extend smax to the pair-level max
    # so the bootstrap covers the same window the star plot now shows.
    return 0, int(PAIR_SMAX[phoneme_pair])


# %% [markdown]
# ## B4 cell definition (per-step balanced pool across ambiguous steps)

# %%
b4_qualified = (
    trial_balance
    .filter(pl.col("is_ambiguous_step"))
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.col("resampled").sort().alias("qualifying_steps"),
        pl.col("min_class").sum().alias("n_per_class"),
        pl.len().alias("n_qualifying"),
    )
    .filter((pl.col("n_qualifying") >= 1) & (pl.col("n_per_class") >= K))
    .join(
        peaks.select(["subject", "electrode_idx", "phoneme_pair",
                      "smin", "smax", "test_roc_auc"])
             .rename({"smin": "phon_smin", "smax": "phon_smax",
                      "test_roc_auc": "acoustic_peak_auc"}),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
)
print(f"B4 qualifying cells (n_qualifying ≥ 1, n_per_class ≥ {K}): "
      f"{b4_qualified.height}")

# %% [markdown]
# ## Bootstrap loop

# %%
def bootstrap_cell(
    *,
    md_pp,
    hga,
    bhv_col: str,
    word_end: str,
    qualifying_steps: list[int],
    acoustic_smin: int,
    acoustic_smax: int,
    behav_smin: int,
    behav_smax: int,
    R: int,
    base_seed: int = 0,
) -> tuple[list[dict], int, int | None]:
    """Run R bootstrap replicates of the cell's mean-diff searchlight.

    Returns (rows_per_replicate_window, n_per_class, preferred_class).
    Preferred class is None for tied cells; rows still carry mean_diff_raw
    and mean_diff_aligned = NaN.
    """
    per_step = per_step_class_counts(
        md_pp, word_end=word_end, qualifying_steps=qualifying_steps,
        group_col=bhv_col,
    )
    n_per_class = n_per_class_from_per_step(per_step)
    preferred = acoustic_preferred_class(
        hga, md_pp, group_col=bhv_col, word_end=word_end,
        acoustic_smin=acoustic_smin, acoustic_smax=acoustic_smax,
    )
    rows: list[dict] = []
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        draws = select_cell_trials_bootstrap(per_step, rng=rng)
        keys = sorted(draws.keys())
        # raw: pos = class==1 (or larger key), neg = class==0 (or smaller key)
        raw_pos_key, raw_neg_key = keys[1], keys[0]
        res = searchlight_mean_diff(
            hga, draws[raw_pos_key], draws[raw_neg_key],
            search_smin=behav_smin, search_smax=behav_smax,
            window_size=WINDOW_SIZE, stride=STRIDE,
        )
        for w in res:
            mean_diff_raw = w.mean_diff
            if preferred is None:
                mean_diff_aligned = float("nan")
            else:
                # If raw_pos_key == preferred, aligned = raw; else aligned = -raw
                mean_diff_aligned = mean_diff_raw if raw_pos_key == preferred \
                    else -mean_diff_raw
            rows.append({
                "replicate": r,
                "smin": w.smin, "smax": w.smax,
                "tmin": w.smin / epoch_sfreq + epoch_tmin,
                "tmax": w.smax / epoch_sfreq + epoch_tmin,
                "mean_pos_raw": w.mean_pos,
                "mean_neg_raw": w.mean_neg,
                "mean_diff_raw": mean_diff_raw,
                "mean_diff_aligned": mean_diff_aligned,
                "n_per_class": n_per_class,
            })
    return rows, n_per_class, preferred


# %% [markdown]
# ## Run B4 cells

# %%
b4_boot_rows: list[dict] = []
b4_cell_manifest: list[dict] = []
b4_failures: list[dict] = []

for row in tqdm(b4_qualified.iter_rows(named=True),
                total=b4_qualified.height, desc="B4 bootstrap"):
    subj = row["subject"]
    if subj not in epochs_dict:
        b4_failures.append({**row, "error": "no epochs for subject"})
        continue
    ep = epochs_dict[subj]
    md = ep.metadata
    bhv_col = resolve_behavior_col(md)
    pp_mask = (md["phoneme_pair"] == row["phoneme_pair"]).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = extract_hga(ep_pp, int(row["electrode_idx"]))
    behav_smin, behav_smax = behav_search_range(row["phoneme_pair"], row["phon_smax"])
    steps = [int(s) for s in row["qualifying_steps"]]
    if behav_smax - behav_smin < WINDOW_SIZE:
        b4_cell_manifest.append({
            "subject": subj,
            "electrode_idx": int(row["electrode_idx"]),
            "phoneme_pair": row["phoneme_pair"],
            "word_end": row["word_end"],
            "mode": "matched_n",
            "resampled_step": None,
            "qualifying_steps": ",".join(str(s) for s in steps),
            "n_per_class": int(row["n_per_class"]),
            "preferred_class": None,
            "status": "search_range_too_narrow",
            "behav_smin": behav_smin, "behav_smax": behav_smax,
        })
        continue
    try:
        rows, n_per_class, preferred = bootstrap_cell(
            md_pp=md_pp, hga=hga, bhv_col=bhv_col,
            word_end=row["word_end"], qualifying_steps=steps,
            acoustic_smin=int(row["phon_smin"]),
            acoustic_smax=int(row["phon_smax"]),
            behav_smin=behav_smin, behav_smax=behav_smax,
            R=R,
        )
        for r in rows:
            b4_boot_rows.append({
                "subject": subj,
                "electrode_idx": int(row["electrode_idx"]),
                "phoneme_pair": row["phoneme_pair"],
                "word_end": row["word_end"],
                "resampled": None,
                "qualifying_steps": ",".join(str(s) for s in steps),
                "n_qualifying_steps": len(steps),
                "mode": "matched_n",
                "acoustic_peak_auc": float(row["acoustic_peak_auc"]),
                **r,
            })
        b4_cell_manifest.append({
            "subject": subj,
            "electrode_idx": int(row["electrode_idx"]),
            "phoneme_pair": row["phoneme_pair"],
            "word_end": row["word_end"],
            "mode": "matched_n",
            "resampled_step": None,
            "qualifying_steps": ",".join(str(s) for s in steps),
            "n_per_class": n_per_class,
            "preferred_class": preferred,
            "status": "ok",
            "behav_smin": behav_smin, "behav_smax": behav_smax,
        })
    except Exception as exc:
        tb = traceback.format_exc()
        b4_failures.append({
            **{k: row[k] for k in
               ("subject", "electrode_idx", "phoneme_pair", "word_end")},
            "qualifying_steps": ",".join(str(s) for s in steps),
            "error": repr(exc), "traceback": tb,
        })
        print(f"FAILED B4: {subj} e{row['electrode_idx']} {row['phoneme_pair']} "
              f"{row['word_end']}\n{tb}")

# Underpowered B4 candidates
b4_drops = (
    trial_balance
    .filter(pl.col("is_ambiguous_step"))
    .join(
        peaks.select(["subject", "electrode_idx", "phoneme_pair"]),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.len().alias("n_ambig_steps"),
        pl.col("min_class").sum().alias("n_per_class_pool"),
    )
    .filter((pl.col("n_ambig_steps") < 1) | (pl.col("n_per_class_pool") < K))
)
for row in b4_drops.iter_rows(named=True):
    b4_cell_manifest.append({
        "subject": row["subject"],
        "electrode_idx": int(row["electrode_idx"]),
        "phoneme_pair": row["phoneme_pair"],
        "word_end": row["word_end"],
        "mode": "matched_n",
        "resampled_step": None,
        "qualifying_steps": "",
        "n_per_class": int(row["n_per_class_pool"]),
        "preferred_class": None,
        "status": "underpowered",
        "behav_smin": None, "behav_smax": None,
    })

b4_boot = pl.DataFrame(b4_boot_rows) if b4_boot_rows else pl.DataFrame()
if b4_boot.height:
    b4_boot.write_parquet(OUT_DIR / "b4_bootstrap.parquet")
print(f"B4 bootstrap rows: {b4_boot.height}  (failures: {len(b4_failures)})")

cell_manifest = pl.DataFrame(b4_cell_manifest)
cell_manifest.write_parquet(OUT_DIR / "cell_manifest.parquet")
print(f"cell_manifest: {cell_manifest.height} rows")
print(cell_manifest.group_by(["status"]).len().sort(["status"]))

# %% [markdown]
# ## Per-window aggregation (CI + empirical p)

# %%
def per_window_summary(boot: pl.DataFrame, cell_keys: list[str]) -> pl.DataFrame:
    if boot.height == 0:
        return pl.DataFrame()
    grouped = (
        boot
        .group_by(cell_keys + ["smin", "smax", "tmin", "tmax"])
        .agg(
            pl.col("mean_diff_raw").median().alias("mean_diff_raw_med"),
            pl.col("mean_diff_raw").quantile(CI_LOW / 100).alias("mean_diff_raw_ci_lo"),
            pl.col("mean_diff_raw").quantile(CI_HIGH / 100).alias("mean_diff_raw_ci_hi"),
            pl.col("mean_diff_raw").std().alias("mean_diff_raw_std"),
            (pl.col("mean_diff_raw") <= 0).cast(pl.Float64).mean().alias("frac_raw_le0"),
            (pl.col("mean_diff_raw") >= 0).cast(pl.Float64).mean().alias("frac_raw_ge0"),

            pl.col("mean_diff_aligned").median().alias("mean_diff_aligned_med"),
            pl.col("mean_diff_aligned").mean().alias("mean_diff_aligned_mean"),
            pl.col("mean_diff_aligned").quantile(CI_LOW / 100).alias("mean_diff_aligned_ci_lo"),
            pl.col("mean_diff_aligned").quantile(CI_HIGH / 100).alias("mean_diff_aligned_ci_hi"),
            pl.col("mean_diff_aligned").std().alias("mean_diff_aligned_std"),
            (pl.col("mean_diff_aligned") <= 0).cast(pl.Float64).mean().alias("frac_aligned_le0"),
            (pl.col("mean_diff_aligned") >= 0).cast(pl.Float64).mean().alias("frac_aligned_ge0"),

            pl.col("n_per_class").first().alias("n_per_class"),
            pl.col("acoustic_peak_auc").first().alias("acoustic_peak_auc"),
            pl.col("replicate").max().alias("R_replicates"),
        )
    )
    grouped = grouped.with_columns([
        # Empirical 2-sided bootstrap p: 2 * min(frac<=0, frac>=0). Clamped to 1.
        pl.min_horizontal(
            2 * pl.min_horizontal("frac_raw_le0", "frac_raw_ge0"),
            pl.lit(1.0),
        ).alias("emp_p_raw"),
        pl.min_horizontal(
            2 * pl.min_horizontal("frac_aligned_le0", "frac_aligned_ge0"),
            pl.lit(1.0),
        ).alias("emp_p_aligned"),
        # CI excludes 0: lo > 0 OR hi < 0.
        ((pl.col("mean_diff_raw_ci_lo") > 0) | (pl.col("mean_diff_raw_ci_hi") < 0))
            .alias("ci_raw_excludes_zero"),
        ((pl.col("mean_diff_aligned_ci_lo") > 0) | (pl.col("mean_diff_aligned_ci_hi") < 0))
            .alias("ci_aligned_excludes_zero"),
    ])
    return grouped.sort(cell_keys + ["smin"])


b4_cell_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
b4_per_window = per_window_summary(b4_boot, b4_cell_keys)
if b4_per_window.height:
    b4_per_window.write_parquet(OUT_DIR / "b4_per_window.parquet")
print(f"B4 per_window rows: {b4_per_window.height}")

# %% [markdown]
# ## Per-cell best-window summary
#
# Best window = window with largest |median(mean_diff_aligned)|. Falls back
# to |raw| when aligned is NaN (tied tuning).

# %%
def per_cell_best(per_window: pl.DataFrame, cell_keys: list[str]) -> pl.DataFrame:
    if per_window.height == 0:
        return pl.DataFrame()
    return (
        per_window
        .with_columns([
            pl.when(pl.col("mean_diff_aligned_med").is_not_null())
                .then(pl.col("mean_diff_aligned_med").abs())
                .otherwise(pl.col("mean_diff_raw_med").abs())
                .alias("__rank"),
        ])
        .sort(cell_keys + ["__rank"], descending=[False] * len(cell_keys) + [True])
        .group_by(cell_keys, maintain_order=True)
        .head(1)
        .drop("__rank")
        .rename({
            "smin": "best_smin", "smax": "best_smax",
            "tmin": "best_tmin", "tmax": "best_tmax",
            "mean_diff_raw_med": "best_mean_diff_raw_med",
            "mean_diff_raw_ci_lo": "best_mean_diff_raw_ci_lo",
            "mean_diff_raw_ci_hi": "best_mean_diff_raw_ci_hi",
            "mean_diff_aligned_med": "best_mean_diff_aligned_med",
            "mean_diff_aligned_ci_lo": "best_mean_diff_aligned_ci_lo",
            "mean_diff_aligned_ci_hi": "best_mean_diff_aligned_ci_hi",
            "emp_p_raw": "best_emp_p_raw",
            "emp_p_aligned": "best_emp_p_aligned",
            "ci_raw_excludes_zero": "best_ci_raw_excludes_zero",
            "ci_aligned_excludes_zero": "best_ci_aligned_excludes_zero",
        })
    )


b4_per_cell = per_cell_best(b4_per_window, b4_cell_keys)
if b4_per_cell.height:
    b4_per_cell.write_parquet(OUT_DIR / "b4_per_cell.parquet")
print(f"B4 per_cell rows: {b4_per_cell.height}")

# Augment with fields needed to regenerate star plots in the filtered gallery.
if b4_per_cell.height:
    b4_per_cell = b4_per_cell.join(
        b4_qualified.select([
            "subject", "electrode_idx", "phoneme_pair", "word_end",
            "qualifying_steps", "phon_smin", "phon_smax",
        ]),
        on=b4_cell_keys, how="left",
    )

# %% [markdown]
# ## Cross-WE pooled pair statistic
#
# Design choices (2026-05-27-causal46-cross-we-pooled-test.md):
#   D1: Extend population_summary.pdf in place (Option A)
#   D2: S_r = (|e0_r| + |e1_r|) / 2; report sign_concordance (Option A)
#   D3: Best pair-window = argmax median S_r over windows shared by both WEs (Option A)
#   D4: Only pairs where BOTH WEs are present with shared windows (revised to Option B/exclude 1-WE)
#   D5: emp_p via replicate-permutation null — NOTE: this tests WE coupling, not magnitude vs. chance;
#       a label-permutation null integrated into bootstrap_cell would be more principled (TBD)
#   D6: Scatter + ROI breakdown + lift waterfall

# %%
PAIR_KEYS = ["subject", "electrode_idx", "phoneme_pair"]
N_NULL_PERM = 999
PAIR_EMP_P_THRESHOLD = 0.05


def cross_we_pair_summary(
    boot: pl.DataFrame,
    per_cell: pl.DataFrame,
    n_null_perm: int = N_NULL_PERM,
    rng_seed: int = 42,
) -> pl.DataFrame:
    """Pair-level magnitude-pooled bootstrap statistic per (subject x electrode x phoneme_pair).

    Only processes pairs where BOTH word-ends are present in boot with at least
    one shared (smin, smax) window.  1-WE pairs are skipped entirely.

    S_r = (|e0_r| + |e1_r|) / 2 at the best shared window (argmax median S_r).
    emp_p: replicate-permutation null — 999 shuffles of WE1's replicate order.
    pair_ci_excludes_zero = pair_emp_p < PAIR_EMP_P_THRESHOLD.
    """
    if boot.height == 0 or per_cell.height == 0:
        return pl.DataFrame()
    rng = np.random.default_rng(rng_seed)
    pc_index = {
        (r["subject"], r["electrode_idx"], r["phoneme_pair"], r["word_end"]): r
        for r in per_cell.iter_rows(named=True)
    }

    def _pc(subj: str, eidx: int, pp: str, we: str) -> dict | None:
        return pc_index.get((subj, eidx, pp, we))

    rows: list[dict] = []
    for group_df in tqdm(
        boot.partition_by(PAIR_KEYS, maintain_order=True),
        desc="cross-WE pairs",
    ):
        subj = str(group_df["subject"][0])
        eidx = int(group_df["electrode_idx"][0])
        pp = str(group_df["phoneme_pair"][0])

        word_ends = sorted(group_df["word_end"].unique().to_list())
        if len(word_ends) != 2:
            continue  # only process pairs where both WEs are present

        we0, we1 = word_ends[0], word_ends[1]

        def _win_map(we: str) -> dict[tuple, np.ndarray]:
            wdf = group_df.filter(pl.col("word_end") == we)
            wmap: dict[tuple, np.ndarray] = {}
            for wg in wdf.partition_by(["smin", "smax"], maintain_order=True):
                wmap[(int(wg["smin"][0]), int(wg["smax"][0]))] = (
                    wg.sort("replicate")["mean_diff_aligned"].to_numpy().astype(float)
                )
            return wmap

        map0 = _win_map(we0)
        map1 = _win_map(we1)
        shared = set(map0.keys()) & set(map1.keys())
        if not shared:
            continue  # no shared windows across WEs

        # Find best window by median pair-statistic
        best_S_med = -1.0
        best_sm = best_sx = 0
        best_e0: np.ndarray | None = None
        best_e1: np.ndarray | None = None
        for win_key in shared:
            e0 = map0[win_key]
            e1 = map1[win_key]
            min_len = min(len(e0), len(e1))
            if min_len < 10:
                continue
            e0, e1 = e0[:min_len], e1[:min_len]
            valid = ~(np.isnan(e0) | np.isnan(e1))
            if valid.sum() < 10:
                continue
            med = float(np.nanmedian((np.abs(e0) + np.abs(e1)) / 2))
            if med > best_S_med:
                best_S_med = med
                best_sm, best_sx = win_key
                best_e0, best_e1 = e0[:min_len], e1[:min_len]
        if best_e0 is None:
            continue

        S_obs = (np.abs(best_e0) + np.abs(best_e1)) / 2
        s_med = float(np.nanmedian(S_obs))
        s_ci_lo = float(np.nanpercentile(S_obs, CI_LOW))
        s_ci_hi = float(np.nanpercentile(S_obs, CI_HIGH))

        valid = ~(np.isnan(best_e0) | np.isnan(best_e1))
        sign_conc = (
            float(np.mean(np.sign(best_e0[valid]) == np.sign(best_e1[valid])))
            if valid.sum() > 0 else float("nan")
        )

        # Replicate-permutation null: shuffle WE1 replicate order independently
        R_act = len(best_e0)
        null_meds = np.empty(n_null_perm)
        for i in range(n_null_perm):
            idx = rng.permutation(R_act)
            null_meds[i] = float(np.nanmedian(
                (np.abs(best_e0) + np.abs(best_e1[idx])) / 2
            ))
        pair_emp_p = float(np.mean(null_meds >= s_med))
        pair_ci_excl = pair_emp_p < PAIR_EMP_P_THRESHOLD

        cells_sig = sum(
            1 for we in word_ends
            if (pcr := _pc(subj, eidx, pp, we)) and pcr.get("best_ci_aligned_excludes_zero")
        )
        auc_list = []
        for we in word_ends:
            pcr = _pc(subj, eidx, pp, we)
            if pcr is not None:
                v = pcr.get("acoustic_peak_auc")
                if v is not None:
                    auc_list.append(float(v))
        auc_max = max(auc_list) if auc_list else float("nan")

        rows.append({
            "subject": subj,
            "electrode_idx": eidx,
            "phoneme_pair": pp,
            "word_ends": ",".join(word_ends),
            "n_we_contributing": 2,
            "pair_smin": best_sm,
            "pair_smax": best_sx,
            "pair_tmin": best_sm / epoch_sfreq + epoch_tmin,
            "pair_tmax": best_sx / epoch_sfreq + epoch_tmin,
            "pair_statistic_med": s_med,
            "pair_statistic_ci_lo": s_ci_lo,
            "pair_statistic_ci_hi": s_ci_hi,
            "pair_emp_p": pair_emp_p,
            "pair_ci_excludes_zero": pair_ci_excl,
            "sign_concordance": sign_conc,
            "acoustic_peak_auc_max": auc_max,
            "cells_individually_sig": cells_sig,
        })

    return pl.DataFrame(rows) if rows else pl.DataFrame()


b4_per_pair = cross_we_pair_summary(b4_boot, b4_per_cell)
if b4_per_pair.height:
    b4_per_pair.write_parquet(OUT_DIR / "b4_per_pair.parquet")
n_pair_sig = b4_per_pair.filter(pl.col("pair_ci_excludes_zero")).height if b4_per_pair.height else 0
print(f"B4 per_pair rows: {b4_per_pair.height}  (pair_ci_excl: {n_pair_sig})")

# Join pair verdict back to b4_per_cell and re-write
if b4_per_pair.height and b4_per_cell.height:
    b4_per_cell = b4_per_cell.join(
        b4_per_pair.select([
            "subject", "electrode_idx", "phoneme_pair",
            "pair_statistic_med", "pair_ci_excludes_zero",
            pl.col("n_we_contributing").alias("pair_n_we_contributing"),
        ]),
        on=PAIR_KEYS, how="left",
    )
    b4_per_cell.write_parquet(OUT_DIR / "b4_per_cell.parquet")
    print(f"Updated b4_per_cell.parquet with pair columns ({b4_per_cell.height} rows)")

# %% [markdown]
# ## ROI lookup

# %%
roi_frames = []
subjects = sorted({*([] if b4_boot.height == 0 else b4_boot["subject"].unique().to_list())})
for subj in subjects:
    try:
        edf = get_electrode_df(subj)
    except Exception as exc:
        print(f"⚠ no electrode_df for {subj}: {exc}")
        continue
    roi_col = "roi" if "roi" in edf.columns else (
        "anat" if "anat" in edf.columns else None
    )
    if roi_col is None:
        print(f"⚠ no ROI/anat column for {subj}: {list(edf.columns)}")
        continue
    edf2 = edf.reset_index().rename(columns={"index": "electrode_idx"}) \
        if "electrode_idx" not in edf.columns else edf
    roi_frames.append(pl.from_pandas(
        edf2[["electrode_idx", roi_col]].assign(subject=subj).rename(
            columns={roi_col: "roi"}
        ).astype({"roi": str})
    ))
electrode_roi = (
    pl.concat(roi_frames, how="diagonal_relaxed")
    if roi_frames else
    pl.DataFrame(schema={"subject": pl.Utf8, "electrode_idx": pl.Int64,
                          "roi": pl.Utf8})
)
print(f"ROI rows: {electrode_roi.height}")

# %% [markdown]
# ## Population summary
#
# Per (mode × window × cut): fraction of cells with `ci_aligned_excludes_zero`
# (and the raw counterpart for sanity), median signed mean_diff_aligned,
# median |mean_diff_aligned|, IQR. Cells with tied tuning (preferred_class
# is None → aligned is NaN) are counted in the raw column but excluded from
# aligned medians and fraction.

# %%
def population_summary(per_window: pl.DataFrame, mode_name: str) -> pl.DataFrame:
    if per_window.height == 0:
        return pl.DataFrame()
    pc = per_window.with_columns(pl.lit(mode_name).alias("mode")) \
        .join(electrode_roi, on=["subject", "electrode_idx"], how="left") \
        .with_columns(pl.col("roi").fill_null("unknown"))
    out: list[pl.DataFrame] = []
    for cut_name, group_cols in [
        ("overall", []),
        ("phoneme_pair", ["phoneme_pair"]),
        ("roi", ["roi"]),
        ("phoneme_pair_x_roi", ["phoneme_pair", "roi"]),
    ]:
        agg = (
            pc
            .group_by(["mode", "smin", "smax", "tmin", "tmax"] + group_cols)
            .agg(
                pl.len().alias("n_cells"),
                pl.col("ci_raw_excludes_zero").cast(pl.Float64).sum().alias("n_ci_raw_sig"),
                pl.col("ci_aligned_excludes_zero").cast(pl.Float64).sum().alias("n_ci_aligned_sig"),
                pl.col("mean_diff_aligned_med").is_not_null().cast(pl.Float64)
                    .sum().alias("n_cells_aligned_defined"),
                pl.col("mean_diff_aligned_med").median().alias("median_signed_aligned"),
                pl.col("mean_diff_aligned_med").abs().median().alias("median_abs_aligned"),
                pl.col("mean_diff_aligned_med").abs().quantile(0.25).alias("q25_abs_aligned"),
                pl.col("mean_diff_aligned_med").abs().quantile(0.75).alias("q75_abs_aligned"),
            )
            .with_columns([
                (pl.col("n_ci_raw_sig") / pl.col("n_cells")).alias("frac_ci_raw"),
                (pl.col("n_ci_aligned_sig") /
                 pl.when(pl.col("n_cells_aligned_defined") > 0)
                   .then(pl.col("n_cells_aligned_defined"))
                   .otherwise(1)).alias("frac_ci_aligned"),
                pl.lit(cut_name).alias("cut"),
            ])
        )
        out.append(agg)
    return pl.concat(out, how="diagonal_relaxed")


b4_pop = population_summary(b4_per_window, "b4")
population = pl.concat([df for df in (b4_pop,) if df.height],
                       how="diagonal_relaxed")
if population.height:
    population.write_csv(OUT_DIR / "population_summary.csv")
print(f"population_summary rows: {population.height}")

# %% [markdown]
# ## Population summary plots

# %%
JON41_GREENLIGHT = 0.40
JON41_REVISIT = 0.15


def _peak_frac(pop: pl.DataFrame, mode: str, frac_col: str) -> float | None:
    if pop.height == 0:
        return None
    sub = pop.filter((pl.col("mode") == mode) & (pl.col("cut") == "overall"))
    if sub.height == 0:
        return None
    return float(sub[frac_col].max())


pdf_path = OUT_DIR / "population_summary.pdf"
with PdfPages(pdf_path) as pdf:
    # Title
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.7, "Bootstrap CI summary — within-completion behavior contrast",
            ha="center", va="center", fontsize=18)
    n_b4_ok = cell_manifest.filter((pl.col('mode')=='matched_n') & (pl.col('status')=='ok')).height
    ax.text(0.5, 0.55,
            f"K = {K}   ·   R = {R}   ·   CI = {CI_LOW}–{CI_HIGH}%\n"
            f"B4 cells (ok): {n_b4_ok}\n"
            f"AS sites: {peaks.height}",
            ha="center", va="center", fontsize=11)
    pdf.savefig(fig); plt.close(fig)

    # Per-time frac-CI-excludes-0 curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode in zip(axes, ("b4",)):
        for col, color, label in (
            ("frac_ci_aligned", "#2166ac", "aligned (CI excludes 0)"),
            ("frac_ci_raw",     "#b2182b", "raw (CI excludes 0)"),
        ):
            sub = (
                population
                .filter((pl.col("mode") == mode) & (pl.col("cut") == "overall"))
                .sort("tmin")
            )
            if sub.height == 0:
                continue
            ax.plot(sub["tmin"].to_numpy(), sub[col].to_numpy(),
                    marker="o", color=color, lw=1.4, label=label)
        ax.axhline(JON41_GREENLIGHT, color="#4dac26", lw=1, ls="--",
                   label=f"greenlight ≥ {JON41_GREENLIGHT:.0%}")
        ax.axhline(JON41_REVISIT, color="#d73027", lw=1, ls="--",
                   label=f"revisit < {JON41_REVISIT:.0%}")
        ax.set_xlabel("Window start (s, post word onset)")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{mode.upper()} — fraction with CI excluding 0")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("fraction of cells")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # Sites x time heatmap of median signed aligned mean_diff
    for mode, per_window, ckeys in (
        ("b4", b4_per_window, b4_cell_keys),
    ):
        if per_window.height == 0:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.5, f"{mode.upper()}: no per-window rows",
                    ha="center", va="center", fontsize=14); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)
            continue
        pivot = (
            per_window
            .pivot(values="mean_diff_aligned_med", index=ckeys, on="tmin",
                   aggregate_function="first")
            .sort(ckeys)
        )
        time_cols_sorted = sorted([c for c in pivot.columns if c not in ckeys],
                                  key=lambda c: float(c))
        mat = pivot.select(time_cols_sorted).to_numpy().astype(float)
        if mat.size == 0:
            continue
        sort_order = np.argsort(-np.nanmax(np.abs(mat), axis=1))
        mat = mat[sort_order]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        finite = np.isfinite(mat) & ~np.isnan(mat)
        vlim = float(np.nanpercentile(np.abs(mat[finite]) if finite.any() else [1], 98)) or 1.0
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vlim, vmax=vlim,
                       extent=[float(time_cols_sorted[0]),
                               float(time_cols_sorted[-1]), mat.shape[0], 0])
        ax.set_xlabel("Window start (s, post word onset)")
        ax.set_ylabel("Cell (sorted by peak |aligned mean_diff|)")
        ax.set_title(f"{mode.upper()} — median signed aligned mean_diff (R={R})  "
                     f"n_cells={mat.shape[0]}")
        cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label("aligned mean_diff (HGA)")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    # Effect-size distributions: best-window aligned medians
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (mode, per_cell) in zip(axes, [
        ("b4", b4_per_cell),
    ]):
        if per_cell.height == 0:
            ax.text(0.5, 0.5, f"{mode}: empty", ha="center", va="center"); ax.axis("off"); continue
        signed = per_cell["best_mean_diff_aligned_med"].drop_nulls().to_numpy()
        ax.hist(signed, bins=30, color="#4393c3", edgecolor="k", alpha=0.7)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("best-window median signed aligned mean_diff")
        ax.set_ylabel("# cells")
        med = float(np.nanmedian(signed)) if len(signed) else float("nan")
        ax.set_title(f"{mode.upper()} — n={len(signed)}, median={med:.2f}")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # Best-window timing scatter — colored by CI significance
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (mode, per_cell) in zip(axes, [("b4", b4_per_cell)]):
        if per_cell.height == 0:
            ax.text(0.5, 0.5, f"{mode}: empty", ha="center", va="center"); ax.axis("off"); continue
        tmin_arr = per_cell["best_tmin"].to_numpy().astype(float)
        sig_mask = per_cell["best_ci_aligned_excludes_zero"].to_numpy().astype(bool)
        rng_jit = np.random.default_rng(42)
        y_jit = rng_jit.uniform(-0.4, 0.4, size=len(tmin_arr))
        ax.scatter(tmin_arr[~sig_mask], y_jit[~sig_mask],
                   color="#d9d9d9", s=22, alpha=0.8, linewidths=0.4,
                   edgecolors="k", label="CI includes 0", zorder=2)
        ax.scatter(tmin_arr[sig_mask], y_jit[sig_mask],
                   color="#2166ac", s=30, alpha=0.85, linewidths=0.4,
                   edgecolors="k", label="CI excludes 0 (aligned)", zorder=3)
        n_sig = int(sig_mask.sum())
        n_tot = len(sig_mask)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xlabel("Best-window tmin (s, post word onset)")
        ax.set_yticks([])
        ax.set_title(f"{mode.upper()} — {n_sig}/{n_tot} significant ({100*n_sig/max(n_tot,1):.0f}%)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Best-window timing per cell", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # Phoneme_pair / ROI breakdowns
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, mode in enumerate(("b4",)):
        for row, cut in enumerate(("phoneme_pair", "roi")):
            ax = axes[row, col]
            sub = population.filter((pl.col("mode") == mode) & (pl.col("cut") == cut))
            if sub.height == 0:
                ax.text(0.5, 0.5, f"{mode}/{cut}: empty",
                        ha="center", va="center"); ax.axis("off"); continue
            peak = (
                sub.group_by(cut)
                   .agg(pl.col("frac_ci_aligned").max().alias("frac_peak"),
                        pl.col("n_cells_aligned_defined").max().alias("n_peak"))
                   .sort("frac_peak", descending=True)
            )
            labels = peak[cut].to_list()
            vals = peak["frac_peak"].to_list()
            ns = peak["n_peak"].to_list()
            ax.barh(labels, vals, color="#4393c3", edgecolor="k")
            ax.axvline(JON41_GREENLIGHT, color="#4dac26", lw=1, ls="--")
            ax.axvline(JON41_REVISIT, color="#d73027", lw=1, ls="--")
            for y, (v, n) in enumerate(zip(vals, ns)):
                ax.text(v + 0.01, y, f"{v:.0%} (n={int(n)})", va="center", fontsize=8)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("peak-window frac with CI excluding 0 (aligned)")
            ax.set_title(f"{mode.upper()} — {cut}")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # Decision callout
    b4_peak = _peak_frac(population, "b4", "frac_ci_aligned")
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    headline = b4_peak if b4_peak is not None else None
    if headline is None:
        verdict = "    INDETERMINATE — no cells qualified"
    elif headline >= JON41_GREENLIGHT:
        verdict = "    GREENLIGHT — option 1 supported by per-site bootstrap CIs"
    elif headline < JON41_REVISIT:
        verdict = "    REVISIT — eyeball/star-plot evidence outruns statistics"
    else:
        verdict = "    AMBIGUOUS — between thresholds; needs discussion"
    lines = [
        "DECISION CALLOUT — JON-41 Group B thresholds",
        "",
        f"  Greenlight (continue option 1):   peak frac CI-aligned ≥ {JON41_GREENLIGHT:.0%}",
        f"  Revisit (rethink AS-site filter): peak frac CI-aligned <  {JON41_REVISIT:.0%}",
        "",
        f"  B4 peak frac CI-aligned: "
        f"{b4_peak:.1%}" if b4_peak is not None else "  B4: n/a (no cells)",
        "",
        "  Verdict (B4 if available, else B3):",
        verdict,
    ]
    ax.text(0.05, 0.95, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=11)
    pdf.savefig(fig); plt.close(fig)

    # ---- Cross-WE pooled pair statistic pages ----
    if b4_per_pair.height:
        # Section header
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.7, "Cross-WE pooled pair statistic",
                ha="center", va="center", fontsize=18)
        n_sig_pair = b4_per_pair.filter(pl.col("pair_ci_excludes_zero")).height
        n_lift = b4_per_pair.filter(
            pl.col("pair_ci_excludes_zero") & (pl.col("cells_individually_sig") < 2)
        ).height if "cells_individually_sig" in b4_per_pair.columns else 0
        ax.text(0.5, 0.5,
                f"Total pairs (both WEs present): {b4_per_pair.height}\n"
                f"pair_ci_excludes_zero: {n_sig_pair}  ·  lift (sig but not both cells): {n_lift}\n"
                f"Method: S_r = (|e0_r| + |e1_r|) / 2  ·  emp_p from {N_NULL_PERM}-rep permutation null",
                ha="center", va="center", fontsize=11)
        pdf.savefig(fig); plt.close(fig)

        # D6a: Per-pair scatter (all pairs have both WEs; n_we_contributing==2 always)
        pairs_2we = b4_per_pair
        if pairs_2we.height and b4_per_cell.height:
            pairs_2we = pairs_2we.with_columns([
                pl.col("word_ends").str.split(",").list.get(0).alias("_we0"),
                pl.col("word_ends").str.split(",").list.get(1).alias("_we1"),
            ])
            eff_lookup = b4_per_cell.select(
                ["subject", "electrode_idx", "phoneme_pair", "word_end",
                 "best_mean_diff_aligned_med"]
            )
            scatter_df = (
                pairs_2we.join(
                    eff_lookup.rename({
                        "word_end": "_we0",
                        "best_mean_diff_aligned_med": "eff_we0",
                    }),
                    on=["subject", "electrode_idx", "phoneme_pair", "_we0"],
                    how="left",
                ).join(
                    eff_lookup.rename({
                        "word_end": "_we1",
                        "best_mean_diff_aligned_med": "eff_we1",
                    }),
                    on=["subject", "electrode_idx", "phoneme_pair", "_we1"],
                    how="left",
                )
            )
            x_arr = scatter_df["eff_we0"].to_numpy().astype(float)
            y_arr = scatter_df["eff_we1"].to_numpy().astype(float)
            sig_arr = scatter_df["pair_ci_excludes_zero"].to_numpy().astype(bool)
            valid_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_v, y_v, sig_v = np.abs(x_arr[valid_mask]), np.abs(y_arr[valid_mask]), sig_arr[valid_mask]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(x_v[~sig_v], y_v[~sig_v], color="#d9d9d9", s=40, alpha=0.8,
                       edgecolors="k", lw=0.5, label=f"pair CI includes 0 (n={int((~sig_v).sum())})")
            ax.scatter(x_v[sig_v], y_v[sig_v], color="#2166ac", s=50, alpha=0.9,
                       edgecolors="k", lw=0.5, label=f"pair CI excludes 0 (n={int(sig_v.sum())})")
            if valid_mask.sum() >= 3:
                from scipy.stats import spearmanr
                r_sp, p_sp = spearmanr(x_v, y_v)
                ax.set_title(
                    f"Cross-WE |effect| scatter — 2-WE pairs (n={valid_mask.sum()})\n"
                    f"Spearman r = {r_sp:.2f}, p = {p_sp:.3f}", fontsize=10
                )
            else:
                ax.set_title(f"Cross-WE |effect| scatter — n={valid_mask.sum()}", fontsize=10)
            lim = max(x_v.max() if len(x_v) else 1, y_v.max() if len(y_v) else 1) * 1.05
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
            ax.set_xlabel("|best aligned effect| — WE0")
            ax.set_ylabel("|best aligned effect| — WE1")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # D6b: ROI breakdown — frac pair_ci_excludes_zero per ROI
        pair_roi = (
            b4_per_pair
            .join(electrode_roi, on=["subject", "electrode_idx"], how="left")
            .with_columns(pl.col("roi").fill_null("unknown"))
        )
        roi_pair_summary = (
            pair_roi
            .group_by("roi")
            .agg(
                pl.len().alias("n_pairs"),
                pl.col("pair_ci_excludes_zero").cast(pl.Float64).sum().alias("n_pair_sig"),
                pl.col("n_we_contributing").eq(2).cast(pl.Float64).mean().alias("frac_2we"),
            )
            .with_columns(
                (pl.col("n_pair_sig") / pl.col("n_pairs")).alias("frac_pair_sig")
            )
            .sort("frac_pair_sig", descending=True)
        )
        fig, ax = plt.subplots(figsize=(8, max(3, roi_pair_summary.height * 0.45)))
        labels = roi_pair_summary["roi"].to_list()
        vals = roi_pair_summary["frac_pair_sig"].to_list()
        ns = roi_pair_summary["n_pairs"].to_list()
        ax.barh(labels, vals, color="#4393c3", edgecolor="k")
        ax.axvline(JON41_GREENLIGHT, color="#4dac26", lw=1, ls="--")
        ax.axvline(JON41_REVISIT, color="#d73027", lw=1, ls="--")
        for y_pos, (v, n) in enumerate(zip(vals, ns)):
            ax.text(v + 0.01, y_pos, f"{v:.0%} (n={int(n)})", va="center", fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("fraction of pairs with pair_ci_excludes_zero")
        ax.set_title("Cross-WE pair significance by ROI", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # D6c: Lift waterfall — all pairs sorted by pair_emp_p
        pairs_2we_lift = b4_per_pair.sort("pair_emp_p")
        if pairs_2we_lift.height:
            emp_p_arr = pairs_2we_lift["pair_emp_p"].to_numpy().astype(float)
            csig_arr = pairs_2we_lift["cells_individually_sig"].to_numpy().astype(int)
            log_p = -np.log10(np.clip(emp_p_arr, 1e-4, 1))
            colors = {0: "#d01c8b", 1: "#f1b6da", 2: "#4dac26"}
            bar_colors = [colors.get(int(c), "#888888") for c in csig_arr]
            fig, ax = plt.subplots(figsize=(min(12, 0.4 * len(log_p) + 2), 4))
            ax.bar(range(len(log_p)), log_p, color=bar_colors, edgecolor="none", width=0.8)
            ax.axhline(-np.log10(PAIR_EMP_P_THRESHOLD), color="k", lw=1, ls="--",
                       label=f"p = {PAIR_EMP_P_THRESHOLD}")
            from matplotlib.patches import Patch
            legend_els = [
                Patch(facecolor=colors[0], label="neither cell individually sig"),
                Patch(facecolor=colors[1], label="1 cell sig"),
                Patch(facecolor=colors[2], label="both cells sig"),
            ]
            ax.legend(handles=legend_els, fontsize=8, loc="upper right")
            ax.set_xlabel("pair (sorted by pair_emp_p)")
            ax.set_ylabel("-log10(pair_emp_p)")
            ax.set_title(
                f"Lift waterfall — cross-WE pairs (n={pairs_2we_lift.height})\n"
                f"sig (p<{PAIR_EMP_P_THRESHOLD}): {int((emp_p_arr < PAIR_EMP_P_THRESHOLD).sum())}  "
                f"of which lift (cells_sig<2): "
                f"{int(((emp_p_arr < PAIR_EMP_P_THRESHOLD) & (csig_arr < 2)).sum())}",
                fontsize=10,
            )
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

print(f"wrote {pdf_path}")

# %% [markdown]
# ## Filtered-gallery hook

# %%
try:
    from pypdf import PdfReader, PdfWriter
    _HAS_PYPDF = True
except ImportError:
    PdfReader = PdfWriter = None  # type: ignore[assignment]
    _HAS_PYPDF = False
    print("⚠ pypdf not installed — will emit filtered_manifest.csv only; "
          "filtered PDFs skipped.")


def site_effect_fig(row: dict, site_per_window: pl.DataFrame) -> plt.Figure:
    """CI-trace figure for one cell: bootstrap mean ± CI band across windows."""
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    if site_per_window.height > 0:
        pw = site_per_window.sort("tmin")
        tcenter = ((pw["tmin"] + pw["tmax"]) / 2).to_numpy().astype(float)
        mn = pw["mean_diff_aligned_mean"].to_numpy().astype(float)
        ci_lo = pw["mean_diff_aligned_ci_lo"].to_numpy().astype(float)
        ci_hi = pw["mean_diff_aligned_ci_hi"].to_numpy().astype(float)
        ax.plot(tcenter, mn, color="#2166ac", lw=1.5,
                label="bootstrap mean aligned diff")
        ax.fill_between(tcenter, ci_lo, ci_hi, color="#2166ac", alpha=0.22,
                        label=f"{CI_LOW}–{CI_HIGH}% bootstrap CI")
        # Highlight best window — spans tmin to tmax so the line center sits inside it
        if row.get("best_tmin") is not None and row.get("best_tmax") is not None:
            ax.axvspan(float(row["best_tmin"]), float(row["best_tmax"]),
                       color="#fdae61", alpha=0.45, label="best window", zorder=0)
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("Window center (s, post word onset)")
    ax.set_ylabel("aligned mean_diff (HGA)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    med_val = row.get("best_mean_diff_aligned_med")
    ci_lo_v = row.get("best_mean_diff_aligned_ci_lo")
    ci_hi_v = row.get("best_mean_diff_aligned_ci_hi")
    p_val = row.get("best_emp_p_aligned")
    sig_str = "CI excludes 0" if row.get("best_ci_aligned_excludes_zero") else "CI includes 0"
    id_str = (f"{row['subject']} e{row['electrode_idx']} "
              f"{row['phoneme_pair']} · {row['word_end']}")
    if row.get("resampled") is not None:
        id_str += f" step {row['resampled']}"
    stat_str = ""
    if med_val is not None:
        stat_str = (f"  |  effect = {med_val:.3f} [{ci_lo_v:.3f}, {ci_hi_v:.3f}]"
                    f"  p = {p_val:.3f}  {sig_str}")
    ax.set_title(id_str + stat_str, fontsize=9)
    fig.tight_layout()
    return fig


def write_annotated_pdfs(
    entries: list[dict],
    per_window: pl.DataFrame,
    cell_keys: list[str],
    out_path: Path,
    epochs_dict: dict | None = None,
    pair_lookup: dict | None = None,
) -> int:
    """Filtered-gallery PDF: regenerated star plot per cell.

    pair_lookup: optional dict keyed by (subject, electrode_idx, phoneme_pair) →
    b4_per_pair row dict. When provided, a colored banner is added at the top of
    each page showing the cross-WE pooled test result for that site/pair.
    """
    if not entries or not _HAS_PYPDF:
        return 0
    # Precompute matched x-axis limit per (subject, electrode_idx, phoneme_pair):
    # both word_ends in a group share the max offset so plots align vertically.
    group_xlim: dict[tuple, float] = {}
    for row in entries:
        key = (row["subject"], row["electrode_idx"], row["phoneme_pair"])
        we_xlim = OFFSET_DICT.get(row["word_end"], 1.0) + 0.1
        group_xlim[key] = max(group_xlim.get(key, 0.0), we_xlim)
    writer = PdfWriter()
    n = 0
    for row in tqdm(entries):
        filt = (
            (pl.col("subject") == row["subject"])
            & (pl.col("electrode_idx") == row["electrode_idx"])
            & (pl.col("phoneme_pair") == row["phoneme_pair"])
            & (pl.col("word_end") == row["word_end"])
        )
        if "resampled" in cell_keys and row.get("resampled") is not None:
            filt = filt & (pl.col("resampled") == row["resampled"])
        site_pw = per_window.filter(filt) if per_window.height else pl.DataFrame()

        sig_wins = None
        mda = None
        if site_pw.height:
            pw_s = site_pw.sort("tmin")
            sig_list = [
                (float(r["tmin"]), float(r["tmax"]))
                for r in pw_s.filter(pl.col("ci_aligned_excludes_zero")).iter_rows(named=True)
            ]
            sig_wins = sig_list or None
            mda = {
                "tcenter": ((pw_s["tmin"] + pw_s["tmax"]) / 2).to_numpy().astype(float),
                "mean": pw_s["mean_diff_aligned_mean"].to_numpy().astype(float),
                "ci_lo": pw_s["mean_diff_aligned_ci_lo"].to_numpy().astype(float),
                "ci_hi": pw_s["mean_diff_aligned_ci_hi"].to_numpy().astype(float),
            }

        qs = row.get("qualifying_steps")
        can_regen = (
            epochs_dict is not None
            and row.get("subject") in epochs_dict
            and qs is not None
            and row.get("phon_smin") is not None
        )
        if not can_regen:
            print(f"  ⚠ skipping {row['subject']} e{row['electrode_idx']}: "
                  "cannot regenerate star plot (missing epochs or qualifying_steps)")
            continue
        if isinstance(qs, str):
            qs = [int(s) for s in qs.split(",") if s]
        key = (row["subject"], row["electrode_idx"], row["phoneme_pair"])
        try:
            fig2 = matched_n_star_plot(
                subject=row["subject"],
                electrode_idx=int(row["electrode_idx"]),
                phoneme_pair=row["phoneme_pair"],
                word_end=row["word_end"],
                qualifying_steps=list(qs),
                epochs_dict=epochs_dict,
                n_per_class=int(row["n_per_class"]),
                phon_smin=int(row["phon_smin"]),
                phon_smax=int(row["phon_smax"]),
                phon_search_smin=AC_SEARCH_SMIN,
                phon_search_smax=AC_SEARCH_SMAX,
                acoustic_peak_auc=row.get("acoustic_peak_auc"),
                sig_windows=sig_wins,
                mean_diff_arrays=mda,
                xlim=group_xlim[key],
            )
            # Cross-WE pooled test banner
            if pair_lookup is not None:
                pair_key_lut = (
                    row["subject"], int(row["electrode_idx"]), row["phoneme_pair"]
                )
                pr = pair_lookup.get(pair_key_lut)
                if pr is not None:
                    pair_sig = bool(pr.get("pair_ci_excludes_zero", False))
                    emp_p = pr.get("pair_emp_p")
                    emp_p_str = f"{float(emp_p):.3f}" if emp_p is not None else "?"
                    sc = pr.get("sign_concordance")
                    sc_str = f"{float(sc):.2f}" if sc is not None and not (
                        isinstance(sc, float) and np.isnan(sc)
                    ) else "?"
                    bar_color = "#4dac26" if pair_sig else "#d9d9d9"
                    text_color = "white" if pair_sig else "#555555"
                    label = (
                        f"cross-WE pooled: {'SIG' if pair_sig else 'ns'}"
                        f"   p = {emp_p_str}"
                        f"   cells_sig = {pr.get('cells_individually_sig', '?')}/2"
                        f"   sign_concordance = {sc_str}"
                    )
                    fig2.suptitle(
                        label, y=1.01, fontsize=7.5, color=text_color,
                        bbox=dict(facecolor=bar_color, alpha=0.9,
                                  edgecolor="none", pad=4,
                                  boxstyle="round,pad=0.3"),
                    )
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="pdf", bbox_inches="tight")
            plt.close(fig2)
            buf2.seek(0)
            for page in PdfReader(buf2).pages:
                writer.add_page(page)
            n += 1
        except Exception as exc:
            print(f"  ⚠ star plot regen failed for {row['subject']} "
                  f"e{row['electrode_idx']}: {exc}")
    if n:
        with out_path.open("wb") as fh:
            writer.write(fh)
    return n


filtered_rows: list[dict] = []

if b4_per_cell.height:
    powered_entries = []
    sig_entries = []
    for row in b4_per_cell.iter_rows(named=True):
        is_sig = bool(row["best_ci_aligned_excludes_zero"])
        powered_entries.append(row)
        if is_sig: sig_entries.append(row)
        manifest_row = cell_manifest.filter(
            (pl.col("mode") == "matched_n")
            & (pl.col("subject") == row["subject"])
            & (pl.col("electrode_idx") == row["electrode_idx"])
            & (pl.col("phoneme_pair") == row["phoneme_pair"])
            & (pl.col("word_end") == row["word_end"])
        )
        qs = manifest_row["qualifying_steps"][0] if manifest_row.height else ""
        filtered_rows.append({
            "mode": "matched_n",
            "subject": row["subject"], "electrode_idx": row["electrode_idx"],
            "phoneme_pair": row["phoneme_pair"], "word_end": row["word_end"],
            "resampled": None, "qualifying_steps": qs,
            "best_smin": row["best_smin"], "best_smax": row["best_smax"],
            "best_mean_diff_aligned_med": row["best_mean_diff_aligned_med"],
            "best_emp_p_aligned": row["best_emp_p_aligned"],
            "best_ci_aligned_excludes_zero": row["best_ci_aligned_excludes_zero"],
            "powered": True, "significant": is_sig,
            "pdf_path": "",
            "status": "ok",
        })
    pair_lut = {
        (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"]): r
        for r in b4_per_pair.iter_rows(named=True)
    } if b4_per_pair.height else None
    n_p = write_annotated_pdfs(powered_entries, b4_per_window, b4_cell_keys,
                               FILT_DIR / "b4_powered.pdf",
                               epochs_dict=epochs_dict, pair_lookup=pair_lut)
    n_s = write_annotated_pdfs(sig_entries, b4_per_window, b4_cell_keys,
                               FILT_DIR / "b4_powered_significant.pdf",
                               epochs_dict=epochs_dict, pair_lookup=pair_lut)
    print(f"B4 filtered PDFs: powered={n_p}  significant={n_s}")

under = cell_manifest.filter(
    pl.col("status").is_in(["underpowered", "search_range_too_narrow"])
)
for row in under.iter_rows(named=True):
    filtered_rows.append({
        "mode": row["mode"],
        "subject": row["subject"], "electrode_idx": row["electrode_idx"],
        "phoneme_pair": row["phoneme_pair"], "word_end": row["word_end"],
        "resampled": row["resampled_step"] if row["mode"] == "single_step" else None,
        "qualifying_steps": row["qualifying_steps"],
        "best_smin": None, "best_smax": None,
        "best_mean_diff_aligned_med": None,
        "best_emp_p_aligned": None,
        "best_ci_aligned_excludes_zero": False,
        "powered": False, "significant": False,
        "pdf_path": "", "status": row["status"],
    })

if filtered_rows:
    pl.DataFrame(filtered_rows).write_csv(FILT_DIR / "filtered_manifest.csv")
    print(f"wrote {FILT_DIR / 'filtered_manifest.csv'}  ({len(filtered_rows)} rows)")
else:
    (FILT_DIR / "filtered_manifest.csv").write_text("")
    print("filtered_manifest.csv: no rows (empty)")

# %% [markdown]
# ## Done

# %%
print("=" * 70)
print(f"K = {K}   R = {R}   CI = {CI_LOW}–{CI_HIGH}%")
print(f"AS sites: {peaks.height}")
print(f"B4 cells (ok): "
      f"{cell_manifest.filter((pl.col('mode')=='matched_n') & (pl.col('status')=='ok')).height}")
b4_peak = _peak_frac(population, "b4", "frac_ci_aligned")
print(f"B4 peak frac CI-aligned: {b4_peak if b4_peak is None else f'{b4_peak:.1%}'}")
print(f"See {pdf_path} for the decision callout.")
print("=" * 70)
print("Cross-WE pooled pair statistic summary:")
if b4_per_pair.height:
    n_sig_pair = b4_per_pair.filter(pl.col("pair_ci_excludes_zero")).height
    n_sig_neither = b4_per_pair.filter(
        pl.col("pair_ci_excludes_zero") & (pl.col("cells_individually_sig") == 0)
    ).height
    n_sig_one = b4_per_pair.filter(
        pl.col("pair_ci_excludes_zero") & (pl.col("cells_individually_sig") == 1)
    ).height
    n_lift = n_sig_neither + n_sig_one
    print(f"  Total cross-WE pairs (both WEs present): {b4_per_pair.height}")
    print(f"  pair_ci_excludes_zero:  {n_sig_pair}")
    print(f"  Lift (pair sig but NOT both cells individually sig): {n_lift}")
    print(f"    of which 0 cells sig: {n_sig_neither}")
    print(f"    of which 1 cell sig:  {n_sig_one}")
    print(f"  See {OUT_DIR / 'b4_per_pair.parquet'}")
else:
    print("  (no pair results — b4_per_pair is empty)")
