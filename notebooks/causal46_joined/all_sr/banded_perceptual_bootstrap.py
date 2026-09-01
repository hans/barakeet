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
# # Banded within-completion bootstrap — all speech-responsive sites
#
# Sibling of `t_tests_all_sr.py`. Same all-SR site universe and same
# per-step class-balanced within-completion bootstrap machinery
# (`_within_completion.py`, unchanged), but **replaces the sliding
# searchlight with 3 fixed, physiologically-motivated bands per
# `(phoneme_pair, word_end)`**:
#
# | band | span | rationale |
# |------|------|-----------|
# | `pre_pod`     | `[50 ms, POD)`          | percept-predictive / anticipatory bias |
# | `post_pod`    | `[POD, offset)`         | disambiguation / resolution |
# | `post_offset` | `[offset, offset+tail)` | the diffuse perceptual tail |
#
# **Why a separate notebook, not an edit to `t_tests_all_sr.py`.** That
# notebook is gated by `t_tests_all_sr_reconciliation.py`, which asserts it
# reproduces `t_tests.py` bit-exact on the AS cells; `perceptual_acoustic_partition`
# hard-fails if that gate doesn't pass. `t_tests.py` still uses the searchlight,
# so a fixed-band scheme legitimately diverges from it — this is a *different*
# test, not a bug. Kept sibling so the reconciled pipeline stays green.
#
# **Why bands, not searchlight.** The searchlight selects its best window on
# the perceptual contrast itself, then reads significance at that same window
# (double-dip), and inflates the within-cell multiple-comparisons burden. A
# small set of *pre-specified* bands removes the within-cell selection (a plain
# Bonferroni over 3 bands is honest), matches the temporally-diffuse perceptual
# response (better SNR than narrow bins), and — crucially — the `pre_pod` band
# starts at 50 ms, not 0, so the pre-evoked baseline (the source of the 40–80 ms
# artifact "sites") is excluded.
#
# **Kept for downstream use:** `mean_diff_raw_null` (per-replicate,
# per-band label-permutation null) — this powers the population count-vs-null
# (the co-localization headline), which does not need the per-cell permutation
# floor.
#
# Outputs (sibling tree `outputs/causal46_joined/banded_perceptual_bootstrap/`):
# - `b4_bootstrap.parquet`      — per (cell, band, replicate) raw + null draws
# - `b4_per_band.parquet`       — per (cell, band) bootstrap CI + empirical p
# - `b4_per_cell.parquet`       — per cell: best band + Bonferroni-over-bands p
# - `cell_manifest.parquet`
# - `population_count_vs_null.csv` — observed vs null count of sig cells, per band

# %%
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from src.stimuli import PHONEME_PAIR_TO_WORD_ENDS, OFFSET_DICT, POD_dict
from src.viz_paper import epoch_sfreq, epoch_tmin
from src.viz_provisional import load_epochs_dict

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    n_per_class_from_per_step,
    per_step_class_counts,
    resolve_behavior_col,
    searchlight_mean_diff,
    select_cell_trials_bootstrap,
)

# %% tags=["parameters"]
sr_site_universe_path = "outputs/causal46_joined/sr_site_universe/sr_site_universe.parquet"
epoch_dir = "outputs/epochs_preprocessed"
trial_balance_path = "outputs/causal46_joined/trial_balance_index_all_sr/trial_balance_index.csv"
outdir = "outputs/causal46_joined/banded_perceptual_bootstrap"
min_class_k = 4
n_bootstrap = 1000
band_a_early_s = 0.050          # pre_pod band starts here (NOT 0 — skip baseline)
word_end_tail_samples = 20      # +200 ms past word offset (sfreq=100)

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
EPOCH_DIR = Path(epoch_dir)

K = min_class_k
R = n_bootstrap
CI_LOW, CI_HIGH = 2.5, 97.5
WORD_END_TAIL_SAMPLES = word_end_tail_samples

print(f"EPOCH_DIR: {EPOCH_DIR}  (exists: {EPOCH_DIR.exists()})")
print(f"K = {K}   R = {R}   CI = [{CI_LOW}, {CI_HIGH}]")
print(f"band_a starts at {band_a_early_s * 1000:.0f} ms; tail = "
      f"{WORD_END_TAIL_SAMPLES} samples")


# %% [markdown]
# ## Band definition (samples) per (phoneme_pair, word_end)

# %%
def _s(t_s: float) -> int:
    """seconds -> sample index."""
    return int(round((t_s - epoch_tmin) * epoch_sfreq))


def behav_bands(phoneme_pair: str, word_end: str, n_times: int) -> list[tuple[str, int, int]]:
    """3 fixed bands as (name, smin, smax); dropped if degenerate / OOB.

    pre_pod   = [band_a_early_s, POD)
    post_pod  = [POD, offset)
    post_offset = [offset, offset + tail]
    """
    a0 = _s(band_a_early_s)
    pod = _s(POD_dict[phoneme_pair])
    off = _s(OFFSET_DICT[word_end])
    c1 = min(off + WORD_END_TAIL_SAMPLES, n_times)
    candidates = [
        ("pre_pod", a0, pod),
        ("post_pod", pod, off),
        ("post_offset", off, c1),
    ]
    return [
        (name, smin, smax)
        for name, smin, smax in candidates
        if 0 <= smin < smax <= n_times
    ]


# %% [markdown]
# ## Load site universe, trial balance, epochs

# %%
site_universe = pl.read_parquet(sr_site_universe_path)
print(f"all-SR sites: {site_universe.height}  "
      f"(acoustic_significant: {int(site_universe['acoustic_significant'].sum())})")

trial_balance = pl.read_csv(trial_balance_path)
print(f"trial_balance: {trial_balance.height} rows")

epochs_dict = load_epochs_dict(EPOCH_DIR)
print(f"epochs loaded: {sorted(epochs_dict)}")

# %% [markdown]
# ## B4 cell definition (per-step balanced pool across ambiguous steps)
#
# Same qualifying-cell derivation as `t_tests_all_sr.py`: ambiguous-step cells
# from the all-SR trial balance, annotated (not filtered) with
# `acoustic_significant` / `phon_smin` / `phon_smax` from the site universe.

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
        site_universe.select([
            "subject", "electrode_idx", "phoneme_pair",
            "acoustic_significant", "phon_smin", "phon_smax",
        ]),
        on=["subject", "electrode_idx", "phoneme_pair"], how="left",
    )
    .with_columns(pl.col("acoustic_significant").fill_null(False))
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
)
print(f"B4 qualifying cells: {b4_qualified.height}")


# %% [markdown]
# ## Bootstrap over the 3 bands
#
# For each replicate: draw per-step class-balanced trials, evaluate raw
# mean-diff on each fixed band, and a within-step label-permutation null on
# the same draw (identical null construction to `t_tests_all_sr.py`). Each
# band is scored by calling `searchlight_mean_diff` with a single window
# spanning it (window_size == stride == band width -> exactly one window).

# %%
def _band_mean_diffs(hga, pos_idx, neg_idx, bands):
    """{band_name: MeanDiffWindow} evaluating each fixed band as one window."""
    out = {}
    for name, smin, smax in bands:
        w = smax - smin
        res = searchlight_mean_diff(
            hga, pos_idx, neg_idx,
            search_smin=smin, search_smax=smax, window_size=w, stride=w,
        )
        if res:
            out[name] = res[0]
    return out


def bootstrap_cell_banded(*, md_pp, hga, bhv_col, word_end, qualifying_steps,
                          bands, R, base_seed=0):
    per_step = per_step_class_counts(
        md_pp, word_end=word_end, qualifying_steps=qualifying_steps,
        group_col=bhv_col,
    )
    n_per_class = n_per_class_from_per_step(per_step)
    rows: list[dict] = []
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        draws = select_cell_trials_bootstrap(per_step, rng=rng)
        keys = sorted(draws.keys())
        raw_pos_key, raw_neg_key = keys[1], keys[0]
        raw = _band_mean_diffs(hga, draws[raw_pos_key], draws[raw_neg_key], bands)

        # Within-step label-permutation null on the same draw (per-step balance
        # preserved) — identical construction to t_tests_all_sr.bootstrap_cell.
        _step_sizes = [
            min(len(v) for v in by_cls.values())
            for by_cls in per_step.values()
            if min(len(v) for v in by_cls.values()) > 0
        ]
        _null_pos_parts, _null_neg_parts, _off = [], [], 0
        for _ns in _step_sizes:
            _pool = np.concatenate([
                draws[raw_pos_key][_off:_off + _ns],
                draws[raw_neg_key][_off:_off + _ns],
            ])
            rng.shuffle(_pool)
            _null_pos_parts.append(_pool[:_ns])
            _null_neg_parts.append(_pool[_ns:])
            _off += _ns
        null = _band_mean_diffs(
            hga, np.concatenate(_null_pos_parts), np.concatenate(_null_neg_parts), bands
        )
        for name, smin, smax in bands:
            if name not in raw:
                continue
            w = raw[name]
            rows.append({
                "replicate": r,
                "band": name,
                "smin": w.smin, "smax": w.smax,
                "tmin": w.smin / epoch_sfreq + epoch_tmin,
                "tmax": w.smax / epoch_sfreq + epoch_tmin,
                "mean_diff_raw": w.mean_diff,
                "mean_diff_raw_null": null[name].mean_diff if name in null else float("nan"),
                "n_per_class": n_per_class,
            })
    return rows, n_per_class


# %% [markdown]
# ## Run — grouped by (subject, phoneme_pair) to load+baseline once

# %%
b4_boot_rows: list[dict] = []
b4_cell_manifest: list[dict] = []
b4_failures: list[dict] = []

b4_groups: dict[tuple[str, str], list[dict]] = {}
for row in b4_qualified.iter_rows(named=True):
    b4_groups.setdefault((row["subject"], row["phoneme_pair"]), []).append(row)
print(f"grouped into {len(b4_groups)} (subject, phoneme_pair) loads")

pbar = tqdm(total=b4_qualified.height, desc="banded bootstrap")
for (subj, pp), group_rows in b4_groups.items():
    if subj not in epochs_dict:
        for row in group_rows:
            b4_failures.append({**row, "error": "no epochs for subject"})
        pbar.update(len(group_rows))
        continue
    ep = epochs_dict[subj]
    md = ep.metadata
    bhv_col = resolve_behavior_col(md)
    pp_mask = (md["phoneme_pair"] == pp).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    all_hga = ep_pp.copy().apply_baseline((None, 0)).get_data()
    n_times = all_hga.shape[2]

    for row in group_rows:
        hga = all_hga[:, int(row["electrode_idx"]), :]
        steps = [int(s) for s in row["qualifying_steps"]]
        bands = behav_bands(row["phoneme_pair"], row["word_end"], n_times)
        manifest_base = {
            "subject": subj,
            "electrode_idx": int(row["electrode_idx"]),
            "phoneme_pair": row["phoneme_pair"],
            "word_end": row["word_end"],
            "qualifying_steps": ",".join(str(s) for s in steps),
            "n_per_class": int(row["n_per_class"]),
            "acoustic_significant": bool(row["acoustic_significant"]),
            "phon_smin": row["phon_smin"],
            "phon_smax": row["phon_smax"],
            "n_bands": len(bands),
        }
        if not bands:
            b4_cell_manifest.append({**manifest_base, "status": "no_valid_band"})
            pbar.update(1)
            continue
        try:
            rows, n_per_class = bootstrap_cell_banded(
                md_pp=md_pp, hga=hga, bhv_col=bhv_col,
                word_end=row["word_end"], qualifying_steps=steps,
                bands=bands, R=R,
            )
            for rr in rows:
                b4_boot_rows.append({
                    "subject": subj,
                    "electrode_idx": int(row["electrode_idx"]),
                    "phoneme_pair": row["phoneme_pair"],
                    "word_end": row["word_end"],
                    "acoustic_significant": bool(row["acoustic_significant"]),
                    **rr,
                })
            b4_cell_manifest.append({**manifest_base, "status": "ok"})
        except Exception as exc:
            tb = traceback.format_exc()
            b4_failures.append({
                **{k: row[k] for k in
                   ("subject", "electrode_idx", "phoneme_pair", "word_end")},
                "error": repr(exc), "traceback": tb,
            })
            print(f"FAILED: {subj} e{row['electrode_idx']} {row['phoneme_pair']} "
                  f"{row['word_end']}\n{tb}")
        pbar.update(1)
pbar.close()

b4_boot = pl.DataFrame(b4_boot_rows) if b4_boot_rows else pl.DataFrame()
if b4_boot.height:
    b4_boot.write_parquet(OUT_DIR / "b4_bootstrap.parquet")
print(f"bootstrap rows: {b4_boot.height}  (failures: {len(b4_failures)})")

cell_manifest = pl.DataFrame(b4_cell_manifest)
cell_manifest.write_parquet(OUT_DIR / "cell_manifest.parquet")
print(cell_manifest.group_by(["status"]).len().sort("status"))

# %% [markdown]
# ## Per-band aggregation (bootstrap CI + empirical p)

# %%
CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]


def per_band_summary(boot: pl.DataFrame) -> pl.DataFrame:
    if boot.height == 0:
        return pl.DataFrame()
    grouped = (
        boot
        .group_by(CELL_KEYS + ["band", "smin", "smax", "tmin", "tmax"])
        .agg(
            pl.col("mean_diff_raw").median().alias("mean_diff_raw_med"),
            pl.col("mean_diff_raw").quantile(CI_LOW / 100).alias("mean_diff_raw_ci_lo"),
            pl.col("mean_diff_raw").quantile(CI_HIGH / 100).alias("mean_diff_raw_ci_hi"),
            (pl.col("mean_diff_raw") <= 0).cast(pl.Float64).mean().alias("frac_raw_le0"),
            (pl.col("mean_diff_raw") >= 0).cast(pl.Float64).mean().alias("frac_raw_ge0"),
            # null-band CI, from the label-permutation draws (for count-vs-null)
            pl.col("mean_diff_raw_null").quantile(CI_LOW / 100).alias("null_ci_lo"),
            pl.col("mean_diff_raw_null").quantile(CI_HIGH / 100).alias("null_ci_hi"),
            pl.col("n_per_class").first().alias("n_per_class"),
            pl.col("acoustic_significant").first().alias("acoustic_significant"),
            pl.col("replicate").max().alias("R_replicates"),
        )
    )
    return grouped.with_columns([
        pl.min_horizontal(2 * pl.min_horizontal("frac_raw_le0", "frac_raw_ge0"),
                          pl.lit(1.0)).alias("emp_p_raw"),
        ((pl.col("mean_diff_raw_ci_lo") > 0) | (pl.col("mean_diff_raw_ci_hi") < 0))
            .alias("ci_raw_excludes_zero"),
        ((pl.col("null_ci_lo") > 0) | (pl.col("null_ci_hi") < 0))
            .alias("null_ci_excludes_zero"),
    ]).sort(CELL_KEYS + ["smin"])


b4_per_band = per_band_summary(b4_boot)
if b4_per_band.height:
    b4_per_band.write_parquet(OUT_DIR / "b4_per_band.parquet")
print(f"per_band rows: {b4_per_band.height}")

# %% [markdown]
# ## Per-cell: best band + Bonferroni over the (pre-specified) bands
#
# Because the bands are fixed a priori, a plain Bonferroni over the cell's
# bands is an honest per-cell correction (no window selection to undo).
# `bonferroni_emp_p = min(1, n_bands * min_band_emp_p)`.

# %%
def per_cell_banded(per_band: pl.DataFrame) -> pl.DataFrame:
    if per_band.height == 0:
        return pl.DataFrame()
    best = (
        per_band
        .with_columns(pl.col("mean_diff_raw_med").abs().alias("__rank"))
        .sort(CELL_KEYS + ["__rank"], descending=[False] * len(CELL_KEYS) + [True])
        .group_by(CELL_KEYS, maintain_order=True)
        .head(1)
        .drop("__rank")
        .rename({
            "band": "best_band", "smin": "best_smin", "smax": "best_smax",
            "tmin": "best_tmin", "tmax": "best_tmax",
            "mean_diff_raw_med": "best_mean_diff_raw_med",
            "emp_p_raw": "best_emp_p_raw",
            "ci_raw_excludes_zero": "best_ci_raw_excludes_zero",
        })
        .select(CELL_KEYS + [
            "best_band", "best_smin", "best_smax", "best_tmin", "best_tmax",
            "best_mean_diff_raw_med", "best_emp_p_raw", "best_ci_raw_excludes_zero",
            "n_per_class", "acoustic_significant",
        ])
    )
    agg = (
        per_band
        .group_by(CELL_KEYS)
        .agg(
            pl.len().alias("n_bands"),
            pl.col("emp_p_raw").min().alias("min_band_emp_p"),
            pl.col("ci_raw_excludes_zero").any().alias("any_band_ci_excl_zero"),
        )
        .with_columns(
            pl.min_horizontal(pl.col("n_bands") * pl.col("min_band_emp_p"), pl.lit(1.0))
            .alias("bonferroni_emp_p")
        )
    )
    return best.join(agg, on=CELL_KEYS, how="left")


b4_per_cell = per_cell_banded(b4_per_band)
if b4_per_cell.height:
    b4_per_cell.write_parquet(OUT_DIR / "b4_per_cell.parquet")
print(f"per_cell rows: {b4_per_cell.height}")

# %% [markdown]
# ## Population count-vs-null (the co-localization headline)
#
# Per band, compare the observed number of cells whose bootstrap CI excludes
# zero against the number under the label-permutation null (`null_ci_excludes_zero`).
# A significant OBSERVED >> NULL excess is population evidence for
# non-acoustic perceptual coding, independent of any single cell surviving
# per-cell FDR. Split by `acoustic_significant` so the non-acoustic cell (the
# scientifically load-bearing partition) is called out.

# %%
if b4_per_band.height:
    pop = (
        b4_per_band
        .group_by(["band", "acoustic_significant"])
        .agg(
            pl.len().alias("n_cells"),
            pl.col("ci_raw_excludes_zero").sum().alias("n_observed_sig"),
            pl.col("null_ci_excludes_zero").sum().alias("n_null_sig"),
        )
        .with_columns([
            (pl.col("n_observed_sig") / pl.col("n_cells")).alias("obs_rate"),
            (pl.col("n_null_sig") / pl.col("n_cells")).alias("null_rate"),
            (pl.col("n_observed_sig") - pl.col("n_null_sig")).alias("excess"),
        ])
        .sort(["band", "acoustic_significant"])
    )
    pop.write_csv(OUT_DIR / "population_count_vs_null.csv")
    print(pop)
else:
    print("no bootstrap output — skipping population count-vs-null")
