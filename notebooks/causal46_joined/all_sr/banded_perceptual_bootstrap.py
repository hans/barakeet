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
# # Banded within-completion perceptual test — all speech-responsive sites
#
# Sibling of `t_tests_all_sr.py`. Same all-SR site universe and the same
# per-step class-balanced within-completion machinery (`_within_completion.py`,
# unchanged), but **replaces the sliding searchlight with 3 fixed,
# physiologically-motivated bands per `(phoneme_pair, word_end)`**:
#
# | band | span | rationale |
# |------|------|-----------|
# | `pre_pod`     | `[50 ms, POD)`          | percept-predictive / anticipatory bias |
# | `post_pod`    | `[POD, offset)`         | disambiguation / resolution |
# | `post_offset` | `[offset, offset+tail)` | the diffuse perceptual tail |
#
# **Cells stay completion-specific** — one per `(subject, electrode_idx,
# phoneme_pair, word_end)`. Nothing is pooled or required to replicate across
# word-ends: completion-specific responses are the hypothesis, not noise.
#
# **Significance = per-cell permutation p, then BH-FDR** across all band-cells.
# For each band-cell we draw `R` replicates; each replicate bootstraps the
# per-step class-balanced trials (→ `mean_diff_raw`) and computes a within-step
# label-permutation of the same draw (→ `mean_diff_raw_null`). The per-cell
# statistics:
#
# - `observed`  = median of the `R` real `mean_diff_raw` draws (effect size);
#   `ci_lo/hi`  = 2.5/97.5 percentiles (for reporting).
# - `perm_p`    = `(1 + #{ |null| >= |observed| }) / (R + 1)` — a Monte-Carlo
#   permutation p (floors at `1/(R+1)`). This REPLACES the earlier
#   `null_ci_excludes_zero` readout, which was miscalibrated: the label-null is
#   re-centered at ~0 every replicate, so its own CI straddles zero by
#   construction and never "excludes zero" — it is not a valid benchmark.
# - `z` = `observed / null_sd` — a floor-free ranking statistic (perm_p ties at
#   the `1/(R+1)` floor for the strongest cells; rank by `|z|` / `|observed|`,
#   decide inclusion by `q`).
#
# **BH-FDR** over the whole band-cell family gives `q_value` / `significant`.
# On this data the operative BH boundary is the rank-k threshold with ~100+
# real effects (≈ `k/N · alpha`), not the rank-1 `alpha/N` — so `R ≈ 1e4`
# resolves it with headroom (floor `1e-4` well below the ~`3.7e-4` boundary).
# Bump `R` only if you want p-level *ranking* of the top sites; `|z|` already
# orders them.
#
# **Scales in storage:** per-cell summaries are computed inside the trial loop;
# the full per-replicate draws are discarded unless `save_draws=True`, so
# storage is O(cells) regardless of `R`.
#
# Outputs (`outputs/causal46_joined/banded_perceptual_bootstrap/`):
# - `b4_per_band.parquet`   — per (cell, band): observed, CI, perm_p, q, significant, z
# - `b4_per_cell.parquet`   — per cell: most-significant band + its q / significant
# - `significant_sites.csv` — BH-significant band-cells, ranked by |z|
# - `population_summary.csv` — n significant per band x acoustic_significant
# - `cell_manifest.parquet`
# - `b4_bootstrap.parquet`  — only if save_draws=True (large at high R)

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
n_bootstrap = 10000             # R: bootstrap + label-permutation replicates per band-cell
alpha_fdr = 0.05                # BH-FDR level
band_a_early_s = 0.050          # pre_pod band starts here (NOT 0 — skip baseline)
word_end_tail_samples = 20      # +200 ms past word offset (sfreq=100)
save_draws = False              # write the full per-replicate draws (large at high R)

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
EPOCH_DIR = Path(epoch_dir)

K = min_class_k
R = n_bootstrap
CI_LOW, CI_HIGH = 2.5, 97.5
WORD_END_TAIL_SAMPLES = word_end_tail_samples

print(f"EPOCH_DIR: {EPOCH_DIR}  (exists: {EPOCH_DIR.exists()})")
print(f"K = {K}   R = {R}   perm_p floor = {1/(R+1):.2e}   alpha_fdr = {alpha_fdr}")
print(f"band_a starts at {band_a_early_s * 1000:.0f} ms; tail = "
      f"{WORD_END_TAIL_SAMPLES} samples;  save_draws = {save_draws}")


# %% [markdown]
# ## Band definition (samples) per (phoneme_pair, word_end)

# %%
def _s(t_s: float) -> int:
    """seconds -> sample index."""
    return int(round((t_s - epoch_tmin) * epoch_sfreq))


def behav_bands(phoneme_pair: str, word_end: str, n_times: int) -> list[tuple[str, int, int]]:
    """3 fixed bands as (name, smin, smax); dropped if degenerate / OOB.

    pre_pod     = [band_a_early_s, POD)
    post_pod    = [POD, offset)
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
# ## Per-band permutation test (summarized inside the loop)
#
# Each band is scored by calling `searchlight_mean_diff` with a single window
# spanning it (window_size == stride == band width -> exactly one window).
# `bootstrap_cell_banded` returns one summary row per band (+ optionally the
# raw draws), so global memory / storage is O(cells), not O(cells x R).

# %%
def _band_mean_diffs(hga, pos_idx, neg_idx, bands):
    """{band_name: mean_diff (float)} evaluating each fixed band as one window."""
    out = {}
    for name, smin, smax in bands:
        w = smax - smin
        res = searchlight_mean_diff(
            hga, pos_idx, neg_idx,
            search_smin=smin, search_smax=smax, window_size=w, stride=w,
        )
        if res:
            out[name] = res[0].mean_diff
    return out


def bootstrap_cell_banded(*, md_pp, hga, bhv_col, word_end, qualifying_steps,
                          bands, R, base_seed=0, keep_draws=False):
    """Return (summary_rows, n_per_class[, draw_rows]).

    summary_rows: one dict per band with observed effect, CI, perm_p, z.
    """
    per_step = per_step_class_counts(
        md_pp, word_end=word_end, qualifying_steps=qualifying_steps,
        group_col=bhv_col,
    )
    n_per_class = n_per_class_from_per_step(per_step)

    real = {name: np.empty(R) for name, _, _ in bands}
    null = {name: np.empty(R) for name, _, _ in bands}
    draw_rows: list[dict] = []
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        draws = select_cell_trials_bootstrap(per_step, rng=rng)
        keys = sorted(draws.keys())
        raw_pos_key, raw_neg_key = keys[1], keys[0]
        rvals = _band_mean_diffs(hga, draws[raw_pos_key], draws[raw_neg_key], bands)

        # within-step label-permutation null on the same draw (per-step balance
        # preserved) — identical construction to t_tests_all_sr.bootstrap_cell.
        _step_sizes = [
            min(len(v) for v in by_cls.values())
            for by_cls in per_step.values()
            if min(len(v) for v in by_cls.values()) > 0
        ]
        _npos, _nneg, _off = [], [], 0
        for _ns in _step_sizes:
            _pool = np.concatenate([
                draws[raw_pos_key][_off:_off + _ns],
                draws[raw_neg_key][_off:_off + _ns],
            ])
            rng.shuffle(_pool)
            _npos.append(_pool[:_ns])
            _nneg.append(_pool[_ns:])
            _off += _ns
        nvals = _band_mean_diffs(hga, np.concatenate(_npos), np.concatenate(_nneg), bands)

        for name, smin, smax in bands:
            real[name][r] = rvals.get(name, np.nan)
            null[name][r] = nvals.get(name, np.nan)
            if keep_draws:
                draw_rows.append({
                    "replicate": r, "band": name,
                    "mean_diff_raw": rvals.get(name, np.nan),
                    "mean_diff_raw_null": nvals.get(name, np.nan),
                })

    summary_rows = []
    for name, smin, smax in bands:
        rv, nv = real[name], null[name]
        observed = float(np.nanmedian(rv))
        null_sd = float(np.nanstd(nv))
        n_null = np.sum(~np.isnan(nv))
        n_exceed = int(np.sum(np.abs(nv) >= abs(observed)))
        perm_p = (1 + n_exceed) / (n_null + 1) if n_null else np.nan
        summary_rows.append({
            "band": name, "smin": smin, "smax": smax,
            "tmin": smin / epoch_sfreq + epoch_tmin,
            "tmax": smax / epoch_sfreq + epoch_tmin,
            "observed": observed,
            "ci_lo": float(np.nanpercentile(rv, CI_LOW)),
            "ci_hi": float(np.nanpercentile(rv, CI_HIGH)),
            "null_sd": null_sd,
            "z": observed / null_sd if null_sd > 0 else np.nan,
            "perm_p": perm_p,
            "n_per_class": n_per_class,
        })
    return (summary_rows, n_per_class, draw_rows) if keep_draws else (summary_rows, n_per_class)


# %% [markdown]
# ## Run — grouped by (subject, phoneme_pair) to load+baseline once

# %%
band_rows: list[dict] = []
draw_rows_all: list[dict] = []
cell_manifest_rows: list[dict] = []
failures: list[dict] = []

groups: dict[tuple[str, str], list[dict]] = {}
for row in b4_qualified.iter_rows(named=True):
    groups.setdefault((row["subject"], row["phoneme_pair"]), []).append(row)
print(f"grouped into {len(groups)} (subject, phoneme_pair) loads")

pbar = tqdm(total=b4_qualified.height, desc=f"banded perm (R={R})")
for (subj, pp), group_rows in groups.items():
    if subj not in epochs_dict:
        for row in group_rows:
            failures.append({**row, "error": "no epochs for subject"})
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
        base = {
            "subject": subj, "electrode_idx": int(row["electrode_idx"]),
            "phoneme_pair": row["phoneme_pair"], "word_end": row["word_end"],
            "qualifying_steps": ",".join(str(s) for s in steps),
            "n_per_class": int(row["n_per_class"]),
            "acoustic_significant": bool(row["acoustic_significant"]),
            "phon_smin": row["phon_smin"], "phon_smax": row["phon_smax"],
            "n_bands": len(bands),
        }
        if not bands:
            cell_manifest_rows.append({**base, "status": "no_valid_band"})
            pbar.update(1)
            continue
        try:
            result = bootstrap_cell_banded(
                md_pp=md_pp, hga=hga, bhv_col=bhv_col,
                word_end=row["word_end"], qualifying_steps=steps,
                bands=bands, R=R, keep_draws=save_draws,
            )
            summary_rows = result[0]
            keys = {k: base[k] for k in
                    ("subject", "electrode_idx", "phoneme_pair", "word_end",
                     "acoustic_significant")}
            for sr in summary_rows:
                band_rows.append({**keys, **sr})
            if save_draws:
                for dr in result[2]:
                    draw_rows_all.append({**keys, **dr})
            cell_manifest_rows.append({**base, "status": "ok"})
        except Exception as exc:
            tb = traceback.format_exc()
            failures.append({
                **{k: row[k] for k in
                   ("subject", "electrode_idx", "phoneme_pair", "word_end")},
                "error": repr(exc), "traceback": tb,
            })
            print(f"FAILED: {subj} e{row['electrode_idx']} {row['phoneme_pair']} "
                  f"{row['word_end']}\n{tb}")
        pbar.update(1)
pbar.close()

cell_manifest = pl.DataFrame(cell_manifest_rows)
cell_manifest.write_parquet(OUT_DIR / "cell_manifest.parquet")
print(cell_manifest.group_by("status").len().sort("status"))
print(f"failures: {len(failures)}")

if save_draws and draw_rows_all:
    pl.DataFrame(draw_rows_all).write_parquet(OUT_DIR / "b4_bootstrap.parquet")
    print(f"wrote b4_bootstrap.parquet ({len(draw_rows_all)} rows)")


# %% [markdown]
# ## BH-FDR across the band-cell family
#
# The family is every tested band-cell (completion-specific, no pooling). BH
# q-values on `perm_p`; `significant = q <= alpha_fdr`. Strongest cells tie at
# the `perm_p` floor — rank them by `|z|` (and `|observed|`), not by p.

# %%
def bh_qvalues(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


b4_per_band = pl.DataFrame(band_rows)
if b4_per_band.height:
    q = bh_qvalues(b4_per_band["perm_p"].to_numpy())
    b4_per_band = b4_per_band.with_columns([
        pl.Series("q_value", q),
        (pl.Series("q_value", q) <= alpha_fdr).alias("significant"),
        pl.col("observed").abs().alias("abs_effect"),
        pl.col("z").abs().alias("abs_z"),
    ]).sort(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin"])
    b4_per_band.write_parquet(OUT_DIR / "b4_per_band.parquet")
    n_sig = int(b4_per_band["significant"].sum())
    print(f"per_band rows: {b4_per_band.height}  BH-significant band-cells: {n_sig}")
else:
    print("no band-cells tested")

# %% [markdown]
# ## Per-cell: most-significant band

# %%
CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
if b4_per_band.height:
    b4_per_cell = (
        b4_per_band
        .sort(CELL_KEYS + ["perm_p", "abs_z"], descending=[False] * 4 + [False, True])
        .group_by(CELL_KEYS, maintain_order=True)
        .head(1)
        .rename({
            "band": "best_band", "observed": "best_observed", "z": "best_z",
            "perm_p": "best_perm_p", "q_value": "best_q_value",
            "significant": "best_significant",
        })
        .select(CELL_KEYS + [
            "best_band", "best_observed", "best_z", "best_perm_p",
            "best_q_value", "best_significant", "n_per_class", "acoustic_significant",
        ])
    )
    b4_per_cell.write_parquet(OUT_DIR / "b4_per_cell.parquet")
    print(f"per_cell rows: {b4_per_cell.height}  "
          f"BH-significant cells: {int(b4_per_cell['best_significant'].sum())}")

# %% [markdown]
# ## Significant-site list (ranked by |z|) + population summary

# %%
if b4_per_band.height:
    sig = (
        b4_per_band
        .filter(pl.col("significant"))
        .sort("abs_z", descending=True)
        .select(CELL_KEYS + ["band", "tmin", "tmax", "observed", "z",
                             "perm_p", "q_value", "n_per_class", "acoustic_significant"])
    )
    sig.write_csv(OUT_DIR / "significant_sites.csv")
    print(f"significant_sites.csv: {sig.height} band-cells "
          f"({sig.select(['subject','electrode_idx']).unique().height} electrodes, "
          f"{sig.filter(~pl.col('acoustic_significant')).height} non-acoustic)")

    pop = (
        b4_per_band
        .group_by(["band", "acoustic_significant"])
        .agg(
            pl.len().alias("n_cells"),
            pl.col("significant").sum().alias("n_significant"),
        )
        .with_columns((pl.col("n_significant") / pl.col("n_cells")).alias("sig_rate"))
        .sort(["band", "acoustic_significant"])
    )
    pop.write_csv(OUT_DIR / "population_summary.csv")
    print(pop)
