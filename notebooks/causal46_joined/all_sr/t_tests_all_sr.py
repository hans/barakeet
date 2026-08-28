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
# # B4 within-completion bootstrap CIs — all speech-responsive sites
#
# Step 3 of the all-SR perceptual fork
# (`docs/superpowers/plans/2026-08-27-all-speech-responsive-perceptual.md`).
# Same per-step class-balanced bootstrap as `t_tests.py`
# (`_within_completion.py` is the shared, unmodified source of truth), run
# over the full speech-responsive site universe instead of only
# acoustic-significant (AS) sites:
#
# - `b4_qualified` **left-joins** `sr_site_universe.parquet` (all-SR, Step 1)
#   instead of inner-joining a p-value-filtered acoustic peaks table — every
#   qualifying SR cell survives; `acoustic_significant` /
#   `phon_smin` / `phon_smax` are carried through as annotations (null
#   smin/smax when not acoustic-significant).
# - **Polarity / aligned labelling is out of scope here** (per plan): it
#   requires an acoustic window to align to, which is undefined for
#   non-acoustic sites, and a constant per-cell sign flip leaves a two-sided
#   CI test invariant anyway. `mean_diff_raw` / `ci_raw_excludes_zero` is the
#   headline contrast — the same raw, uncorrected bootstrap CI that
#   `t_tests.py`'s output is consumed as everywhere downstream in the real
#   pipeline (`plot_for_paper`, `behavioral_discriminative_windows.py`,
#   `late_perceptual_projection.py`'s candidate gate). No maxstat / BH-FDR /
#   TFCE correction is applied — an earlier draft added a max-|z| + BH-FDR
#   correction modeled on `late_integration_maxstat_significance.py`, but
#   that notebook has no Snakefile rule and isn't part of what actually
#   feeds `plot_for_paper`; it wasn't real precedent for this statistic.
#   Matching how `t_tests.py`'s own output is used elsewhere keeps this
#   fork consistent with the rest of the codebase.
#
# No star-plot gallery / cross-WE pooled pair statistic here (both depend on
# the aligned contrast, which is out of scope).
#
# Outputs (mirrors `t_tests/` naming, sibling tree):
# - `outputs/causal46_joined/t_tests_all_sr/b4_bootstrap.parquet`
# - `outputs/causal46_joined/t_tests_all_sr/b4_per_window.parquet`
# - `outputs/causal46_joined/t_tests_all_sr/b4_per_cell.parquet`
# - `outputs/causal46_joined/t_tests_all_sr/cell_manifest.parquet`
# - `outputs/causal46_joined/t_tests_all_sr/population_summary.csv`
# - `outputs/causal46_joined/t_tests_all_sr/population_summary.pdf`

# %%
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from src.data import get_electrode_df
from src.stimuli import PHONEME_PAIR_TO_WORD_ENDS, OFFSET_DICT
from src.viz_paper import epoch_sfreq, epoch_tmin
from src.viz_provisional import load_epochs_dict

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    extract_hga,
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
outdir = "outputs/causal46_joined/t_tests_all_sr"
min_class_k = 4
window_size = 10
stride = 10
n_bootstrap = 1000

# %%
REPO = Path(".").resolve()
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_DIR = Path(epoch_dir)

WINDOW_SIZE = window_size
STRIDE = stride
WORD_END_TAIL_SAMPLES = 20  # +200 ms past word offset (sfreq=100)

K = min_class_k
R = n_bootstrap        # bootstrap replicates per cell
CI_LOW, CI_HIGH = 2.5, 97.5

print(f"REPO:      {REPO}")
print(f"EPOCH_DIR: {EPOCH_DIR}  (exists: {EPOCH_DIR.exists()})")
print(f"K = {K}   R = {R}   CI = [{CI_LOW}, {CI_HIGH}]")
print(f"window={WINDOW_SIZE}  stride={STRIDE}")

# %% [markdown]
# ## Load all-SR site universe, trial balance, and epochs

# %%
site_universe = pl.read_parquet(sr_site_universe_path)
print(f"all-SR sites: {site_universe.height}  "
      f"(acoustic_significant: {int(site_universe['acoustic_significant'].sum())})")

trial_balance = pl.read_csv(trial_balance_path)
print(f"trial_balance: {trial_balance.height} rows")

epochs_dict = load_epochs_dict(EPOCH_DIR)
print(f"epochs loaded: {sorted(epochs_dict)}")

# %% [markdown]
# ## Word-end behavioral search bound (samples)
#
# Pair-level max across word-ends (matches `t_tests.py`'s dev override: the
# acoustic window is not used to bound the search here either).

# %%
def word_end_search_smax(word_end: str) -> int:
    offset_s = OFFSET_DICT[word_end]
    sample = int(round((offset_s - epoch_tmin) * epoch_sfreq))
    return sample + WORD_END_TAIL_SAMPLES


WE_SMAX = {we: word_end_search_smax(we) for we in OFFSET_DICT.keys()}
print(f"word-end search_smax (samples): {WE_SMAX}")

PAIR_SMAX = {
    pp: max(WE_SMAX[we] for we in wes)
    for pp, wes in PHONEME_PAIR_TO_WORD_ENDS.items()
}
print(f"pair search_smax (samples): {PAIR_SMAX}")


def behav_search_range(phoneme_pair: str) -> tuple[int, int]:
    return 0, int(PAIR_SMAX[phoneme_pair])


# %% [markdown]
# ## B4 cell definition (per-step balanced pool across ambiguous steps)
#
# Left join against the all-SR universe: every qualifying SR cell survives,
# `acoustic_significant` / `phon_smin` / `phon_smax` come along as labels.

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
            "acoustic_significant", "phon_smin", "phon_smax", "acoustic_peak_auc",
        ]),
        on=["subject", "electrode_idx", "phoneme_pair"], how="left",
    )
    .with_columns(pl.col("acoustic_significant").fill_null(False))
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
)
n_as_qualified = int(b4_qualified["acoustic_significant"].sum())
print(f"B4 qualifying cells (n_qualifying ≥ 1, n_per_class ≥ {K}): "
      f"{b4_qualified.height}  (acoustic_significant: {n_as_qualified})")

# %% [markdown]
# ## Bootstrap loop
#
# Raw contrast only — no acoustic-aligned polarity (out of scope; see module
# docstring). `mean_diff_raw_null` is a within-step label-permutation null on
# the SAME bootstrap trial draw (kept for parity with `t_tests.py`'s row
# shape; not consumed by any correction here).
#
# This is a trimmed copy of `t_tests.py`'s `bootstrap_cell` (same RNG call
# order on `per_step`/`select_cell_trials_bootstrap`/label-permutation
# shuffle, so an AS cell's draws are bit-identical between the two files —
# minus the acoustic-aligned/`preferred`-class block, out of scope here).
# Deliberately not extracted into a shared helper: `t_tests.py` is the frozen
# AS-restricted pipeline this fork must not touch. `t_tests_all_sr_reconciliation.py`
# is the sanctioned drift detector — if this copy and `t_tests.py`'s diverge
# on the shared (raw) computation, that notebook's blocking gate fails loudly
# the next time both pipelines run. Keep this in sync by inspection; don't
# rely on "it compiles" as evidence it still matches.

# %%
def bootstrap_cell(
    *,
    md_pp,
    hga,
    bhv_col: str,
    word_end: str,
    qualifying_steps: list[int],
    behav_smin: int,
    behav_smax: int,
    R: int,
    base_seed: int = 0,
) -> tuple[list[dict], int]:
    """Run R bootstrap replicates of the cell's raw mean-diff searchlight.

    Returns (rows_per_replicate_window, n_per_class). Each row carries
    mean_diff_raw and mean_diff_raw_null (label-permutation null from the
    same trial draw, preserving per-step class balance).
    """
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
        # raw: pos = class==1 (or larger key), neg = class==0 (or smaller key)
        raw_pos_key, raw_neg_key = keys[1], keys[0]
        res = searchlight_mean_diff(
            hga, draws[raw_pos_key], draws[raw_neg_key],
            search_smin=behav_smin, search_smax=behav_smax,
            window_size=WINDOW_SIZE, stride=STRIDE,
        )
        # Label-permutation null: within each step, pool that step's pos+neg
        # draws and randomly re-split equally (preserves per-step class balance).
        _step_sizes = [
            min(len(v) for v in by_cls.values())
            for by_cls in per_step.values()
            if min(len(v) for v in by_cls.values()) > 0
        ]
        _null_pos_parts: list[np.ndarray] = []
        _null_neg_parts: list[np.ndarray] = []
        _off = 0
        for _ns in _step_sizes:
            _step_pool = np.concatenate([
                draws[raw_pos_key][_off:_off + _ns],
                draws[raw_neg_key][_off:_off + _ns],
            ])
            rng.shuffle(_step_pool)
            _null_pos_parts.append(_step_pool[:_ns])
            _null_neg_parts.append(_step_pool[_ns:])
            _off += _ns
        null_pos = np.concatenate(_null_pos_parts)
        null_neg = np.concatenate(_null_neg_parts)
        res_null = searchlight_mean_diff(
            hga, null_pos, null_neg,
            search_smin=behav_smin, search_smax=behav_smax,
            window_size=WINDOW_SIZE, stride=STRIDE,
        )
        null_diff_by_window = {(w.smin, w.smax): w.mean_diff for w in res_null}
        for w in res:
            rows.append({
                "replicate": r,
                "smin": w.smin, "smax": w.smax,
                "tmin": w.smin / epoch_sfreq + epoch_tmin,
                "tmax": w.smax / epoch_sfreq + epoch_tmin,
                "mean_pos_raw": w.mean_pos,
                "mean_neg_raw": w.mean_neg,
                "mean_diff_raw": w.mean_diff,
                "mean_diff_raw_null": null_diff_by_window.get((w.smin, w.smax), float("nan")),
                "n_per_class": n_per_class,
            })
    return rows, n_per_class


# %% [markdown]
# ## Run B4 cells

# %%
b4_boot_rows: list[dict] = []
b4_cell_manifest: list[dict] = []
b4_failures: list[dict] = []

for row in tqdm(b4_qualified.iter_rows(named=True),
                total=b4_qualified.height, desc="B4 all-SR bootstrap"):
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
    behav_smin, behav_smax = behav_search_range(row["phoneme_pair"])
    steps = [int(s) for s in row["qualifying_steps"]]
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
    }
    if behav_smax - behav_smin < WINDOW_SIZE:
        b4_cell_manifest.append({
            **manifest_base,
            "status": "search_range_too_narrow",
            "behav_smin": behav_smin, "behav_smax": behav_smax,
        })
        continue
    try:
        rows, n_per_class = bootstrap_cell(
            md_pp=md_pp, hga=hga, bhv_col=bhv_col,
            word_end=row["word_end"], qualifying_steps=steps,
            behav_smin=behav_smin, behav_smax=behav_smax,
            R=R,
        )
        for r in rows:
            b4_boot_rows.append({
                "subject": subj,
                "electrode_idx": int(row["electrode_idx"]),
                "phoneme_pair": row["phoneme_pair"],
                "word_end": row["word_end"],
                "qualifying_steps": ",".join(str(s) for s in steps),
                "n_qualifying_steps": len(steps),
                "acoustic_significant": bool(row["acoustic_significant"]),
                **r,
            })
        b4_cell_manifest.append({
            **manifest_base,
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

# Underpowered candidates (n_qualifying < 1 or n_per_class < K) — for
# completeness in the manifest, ALL SR cells, not just AS ones.
b4_drops = (
    trial_balance
    .filter(pl.col("is_ambiguous_step"))
    .join(
        site_universe.select([
            "subject", "electrode_idx", "phoneme_pair",
            "acoustic_significant", "phon_smin", "phon_smax",
        ]),
        on=["subject", "electrode_idx", "phoneme_pair"], how="inner",
    )
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.len().alias("n_ambig_steps"),
        pl.col("min_class").sum().alias("n_per_class_pool"),
        pl.col("acoustic_significant").first().alias("acoustic_significant"),
        pl.col("phon_smin").first().alias("phon_smin"),
        pl.col("phon_smax").first().alias("phon_smax"),
    )
    .filter((pl.col("n_ambig_steps") < 1) | (pl.col("n_per_class_pool") < K))
)
for row in b4_drops.iter_rows(named=True):
    b4_cell_manifest.append({
        "subject": row["subject"],
        "electrode_idx": int(row["electrode_idx"]),
        "phoneme_pair": row["phoneme_pair"],
        "word_end": row["word_end"],
        "qualifying_steps": "",
        "n_per_class": int(row["n_per_class_pool"]),
        "acoustic_significant": bool(row["acoustic_significant"]),
        "phon_smin": row["phon_smin"],
        "phon_smax": row["phon_smax"],
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
# ## Per-window aggregation (CI + empirical p) — raw only

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
            pl.col("n_per_class").first().alias("n_per_class"),
            pl.col("acoustic_significant").first().alias("acoustic_significant"),
            pl.col("replicate").max().alias("R_replicates"),
        )
    )
    grouped = grouped.with_columns([
        pl.min_horizontal(
            2 * pl.min_horizontal("frac_raw_le0", "frac_raw_ge0"),
            pl.lit(1.0),
        ).alias("emp_p_raw"),
        ((pl.col("mean_diff_raw_ci_lo") > 0) | (pl.col("mean_diff_raw_ci_hi") < 0))
            .alias("ci_raw_excludes_zero"),
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
# Best window = window with largest |median(mean_diff_raw)|. (For an AS cell,
# this is identical to ranking by |aligned| in `t_tests.py` — aligned is a
# constant per-cell sign flip of raw, so the argmax over windows agrees; see
# `t_tests_all_sr_reconciliation.py`.)

# %%
def per_cell_best(per_window: pl.DataFrame, cell_keys: list[str]) -> pl.DataFrame:
    if per_window.height == 0:
        return pl.DataFrame()
    return (
        per_window
        .with_columns(pl.col("mean_diff_raw_med").abs().alias("__rank"))
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
            "emp_p_raw": "best_emp_p_raw",
            "ci_raw_excludes_zero": "best_ci_raw_excludes_zero",
        })
    )


b4_per_cell = per_cell_best(b4_per_window, b4_cell_keys)
if b4_per_cell.height:
    b4_per_cell.write_parquet(OUT_DIR / "b4_per_cell.parquet")
print(f"B4 per_cell rows: {b4_per_cell.height}")
n_sig = int(b4_per_cell["best_ci_raw_excludes_zero"].fill_null(False).sum()) if b4_per_cell.height else 0
print(f"per-cell significance — best-window CI excludes zero: {n_sig}/{b4_per_cell.height}")

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
# Per (window × cut): fraction of cells with `ci_raw_excludes_zero`, median
# signed/|abs| raw effect. `acoustic_significant` is an explicit cut — this
# is the number the fork exists to produce: does the non-acoustic-significant
# slice show perceptual selectivity too?

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
        ("acoustic_significant", ["acoustic_significant"]),
        ("phoneme_pair_x_roi", ["phoneme_pair", "roi"]),
    ]:
        agg = (
            pc
            .group_by(["mode", "smin", "smax", "tmin", "tmax"] + group_cols)
            .agg(
                pl.len().alias("n_cells"),
                pl.col("ci_raw_excludes_zero").cast(pl.Float64).sum().alias("n_ci_raw_sig"),
                pl.col("mean_diff_raw_med").median().alias("median_signed_raw"),
                pl.col("mean_diff_raw_med").abs().median().alias("median_abs_raw"),
                pl.col("mean_diff_raw_med").abs().quantile(0.25).alias("q25_abs_raw"),
                pl.col("mean_diff_raw_med").abs().quantile(0.75).alias("q75_abs_raw"),
            )
            .with_columns([
                (pl.col("n_ci_raw_sig") / pl.col("n_cells")).alias("frac_ci_raw"),
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
def _peak_frac(pop: pl.DataFrame, cut: str, sub_key=None, sub_val=None) -> float | None:
    if pop.height == 0:
        return None
    sub = pop.filter((pl.col("mode") == "b4") & (pl.col("cut") == cut))
    if sub_key is not None:
        sub = sub.filter(pl.col(sub_key) == sub_val)
    if sub.height == 0:
        return None
    return float(sub["frac_ci_raw"].max())


pdf_path = OUT_DIR / "population_summary.pdf"
with PdfPages(pdf_path) as pdf:
    # Title
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.75, "All-speech-responsive within-completion contrast",
            ha="center", va="center", fontsize=16)
    n_ok = cell_manifest.filter(pl.col("status") == "ok").height
    n_as_ok = int(b4_per_cell["acoustic_significant"].sum()) if b4_per_cell.height else 0
    ax.text(0.5, 0.55,
            f"K = {K}   ·   R = {R}   ·   CI = {CI_LOW}–{CI_HIGH}%\n"
            f"cells (ok): {n_ok}   (acoustic_significant: {n_as_ok})\n\n"
            f"best-window CI-excludes-0:  {n_sig} / {n_ok}\n\n"
            f"(raw, uncorrected bootstrap CI — matches how t_tests.py's output\n"
            f"is used everywhere else downstream)",
            ha="center", va="center", fontsize=10)
    pdf.savefig(fig); plt.close(fig)

    # Per-time frac-CI-excludes-0 curve, split by acoustic_significant
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for as_val, color, label in ((True, "#2166ac", "acoustic_significant"),
                                  (False, "#b2182b", "not acoustic_significant")):
        sub = (
            population
            .filter((pl.col("mode") == "b4") & (pl.col("cut") == "acoustic_significant")
                     & (pl.col("acoustic_significant") == as_val))
            .sort("tmin")
        )
        if sub.height == 0:
            continue
        ax.plot(sub["tmin"].to_numpy(), sub["frac_ci_raw"].to_numpy(),
                marker="o", color=color, lw=1.4, label=f"{label} (n={int(sub['n_cells'][0])})")
    ax.set_xlabel("Window start (s, post word onset)")
    ax.set_ylabel("fraction of cells, CI excludes 0")
    ax.set_ylim(0, 1.02)
    ax.set_title("B4 — per-window fraction significant, by acoustic_significant")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # Sites x time heatmap of median signed raw mean_diff
    if b4_per_window.height:
        pivot = (
            b4_per_window
            .pivot(values="mean_diff_raw_med", index=b4_cell_keys, on="tmin",
                   aggregate_function="first")
            .sort(b4_cell_keys)
        )
        time_cols_sorted = sorted([c for c in pivot.columns if c not in b4_cell_keys],
                                  key=lambda c: float(c))
        mat = pivot.select(time_cols_sorted).to_numpy().astype(float)
        if mat.size:
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
            ax.set_ylabel("Cell (sorted by peak |raw mean_diff|)")
            ax.set_title(f"B4 — median signed raw mean_diff (R={R})  n_cells={mat.shape[0]}")
            cb = fig.colorbar(im, ax=ax, shrink=0.6); cb.set_label("raw mean_diff (HGA)")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

    # Effect-size distribution: best-window raw medians
    if b4_per_cell.height:
        fig, ax = plt.subplots(figsize=(6, 4))
        signed = b4_per_cell["best_mean_diff_raw_med"].drop_nulls().to_numpy()
        ax.hist(signed, bins=30, color="#4393c3", edgecolor="k", alpha=0.7)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("best-window median signed raw mean_diff")
        ax.set_ylabel("# cells")
        med = float(np.nanmedian(signed)) if len(signed) else float("nan")
        ax.set_title(f"B4 — n={len(signed)}, median={med:.2f}")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    # Best-window timing scatter, colored by ci_raw_excludes_zero
    if b4_per_cell.height:
        fig, ax = plt.subplots(figsize=(7, 4))
        tmin_arr = b4_per_cell["best_tmin"].to_numpy().astype(float)
        sig_mask = b4_per_cell["best_ci_raw_excludes_zero"].fill_null(False).to_numpy().astype(bool)
        rng_jit = np.random.default_rng(42)
        y_jit = rng_jit.uniform(-0.4, 0.4, size=len(tmin_arr))
        ax.scatter(tmin_arr[~sig_mask], y_jit[~sig_mask],
                   color="#d9d9d9", s=22, alpha=0.8, linewidths=0.4,
                   edgecolors="k", label="CI includes 0", zorder=2)
        ax.scatter(tmin_arr[sig_mask], y_jit[sig_mask],
                   color="#2166ac", s=30, alpha=0.85, linewidths=0.4,
                   edgecolors="k", label="CI excludes 0", zorder=3)
        n_sig_scatter = int(sig_mask.sum())
        n_tot = len(sig_mask)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_xlabel("Best-window tmin (s, post word onset)")
        ax.set_yticks([])
        ax.set_title(f"B4 — {n_sig_scatter}/{n_tot} significant ({100*n_sig_scatter/max(n_tot,1):.0f}%)")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    # phoneme_pair / roi / acoustic_significant breakdowns
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, cut in zip(axes, ("phoneme_pair", "roi", "acoustic_significant")):
        sub = population.filter((pl.col("mode") == "b4") & (pl.col("cut") == cut))
        if sub.height == 0:
            ax.text(0.5, 0.5, f"{cut}: empty", ha="center", va="center"); ax.axis("off"); continue
        peak = (
            sub.group_by(cut)
               .agg(pl.col("frac_ci_raw").max().alias("frac_peak"),
                    pl.col("n_cells").max().alias("n_peak"))
               .sort("frac_peak", descending=True)
        )
        labels = [str(v) for v in peak[cut].to_list()]
        vals = peak["frac_peak"].to_list()
        ns = peak["n_peak"].to_list()
        ax.barh(labels, vals, color="#4393c3", edgecolor="k")
        for y, (v, n) in enumerate(zip(vals, ns)):
            ax.text(v + 0.01, y, f"{v:.0%} (n={int(n)})", va="center", fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("peak-window frac CI excludes 0")
        ax.set_title(f"B4 — {cut}")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)

print(f"wrote {pdf_path}")

# %% [markdown]
# ## Done

# %%
print("=" * 70)
print(f"K = {K}   R = {R}   CI = {CI_LOW}–{CI_HIGH}%")
print(f"all-SR sites: {site_universe.height}  "
      f"(acoustic_significant: {int(site_universe['acoustic_significant'].sum())})")
print(f"B4 cells (ok): {cell_manifest.filter(pl.col('status') == 'ok').height}")
if b4_per_cell.height:
    print(f"best-window CI-excludes-0: {n_sig}/{b4_per_cell.height}")
print(f"See {pdf_path}")
print("=" * 70)
