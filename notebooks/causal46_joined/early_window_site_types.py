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
# # Early-window site-type classification (per subject)
#
# Classifies each (electrode × phoneme_pair) site present in the manual
# annotation manifest into five types based on whether, within the acoustic
# decoder's peak-selection search range (the early window), it shows:
#
# - **Type 1** (`type1_acoustic_only`): clean acoustic response, no behavioral
#   split in early window.
# - **Type 2** (`type2_early_perceptual`): acoustic response + aligned behavioral
#   split at BOTH word-ends.
# - **Type 3** (`type3_asymmetric`): acoustic response + aligned behavioral split
#   at exactly ONE word-end; the other is fully null.
# - **Grab-bag** (`grab_bag`): any anti-aligned behavioral window in early range.
# - **Complex** (`complex`): multi-peak / complex acoustic tuning in manifest.
#
# Three bootstraps are run per site over the same acoustic search range
# `[word_onset, min(max_word_end_offset, 1.3 s)]`:
#   - **A**: endpoint-balanced acoustic bootstrap (steps 1 vs 6, pooled WEs)
#   - **B₁**: within-completion behavioral bootstrap for WE0 (e.g. "desolate")
#   - **B₂**: within-completion behavioral bootstrap for WE1 (e.g. "necessary")
#
# Alignment sign (`acoustic_sign`) is derived from the A bootstrap's best window
# — the window maximising |median(mean_diff_raw_A)| among A-significant windows.
#
# Outputs (per subject):
#   - `A_per_window.parquet`      — A bootstrap per (site × window)
#   - `B_per_window.parquet`      — B bootstrap per (cell × window)
#   - `site_type_assignments.parquet` — one row per (subject × electrode × phoneme_pair)

# %%
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

# Thread-count limits — set before any BLAS import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

from src.data import add_metadata_features
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    bootstrap_A_site,
    early_window_star_plot,
    extract_hga,
    n_per_class_from_per_step,
    per_step_class_counts,
    resolve_behavior_col,
    searchlight_mean_diff,
    select_cell_trials_bootstrap,
)

# %% tags=["parameters"]
subject = "EC250"
manifest_path = "outputs/causal46_joined/filtered_manifest.csv"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
epoch_dir = "outputs/epochs_preprocessed"
trial_balance_path = "outputs/causal46_joined/trial_balance_index.csv"
outdir = "outputs/causal46_joined/early_window_site_types/EC250"
min_class_k = 4
window_size = 10
stride = 10
R = 1000
# Acoustic decoder peak-selection search range — used as B bootstrap search range.
# Matches acoustic_decoding_peaks.py so B only evaluates the same early window
# where the acoustic response was originally found.
ac_search_smin = 45
ac_search_smax = 68

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = window_size
STRIDE = stride
K = min_class_k
R = int(R)
CI_LOW, CI_HIGH = 2.5, 97.5
# B bootstrap searches only this acoustic window (not the full word).
B_SEARCH_SMIN = int(ac_search_smin)
B_SEARCH_SMAX = int(ac_search_smax)

# Acoustic search range (matches acoustic_decoding_peaks.py peak-selection bounds)
SMIN_LO = int((0.0 - epoch_tmin) * epoch_sfreq)          # word onset = sample 40
ALL_RESPONSE_TMAX_MAX = 1.3
SMAX_HI_ABS = int((ALL_RESPONSE_TMAX_MAX - epoch_tmin) * epoch_sfreq)  # sample 170


def max_we_offset_sample(pair: str) -> int:
    return max(
        int(round((OFFSET_DICT[we] - epoch_tmin) * epoch_sfreq))
        for we in PHONEME_PAIR_TO_WORD_ENDS[pair]
    )


PAIR_SMAX_HI: dict[str, int] = {
    pp: min(max_we_offset_sample(pp), SMAX_HI_ABS)
    for pp in PHONEME_PAIR_TO_WORD_ENDS
}
PAIR_WE: dict[str, list[str]] = dict(PHONEME_PAIR_TO_WORD_ENDS)  # {pair: [we0, we1]}

print(f"subject={subject}  K={K}  R={R}  window={WINDOW_SIZE}  stride={STRIDE}")
print(f"Acoustic search range: smin_lo={SMIN_LO}")
print(f"Per-pair smax_hi: {PAIR_SMAX_HI}")

# %% [markdown]
# ## Load inputs

# %%
manifest_all = pl.read_csv(manifest_path)
if "subject" in manifest_all.columns:
    manifest = manifest_all.filter(pl.col("subject") == subject)
else:
    raise ValueError("filtered_manifest.csv must have a 'subject' column")
print(f"Manifest rows for {subject}: {manifest.height}")

trial_balance_all = pl.read_csv(trial_balance_path)
trial_balance = trial_balance_all.filter(pl.col("subject") == subject)
print(f"trial_balance rows for {subject}: {trial_balance.height}")

# %%
epochs_path = Path(epoch_dir) / f"{subject}_epo.fif"
ep = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
md_raw = ep.metadata
md = add_metadata_features(md_raw).reset_index(drop=True)
ep.metadata = md
bhv_col = resolve_behavior_col(md)
print(f"Loaded {len(ep)} epochs; behavior col: {bhv_col}")

# %% [markdown]
# ## Helper — per-window aggregate (A and B)

# %%
def agg_A_per_window(rows: list[dict]) -> pl.DataFrame:
    """Aggregate A bootstrap rows to per-window CI summary."""
    if not rows:
        return pl.DataFrame()
    boot = pl.DataFrame(rows)
    return (
        boot
        .group_by(["smin", "smax", "tmin", "tmax"])
        .agg(
            pl.col("mean_diff_raw").median().alias("mean_diff_raw_med"),
            pl.col("mean_diff_raw").quantile(CI_LOW / 100).alias("ci_lo"),
            pl.col("mean_diff_raw").quantile(CI_HIGH / 100).alias("ci_hi"),
            (pl.col("mean_diff_raw") <= 0).cast(pl.Float64).mean().alias("frac_le0"),
            (pl.col("mean_diff_raw") >= 0).cast(pl.Float64).mean().alias("frac_ge0"),
            pl.col("n_per_class").first(),
        )
        .with_columns([
            pl.min_horizontal(
                2 * pl.min_horizontal("frac_le0", "frac_ge0"), pl.lit(1.0)
            ).alias("emp_p"),
            ((pl.col("ci_lo") > 0) | (pl.col("ci_hi") < 0)).alias("ci_excludes_zero"),
        ])
        .drop(["frac_le0", "frac_ge0"])
        .sort("smin")
    )


def agg_B_per_window(rows: list[dict]) -> pl.DataFrame:
    """Aggregate B bootstrap rows to per-window CI summary."""
    if not rows:
        return pl.DataFrame()
    boot = pl.DataFrame(rows)
    return (
        boot
        .group_by(["smin", "smax", "tmin", "tmax"])
        .agg(
            pl.col("mean_diff_raw").median().alias("mean_diff_raw_med"),
            pl.col("mean_diff_raw").quantile(CI_LOW / 100).alias("ci_raw_lo"),
            pl.col("mean_diff_raw").quantile(CI_HIGH / 100).alias("ci_raw_hi"),
            pl.col("mean_diff_aligned").median().alias("mean_diff_aligned_med"),
            pl.col("mean_diff_aligned").quantile(CI_LOW / 100).alias("ci_aligned_lo"),
            pl.col("mean_diff_aligned").quantile(CI_HIGH / 100).alias("ci_aligned_hi"),
            (pl.col("mean_diff_aligned") <= 0).cast(pl.Float64).mean().alias("frac_aligned_le0"),
            (pl.col("mean_diff_aligned") >= 0).cast(pl.Float64).mean().alias("frac_aligned_ge0"),
            pl.col("n_per_class").first(),
        )
        .with_columns([
            pl.min_horizontal(
                2 * pl.min_horizontal("frac_aligned_le0", "frac_aligned_ge0"),
                pl.lit(1.0),
            ).alias("emp_p_aligned"),
            ((pl.col("ci_aligned_lo") > 0) | (pl.col("ci_aligned_hi") < 0))
                .alias("ci_aligned_excludes_zero"),
        ])
        .drop(["frac_aligned_le0", "frac_aligned_ge0"])
        .sort("smin")
    )

# %% [markdown]
# ## Helper — B bootstrap cell

# %%
def bootstrap_B_cell(
    *,
    md_pp: pd.DataFrame,
    hga: np.ndarray,
    word_end: str,
    qualifying_steps: list[int],
    acoustic_sign: float,
    search_smin: int,
    search_smax: int,
    R: int,
    base_seed: int = 0,
) -> tuple[list[dict], int]:
    """Run R bootstrap replicates of the B behavioral searchlight.

    Returns (rows, n_per_class).
    mean_diff_aligned = acoustic_sign × mean_diff_raw
    where mean_diff_raw = mean(HGA[class=1]) − mean(HGA[class=0]).
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
        raw_pos_key, raw_neg_key = keys[1], keys[0]  # class=1 is pos
        windows = searchlight_mean_diff(
            hga, draws[raw_pos_key], draws[raw_neg_key],
            search_smin=search_smin, search_smax=search_smax,
            window_size=WINDOW_SIZE, stride=STRIDE,
        )
        for w in windows:
            mean_diff_raw = w.mean_diff
            if np.isnan(acoustic_sign):
                mean_diff_aligned = float("nan")
            else:
                mean_diff_aligned = float(acoustic_sign) * mean_diff_raw
            rows.append({
                "replicate": r,
                "smin": w.smin, "smax": w.smax,
                "tmin": w.smin / epoch_sfreq + epoch_tmin,
                "tmax": w.smax / epoch_sfreq + epoch_tmin,
                "mean_diff_raw": mean_diff_raw,
                "mean_diff_aligned": mean_diff_aligned,
                "n_per_class": n_per_class,
            })
    return rows, n_per_class

# %% [markdown]
# ## Helper — B cell qualification from trial balance

# %%
def get_b_cell_qualification(
    electrode_idx: int,
    pair: str,
    word_end: str,
) -> tuple[list[int], int] | None:
    """Return (qualifying_steps, n_per_class) from trial_balance, or None if underpowered."""
    cell = (
        trial_balance
        .filter(
            (pl.col("electrode_idx") == electrode_idx)
            & (pl.col("phoneme_pair") == pair)
            & (pl.col("word_end") == word_end)
            & pl.col("is_ambiguous_step")
        )
    )
    if cell.height == 0:
        return None
    n_per_class = int(cell["min_class"].sum())
    if n_per_class < K:
        return None
    steps = sorted(int(float(s)) for s in cell["resampled"].to_list())
    return steps, n_per_class

# %% [markdown]
# ## Helper — site-type assignment

# %%
_COMPLEX_TUNING_VALUES = {"both", "complex", "two peaks"}


def assign_site_type(
    *,
    manifest_tuning: str,
    A_significant: bool,
    acoustic_sign: float,
    B1_aligned_sig: bool,
    B1_anti_sig: bool,
    B2_aligned_sig: bool,
    B2_anti_sig: bool,
    B1_powered: bool,
    B2_powered: bool,
) -> tuple[str, str]:
    """Return (site_type, status) per the plan's decision tree."""
    tuning_clean = str(manifest_tuning).strip().lower() if manifest_tuning else ""

    # Type C: complex acoustic tuning
    if tuning_clean in _COMPLEX_TUNING_VALUES:
        return "complex", "ok"

    # A not significant (or endpoint-underpowered)
    if not A_significant or not np.isfinite(acoustic_sign):
        return "A_unsigned", "ok"

    # B power check — requires both cells to be classified
    if not B1_powered or not B2_powered:
        return "unknown", "unclassifiable_B_power"

    # Classify by aligned/anti-aligned pattern
    b1_any = B1_aligned_sig or B1_anti_sig
    b2_any = B2_aligned_sig or B2_anti_sig

    if not b1_any and not b2_any:
        return "type1_acoustic_only", "ok"

    if B1_aligned_sig and B2_aligned_sig and not B1_anti_sig and not B2_anti_sig:
        return "type2_early_perceptual", "ok"

    if (B1_aligned_sig and not B2_aligned_sig and not B2_anti_sig and not B1_anti_sig):
        return "type3_asymmetric", "ok"
    if (B2_aligned_sig and not B1_aligned_sig and not B1_anti_sig and not B2_anti_sig):
        return "type3_asymmetric", "ok"

    return "grab_bag", "ok"

# %% [markdown]
# ## Build site pool from manifest

# %%
# Group manifest by (electrode_idx, phoneme_pair) to get unique sites.
# Verify acoustic_tuning consistency across word_end rows.
_SITE_KEYS = ["electrode_idx", "phoneme_pair"]

sites: list[dict] = []
for group_df in manifest.partition_by(_SITE_KEYS, maintain_order=True):
    ei = int(group_df["electrode_idx"][0])
    pp = str(group_df["phoneme_pair"][0])

    # Collect acoustic_tuning values across both WE rows
    if "acoustic tuning" in group_df.columns:
        tuning_vals = (
            group_df["acoustic tuning"]
            .drop_nulls()
            .cast(pl.Utf8)
            .to_list()
        )
        tuning_vals = [v.strip() for v in tuning_vals if v.strip()]
    else:
        tuning_vals = []

    if not tuning_vals:
        manifest_tuning = ""
        tuning_conflict = False
    elif len(set(tuning_vals)) == 1:
        manifest_tuning = tuning_vals[0]
        tuning_conflict = False
    else:
        # Conflicting acoustic_tuning across WE rows → conservatively treat as complex
        manifest_tuning = "complex"
        tuning_conflict = True
        print(f"  TUNING CONFLICT: {subject} e{ei} {pp} — {set(tuning_vals)!r}; "
              "treating as complex")

    sites.append({
        "electrode_idx": ei,
        "phoneme_pair": pp,
        "manifest_tuning": manifest_tuning,
        "tuning_conflict": tuning_conflict,
    })

print(f"Unique sites (electrode × pair) for {subject}: {len(sites)}")

# %% [markdown]
# ## Main bootstrap loop

# %%
A_window_rows: list[dict] = []
B_window_rows: list[dict] = []
site_type_rows: list[dict] = []
failures: list[dict] = []

for site in tqdm(sites, desc="early_window sites"):
    ei = site["electrode_idx"]
    pp = site["phoneme_pair"]
    manifest_tuning = site["manifest_tuning"]
    search_smin = SMIN_LO
    search_smax = PAIR_SMAX_HI.get(pp, SMAX_HI_ABS)

    if search_smax - search_smin < WINDOW_SIZE:
        site_type_rows.append({
            "subject": subject, "electrode_idx": ei, "phoneme_pair": pp,
            "manifest_tuning": manifest_tuning, "tuning_conflict": site["tuning_conflict"],
            "acoustic_sign": float("nan"), "A_significant": False,
            "B1_aligned_sig": False, "B1_anti_sig": False,
            "B2_aligned_sig": False, "B2_anti_sig": False,
            "B1_qualifying_steps": "", "B2_qualifying_steps": "",
            "B1_n_per_class": 0, "B2_n_per_class": 0,
            "site_type": "unknown", "status": "search_range_too_narrow",
        })
        continue

    try:
        # ------------------------------------------------------------------
        # Epoch subset for this phoneme pair
        # ------------------------------------------------------------------
        pp_mask = (md["phoneme_pair"] == pp).values
        ep_pp = ep[pp_mask]
        md_pp = md[pp_mask].reset_index(drop=True)
        hga = extract_hga(ep_pp, ei)

        # ------------------------------------------------------------------
        # A bootstrap (endpoint-balanced, all word_ends pooled)
        # ------------------------------------------------------------------
        a_result = bootstrap_A_site(
            hga, md_pp,
            search_smin=search_smin, search_smax=search_smax,
            window_size=WINDOW_SIZE, stride=STRIDE,
            R=R, min_n=K,
        )

        if a_result is None:
            # Endpoints underpowered
            n_lo, n_hi = (
                int((md_pp["resampled"] == 1).sum()),
                int((md_pp["resampled"] == 6).sum()),
            )
            site_type_rows.append({
                "subject": subject, "electrode_idx": ei, "phoneme_pair": pp,
                "manifest_tuning": manifest_tuning, "tuning_conflict": site["tuning_conflict"],
                "acoustic_sign": float("nan"), "A_significant": False,
                "B1_aligned_sig": False, "B1_anti_sig": False,
                "B2_aligned_sig": False, "B2_anti_sig": False,
                "B1_qualifying_steps": "", "B2_qualifying_steps": "",
                "B1_n_per_class": 0, "B2_n_per_class": 0,
                "site_type": "A_unsigned", "status": "A_endpoint_underpowered",
                "A_n_step1": n_lo, "A_n_step6": n_hi,
            })
            continue

        a_rows, n_lo, n_hi = a_result

        # Per-window A summary
        a_pw = agg_A_per_window(a_rows)
        a_pw = a_pw.with_columns([
            pl.lit(subject).alias("subject"),
            pl.lit(ei).alias("electrode_idx"),
            pl.lit(pp).alias("phoneme_pair"),
            pl.lit(n_lo).alias("n_step1"),
            pl.lit(n_hi).alias("n_step6"),
        ])
        A_window_rows.append(a_pw)

        # Acoustic sign from best significant A window
        a_sig = a_pw.filter(pl.col("ci_excludes_zero"))
        A_significant = a_sig.height > 0
        if A_significant:
            best_a = (
                a_sig
                .with_columns(pl.col("mean_diff_raw_med").abs().alias("__abs"))
                .sort("__abs", descending=True)
                .head(1)
            )
            acoustic_sign = float(np.sign(float(best_a["mean_diff_raw_med"][0])))
        else:
            acoustic_sign = float("nan")

        # ------------------------------------------------------------------
        # B₁ and B₂ bootstraps — restricted to acoustic peak search window
        # B_SEARCH_SMIN/SMAX matches acoustic_decoding_peaks.py so we only
        # evaluate the early acoustic period, not late lexical effects.
        # ------------------------------------------------------------------
        word_ends = PAIR_WE.get(pp, [])
        we_results: dict[str, dict] = {}

        for we in word_ends:
            b_qual = get_b_cell_qualification(ei, pp, we)
            if b_qual is None:
                we_results[we] = {
                    "powered": False, "n_per_class": 0, "qualifying_steps": [],
                    "aligned_sig": False, "anti_sig": False,
                }
                continue

            q_steps, n_per_class = b_qual

            b_rows, _ = bootstrap_B_cell(
                md_pp=md_pp, hga=hga,
                word_end=we,
                qualifying_steps=q_steps,
                acoustic_sign=acoustic_sign,
                search_smin=B_SEARCH_SMIN, search_smax=B_SEARCH_SMAX,
                R=R,
            )

            b_pw = agg_B_per_window(b_rows)

            if b_pw.height > 0:
                # n_per_step_per_class: {str(step): min_class_at_step}
                per_step = per_step_class_counts(
                    md_pp, word_end=we, qualifying_steps=q_steps, group_col=bhv_col,
                )
                n_per_step = {
                    str(s): int(min(len(v) for v in by_cls.values()))
                    for s, by_cls in per_step.items()
                }
                b_pw = b_pw.with_columns([
                    pl.lit(subject).alias("subject"),
                    pl.lit(ei).alias("electrode_idx"),
                    pl.lit(pp).alias("phoneme_pair"),
                    pl.lit(we).alias("word_end"),
                    pl.lit(",".join(str(s) for s in q_steps)).alias("qualifying_steps"),
                    pl.lit(json.dumps(n_per_step)).alias("n_per_step_per_class"),
                    pl.lit(n_per_class).alias("n_per_class_cell"),
                    pl.lit(acoustic_sign).alias("acoustic_sign"),
                ])
                B_window_rows.append(b_pw)

                # B present / anti-present: any window with aligned CI excluding zero
                b_aligned_sig = bool(
                    b_pw.filter(
                        pl.col("ci_aligned_excludes_zero")
                        & (pl.col("mean_diff_aligned_med") > 0)
                    ).height > 0
                )
                b_anti_sig = bool(
                    b_pw.filter(
                        pl.col("ci_aligned_excludes_zero")
                        & (pl.col("mean_diff_aligned_med") < 0)
                    ).height > 0
                )
            else:
                b_aligned_sig = False
                b_anti_sig = False

            we_results[we] = {
                "powered": True, "n_per_class": n_per_class,
                "qualifying_steps": q_steps,
                "aligned_sig": b_aligned_sig, "anti_sig": b_anti_sig,
            }

        # ------------------------------------------------------------------
        # Site-type assignment
        # ------------------------------------------------------------------
        we0, we1 = (word_ends + ["", ""])[:2]
        r0 = we_results.get(we0, {"powered": False, "n_per_class": 0,
                                   "qualifying_steps": [], "aligned_sig": False,
                                   "anti_sig": False})
        r1 = we_results.get(we1, {"powered": False, "n_per_class": 0,
                                   "qualifying_steps": [], "aligned_sig": False,
                                   "anti_sig": False})

        site_type, status = assign_site_type(
            manifest_tuning=manifest_tuning,
            A_significant=A_significant,
            acoustic_sign=acoustic_sign,
            B1_aligned_sig=r0["aligned_sig"],
            B1_anti_sig=r0["anti_sig"],
            B2_aligned_sig=r1["aligned_sig"],
            B2_anti_sig=r1["anti_sig"],
            B1_powered=r0["powered"],
            B2_powered=r1["powered"],
        )

        site_type_rows.append({
            "subject": subject,
            "electrode_idx": ei,
            "phoneme_pair": pp,
            "manifest_tuning": manifest_tuning,
            "tuning_conflict": site["tuning_conflict"],
            "acoustic_sign": acoustic_sign,
            "A_significant": A_significant,
            "B1_word_end": we0,
            "B2_word_end": we1,
            "B1_aligned_sig": r0["aligned_sig"],
            "B1_anti_sig": r0["anti_sig"],
            "B2_aligned_sig": r1["aligned_sig"],
            "B2_anti_sig": r1["anti_sig"],
            "B1_qualifying_steps": ",".join(str(s) for s in r0["qualifying_steps"]),
            "B2_qualifying_steps": ",".join(str(s) for s in r1["qualifying_steps"]),
            "B1_n_per_class": r0["n_per_class"],
            "B2_n_per_class": r1["n_per_class"],
            "A_n_step1": n_lo,
            "A_n_step6": n_hi,
            "site_type": site_type,
            "status": status,
        })

    except Exception as exc:
        tb = traceback.format_exc()
        failures.append({
            "subject": subject, "electrode_idx": ei, "phoneme_pair": pp,
            "error": repr(exc), "traceback": tb,
        })
        print(f"FAILED: {subject} e{ei} {pp}\n{tb}")

print(f"Sites processed: {len(site_type_rows)}  failures: {len(failures)}")

# %% [markdown]
# ## Save outputs

# %%
# A_per_window.parquet
if A_window_rows:
    A_per_window = pl.concat(A_window_rows, how="diagonal_relaxed")
    # Reorder columns: keys first
    key_cols = ["subject", "electrode_idx", "phoneme_pair"]
    other_cols = [c for c in A_per_window.columns if c not in key_cols]
    A_per_window = A_per_window.select(key_cols + other_cols)
    A_per_window.write_parquet(OUT_DIR / "A_per_window.parquet")
    print(f"A_per_window: {A_per_window.height} rows")
else:
    A_per_window = pl.DataFrame()
    pl.DataFrame().write_parquet(OUT_DIR / "A_per_window.parquet")
    print("A_per_window: 0 rows (empty)")

# %%
# B_per_window.parquet
if B_window_rows:
    B_per_window = pl.concat(B_window_rows, how="diagonal_relaxed")
    key_cols = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
    other_cols = [c for c in B_per_window.columns if c not in key_cols]
    B_per_window = B_per_window.select(key_cols + other_cols)
    B_per_window.write_parquet(OUT_DIR / "B_per_window.parquet")
    print(f"B_per_window: {B_per_window.height} rows")
else:
    B_per_window = pl.DataFrame()
    pl.DataFrame().write_parquet(OUT_DIR / "B_per_window.parquet")
    print("B_per_window: 0 rows (empty)")

# %%
# site_type_assignments.parquet
site_type_df = pl.DataFrame(site_type_rows) if site_type_rows else pl.DataFrame()
site_type_df.write_parquet(OUT_DIR / "site_type_assignments.parquet")
print(f"site_type_assignments: {site_type_df.height} rows")
if site_type_df.height > 0 and "site_type" in site_type_df.columns:
    print(site_type_df.group_by("site_type").len().sort("site_type"))

# %% [markdown]
# ## Star plot gallery

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load phon_peaks for acoustic AUC annotation
_phon_peaks_all = pl.read_parquet(phon_peaks_path)
_phon_peaks_subj = _phon_peaks_all.filter(pl.col("subject") == subject)
phon_auc: dict[tuple, float] = {
    (int(r["electrode_idx"]), str(r["phoneme_pair"])): float(r["test_roc_auc"])
    for r in _phon_peaks_subj.iter_rows(named=True)
    if r.get("test_roc_auc") is not None
}
print(f"phon_peaks loaded: {_phon_peaks_subj.height} rows for {subject}  "
      f"({len(phon_auc)} auc entries)")

# %%
# Sort sites by type for PDF ordering
_SITE_TYPE_ORDER = [
    "type2_early_perceptual",
    "type3_asymmetric",
    "grab_bag",
    "type1_acoustic_only",
    "complex",
    "A_unsigned",
    "unknown",
]

def _site_sort_key(row: dict) -> int:
    t = row.get("site_type", "unknown") or "unknown"
    return _SITE_TYPE_ORDER.index(t) if t in _SITE_TYPE_ORDER else len(_SITE_TYPE_ORDER)


sorted_sites = sorted(site_type_df.iter_rows(named=True), key=_site_sort_key)

pdf_path_gallery = OUT_DIR / "star_plots_early.pdf"
_n_plotted = 0
_failures_plot: list[tuple] = []

with PdfPages(pdf_path_gallery) as _pdf:
    if not sorted_sites:
        _fig0, _ax0 = plt.subplots(figsize=(8.5, 4))
        _ax0.text(0.5, 0.5, f"{subject}: no sites in manifest",
                  ha="center", va="center", fontsize=14)
        _ax0.axis("off")
        _pdf.savefig(_fig0)
        plt.close(_fig0)
    for _row in tqdm(sorted_sites, desc="star plots"):
        _ei = int(_row["electrode_idx"])
        _pp = _row["phoneme_pair"]
        _we0 = _row.get("B1_word_end") or ""
        _we1 = _row.get("B2_word_end") or ""
        _b1_steps = [int(float(s)) for s in (_row["B1_qualifying_steps"] or "").split(",") if s]
        _b2_steps = [int(float(s)) for s in (_row["B2_qualifying_steps"] or "").split(",") if s]
        _a_sign = float(_row["acoustic_sign"]) if _row["acoustic_sign"] is not None else float("nan")

        _site_a_pw = (
            A_per_window.filter(
                (pl.col("electrode_idx") == _ei) & (pl.col("phoneme_pair") == _pp)
            ) if A_per_window.height > 0 else pl.DataFrame()
        )
        _site_b1_pw = (
            B_per_window.filter(
                (pl.col("electrode_idx") == _ei)
                & (pl.col("phoneme_pair") == _pp)
                & (pl.col("word_end") == _we0)
            ) if B_per_window.height > 0 and _we0 else pl.DataFrame()
        )
        _site_b2_pw = (
            B_per_window.filter(
                (pl.col("electrode_idx") == _ei)
                & (pl.col("phoneme_pair") == _pp)
                & (pl.col("word_end") == _we1)
            ) if B_per_window.height > 0 and _we1 else pl.DataFrame()
        )

        try:
            _fig = early_window_star_plot(
                subject, _ei, _pp,
                ep=ep,
                bhv_col=bhv_col,
                a_per_window=_site_a_pw,
                b1_per_window=_site_b1_pw,
                b2_per_window=_site_b2_pw,
                we0=_we0,
                we1=_we1,
                b1_qualifying_steps=_b1_steps,
                b2_qualifying_steps=_b2_steps,
                b1_n_per_class=int(_row.get("B1_n_per_class") or 0),
                b2_n_per_class=int(_row.get("B2_n_per_class") or 0),
                a_n_step1=int(_row.get("A_n_step1") or 0),
                a_n_step6=int(_row.get("A_n_step6") or 0),
                acoustic_sign=_a_sign,
                site_type=str(_row.get("site_type") or "unknown"),
                manifest_tuning=str(_row.get("manifest_tuning") or ""),
                acoustic_peak_auc=phon_auc.get((_ei, _pp)),
                search_smin=SMIN_LO,
                search_smax=PAIR_SMAX_HI.get(_pp, SMAX_HI_ABS),
                b_search_smin=B_SEARCH_SMIN,
                b_search_smax=B_SEARCH_SMAX,
            )
            _pdf.savefig(_fig, bbox_inches="tight")
            plt.close(_fig)
            _n_plotted += 1
        except Exception as _exc:
            _failures_plot.append((_ei, _pp, repr(_exc)))
            plt.close("all")

    if _n_plotted == 0 and sorted_sites:
        _fig0, _ax0 = plt.subplots(figsize=(8.5, 4))
        _ax0.text(0.5, 0.5, f"{subject}: all {len(sorted_sites)} sites failed to render",
                  ha="center", va="center", fontsize=12)
        _ax0.axis("off")
        _pdf.savefig(_fig0)
        plt.close(_fig0)

print(f"star_plots_early.pdf: {_n_plotted} pages  ({len(_failures_plot)} failures)")
if _failures_plot:
    for _ei, _pp, _err in _failures_plot:
        print(f"  e{_ei} {_pp}: {_err}")

# %% [markdown]
# ## Done

# %%
print("=" * 70)
print(f"subject={subject}  K={K}  R={R}")
print(f"Sites in manifest: {len(sites)}")
if site_type_df.height > 0 and "site_type" in site_type_df.columns:
    for row in site_type_df.group_by(["site_type", "status"]).len().sort(["site_type", "status"]).iter_rows(named=True):
        print(f"  {row['site_type']:30s}  status={row['status']:30s}  n={row['len']}")
if failures:
    print(f"\n⚠ Failures ({len(failures)}):")
    for f in failures:
        print(f"  {f['subject']} e{f['electrode_idx']} {f['phoneme_pair']}: {f['error']}")
print("=" * 70)
