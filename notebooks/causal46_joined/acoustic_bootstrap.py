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
# # Acoustic bootstrap (endpoint contrast)
#
# Runs `bootstrap_A_site` (step6 − step1 endpoint contrast) for every annotated
# acoustic site in the early_acoustic_window manifest, searching in
# `[t=0, phon_smax]` with the same window_size and stride as `b4_bootstrap`.
#
# Two output tiers:
# - `a_*_all.parquet` — all annotated sites. Consumed by `contrast_plot.py` to
#   fix each acoustic site's sign/window from the clean endpoint contrast.
# - `a_*.parquet` — the type1 subset, under the original names, feeding the type1
#   comparison section of `early_perceptual_windows.py` (which assumes type1-only
#   rows). Per-site RNG is independent of the site loop, so these are
#   content-identical to a type1-only run.
#
# **No behavioural (ambiguous) trials are used here** — only unambiguous endpoint
# steps 1 and 6, pooled across both word_ends per phoneme_pair.

# %% tags=["parameters"]
early_annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/acoustic_bootstrap"

R = 1000
window_size = 5   # must match causal46_joined.window_size (b4_bootstrap grid)
stride = 5        # must match causal46_joined.stride
min_n = 4
ci_low = 2.5
ci_high = 97.5

# %%
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import polars as pl

from src.data import add_metadata_features
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import bootstrap_A_site, extract_hga  # noqa: E402
from _windows import _find_maximal_runs, _window_sign  # noqa: E402
from _within_completion import summarize_replicate_array  # noqa: E402
from _acoustic_step_bootstrap import per_window_summary  # noqa: E402

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]

# t=0 in sample space
SAMPLE_T0 = int(round((0.0 - epoch_tmin) * epoch_sfreq))
print(f"SAMPLE_T0={SAMPLE_T0}  window_size={window_size}  stride={stride}  R={R}")

# Full-timecourse upper bound per phoneme pair (mirrors t_tests.py PAIR_SMAX).
# Used for the second bootstrap that backs full-trial significance bars on ax_top.
WORD_END_TAIL_SAMPLES = 20  # +200 ms tail past word offset (sfreq=100 Hz)
_WE_SMAX = {
    we: int(round((OFFSET_DICT[we] - epoch_tmin) * epoch_sfreq)) + WORD_END_TAIL_SAMPLES
    for we in OFFSET_DICT
}
PAIR_SMAX = {
    pp: max(_WE_SMAX[we] for we in wes)
    for pp, wes in PHONEME_PAIR_TO_WORD_ENDS.items()
}
print(f"PAIR_SMAX (samples): {PAIR_SMAX}")

# %% [markdown]
# ## Load inputs

# %%
phon_peaks_df = pl.read_parquet(phon_peaks_path)
print(f"phon_peaks_df: {phon_peaks_df.height} rows, cols: {phon_peaks_df.columns}")

# phon_peaks smin/smax are the acoustic decoder peak window per site.
# We use phon_smax as the upper bound for the acoustic bootstrap search.
phon_peak_lookup: dict[tuple, tuple[int, int]] = {
    (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"]): (int(r["smin"]), int(r["smax"]))
    for r in phon_peaks_df.iter_rows(named=True)
}

# %%
early_annotation_df = pl.read_csv(early_annotations_path)
print(f"early_annotation_df: {early_annotation_df.height} rows, cols: {early_annotation_df.columns}")

# Run the endpoint bootstrap over ALL annotated acoustic sites (not just type1)
# so that contrast_plot.py can orient every acoustic site by its endpoint sign.
# The type1 subset is written back under the original output names for the
# existing type1-only consumers (early_perceptual_windows.py etc.).
all_sites = (
    early_annotation_df
    .select(SITE_KEYS)
    .with_columns(pl.col("electrode_idx").cast(pl.Int64))
    .unique()
)
type1_sites = (
    early_annotation_df
    .filter(pl.col("site_type_relabel") == "type1_acoustic_only")
    .select(SITE_KEYS)
    .with_columns(pl.col("electrode_idx").cast(pl.Int64))
    .unique()
)
print(f"all annotated sites: {all_sites.height}; type1 sites: {type1_sites.height}")

# %% [markdown]
# ## Per-site acoustic bootstrap

# %%
boot_rows: list[dict] = []
boot_rows_full: list[dict] = []
per_site_rows: list[dict] = []
n_skipped_no_peaks = 0
n_skipped_underpowered = 0

for subject, subj_sites in all_sites.to_pandas().groupby("subject"):
    ep_path = Path(epoch_dir) / f"{subject}_epo.fif"
    ep_full = mne.read_epochs(str(ep_path), preload=True, verbose="WARNING")
    ep_full.metadata = add_metadata_features(ep_full.metadata.copy())
    md = ep_full.metadata

    print(f"\n{subject}: {len(subj_sites)} sites")

    for _, site_row in subj_sites.iterrows():
        eidx = int(site_row["electrode_idx"])
        pp   = site_row["phoneme_pair"]

        peak = phon_peak_lookup.get((subject, eidx, pp))
        if peak is None:
            warnings.warn(f"No phon_peaks entry for {subject} e{eidx} {pp} — skipping")
            n_skipped_no_peaks += 1
            continue
        phon_smin, phon_smax = peak

        pp_mask = (md["phoneme_pair"] == pp).values
        ep_pp   = ep_full[pp_mask]
        md_pp   = md[pp_mask].reset_index(drop=True)
        hga     = extract_hga(ep_pp, eidx)

        result = bootstrap_A_site(
            hga, md_pp,
            search_smin=SAMPLE_T0,
            search_smax=phon_smax,
            window_size=window_size,
            stride=stride,
            R=R,
            min_n=min_n,
        )
        if result is None:
            warnings.warn(
                f"{subject} e{eidx} {pp}: fewer than {min_n} endpoint trials — skipping"
            )
            n_skipped_underpowered += 1
            continue

        rows, n_lo, n_hi = result
        for r in rows:
            boot_rows.append({
                "subject": subject,
                "electrode_idx": eidx,
                "phoneme_pair": pp,
                "replicate": r["replicate"],
                "smin": r["smin"],
                "smax": r["smax"],
                "mean_diff_raw": r["mean_diff_raw"],
                "n_per_class": r["n_per_class"],
            })

        per_site_rows.append({
            "subject": subject,
            "electrode_idx": eidx,
            "phoneme_pair": pp,
            "phon_smin": phon_smin,
            "phon_smax": phon_smax,
            "n_lo": n_lo,
            "n_hi": n_hi,
            "n_per_class": min(n_lo, n_hi),
        })

        # Second pass: full-timecourse bootstrap for star-plot ax_top bars.
        # Same seed, same trial draws — windows within [SAMPLE_T0, phon_smax]
        # are identical; additional windows cover [phon_smax, PAIR_SMAX[pp]].
        result_full = bootstrap_A_site(
            hga, md_pp,
            search_smin=SAMPLE_T0,
            search_smax=PAIR_SMAX[pp],
            window_size=window_size,
            stride=stride,
            R=R,
            min_n=min_n,
        )
        if result_full is not None:
            rows_full, _, _ = result_full
            for r in rows_full:
                boot_rows_full.append({
                    "subject": subject,
                    "electrode_idx": eidx,
                    "phoneme_pair": pp,
                    "replicate": r["replicate"],
                    "smin": r["smin"],
                    "smax": r["smax"],
                    "mean_diff_raw": r["mean_diff_raw"],
                    "n_per_class": r["n_per_class"],
                })

        print(f"  e{eidx} {pp}: n_lo={n_lo} n_hi={n_hi}  {len(rows)} rows (early)  {len(rows_full) if result_full else 0} rows (full)")

print(
    f"\nDone: {len(per_site_rows)} sites processed, "
    f"{n_skipped_no_peaks} skipped (no phon_peaks), "
    f"{n_skipped_underpowered} skipped (underpowered)"
)

# %% [markdown]
# ## Write outputs

# %%
A_BOOT_COLS = ["subject", "electrode_idx", "phoneme_pair",
               "replicate", "smin", "smax", "mean_diff_raw", "n_per_class"]
A_SITE_COLS = ["subject", "electrode_idx", "phoneme_pair",
               "phon_smin", "phon_smax", "n_lo", "n_hi", "n_per_class"]


def _type1_subset(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(type1_sites, on=SITE_KEYS, how="semi") if df.height else df


if boot_rows:
    a_bootstrap      = pl.DataFrame(boot_rows)
    a_bootstrap_full = pl.DataFrame(boot_rows_full) if boot_rows_full else pl.DataFrame(
        {c: pl.Series([], dtype=pl.Utf8) for c in A_BOOT_COLS}
    ).cast({"electrode_idx": pl.Int64, "replicate": pl.Int64,
            "smin": pl.Int64, "smax": pl.Int64,
            "mean_diff_raw": pl.Float64, "n_per_class": pl.Int64})
    a_per_site  = pl.DataFrame(per_site_rows)
else:
    a_bootstrap = pl.DataFrame(
        {c: pl.Series([], dtype=pl.Utf8) for c in A_BOOT_COLS}
    ).cast({"electrode_idx": pl.Int64, "replicate": pl.Int64,
            "smin": pl.Int64, "smax": pl.Int64,
            "mean_diff_raw": pl.Float64, "n_per_class": pl.Int64})
    a_bootstrap_full = a_bootstrap.clone()
    a_per_site = pl.DataFrame(
        {c: pl.Series([], dtype=pl.Utf8) for c in A_SITE_COLS}
    ).cast({"electrode_idx": pl.Int64,
            "phon_smin": pl.Int64, "phon_smax": pl.Int64,
            "n_lo": pl.Int64, "n_hi": pl.Int64, "n_per_class": pl.Int64})

# Per-window CI summary for star-plot significance bars on ax_top.
# mean_diff_aligned = mean_diff_raw (step6 > step1 polarity is fixed).
def _boot_to_per_window(df: pl.DataFrame) -> pl.DataFrame:
    aligned = df.with_columns(
        pl.col("mean_diff_raw").alias("mean_diff_aligned"),
        (pl.col("smin") / epoch_sfreq + epoch_tmin).alias("tmin"),
        (pl.col("smax") / epoch_sfreq + epoch_tmin).alias("tmax"),
        pl.lit(None).cast(pl.Float64).alias("acoustic_peak_auc"),
    )
    return per_window_summary(aligned, SITE_KEYS)


a_per_window      = _boot_to_per_window(a_bootstrap)
a_per_window_full_df = _boot_to_per_window(a_bootstrap_full)

# Full (all-site) early-window outputs — consumed by contrast_plot.py to orient
# every acoustic site by its endpoint sign (must not include late windows).
a_bootstrap.write_parquet(OUT_DIR / "a_bootstrap_all.parquet")
a_per_site.write_parquet(OUT_DIR / "a_per_site_all.parquet")
a_per_window.write_parquet(OUT_DIR / "a_per_window_all.parquet")

# Full-timecourse outputs — star-plot ax_top significance bars across full trial.
a_per_window_full_df.write_parquet(OUT_DIR / "a_per_window_full_all.parquet")
_type1_subset(a_per_window_full_df).write_parquet(OUT_DIR / "a_per_window_full.parquet")


# Type1 subset under the original names, preserving existing type1-only
# consumers. Per-site RNG is independent of the site loop, so these rows are
# content-identical to a type1-only run.
a_bootstrap_t1 = _type1_subset(a_bootstrap)
a_per_site_t1 = _type1_subset(a_per_site)
a_per_window_t1 = _type1_subset(a_per_window)
a_bootstrap_t1.write_parquet(OUT_DIR / "a_bootstrap.parquet")
a_per_site_t1.write_parquet(OUT_DIR / "a_per_site.parquet")
a_per_window_t1.write_parquet(OUT_DIR / "a_per_window.parquet")

print(f"a_bootstrap:       {a_bootstrap.height:,} rows (all)  {a_bootstrap_t1.height:,} rows (type1)")
print(f"a_per_site:        {a_per_site.height} rows (all)  {a_per_site_t1.height} rows (type1)")
print(f"a_per_window:      {a_per_window.height} rows (all)  {a_per_window_t1.height} rows (type1)")
print(f"a_per_window_full: {a_per_window_full_df.height} rows (all)  {_type1_subset(a_per_window_full_df).height} rows (type1)")
if a_per_site.height > 0:
    print(a_per_site)
