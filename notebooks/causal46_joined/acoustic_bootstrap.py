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
# # Acoustic bootstrap for type1 sites
#
# Runs `bootstrap_A_site` (step6 − step1 endpoint contrast) for each
# type1 / acoustic-only site in the early_acoustic_window manifest, searching
# in `[t=0, phon_smax]` with the same window_size and stride as `b4_bootstrap`.
#
# The result (`a_bootstrap.parquet`) feeds the type1 comparison section of
# `early_perceptual_windows.py`, replacing the previous behavioural-bootstrap
# proxy so the acoustic-onset timing is measured with a matching method.
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
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import bootstrap_A_site, extract_hga  # noqa: E402
from _windows import _find_maximal_runs, _window_sign  # noqa: E402
from _within_completion import summarize_replicate_array  # noqa: E402

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]

# t=0 in sample space
SAMPLE_T0 = int(round((0.0 - epoch_tmin) * epoch_sfreq))
print(f"SAMPLE_T0={SAMPLE_T0}  window_size={window_size}  stride={stride}  R={R}")

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

type1_sites = (
    early_annotation_df
    .filter(pl.col("site_type_relabel") == "type1_acoustic_only")
    .select(SITE_KEYS)
    .unique()
)
print(f"type1 sites: {type1_sites.height}")

# %% [markdown]
# ## Per-site acoustic bootstrap

# %%
boot_rows: list[dict] = []
per_site_rows: list[dict] = []
n_skipped_no_peaks = 0
n_skipped_underpowered = 0

for subject, subj_sites in type1_sites.to_pandas().groupby("subject"):
    ep_path = Path(epoch_dir) / f"{subject}_epo.fif"
    ep_full = mne.read_epochs(str(ep_path), preload=True, verbose="WARNING")
    ep_full.metadata = add_metadata_features(ep_full.metadata.copy())
    md = ep_full.metadata

    print(f"\n{subject}: {len(subj_sites)} type1 sites")

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
        print(f"  e{eidx} {pp}: n_lo={n_lo} n_hi={n_hi}  {len(rows)} rows")

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

if boot_rows:
    a_bootstrap = pl.DataFrame(boot_rows)
    a_per_site  = pl.DataFrame(per_site_rows)
else:
    a_bootstrap = pl.DataFrame(
        {c: pl.Series([], dtype=pl.Utf8) for c in A_BOOT_COLS}
    ).cast({"electrode_idx": pl.Int64, "replicate": pl.Int64,
            "smin": pl.Int64, "smax": pl.Int64,
            "mean_diff_raw": pl.Float64, "n_per_class": pl.Int64})
    a_per_site = pl.DataFrame(
        {c: pl.Series([], dtype=pl.Utf8) for c in A_SITE_COLS}
    ).cast({"electrode_idx": pl.Int64,
            "phon_smin": pl.Int64, "phon_smax": pl.Int64,
            "n_lo": pl.Int64, "n_hi": pl.Int64, "n_per_class": pl.Int64})

a_bootstrap.write_parquet(OUT_DIR / "a_bootstrap.parquet")
a_per_site.write_parquet(OUT_DIR / "a_per_site.parquet")

print(f"a_bootstrap: {a_bootstrap.height:,} rows")
print(f"a_per_site:  {a_per_site.height} rows")
if a_per_site.height > 0:
    print(a_per_site)
