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
# # Late (integration-window) Perceptual Projection (per subject)
#
# Port of `early_perceptual_projection.py` (ADR-0002) to the **late (integration)
# window** (ADR-0003 / spec `2026-07-21-late-perceptual-projection-percell-spec.md`,
# issue #22). Per **cell** = (subject, electrode_idx, phoneme_pair, word_end),
# **strict same-word-end (no pooling)**, measures how much the within-completion
# **late** percept contrast re-expresses **that same word-end's own** unambiguous
# /d/–/n/ tuning:
#
# - `â(w)` = late acoustic template = bootstrap median of (step6 − step1) HGA on
#   **unambiguous** trials of this word_end (`bootstrap_endpoint_beta`), per window.
# - `p(w)` = deterministic B4 min_class-weighted within-completion percept contrast
#   on **ambiguous** trials of this word_end (`windowed_deterministic_p`), per window.
# - **Window rule 1c** (ADR-0003 §5): mark windows reliable (β_unamb CI excludes 0),
#   anchor at argmax|median β_unamb|, take the maximal contiguous reliable run `R`,
#   unit-L2-normalize â over `R`; **π_anchored = ⟨â_unit, p|_R⟩** (NaN if no
#   reliable window).
# - Also exported (diagnostic, for #21's reliable-vs-all comparison): **π_peak**
#   (single argmax|β_unamb| window, reliability-ignored) and â-reliability descriptors.
# - Per-cell null: within-step percept-label permutation, â_unit and `R` held fixed.
#
# â comes from unambiguous trials, p from ambiguous trials — structurally
# independent (ADR-0002 non-circularity, preserved). No per-cell sign flip.
#
# ## ⚠ Grid provenance (READ — flagged to #22)
#
# The spec/ADR describe the b4_bootstrap grid as *already* the late,
# word-end-anchored grid ([acoustic-peak, word_offset+tail]). The **live**
# `t_tests.py::behav_search_range` has a DEV override searching `(0, PAIR_SMAX)` —
# from **onset**, **pair-level** upper bound. Applied unfiltered, the â-anchor
# (argmax|β_unamb|) would land on the **early acoustic peak**, making this the
# early projection again. This notebook therefore reconstructs the intended late
# grid by filtering the live grid on **both** bounds:
# `smin >= phon_smax_c6` (per-site acoustic-peak end) **and**
# `smax <= word_end_offset + WORD_END_TAIL_SAMPLES` (per word_end). Controlled by
# `late_cutoff_mode`. This is a faithful build of the LOCKED design, not a
# re-opening — but the discrepancy is surfaced to Jon before the prod run.

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

from src.data import add_metadata_features
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    extract_hga,
    resolve_behavior_col,
)
from _projection import compute_cell_projection  # noqa: E402

# %% tags=["parameters"]
subject = "EC250"
# A_significant acoustic-responsive universe (same pool as early projection).
site_pool_path = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv"
# Late searchlight window grid + observed-p cross-check (prod-only).
b4_bootstrap_path = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"
# Acoustic-peak window per site (its smax = phon_smax_c6 = late-grid lower bound).
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/late_perceptual_projection/EC250"
min_class_k = 3
# â (endpoint bootstrap) settings — match strong_generator defaults.
r_unamb = 1000
ci_low = 2.5
ci_high = 97.5
min_endpoint_n = 3
# Late-grid construction. "phon_smax": smin >= per-site acoustic-peak smax AND
# smax <= word_end offset + tail (the spec's word-end-anchored late grid).
# "pod": window-center >= pod_min_s fixed cutoff (strong_generator_scan precedent,
# for the reconciliation check). Word-end offset cap always applied.
late_cutoff_mode = "phon_smax"
pod_min_s = 0.30
word_end_tail_samples = 20
# â-anchor reading (ADR-0003 §5 is ambiguous — ratification point, issue #22).
# "reliable_max": anchor argmax|β| among reliable windows (any reliable ⇒ non-NaN).
# "global_max": anchor argmax|β| over all windows; NaN unless that window is reliable.
anchor_mode = "reliable_max"
n_perms = 10000
master_seed = 42
fdr_alpha = 0.05

# %%
OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = int(min_class_k)
R_UNAMB = int(r_unamb)
CI_LOW, CI_HIGH = float(ci_low), float(ci_high)
MIN_ENDPOINT_N = int(min_endpoint_n)
N_PERMS = int(n_perms)
MASTER_SEED = int(master_seed)
FDR_ALPHA = float(fdr_alpha)
WORD_END_TAIL_SAMPLES = int(word_end_tail_samples)
POD_MIN_S = float(pod_min_s)

print(f"subject={subject}  K={K}  R_unamb={R_UNAMB}  N_PERMS={N_PERMS}")
print(f"late_cutoff_mode={late_cutoff_mode}  (word_end tail={WORD_END_TAIL_SAMPLES} samples)")


def s_to_t(s) -> float:
    return s / epoch_sfreq + epoch_tmin


def word_end_search_smax(word_end: str) -> int:
    """Upper late-grid bound per word_end = offset sample + tail (mirrors
    t_tests.py::word_end_search_smax)."""
    offset_s = OFFSET_DICT[word_end]
    return int(round((offset_s - epoch_tmin) * epoch_sfreq)) + WORD_END_TAIL_SAMPLES


# %% [markdown]
# ## Load inputs

# %%
site_pool_all = pd.read_csv(site_pool_path)
site_pool_subj = site_pool_all[site_pool_all["subject"] == subject]
included_sites = (
    site_pool_subj[site_pool_subj["A_significant"]]
    [["subject", "electrode_idx", "phoneme_pair"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
print(f"Sites in pool for {subject}: {len(site_pool_subj)}")
print(f"A_significant sites (included): {len(included_sites)}")

# Late searchlight window grid (shared window bounds across cells; filtered to the
# late word-end range per cell below).
b4 = pl.read_parquet(b4_bootstrap_path)
GRID = [
    (int(r["smin"]), int(r["smax"]))
    for r in b4.select(["smin", "smax"]).unique().sort("smin").iter_rows(named=True)
]
print(f"b4 grid windows: {len(GRID)}  (smin {GRID[0][0]}..{GRID[-1][0]})")

# Acoustic-peak window per site → phon_smax_c6 (late-grid lower bound).
phon = pl.read_parquet(phon_peaks_path)
phon_smax_map = {
    (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"]): int(r["smax"])
    for r in phon.select(["subject", "electrode_idx", "phoneme_pair", "smax"]).iter_rows(named=True)
}

# %%
epochs_path = Path(epoch_dir) / f"{subject}_epo.fif"
ep = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
md = add_metadata_features(ep.metadata).reset_index(drop=True)
md["subject"] = subject
ep.metadata = md
bhv_col = resolve_behavior_col(md)
print(f"Loaded {len(ep)} epochs; behavior col: {bhv_col}")


# %% [markdown]
# ## Late-grid construction per cell

# %%
def late_grid_for_cell(phon_smax, word_end):
    """The late, word-end-anchored window list for one cell (ADR-0003 §2 + the
    grid-provenance note above). Both bounds applied to the live b4 grid."""
    we_smax = word_end_search_smax(word_end)
    if late_cutoff_mode == "phon_smax":
        if phon_smax is None:
            return []
        lo_ok = lambda smin: smin >= phon_smax
    elif late_cutoff_mode == "pod":
        lo_ok = lambda smin, smax=None: True  # replaced per-window below
    else:
        raise ValueError(f"unknown late_cutoff_mode={late_cutoff_mode!r}")
    out = []
    for (smin, smax) in GRID:
        if smax > we_smax:
            continue
        if late_cutoff_mode == "phon_smax":
            if smin < phon_smax:
                continue
        else:  # pod: center >= POD_MIN_S
            if s_to_t(0.5 * (smin + smax)) < POD_MIN_S:
                continue
        out.append((smin, smax))
    return out


# %% [markdown]
# ## Per-cell projection loop
#
# The claim-critical per-cell path (â over the late grid → 1c anchored run →
# π_anchored + π_peak + nulls) lives in `_projection.compute_cell_projection`
# (unit-tested on synthetic data — the notebook body is otherwise unexercised in
# dev where epochs are absent). This loop only handles grid construction, epoch
# subsetting, and record assembly.

# %%
results = []
null_arrays = {}       # cell_key → (N_PERMS,) anchored null π
null_arrays_peak = {}  # cell_key → (N_PERMS,) peak null π (diagnostic)

cells = []
for _, srow in included_sites.iterrows():
    pp = srow["phoneme_pair"]
    for we in PHONEME_PAIR_TO_WORD_ENDS[pp]:
        cells.append((int(srow["electrode_idx"]), pp, we))

# Cache per-(pp) epoch subset + per-(pp, elec) HGA to avoid recomputation.
for cell_i, (elec_idx, pp, we) in enumerate(tqdm(cells, desc="cells")):
    cell_key = f"{subject}_{elec_idx}_{pp}_{we}"
    rng = np.random.default_rng(MASTER_SEED + cell_i)

    phon_smax = phon_smax_map.get((subject, elec_idx, pp))
    windows = late_grid_for_cell(phon_smax, we)

    ep_pp = ep[ep.metadata["phoneme_pair"] == pp]
    md_pp = ep_pp.metadata.reset_index(drop=True)
    hga = extract_hga(ep_pp, elec_idx)

    we_mask = (md_pp["word_end"] == we).values
    n_step1 = int((we_mask & (md_pp["resampled"] == 1).values).sum())
    n_step6 = int((we_mask & (md_pp["resampled"] == 6).values).sum())

    base_rec = dict(
        subject=subject, electrode_idx=elec_idx, phoneme_pair=pp, word_end=we,
        phon_smax=(int(phon_smax) if phon_smax is not None else -1),
        n_grid_windows=len(windows), grid_smin=(windows[0][0] if windows else -1),
        grid_smax=(windows[-1][1] if windows else -1),
        n_step1=n_step1, n_step6=n_step6,
        r_unamb=R_UNAMB, n_perms=N_PERMS, master_seed=MASTER_SEED, cell_offset=cell_i,
        late_cutoff_mode=late_cutoff_mode, anchor_mode=anchor_mode,
    )

    metrics, null_pi, null_peak = compute_cell_projection(
        hga, md_pp, word_end=we, group_col=bhv_col, windows=windows, K=K,
        n_perms=N_PERMS, rng=rng, r_unamb=R_UNAMB, min_endpoint_n=MIN_ENDPOINT_N,
        ci_low=CI_LOW, ci_high=CI_HIGH, anchor_mode=anchor_mode,
    )
    results.append({**base_rec, **metrics})
    if null_pi is not None:
        null_arrays[cell_key] = null_pi
    if null_peak is not None:
        null_arrays_peak[cell_key] = null_peak

results_df = pd.DataFrame(results)
print(f"\nTotal cells processed: {len(results_df)}")
print("skip_reason counts:\n" + results_df["skip_reason"].value_counts(dropna=False).to_string())

# %% [markdown]
# ## Population preview (aggregate does the pre-registered CPO test)

# %%
if len(results_df) > 0:
    n_estimable = int(results_df["pi_peak"].notna().sum())
    n_reliable_cells = int(results_df["pi_anchored"].notna().sum())
    print(f"â-estimable (π_peak non-NaN)          : {n_estimable}")
    print(f"cells                                : {len(results_df)}")
    print(f"â-reliable (π_anchored non-NaN)       : {n_reliable_cells}   <- claim-bearing population")
    if n_reliable_cells > 0:
        rel = results_df[results_df["pi_anchored"].notna()]
        n_sig_1t = int((rel["p_one_tailed"] < FDR_ALPHA).sum())
        n_pos = int((rel["pi_anchored"] > 0).sum())
        print(f"  uncorrected one-tailed p<{FDR_ALPHA}       : {n_sig_1t} / {n_reliable_cells}")
        print(f"  π_anchored > 0                     : {n_pos} / {n_reliable_cells}")
    # Diagnostic reliable-vs-all (the map's spine): π_peak among all estimable.
    est = results_df[results_df["pi_peak"].notna()]
    if len(est) > 0:
        n_peak_pos = int((est["pi_peak"] > 0).sum())
        print(f"[diagnostic] π_peak estimable cells   : {len(est)}  (π_peak>0: {n_peak_pos})")

# %% [markdown]
# ## Diagnostics plot

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(f"{subject} — late perceptual projection diagnostics")

rel_df = results_df[results_df["pi_anchored"].notna()] if len(results_df) else results_df

ax = axes[0]
if len(rel_df) > 0:
    pi_vals = rel_df["pi_anchored"].values
    pooled_null = np.concatenate(list(null_arrays.values())) if null_arrays else np.array([])
    if len(pooled_null) > 0:
        ax.hist(pooled_null, bins=60, density=True, alpha=0.4, color="gray", label="pooled null (display)")
    ax.scatter(pi_vals, np.zeros_like(pi_vals), zorder=5, s=40, c="steelblue", label="π_anchored")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("π_anchored"); ax.set_ylabel("density")
    ax.set_title("π_anchored (â-reliable cells)")
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, "no â-reliable cells", ha="center", va="center", transform=ax.transAxes)

ax = axes[1]
if len(rel_df) > 0:
    ax.scatter(rel_df["a_raw_norm"].values, rel_df["pi_anchored"].values, s=30, alpha=0.7, c="steelblue")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("‖â_raw‖ over run"); ax.set_ylabel("π_anchored")
    ax.set_title("π_anchored vs ‖â‖\n(reliability gate defuses low-‖â‖ inflation)")
else:
    ax.text(0.5, 0.5, "no â-reliable cells", ha="center", va="center", transform=ax.transAxes)

ax = axes[2]
if len(results_df) > 0:
    ax.hist(results_df["n_reliable_windows"].dropna().values,
            bins=range(0, int(np.nanmax(results_df["n_reliable_windows"].values)) + 2 if results_df["n_reliable_windows"].notna().any() else 2),
            color="darkorange", alpha=0.7, edgecolor="white")
    ax.set_xlabel("# reliable late windows"); ax.set_ylabel("cells")
    ax.set_title("â-reliability across cells")
else:
    ax.text(0.5, 0.5, "no cells", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
fig.savefig(OUT_DIR / "pi_dist.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("Saved pi_dist.png")

# %% [markdown]
# ## Save outputs

# %%
results_df.to_csv(OUT_DIR / "site_results.csv", index=False)
print(f"Saved site_results.csv  ({len(results_df)} rows)")

np.savez_compressed(str(OUT_DIR / "null_pi.npz"), **{k: v for k, v in null_arrays.items()})
np.savez_compressed(str(OUT_DIR / "null_pi_peak.npz"), **{k: v for k, v in null_arrays_peak.items()})
print(f"Saved null_pi.npz  ({len(null_arrays)} anchored) + null_pi_peak.npz  ({len(null_arrays_peak)} peak)")

print("\nDone.")
