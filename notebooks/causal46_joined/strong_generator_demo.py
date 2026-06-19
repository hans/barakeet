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
# # Strong-generator demo: does the behavioral slope reproduce the unambiguous response?
#
# **Interactive, not part of the Snakemake pipeline.**
#
# Pick one `(subject, electrode_idx, phoneme_pair, word_end)` cell and a time
# window you *claim* is behaviorally selective. The notebook:
#
# 1. Renders the B4 star plot (top facet = unambiguous steps 1 vs 6 within the
#    word_end; bottom facet = ambiguous percept-balanced bootstrap) and marks
#    your claimed window on every facet.
# 2. **β_amb** — reads the ambiguous within-completion slope straight from the
#    t-test output (`b4_bootstrap.parquet`), at the grid window nearest your
#    claim. This is the per-step-balanced percept difference on a FIXED reference
#    (heard-/n/ − heard-/d/; binary report, belief difference ≡ 1), with no
#    per-electrode acoustic-polarity alignment so signs are comparable across
#    electrodes.
# 3. **β_unamb** — computes the *same-window* endpoint (step 1 vs 6 = /n/ − /d/)
#    difference on the unambiguous trials of this word_end, same fixed reference,
#    with a matched balanced bootstrap.
# 4. **Applies** the ambiguous slope to the unambiguous window. **Primary
#    statistic = the difference `β_amb − β_unamb`** (well-defined even when the
#    late-window endpoint response β_unamb straddles zero). The strong
#    (binary-belief) generator predicts `difference ≈ 0`. Verdict is asymmetric:
#    `difference ≫ 0` **rules out** a single generator for this cell;
#    `difference ≈ 0` is only *consistent with* one. A `ratio = β_amb / β_unamb`
#    ("fraction explained") is shown only when both slopes are reliably nonzero.
#
# ⚠️ **Binary-report caveat.** Even a clean `difference ≈ 0` may reflect equal
# belief strength rather than one generator; `difference ≫ 0` may reflect a
# separate acoustic generator OR weaker belief at ambiguous steps. Only a
# graded-posterior rescaling (future toggle) disambiguates magnitude.
#
# **Data note — runs on prod.** β_amb is read straight from the raw per-replicate
# `b4_bootstrap.parquet` (the t-test output), so the difference/ratio CIs are
# exact (Monte-Carlo over the two real bootstrap replicate arrays — no Normal
# approximation). That file is NOT synced to the local mount; this notebook is
# meant to run on prod (or anywhere the pipeline `outputs/` are present). Using it
# as the SOLE source also keeps grid, cells, and slope self-consistent with the
# star-plot gallery (same `t_tests.py` run).

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    bootstrap_endpoint_beta,
    extract_hga,
    matched_n_star_plot,
    resolve_behavior_col,
)

# %% tags=["parameters"]
# ── Cell + claimed window (EDIT THESE) ───────────────────────────────────────
# Default = an illustrative cell: strong reliable late percept slope (β_amb≈+2.3)
# but no matching late endpoint response → the test rules out a single generator.
subject = "EC243"
electrode_idx = 101
phoneme_pair = "dn"
word_end = "necessary"
# Claimed behaviorally-selective window, in seconds post word onset:
claim_smin = 95
claim_tmin_s = claim_smin / 100 - 0.4
claim_tmax_s = (claim_smin + 15) / 100 - 0.4

# ── Data sources (prod-canonical outputs/; the raw bootstrap is not synced) ──
b4_bootstrap_path = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
epoch_dir = "outputs/epochs_preprocessed"
textgrid_dir = "textgrids"  # available on prod; render step needs it

# ── Bootstrap / CI settings ──────────────────────────────────────────────────
R_UNAMB = 1000        # endpoint bootstrap replicates
N_RATIO_MC = 50_000   # Monte-Carlo draws for the difference / ratio CI
CI_LOW, CI_HIGH = 2.5, 97.5
MIN_ENDPOINT_N = 3    # min trials per endpoint step (within word_end)

CELL = dict(subject=subject, electrode_idx=electrode_idx,
            phoneme_pair=phoneme_pair, word_end=word_end)
print(f"cell: {CELL}")
print(f"claimed window: [{claim_tmin_s}, {claim_tmax_s}] s")


# %%
def s_to_t(s) -> float:
    return s / epoch_sfreq + epoch_tmin

def t_to_s(t: float) -> float:
    return (t - epoch_tmin) * epoch_sfreq


# %% [markdown]
# ## Locate this cell in the raw bootstrap output

# %%
if not Path(b4_bootstrap_path).exists():
    raise SystemExit(
        f"{b4_bootstrap_path} not found. This notebook reads the RAW per-replicate "
        f"bootstrap, which is not synced to the local mount — run on prod (or point "
        f"b4_bootstrap_path at a location where the pipeline output is present)."
    )
b4 = pl.read_parquet(b4_bootstrap_path)
cell_boot = b4.filter(
    (pl.col("subject") == subject) & (pl.col("electrode_idx") == electrode_idx)
    & (pl.col("phoneme_pair") == phoneme_pair) & (pl.col("word_end") == word_end)
)
if cell_boot.height == 0:
    avail = (b4.select(["subject", "electrode_idx", "phoneme_pair", "word_end"])
             .unique().sort(["subject", "electrode_idx", "phoneme_pair", "word_end"]))
    raise SystemExit(
        f"Cell {CELL} not in b4_bootstrap.parquet "
        f"({avail.height} cells available). First rows:\n{avail.head(20)}"
    )

qualifying_steps = [int(s) for s in cell_boot["qualifying_steps"][0].split(",")]
n_per_class = int(cell_boot["n_per_class"][0])
ac_peak_auc = float(cell_boot["acoustic_peak_auc"][0])
print(f"qualifying ambiguous steps: {qualifying_steps}")
print(f"n_per_class (ambiguous, per replicate): {n_per_class}")
print(f"acoustic peak AUC: {ac_peak_auc:.3f}")


# %% [markdown]
# ## Snap the claimed window to the nearest grid window
#
# β_amb comes from `b4_bootstrap` on a fixed searchlight grid (read directly from
# the file — do not assume config constants). We snap by window *center*; β_unamb
# is then computed on this same snapped `(smin, smax)` so the comparison is
# apples-to-apples.

# %%
grid = cell_boot.select(["smin", "smax"]).unique().sort("smin").to_numpy()
grid_centers_s = grid.mean(axis=1)
claim_center_s = 0.5 * (t_to_s(claim_tmin_s) + t_to_s(claim_tmax_s))
i_snap = int(np.argmin(np.abs(grid_centers_s - claim_center_s)))
SMIN, SMAX = int(grid[i_snap, 0]), int(grid[i_snap, 1])
snap_tmin_s, snap_tmax_s = s_to_t(SMIN), s_to_t(SMAX)
print(f"grid window width = {grid[0,1]-grid[0,0]} samples; "
      f"{grid.shape[0]} windows from {s_to_t(grid[0,0]):.3f}s to {s_to_t(grid[-1,1]):.3f}s")
print(f"claimed center {s_to_t(claim_center_s):.3f}s → snapped window "
      f"[{snap_tmin_s:.3f}, {snap_tmax_s:.3f}]s  (smin={SMIN}, smax={SMAX})")


# %% [markdown]
# ## β_amb — ambiguous within-completion slope (from the t-test output)
#
# Per-replicate `mean_diff_raw` at the snapped window: per-step-balanced, on a
# FIXED reference HGA[behavior=1] − HGA[behavior=0] = heard-/n/ − heard-/d/ (and
# the analogous step6-side − step1-side for bm/pb). No per-electrode acoustic
# polarity — the reference is consistent across electrodes, so signs are
# comparable for any population aggregation. We keep the full bootstrap array so
# the downstream difference/ratio CIs are exact.

# %%
beta_amb_arr = (cell_boot.filter((pl.col("smin") == SMIN) & (pl.col("smax") == SMAX))
                ["mean_diff_raw"].to_numpy())
beta_amb_arr = beta_amb_arr[np.isfinite(beta_amb_arr)]
if beta_amb_arr.size == 0:
    raise SystemExit(f"No finite β_amb replicates at window (smin={SMIN}, smax={SMAX}).")
beta_amb_med = float(np.median(beta_amb_arr))
beta_amb_ci = np.percentile(beta_amb_arr, [CI_LOW, CI_HIGH])
beta_amb_reliable = bool(beta_amb_ci[0] > 0 or beta_amb_ci[1] < 0)
print(f"β_amb  = {beta_amb_med:+.4f}  CI[{beta_amb_ci[0]:+.4f}, {beta_amb_ci[1]:+.4f}]  "
      f"(R={beta_amb_arr.size} replicates)")
print(f"  CI excludes zero: {beta_amb_reliable}")


# %% [markdown]
# ## Load epochs (single subject) and the acoustic window for this site
#
# The acoustic window (causal6 peak) is only used for the star plot's green
# "acoustic peak" shading — it plays no role in β now that both slopes use the
# fixed /n/−/d/ reference.

# %%
import mne  # noqa: E402
from src.data import add_metadata_features  # noqa: E402

ep_path = Path(epoch_dir) / f"{subject}_epo.fif"
ep_full = mne.read_epochs(str(ep_path), preload=True, verbose="WARNING")
ep_full.metadata = add_metadata_features(ep_full.metadata.copy())

md = ep_full.metadata
bhv_col = resolve_behavior_col(md)
pp_mask = (md["phoneme_pair"] == phoneme_pair).values
ep_pp = ep_full[pp_mask]
md_pp = md[pp_mask].reset_index(drop=True)
hga = extract_hga(ep_pp, electrode_idx)   # (n_trials_pp, n_times), aligned to md_pp
print(f"loaded {ep_path.name}: {hga.shape[0]} {phoneme_pair} trials")

peaks = pl.read_parquet(phon_peaks_path).filter(
    (pl.col("subject") == subject) & (pl.col("electrode_idx") == electrode_idx)
    & (pl.col("phoneme_pair") == phoneme_pair)
)
if peaks.height == 0:
    raise SystemExit(f"No acoustic peak row for {subject} e{electrode_idx} {phoneme_pair}")
peaks = peaks.sort("test_roc_auc", descending=True)  # peak window = best AUC
AC_SMIN, AC_SMAX = int(peaks["smin"][0]), int(peaks["smax"][0])
print(f"acoustic window: [{s_to_t(AC_SMIN):.3f}, {s_to_t(AC_SMAX):.3f}]s "
      f"(smin={AC_SMIN}, smax={AC_SMAX})")


# %% [markdown]
# ## β_unamb — same-window endpoint difference, fixed /n/−/d/ reference
#
# Within this word_end only (matches the star plot's top facet; suffix acoustics
# held fixed). Balanced bootstrap between the two endpoint steps; fixed reference
# `step6 − step1` (= /n/ − /d/ at endpoints), the SAME physical convention as
# β_amb — no acoustic-polarity alignment.

# %%
we_mask = (md_pp["word_end"] == word_end).values
n_lo = int((we_mask & (md_pp["resampled"] == 1).values).sum())
n_hi = int((we_mask & (md_pp["resampled"] == 6).values).sum())
print(f"endpoint trials within {word_end}: step1 n={n_lo}, step6 n={n_hi}")
n_bal = min(n_lo, n_hi)

beta_unamb_arr = bootstrap_endpoint_beta(
    hga, md_pp,
    word_end=word_end, smin=SMIN, smax=SMAX,
    R=R_UNAMB, min_n=MIN_ENDPOINT_N,
)
if beta_unamb_arr is None:
    raise SystemExit(
        f"Too few endpoint trials within word_end (need ≥{MIN_ENDPOINT_N} each)."
    )

beta_unamb_med = float(np.median(beta_unamb_arr))
beta_unamb_ci = np.percentile(beta_unamb_arr, [CI_LOW, CI_HIGH])
beta_unamb_reliable = bool(beta_unamb_ci[0] > 0 or beta_unamb_ci[1] < 0)
print(f"β_unamb = {beta_unamb_med:+.4f}  CI[{beta_unamb_ci[0]:+.4f}, {beta_unamb_ci[1]:+.4f}]  "
      f"reliable={beta_unamb_reliable}  (R={R_UNAMB}, n_bal={n_bal}/step)")


# %% [markdown]
# ## Apply the ambiguous slope to the unambiguous window
#
# Primary = difference `β_amb − β_unamb` (strong binary-belief generator ⇒ ≈ 0).
# Ratio shown only when both slopes are reliably nonzero (else it is Cauchy-tailed
# and uninterpretable). The two bootstraps are independent, so the CIs come from
# Monte-Carlo over the two real replicate arrays — exact, no parametric model.

# %%
rng = np.random.default_rng(0)
a_draws = rng.choice(beta_amb_arr, size=N_RATIO_MC, replace=True)
b_draws = rng.choice(beta_unamb_arr, size=N_RATIO_MC, replace=True)

diff_draws = a_draws - b_draws
diff_point = beta_amb_med - beta_unamb_med
diff_ci = np.percentile(diff_draws, [CI_LOW, CI_HIGH])
diff_excludes_zero = diff_ci[0] > 0 or diff_ci[1] < 0

# The DIFFERENCE verdict needs only a real ambiguous slope to "apply" (β_amb
# reliable); its CI already absorbs β_unamb's variance, so it is valid even when
# β_unamb ≈ 0 — indeed that (large β_amb, no late endpoint response) is the
# STRONGEST refutation of a single generator. The RATIO additionally needs
# β_unamb reliably nonzero (else it is Cauchy-tailed).
test_meaningful = beta_amb_reliable
ratio_meaningful = beta_amb_reliable and beta_unamb_reliable
same_sign = np.sign(beta_amb_med) == np.sign(beta_unamb_med)

print(f"β_amb            = {beta_amb_med:+.4f}  CI[{beta_amb_ci[0]:+.4f}, {beta_amb_ci[1]:+.4f}]  reliable={beta_amb_reliable}")
print(f"β_unamb          = {beta_unamb_med:+.4f}  CI[{beta_unamb_ci[0]:+.4f}, {beta_unamb_ci[1]:+.4f}]  reliable={beta_unamb_reliable}")
print(f"\nPRIMARY  β_amb − β_unamb = {diff_point:+.4f}  CI[{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")
print(f"         same sign = {same_sign}")

if not test_meaningful:
    print("\n⟹ NO BEHAVIORAL SLOPE TO APPLY: β_amb CI includes zero at the claimed "
          "window — the t-test does not support a reliable percept slope here, so "
          "the difference verdict is not meaningful. (You claimed this window is "
          "behaviorally selective; the pipeline disagrees at this window.)")
elif diff_excludes_zero:
    print("\n⟹ difference CI EXCLUDES zero → endpoints do NOT reproduce the "
          "ambiguous slope → RULES OUT a single generator here."
          + ("  (β_unamb ≈ 0: a strong ambiguity-specific response — the sharpest "
             "refutation.)" if not beta_unamb_reliable else ""))
elif beta_unamb_reliable and same_sign:
    print("\n⟹ difference CI INCLUDES zero AND β_unamb reliably reproduces the "
          "slope (same sign, nonzero) → CONSISTENT WITH one belief-driven generator "
          "(not proof — a coincidental magnitude match is possible).")
else:
    print("\n⟹ INCONCLUSIVE: difference CI includes zero only because β_unamb is "
          "too uncertain"
          + (" / opposite-signed" if (beta_unamb_reliable and not same_sign) else "")
          + " to detect a β_amb-sized effect → UNDERPOWERED, not evidence for one "
          "generator.")

if ratio_meaningful:
    ratio_draws = a_draws / b_draws
    ratio_draws = ratio_draws[np.isfinite(ratio_draws)]
    ratio_point = beta_amb_med / beta_unamb_med
    ratio_ci = np.percentile(ratio_draws, [CI_LOW, CI_HIGH])
    print(f"\n  (display) fraction explained β_amb/β_unamb = {ratio_point:+.3f}  "
          f"CI[{ratio_ci[0]:+.3f}, {ratio_ci[1]:+.3f}]")
else:
    ratio_point = np.nan
    ratio_ci = np.array([np.nan, np.nan])
    print("\n  (ratio suppressed: denominator not reliably nonzero — would be Cauchy-tailed)")

print("\n⚠ binary-report caveat: even a clean difference ≈ 0 may reflect equal "
      "belief strength rather than one generator; difference ≫ 0 may reflect a "
      "separate acoustic generator OR weaker belief at ambiguous steps. Only a "
      "graded-posterior rescaling (future toggle) disambiguates magnitude.")


# %% [markdown]
# ## Star plot with the claimed window marked on every facet
#
# (Rendering needs the textgrids, available on prod.) Snapped window = solid
# orange span; raw claimed window = dashed edges.

# %%
fig = matched_n_star_plot(
    subject, electrode_idx, phoneme_pair, word_end, qualifying_steps,
    epochs_dict={subject: ep_full},
    n_per_class=n_per_class,
    phon_smin=AC_SMIN, phon_smax=AC_SMAX,
    phon_search_smin=None, phon_search_smax=None,
    textgrid_dir=textgrid_dir,
    acoustic_peak_auc=ac_peak_auc,
    R_plot=200,
)
for ax in fig.axes:
    ax.axvspan(snap_tmin_s, snap_tmax_s, color="#fdae61", alpha=0.30, zorder=0)
    for edge in (claim_tmin_s, claim_tmax_s):
        ax.axvline(edge, color="#b35806", lw=1.0, ls="--", alpha=0.8, zorder=6)
fig.suptitle(
    f"claimed [{claim_tmin_s:.2f},{claim_tmax_s:.2f}]s → snapped "
    f"[{snap_tmin_s:.2f},{snap_tmax_s:.2f}]s   "
    f"β_amb={beta_amb_med:+.3f}  β_unamb={beta_unamb_med:+.3f}  Δ={diff_point:+.3f}"
    + ("" if test_meaningful else "  (β_amb CI spans 0: no slope to apply)"),
    fontsize=9, y=1.02,
)
fig


# %% [markdown]
# ## Comparison panel: β_amb vs β_unamb in the same window

# %%
fig2, ax = plt.subplots(figsize=(4.2, 4.0))
labels = ["β_amb\n(ambiguous,\npercept)", "β_unamb\n(endpoints,\nstep 1 vs 6)"]
meds = [beta_amb_med, beta_unamb_med]
cis = [beta_amb_ci, beta_unamb_ci]
colors = ["#2166ac", "#d73027"]
for i, (m, ci, c) in enumerate(zip(meds, cis, colors)):
    ax.bar(i, m, color=c, alpha=0.75, width=0.6)
    ax.errorbar(i, m, yerr=[[m - ci[0]], [ci[1] - m]], color="k", capsize=4, lw=1.2)
ax.axhline(beta_amb_med, color="#2166ac", lw=1.0, ls="--", alpha=0.7,
           label="β_amb predicted level")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("HGA difference (z)\n(heard-/n/ − heard-/d/)", fontsize=8)
_ratio_str = (f"   ratio={ratio_point:+.2f}[{ratio_ci[0]:+.2f},{ratio_ci[1]:+.2f}]"
              if ratio_meaningful else "   (ratio N/A)")
ax.set_title(
    f"{subject} e{electrode_idx} {phoneme_pair}·{word_end}\n"
    f"window [{snap_tmin_s:.2f},{snap_tmax_s:.2f}]s   "
    f"Δ={diff_point:+.3f}[{diff_ci[0]:+.3f},{diff_ci[1]:+.3f}]" + _ratio_str,
    fontsize=8,
)
ax.legend(fontsize=7)
fig2.tight_layout()
fig2
