# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
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
# # Acoustic decoding — single-electrode window inspector
#
# Interactive notebook to inspect a hand-picked time window on one electrode.
# Visualises the causal46 star plot with the new window highlighted, fits a
# fresh acoustic decoder on both the source (peak) and new windows in one
# `run_acoustic_searchlight` call, then transfers the source decoder onto the
# new window and reports both AUCs.
#
# **Usage:** set the params cell below, then run all cells.  No files are
# written unless you un-comment the `savefig` line.

# %% tags=["parameters"]

# --- Site ---
subject = "EC282"
electrode_idx = 106
phoneme_pair = "dn"
word_end = "necessary"

# --- New window onset ---
# If both are None the window is auto-resolved from the peak b4 behavioral
# contrast (b4_per_window_path must point to a valid parquet).
# new_window_onset_s takes priority over new_window_onset_sample when set.
new_window_onset_s = None       # seconds post word onset (or None for b4 auto-resolve)
new_window_onset_sample = None  # fallback sample offset into epoch

# --- Source-window override (for sites without a recorded peak) ---
# Leave both as None to use the peak acoustic window as source.
source_smin = None
source_smax = None

# --- Paths ---
config_path = "config.yaml"
reg_lambda_winners_path = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json"
phon_peaks_path = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
b4_per_window_path = "outputs/causal46_joined/t_tests/b4_per_window.parquet"
epoch_dir = "outputs/epochs_preprocessed"
behav_dec_full_root = "outputs/causal46_joined/behavior_decoding_single_electrode"
behav_dec_hga_only_root = "outputs/causal46_joined/behavior_decoding_single_electrode_hga_only"
# Full acoustic searchlight scores for this subject (timecourse plot in cell 7).
# Defaults to the causal6 per-subject scores parquet; set to None to skip.
acoustic_scores_path = None  # auto-resolved below if left None

# --- Fit options ---
device = "cpu"           # "cuda" for GPU; "cpu" is sufficient for 1 electrode × 2 windows

# %% [markdown]
# ## Load config and resolve hyperparameters

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

# Resolve repo root so relative imports work from any CWD.
REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks" / "causal46_joined"))

from src.models.causal6 import run_acoustic_searchlight
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin
from src.viz_provisional import load_epochs_dict
from _within_completion import (
    load_behav_decoding_scores,
    matched_n_star_plot,
    n_per_class_from_per_step,
    per_step_class_counts,
    resolve_behavior_col,
)
from _acoustic_window_inspect import (
    build_new_window,
    compute_transfer_auc,
    pick_window_from_b4,
    resolve_source_and_peak_windows,
)

# %%
_cfg = yaml.safe_load(Path(config_path).read_text())
C6 = _cfg["causal6"]
C46 = _cfg["causal46_joined"]
DEC = _cfg["analysis"]["decoding"]

min_class_k = int(C46["min_class_k"])         # min trials per class per step for qualifying_steps

_winners = json.loads(Path(reg_lambda_winners_path).read_text())
reg_lambda = float(_winners["reg_lambda_acoustic"])

min_sample = int(DEC["min_sample"])
window_size = int(DEC["window_size"])
stride = int(DEC["stride"])
n_folds = int(C6["n_folds"])
cv_random_state = int(C6["cv_random_state"])
tol = float(C6["tol"])
max_iter = int(C6["max_iter"])

print(f"min_class_k={min_class_k}")
print(
    f"reg_lambda={reg_lambda}  window_size={window_size}  stride={stride}"
    f"  min_sample={min_sample}"
)
print(
    f"n_folds={n_folds}  cv_random_state={cv_random_state}"
    f"  tol={tol}  max_iter={max_iter}  device={device!r}"
)

# %%
epochs_dict = load_epochs_dict(Path(epoch_dir))
print(f"Epochs loaded: {sorted(epochs_dict)}")

phon_peaks = pl.read_parquet(phon_peaks_path)
print(f"Peak rows: {phon_peaks.height}")

b4_per_window = pl.read_parquet(b4_per_window_path) if Path(b4_per_window_path).exists() else None
print(f"B4 per-window rows: {b4_per_window.height if b4_per_window is not None else 'n/a (file not found)'}")

behav_decoding_df = load_behav_decoding_scores(
    f"{behav_dec_full_root}/{subject}/scores.parquet",
    f"{behav_dec_hga_only_root}/{subject}/scores.parquet",
)
print(f"Behavioral decoding scores loaded: {behav_decoding_df is not None}")

if acoustic_scores_path is None:
    _auto = (
        f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet"
    )
    acoustic_scores_path = _auto if Path(_auto).exists() else None
print(f"Acoustic searchlight scores: {acoustic_scores_path}")

# %% [markdown]
# ## Resolve windows

# %%
ep = epochs_dict[subject]
n_times = ep.get_data(picks=[electrode_idx]).shape[-1]

peak_smin, peak_smax, peak_auc, source_smin_eff, source_smax_eff, width = (
    resolve_source_and_peak_windows(
        phon_peaks, subject, electrode_idx, phoneme_pair,
        source_smin_override=source_smin,
        source_smax_override=source_smax,
    )
)

_b4_peak_info = None  # (b4_smin, b4_smax, contrast_med) if auto-resolved

if new_window_onset_s is None and new_window_onset_sample is None:
    if b4_per_window is None:
        raise RuntimeError(
            "new_window_onset_s and new_window_onset_sample are both None, "
            "but b4_per_window_path does not exist. "
            "Set new_window_onset_s to an explicit value."
        )
    new_smin, new_smax, _b4_smin, _b4_smax, _b4_contrast = pick_window_from_b4(
        b4_per_window, subject, electrode_idx, phoneme_pair, word_end,
        width, epoch_tmin, epoch_sfreq, min_sample, n_times,
    )
    _b4_peak_info = (_b4_smin, _b4_smax, _b4_contrast)
    print(
        f"Auto-resolved new window from b4 peak behavioral contrast:\n"
        f"  b4 peak: smin={_b4_smin} smax={_b4_smax}  "
        f"({_b4_smin/epoch_sfreq+epoch_tmin:.3f}s–{_b4_smax/epoch_sfreq+epoch_tmin:.3f}s)  "
        f"mean_diff_aligned_med={_b4_contrast:.4f}"
    )
else:
    new_smin, new_smax = build_new_window(
        onset_s=new_window_onset_s,
        onset_sample=new_window_onset_sample,
        width=width,
        epoch_tmin=epoch_tmin,
        epoch_sfreq=epoch_sfreq,
        min_sample=min_sample,
        n_times=n_times,
    )

print(f"Peak window:   smin={peak_smin}  smax={peak_smax}  "
      f"({peak_smin/epoch_sfreq+epoch_tmin:.3f}s – {peak_smax/epoch_sfreq+epoch_tmin:.3f}s)  "
      f"AUC={peak_auc:.3f}" if peak_auc is not None else "Peak window: none")
print(f"Source window: smin={source_smin_eff}  smax={source_smax_eff}  "
      f"({source_smin_eff/epoch_sfreq+epoch_tmin:.3f}s – "
      f"{source_smax_eff/epoch_sfreq+epoch_tmin:.3f}s)")
print(f"New window:    smin={new_smin}  smax={new_smax}  "
      f"({new_smin/epoch_sfreq+epoch_tmin:.3f}s – {new_smax/epoch_sfreq+epoch_tmin:.3f}s)  "
      f"width={width}")

# %% [markdown]
# ## Derive qualifying steps for the star plot

# %%
md = ep.metadata
bhv_col = resolve_behavior_col(md)
pp_mask = (md["phoneme_pair"] == phoneme_pair).values
md_pp = md[pp_mask].reset_index(drop=True)

_candidate_steps = [
    int(s) for s in sorted(md_pp["resampled"].dropna().unique())
    if s not in (1, 6)
]
_per_step = per_step_class_counts(
    md_pp, word_end=word_end, qualifying_steps=_candidate_steps, group_col=bhv_col
)
qualifying_steps = [
    s for s, by_class in _per_step.items()
    if len(by_class) >= 2 and all(len(v) >= min_class_k for v in by_class.values())
]
n_per_class = n_per_class_from_per_step({s: _per_step[s] for s in qualifying_steps})

print(f"qualifying_steps={qualifying_steps}  n_per_class={n_per_class}")

# %% [markdown]
# ## Star plot with both windows highlighted

# %%
_AC_SEARCH_SMIN = int(_cfg["analysis"]["decoding"].get("acoustic_peak_search_smin", 0))
_AC_SEARCH_SMAX = int(_cfg["analysis"]["decoding"].get("acoustic_peak_search_smax", 50))

fig = matched_n_star_plot(
    subject=subject,
    electrode_idx=electrode_idx,
    phoneme_pair=phoneme_pair,
    word_end=word_end,
    qualifying_steps=qualifying_steps,
    epochs_dict=epochs_dict,
    n_per_class=n_per_class,
    phon_smin=peak_smin,
    phon_smax=peak_smax,
    phon_search_smin=_AC_SEARCH_SMIN,
    phon_search_smax=_AC_SEARCH_SMAX,
    acoustic_peak_auc=peak_auc,
    behav_decoding_df=behav_decoding_df,
    early_smax_s=_AC_SEARCH_SMAX,
)

# Post-hoc: overlay the new window as a purple axvspan on all panels.
_new_tmin = new_smin / epoch_sfreq + epoch_tmin
_new_tmax = new_smax / epoch_sfreq + epoch_tmin
_new_color = "#8856a7"

_panels = list(fig.axes)
for _ax in _panels:
    _ax.axvspan(_new_tmin, _new_tmax, color=_new_color, alpha=0.18, label="new window")

# Add legend entry only on ax_top to avoid cluttering every panel.
fig.axes[0].legend(fontsize=7, loc="upper left", framealpha=0.7)
fig.suptitle(
    f"{subject} e{electrode_idx} {phoneme_pair} · {word_end}",
    fontsize=9, y=1.01,
)
plt.tight_layout()
plt.show()
# plt.savefig(f"{subject}_e{electrode_idx}_{phoneme_pair}_{word_end}_inspect.pdf",
#             bbox_inches="tight")

# %% [markdown]
# ## Fit both decoders in one searchlight call

# %%
# Dedup in case source == new (e.g. user set source override == new window).
_windows_list = [[source_smin_eff, source_smax_eff]]
if new_smin != source_smin_eff or new_smax != source_smax_eff:
    _windows_list.append([new_smin, new_smax])
windows = np.array(_windows_list, dtype=np.int64)

print(f"Fitting {len(windows)} window(s): {windows.tolist()}")

scores, predictions, coefficients = run_acoustic_searchlight(
    ep,
    subject=subject,
    electrode_idxs=[electrode_idx],
    windows=windows,
    target="categorical_acoustic_cue",
    reg_lambda=reg_lambda,
    n_folds=n_folds,
    cv_random_state=cv_random_state,
    device=device,
    tol=tol,
    max_iter=max_iter,
)

# Filter to this phoneme pair.
scores = scores.filter(pl.col("phoneme_pair") == phoneme_pair)
predictions = predictions.filter(pl.col("phoneme_pair") == phoneme_pair)
coefficients = coefficients.filter(pl.col("phoneme_pair") == phoneme_pair)

# Sanity: both windows present in scores.
_score_windows = set(
    zip(scores["smin"].to_list(), scores["smax"].to_list())
)
for _w in _windows_list:
    assert tuple(_w) in _score_windows, (
        f"Window {_w} missing from scores — check phoneme pair filter."
    )
print("Both windows present in scores. Done.")

# %% [markdown]
# ## Transfer: source decoder → new window

# %%
# epoch_data: (N_total_epochs, N_times) indexed by metadata index labels.
# Mirrors evaluate_phonetic_transfer's `t_epoch_data[epoch_idxs]` pattern.
_epoch_data = ep.get_data(picks=[electrode_idx]).squeeze(1)

# sanity check: transfer from source -> source should match original AUC.
orig_mean, orig_folds = compute_transfer_auc(
    epoch_data=_epoch_data,
    phoneme_pair=phoneme_pair,
    source_smin=source_smin_eff,
    source_smax=source_smax_eff,
    new_smin=source_smin_eff,  # use source window as "new" for original AUC
    new_smax=source_smax_eff,
    predictions=predictions,
    coefficients=coefficients,
    n_folds=n_folds,
)
print(f"Original AUC (source → source): {orig_mean:.4f}  "
      f"fold AUCs={[round(a, 4) for a in orig_folds]}")
assert np.isclose(orig_mean, peak_auc, atol=1e-4), (
    f"Original AUC {orig_mean:.4f} does not match peak AUC {peak_auc:.4f} — "
    f"check that source window is correctly aligned with scored window."
)

transfer_mean, transfer_folds = compute_transfer_auc(
    epoch_data=_epoch_data,
    phoneme_pair=phoneme_pair,
    source_smin=source_smin_eff,
    source_smax=source_smax_eff,
    new_smin=new_smin,
    new_smax=new_smax,
    predictions=predictions,
    coefficients=coefficients,
    n_folds=n_folds,
)

print(f"Transfer AUC (source → new): {transfer_mean:.4f}  "
      f"fold AUCs={[round(a, 4) for a in transfer_folds]}")

# %% [markdown]
# ## Report

# %%
def _window_auc(sc, smin, smax):
    """Fold-mean AUC for one window from a scores DataFrame."""
    row = sc.filter((pl.col("smin") == smin) & (pl.col("smax") == smax))
    if row.height == 0:
        return float("nan"), float("nan")
    vals = row["test_roc_auc"].to_numpy()
    return float(vals.mean()), float(vals.std())


src_mean, src_std = _window_auc(scores, source_smin_eff, source_smax_eff)
new_mean, new_std = _window_auc(scores, new_smin, new_smax)

print("\n=== Decoder AUCs ===")
print(f"  Source (peak) window  [{source_smin_eff}:{source_smax_eff}]  "
      f"{source_smin_eff/epoch_sfreq+epoch_tmin:.3f}–{source_smax_eff/epoch_sfreq+epoch_tmin:.3f}s")
print(f"    Retrained-on-source :  {src_mean:.4f} ± {src_std:.4f}")
print()
print(f"  New window            [{new_smin}:{new_smax}]  "
      f"{new_smin/epoch_sfreq+epoch_tmin:.3f}–{new_smax/epoch_sfreq+epoch_tmin:.3f}s")
print(f"    Retrained-on-new    :  {new_mean:.4f} ± {new_std:.4f}")
print(f"    Transfer (src → new):  {transfer_mean:.4f} ± {np.std(transfer_folds):.4f}")
print()
if peak_auc is not None:
    print(f"  (On-disk peak AUC from parquet: {peak_auc:.4f} — display context only;")
    print(f"   may differ slightly from retrained-on-source due to device/seed.)")

# %%
# Bar chart summary.
fig2, ax = plt.subplots(figsize=(5, 3))
labels = ["Source\n(retrained)", "New\n(retrained)", "Transfer\n(src→new)"]
means = [src_mean, new_mean, transfer_mean]
stds = [src_std, new_std, float(np.std(transfer_folds))]
colors = ["#4dac26", "#8856a7", "#d95f0e"]
xs = np.arange(len(labels))
ax.bar(xs, means, yerr=stds, color=colors, alpha=0.8, capsize=4, width=0.5)
ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.6, label="chance")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("ROC AUC")
ax.set_ylim(0.4, 1.0)
ax.set_title(
    f"{subject} e{electrode_idx} {phoneme_pair} · {word_end}\n"
    f"source=[{source_smin_eff}:{source_smax_eff}]  "
    f"new=[{new_smin}:{new_smax}]",
    fontsize=8,
)
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# Searchlight time-course (from on-disk per-subject scores, if available).
if acoustic_scores_path is not None and Path(acoustic_scores_path).exists():
    _all_scores = (
        pl.read_parquet(acoustic_scores_path)
        .filter(
            (pl.col("electrode_idx") == electrode_idx)
            & (pl.col("phoneme_pair") == phoneme_pair)
        )
        .group_by(["smin", "smax"])
        .agg(pl.col("test_roc_auc").mean())
        .with_columns(
            ((pl.col("smin") + pl.col("smax")) / 2 / epoch_sfreq + epoch_tmin).alias("tcenter")
        )
        .sort("tcenter")
    )
    fig3, ax3 = plt.subplots(figsize=(7, 2.5))
    ax3.plot(
        _all_scores["tcenter"].to_numpy(),
        _all_scores["test_roc_auc"].to_numpy(),
        color="k", lw=1.2, label="searchlight AUC",
    )
    ax3.axhline(0.5, color="k", lw=0.6, ls="--", alpha=0.5)
    # Source (peak) window
    _src_t = (source_smin_eff + source_smax_eff) / 2 / epoch_sfreq + epoch_tmin
    ax3.axvspan(
        source_smin_eff / epoch_sfreq + epoch_tmin,
        source_smax_eff / epoch_sfreq + epoch_tmin,
        color="#4dac26", alpha=0.25, label=f"source [{source_smin_eff}:{source_smax_eff}]",
    )
    # New window
    ax3.axvspan(
        _new_tmin, _new_tmax,
        color=_new_color, alpha=0.25, label=f"new [{new_smin}:{new_smax}]",
    )
    ax3.set_xlabel("Time (s post word onset)")
    ax3.set_ylabel("ROC AUC")
    ax3.set_title(
        f"Acoustic searchlight — {subject} e{electrode_idx} {phoneme_pair}",
        fontsize=8,
    )
    ax3.legend(fontsize=7)
    plt.tight_layout()
    plt.show()
else:
    print(
        "Searchlight timecourse not shown: acoustic_scores_path not found or not set.\n"
        f"Set acoustic_scores_path to "
        f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet"
    )
