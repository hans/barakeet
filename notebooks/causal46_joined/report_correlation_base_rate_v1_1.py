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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Report-correlation base-rate test — v1.1 (windowed t-test classifier)
#
# **Problem with v1**: the null classified sites with a single-window scalar AUC.
# The real analysis (`early_perceptual_windows.py`) sweeps contiguous 10-sample
# windows and flags if ANY window survives. The windowed sweep has more chances to
# cross threshold by chance, so v1 understated the null false-positive rate. The
# two-sided result likely survives (11 vs null mean ~0.3), but the one-sided result
# (18 vs null mean ~11, v1 p=.026) may not hold under a corrected null.
#
# **Fix**: use the real windowed classifier — identical to `early_perceptual_windows.py`
# — inside the permutation loop.
#
# **Classifier (same for observed and null) — validity claim:**
# Contiguous 10-sample (100ms) windows from t=0 to pair word-offset+200ms (same
# bounds and stride as `t_tests.py`/`early_perceptual_windows.py`). Per window:
# Welch two-sample t-test on ambiguous trials (report 0 vs 1), two-sided α=0.05.
# Flag word-end if ANY window survives. Classify site: two-sided = both word-ends
# flagged; one-sided = exactly one; absent = neither.
#
# **Null**: M=1000 iterations. Permute report labels WITHIN each (step × word_end)
# cell — real HGA unchanged. Run identical windowed t-test on permuted labels.
#
# **Observed**: Same windowed t-test classifier on true labels (re-derived here
# so observed and null are provably under the same rule). Original B1/B2_aligned_sig
# counts shown for comparison.
#
# **Expected / interpretation:**
# - Two-sided: should remain p << 0.01 (v1 margin: 11 vs 0.3).
# - One-sided: null mean will RISE vs v1 (windowed sweep inflates false-positive
#   rate). Honest outcome may be one-sided no longer significant; report whatever it is.

# %%
import os
import sys
from pathlib import Path

REPO = Path("/workdir")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import warnings
import numpy as np
import pandas as pd
import polars as pl
import mne
import matplotlib.pyplot as plt
from scipy import stats
from tqdm.auto import tqdm

from src.data import add_metadata_features, get_ambiguous_resampled_steps
from src.stimuli import OFFSET_DICT, PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_tmin, epoch_sfreq

# %% tags=["parameters"]
early_acoustic_window_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
epoch_dir = "outputs/epochs_preprocessed"
v1_summary_path = "outputs/causal46_joined/report_correlation_base_rate_v1/summary.csv"
outdir = "outputs/causal46_joined/report_correlation_base_rate_v1_1"

# Classifier parameters (match t_tests.py / early_perceptual_windows.py)
WINDOW_SIZE = 10    # samples = 100ms at 100 Hz
STRIDE = 10         # samples = 100ms
ALPHA = 0.05        # per-window two-sided t-test threshold
MIN_CLASS_K = 4     # min balanced trials per class (sum of min_class[s] across steps)
M_NULL = 1000       # permutation null iterations
RNG_SEED = 42

# %%
OUTDIR = Path(outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

def t_to_sample(t_s: float) -> int:
    return int(round((t_s - epoch_tmin) * epoch_sfreq))

SAMPLE_T0 = t_to_sample(0.0)
print(f"epoch_tmin={epoch_tmin}, epoch_sfreq={epoch_sfreq}, SAMPLE_T0={SAMPLE_T0}")

# Word-end search bounds (same formula as t_tests.py)
WORD_END_TAIL = 20  # +200ms past word offset

def _we_smax(word_end: str) -> int:
    return t_to_sample(OFFSET_DICT[word_end]) + WORD_END_TAIL

WE_SMAX   = {we: _we_smax(we) for we in OFFSET_DICT}
PAIR_SMAX = {
    pp: max(WE_SMAX[we] for we in wes)
    for pp, wes in PHONEME_PAIR_TO_WORD_ENDS.items()
}
print(f"WE_SMAX:   {WE_SMAX}")
print(f"PAIR_SMAX: {PAIR_SMAX}")


def make_windows(pp: str) -> list[tuple[int, int]]:
    """Analysis windows for phoneme_pair pp: [0, PAIR_SMAX[pp]) in WINDOW_SIZE steps."""
    end = PAIR_SMAX[pp]
    wins = []
    smin = 0
    while smin + WINDOW_SIZE <= end:
        wins.append((smin, smin + WINDOW_SIZE))
        smin += STRIDE
    return wins


WINDOWS = {pp: make_windows(pp) for pp in PHONEME_PAIR_TO_WORD_ENDS}
for pp, wins in WINDOWS.items():
    print(f"  {pp}: {len(wins)} windows over [0, {PAIR_SMAX[pp]})")

# %% [markdown]
# ## Original observed distribution (reference from early_acoustic_window.csv)

# %%
ea = pd.read_csv(early_acoustic_window_path)
ea["n_we_sig"]      = ea["B1_aligned_sig"].astype(int) + ea["B2_aligned_sig"].astype(int)
ea["orig_category"] = ea["n_we_sig"].map({0: "absent", 1: "one-sided", 2: "two-sided"})

ORIG_TWOSIDED = int((ea["n_we_sig"] == 2).sum())
ORIG_ONESIDED = int((ea["n_we_sig"] == 1).sum())
ORIG_ABSENT   = int((ea["n_we_sig"] == 0).sum())

print(f"Original observed (B1/B2_aligned_sig from early_acoustic_window.csv):")
print(f"  N_sites = {len(ea)}")
print(f"  Two-sided: {ORIG_TWOSIDED}, One-sided: {ORIG_ONESIDED}, Absent: {ORIG_ABSENT}")

# v1 null baseline for sanity check (from summary.csv)
try:
    v1_summary = pd.read_csv(v1_summary_path, header=None, index_col=0).squeeze()
    V1_NULL_TWO_MEAN = float(v1_summary["null_two_mean"])
    V1_NULL_ONE_MEAN = float(v1_summary["null_one_mean"])
except Exception:
    V1_NULL_TWO_MEAN, V1_NULL_ONE_MEAN = 0.344, 11.036  # fallback from known v1 run
V1_IMPLIED_FLAG_RATE = (2 * V1_NULL_TWO_MEAN + V1_NULL_ONE_MEAN) / (2 * len(ea))
print(f"\nv1 null baseline (for sanity check):")
print(f"  null_two_mean={V1_NULL_TWO_MEAN:.3f}, null_one_mean={V1_NULL_ONE_MEAN:.3f}")
print(f"  implied per-WE false-positive rate: {V1_IMPLIED_FLAG_RATE:.3f}")

# %% [markdown]
# ## Load epoch metadata (all subjects) for ambiguous step identification

# %%
subjects = sorted(ea["subject"].unique().astype(str))
print(f"Subjects in early_acoustic_window: {subjects}")

all_md_parts = []
for subj in tqdm(subjects, desc="loading metadata"):
    ep_path = Path(epoch_dir) / f"{subj}_epo.fif"
    if not ep_path.exists():
        print(f"  WARNING: {ep_path} not found")
        continue
    ep = mne.read_epochs(str(ep_path), verbose=False, preload=False)
    ep.metadata = add_metadata_features(ep.metadata)
    md = ep.metadata[["phoneme_pair", "word_end", "resampled",
                       "behavior_dummy_forced"]].copy()
    md["subject"] = subj
    all_md_parts.append(md)

all_md_pd = pd.concat(all_md_parts, ignore_index=True)
all_md_pl = pl.from_pandas(
    all_md_pd[["subject", "phoneme_pair", "word_end", "resampled",
               "behavior_dummy_forced"]]
)
del all_md_parts

ambig_steps = get_ambiguous_resampled_steps(all_md_pl, ambiguous_response_threshold=2)
print(f"Qualifying ambiguous step groups: {len(ambig_steps)}")

# %% [markdown]
# ## Precompute per-trial per-window HGA
#
# For each (subject × electrode × phoneme_pair × word_end) cell: load epoch once
# per subject, apply baseline correction once, extract per-trial per-window mean
# HGA for the electrode. One subject held in memory at a time (~600MB peak).

# %%
def compute_n_per_class(steps_arr: np.ndarray, labels_arr: np.ndarray,
                        q_steps: list) -> int:
    """Balanced per-class count = sum of min_class[s] across qualifying steps."""
    return sum(
        min(
            int(((steps_arr == s) & (labels_arr == 0)).sum()),
            int(((steps_arr == s) & (labels_arr == 1)).sum()),
        )
        for s in q_steps
    )


cell_data: list[dict] = []  # small arrays only; kept in RAM throughout null loop

for subj in tqdm(subjects, desc="subjects"):
    ep_path = Path(epoch_dir) / f"{subj}_epo.fif"
    if not ep_path.exists():
        continue
    ep = mne.read_epochs(str(ep_path), verbose=False)
    ep.metadata = add_metadata_features(ep.metadata)
    ep_bl = ep.copy().apply_baseline((None, 0))  # baseline-correct all trials once
    md = ep.metadata

    subj_rows = ea[ea["subject"] == subj]

    # Group by electrode to extract HGA once per electrode (may appear in multiple pairs)
    for eidx_int, elec_rows in subj_rows.groupby("electrode_idx"):
        eidx = int(eidx_int)
        try:
            hga_all = ep_bl.get_data(picks=[eidx]).squeeze(1)  # (n_all_trials, n_times)
        except Exception as exc:
            print(f"  ERROR: e{eidx} {subj}: {exc}")
            continue

        for _, row in elec_rows.iterrows():
            pp      = row["phoneme_pair"]
            windows = WINDOWS[pp]

            for we_key in ("B1_word_end", "B2_word_end"):
                we      = row[we_key]
                q_steps = ambig_steps.get((subj, pp, we), [])

                mask_pp  = (md["phoneme_pair"] == pp).values
                mask_we  = (md["word_end"] == we).values
                mask_amb = (np.isin(md["resampled"].values, q_steps)
                            if q_steps else np.zeros(len(md), bool))
                mask = mask_pp & mask_we & mask_amb

                rec: dict = dict(
                    subject=subj, electrode_idx=eidx,
                    phoneme_pair=pp, word_end=we, we_key=we_key,
                    q_steps=list(q_steps),
                )

                if not q_steps or not mask.any():
                    rec.update(hga_pw=None, labels=None, steps=None,
                               powered=False, n_per_class=0)
                    cell_data.append(rec)
                    continue

                hga_filt    = hga_all[mask, :]
                steps_filt  = md["resampled"].values[mask].astype(int)
                labels_filt = md["behavior_dummy_forced"].values[mask].astype(int)

                n_per_class = compute_n_per_class(steps_filt, labels_filt, q_steps)
                powered = (
                    len(np.unique(labels_filt)) == 2
                    and n_per_class >= MIN_CLASS_K
                )

                # Per-trial per-window mean HGA (n_trials × n_windows)
                hga_pw = np.column_stack([
                    hga_filt[:, smin:smax].mean(axis=1)
                    for smin, smax in windows
                ])

                rec.update(hga_pw=hga_pw, labels=labels_filt, steps=steps_filt,
                           powered=powered, n_per_class=n_per_class)
                cell_data.append(rec)

    del ep, ep_bl

print(f"\nTotal cells precomputed: {len(cell_data)}")
n_powered = sum(c["powered"] for c in cell_data)
print(f"Powered (≥{MIN_CLASS_K} per class): {n_powered}")
print(f"Underpowered: {len(cell_data) - n_powered}")

# %% [markdown]
# ## Classifier: windowed Welch t-test sweep
#
# Flag a word-end if any analysis window has two-sided Welch t p < ALPHA.
# Vectorized across all windows for performance.

# %%
def is_flagged(hga_pw: np.ndarray, labels: np.ndarray) -> bool:
    """Windowed sweep flag (matches early_perceptual_windows.py logic).

    hga_pw : (n_trials, n_windows) — per-trial per-window mean HGA
    labels : (n_trials,)           — binary report labels (0 or 1)
    Returns True if ANY window has Welch two-sided t-test p < ALPHA.
    """
    mask1 = labels == 1
    mask0 = labels == 0
    n1, n0 = int(mask1.sum()), int(mask0.sum())
    if n1 < 2 or n0 < 2:
        return False

    x1 = hga_pw[mask1, :]   # (n1, n_windows)
    x0 = hga_pw[mask0, :]   # (n0, n_windows)

    m1, m0 = x1.mean(axis=0), x0.mean(axis=0)
    v1 = x1.var(axis=0, ddof=1) / n1
    v0 = x0.var(axis=0, ddof=1) / n0
    se = np.sqrt(v1 + v0)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 1e-15, (m1 - m0) / se, 0.0)

    # Welch degrees of freedom (per window, vectorized)
    num = (v1 + v0) ** 2
    den = v1 ** 2 / max(n1 - 1, 1) + v0 ** 2 / max(n0 - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = np.where(den > 1e-15, num / den, float(min(n1, n0) - 1))
    df = np.maximum(df, 1.0)

    p = 2.0 * stats.t.sf(np.abs(t_stat), df)
    return bool(np.any(p < ALPHA))


def permute_within_steps(labels: np.ndarray, steps: np.ndarray,
                          q_steps: list, rng: np.random.Generator) -> np.ndarray:
    """Permute labels within each qualifying step (preserves per-step class counts)."""
    perm = labels.copy()
    for s in q_steps:
        idx = np.where(steps == s)[0]
        perm[idx] = rng.permutation(perm[idx])
    return perm

# %% [markdown]
# ## Re-derived observed counts (same windowed t-test classifier)

# %%
obs_flag_map: dict[tuple, bool] = {}
for cell in cell_data:
    key = (cell["subject"], cell["electrode_idx"],
           cell["phoneme_pair"], cell["word_end"])
    obs_flag_map[key] = (
        is_flagged(cell["hga_pw"], cell["labels"]) if cell["powered"] else False
    )


def classify_sites(flag_map: dict, rows_df: pd.DataFrame) -> tuple[int, int, int]:
    """Count (two-sided, one-sided, absent) across all site×pair rows."""
    n_two = n_one = n_abs = 0
    for _, row in rows_df.iterrows():
        k1 = (row["subject"], int(row["electrode_idx"]),
               row["phoneme_pair"], row["B1_word_end"])
        k2 = (row["subject"], int(row["electrode_idx"]),
               row["phoneme_pair"], row["B2_word_end"])
        n = int(flag_map.get(k1, False)) + int(flag_map.get(k2, False))
        if n == 2:
            n_two += 1
        elif n == 1:
            n_one += 1
        else:
            n_abs += 1
    return n_two, n_one, n_abs


OBS_TWO, OBS_ONE, OBS_ABS = classify_sites(obs_flag_map, ea)
print("Re-derived observed (windowed t-test on true labels):")
print(f"  Two-sided: {OBS_TWO}, One-sided: {OBS_ONE}, Absent: {OBS_ABS}")
print(f"Original B1/B2_aligned_sig:")
print(f"  Two-sided: {ORIG_TWOSIDED}, One-sided: {ORIG_ONESIDED}, Absent: {ORIG_ABSENT}")

# %% [markdown]
# ## Null permutation loop
#
# For each of M=1000 iterations: permute report labels within each (step × word_end)
# cell (real HGA unchanged) then run the identical windowed t-test sweep.

# %%
rng = np.random.default_rng(RNG_SEED)

null_two    = np.zeros(M_NULL, dtype=int)
null_one    = np.zeros(M_NULL, dtype=int)
null_absent = np.zeros(M_NULL, dtype=int)

# Track per-WE flag rate for sanity check
null_we_flag_counts  = np.zeros(M_NULL, dtype=int)
n_powered_we_total   = n_powered  # denominator

for m in tqdm(range(M_NULL), desc="null iterations"):
    perm_flag_map: dict[tuple, bool] = {}
    n_flagged_this = 0

    for cell in cell_data:
        key = (cell["subject"], cell["electrode_idx"],
               cell["phoneme_pair"], cell["word_end"])
        if not cell["powered"]:
            perm_flag_map[key] = False
            continue

        perm_labels = permute_within_steps(
            cell["labels"], cell["steps"], cell["q_steps"], rng
        )
        flagged = is_flagged(cell["hga_pw"], perm_labels)
        perm_flag_map[key] = flagged
        if flagged:
            n_flagged_this += 1

    n_two, n_one, n_abs = classify_sites(perm_flag_map, ea)
    null_two[m]    = n_two
    null_one[m]    = n_one
    null_absent[m] = n_abs
    null_we_flag_counts[m] = n_flagged_this

null_we_flag_rate = null_we_flag_counts / max(n_powered_we_total, 1)

print(f"\nNull complete (M={M_NULL}):")
print(f"  Two-sided:  mean={null_two.mean():.2f}  ±{null_two.std():.2f}")
print(f"  One-sided:  mean={null_one.mean():.2f}  ±{null_one.std():.2f}")
print(f"  Absent:     mean={null_absent.mean():.2f} ±{null_absent.std():.2f}")
print(f"  Per-WE flag rate: mean={null_we_flag_rate.mean():.4f}")

# %%
# p-values: fraction of null >= observed (re-derived and original)
p_two_red  = float((null_two >= OBS_TWO  ).mean())
p_one_red  = float((null_one >= OBS_ONE  ).mean())
p_two_orig = float((null_two >= ORIG_TWOSIDED).mean())
p_one_orig = float((null_one >= ORIG_ONESIDED).mean())

print("\n=== HEADLINE READOUT ===")
print(f"  Re-derived observed:    {OBS_TWO} two-sided, {OBS_ONE} one-sided, {OBS_ABS} absent")
print(f"  Original B1/B2 obs:     {ORIG_TWOSIDED} two-sided, {ORIG_ONESIDED} one-sided, {ORIG_ABSENT} absent")
print(f"  Null mean:              {null_two.mean():.2f} two-sided, {null_one.mean():.2f} one-sided")
print(f"  Null 95th pct:          {np.percentile(null_two, 95):.1f} two-sided, "
      f"{np.percentile(null_one, 95):.1f} one-sided")
print(f"\n  p (re-derived observed):")
print(f"    two-sided p = {p_two_red:.4f}, one-sided p = {p_one_red:.4f}")
print(f"  p (original B1/B2 observed):")
print(f"    two-sided p = {p_two_orig:.4f}, one-sided p = {p_one_orig:.4f}")

# %% [markdown]
# ## Sanity check: windowed sweep inflates per-WE false-positive rate vs v1 scalar
#
# If the windowed classifier is working correctly, the per-WE flag rate under the
# null must EXCEED v1's implied rate. If it doesn't, the sweep is degenerate.

# %%
v1_1_flag_rate = float(null_we_flag_rate.mean())
print("=== SANITY CHECK ===")
print(f"  v1 implied per-WE rate (scalar AUC null): {V1_IMPLIED_FLAG_RATE:.4f}")
print(f"  v1.1 per-WE rate (windowed t-test null):  {v1_1_flag_rate:.4f}")
if v1_1_flag_rate > V1_IMPLIED_FLAG_RATE:
    print("  PASS — windowed sweep correctly inflates false-positive rate vs v1.")
else:
    print("  FAIL — windowed sweep rate NOT higher than v1 scalar. Investigate!")

# %% [markdown]
# ## Figures

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle(
    "Base-rate test — v1.1 (real HGA, permuted labels, windowed Welch t-test)\n"
    f"Classifier: {WINDOW_SIZE}-sample windows, stride={STRIDE}, α={ALPHA}; "
    "same structure as early_perceptual_windows.py",
    fontsize=9,
)

for ax, (null_arr, obs_n, orig_n, label, color) in zip(axes, [
    (null_two,    OBS_TWO, ORIG_TWOSIDED, "Two-sided",  "#1b7837"),
    (null_one,    OBS_ONE, ORIG_ONESIDED, "One-sided",  "#762a83"),
    (null_absent, OBS_ABS, ORIG_ABSENT,   "Absent",     "#7fbfff"),
]):
    lo = max(0, null_arr.min() - 1)
    hi = max(null_arr.max(), obs_n, orig_n) + 2
    ax.hist(null_arr, bins=range(lo, hi + 1),
            color=color, alpha=0.65, edgecolor="white", lw=0.4)
    ax.axvline(obs_n,  color="crimson", lw=2.2, zorder=5,
               label=f"Re-derived obs (n={obs_n})")
    ax.axvline(orig_n, color="darkorange", lw=1.5, ls="--", zorder=4,
               label=f"B1/B2_aligned_sig (n={orig_n})")
    ax.set_xlabel(f"N {label} sites")
    ax.set_ylabel("Null iterations" if ax is axes[0] else "")
    ax.set_title(f"{label}\nnull mean={null_arr.mean():.1f}, 95th={np.percentile(null_arr, 95):.1f}")
    ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(OUTDIR / "null_distribution.pdf", bbox_inches="tight")
plt.show()
print(f"Saved → {OUTDIR}/null_distribution.pdf")

# %%
fig2, ax2 = plt.subplots(figsize=(6, 5))
hi_one = max(null_one.max(), ORIG_ONESIDED, OBS_ONE) + 2
hi_two = max(null_two.max(), ORIG_TWOSIDED, OBS_TWO) + 2
h = ax2.hist2d(null_one, null_two,
               bins=[range(0, hi_one), range(0, hi_two)],
               cmap="Blues")
plt.colorbar(h[3], ax=ax2, label="Null iterations")
ax2.scatter(OBS_ONE,       OBS_TWO,
            color="crimson", s=140, zorder=5, marker="*",
            label=f"Re-derived ({OBS_ONE}, {OBS_TWO})")
ax2.scatter(ORIG_ONESIDED, ORIG_TWOSIDED,
            color="darkorange", s=90, zorder=4, marker="D",
            label=f"B1/B2_aligned_sig ({ORIG_ONESIDED}, {ORIG_TWOSIDED})")
ax2.set_xlabel("N one-sided sites")
ax2.set_ylabel("N two-sided sites")
ax2.set_title("Joint null (one-sided × two-sided)\nv1.1 windowed t-test")
ax2.legend(fontsize=8)
plt.tight_layout()
fig2.savefig(OUTDIR / "null_2d.pdf", bbox_inches="tight")
plt.show()
print(f"Saved → {OUTDIR}/null_2d.pdf")

# %% [markdown]
# ## Save outputs

# %%
null_df = pd.DataFrame({
    "null_iter":    np.arange(M_NULL),
    "n_two_sided":  null_two,
    "n_one_sided":  null_one,
    "n_absent":     null_absent,
    "we_flag_rate": null_we_flag_rate,
})
null_df.to_parquet(OUTDIR / "null_counts.parquet", index=False)

summary = {
    "obs_two_sided_rederived": OBS_TWO,
    "obs_one_sided_rederived": OBS_ONE,
    "obs_absent_rederived":    OBS_ABS,
    "obs_two_sided_orig":      ORIG_TWOSIDED,
    "obs_one_sided_orig":      ORIG_ONESIDED,
    "obs_absent_orig":         ORIG_ABSENT,
    "null_two_mean":           float(null_two.mean()),
    "null_two_std":            float(null_two.std()),
    "null_two_p95":            float(np.percentile(null_two, 95)),
    "null_one_mean":           float(null_one.mean()),
    "null_one_std":            float(null_one.std()),
    "null_one_p95":            float(np.percentile(null_one, 95)),
    "p_two_rederived":         p_two_red,
    "p_one_rederived":         p_one_red,
    "p_two_orig":              p_two_orig,
    "p_one_orig":              p_one_orig,
    "null_we_flag_rate_mean":  v1_1_flag_rate,
    "v1_implied_flag_rate":    V1_IMPLIED_FLAG_RATE,
    "sanity_pass":             bool(v1_1_flag_rate > V1_IMPLIED_FLAG_RATE),
    "M_NULL":                  M_NULL,
    "ALPHA":                   ALPHA,
    "WINDOW_SIZE":             WINDOW_SIZE,
    "STRIDE":                  STRIDE,
    "MIN_CLASS_K":             MIN_CLASS_K,
    "n_powered_cells":         n_powered,
    "n_total_cells":           len(cell_data),
}
pd.Series(summary).to_csv(OUTDIR / "summary.csv", header=False)

print("\n=== FINAL SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")
