# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Speech-responsive screen — causal6
#
# Replaces causal5's screen, which used a paired t-test on the **full** post-onset
# window `[0, tmax]` and a one-sided threshold `t > 7`. That criterion has two
# documented failure modes (see `notebooks/causal6/cross_check_decoders.py`
# section 0 and `scripts/refined_speech_responsive.py`):
#
#   1. The post-window is ~3s, so transient acoustic responses (~150–250ms peaks)
#      get diluted to a few percent of their peak in the post-mean. This is the
#      population the paper is built on; the wide-window t-test silently drops
#      ~10% of the headline acoustic-decoder set (7 of 64 sites at threshold 0.65).
#   2. `t > 7` is one-sided, so suppression sites are silently dropped.
#
# This screen uses a paired t-test (post − pre per trial, paired across trials)
# but with:
#   - post-window restricted to `[0, post_tmax_s]` (default 0.6s) — covers the
#     acoustic + early perceptual response window without diluting it.
#   - two-sided threshold `|t| > t_threshold` — catches suppression.
#
# CSV schema is preserved from causal5 so downstream callers keep working:
#   - `speech_responsive`              : final boolean (|t_short| > t_threshold)
#   - `speech_responsive_test_value`   : max|baselined evoked| in [0, 0.9s]  (causal4-style amp; legacy diagnostic)
#   - `speech_responsive_tval`         : the t statistic used by the new criterion (short-window paired t)
#   - `speech_responsive_pval`         : its two-sided p-value
#   - `speech_responsive_ttest`        : same as `speech_responsive` (|t_short| > t_threshold)
#
# Plus two diagnostic columns for the cross-check notebook:
#   - `speech_responsive_t_full`       : t at the legacy [0, tmax] post-window
#   - `speech_responsive_post_tmax_s`  : the post-window used (so config sweeps are auditable per-row)

# %%
import re

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.data import get_electrode_df

# %% tags=["parameters"]
epochs_path = "outputs/epochs_preprocessed/EC260_epo.fif"
outdir = "."

# Refined-abs criterion: paired t-test on a short post-window, two-sided.
post_tmax_s = 0.6     # post-onset window for the t-test, in seconds
t_threshold = 7.0     # |t| threshold (≈ p < 1e-10 at typical n_trials)

# Legacy amplitude diagnostic (causal4 criterion). Kept as a column for the
# cross-check; not used in the final boolean.
amp_threshold = 0.3
amp_tmax_s = 0.9

# %%
subject_name = re.findall(r"(EC[\d]+)_epo", epochs_path)[0]
print(f"subject={subject_name}  post_tmax_s={post_tmax_s}  t_threshold={t_threshold}")

# %%
epochs = mne.read_epochs(epochs_path)
print(f"epoch span: tmin={epochs.tmin}  tmax={epochs.tmax}  sfreq={epochs.info['sfreq']}")

# %%
electrode_df = get_electrode_df(subject_name)
electrode_df["roi"] = electrode_df.roi.astype(str)
electrode_df = electrode_df.droplevel("electrode_name")

# Drop electrode metadata rows that have no corresponding data channel.
electrode_df["keep"] = np.arange(len(electrode_df)) < len(epochs.info["ch_names"])
electrode_df = electrode_df[electrode_df["keep"]].drop(columns="keep")
n_chans = len(electrode_df)

# %% [markdown]
# ## Legacy amplitude diagnostic
#
# Computes max|baselined evoked| in [0, 0.9s] per channel. Stored as
# `speech_responsive_test_value` for cross-pipeline diagnostic continuity with
# causal4 / causal5 — not used in the final boolean.

# %%
evoked_data = (
    epochs.copy()
    .apply_baseline((-0.1, 0))
    .average()
    .crop(tmin=0, tmax=amp_tmax_s)
    .get_data()
)
amp_value = np.abs(evoked_data).max(axis=1)[:n_chans]
amp_flag = amp_value > amp_threshold

# %% [markdown]
# ## Refined paired t-test
#
# Per trial: post-mean over [0, post_tmax_s] vs pre-mean over [tmin, 0].
# `ttest_rel` pairs them across trials, so the test is on the trial-level
# (post − pre) distribution. Two-sided.
#
# `speech_responsive_t_full` (the long-window t) is also computed for the
# cross-check, so we can see per-row how much the wide post-window dilutes
# things.

# %%
data = epochs.get_data()  # (n_epochs, n_chans, n_times)

s_pre_lo,    s_pre_hi    = epochs.time_as_index([epochs.tmin, 0])
s_full_lo,   s_full_hi   = epochs.time_as_index([0, epochs.tmax])
s_short_lo,  s_short_hi  = epochs.time_as_index([0, post_tmax_s])

pre_means        = data[:, :, s_pre_lo:s_pre_hi].mean(axis=2)
post_means_full  = data[:, :, s_full_lo:s_full_hi].mean(axis=2)
post_means_short = data[:, :, s_short_lo:s_short_hi].mean(axis=2)

t_full,  _       = ttest_rel(post_means_full,  pre_means, axis=0)
t_short, p_short = ttest_rel(post_means_short, pre_means, axis=0)

t_full        = t_full[:n_chans]
t_short       = t_short[:n_chans]
p_short       = p_short[:n_chans]

speech_responsive = np.abs(t_short) > t_threshold

# %% [markdown]
# ## Diagnostics

# %%
print(f"  amp criterion (causal4-style):    {amp_flag.sum()}  / {n_chans}  ({amp_flag.mean():.1%})")
print(f"  full-window t > {t_threshold}: {(t_full > t_threshold).sum()}  / {n_chans}  ({(t_full > t_threshold).mean():.1%})")
print(f"  refined  |t| > {t_threshold} (used): {speech_responsive.sum()}  / {n_chans}  ({speech_responsive.mean():.1%})")
n_supp = ((t_short < -t_threshold)).sum()
print(f"    of which suppression-like (t_short < -{t_threshold}): {n_supp}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(t_full, t_short, s=10, alpha=0.5)
lim = max(np.nanmax(np.abs(t_full)), np.nanmax(np.abs(t_short))) * 1.05
axes[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
for thr in (-t_threshold, t_threshold):
    axes[0].axhline(thr, color="r", lw=0.4)
    axes[0].axvline(thr, color="r", lw=0.4)
axes[0].set_xlabel("t (full post-window [0, tmax])")
axes[0].set_ylabel(f"t (short post-window [0, {post_tmax_s}s])")
axes[0].set_title(f"{subject_name}: window-length effect on paired t")

axes[1].scatter(amp_value, t_short, s=10, alpha=0.5)
axes[1].axvline(amp_threshold, color="r", lw=0.4)
axes[1].axhline(t_threshold, color="r", lw=0.4)
axes[1].axhline(-t_threshold, color="r", lw=0.4)
axes[1].set_xlabel("max|baselined evoked| in [0, 0.9s]")
axes[1].set_ylabel(f"t (short post-window [0, {post_tmax_s}s])")
axes[1].set_title(f"{subject_name}: refined t vs causal4 amp")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Write CSV

# %%
electrode_df = electrode_df.assign(
    speech_responsive=speech_responsive.astype(bool),
    speech_responsive_test_value=amp_value,
    speech_responsive_tval=t_short,
    speech_responsive_pval=p_short,
    speech_responsive_ttest=speech_responsive.astype(bool),
    speech_responsive_t_full=t_full,
    speech_responsive_post_tmax_s=float(post_tmax_s),
    subject=subject_name,
)

# %%
electrode_df.to_csv(f"{outdir}/{subject_name}_results.csv")
print(f"wrote {outdir}/{subject_name}_results.csv  ({n_chans} channels)")
