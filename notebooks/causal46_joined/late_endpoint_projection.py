# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Late perceptual projection, gated on ENDPOINT persistence
#
# Prototype variant of `late_perceptual_projection.py`. **Only the gate changes.**
#
# - Current late projection gates each cell on a *behavioral* discriminative run
#   (`b_windows`, the ambiguous-trial /n/-/d/ contrast) — the very statistic that
#   sits at chance at the population level, and selection on the same `p` the
#   projection then tests.
# - This variant gates instead on a **late endpoint run**: a post-acoustic
#   (`smin >= phon_smax`) maximal run of significant step6-step1 (endpoint,
#   unambiguous) HGA windows. Post-POD the two endpoints are acoustically
#   identical within a completion, so a late endpoint contrast is *neural
#   persistence of the acoustic code*, not a stimulus difference. Endpoints have
#   many trials → a cleaner, lower-noise acoustic template `â`.
#
# This makes late parallel to the early perceptual projection: pool = acoustic-
# reliable sites (`A_significant`), gate = endpoint reliability (not middle
# trials), projection `π = ⟨p, â⟩` in the gated window, acoustic-label-shuffle
# null. The ONLY differences from `late_perceptual_projection.py` are (a) the
# window source and (b) the site pool is not further restricted to type1/2.
#
# The projection machinery (raw `⟨p, â⟩`, per-window null, max-over-window,
# BH-FDR) is byte-for-byte the current late one, so results are directly
# comparable — the gate is the single manipulated variable.
#
# NOTE: not yet wired into the Snakefile — run manually. Scratch site-set
# search (no epochs) lives in the plan doc; here we run the full projection.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

from src.data import add_metadata_features
from src.stimuli import PHONEME_PAIR_TO_WORD_ENDS
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import resolve_behavior_col  # noqa: E402
from _windows import _find_maximal_runs  # noqa: E402
from _acoustic_offset import find_early_offset_smin  # noqa: E402
from _late_projection import (  # noqa: E402
    compute_a_vector,
    compute_a_vector_null,
    compute_p,
)

# %% tags=["parameters"]
# Endpoint per-window bootstrap over the FULL epoch range (step6 - step1,
# unambiguous). Source of the late endpoint runs. Per word_end and full word
# timecourse, so the late endpoint window may differ between the two completions
# (post-POD the two words diverge acoustically).
a_per_window_by_word_end_path = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_by_word_end_all.parquet"

# Computed site-type table; consumed ONLY for its A_significant column (site pool)
# and phon_smax (post-acoustic boundary).
site_pool_path = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv"

# Endpoint-window table: source of phon_smax per site.
a_windows_path = "outputs/causal46_joined/acoustic_endpoint_windows/a_windows.parquet"

epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/late_endpoint_projection"

min_class_k = 3
min_component_windows = 2  # a run must span >= this many endpoint windows

window_size = 2
stride = 2

n_perms = 50000
master_seed = 42
fdr_alpha = 0.05

# %%
Path(outdir).mkdir(parents=True, exist_ok=True)

KEY = ["subject", "electrode_idx", "phoneme_pair"]
SF, T0 = epoch_sfreq, epoch_tmin

# %% [markdown]
# ## Build the late-endpoint gate  (per word_end)
#
# Per A_significant site **x word_end**: significant endpoint windows unified into
# maximal same-sign runs (same `_find_maximal_runs` as the discriminative-window
# notebooks). Each run of >= `min_component_windows` becomes a projection window.
# The endpoint bootstrap is read per word_end (`a_per_window_by_word_end_all`), so
# the two completions can carry **different** late endpoint windows — post-POD
# they diverge acoustically, and the within-completion projection is per word_end
# anyway.
#
# The run-candidate floor `smin` is set **per cell** to the *disappearance of the
# initial acoustic response*: `find_early_offset_smin` (shared with
# `acoustic_late.py`) returns the first window at/after `phon_smax` where the
# endpoint contrast has returned to non-significance. A late endpoint run must
# therefore be a **re-emergence** of the endpoint contrast after the initial
# response has diminished — not the sustained tail of a multi-phase early
# response. `phon_smax` is per-site (pre-lexical); the offset is recomputed per
# word_end. Cells whose contrast never returns to non-significance past
# `phon_smax` (no dissociable late region) are dropped from the gate and reported.

# %%
a_per_window = pd.read_parquet(a_per_window_by_word_end_path)
a_windows = pd.read_parquet(a_windows_path)
site_pool = pd.read_csv(site_pool_path)

phon = a_windows[KEY + ["phon_smax"]].drop_duplicates()
included_sites = (
    site_pool[site_pool["A_significant"]][KEY].drop_duplicates()
    .merge(phon, on=KEY, how="left")
)
n_missing_phon = int(included_sites["phon_smax"].isna().sum())
print(f"A_significant sites: {len(included_sites)}  (phon_smax missing: {n_missing_phon})")

# %%
run_rows = []
no_offset_cells = []
for _, r in included_sites.iterrows():
    if pd.isna(r["phon_smax"]):
        continue
    for word_end in PHONEME_PAIR_TO_WORD_ENDS[r["phoneme_pair"]]:
        sub = a_per_window[(a_per_window.subject == r.subject)
                           & (a_per_window.electrode_idx == r.electrode_idx)
                           & (a_per_window.phoneme_pair == r.phoneme_pair)
                           & (a_per_window.word_end == word_end)].sort_values("smin")
        if sub.empty:
            continue
        # Per-cell floor: disappearance of the initial acoustic response.
        s_early_offset = find_early_offset_smin(sub, int(r["phon_smax"]))
        if s_early_offset is None:
            # Sustained endpoint contrast: no dissociable late region.
            no_offset_cells.append({**{k: r[k] for k in KEY}, "word_end": word_end})
            continue
        cand = sub[sub.smin >= s_early_offset]
        sig = cand[cand.ci_raw_excludes_zero]
        if len(sig) == 0:
            continue
        sig_windows = [(int(a), int(b)) for a, b in zip(sig.smin, sig.smax)]
        medians = {int(a): float(m) for a, m in zip(cand.smin, cand.mean_diff_raw_med)}
        for run in _find_maximal_runs(sig_windows, medians):
            if len(run) < min_component_windows:
                continue
            run_rows.append({
                "subject": r.subject, "electrode_idx": r.electrode_idx,
                "phoneme_pair": r.phoneme_pair, "word_end": word_end,
                "s_early_offset": s_early_offset,
                "smin": run[0][0], "smax": run[-1][1],
                "n_component_windows": len(run),
                "beta_endpoint_median": float(np.mean([medians[a] for a, _ in run])),
                "sign": int(np.sign(medians[run[0][0]])),
            })

cell_pool = pd.DataFrame(run_rows)
CELL_KEY = KEY + ["word_end"]
gated_cells = cell_pool[CELL_KEY].drop_duplicates()
print(f"Cells with no dissociable late region (dropped): {len(no_offset_cells)}")
print(f"Gated cells (>=1 late endpoint run): {len(gated_cells)}")
print(f"Gated sites: {cell_pool[KEY].drop_duplicates().shape[0]}")
print(f"Total runs: {len(cell_pool)}")

# %% [markdown]
# ## Per-cell projection  (identical machinery to late_perceptual_projection.py)

# %%
epochs_dict = {}
for p in Path(epoch_dir).glob("*.fif"):
    ep = mne.read_epochs(p, preload=True, verbose=False)
    ep.metadata = add_metadata_features(ep.metadata)
    epochs_dict[p.stem.rstrip("_epo")] = ep

bhv_col = resolve_behavior_col(ep.metadata)

hga_dict = {subject: epochs.copy().apply_baseline((None, 0)).get_data()
            for subject, epochs in epochs_dict.items()}

# %%
results = []
rng = np.random.default_rng(master_seed)

for i, (_, site_group) in enumerate(tqdm(
        cell_pool.groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"]))):
    site_row = site_group.iloc[0]
    subject = site_row["subject"]
    electrode_idx = site_row["electrode_idx"]
    phoneme_pair = site_row["phoneme_pair"]
    word_end = site_row["word_end"]

    ep = epochs_dict[subject]
    assert ep.metadata is not None
    ep_i = ep[(ep.metadata["phoneme_pair"] == phoneme_pair)
              & (ep.metadata["word_end"] == word_end)]
    md_i = ep_i.metadata
    if md_i is None or len(md_i) == 0:
        continue
    hga_i = hga_dict[subject][ep_i.selection, electrode_idx]

    rng_i = np.random.default_rng(master_seed + i)

    window_results = []
    for _, site_window in site_group.iterrows():
        smin, smax = int(site_window.smin), int(site_window.smax)

        p_vec, min_classes, per_step_filtered, N, p_traces = compute_p(
            hga=hga_i, md_pp=md_i, word_end=word_end, group_col=bhv_col,
            smin=smin, smax=smax, window_size=window_size, stride=stride,
            K=min_class_k,
        )
        if p_vec is None:
            continue

        a_vec = compute_a_vector(hga_i, md_i, smin, smax,
                                 window_size=window_size, stride=stride)
        projection = np.dot(p_vec, a_vec)

        a_null = compute_a_vector_null(hga_i, md_i, smin, smax,
                                       window_size=window_size, stride=stride,
                                       n_perms=n_perms, rng=rng_i)
        projection_null = np.dot(p_vec, a_null.T)

        window_results.append({
            "smin": smin, "smax": smax,
            "projection": projection, "projection_null": projection_null,
            "n_subwindows": len(p_vec),
            "beta_endpoint_median": site_window.beta_endpoint_median,
            "n_component_windows": site_window.n_component_windows,
        })

    if not window_results:
        continue

    all_projection = np.array([wr["projection"] for wr in window_results])
    all_projection_null = np.concatenate(
        [wr["projection_null"][:, np.newaxis] for wr in window_results], axis=1)

    stat_obs = all_projection.max()
    stat_null = all_projection_null.max(axis=1)
    pval = (np.sum(stat_null >= stat_obs) + 1) / (n_perms + 1)

    obs_best_window_id = int(np.argmax(all_projection))
    obs_best_window = window_results[obs_best_window_id]
    stat_null_ci_high = np.percentile(stat_null, 95)

    results.append({
        "subject": subject, "electrode_idx": electrode_idx,
        "phoneme_pair": phoneme_pair, "word_end": word_end,
        "window_id": obs_best_window_id,
        **{k: obs_best_window[k] for k in
           ["smin", "smax", "n_subwindows", "projection",
            "beta_endpoint_median", "n_component_windows"]},
        "projection_null_mean": stat_null.mean(),
        "projection_null_ci_low": -np.inf,
        "projection_null_ci_high": stat_null_ci_high,
        "projection_p_value": pval,
    })

results_df = pd.DataFrame(results)
print(f"\nCells with a testable projection: {len(results_df)}")

# %%
p_sig, q, _, _ = multipletests(results_df["projection_p_value"], alpha=fdr_alpha,
                               method="fdr_bh")
results_df["projection_significant"] = p_sig
results_df["projection_significant_uncorrected"] = results_df["projection_p_value"] < fdr_alpha
results_df["projection_q_value"] = q
results_df["projection_significant_ci"] = (
    results_df.projection > results_df.projection_null_ci_high)
results_df["tmin"] = results_df["smin"] / SF + T0
results_df["tmax"] = results_df["smax"] / SF + T0

results_df.to_csv(Path(outdir) / "results.csv", index=False)
cell_pool.to_parquet(Path(outdir) / "late_endpoint_runs.parquet", index=False)

# %%
print("Cells tested                :", len(results_df))
print("Sites tested                :", results_df[KEY].drop_duplicates().shape[0])
print("Uncorrected projection sig  :", int(results_df.projection_significant_uncorrected.sum()))
print("CI projection sig           :", int(results_df.projection_significant_ci.sum()))
print("BH-FDR projection sig       :", int(results_df.projection_significant.sum()))
results_df.sort_values("projection_q_value").head(20)
