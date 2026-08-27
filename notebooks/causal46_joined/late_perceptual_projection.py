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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import os
import sys
from math import comb
from pathlib import Path

import matplotlib
# matplotlib.use("Agg")
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
from _within_completion import (  # noqa: E402
    extract_hga,
    resolve_behavior_col,
)
from _late_projection import (  # noqa: E402
    compute_a_vector,
    compute_a_vector_null,
    compute_p,
)

# %% tags=["parameters"]
# Computed site-type table (early_window_site_types); consumed ONLY for its
# A_significant column to define the projection site pool. NOT the manual
# type1-5 authority (that is early_acoustic_window.csv, read by the aggregate).
site_pool_path = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv"

early_window_path = "outputs/causal46_joined/early_perceptual_projection/site_class.parquet"

# Individual b4 bootstrap results per window
b4_windows_path = "outputs/causal46_joined/t_tests/b4_per_window.parquet"

# Individual acoustic endpoint bootstrap results per window
a_windows_path = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_full_all.parquet"

# Unified behaviorally discriminative windows
b_windows_path = "outputs/causal46_joined/behavioral_discriminative_windows_all/b_windows.parquet"

epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/late_perceptual_projection"
min_class_k = 3

min_component_windows = 2

# window parameters for the HGA sampling
window_size = 2
stride = 2

n_perms = 50000
master_seed = 42
fdr_alpha = 0.05

# %%
Path(outdir).mkdir(parents=True, exist_ok=True)

# %%
early_window_df = pd.read_parquet(early_window_path)

# %%
b_windows = pd.read_parquet(b_windows_path)

# %%
b4_windows = pd.read_parquet("outputs/causal46_joined/t_tests/b4_per_window.parquet")

# %%
a_windows = pd.read_parquet(a_windows_path)

# %%
# Site pool: A_significant sites (same universe as other causal46_joined analyses).
# A_significant = True means the acoustic searchlight test was significant in early_window_site_types.
site_pool = pd.read_csv(site_pool_path)
included_sites = (
    site_pool[site_pool["A_significant"]]
    [["subject", "electrode_idx", "phoneme_pair"]]
    .reset_index(drop=True)
)

# Retain sites that are identified as type1 or type2
included_sites = pd.merge(
    included_sites,
    early_window_df[["subject", "electrode_idx", "phoneme_pair", "early_response_class"]],
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="left",
)
included_sites = included_sites[included_sites["early_response_class"] != "neither"]

n_total_in_pool = len(site_pool)
print(f"Sites initially in pool: {n_total_in_pool}")
print(f"Sites retained: {len(included_sites)}")
if len(included_sites) > 0:
    print(included_sites[["subject", "electrode_idx", "phoneme_pair"]].to_string(index=False))

# %%
cell_pool = pd.merge(
    included_sites,
    b_windows[b_windows.ci_excludes_zero
              & (b_windows.n_component_windows >= min_component_windows)],
    on=["subject", "electrode_idx", "phoneme_pair"],
    how="left",
    indicator=True
)

cell_pool_counts = cell_pool._merge.value_counts()
assert cell_pool_counts.get("right_only", 0) == 0, f"Some behaviorally significant sites are missing from site pool: {cell_pool_counts}"

print(f"Behaviorally significant sites in pool: {cell_pool_counts.get('both', 0)}")
cell_pool = cell_pool.query("_merge == 'both'").drop(columns="_merge")

# %%
epochs_dict = {}
for p in Path(epoch_dir).glob("*.fif"):
    ep = mne.read_epochs(p, preload=True, verbose=False)
    ep.metadata = add_metadata_features(ep.metadata)
    epochs_dict[p.stem.rstrip("_epo")] = ep

bhv_col = resolve_behavior_col(ep.metadata)


# %% [markdown]
# ## Per-cell projection

# %%
hga_dict = {subject: epochs.copy().apply_baseline((None, 0)).get_data()
            for subject, epochs in epochs_dict.items()}

# %%
results = []

rng = np.random.default_rng(master_seed)

for i, (_, site_group) in enumerate(tqdm(cell_pool.groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"]))):
    site_row = site_group.iloc[0]
    subject = site_row["subject"]
    electrode_idx = site_row["electrode_idx"]
    phoneme_pair = site_row["phoneme_pair"]
    word_end = site_row["word_end"]

    # Filter epochs for this cell
    ep = epochs_dict[subject]
    assert ep.metadata is not None
    ep_i = ep[(ep.metadata["phoneme_pair"] == phoneme_pair)
              & (ep.metadata["word_end"] == word_end)]
    md_i = ep_i.metadata
    assert md_i is not None

    if len(md_i) == 0:
        continue
    hga_i = hga_dict[subject][ep_i.selection, electrode_idx]

    # reuse the same RNG across windows so that
    # label permutation is the same across windows
    rng_i = np.random.default_rng(master_seed + i)

    window_results = []
    for _, site_window in site_group.iterrows():
        smin, smax = int(site_window.smin), int(site_window.smax)

        # Compute perceptual template
        p_vec, min_classes, per_step_filtered, N, p_traces = compute_p(
            hga=hga_i,
            md_pp=md_i,
            word_end=word_end,
            group_col=bhv_col,
            smin=smin,
            smax=smax,
            window_size=window_size,
            stride=stride,
            K=min_class_k
        )
        if p_vec is None:
            print(f"No qualifying steps for {subject} {electrode_idx} {phoneme_pair} {word_end}")
            continue

        # Compute acoustic vector
        a_vec = compute_a_vector(
            hga_i, md_i, smin, smax,
            window_size=window_size,
            stride=stride
        )

        projection = np.dot(p_vec, a_vec)

        # Compute null distribution of projection
        a_null = compute_a_vector_null(
            hga_i, md_i, smin, smax,
            window_size=window_size,
            stride=stride,
            n_perms=n_perms,
            rng=rng_i,
        )
        projection_null = np.dot(p_vec, a_null.T)
        projection_null_ci_low, projection_null_ci_high = \
            np.percentile(projection_null, [0, 95])

        window_results.append({
            "smin": smin,
            "smax": smax,
            "projection": projection,
            "projection_null": projection_null,
            "n_subwindows": len(p_vec),

            "beta_ambig_mean": site_window.beta_ambig_mean,
            "beta_ambig_median": site_window.beta_ambig_median,
        })

    # (n_windows,)
    all_projection = np.array([wr["projection"] for wr in window_results])
    # (n_perms, n_windows)
    all_projection_null = np.concatenate([wr["projection_null"][:, np.newaxis] for wr in window_results], axis=1)

    # Aggregate max-over-window statistics in observed and null
    stat_obs = all_projection.max()
    stat_null = all_projection_null.max(axis=1)
    pval = (np.sum(stat_null >= stat_obs) + 1) / (n_perms + 1)

    obs_best_window_id = np.argmax(all_projection)
    obs_best_window = window_results[obs_best_window_id]

    stat_null_ci_low, stat_null_ci_high = -np.inf, np.percentile(stat_null, 95)

    # pval (before max-over-window correction)
    # pval = (np.sum(projection_null >= projection) + 1) / (n_perms + 1)

    results.append({
        "subject": subject,
        "electrode_idx": electrode_idx,
        "phoneme_pair": phoneme_pair,
        "word_end": word_end,

        # properties of the site (shared across windows)
        "n_per_class": site_row.n_per_class,
        "acoustic_peak_auc": site_row.acoustic_peak_auc,
        "phon_smin": site_row.phon_smin,
        "phon_smax": site_row.phon_smax,

        # properties of the best window
        "window_id": obs_best_window_id,
        **{k: obs_best_window[k] for k in ["smin", "smax", "n_subwindows", "projection",
                                           "beta_ambig_mean", "beta_ambig_median"]},

        "projection_null_mean": stat_null.mean(),
        "projection_null_ci_low": stat_null_ci_low,
        "projection_null_ci_high": stat_null_ci_high,
        "projection_p_value": pval,
    })


results_df = pd.DataFrame(results)
print(f"\nTotal sites processed: {len(results_df)}")

# %%
p_sig, q, _, _ = multipletests(
    results_df["projection_p_value"],
    alpha=fdr_alpha,
    method="fdr_bh",
    is_sorted=False,
    returnsorted=False
)
results_df["projection_significant"] = p_sig
results_df["projection_significant_uncorrected"] = results_df["projection_p_value"] < fdr_alpha
results_df["projection_q_value"] = q

results_df["projection_significant_ci"] = (
    results_df.projection > results_df.projection_null_ci_high
)

results_df["tmin"] = results_df["smin"] / epoch_sfreq + epoch_tmin
results_df["tmax"] = results_df["smax"] / epoch_sfreq + epoch_tmin

# %%
results_df.to_csv(Path(outdir) / "results.csv", index=False)

# %%
results_df.sort_values("projection_q_value").head(20)

# %%
(results_df.projection > results_df.projection_null_ci_high).sum()

# %%
(results_df.projection > results_df.projection_null_ci_high).mean()

# %%
xs = ep_i.times

for step in [1, 6]:
    step_mask = md_i["resampled"] == step
    plt.plot(xs, hga_i[step_mask].mean(axis=0), label=f"Step {step}")

plt.xlim(-0.05, 0.8)
ax = plt.gca()
ax.axvspan(site_row.smin / 100 - 0.4, site_row.smax / 100 - 0.4,
           alpha=0.3)
ax.legend()

# %%
plt.plot(xs, p_traces[0], label="Perceptual class 0", color="blue")
plt.plot(xs, p_traces[1], label="Perceptual class 1", color="orange")
plt.xlim(-0.05, 0.8)
plt.axvspan(site_row.smin / 100 - 0.4, site_row.smax / 100 - 0.4,
              alpha=0.3)
