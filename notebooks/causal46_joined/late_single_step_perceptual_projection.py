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
# # Single-step late-perceptual projection gate
#
# `late_perceptual_projection.py` calls a site "late-perceptual" via a projection
# gate whose within-completion percept contrast (`compute_p`) pools over ALL
# qualifying ambiguous steps. This notebook asks: how many of those same B4 cells
# still pass the SAME gate if the percept contrast is restricted to just ONE
# step — the most ambiguous one (report proportion nearest 50/50)?
#
# Reuses the exact `b_windows`/cell pool from `late_perceptual_projection.py`
# (Path A per the design doc); the only change is `compute_p(..., restrict_steps=[s])`.
# See `docs/superpowers/plans/2026-08-27-single-step-perceptual-projection.md`.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

from src.data import add_metadata_features
from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    per_step_class_counts,
    resolve_behavior_col,
)
from _late_projection import (  # noqa: E402
    compute_a_vector,
    compute_a_vector_null,
    compute_p,
    get_qualifying_steps,
)

# %% tags=["parameters"]
# Input paths identical to late_perceptual_projection.py — same cell pool, same
# windows (Path A: reuse B4 unchanged, only compute_p's step restriction differs).
site_pool_path = "outputs/causal46_joined/early_window_site_types/site_type_relabel.csv"
early_window_path = "outputs/causal46_joined/early_perceptual_projection/site_class.parquet"
b_windows_path = "outputs/causal46_joined/behavioral_discriminative_windows_all/b_windows.parquet"

# Reference: the B4 gate this notebook is paired against.
late_perceptual_projection_results_path = "outputs/causal46_joined/late_perceptual_projection/results.csv"

# Early perceptual projection per-site results (plot_for_paper's epp_path) —
# used for the early x late cross-check at the end of this notebook.
epp_path = "outputs/causal46_joined/early_perceptual_projection/all_sites.csv"

epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/single_step_perceptual_projection"
min_class_k = 3

min_component_windows = 2

# Window parameters for the HGA sampling. NOT the notebook-default 2/2 —
# late_perceptual_projection is actually run by the Snakefile with 1/1
# (window_size/stride are overridden there, independent of the general
# causal46_joined config). This notebook reuses the same b_windows and must
# tile them identically to the B4 run it's paired against, so 1/1 here too.
window_size = 1
stride = 1

n_perms = 50000
master_seed = 42
fdr_alpha = 0.05

# %%
Path(outdir).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load cell pool + windows + epochs
#
# Byte-identical to `late_perceptual_projection.py`'s setup — same site pool,
# same `A_significant`/type1-2 filter, same `b_windows` join.

# %%
early_window_df = pd.read_parquet(early_window_path)

# %%
b_windows = pd.read_parquet(b_windows_path)

# %%
site_pool = pd.read_csv(site_pool_path)
included_sites = (
    site_pool[site_pool["A_significant"]]
    [["subject", "electrode_idx", "phoneme_pair"]]
    .reset_index(drop=True)
)

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

# %%
hga_dict = {subject: epochs.copy().apply_baseline((None, 0)).get_data()
            for subject, epochs in epochs_dict.items()}

# %% [markdown]
# ## Most-ambiguous step per (subject, phoneme_pair, word_end)
#
# Behavior-only selection — uses report proportions, never HGA (no double-dipping).
# `word_end`-level metadata doesn't depend on `electrode_idx`, so this is computed
# once per (subject, phoneme_pair, word_end), not once per cell.

# %%
most_ambiguous = {}

for (subject, phoneme_pair, word_end), _ in cell_pool.groupby(
    ["subject", "phoneme_pair", "word_end"]
):
    md = epochs_dict[subject].metadata
    mask = (md["phoneme_pair"] == phoneme_pair) & (md["word_end"] == word_end)
    md_i = md[mask].reset_index(drop=True)

    qualifying = get_qualifying_steps(md_i, word_end=word_end, group_col=bhv_col)
    if not qualifying:
        continue

    per_step = per_step_class_counts(
        md_i, word_end=word_end, qualifying_steps=qualifying, group_col=bhv_col
    )

    best_step, best_metric, best_min_class = None, -1.0, -1
    for s in qualifying:
        by_cls = per_step[s]
        if 0 not in by_cls or 1 not in by_cls:
            continue
        n0, n1 = len(by_cls[0]), len(by_cls[1])
        if n0 + n1 == 0:
            continue
        p1 = n1 / (n0 + n1)
        metric = p1 * (1 - p1)
        min_class = min(n0, n1)
        if (metric > best_metric) or (metric == best_metric and min_class > best_min_class):
            best_step, best_metric, best_min_class = s, metric, min_class

    if best_step is not None:
        most_ambiguous[(subject, phoneme_pair, word_end)] = best_step

print(f"Most-ambiguous step assigned for {len(most_ambiguous)} (subject, phoneme_pair, word_end) groups")

# %% [markdown]
# ## Landscape: per-cell, per-qualifying-step projection gate
#
# Same per-window loop as `late_perceptual_projection.py`, wrapped in an extra
# loop over each cell's qualifying steps, with `compute_p(..., restrict_steps=[s])`.
# One row per (cell, step). Untestable (step, cell) combos — where the K-gate
# leaves no trials once restricted to a single step — are recorded with
# `testable=False`.

# %%
landscape = []

for i, (cell_key, site_group) in enumerate(
    tqdm(cell_pool.groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"]))
):
    site_row = site_group.iloc[0]
    subject, electrode_idx, phoneme_pair, word_end = cell_key

    ep = epochs_dict[subject]
    assert ep.metadata is not None
    ep_i = ep[(ep.metadata["phoneme_pair"] == phoneme_pair)
              & (ep.metadata["word_end"] == word_end)]
    md_i = ep_i.metadata
    assert md_i is not None
    if len(md_i) == 0:
        continue
    hga_i = hga_dict[subject][ep_i.selection, electrode_idx]

    qualifying_steps = get_qualifying_steps(md_i, word_end=word_end, group_col=bhv_col)
    ma_step = most_ambiguous.get((subject, phoneme_pair, word_end))

    for s in qualifying_steps:
        # Distinct, deterministic RNG per (cell, step) — reused across that
        # step's windows so label permutation is the same across windows,
        # same convention as late_perceptual_projection.py's per-cell rng_i.
        rng_i_s = np.random.default_rng(master_seed + i * 1000 + s)

        window_results = []
        testable = True
        for _, site_window in site_group.iterrows():
            smin, smax = int(site_window.smin), int(site_window.smax)

            p_vec, min_classes, per_step_filtered, N, p_traces = compute_p(
                hga=hga_i,
                md_pp=md_i,
                word_end=word_end,
                group_col=bhv_col,
                smin=smin,
                smax=smax,
                window_size=window_size,
                stride=stride,
                K=min_class_k,
                restrict_steps=[s],
            )
            if p_vec is None:
                testable = False
                break

            a_vec = compute_a_vector(
                hga_i, md_i, smin, smax,
                window_size=window_size,
                stride=stride,
            )
            projection = np.dot(p_vec, a_vec)

            a_null = compute_a_vector_null(
                hga_i, md_i, smin, smax,
                window_size=window_size,
                stride=stride,
                n_perms=n_perms,
                rng=rng_i_s,
            )
            projection_null = np.dot(p_vec, a_null.T)

            window_results.append({
                "smin": smin,
                "smax": smax,
                "projection": projection,
                "projection_null": projection_null,
                "n_subwindows": len(p_vec),
                "n_trials": N,
            })

        if not testable or not window_results:
            landscape.append({
                "subject": subject,
                "electrode_idx": electrode_idx,
                "phoneme_pair": phoneme_pair,
                "word_end": word_end,
                "step": s,
                "is_most_ambiguous": s == ma_step,
                "testable": False,
            })
            continue

        all_projection = np.array([wr["projection"] for wr in window_results])
        all_projection_null = np.concatenate(
            [wr["projection_null"][:, np.newaxis] for wr in window_results], axis=1
        )

        stat_obs = all_projection.max()
        stat_null = all_projection_null.max(axis=1)
        pval = (np.sum(stat_null >= stat_obs) + 1) / (n_perms + 1)

        obs_best_window_id = np.argmax(all_projection)
        obs_best_window = window_results[obs_best_window_id]
        stat_null_ci_high = np.percentile(stat_null, 95)

        landscape.append({
            "subject": subject,
            "electrode_idx": electrode_idx,
            "phoneme_pair": phoneme_pair,
            "word_end": word_end,
            "step": s,
            "is_most_ambiguous": s == ma_step,
            "testable": True,

            "n_per_class": site_row.n_per_class,
            "acoustic_peak_auc": site_row.acoustic_peak_auc,
            "phon_smin": site_row.phon_smin,
            "phon_smax": site_row.phon_smax,

            "window_id": obs_best_window_id,
            "smin": obs_best_window["smin"],
            "smax": obs_best_window["smax"],
            "n_subwindows": obs_best_window["n_subwindows"],
            "n_trials": obs_best_window["n_trials"],
            "projection": obs_best_window["projection"],

            "projection_null_mean": stat_null.mean(),
            "projection_null_ci_high": stat_null_ci_high,
            "projection_p_value": pval,
        })

landscape_df = pd.DataFrame(landscape)
landscape_df["projection_significant_uncorrected"] = landscape_df["projection_p_value"] < fdr_alpha
landscape_df["tmin"] = landscape_df["smin"] / epoch_sfreq + epoch_tmin
landscape_df["tmax"] = landscape_df["smax"] / epoch_sfreq + epoch_tmin

print(f"\nLandscape rows (cell × step): {len(landscape_df)}")
print(f"Testable: {landscape_df['testable'].sum()} / {len(landscape_df)}")

# %%
landscape_df.to_parquet(Path(outdir) / "single_step_projection_landscape.parquet", index=False)

# %% [markdown]
# ## Gate + headline
#
# Filter to `is_most_ambiguous`. `sig_uncorrected = p < 0.05`, no FDR — matches
# the flag `plot_for_paper` sums into `late_category`
# (`projection_significant_uncorrected`).

# %%
ma_rows = landscape_df[landscape_df["is_most_ambiguous"]].copy()

n_untestable = int((~ma_rows["testable"]).sum())
ma_testable = ma_rows[ma_rows["testable"]]
n_testable = len(ma_testable)
n_pass = int(ma_testable["projection_significant_uncorrected"].sum())

print("=== Cell-level headline (most-ambiguous step only) ===")
print(f"n_pass / n_testable = {n_pass} / {n_testable}")
print(f"n_untestable (most-ambiguous step failed the K-gate) = {n_untestable}")
# Expected to be 0 at the current config: `most_ambiguous` is chosen among
# get_qualifying_steps (minority class > 2, i.e. >= 3), and compute_p's K-gate
# is min_class_k=3 (>= 3) — identical thresholds, so a step qualifying as
# most-ambiguous always also clears the K-gate. n_untestable > 0 would only
# arise if min_class_k were raised above ambiguous_threshold + 1.

# %%
b4_results = pd.read_csv(late_perceptual_projection_results_path)
CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

paired = pd.merge(
    ma_testable[CELL_KEYS + ["projection_significant_uncorrected"]]
        .rename(columns={"projection_significant_uncorrected": "single_step_sig"}),
    b4_results[CELL_KEYS + ["projection_significant_uncorrected"]]
        .rename(columns={"projection_significant_uncorrected": "b4_sig"}),
    on=CELL_KEYS,
    how="inner",
)
assert len(paired) == n_testable, "paired join should cover every testable most-ambiguous cell"

print("\n=== Paired 2x2 (B4 pass/fail x single-step pass/fail), testable cells only ===")
print(pd.crosstab(paired["b4_sig"], paired["single_step_sig"],
                   rownames=["B4 sig"], colnames=["single-step sig"]))

# %% [markdown]
# ### Plot

# %%
from src._star_gallery import matched_n_star_plot_paper
from src.viz_paper import resampled_cmap

# %%
# cells which are sig in both b4 and single-step
_to_plot = paired.query("b4_sig & single_step_sig")
_to_plot = pd.merge(
    _to_plot,
    ma_testable,
    on=CELL_KEYS,
    how="left",).sort_values("projection_p_value")
for _, row in tqdm(_to_plot.iterrows(), total=len(_to_plot)):
    subject = row["subject"]
    electrode_idx = row["electrode_idx"]
    phoneme_pair = row["phoneme_pair"]
    word_end = row["word_end"]
    f = matched_n_star_plot_paper(
        subject=subject,
        electrode_idx=electrode_idx,
        phoneme_pair=phoneme_pair,
        word_end=word_end,
        qualifying_steps=[row["step"]],
        epochs_dict=epochs_dict,
        bottom_late_window=(row["smin"], row["smax"]),
        textgrid_dir="textgrids",
        resampled_cmap=resampled_cmap,
        figsize=(3, 3),
    )
    f.suptitle(f"{subject} {electrode_idx} {phoneme_pair} {word_end}@{row['step']}")

# %% [markdown]
# ### Site-level roll-up
#
# Replicates `plot_for_paper`'s `late_category` construction: sum
# `sig_uncorrected` over `word_end` per (subject, electrode_idx, phoneme_pair)
# -> {absent: 0, one-sided: 1, two-sided: 2}. Denominator matches the B4
# cell-pool site count (31).

# %%
SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]

single_step_site = (
    ma_testable
    .groupby(SITE_KEYS)[["projection_significant_uncorrected"]]
    .sum()
    .rename(columns={"projection_significant_uncorrected": "n_we_sig"})
)
# Sites in the cell pool but entirely absent from ma_testable (e.g. both
# word_ends untestable) are absent (0 sig word_ends) by construction.
all_sites = cell_pool[SITE_KEYS].drop_duplicates().set_index(SITE_KEYS)
single_step_site = single_step_site.reindex(all_sites.index, fill_value=0)

single_step_site["late_category"] = single_step_site["n_we_sig"].map(
    {0: "absent", 1: "one-sided", 2: "two-sided"}
)

n_sites = len(single_step_site)
n_present = int((single_step_site["late_category"] != "absent").sum())

print(f"Single-step sites late-present (>=1 completion): {n_present} / {n_sites}")
print("(B4 reference: 11 / 31)")
print(single_step_site["late_category"].value_counts())

# %% [markdown]
# ## Cross-check: early perceptual sites
#
# Replicates `plot_for_paper`'s early x late conjunction table
# (`CONJUNCTION_CATEGORIES`: Acoustic / Late perceptual / Early perceptual /
# Early + late perceptual — see `src/viz_paper.py`), substituting the
# single-step site-level `late_category` for the pooled B4 one. This asks:
# of the sites that show an early perceptual (type2_aligned) response, how
# many still show a LATE response once the late percept contrast is
# restricted to a single acoustic step? "Early + late perceptual" is the
# reactivation-candidate cell — same code base for both windows is the
# minimum bar reactivation needs to clear, and single-step is a stricter,
# less pooled-power-dependent version of that bar.
#
# `epp` (early_category) universe and filter are byte-identical to
# `plot_for_paper`: acoustic_only sites, plus type2_aligned sites that are
# also uncorrected-significant (`p_one_tailed < 0.05`) — same early
# perceptual gate the paper's flow table uses. This is a strictly larger
# site universe than the B4/single-step cell pool (52 vs 31 in the
# `outputs_prod` reference run): sites failing the B4-windows join
# (`ci_excludes_zero & n_component_windows >= 2`) are absent from the cell
# pool but still appear here with `late_category = "absent"` — matching
# `plot_for_paper`'s outer-merge + fillna("absent") treatment.

# %%
from src.viz_paper import CONJUNCTION_CATEGORIES, F_EARLY, F_LATE  # noqa: E402

epp_orig = pd.read_csv(epp_path)
epp_orig["significant_uncorrected"] = epp_orig["p_one_tailed"] < fdr_alpha
epp = epp_orig[
    (epp_orig["early_response_class"] == "acoustic_only")
    | ((epp_orig["early_response_class"] == "type2_aligned") & epp_orig["significant_uncorrected"])
]
early_cat = (
    epp
    .assign(early_category=lambda df: df["early_response_class"].replace({
        "acoustic_only": "acoustic",
        "type2_aligned": "perceptual",
    }))
    [SITE_KEYS + ["early_category"]]
)
print(f"Early-category site universe (epp): {len(early_cat)}")

# B4 (pooled) site-level late_category — same recipe as plot_for_paper's
# `late_pres`, for a side-by-side, apples-to-apples comparison.
b4_site = (
    b4_results
    .groupby(SITE_KEYS)[["projection_significant_uncorrected"]]
    .sum()
    .rename(columns={"projection_significant_uncorrected": "n_we_sig"})
)
b4_site["late_category"] = b4_site["n_we_sig"].map(
    {0: "absent", 1: "one-sided", 2: "two-sided"}
)


def _conjunction_table(late_df, label):
    merged = pd.merge(
        early_cat,
        late_df[["late_category"]].reset_index(),
        on=SITE_KEYS,
        how="outer",
    )
    merged["late_category"] = merged["late_category"].fillna("absent")
    n_dropped = merged["early_category"].isna().sum()
    merged = merged.dropna(subset=["early_category"])

    ct = (
        merged
        .groupby(["early_category", "late_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(F_EARLY.order, axis=0)
        .reindex(list(F_LATE.order), axis=1, fill_value=0)
    )
    print(f"\n=== {label} ===")
    print(f"Site x pair cells total: {len(merged)}  (dropped {n_dropped} with no early_category)")
    print(ct)

    conj_counts = {}
    for cat_name, cat_def in CONJUNCTION_CATEGORIES.items():
        late_type = cat_def["late_category"]
        late_type = [late_type] if isinstance(late_type, str) else late_type
        n = int((
            (merged["early_category"] == cat_def["early_category"])
            & (merged["late_category"].isin(late_type))
        ).sum())
        conj_counts[cat_name] = n
    print(conj_counts)
    return merged, ct, conj_counts


_, ct_b4, conj_b4 = _conjunction_table(b4_site, "B4 (pooled) early x late")
_, ct_single, conj_single = _conjunction_table(single_step_site, "Single-step early x late")

print("\n=== 'Early + late perceptual' (reactivation-candidate) sites ===")
print(f"B4 pooled:    {conj_b4['Early + late perceptual']}")
print(f"Single-step:  {conj_single['Early + late perceptual']}")
