# ---
# jupyter:
#   jupytext:
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
# # A_neurometrics_provisional — causal6 (AUC-thresholded)
#
# Mirror of `A_neurometrics.py` that does NOT require the `prepare_neurometrics`
# parquet bundle and does NOT use FDR/permutation significance. Suitable for
# running before all subjects' permutation tests have finished.
#
# Inputs (consumed under `outputs_root`, default `outputs/causal6`):
#
#   - `brain_plot_acoustic_tstats.parquet` (per-(subj, elec, pp) acoustic peak)
#   - `brain_plot_behav_tstats.parquet`    (per-(subj, elec, pp, word_end) HGA-only behav peak)
#   - `brain_plot_ganong_tstats.parquet`   (per-(subj, elec, pp) Ganong peak — not used here)
#   - `brain_plot_behav_full_tstats.parquet`  (optional: full vs baseline behavior peak diff)
#   - `acoustic_decoding_single_electrode/{subj}/scores.parquet`  (Section #3 lookup)
#   - `behavior_decoding_single_electrode/{subj}/scores.parquet`  (Section #4, if present)
#
# These artifacts are produced by `notebooks/causal6/view_provisional_results.py`
# (the brain_plot_*.parquet) and by the pipeline rules `acoustic_decoding_single_electrode`
# and `behavior_decoding_single_electrode` (the raw scores).
#
# Sections implemented vs full `A_neurometrics.py`:
#   - #1 Overlap contingency  — replace `_is_significant()` with AUC threshold
#   - #2 Peak timing KDE      — same logic, threshold replaces significance
#   - #3 Cross-window phon decoding — raw acoustic searchlight from scores.parquet
#   - #4 Behavioral decoding improvement  — provisional; full vs baseline if available
#   - #5/#6 Transfer  — TODO stubs (decoder_weights_*.parquet not produced for causal6)
#   - #7a Star plots PDF      — provisional 3-panel plot via `provisional_star_plot`
#   - #7b Condition contrasts — SKIPPED (needs polarity from prepare_neurometrics)

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
import seaborn as sns
import yaml
from loguru import logger as L
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

# %%
from src.viz_paper import add_textgrid
from src.viz_provisional import (
    compute_2x2_contingency,
    load_ambig_steps,
    load_epochs_dict,
    provisional_star_plot,
)

# %% tags=["parameters"]
outputs_root = "outputs/causal6"
epochs_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal6/A_neurometrics_provisional"
epoch_tmin = -0.4
epoch_sfreq = 100
ambiguous_response_threshold = 2
acoustic_auc_threshold = 0.75
behav_auc_threshold = 0.80
textgrid_dir = "data/stimuli/textgrid"
top_k_single_modality = 10

# %%
outputs_root = Path(outputs_root)
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

with open("config.yaml") as _f:
    _config = yaml.safe_load(_f)

_AC_PEAK_SEARCH_SMIN = _config["analysis"]["decoding"]["acoustic_peak_search_smin"]
_AC_PEAK_SEARCH_SMAX = _config["analysis"]["decoding"]["acoustic_peak_search_smax"]
_AC_TARGET = "categorical_acoustic_cue"

# %% [markdown]
# ## Load brain_plot parquets (already produced by view_provisional_results)

# %%
def _read_brain_plot(name: str) -> pl.DataFrame | None:
    p = outputs_root / name
    if not p.exists():
        L.warning(f"missing {p} — sections that require it will be skipped")
        return None
    return pl.read_parquet(p)


ac_brain = _read_brain_plot("brain_plot_acoustic_tstats.parquet")
beh_brain = _read_brain_plot("brain_plot_behav_tstats.parquet")
beh_full_brain = _read_brain_plot("brain_plot_behav_full_tstats.parquet")
# gan_brain not used in the analogue of A_neurometrics; loaded for symmetry only
gan_brain = _read_brain_plot("brain_plot_ganong_tstats.parquet")

assert ac_brain is not None, "brain_plot_acoustic_tstats.parquet required for Sections #1-#3"
assert beh_brain is not None, "brain_plot_behav_tstats.parquet required for Sections #1, #2, #4, #7a"

print(f"ac_brain:  {ac_brain.shape}  subjects={sorted(ac_brain['subject'].unique().to_list())}")
print(f"beh_brain: {beh_brain.shape}  subjects={sorted(beh_brain['subject'].unique().to_list())}")
if beh_full_brain is not None:
    print(f"beh_full:  {beh_full_brain.shape}  subjects={sorted(beh_full_brain['subject'].unique().to_list())}")

# %% [markdown]
# ----
# # Section #1 — Overlap contingency between acoustic and behavior selectivity
#
# Replaces `_is_significant(...)` with `peak_auc >= threshold`. Universe is
# every (subject, electrode_idx, phoneme_pair) that appears in `ac_brain`
# (one row per site). Behavior-selective means **any** word_end of that site
# has HGA-only peak_auc above threshold.

# %%
_AC_KEYS = ["subject", "electrode_idx", "phoneme_pair"]

universe = ac_brain.select(_AC_KEYS).unique()

ac_selective = (
    ac_brain.filter(pl.col("peak_auc") >= acoustic_auc_threshold)
    .select(_AC_KEYS).unique()
)
beh_selective = (
    beh_brain.filter(pl.col("peak_auc") >= behav_auc_threshold)
    .select(_AC_KEYS).unique()
)

contingency = compute_2x2_contingency(
    sites_a=ac_selective,
    sites_b=beh_selective,
    universe=universe,
    label_a=f"acoustic (peak_auc >= {acoustic_auc_threshold:.2f})",
    label_b=f"behavior_hga_only (peak_auc >= {behav_auc_threshold:.2f})",
)

contingency_summary = pd.DataFrame([{
    "label_a": contingency["label_a"], "label_b": contingency["label_b"],
    "n_universe": contingency["n_universe"], "n_a": contingency["n_a"],
    "n_b": contingency["n_b"], "n_both": contingency["n_both"],
    "n_neither": contingency["n_neither"],
    "p_b_given_a": contingency["p_b_given_a"],
    "p_a_given_b": contingency["p_a_given_b"],
    "chi2": contingency["chi2"], "p_chi2": contingency["p_chi2"],
    "odds_ratio": contingency["odds_ratio"],
    "or_lo": contingency["or_95ci"][0], "or_hi": contingency["or_95ci"][1],
}])
contingency_summary.to_csv(outdir / "electrode_distribution_summary_provisional.csv", index=False)
print(contingency_summary.to_string(index=False))

# %%
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.5, 3.2))
table = contingency["table"]
n_universe = contingency["n_universe"]
n_a, n_b = contingency["n_a"], contingency["n_b"]
ax_left.set_title(f"{contingency['label_b']}", fontsize=9, loc="left")
ax_left.bar(
    ["P(B | A)", "P(B | ~A)"],
    [contingency["n_both"] / max(n_a, 1),
     table[1, 0] / max(n_universe - n_a, 1)],
    color=["#2c7fb8", "#a6bddb"],
)
ax_left.set_ylim(0, 1)
ax_left.set_ylabel(f"P(behavior | acoustic)")

ax_right.bar(
    ["P(A | B)", "P(A | ~B)"],
    [contingency["n_both"] / max(n_b, 1),
     table[0, 1] / max(n_universe - n_b, 1)],
    color=["#d95f0e", "#fec44f"],
)
ax_right.set_ylim(0, 1)
ax_right.set_ylabel("P(acoustic | behavior)")
ax_right.annotate(
    f"OR={contingency['odds_ratio']:.2f} "
    f"[{contingency['or_95ci'][0]:.2f}, {contingency['or_95ci'][1]:.2f}]\n"
    f"chi^2={contingency['chi2']:.1f}, p={contingency['p_chi2']:.2g}",
    xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
    fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
)
fig.suptitle(
    f"Section #1 — AUC-thresholded contingency  "
    f"(ac>={acoustic_auc_threshold:.2f}, beh>={behav_auc_threshold:.2f})",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(outdir / "electrode_distribution_provisional.pdf")
plt.close(fig)

# %% [markdown]
# ----
# # Section #2 — Peak timing dynamics
#
# KDE of acoustic-peak vs behavior-peak times, per word ending. Peak time is
# the window midpoint (matches `A_neurometrics.py` Section #2; differs from
# `view_provisional_results.smin_to_ms` which uses `smin` alone).

# %%
def _peak_t_midpoint(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (((pl.col("peak_smin") + pl.col("peak_smax")) / 2) / epoch_sfreq + epoch_tmin).alias("peak_t")
    )


_ac_pass = _peak_t_midpoint(ac_brain.filter(pl.col("peak_auc") >= acoustic_auc_threshold))
_beh_pass = _peak_t_midpoint(beh_brain.filter(pl.col("peak_auc") >= behav_auc_threshold))

for plot_word_end in ["necessary", "desolate"]:
    beh_we = _beh_pass.filter(pl.col("word_end") == plot_word_end).to_pandas()
    beh_we["kind"] = "behavior"
    # Acoustic peaks have no word_end — replicate filtered rows for this word_end.
    ac_pd = _ac_pass.to_pandas().copy()
    ac_pd["word_end"] = plot_word_end
    ac_pd["kind"] = "acoustic"

    plot_df = pd.concat(
        [ac_pd[["subject", "electrode_idx", "phoneme_pair", "word_end", "peak_t", "kind"]],
         beh_we[["subject", "electrode_idx", "phoneme_pair", "word_end", "peak_t", "kind"]]],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.kdeplot(data=plot_df, x="peak_t", hue="kind", common_norm=False, ax=ax,
                fill=True, alpha=0.4)
    ax.set_xlabel("Peak time (s, post word onset)")
    ax.set_xlim(0, 1.3)
    ax.set_title(
        f"Peak timing — {plot_word_end}\n"
        f"(ac n={len(ac_pd)} sites, beh n={len(beh_we)} sites)"
    )
    try:
        add_textgrid(ax, textgrid_dir=textgrid_dir,
                     textgrid_file=f"11_{plot_word_end}_dn_002.TextGrid",
                     vline_extent=1.0)
    except Exception as e:
        L.warning(f"textgrid overlay skipped for {plot_word_end}: {e}")
    fig.tight_layout()
    fig.savefig(outdir / f"decoding_timing-{plot_word_end}_provisional.pdf")
    plt.close(fig)

# %% [markdown]
# ----
# # Section #3 — Cross-window phonetic decoding (lookup, no transfer)
#
# For each acoustic-AUC-passing site, compare phon AUC at:
#   (a) its acoustic-peak window — read from `ac_brain.peak_auc` directly.
#   (b) its behavior-peak window — looked up in raw acoustic searchlight
#       scores at the behavior peak's (smin, smax).
#
# Behavior windows that fall outside the acoustic searchlight's range will
# drop in the inner join; the drop count is printed.

# %%
def _load_acoustic_searchlight() -> pl.DataFrame:
    paths = sorted(outputs_root.glob("acoustic_decoding_single_electrode/*/scores.parquet"))
    frames = []
    for p in paths:
        df = (
            pl.read_parquet(p)
            .filter(pl.col("target") == _AC_TARGET)
            .group_by(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
            .agg(pl.col("test_roc_auc").mean().alias("mean_test_roc_auc"))
        )
        frames.append(df)
    return pl.concat(frames) if frames else pl.DataFrame()


ac_searchlight = _load_acoustic_searchlight()
print(f"acoustic searchlight: {ac_searchlight.shape}  "
      f"smin range [{ac_searchlight['smin'].min()}, {ac_searchlight['smin'].max()}]")

# Acoustic-selective sites at their acoustic peak window (the AUC IS peak_auc).
phon_at_phon_peak = (
    ac_brain.filter(pl.col("peak_auc") >= acoustic_auc_threshold)
    .select(["subject", "electrode_idx", "phoneme_pair", "peak_auc"])
    .rename({"peak_auc": "mean_test_roc_auc"})
    .with_columns(pl.lit("acoustic_window").alias("evaluation"))
)

# Same sites' acoustic AUC at the BEHAVIOR peak window. Each (subj, elec, pp)
# has up to 2 word_ends; both rows participate.
_ac_pass_sites = ac_brain.filter(pl.col("peak_auc") >= acoustic_auc_threshold).select(_AC_KEYS).unique()
_beh_windows_for_ac_sites = (
    beh_brain
    .join(_ac_pass_sites, on=_AC_KEYS, how="inner")
    .select(["subject", "electrode_idx", "phoneme_pair", "word_end",
             "peak_smin", "peak_smax"])
    .rename({"peak_smin": "smin", "peak_smax": "smax"})
)
phon_at_behav_peak = (
    _beh_windows_for_ac_sites
    .join(ac_searchlight,
          on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
          how="inner")
    .with_columns(pl.lit("perceptual_window").alias("evaluation"))
    .select(["subject", "electrode_idx", "phoneme_pair", "mean_test_roc_auc", "evaluation"])
)

_dropped = _beh_windows_for_ac_sites.height - phon_at_behav_peak.height
print(f"Section #3: {_beh_windows_for_ac_sites.height} candidate (site, word_end) rows; "
      f"{_dropped} dropped (behavior window outside acoustic searchlight range)")

phon_roc_auc_comparison = pl.concat([phon_at_phon_peak, phon_at_behav_peak])

phon_subj_means = (
    phon_roc_auc_comparison
    .group_by(["subject", "evaluation"])
    .agg(pl.col("mean_test_roc_auc").mean().alias("mean_roc_auc"))
    .pivot(on="evaluation", index="subject", values="mean_roc_auc")
    .drop_nulls()
)

if phon_subj_means.height >= 2:
    t, p = stats.ttest_rel(
        phon_subj_means["acoustic_window"].to_numpy(),
        phon_subj_means["perceptual_window"].to_numpy(),
    )
    print(f"phon decoding (acoustic vs perceptual peak window): "
          f"t={t:.3f}, p={p:.3g}, n_subjects={phon_subj_means.height}")
else:
    print(f"phon decoding spaghetti: only {phon_subj_means.height} subjects; t-test skipped")

fig, ax = plt.subplots(figsize=(3.2, 3))
for row in phon_subj_means.iter_rows(named=True):
    ax.plot([0, 1], [row["acoustic_window"], row["perceptual_window"]],
            color="gray", alpha=0.6, marker="o")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Acoustic\nwindow", "Perceptual\nwindow"])
ax.set_ylabel("Acoustic ROC-AUC (mean across sites)")
ax.set_ylim(0.4, 1.0)
ax.set_title(f"Cross-window acoustic decoding\n(n_subjects={phon_subj_means.height})")
fig.tight_layout()
fig.savefig(outdir / "decoding_phonetic_provisional.pdf")
plt.close(fig)

# %% [markdown]
# ----
# # Section #4 — Behavioral decoding improvement (provisional)
#
# Compare AUCs at each behavioral-peak site:
#   - **HGA-only** AUC = `beh_brain.peak_auc` directly.
#   - **Full** (HGA + control) and **baseline** (control-only) AUCs at the
#     SAME (smin, smax) as the HGA-only peak — looked up in raw
#     `behavior_decoding_single_electrode/{subj}/scores.parquet`.
#
# When `behavior_decoding_single_electrode/` is absent (no full+baseline data),
# only the HGA-only AUC distribution is shown.

# %%
_beh_full_paths = sorted(outputs_root.glob("behavior_decoding_single_electrode/*/scores.parquet"))
has_full_baseline = len(_beh_full_paths) > 0

behav_peak_set = (
    beh_brain.filter(pl.col("peak_auc") >= behav_auc_threshold)
    .select(["subject", "electrode_idx", "phoneme_pair", "word_end",
             "peak_smin", "peak_smax", "peak_auc"])
    .rename({"peak_smin": "smin", "peak_smax": "smax",
             "peak_auc": "hga_only_roc_auc"})
)

if has_full_baseline:
    _full_frames = []
    _base_frames = []
    for p in _beh_full_paths:
        raw = pl.read_parquet(p)
        full = (
            raw.filter(pl.col("model") == "full")
            .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"])
            .agg(pl.col("test_roc_auc").mean().alias("full_roc_auc"))
        )
        # Baseline is electrode-/window-independent (one number per fold per
        # (subject, phoneme_pair, word_end)). Mean across folds.
        base = (
            raw.filter(pl.col("model") == "baseline")
            .group_by(["subject", "phoneme_pair", "word_end"])
            .agg(pl.col("test_roc_auc").mean().alias("baseline_roc_auc"))
        )
        _full_frames.append(full)
        _base_frames.append(base)
    full_lookup = pl.concat(_full_frames)
    base_lookup = pl.concat(_base_frames)

    behav_at_peaks = (
        behav_peak_set
        .join(full_lookup,
              on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
              how="left")
        .join(base_lookup,
              on=["subject", "phoneme_pair", "word_end"],
              how="left")
    )

    behav_summary = (
        behav_at_peaks
        .group_by("subject")
        .agg([
            pl.col("hga_only_roc_auc").mean().alias("hga_only_mean"),
            pl.col("full_roc_auc").mean().alias("full_mean"),
            pl.col("baseline_roc_auc").mean().alias("baseline_mean"),
        ])
        .drop_nulls()
    )

    print("Section #4 subject means:")
    print(behav_summary.to_pandas().to_string(index=False))

    fig, ax = plt.subplots(figsize=(3.5, 3))
    for row in behav_summary.iter_rows(named=True):
        ax.plot(
            [0, 1, 2],
            [row["baseline_mean"], row["hga_only_mean"], row["full_mean"]],
            color="gray", alpha=0.6, marker="o",
        )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Control\nbaseline", "HGA only", "HGA + control"])
    ax.set_ylabel("Behavioral ROC-AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"Behavioral decoding improvement\n(n_subjects={behav_summary.height})")
    fig.tight_layout()
    fig.savefig(outdir / "decoding_behavioral_improvement_provisional.pdf")
    plt.close(fig)
else:
    print("Section #4 (provisional): `behavior_decoding_single_electrode/` not present — "
          "showing HGA-only AUC distribution at peak only.")

    fig, ax = plt.subplots(figsize=(4, 3))
    for subj, sub in behav_peak_set.group_by("subject"):
        ax.scatter(np.full(sub.height, sorted(behav_peak_set["subject"].unique().to_list()).index(subj[0])),
                   sub["hga_only_roc_auc"].to_numpy(),
                   alpha=0.4, s=14)
    subj_order = sorted(behav_peak_set["subject"].unique().to_list())
    ax.set_xticks(range(len(subj_order)))
    ax.set_xticklabels(subj_order, rotation=45, ha="right", fontsize=8)
    ax.axhline(0.5, color="k", lw=0.5, ls="--", label="chance")
    ax.axhline(behav_auc_threshold, color="tomato", lw=0.5, ls=":",
               label=f"threshold {behav_auc_threshold:.2f}")
    ax.set_ylabel("HGA-only ROC-AUC at peak")
    ax.set_title("Behavioral decoding (HGA-only) at peak window\n"
                 "[full+baseline not yet available]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "decoding_behavioral_improvement_provisional.pdf")
    plt.close(fig)

# %% [markdown]
# ----
# # Section #5 / #6 — TODO (decoder transfer)
#
# Causal4/5 evaluate cross-window decoder TRANSFER (apply weights from one
# window to features from another). That requires `decoder_weights_*.parquet`
# from `prepare_neurometrics`, which hasn't been run for causal6. Skipping
# until those weights are produced.

# %%
for stub_name, stub_label in [
    ("decoding_acoustic_transfer_provisional.pdf",
     "Section #5 — TODO\n(apply acoustic weights\nto perceptual window)"),
    ("decoding_phon_decoder_on_behav_window_provisional.pdf",
     "Section #6 — TODO\n(apply behav weights\nto acoustic window)"),
]:
    fig, ax = plt.subplots(figsize=(3, 2.75))
    ax.text(0.5, 0.5, stub_label, ha="center", va="center", fontsize=10)
    ax.axis("off")
    fig.savefig(outdir / stub_name)
    plt.close(fig)

# %% [markdown]
# ----
# # Section #7a — Provisional star plots PDF
#
# Three PDFs:
#   - `provisional_star_plots.pdf`: sites passing BOTH AUC thresholds.
#   - `provisional_star_plots_behav_only.pdf`: top-K HGA-only behav sites
#     that don't pass the acoustic threshold.
#   - `provisional_star_plots_acoustic_only.pdf`: top-K acoustic sites that
#     don't pass the behavioral threshold, with one page per word_end with
#     ambiguous-step data.
#
# Each page is a 3-panel `provisional_star_plot` (unambig / controlled
# ambiguous / all trials). No polarity correction is applied (that requires
# `prepare_neurometrics`).

# %%
# Build a per-electrode contingency: best acoustic vs best behavioral peak per
# (subject, electrode_idx). Mirrors view_provisional_results.py section 8.
_ac_pd = ac_brain.to_pandas()
_beh_pd = beh_brain.to_pandas()

acoustic_passes = _ac_pd[_ac_pd.peak_auc >= acoustic_auc_threshold].copy()
behav_passes = _beh_pd[_beh_pd.peak_auc >= behav_auc_threshold].copy()

behav_plot = (
    behav_passes
    .sort_values("peak_auc", ascending=False)
    .groupby(["subject", "electrode_idx"], observed=True, as_index=False)
    .first()
)
A_phonetic_plot = (
    acoustic_passes
    .sort_values("peak_auc", ascending=False)
    .groupby(["subject", "electrode_idx"], observed=True, as_index=False)
    .first()
)

contingency_df = pd.merge(
    A_phonetic_plot.drop_duplicates(["subject", "electrode_idx"]),
    behav_plot.drop_duplicates(["subject", "electrode_idx"]),
    on=["subject", "electrode_idx"],
    how="outer",
    suffixes=("_phon", "_behav"),
)
contingency_df["outcome"] = "all"
contingency_df.loc[contingency_df["peak_auc_behav"].isna(), "outcome"] = "phonetic"
contingency_df.loc[contingency_df["peak_auc_phon"].isna(), "outcome"] = "behav"
contingency_df = (
    contingency_df
    .sort_values(["subject", "electrode_idx", "outcome"])
    .drop_duplicates(["subject", "electrode_idx"])
    .reset_index(drop=True)
)

print(contingency_df.outcome.value_counts(normalize=False))

_both = contingency_df[contingency_df.outcome == "all"].copy()

_beh_full_pd = beh_full_brain.to_pandas() if beh_full_brain is not None else None


def _beh_full_diff(subject, electrode_idx, phoneme_pair, word_end):
    if _beh_full_pd is None:
        return None
    mask = (
        (_beh_full_pd["subject"] == subject)
        & (_beh_full_pd["electrode_idx"] == electrode_idx)
        & (_beh_full_pd["phoneme_pair"] == phoneme_pair)
        & (_beh_full_pd["word_end"] == word_end)
    )
    rows = _beh_full_pd[mask]
    return float(rows["peak_diff"].iloc[0]) if len(rows) > 0 else None


def _beh_hga_auc(subject, electrode_idx, phoneme_pair, word_end):
    mask = (
        (_beh_pd["subject"] == subject)
        & (_beh_pd["electrode_idx"] == electrode_idx)
        & (_beh_pd["phoneme_pair"] == phoneme_pair)
        & (_beh_pd["word_end"] == word_end)
    )
    rows = _beh_pd[mask]
    return float(rows["peak_auc"].iloc[0]) if len(rows) > 0 else None


# %%
epochs_dict = load_epochs_dict(Path(epochs_dir))
print(f"epochs_dict subjects: {sorted(epochs_dict)}")
ambig_steps = load_ambig_steps(epochs_dict,
                                ambiguous_response_threshold=ambiguous_response_threshold)
print(f"ambig_steps: {len(ambig_steps)} (subject, phoneme_pair, word_end) keys")

# %% [markdown]
# ### Both — sites passing both thresholds

# %%
_both_out = outdir / "provisional_star_plots.pdf"
_missing = [s for s in _both["subject"].unique() if s not in epochs_dict]
if _missing:
    print(f"Warning: epochs not loaded for {_missing} — those sites will be skipped.")

with PdfPages(_both_out) as _pdf:
    for _, row in tqdm(_both.iterrows(), total=len(_both),
                       desc="both: star plots"):
        if row["subject"] not in epochs_dict:
            continue
        try:
            fig = provisional_star_plot(
                subject=row["subject"],
                electrode_idx=int(row["electrode_idx"]),
                phoneme_pair=row["phoneme_pair_behav"],
                word_end=row["word_end"],
                epochs_dict=epochs_dict,
                ambig_steps=ambig_steps,
                phon_smin=int(row["peak_smin_phon"]),
                phon_smax=int(row["peak_smax_phon"]),
                behav_smin=int(row["peak_smin_behav"]),
                behav_smax=int(row["peak_smax_behav"]),
                textgrid_dir=textgrid_dir,
                epoch_tmin=epoch_tmin, epoch_sfreq=epoch_sfreq,
                acoustic_peak_auc=float(row["peak_auc_phon"]),
                behav_full_peak_diff=_beh_full_diff(
                    row["subject"], int(row["electrode_idx"]),
                    row["phoneme_pair_behav"], row["word_end"]),
                behav_hga_peak_auc=float(row["peak_auc_behav"]),
            )
            _pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print(f"  skipped {row['subject']} e{int(row['electrode_idx'])}: {e}")

print(f"Written {len(_both)} pages → {_both_out}")

# %% [markdown]
# ### Top-K behavioral-only sites (no acoustic pass)

# %%
_behav_only = (
    contingency_df[contingency_df.outcome == "behav"]
    .sort_values("peak_auc_behav", ascending=False)
    .head(top_k_single_modality)
)
print(f"Top {top_k_single_modality} behavioral-only sites:")
print(_behav_only[["subject", "electrode_idx", "phoneme_pair_behav",
                   "word_end", "peak_auc_behav"]].to_string(index=False))

_bonly_out = outdir / "provisional_star_plots_behav_only.pdf"
with PdfPages(_bonly_out) as _pdf:
    for _, row in _behav_only.iterrows():
        if row["subject"] not in epochs_dict:
            continue
        try:
            fig = provisional_star_plot(
                subject=row["subject"],
                electrode_idx=int(row["electrode_idx"]),
                phoneme_pair=row["phoneme_pair_behav"],
                word_end=row["word_end"],
                epochs_dict=epochs_dict,
                ambig_steps=ambig_steps,
                behav_smin=int(row["peak_smin_behav"]),
                behav_smax=int(row["peak_smax_behav"]),
                textgrid_dir=textgrid_dir,
                epoch_tmin=epoch_tmin, epoch_sfreq=epoch_sfreq,
                behav_full_peak_diff=_beh_full_diff(
                    row["subject"], int(row["electrode_idx"]),
                    row["phoneme_pair_behav"], row["word_end"]),
                behav_hga_peak_auc=float(row["peak_auc_behav"]),
            )
            _pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print(f"  skipped {row['subject']} e{int(row['electrode_idx'])}: {e}")
print(f"Written {len(_behav_only)} pages → {_bonly_out}")

# %% [markdown]
# ### Top-K acoustic-only sites (no behavioral pass)
#
# Acoustic peaks have no `word_end` — render one page per word_end available
# for that (subject, phoneme_pair) in `ambig_steps`.

# %%
_acoustic_only = (
    contingency_df[contingency_df.outcome == "phonetic"]
    .sort_values("peak_auc_phon", ascending=False)
    .head(top_k_single_modality)
)
print(f"Top {top_k_single_modality} acoustic-only sites:")
print(_acoustic_only[["subject", "electrode_idx", "phoneme_pair_phon",
                      "peak_auc_phon"]].to_string(index=False))

_aonly_out = outdir / "provisional_star_plots_acoustic_only.pdf"
with PdfPages(_aonly_out) as _pdf:
    _n_pages = 0
    for _, row in _acoustic_only.iterrows():
        subj = row["subject"]
        pp = row["phoneme_pair_phon"]
        if subj not in epochs_dict:
            continue
        word_ends = sorted({
            we for (s, p, we) in ambig_steps.keys()
            if s == subj and p == pp
        })
        if not word_ends:
            word_ends = sorted(
                epochs_dict[subj].metadata
                .loc[epochs_dict[subj].metadata["phoneme_pair"] == pp, "word_end"]
                .dropna().unique().tolist()
            )
        for we in word_ends:
            try:
                fig = provisional_star_plot(
                    subject=subj,
                    electrode_idx=int(row["electrode_idx"]),
                    phoneme_pair=pp,
                    word_end=we,
                    epochs_dict=epochs_dict,
                    ambig_steps=ambig_steps,
                    phon_smin=int(row["peak_smin_phon"]),
                    phon_smax=int(row["peak_smax_phon"]),
                    textgrid_dir=textgrid_dir,
                    epoch_tmin=epoch_tmin, epoch_sfreq=epoch_sfreq,
                    acoustic_peak_auc=float(row["peak_auc_phon"]),
                    behav_full_peak_diff=_beh_full_diff(
                        subj, int(row["electrode_idx"]), pp, we),
                    behav_hga_peak_auc=_beh_hga_auc(
                        subj, int(row["electrode_idx"]), pp, we),
                )
                _pdf.savefig(fig)
                plt.close(fig)
                _n_pages += 1
            except Exception as e:
                print(f"  skipped {subj} e{int(row['electrode_idx'])} {we}: {e}")
print(f"Written {_n_pages} pages → {_aonly_out}")

# %% [markdown]
# ----
# # Section #7b — SKIPPED
#
# Polarity-corrected mean HGA condition contrasts require `early_polarity` /
# `late_polarity` from `prepare_neurometrics`. Run `prepare_neurometrics` for
# causal6 before producing this figure (or accept a no-polarity-correction
# variant separately).
