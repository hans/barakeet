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
# # A_neurometrics — causal6 edition
#
# Population analyses + figures from the prepare_neurometrics bundle.
# Ports causal4 sections #1, #2, #3, #4, #5, #6, #7. Sections #8-#14 are
# TODO stubs at the bottom of the notebook with pointers into causal4.
#
# Key adaptations from causal4/causal5:
#
#   - Section #1 (overlap contingency) computes the full 2x2 contingency,
#     both conditional probabilities (P(B|A) and P(A|B)), chi-square, and
#     odds ratio. Reported per peak-flavor (tstat_maxstat & tstat_tfce);
#     behavior_full reported alongside behavior_hga_only as preliminary.
#
#   - Section #4 compares HGA-only / full / control-baseline AUC. The
#     control-only baseline lives in behav_baseline_df from prepare_neurometrics.
#
#   - Sections #5/#6 (cross-window transfer) use the small `apply_decoder_weights`
#     helper below — causal6 persists decoders as parquet (coef, mean, scale
#     list-columns), so we apply the linear weights directly rather than
#     using viz_paper.evaluate_phonetic_transfer (which expects sklearn
#     pipeline objects).
#
#   - Section #7 emits an exploratory multi-page PdfPages over all selective
#     sites, no exemplar curation.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
import seaborn as sns
from loguru import logger as L
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

# %%
from src.data import add_metadata_features
from src.viz_paper import (
    PaperData,
    add_textgrid,
    extract_hga_windows_df,
    pl_roc_auc,
    plot_behav_barplot,
    plot_condition_contrast,
    plot_condition_contrasts_single_figure,
    zoomin_hga,
)

# %% tags=["parameters"]
neurometrics_dir = "outputs/causal6/prepare_neurometrics"
all_epochs = sorted(str(p) for p in Path("outputs/epochs_preprocessed").glob("*_epo.fif"))
outdir = "outputs/causal6/A_neurometrics"

epoch_tmin = -0.4
epoch_sfreq = 100
ambiguous_response_threshold = 2
primary_peak_flavor = "tstat_maxstat"
textgrid_dir = "data/stimuli/textgrid"

# %%
neurometrics_dir = Path(neurometrics_dir)
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load PaperData bundle

# %%
def _read(name):
    p = neurometrics_dir / name
    if name.endswith(".parquet"):
        try:
            return pl.read_parquet(p)
        except Exception:
            return pd.read_parquet(p)
    raise ValueError(f"unsupported: {name}")


electrode_df = _read("electrode_df.parquet")
all_md = _read("all_md.parquet")
word_end_df = _read("word_end_df.parquet")

phon_peaks_foldmean_maxstat = _read("phon_peaks_foldmean_maxstat.parquet")
phon_peaks_tstat_maxstat = _read("phon_peaks_tstat_maxstat.parquet")

behav_hga_only_peaks = {
    "foldmean_maxstat": _read("behav_hga_only_peaks_foldmean_maxstat.parquet"),
    "tstat_maxstat":    _read("behav_hga_only_peaks_tstat_maxstat.parquet"),
    "tstat_tfce":       _read("behav_hga_only_peaks_tstat_tfce.parquet"),
}
behav_full_peaks = {
    "foldmean_maxstat": _read("behav_full_peaks_foldmean_maxstat.parquet"),
    "tstat_maxstat":    _read("behav_full_peaks_tstat_maxstat.parquet"),
    "tstat_tfce":       _read("behav_full_peaks_tstat_tfce.parquet"),
}

phon_roc_auc_searchlight_df = _read("phon_roc_auc_searchlight_df.parquet")
behav_roc_auc_searchlight_df = _read("behav_roc_auc_searchlight_df.parquet")
behav_baseline_df = _read("behav_baseline_df.parquet")

plot_phon_phon_df = _read("plot_phon_phon_df.parquet")
plot_behav_phon_df = _read("plot_behav_phon_df.parquet")
plot_phon_behav_df = _read("plot_phon_behav_df.parquet")
plot_behav_behav_df = _read("plot_behav_behav_df.parquet")
zoomin_keys = _read("zoomin_keys.parquet")

hga_df = pd.read_parquet(neurometrics_dir / "hga_df.parquet")
early_polarity = pd.read_parquet(neurometrics_dir / "early_polarity.parquet").set_index(
    ["subject", "electrode_idx", "phoneme_pair", "word_end"]
)["early_polarity"]
late_polarity = pd.read_parquet(neurometrics_dir / "late_polarity.parquet").set_index(
    ["subject", "electrode_idx", "phoneme_pair", "word_end"]
)["late_polarity"]
reg_df = pd.read_parquet(neurometrics_dir / "reg_df.parquet")

decoder_weights_acoustic = _read("decoder_weights_acoustic.parquet")
decoder_weights_behav_hga_only = _read("decoder_weights_behav_hga_only.parquet")

# %%
epochs = {}
for p in tqdm(all_epochs, desc="loading epochs"):
    import re as _re
    m = _re.search(r"(EC\d+)_epo", str(p))
    assert m is not None
    subj = m.group(1)
    ep = mne.read_epochs(str(p), preload=False, verbose="WARNING")
    ep.metadata = add_metadata_features(ep.metadata.copy())
    epochs[subj] = ep

# %%
paper_data = PaperData(
    electrode_df=electrode_df,
    plot_phon_phon_df=plot_phon_phon_df,
    plot_behav_phon_df=plot_behav_phon_df,
    plot_behav_behav_df=plot_behav_behav_df,
    plot_phon_behav_df=plot_phon_behav_df,
    behav_roc_auc_searchlight_df=behav_roc_auc_searchlight_df,
    phon_roc_auc_searchlight_df=phon_roc_auc_searchlight_df,
    all_md=all_md,
    word_end_df=word_end_df,
    epochs=epochs,
    phon_peaks_df=phon_peaks_foldmean_maxstat,
    behav_peaks_df=behav_hga_only_peaks[primary_peak_flavor],
    behav_peaks_df_unfiltered=behav_hga_only_peaks[primary_peak_flavor],
    zoomin_keys=zoomin_keys,
    early_polarity=early_polarity,
    late_polarity=late_polarity,
    hga_df=hga_df,
    reg_df=reg_df,
)

# %% [markdown]
# ## Inline helpers

# %%
def _is_significant(df: pl.DataFrame) -> pl.Series:
    """FDR-significance column with safe fallback. The aggregator
    (notebooks/causal6/significance_aggregate.py) writes a `significant`
    bool column from BH-FDR on `p_value` at fdr_alpha; if that column is
    absent (e.g. consumed pre-aggregation), fall back to uncorrected
    p_value < 0.05."""
    if "significant" in df.columns:
        return df.get_column("significant")
    return df.get_column("p_value") < 0.05


def site_keys(df: pl.DataFrame, *extra) -> pl.DataFrame:
    return df.select(["subject", "electrode_idx", "phoneme_pair", *extra]).unique()


def compute_2x2_contingency(
    sites_a: pl.DataFrame,
    sites_b: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    label_a: str,
    label_b: str,
) -> dict:
    """Build a 2x2 contingency over a universe of sites (e.g. all speech-responsive
    electrodes), partitioning by membership in set A vs set B. Returns counts,
    both conditional probabilities, chi-square, and odds ratio with 95% CI.

    Each input frame must have at least the columns
    ["subject", "electrode_idx", "phoneme_pair"].
    """
    join_keys = ["subject", "electrode_idx", "phoneme_pair"]
    u = universe.select(join_keys).unique()
    a = sites_a.select(join_keys).unique().with_columns(pl.lit(True).alias("_in_a"))
    b = sites_b.select(join_keys).unique().with_columns(pl.lit(True).alias("_in_b"))
    df = (
        u.join(a, on=join_keys, how="left")
         .join(b, on=join_keys, how="left")
         .with_columns([
             pl.col("_in_a").fill_null(False).alias("_in_a"),
             pl.col("_in_b").fill_null(False).alias("_in_b"),
         ])
    )
    n11 = df.filter(pl.col("_in_a") & pl.col("_in_b")).height
    n10 = df.filter(pl.col("_in_a") & ~pl.col("_in_b")).height
    n01 = df.filter(~pl.col("_in_a") & pl.col("_in_b")).height
    n00 = df.filter(~pl.col("_in_a") & ~pl.col("_in_b")).height
    table = np.array([[n11, n10], [n01, n00]])

    chi2, p_chi2, dof, expected = stats.chi2_contingency(table, correction=False)

    # Odds ratio with 95% CI via Haldane-Anscombe correction for empty cells
    a11, a10, a01, a00 = (table + 0.5).flatten()
    odds_ratio = (a11 * a00) / (a10 * a01)
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1 / a11 + 1 / a10 + 1 / a01 + 1 / a00)
    ci_log = (log_or - 1.96 * se_log_or, log_or + 1.96 * se_log_or)
    or_ci = (np.exp(ci_log[0]), np.exp(ci_log[1]))

    p_b_given_a = n11 / (n11 + n10) if (n11 + n10) else float("nan")
    p_a_given_b = n11 / (n11 + n01) if (n11 + n01) else float("nan")

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_universe": df.height,
        "n_a": n11 + n10,
        "n_b": n11 + n01,
        "n_both": n11,
        "n_neither": n00,
        "table": table,
        "p_b_given_a": p_b_given_a,
        "p_a_given_b": p_a_given_b,
        "chi2": chi2,
        "p_chi2": p_chi2,
        "dof": dof,
        "odds_ratio": odds_ratio,
        "or_95ci": or_ci,
    }


def apply_decoder_weights(
    coef: np.ndarray, mean: np.ndarray, scale: np.ndarray, intercept: float | None,
    X: np.ndarray,
) -> np.ndarray:
    """Apply a stored linear LR (coef + StandardScaler stats) to features X.
    Returns predicted probabilities of class 1.

    coef, mean, scale: 1-D arrays of length d.
    X: (n_trials, d). intercept defaults to 0 if missing (causal6 _fit_batched_cv
    fits without explicit bias storage; verify and adjust if your coefficients
    parquet has an `intercept` column).
    """
    Xs = (X - mean[None, :]) / scale[None, :]
    logit = Xs @ coef + (intercept if intercept is not None else 0.0)
    return 1.0 / (1.0 + np.exp(-logit))


# %% [markdown]
# ----
# # Section #1 — Overlap contingency between acoustic and behavior selectivity
#
# Causal4 only computed P(behavioral | acoustic). Now compute the full 2x2
# contingency over all speech-responsive electrodes, with both conditional
# probabilities, chi-square, and odds ratio. Run per peak-flavor.

# %%
contingency_universe = (
    pl.from_pandas(electrode_df.to_pandas() if isinstance(electrode_df, pl.DataFrame) else electrode_df)
    if False else electrode_df
)

# Acoustic-selective sites: FDR-significant peaks (foldmean_maxstat is canonical here;
# tstat_maxstat reported alongside).
phon_sig_foldmean = phon_peaks_foldmean_maxstat.filter(_is_significant(phon_peaks_foldmean_maxstat))
phon_sig_tstat    = phon_peaks_tstat_maxstat.filter(_is_significant(phon_peaks_tstat_maxstat))

# Behavior-selective: per flavor, per decoder.
behav_hga_only_sig = {
    flavor: df.filter(_is_significant(df))
    for flavor, df in behav_hga_only_peaks.items()
}
behav_full_sig = {
    flavor: df.filter(_is_significant(df))
    for flavor, df in behav_full_peaks.items()
}

# Universe = electrode_df (all speech-responsive electrodes per subject).
# For "site" we expand each electrode against the phoneme pairs the subject has.
universe_sites = (
    all_md.select(["subject", "phoneme_pair"]).unique()
    .join(electrode_df.select(["subject", "electrode_idx"]).unique(), on="subject", how="inner")
)

# %%
contingency_records = []
for flavor in ["tstat_maxstat", "tstat_tfce"]:
    # Acoustic side — acoustic peaks don't have tfce, so reuse tstat_maxstat for the
    # tfce row to keep the table shape consistent.
    phon_sig = phon_sig_tstat if flavor == "tstat_maxstat" else phon_sig_tstat
    for behav_label, behav_dict in [
        ("behavior_hga_only", behav_hga_only_sig),
        ("behavior_full (preliminary)", behav_full_sig),
    ]:
        result = compute_2x2_contingency(
            sites_a=phon_sig,
            sites_b=behav_dict[flavor],
            universe=universe_sites,
            label_a="acoustic_selective",
            label_b=behav_label,
        )
        result["flavor"] = flavor
        contingency_records.append(result)

contingency_summary = pd.DataFrame([
    {
        "flavor": r["flavor"], "label_a": r["label_a"], "label_b": r["label_b"],
        "n_universe": r["n_universe"], "n_a": r["n_a"], "n_b": r["n_b"],
        "n_both": r["n_both"], "n_neither": r["n_neither"],
        "p_b_given_a": r["p_b_given_a"], "p_a_given_b": r["p_a_given_b"],
        "chi2": r["chi2"], "p_chi2": r["p_chi2"],
        "odds_ratio": r["odds_ratio"],
        "or_lo": r["or_95ci"][0], "or_hi": r["or_95ci"][1],
    }
    for r in contingency_records
])
contingency_summary.to_csv(outdir / "electrode_distribution_summary.csv", index=False)
print(contingency_summary.to_string(index=False))

# %%
fig, axes = plt.subplots(
    nrows=len(contingency_records), ncols=2,
    figsize=(7, 2.5 * len(contingency_records)),
    sharey="col",
)
if axes.ndim == 1:
    axes = axes[None, :]

for row_idx, r in enumerate(contingency_records):
    ax_left, ax_right = axes[row_idx]
    title = f"{r['flavor']} — {r['label_b']}"
    ax_left.set_title(title, fontsize=10, loc="left")
    # Left panel: P(B | A)
    ax_left.bar(["P(B | A)", "P(B | ~A)"],
                [r["n_both"] / max(r["n_a"], 1),
                 r["table"][1, 0] / max(r["n_universe"] - r["n_a"], 1)],
                color=["#2c7fb8", "#a6bddb"])
    ax_left.set_ylim(0, 1)
    ax_left.set_ylabel(f"P({r['label_b']} | acoustic)")
    # Right panel: P(A | B)
    ax_right.bar(["P(A | B)", "P(A | ~B)"],
                 [r["n_both"] / max(r["n_b"], 1),
                  r["table"][0, 1] / max(r["n_universe"] - r["n_b"], 1)],
                 color=["#d95f0e", "#fec44f"])
    ax_right.set_ylim(0, 1)
    ax_right.set_ylabel("P(acoustic | behavior)")
    # Annotate stats
    ax_right.annotate(
        f"OR={r['odds_ratio']:.2f} [{r['or_95ci'][0]:.2f}, {r['or_95ci'][1]:.2f}]\n"
        f"chi^2={r['chi2']:.1f}, p={r['p_chi2']:.2g}",
        xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
    )

fig.tight_layout()
fig.savefig(outdir / "electrode_distribution.pdf")
plt.close(fig)

# %% [markdown]
# ----
# # Section #2 — Peak timing dynamics
#
# KDE of acoustic-peak vs behavior-peak times, per word ending. Reads peaks
# parquets directly.

# %%
def _peaks_to_pd(df: pl.DataFrame, kind: str) -> pd.DataFrame:
    return (
        df.filter(_is_significant(df))
          .with_columns([
              ((pl.col("smin") + pl.col("smax")) / 2 / epoch_sfreq + epoch_tmin).alias("peak_t"),
              pl.lit(kind).alias("kind"),
          ])
          .select(["subject", "electrode_idx", "phoneme_pair", "word_end", "peak_t", "kind"]
                  if "word_end" in df.columns
                  else ["subject", "electrode_idx", "phoneme_pair", "peak_t", "kind"])
          .to_pandas()
    )


for plot_word_end in ["necessary", "desolate"]:
    behav_peaks = behav_hga_only_peaks[primary_peak_flavor]
    behav_filtered = behav_peaks.filter(pl.col("word_end") == plot_word_end)
    phon_pd = _peaks_to_pd(phon_peaks_foldmean_maxstat, "acoustic")
    behav_pd = _peaks_to_pd(behav_filtered, "behavior")
    if "word_end" not in phon_pd.columns:
        # Acoustic peaks aren't per-word_end; replicate across word ends for display.
        pp = phon_pd.copy()
        pp["word_end"] = plot_word_end
        phon_pd = pp
    plot_df = pd.concat([phon_pd, behav_pd], ignore_index=True)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.kdeplot(data=plot_df, x="peak_t", hue="kind", common_norm=False, ax=ax,
                fill=True, alpha=0.4)
    ax.set_xlabel("Peak time (s, post word onset)")
    ax.set_xlim(0, 1.3)
    ax.set_title(f"Peak timing — {plot_word_end}")
    try:
        add_textgrid(ax, textgrid_dir=textgrid_dir,
                     textgrid_file=f"11_{plot_word_end}_dn_002.TextGrid",
                     vline_extent=1.0)
    except Exception as e:
        L.warning(f"textgrid overlay skipped for {plot_word_end}: {e}")
    fig.tight_layout()
    fig.savefig(outdir / f"decoding_timing-{plot_word_end}.pdf")
    plt.close(fig)

# %% [markdown]
# ----
# # Section #3 — Cross-window phonetic decoding (lookup, no transfer)
#
# For each electrode, look up the in-window acoustic ROC-AUC at:
#   (a) its acoustic-peak window
#   (b) its behavior-peak window
# Both come from `phon_roc_auc_searchlight_df` (per-fold AUCs at every
# searchlight position). No transfer, no decoder weights. Spaghetti plot
# of subject means.

# %%
phon_at_phon_peak = (
    phon_peaks_foldmean_maxstat.filter(_is_significant(phon_peaks_foldmean_maxstat))
    .select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
    .join(phon_roc_auc_searchlight_df,
          on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"], how="left")
    .with_columns(pl.lit("acoustic_window").alias("evaluation"))
)
phon_at_behav_peak = (
    behav_hga_only_peaks[primary_peak_flavor]
    .filter(_is_significant(behav_hga_only_peaks[primary_peak_flavor]))
    .select(["subject", "electrode_idx", "phoneme_pair", "smin", "smax"])
    .join(phon_roc_auc_searchlight_df,
          on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"], how="left")
    .with_columns(pl.lit("perceptual_window").alias("evaluation"))
)

phon_roc_auc_comparison_df = pl.concat([phon_at_phon_peak, phon_at_behav_peak])
phon_subj_means = (
    phon_roc_auc_comparison_df
    .group_by(["subject", "evaluation"])
    .agg(pl.col("test_roc_auc").mean().alias("mean_roc_auc"))
    .pivot(on="evaluation", index="subject", values="mean_roc_auc")
    .drop_nulls()
)

if phon_subj_means.height >= 2:
    t, p = stats.ttest_rel(
        phon_subj_means["acoustic_window"].to_numpy(),
        phon_subj_means["perceptual_window"].to_numpy(),
    )
    print(f"phon decoding (acoustic vs perceptual peak window): t={t:.3f}, p={p:.3g}")

fig, ax = plt.subplots(figsize=(3, 2.75))
for row in phon_subj_means.iter_rows(named=True):
    ax.plot([0, 1], [row["acoustic_window"], row["perceptual_window"]],
            color="gray", alpha=0.6, marker="o")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Acoustic\nwindow", "Perceptual\nwindow"])
ax.set_ylabel("Acoustic ROC-AUC")
ax.set_ylim(0.4, 1.0)
ax.set_title("Cross-window acoustic decoding")
fig.tight_layout()
fig.savefig(outdir / "decoding_phonetic.pdf")
plt.close(fig)

# %% [markdown]
# ----
# # Section #4 — Behavioral decoding improvement
#
# Compare HGA-only / full (HGA + control) / control-baseline AUC at each
# behavioral-peak site. Spaghetti per subject + paired t-tests.

# %%
behav_peak_set = (
    behav_hga_only_peaks[primary_peak_flavor]
    .filter(_is_significant(behav_hga_only_peaks[primary_peak_flavor]))
    .select(["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"])
)

behav_at_peaks = (
    behav_peak_set
    .join(behav_roc_auc_searchlight_df,
          on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
          how="left")
)

# Subject means across peaks/folds
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

# Spaghetti plot
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
ax.set_title("Behavioral decoding improvement")
fig.tight_layout()
fig.savefig(outdir / "decoding_behavioral_improvement.pdf")
plt.close(fig)

# %% [markdown]
# ----
# # Section #5 — Phonetic cross-window transfer
#
# Apply trained acoustic-window decoder weights to perceptual-window HGA
# features; predict same target (acoustic). Uses `apply_decoder_weights`.
#
# Implementation outline (filled in as the parquet schemas stabilize):
#   1. For each acoustic-peak site, find its acoustic-window decoder weights
#      in decoder_weights_acoustic (rows keyed by subject/electrode_idx/
#      phoneme_pair/smin/smax/fold).
#   2. Identify the matching behavior-peak window (smin_behav/smax_behav)
#      for the same site from behav_hga_only_peaks.
#   3. Extract per-trial HGA averaged over the behavior-peak window from
#      epochs[subject], for each fold's test set.
#   4. apply_decoder_weights(...) -> proba; compute ROC-AUC vs decoder_target.

# %%
# TODO: Implement transfer evaluation. Skeleton placeholder below; emits an
# empty PDF so the rule's expected output exists.
fig, ax = plt.subplots(figsize=(3, 2.75))
ax.text(0.5, 0.5,
        "Section #5 — TODO\n(apply acoustic weights\nto perceptual window)",
        ha="center", va="center", fontsize=10)
ax.axis("off")
fig.savefig(outdir / "decoding_acoustic_transfer.pdf")
plt.close(fig)

# %% [markdown]
# ----
# # Section #6 — Behavioral cross-window transfer
#
# Apply trained behavior-window (HGA-only) decoder weights to acoustic-window
# HGA features; predict behavioral target. Same mechanism as #5 with windows
# swapped and target swapped.

# %%
# TODO: Implement transfer evaluation. Skeleton placeholder below.
fig, ax = plt.subplots(figsize=(3, 2.75))
ax.text(0.5, 0.5,
        "Section #6 — TODO\n(apply behav weights\nto acoustic window)",
        ha="center", va="center", fontsize=10)
ax.axis("off")
fig.savefig(outdir / "decoding_phon_decoder_on_behav_window.pdf")
plt.close(fig)

# %% [markdown]
# ----
# # Section #7a — Exploratory star plots (PdfPages)
#
# Multi-page PDF over all selective sites, no exemplar pre-selection.
# Modeled on hga_zoomin_search.pdf in causal4 A_neurometrics. Each page
# shows the zoomin_hga + behav_barplot for one (site, completion).

# %%
zoomin_targets = (
    zoomin_keys
    .join(behav_hga_only_peaks[primary_peak_flavor]
            .filter(_is_significant(behav_hga_only_peaks[primary_peak_flavor]))
            .select(["subject", "electrode_idx", "phoneme_pair", "word_end"]).unique(),
          on=["subject", "electrode_idx", "phoneme_pair"],
          how="inner")
    .to_pandas()
)
print(f"zoomin exploratory PDF: {len(zoomin_targets)} (site, word_end) pages")

with PdfPages(outdir / "zoomin_exploratory.pdf") as pdf:
    for _, row in tqdm(zoomin_targets.iterrows(), total=len(zoomin_targets),
                       desc="rendering star plots"):
        try:
            fb = zoomin_hga(
                paper_data,
                subject=row["subject"], electrode_idx=int(row["electrode_idx"]),
                phoneme_pair=row["phoneme_pair"], word_end=row["word_end"],
                textgrid_dir=textgrid_dir,
                figsize=(6, 4),
                title=f"{row['subject']} e{row['electrode_idx']} {row['phoneme_pair']} {row['word_end']}",
            )
            fig = fb.fig if hasattr(fb, "fig") else plt.gcf()
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            L.warning(f"zoomin failed for {row.to_dict()}: {e}")

# %% [markdown]
# ## Section #7b — Average condition contrasts (population mean HGA contrasts)
#
# Polarity-corrected mean HGA difference between heard-/d/ and heard-/n/ at
# behaviorally-selective sites. One figure per word ending.

# %%
for word_end in ["necessary", "desolate"]:
    fb = plot_condition_contrasts_single_figure(
        paper_data,
        textgrid_dir=textgrid_dir,
        plot_word_ends=(word_end,),
        ambiguous_response_threshold=ambiguous_response_threshold,
    )
    fig = fb.fig if hasattr(fb, "fig") else plt.gcf()
    fig.savefig(outdir / f"condition_contrasts-{word_end}.pdf")
    plt.close(fig)

# %% [markdown]
# ----
# # TODO: Sections #8 - #14 (deferred)
#
# These sections from causal4 A_neurometrics are out of scope for v1. Each
# stub below describes what the section does and points to its location in
# notebooks/causal4/A_neurometrics.py for reference.
#
# ## Section #8 — Early polarity x late-response-presence contingency
# Causal4 lines ~2126-2368. For each electrode with an acoustic response,
# tabulate acoustic tuning (early polarity) against whether the site also
# shows a perceptual response on each word ending. Heatmaps + stacked bar
# with chi-square.
#
# ## Section #9 — Early x late polarity contingency
# Causal4 lines ~2505-2630. Crosstab of early vs late preference direction;
# binomial test for congruence above chance.
#
# ## Section #10 — Polarity transfer per-electrode exploratory
# Causal4 lines ~2636-2772. Multi-page PDF of zoomin plots split by congruency,
# sorted by transfer effect magnitude. Builds on sections #5/#6 outputs.
#
# ## Section #11 — Late unambiguous-response generalization
# Causal4 lines ~2786-3071. Per site, t-test the perceptual-window HGA
# contrast on unambiguous trials (resampled 1, 6). Tags sites as
# late_on_unambig=True/False; emits unambig_late_df.csv.
#
# ## Section #12 — Late polarity x unambig generalization
# Causal4 lines ~3073-3109. Stackbar contingency of late perceptual tuning
# against late_on_unambig from #11.
#
# ## Section #13 — Behavioral contrast on unambiguous, split by generalization
# Causal4 lines ~3119-3382. plot_condition_contrast on unambiguous trials,
# split by late_on_unambig.
#
# ## Section #14 — Behavioral contrast on ambiguous, split by generalization
# Causal4 lines ~3384-3455. Same as #13 but on ambiguous trials.
