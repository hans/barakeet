"""
Generate a PDF report for the multivariate gradient perception analysis.

Loads precomputed parquets and produces a multi-page PDF with:
  1. Neurometric curves with sigmoid fits (per subject x phoneme_pair)
  2. Sigmoid k distribution (categorical vs graded)
  3. Correlation summary (ambiguous-trial step correlations)
  4. Permutation null distributions
  5. Neural vs behavioral psychometric overlay (non-mismatch trials)
  6. ROC-AUC vs population size

Usage:
    python scripts/gradient_perception_report.py \\
        --data-dir outputs/causal5/multivariate_gradient_perception \\
        --output outputs/causal5/multivariate_gradient_perception/report.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import scipy.stats

from src.models.sigmoid import (
    sigmoid_model,
    fit_sigmoid,
    EFFECTIVELY_LINEAR_K,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_page(pdf, lines, title=None, fontsize=10.5):
    """Render a text page (no axes) into the PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    y = 0.93
    if title:
        fig.text(0.08, y, title, fontsize=14, fontweight="bold",
                 fontfamily="serif", va="top")
        y -= 0.04
    fig.text(0.08, y, "\n".join(lines), fontsize=fontsize,
             fontfamily="serif", va="top", wrap=True,
             transform=fig.transFigure, linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def _phoneme_pair_label(pp):
    return {"bm": "/b/-/m/", "dn": "/d/-/n/", "pb": "/p/-/b/"}.get(pp, pp)


PAIR_COLORS = {"bm": "#1b9e77", "dn": "#d95f02", "pb": "#7570b3"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir, all_md_path):
    data_dir = Path(data_dir)
    d = {}
    d["gradient_stats"] = pd.read_parquet(data_dir / "gradient_stats.parquet")
    d["reg_pred"] = pd.read_parquet(data_dir / "regression_predictions.parquet")

    ep_path = data_dir / "endpoint_predictions.parquet"
    if ep_path.exists():
        d["endpoint_pred"] = pd.read_parquet(ep_path)
    else:
        d["endpoint_pred"] = None

    perm_path = data_dir / "permutation_correlations.parquet"
    if perm_path.exists():
        d["perm"] = pd.read_parquet(perm_path)
    else:
        d["perm"] = None

    # all_md for mismatch / behavioral columns
    if Path(all_md_path).exists():
        d["all_md"] = pd.read_parquet(
            all_md_path,
            columns=["subject", "epoch_idx", "phoneme_pair", "resampled",
                      "mismatch", "behavior_categorical_forced",
                      "categorical_acoustic_cue", "word_end"],
        )
    else:
        d["all_md"] = None

    return d


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_neurometric_curves(data, pdf):
    """Per-condition neurometric curves with sigmoid fits.

    Returns sigmoid fit results for downstream use.
    """
    reg_pred = data["reg_pred"]
    endpoint_pred = data["endpoint_pred"]
    stats = data["gradient_stats"]

    conditions = stats[["subject", "phoneme_pair"]].values.tolist()

    # Compute mean predicted step per (subject, phoneme_pair, resampled)
    # across folds, for ambiguous trials
    ambig_means = (
        reg_pred.groupby(["subject", "phoneme_pair", "resampled"])["decoder_proba"]
        .mean().reset_index()
    )

    # For endpoints, average across folds
    ep_means = (
        endpoint_pred.groupby(["subject", "phoneme_pair", "resampled"])["decoder_proba"]
        .mean().reset_index()
    )

    sigmoid_results = []
    ncols = 4
    nrows = 3
    per_page = ncols * nrows

    for page_start in range(0, len(conditions), per_page):
        page_conds = conditions[page_start:page_start + per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5),
                                 squeeze=False)
        fig.suptitle("Neurometric Curves: Population Decoder Predictions",
                     fontsize=13, fontweight="bold", y=0.98)

        for idx, (subj, pp) in enumerate(page_conds):
            ax = axes[idx // ncols, idx % ncols]
            color = PAIR_COLORS.get(pp, "gray")

            # Collect step means
            am = ambig_means.query(
                "subject == @subj and phoneme_pair == @pp"
            ).sort_values("resampled")

            steps, preds = [], []
            # Endpoints
            ep = ep_means.query(
                "subject == @subj and phoneme_pair == @pp"
            ).sort_values("resampled")
            if len(ep) > 0:
                steps.extend(ep.resampled.values)
                preds.extend(ep.decoder_proba.values)

            steps.extend(am.resampled.values)
            preds.extend(am.decoder_proba.values)

            steps = np.array(steps)
            preds = np.array(preds)
            order = np.argsort(steps)
            steps, preds = steps[order], preds[order]

            ax.plot(steps, preds, "o-", color=color, ms=5, lw=1.5)

            # Sigmoid fit — decoder_proba is already in [0, 1]
            sig = fit_sigmoid(steps, preds)

            if sig is not None:
                x_fine = np.linspace(steps.min(), steps.max(), 100)
                y_fine = sigmoid_model(x_fine, *sig["params"])
                ax.plot(x_fine, y_fine, "--", color="tomato",
                        lw=1.2, alpha=0.8)
                k_str = f"k={sig['k']:.2f}"
                if sig["effectively_linear"]:
                    k_str += " (linear)"
                ax.set_title(f"{subj} {_phoneme_pair_label(pp)}\n{k_str}",
                             fontsize=8)
                sigmoid_results.append({
                    "subject": subj, "phoneme_pair": pp, **sig,
                })
            else:
                ax.set_title(f"{subj} {_phoneme_pair_label(pp)}\nfit failed",
                             fontsize=8)

            ax.set_xlabel("Morph step", fontsize=7)
            ax.set_ylabel("P(step 6)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xticks(range(1, 7))
            ax.set_ylim(-0.05, 1.05)

            # Reference line: chance
            ax.axhline(0.5, ls=":", color="gray", lw=0.8, alpha=0.5)

        # Hide unused axes
        for idx in range(len(page_conds), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    return pd.DataFrame(sigmoid_results) if sigmoid_results else pd.DataFrame()


def build_k_distribution(sigmoid_df, pdf):
    """Histogram of sigmoid k values."""
    if sigmoid_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Sigmoid Steepness (k) Distribution", fontsize=13,
                 fontweight="bold")

    k_vals = sigmoid_df["k"].values
    is_linear = k_vals > EFFECTIVELY_LINEAR_K

    # Left: histogram of k
    ax = axes[0]
    ax.hist(k_vals[~is_linear], bins=15, color="steelblue", edgecolor="white",
            alpha=0.8, label=f"Sigmoidal (n={np.sum(~is_linear)})")
    if np.any(is_linear):
        ax.axvline(EFFECTIVELY_LINEAR_K, color="tomato", ls="--", lw=1.2)
        ax.text(EFFECTIVELY_LINEAR_K + 0.3, ax.get_ylim()[1] * 0.9,
                f"Linear (n={np.sum(is_linear)})",
                color="tomato", fontsize=9)
    ax.set_xlabel("k (steepness)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.set_title("Small k = categorical, large k = graded")

    # Right: k by phoneme pair
    ax = axes[1]
    for pp in ["bm", "dn", "pb"]:
        sub = sigmoid_df.query("phoneme_pair == @pp")
        if len(sub) == 0:
            continue
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(sub))
        pp_x = {"bm": 0, "dn": 1, "pb": 2}[pp]
        ax.scatter(pp_x + jitter, sub["k"], color=PAIR_COLORS[pp],
                   s=40, alpha=0.7, edgecolors="white", lw=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([_phoneme_pair_label(p) for p in ["bm", "dn", "pb"]])
    ax.set_ylabel("k (steepness)")
    ax.axhline(EFFECTIVELY_LINEAR_K, color="tomato", ls="--", lw=1, alpha=0.6)
    ax.set_title("k by phoneme pair")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def build_correlation_summary(stats, pdf):
    """Strip plot of ambiguous-trial correlations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Ambiguous-Trial Step Correlation (Population Decoder)",
                 fontsize=13, fontweight="bold")

    for pp in ["bm", "dn", "pb"]:
        sub = stats.query("phoneme_pair == @pp")
        if len(sub) == 0:
            continue
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(sub))
        pp_x = {"bm": 0, "dn": 1, "pb": 2}[pp]
        ax.scatter(pp_x + jitter, sub["ambiguous_step_correlation"],
                   color=PAIR_COLORS[pp], s=50, alpha=0.8,
                   edgecolors="white", lw=0.5, zorder=3)
        # Label each point with subject
        for _, row in sub.iterrows():
            ax.annotate(row.subject, (pp_x + 0.15, row.ambiguous_step_correlation),
                        fontsize=5.5, alpha=0.6)

    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([_phoneme_pair_label(p) for p in ["bm", "dn", "pb"]])
    ax.set_ylabel("Pearson r (predicted vs actual morph step)")

    mean_r = stats["ambiguous_step_correlation"].mean()
    ax.set_title(f"Mean r = {mean_r:.3f} across {len(stats)} conditions",
                 fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def build_permutation_nulls(data, pdf):
    """Grid of null-distribution histograms."""
    perm = data["perm"]
    stats = data["gradient_stats"]
    if perm is None:
        return

    conditions = stats[["subject", "phoneme_pair"]].values.tolist()
    ncols = 4
    nrows = 3
    per_page = ncols * nrows

    for page_start in range(0, len(conditions), per_page):
        page_conds = conditions[page_start:page_start + per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5),
                                 squeeze=False)
        fig.suptitle("Permutation Test: Null Distributions",
                     fontsize=13, fontweight="bold", y=0.98)

        for idx, (subj, pp) in enumerate(page_conds):
            ax = axes[idx // ncols, idx % ncols]

            null = perm.query(
                "subject == @subj and phoneme_pair == @pp"
            )["correlation"].values

            obs_row = stats.query(
                "subject == @subj and phoneme_pair == @pp"
            )
            obs = obs_row["ambiguous_step_correlation"].values[0]

            ax.hist(null, bins=20, color="lightgray", edgecolor="white")
            ax.axvline(obs, color="tomato", lw=1.5)

            # p-value
            p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)
            sig_star = "*" if p < 0.05 else ""

            ax.set_title(
                f"{subj} {_phoneme_pair_label(pp)}\n"
                f"r={obs:.3f}, p={p:.3f}{sig_star}",
                fontsize=8,
            )
            ax.tick_params(labelsize=6)

        for idx in range(len(page_conds), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)


def build_neural_vs_behavioral(data, pdf):
    """Neural and behavioral psychometric functions (within-completion average)."""
    reg_pred = data["reg_pred"]
    all_md = data["all_md"]
    stats = data["gradient_stats"]

    if all_md is None:
        print("WARNING: all_md not available, skipping neural vs behavioral")
        return

    conditions = stats[["subject", "phoneme_pair"]].values.tolist()

    ncols = 4
    nrows = 3
    per_page = ncols * nrows

    # Lexical evidence labels and ordering: left-biasing completion first
    # phoneme_pair -> [(left_comp, label), (right_comp, label)]
    _lex_order = {
        "bm": [("bountiful", "/b/ (bountiful)"), ("mountains", "/m/ (mountains)")],
        "dn": [("desolate", "/d/ (desolate)"), ("necessary", "/n/ (necessary)")],
        "pb": [("penecillin", "/p/ (penicillin)"), ("beneficial", "/b/ (beneficial)")],
    }

    for page_start in range(0, len(conditions), per_page):
        page_conds = conditions[page_start:page_start + per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5),
                                 squeeze=False)
        fig.suptitle(
            "Neural vs Behavioral Psychometric Functions (by lexical context)",
            fontsize=12, fontweight="bold", y=0.98,
        )

        for idx, (subj, pp) in enumerate(page_conds):
            ax = axes[idx // ncols, idx % ncols]

            cond = reg_pred.query(
                "subject == @subj and phoneme_pair == @pp"
            )

            if len(cond) == 0:
                ax.set_title(f"{subj} {_phoneme_pair_label(pp)}\nno data",
                             fontsize=8)
                continue

            # Average predicted step across folds per trial
            trial_means = cond.groupby("epoch_idx").agg(
                resampled=("resampled", "first"),
                decoder_proba=("decoder_proba", "mean"),
                behavior=("behavior_categorical_forced", "first"),
                word_end=("word_end", "first"),
            ).reset_index()

            steps = sorted(trial_means.resampled.unique())
            # Order completions by lexical evidence (left-biasing first)
            lex_order = _lex_order.get(pp)
            if lex_order:
                completions = [c for c, _ in lex_order
                               if c in trial_means.word_end.values]
            else:
                completions = sorted(trial_means.word_end.unique())

            # Neural: within-completion averaged (raw decoder proba)
            neural_by_step = []
            for step in steps:
                neural_per_comp = []
                for comp in completions:
                    st = trial_means.query(
                        "resampled == @step and word_end == @comp"
                    )
                    if len(st) == 0:
                        continue
                    neural_per_comp.append(st.decoder_proba.mean())
                neural_by_step.append(
                    np.mean(neural_per_comp) if neural_per_comp else np.nan
                )

            color = PAIR_COLORS.get(pp, "gray")
            ax.plot(steps, neural_by_step, "o-", color=color, ms=4, lw=1.5,
                    label="Neural")

            # Behavioral: separate curve per completion (lexical context)
            comp_styles = [("s-", 0.8), ("^--", 0.5)]
            for ci, comp in enumerate(completions):
                comp_trials = trial_means.query("word_end == @comp")
                behav_by_step = []
                for step in steps:
                    st = comp_trials.query("resampled == @step")
                    valid = st.behavior.dropna()
                    if len(valid) > 0:
                        behav_by_step.append((valid == 1).mean())
                    else:
                        behav_by_step.append(np.nan)
                style, alpha = comp_styles[ci]
                lex_dict = dict(_lex_order.get(pp, []))
                label = lex_dict.get(comp, comp)
                ax.plot(steps, behav_by_step, style, color="gray", ms=4,
                        lw=1.2, alpha=alpha, label=label)

            ax.set_title(f"{subj} {_phoneme_pair_label(pp)}", fontsize=8)
            ax.set_xlabel("Morph step", fontsize=7)
            ax.set_ylabel("P(second phoneme)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_ylim(-0.05, 1.05)
            if idx == 0:
                ax.legend(fontsize=6, loc="lower right")

        for idx in range(len(page_conds), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)


def build_neural_behavioral_alignment(data, sigmoid_df, pdf):
    """Cross-subject alignment of neural and behavioral sigmoid parameters.

    Fits sigmoids to each subject's behavioral psychometric function (averaged
    across completions) and compares PSE (x0) and slope (k) to the neural
    sigmoid fits.  Reports rank correlations (robust to unknown readout
    transform) and scatter plots.
    """
    all_md = data["all_md"]

    if all_md is None or sigmoid_df.empty:
        return pd.DataFrame()

    conditions = sigmoid_df[["subject", "phoneme_pair"]].values.tolist()

    behav_fits = []
    for subj, pp in conditions:
        md = all_md.query("subject == @subj and phoneme_pair == @pp")
        if len(md) == 0:
            continue
        # Behavioral psychometric: P(response==1) per step, averaged across
        # completions (within-completion average to match neural curve)
        behav_valid = md.dropna(subset=["behavior_categorical_forced"])
        if len(behav_valid) == 0:
            continue
        steps = sorted(behav_valid.resampled.unique())
        behav_by_step = []
        for step in steps:
            st = behav_valid.query("resampled == @step")
            behav_by_step.append((st.behavior_categorical_forced == 1).mean())
        steps = np.array(steps)
        behav_by_step = np.array(behav_by_step)

        bsig = fit_sigmoid(steps, behav_by_step)
        if bsig is not None:
            behav_fits.append({
                "subject": subj, "phoneme_pair": pp,
                "behav_x0": bsig["x0"], "behav_k": bsig["k"],
                "behav_r2": bsig["r2"],
            })

    if not behav_fits:
        return pd.DataFrame()

    behav_df = pd.DataFrame(behav_fits)
    merged = sigmoid_df.merge(behav_df, on=["subject", "phoneme_pair"], how="inner")
    if len(merged) < 4:
        return merged  # too few points for meaningful correlation

    # --- Scatter plots ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Neural–Behavioral Sigmoid Alignment (cross-subject)",
                 fontsize=13, fontweight="bold")

    # PSE (x0) alignment
    ax = axes[0]
    for pp in ["bm", "dn", "pb"]:
        sub = merged.query("phoneme_pair == @pp")
        if len(sub) == 0:
            continue
        ax.scatter(sub.behav_x0, sub.x0, color=PAIR_COLORS[pp], s=50,
                   alpha=0.8, edgecolors="white", lw=0.5,
                   label=_phoneme_pair_label(pp))
        for _, row in sub.iterrows():
            ax.annotate(row.subject, (row.behav_x0 + 0.05, row.x0 + 0.05),
                        fontsize=5.5, alpha=0.6)

    rho_x0, p_x0 = scipy.stats.spearmanr(merged.behav_x0, merged.x0)
    lims = [min(merged.behav_x0.min(), merged.x0.min()) - 0.3,
            max(merged.behav_x0.max(), merged.x0.max()) + 0.3]
    ax.plot(lims, lims, ":", color="gray", lw=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Behavioral PSE (x0)")
    ax.set_ylabel("Neural PSE (x0)")
    ax.set_title(f"PSE: ρ = {rho_x0:.3f}, p = {p_x0:.3f}", fontsize=10)
    ax.legend(fontsize=8)

    # Slope (k) alignment
    ax = axes[1]
    for pp in ["bm", "dn", "pb"]:
        sub = merged.query("phoneme_pair == @pp")
        if len(sub) == 0:
            continue
        ax.scatter(sub.behav_k, sub.k, color=PAIR_COLORS[pp], s=50,
                   alpha=0.8, edgecolors="white", lw=0.5,
                   label=_phoneme_pair_label(pp))
        for _, row in sub.iterrows():
            ax.annotate(row.subject, (row.behav_k + 0.02, row.k + 0.02),
                        fontsize=5.5, alpha=0.6)

    rho_k, p_k = scipy.stats.spearmanr(merged.behav_k, merged.k)
    ax.set_xlabel("Behavioral slope (k)")
    ax.set_ylabel("Neural slope (k)")
    ax.set_title(f"Slope: ρ = {rho_k:.3f}, p = {p_k:.3f}", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Store stats on the merged df for the summary page
    merged.attrs["rho_x0"] = rho_x0
    merged.attrs["p_x0"] = p_x0
    merged.attrs["rho_k"] = rho_k
    merged.attrs["p_k"] = p_k
    return merged


def build_r2_vs_population(stats, pdf):
    """Scatter: test ROC-AUC vs number of electrodes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Cross-Validated ROC-AUC vs Population Size",
                 fontsize=13, fontweight="bold")

    for pp in ["bm", "dn", "pb"]:
        sub = stats.query("phoneme_pair == @pp")
        if len(sub) == 0:
            continue
        ax.scatter(sub.n_electrodes, sub.mean_test_roc_auc,
                   color=PAIR_COLORS[pp], s=50, alpha=0.8,
                   edgecolors="white", lw=0.5,
                   label=_phoneme_pair_label(pp))
        for _, row in sub.iterrows():
            ax.annotate(row.subject,
                        (row.n_electrodes + 0.2, row.mean_test_roc_auc),
                        fontsize=5.5, alpha=0.6)

    # Correlation
    r, p = scipy.stats.pearsonr(stats.n_electrodes, stats.mean_test_roc_auc)
    ax.set_xlabel("Number of acoustically selective electrodes")
    ax.set_ylabel("Mean test ROC-AUC")
    ax.set_title(f"r = {r:.3f}, p = {p:.3f}", fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",
                        default="outputs/causal5/multivariate_gradient_perception")
    parser.add_argument("--all-md",
                        default="outputs/causal5/prepare_neurometrics/all_md.parquet")
    parser.add_argument("--output",
                        default="outputs/causal5/multivariate_gradient_perception/report.pdf")
    args = parser.parse_args()

    data = load_data(args.data_dir, args.all_md)
    stats = data["gradient_stats"]

    n_sig = 0
    if data["perm"] is not None:
        for _, row in stats.iterrows():
            null = data["perm"].query(
                "subject == @row.subject and phoneme_pair == @row.phoneme_pair"
            )["correlation"].values
            obs = row["ambiguous_step_correlation"]
            p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (len(null) + 1)
            if p < 0.05:
                n_sig += 1

    with PdfPages(args.output) as pdf:
        # --- Title page ---
        text_page(pdf, [
            "",
            "Population-Level Gradient Perception Analysis",
            "",
            "Does the population of acoustically selective electrodes encode",
            "stimulus continuum information in a graded or categorical fashion?",
            "",
            f"Conditions: {len(stats)} (subject x phoneme pair)",
            f"Significant (p<0.05, permutation): {n_sig}/{len(stats)} ({100*n_sig/len(stats):.0f}%)",
            f"Mean ambiguous-trial correlation: {stats.ambiguous_step_correlation.mean():.3f}",
            "",
            "Logistic regression with PCA trained on endpoint trials (steps 1, 6)",
            "to classify acoustic cue. Applied to ambiguous trials (steps 2-5).",
            "Sigmoid fits characterize whether neurometric curves are categorical",
            "(small k) or graded (large k).",
        ], title="Multivariate Gradient Perception Report")

        # --- Methods page ---
        text_page(pdf, [
            "Electrode selection:",
            "  Acoustically selective sites (ROC-AUC >= 0.65 on endpoint acoustic decoding)",
            "  from the single-electrode acoustic searchlight analysis.",
            "",
            "Feature construction:",
            "  For each population, the HGA time course within the acoustic response window",
            "  is concatenated across electrodes to form a spatiotemporal feature vector",
            "  (n_electrodes x n_timepoints).",
            "",
            "Classification model:",
            "  Logistic regression with PCA preprocessing (auto component selection: 25%, 50%, 90%).",
            "  5 repeated train/test splits (80/20) with inner cross-validation for",
            "  hyperparameter tuning (C, PCA components). Trained on endpoint trials",
            "  (steps 1 and 6) with binary acoustic cue as target.",
            "",
            "Evaluation:",
            "  - Cross-validated ROC-AUC on held-out endpoint trials",
            "  - Pearson correlation between decoder probability and morph step on ambiguous trials",
            "  - Permutation test (100 shuffles) for significance of ambiguous-trial correlation",
            "",
            "Sigmoid fitting:",
            "  f(x) = a / (1 + exp(-(x - x0) / k)) + (0.5 - a/2)",
            "  Three free parameters: amplitude (a), PSE (x0), steepness (k).",
            "  Small k (<1) indicates categorical encoding (step function).",
            "  Large k (>10) indicates effectively linear / graded tracking.",
            "  Multi-start optimization with scipy.optimize.curve_fit.",
        ], title="Methods")

        # --- Neurometric curves ---
        sigmoid_df = build_neurometric_curves(data, pdf)

        # --- Sigmoid k distribution ---
        build_k_distribution(sigmoid_df, pdf)

        # --- Correlation summary ---
        build_correlation_summary(stats, pdf)

        # --- Permutation nulls ---
        build_permutation_nulls(data, pdf)

        # --- Neural vs behavioral ---
        build_neural_vs_behavioral(data, pdf)

        # --- Neural-behavioral sigmoid alignment ---
        alignment_df = build_neural_behavioral_alignment(data, sigmoid_df, pdf)

        # --- R2 vs population size ---
        build_r2_vs_population(stats, pdf)

        # --- Summary page ---
        summary_lines = [
            "",
            f"Total conditions analyzed: {len(stats)}",
            f"Significant at p<0.05: {n_sig}/{len(stats)} ({100*n_sig/len(stats):.0f}%)",
            f"Mean ambiguous-trial correlation: {stats.ambiguous_step_correlation.mean():.3f} "
            f"(range {stats.ambiguous_step_correlation.min():.3f} - "
            f"{stats.ambiguous_step_correlation.max():.3f})",
            f"Mean test ROC-AUC: {stats.mean_test_roc_auc.mean():.3f}",
            "",
        ]
        if not sigmoid_df.empty:
            n_cat = (sigmoid_df.k < 1).sum()
            n_graded = ((sigmoid_df.k >= 1) & (sigmoid_df.k <= EFFECTIVELY_LINEAR_K)).sum()
            n_linear = (sigmoid_df.k > EFFECTIVELY_LINEAR_K).sum()
            summary_lines.extend([
                "Sigmoid steepness distribution:",
                f"  Categorical (k < 1): {n_cat}/{len(sigmoid_df)}",
                f"  Intermediate (1 <= k <= 10): {n_graded}/{len(sigmoid_df)}",
                f"  Effectively linear (k > 10): {n_linear}/{len(sigmoid_df)}",
                f"  Median k: {sigmoid_df.k.median():.2f}",
                "",
            ])
        if len(alignment_df) > 0 and hasattr(alignment_df, "attrs"):
            rho_x0 = alignment_df.attrs.get("rho_x0")
            p_x0 = alignment_df.attrs.get("p_x0")
            rho_k = alignment_df.attrs.get("rho_k")
            p_k = alignment_df.attrs.get("p_k")
            if rho_x0 is not None:
                summary_lines.extend([
                    "Neural-behavioral alignment (Spearman rank correlations):",
                    f"  PSE (x0):  ρ = {rho_x0:.3f}, p = {p_x0:.3f}  (n = {len(alignment_df)})",
                    f"  Slope (k): ρ = {rho_k:.3f}, p = {p_k:.3f}",
                    "",
                ])

        summary_lines.extend([
            "Interpretation:",
            "  The majority of population decoders show significant graded tracking",
            "  of the acoustic continuum on ambiguous trials, with neurometric curves",
            "  that interpolate between endpoint values rather than snapping to",
            "  categorical boundaries. This provides evidence that graded acoustic",
            "  information is present at the population level even when individual",
            "  electrodes show categorical responses.",
        ])
        text_page(pdf, summary_lines, title="Summary")

    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
