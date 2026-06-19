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
# # Type-1 early acoustic decoder output on ambiguous trials
#
# At sites classified as **type1 (acoustic only)** — no detectable perceptual response in
# the early window — we ask what the early acoustic decoder "thinks" when it sees an
# ambiguous stimulus.
#
# Two competing predictions:
# - **Gradient encoding**: decoder output tracks subphonemic acoustic variation; output
#   drifts smoothly from ~0 at step 1 through intermediate values at steps 2–5 to ~1
#   at step 6.
# - **Categorical encoding**: decoder output is committed even on ambiguous stimuli; the
#   distribution at steps 2–5 is bimodal near 0 and 1.
#
# Secondary question: does decoder output on ambiguous steps predict the participant's
# reported percept (hue split by behavioral response)?
#
# **Note**: this notebook requires epoch files in `epoch_dir`; subjects without a `.fif`
# file are skipped with a warning. Run in production where all subjects' files are present.

# %% tags=["parameters"]
annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
phon_peaks_root  = "outputs/causal6/acoustic_decoding_peaks"
epoch_dir        = "outputs/epochs_preprocessed"
config_path      = "config.yaml"
reg_lambda_winners_path = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json"
outdir           = "outputs/causal46_joined/type1_early_decoder_on_ambiguous"
device           = "cpu"

# %% [markdown]
# ## Imports and configuration

# %%
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import polars as pl
import yaml
from loguru import logger as L
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "notebooks" / "causal46_joined"))

from src.data import add_metadata_features
from src.models.causal6 import run_acoustic_searchlight

from _within_completion import resolve_behavior_col

# %%
_cfg = yaml.safe_load(Path(config_path).read_text())
_winners = json.loads(Path(reg_lambda_winners_path).read_text())

reg_lambda      = float(_winners["reg_lambda_acoustic"])
n_folds         = int(_cfg["causal6"]["n_folds"])
cv_random_state = int(_cfg["causal6"]["cv_random_state"])
tol             = float(_cfg["causal6"]["tol"])
max_iter        = int(_cfg["causal6"]["max_iter"])

print(
    f"reg_lambda={reg_lambda}  n_folds={n_folds}  cv_random_state={cv_random_state}"
    f"  tol={tol}  max_iter={max_iter}  device={device!r}"
)

# %% [markdown]
# ## Synthetic reconstruction check
#
# Verifies the no-intercept sigmoid reconstruction formula against sklearn before
# applying it to real coefficients. This is the only correctness check we can run
# locally without epoch files.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
N, D = 60, 15
X_toy = rng.standard_normal((N, D))
y_toy = (rng.random(N) > 0.5).astype(int)

sc = StandardScaler()
X_toy_std = sc.fit_transform(X_toy)

sk = LogisticRegression(C=1.0, fit_intercept=False, solver="lbfgs", max_iter=200)
sk.fit(X_toy_std, y_toy)
sk_proba = sk.predict_proba(X_toy_std)[:, 1]

w = sk.coef_[0].astype(np.float64)
mu = sc.mean_.astype(np.float64)
sigma = sc.scale_.astype(np.float64)
X_rec = (X_toy - mu) / sigma
rec_proba = 1.0 / (1.0 + np.exp(-(X_rec @ w)))

max_diff = np.abs(sk_proba - rec_proba).max()
assert max_diff < 1e-10, f"Reconstruction mismatch: max_diff={max_diff:.2e}"
print(f"Reconstruction check passed (max_diff={max_diff:.2e})")

del rng, N, D, X_toy, y_toy, sc, X_toy_std, sk, sk_proba, w, mu, sigma, X_rec, rec_proba, max_diff

# %% [markdown]
# ## Load type1 site list

# %%
annotations = pl.read_csv(annotations_path)
type1_sites = annotations.filter(
    pl.col("site_type_relabel") == "type1_acoustic_only"
).select(["subject", "electrode_idx", "phoneme_pair", "status", "acoustic_sign"])

n_type1 = len(type1_sites)
n_ok    = int((type1_sites["status"] == "ok").sum())
L.info(f"Type1 sites: {n_type1} total ({n_ok} ok, {n_type1 - n_ok} non-ok status)")
print(type1_sites["status"].value_counts())
print(type1_sites["subject"].value_counts())

# %% [markdown]
# ## Phoneme-pair → label mapping

# %%
_PHONEME_LABELS = {
    "dn": {0: "heard /d/", 1.0: "heard /n/"},
    "bm": {0: "heard /b/", 1.0: "heard /m/"},
    "pb": {0: "heard /p/", 1.0: "heard /b/"},
}

def _bhv_label(phoneme_pair: str, val: float) -> str:
    return _PHONEME_LABELS.get(phoneme_pair, {}).get(val, str(val))

# %% [markdown]
# ## Main loop: refit acoustic decoder per type1 site, score ambiguous trials

# %%
epochs_cache: dict[str, mne.Epochs] = {}

def _load_epochs(subject: str) -> mne.Epochs | None:
    fif_path = Path(epoch_dir) / f"{subject}_epo.fif"
    if not fif_path.exists():
        L.warning(f"Epoch file not found for {subject}: {fif_path} — skipping")
        return None
    ep = mne.read_epochs(str(fif_path), preload=True, verbose="WARNING")
    ep.metadata = add_metadata_features(ep.metadata.copy())
    L.info(f"Loaded epochs for {subject}: {len(ep)} epochs")
    return ep


def _reconstruct_proba(coefs_pp: pl.DataFrame, X: np.ndarray) -> np.ndarray:
    """Average sigmoid(X_std @ w) across folds. X shape: (n_trials, win_size)."""
    fold_probas = []
    for row in coefs_pp.iter_rows(named=True):
        w     = np.array(row["coef"],  dtype=np.float64)
        mu    = np.array(row["mean"],  dtype=np.float64)
        sigma = np.array(row["scale"], dtype=np.float64)
        X_std = (X - mu) / sigma
        proba = 1.0 / (1.0 + np.exp(-(X_std @ w)))
        fold_probas.append(proba)
    return np.mean(fold_probas, axis=0)


# Per-subject group iteration so we load epochs once per subject.
subjects_in_annotations = type1_sites["subject"].unique().to_list()

all_rows: list[dict] = []
tier_a_done = False  # run Tier A exactly once

for subject in sorted(subjects_in_annotations):
    ep = _load_epochs(subject)
    if ep is None:
        continue

    assert ep.metadata is not None
    md = ep.metadata
    # Assert contiguous 0..N-1 index (needed for get_data positional slicing in Tier A).
    assert list(md.index) == list(range(len(md))), (
        f"[{subject}] Unexpected metadata index — expected RangeIndex(0..N-1)"
    )
    epoch_data_full = ep.get_data()  # (N_total, n_electrodes, n_times)

    # Load peak windows for this subject.
    peaks_path = Path(phon_peaks_root) / subject / "phon_peaks.parquet"
    if not peaks_path.exists():
        L.warning(f"[{subject}] phon_peaks.parquet not found at {peaks_path} — skipping")
        continue
    phon_peaks = pl.read_parquet(peaks_path)

    # Subject's type1 sites.
    subject_sites = type1_sites.filter(pl.col("subject") == subject)

    # ---------- run_acoustic_searchlight for all electrodes of this subject at once ----------
    # We call it per phoneme_pair to avoid combining phoneme pairs in one batch.
    pp_groups = subject_sites.group_by("phoneme_pair")

    for (phoneme_pair,), pp_rows in pp_groups:
        # pp_rows is a DataFrame of sites for one phoneme_pair.
        electrode_idxs_pp = pp_rows["electrode_idx"].to_list()

        # Gather (smin, smax) windows from phon_peaks for these electrodes.
        windows_list = []
        valid_electrode_idxs = []
        for eidx in electrode_idxs_pp:
            peak_row = phon_peaks.filter(
                (pl.col("electrode_idx") == eidx) & (pl.col("phoneme_pair") == phoneme_pair)
            )
            if len(peak_row) == 0:
                L.warning(f"[{subject} e{eidx} {phoneme_pair}] No peak row in phon_peaks — skipping electrode")
                continue
            smin = int(peak_row["smin"][0])
            smax = int(peak_row["smax"][0])
            windows_list.append([smin, smax])
            valid_electrode_idxs.append(eidx)

        if not valid_electrode_idxs:
            continue

        windows_arr = np.array(windows_list, dtype=np.int64)  # (n_elec, 2)

        # Each electrode gets its own window, so we call one electrode at a time
        # to avoid the all-combinations expansion in run_acoustic_searchlight.
        for eidx, (smin, smax) in zip(valid_electrode_idxs, windows_list):
            site_row = pp_rows.filter(pl.col("electrode_idx") == eidx).row(0, named=True)
            acoustic_sign = float(site_row["acoustic_sign"])
            status        = str(site_row["status"])
            if status != "ok":
                L.warning(f"[{subject} e{eidx} {phoneme_pair}] status={status!r} (non-ok; proceeding anyway)")

            peak_row = phon_peaks.filter(
                (pl.col("electrode_idx") == eidx) & (pl.col("phoneme_pair") == phoneme_pair)
            )
            logged_auc = float(peak_row["test_roc_auc"][0])

            scores, preds, coefs = run_acoustic_searchlight(
                ep,
                subject=subject,
                electrode_idxs=[eidx],
                windows=np.array([[smin, smax]], dtype=np.int64),
                target="categorical_acoustic_cue",
                reg_lambda=reg_lambda,
                n_folds=n_folds,
                cv_random_state=cv_random_state,
                device=device,
                tol=tol,
                max_iter=max_iter,
            )

            # Filter to this phoneme_pair's results.
            coefs_pp = coefs.filter(
                (pl.col("phoneme_pair") == phoneme_pair)
                & (pl.col("electrode_idx") == eidx)
                & (pl.col("smin") == smin) & (pl.col("smax") == smax)
            )
            preds_pp = preds.filter(
                (pl.col("phoneme_pair") == phoneme_pair)
                & (pl.col("electrode_idx") == eidx)
                & (pl.col("smin") == smin) & (pl.col("smax") == smax)
            )
            scores_pp = scores.filter(
                (pl.col("phoneme_pair") == phoneme_pair)
                & (pl.col("electrode_idx") == eidx)
                & (pl.col("smin") == smin) & (pl.col("smax") == smax)
            )

            # ---------- Tier A: per-trial proba sanity check (first site with canonical preds) ----------
            if not tier_a_done:
                canonical_path = Path(f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/predictions.parquet")
                if canonical_path.exists():
                    canonical = pl.read_parquet(canonical_path)
                    canonical_site = canonical.filter(
                        (pl.col("electrode_idx") == eidx)
                        & (pl.col("phoneme_pair") == phoneme_pair)
                        & (pl.col("smin") == smin) & (pl.col("smax") == smax)
                        & (pl.col("target") == "categorical_acoustic_cue")
                    )
                    if len(canonical_site) > 0:
                        fold0_row = coefs_pp.filter(pl.col("fold") == 0).row(0, named=True)
                        w_f0    = np.array(fold0_row["coef"],  dtype=np.float64)
                        mu_f0   = np.array(fold0_row["mean"],  dtype=np.float64)
                        sig_f0  = np.array(fold0_row["scale"], dtype=np.float64)
                        fold0_canon = canonical_site.filter(pl.col("fold") == 0).sort("epoch_idx")
                        ep_idxs_f0  = fold0_canon["epoch_idx"].to_numpy().astype(int)  # labels == positions (asserted above)
                        X_f0 = epoch_data_full[ep_idxs_f0, eidx, smin:smax].astype(np.float64)
                        X_f0_std = (X_f0 - mu_f0) / sig_f0
                        rec_proba = 1.0 / (1.0 + np.exp(-(X_f0_std @ w_f0)))
                        canon_proba = fold0_canon["decoder_proba"].to_numpy().astype(np.float64)
                        max_diff = np.abs(rec_proba - canon_proba).max()
                        assert max_diff < 0.02, (
                            f"Tier A proba mismatch (max_diff={max_diff:.5f}) for "
                            f"{subject} e{eidx} {phoneme_pair}"
                        )
                        L.info(f"Tier A passed: max_diff={max_diff:.5f} for {subject} e{eidx} {phoneme_pair}")
                        tier_a_done = True

            # ---------- Tier B: fold-mean AUC check ----------
            refitted_auc = float(scores_pp["test_roc_auc"].mean())
            auc_diff = abs(refitted_auc - logged_auc)
            if auc_diff >= 0.02:
                L.warning(
                    f"[{subject} e{eidx} {phoneme_pair}] Tier B AUC mismatch: "
                    f"refitted={refitted_auc:.4f} logged={logged_auc:.4f} diff={auc_diff:.4f}"
                )
            else:
                L.info(
                    f"[{subject} e{eidx} {phoneme_pair}] Tier B OK: "
                    f"refitted={refitted_auc:.4f} logged={logged_auc:.4f}"
                )

            # ---------- Endpoint predictions (steps 1 and 6) from test-fold held-out preds ----------
            pp_ep_mask = (md["phoneme_pair"] == phoneme_pair).values & md["resampled"].isin([1, 6]).values
            ep_md_endpt = md[pp_ep_mask]
            endpt_epoch_idxs = ep_md_endpt.index.to_numpy()

            preds_pp_pd = preds_pp.to_pandas()
            # Each endpoint trial appears exactly once across folds.
            for _, pred_row in preds_pp_pd.iterrows():
                ep_idx = int(pred_row["epoch_idx"])
                md_row = md.loc[ep_idx]
                decoder_proba = float(pred_row["decoder_proba"])
                aligned = decoder_proba if acoustic_sign >= 0 else 1.0 - decoder_proba
                all_rows.append({
                    "subject":                   subject,
                    "electrode_idx":             eidx,
                    "phoneme_pair":              phoneme_pair,
                    "epoch_idx":                 ep_idx,
                    "resampled":                 float(md_row["resampled"]),
                    "word_end":                  str(md_row["word_end"]) if "word_end" in md.columns else "",
                    "behavior_categorical_forced": float(md_row["behavior_categorical_forced"]),
                    "behavior_dummy_forced":     md_row["behavior_dummy_forced"],
                    "decoder_proba":             decoder_proba,
                    "decoder_proba_aligned":     aligned,
                    "split":                     "endpoint",
                    "acoustic_sign":             acoustic_sign,
                    "logged_auc":                logged_auc,
                    "site_label":                f"{subject} e{eidx} {phoneme_pair}",
                })

            # ---------- Ambiguous trials (steps 2-5) ----------
            pp_ambig_mask = (
                (md["phoneme_pair"] == phoneme_pair).values
                & md["resampled"].isin([2, 3, 4, 5]).values
            )
            if pp_ambig_mask.sum() == 0:
                L.warning(f"[{subject} e{eidx} {phoneme_pair}] No ambiguous trials found")
                continue

            X_ambig = epoch_data_full[pp_ambig_mask, eidx, smin:smax].astype(np.float64)
            proba_ambig = _reconstruct_proba(coefs_pp, X_ambig)

            md_ambig = md[pp_ambig_mask].copy()
            md_ambig_epoch_idxs = md_ambig.index.to_numpy()

            for i, ep_idx in enumerate(md_ambig_epoch_idxs):
                md_row = md_ambig.iloc[i]
                decoder_proba = float(proba_ambig[i])
                aligned = decoder_proba if acoustic_sign >= 0 else 1.0 - decoder_proba
                all_rows.append({
                    "subject":                   subject,
                    "electrode_idx":             eidx,
                    "phoneme_pair":              phoneme_pair,
                    "epoch_idx":                 int(ep_idx),
                    "resampled":                 float(md_row["resampled"]),
                    "word_end":                  str(md_row["word_end"]) if "word_end" in md.columns else "",
                    "behavior_categorical_forced": float(md_row["behavior_categorical_forced"]),
                    "behavior_dummy_forced":     md_row["behavior_dummy_forced"],
                    "decoder_proba":             decoder_proba,
                    "decoder_proba_aligned":     aligned,
                    "split":                     "ambiguous",
                    "acoustic_sign":             acoustic_sign,
                    "logged_auc":                logged_auc,
                    "site_label":                f"{subject} e{eidx} {phoneme_pair}",
                })

# %% [markdown]
# ## Concatenate and write trial_df.parquet

# %%
if not all_rows:
    print("WARNING: No rows collected — no epoch files were found. Outputs cannot be written.")
else:
    trial_df = pd.DataFrame(all_rows)
    n_sites_collected = trial_df[["subject", "electrode_idx", "phoneme_pair"]].drop_duplicates().shape[0]
    print(f"Collected {len(trial_df)} rows from {n_sites_collected} sites")
    print(trial_df.dtypes)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    trial_df_pl = pl.from_pandas(trial_df)
    trial_df_pl.write_parquet(outdir_path / "trial_df.parquet")
    print(f"Wrote trial_df.parquet to {outdir_path}")

# %% [markdown]
# ## Figure 1 — Per-site catplots
#
# One page per site. x-axis = resampled step (categorical), y-axis = decoder_proba.
# Hue = behavioral response. Two columns = word_end facets.

# %%
if all_rows:
    _BHV_COLORS = {
        "heard /d/": "#2166ac", "heard /n/": "#d6604d",
        "heard /b/": "#2166ac", "heard /m/": "#d6604d",
        "heard /p/": "#2166ac", "heard /b/": "#d6604d",
    }
    _DEFAULT_COLORS = ["#2166ac", "#d6604d"]

    bhv_col = resolve_behavior_col(trial_df)

    def _ci95_bootstrap(x: np.ndarray, n_boot: int = 2000, rng_seed: int = 0) -> tuple[float, float]:
        rng = np.random.default_rng(rng_seed)
        boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    site_order_for_pdf = sorted(trial_df["site_label"].unique())

    with PdfPages(outdir_path / "per_site_catplots.pdf") as pdf:
        for site_label in site_order_for_pdf:
            site_df = trial_df[trial_df["site_label"] == site_label]
            subject   = str(site_df["subject"].iloc[0])
            eidx      = int(site_df["electrode_idx"].iloc[0])
            pp        = str(site_df["phoneme_pair"].iloc[0])
            auc       = float(site_df["logged_auc"].iloc[0])
            word_ends = sorted(site_df["word_end"].dropna().unique())
            n_we      = max(1, len(word_ends))

            fig, axes = plt.subplots(1, n_we, figsize=(5 * n_we, 4), sharey=True, squeeze=False)

            for col_i, we in enumerate(word_ends if word_ends else [None]):
                ax = axes[0, col_i]
                if we is not None:
                    sub_df = site_df[site_df["word_end"] == we]
                    ax.set_title(we, fontsize=9)
                else:
                    sub_df = site_df
                    ax.set_title("(all word_ends)", fontsize=9)

                steps = sorted(sub_df["resampled"].dropna().unique())
                steps_cat = [str(int(s)) for s in steps]
                bhv_vals  = sorted(sub_df[bhv_col].dropna().unique())
                colors    = _DEFAULT_COLORS[:len(bhv_vals)]

                for bi, bval in enumerate(bhv_vals):
                    blabel = _bhv_label(pp, bval)
                    color  = colors[bi]
                    x_dodge = (bi - (len(bhv_vals) - 1) / 2) * 0.15

                    # means, lo95, hi95 = [], [], []
                    for step in steps:
                        mask = (sub_df["resampled"] == step) & (sub_df[bhv_col] == bval)
                        vals = sub_df.loc[mask, "decoder_proba"].values
                        if len(vals) == 0:
                            # means.append(np.nan); lo95.append(np.nan); hi95.append(np.nan)
                            continue
                        # m = float(vals.mean())
                        # means.append(m)
                        # if len(vals) >= 5:
                        #     lo, hi = _ci95_bootstrap(vals)
                        # else:
                        #     lo, hi = m, m
                        # lo95.append(lo); hi95.append(hi)

                        x_strip = np.full(len(vals), steps.index(step)) + x_dodge
                        x_jitter = x_strip + np.random.default_rng(42).uniform(-0.06, 0.06, len(vals))
                        ax.scatter(x_jitter, vals, color=color, alpha=0.25, s=10, zorder=1)

                    
                    # ax.plot(x_line, means, color=color, lw=1.5, label=blabel, zorder=3)
                    # ax.errorbar(
                    #     x_line, means,
                    #     yerr=[
                    #         np.array(means) - np.array(lo95),
                    #         np.array(hi95) - np.array(means),
                    #     ],
                    #     fmt="none", color=color, capsize=3, zorder=3,
                    # )

                # plot mean activation at different resampled steps, not split by behavior
                means = sub_df.groupby("resampled")["decoder_proba"].mean().reindex(steps).values
                overall_bootstrap = sub_df.groupby("resampled")["decoder_proba"].apply(lambda x: _ci95_bootstrap(x.to_numpy())).reindex(steps)
                overall_lo95 = [lo for lo, hi in overall_bootstrap]
                overall_hi95 = [hi for lo, hi in overall_bootstrap]
                x_line = np.arange(len(steps)) + x_dodge
                
                ax.plot(x_line, sub_df.groupby("resampled")["decoder_proba"].mean().reindex(steps).values,
                        color="black", lw=2.0, zorder=2)
                ax.errorbar(
                    x_line, means,
                    yerr=[
                        means - np.array(overall_lo95),
                        np.array(overall_hi95) - means,
                    ],
                    fmt="none", color="black", capsize=5, zorder=3,
                )

                ax.set_xticks(range(len(steps)))
                ax.set_xticklabels(steps_cat)
                ax.set_xlabel("resampled step")
                ax.set_ylabel("decoder_proba")
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.5, color="gray", ls="--", lw=0.5, zorder=0)
                if col_i == n_we - 1:
                    ax.legend(fontsize=7, loc="upper left")

            fig.suptitle(
                f"{subject} e{eidx} {pp}  (site_type=type1)  AUC={auc:.3f}",
                fontsize=10,
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote per_site_catplots.pdf ({len(site_order_for_pdf)} pages)")

# %% [markdown]
# ## Figure 2 — Aggregate summary
#
# One column per site (sorted by AUC descending), one dot per resampled step,
# colored by step using a diverging colormap.

# %%
if all_rows:
    # Site order: sort by logged_auc descending.
    site_auc = (
        trial_df.groupby("site_label")["logged_auc"].first()
        .sort_values(ascending=False)
    )
    site_order = list(site_auc.index)
    n_sites = len(site_order)

    step_values = [1, 2, 3, 4, 5, 6]
    step_colors = plt.cm.RdBu_r(np.linspace(0.05, 0.95, len(step_values)))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_sites * 0.35), 7), sharex=True)

    # --- Panel A: mean across all trials (pooled behavior) ---
    ax = axes[0]
    for step_i, step in enumerate(step_values):
        x_base = np.arange(n_sites) + step_i * 0.12
        site_means = []
        for sl in site_order:
            vals = trial_df.loc[
                (trial_df["site_label"] == sl) & (trial_df["resampled"] == step),
                "decoder_proba_aligned"
            ].values
            site_means.append(float(vals.mean()) if len(vals) > 0 else np.nan)
        ax.scatter(
            x_base, site_means,
            color=step_colors[step_i], s=20, alpha=0.85, label=f"step {step}",
        )
    ax.set_ylabel("decoder_proba_aligned")
    ax.set_title("All trials pooled")
    ax.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="resampled step", loc="upper right", fontsize=7, ncol=3)

    # --- Panel B: split by behavior (-1 vs +1) ---
    ax = axes[1]
    bhv_styles = {-1.0: dict(marker="v", alpha=0.5), 1.0: dict(marker="^", alpha=0.5)}
    for step_i, step in enumerate(step_values):
        for bi, bval in enumerate([-1.0, 1.0]):
            x_base = np.arange(n_sites) + step_i * 0.12 + bi * 0.04
            site_means = []
            for sl in site_order:
                vals = trial_df.loc[
                    (trial_df["site_label"] == sl)
                    & (trial_df["resampled"] == step)
                    & (trial_df["behavior_categorical_forced"] == bval),
                    "decoder_proba_aligned",
                ].values
                site_means.append(float(vals.mean()) if len(vals) > 0 else np.nan)
            ax.scatter(
                x_base, site_means,
                color=step_colors[step_i], s=15,
                **bhv_styles[bval],
            )
    ax.set_ylabel("decoder_proba_aligned")
    ax.set_title("Split by behavior (▲=+1, ▼=−1)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(n_sites))
    ax.set_xticklabels(site_order, rotation=90, fontsize=5)

    plt.tight_layout()
    fig.savefig(outdir_path / "aggregate_figure.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote aggregate_figure.pdf ({n_sites} sites)")
