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
annotations_path    = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
trial_balance_path  = "outputs/causal46_joined/trial_balance_index.csv"
phon_peaks_root     = "outputs/causal6/acoustic_decoding_peaks"
epoch_dir           = "outputs/epochs_preprocessed"
config_path         = "config.yaml"
reg_lambda_winners_path = "outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json"
outdir              = "outputs/causal46_joined/type1_early_decoder_on_ambiguous"
device              = "cpu"

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
from src.stimuli import PHONEME_PAIR_TO_WORD_ENDS
from _within_completion import (
    n_per_class_from_per_step,
    per_step_class_counts,
    select_cell_trials_bootstrap,
)

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
# ## Load trial balance and build qualifying-steps lookup
#
# `is_ambiguous_step` is electrode-independent (purely behavioral), so we can
# build the lookup from any electrode and use it for all type1 sites.

# %%
_trial_balance = pl.read_csv(trial_balance_path)
_ambig_rows = (
    _trial_balance
    .filter(pl.col("is_ambiguous_step"))
    .select(["subject", "phoneme_pair", "word_end", "resampled"])
    .unique()
)
# step_lookup[(subject, phoneme_pair, word_end)] → sorted list of qualifying resampled steps
step_lookup: dict[tuple, list[float]] = {}
for _r in _ambig_rows.iter_rows(named=True):
    _key = (_r["subject"], _r["phoneme_pair"], _r["word_end"])
    step_lookup.setdefault(_key, []).append(float(_r["resampled"]))
for _key in step_lookup:
    step_lookup[_key] = sorted(step_lookup[_key])

print(f"step_lookup: {len(step_lookup)} (subject, phoneme_pair, word_end) combinations")
print("Example (EC243, dn, necessary):",
      step_lookup.get(("EC243", "dn", "necessary"), "not found"))

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

            # ---------- Ambiguous trials: per word_end, using subject-specific qualifying steps ----------
            # qualifying steps come from trial_balance_index.csv (is_ambiguous_step==True),
            # which reflects where the subject actually gave both response types.
            pp_word_ends = PHONEME_PAIR_TO_WORD_ENDS.get(phoneme_pair, [])
            any_ambig = False

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
            for we in pp_word_ends:
                qualifying_steps = step_lookup.get((subject, phoneme_pair, we), [])
                if not qualifying_steps:
                    L.warning(
                        f"[{subject} e{eidx} {phoneme_pair} {we}] "
                        "No qualifying ambiguous steps in trial_balance"
                    )
                    continue

                pp_we_mask = (
                    (md["phoneme_pair"] == phoneme_pair).values
                    & (md["word_end"] == we).values
                    & md["resampled"].isin(qualifying_steps).values
                )
                if pp_we_mask.sum() == 0:
                    L.warning(
                        f"[{subject} e{eidx} {phoneme_pair} {we}] "
                        f"Qualifying steps {qualifying_steps} but no matching epochs"
                    )
                    continue

                any_ambig = True
                X_ambig = epoch_data_full[pp_we_mask, eidx, smin:smax].astype(np.float64)
                proba_ambig = _reconstruct_proba(coefs_pp, X_ambig)

                md_ambig = md[pp_we_mask].copy()
                qualifying_steps_str = ",".join(str(int(s)) for s in qualifying_steps)

                for i, (_, md_row) in enumerate(md_ambig.iterrows()):
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
                        "qualifying_steps":           qualifying_steps_str,
                        "acoustic_sign":             acoustic_sign,
                        "logged_auc":                logged_auc,
                        "site_label":                f"{subject} e{eidx} {phoneme_pair}",
                    })

            if not any_ambig:
                L.warning(f"[{subject} e{eidx} {phoneme_pair}] No qualifying ambiguous trials across any word_end")

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
# One page per site. x-axis = qualifying ambiguous resampled steps (subject-specific),
# y-axis = decoder_proba. Hue = behavioral response. Two columns = word_end facets.
# Only `split == "ambiguous"` trials are shown — these are steps where the subject
# gave both response types (from trial_balance_index.csv).

# %%
if all_rows:
    _DEFAULT_COLORS = ["#2166ac", "#d6604d"]

    # trial_df uses "behavior_categorical_forced" (built above); don't call
    # resolve_behavior_col which looks for different column names.
    bhv_col = "behavior_categorical_forced"

    R_PLOT = 200  # bootstrap replicates for mean ± SE overlay

    site_order_for_pdf = sorted(trial_df["site_label"].unique())

    with PdfPages(outdir_path / "per_site_catplots.pdf") as pdf:
        for site_label in site_order_for_pdf:
            site_df = trial_df[
                (trial_df["site_label"] == site_label) & (trial_df["split"] == "ambiguous")
            ]
            if len(site_df) == 0:
                continue
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
                    qs_str = sub_df["qualifying_steps"].iloc[0] if len(sub_df) else ""
                else:
                    sub_df = site_df
                    qs_str = ""

                qualifying_steps_ints = (
                    [int(s) for s in qs_str.split(",") if s.strip()]
                    if qs_str else []
                )
                sub_df_reset = sub_df.reset_index(drop=True)

                if not qualifying_steps_ints or len(sub_df_reset) == 0:
                    title_we = we if we else "(all word_ends)"
                    ax.set_title(f"{title_we}  (no qualifying steps)", fontsize=9)
                    continue

                # word_end for per_step_class_counts: sub_df is already filtered to
                # this we, so the internal we_mask is all-True; pass the real value
                # so the function's column check works.
                _we_for_counts = we if we is not None else str(sub_df_reset["word_end"].iloc[0])
                per_step = per_step_class_counts(
                    sub_df_reset,
                    word_end=_we_for_counts,
                    qualifying_steps=qualifying_steps_ints,
                    group_col=bhv_col,
                )
                n_per_class = n_per_class_from_per_step(per_step)

                title_we = we if we else "(all word_ends)"
                ax.set_title(
                    f"{title_we}  steps:{qs_str}  n≈{n_per_class}/class",
                    fontsize=9,
                )

                steps = sorted(per_step.keys())
                steps_cat = [str(s) for s in steps]
                bhv_vals = sorted(sub_df_reset[bhv_col].dropna().unique())
                colors = _DEFAULT_COLORS[:len(bhv_vals)]

                # --- Bootstrap replicates: per-step per-class mean decoder_proba ---
                boot_step_class: dict = {s: {int(b): [] for b in bhv_vals} for s in steps}
                for r_i in range(R_PLOT):
                    rng_r = np.random.default_rng(r_i)
                    for step, by_class in per_step.items():
                        n_s = min(len(v) for v in by_class.values())
                        if n_s == 0:
                            continue
                        for cls, idxs in by_class.items():
                            drawn = rng_r.choice(idxs, size=n_s, replace=True)
                            boot_step_class[step][cls].append(
                                float(sub_df_reset.loc[drawn, "decoder_proba"].values.mean())
                            )

                # --- Scatter: one balanced draw (seed=0), split by class ---
                draws0 = select_cell_trials_bootstrap(per_step, rng=np.random.default_rng(0))
                for bi, bval in enumerate(bhv_vals):
                    bval_int = int(bval)
                    color = colors[bi]
                    blabel = _bhv_label(pp, bval)
                    if bval_int not in draws0:
                        continue
                    idx0 = draws0[bval_int]
                    y_all = sub_df_reset.loc[idx0, "decoder_proba"].values
                    step_all = sub_df_reset.loc[idx0, "resampled"].values
                    x_dodge = (bi - (len(bhv_vals) - 1) / 2) * 0.15
                    first = True
                    for step_i, step in enumerate(steps):
                        mask_s = step_all == step
                        y_s = y_all[mask_s]
                        if len(y_s) == 0:
                            continue
                        jitter = np.random.default_rng(42 + step_i * 10 + bi).uniform(
                            -0.06, 0.06, len(y_s)
                        )
                        ax.scatter(
                            np.full(len(y_s), step_i) + x_dodge + jitter,
                            y_s,
                            color=color, alpha=0.30, s=10, zorder=1,
                            label=blabel if first else None,
                        )
                        first = False

                # --- Mean ± bootstrap SE overlay per class ---
                for bi, bval in enumerate(bhv_vals):
                    bval_int = int(bval)
                    color = colors[bi]
                    x_line, means_b, ses_b = [], [], []
                    for step_i, step in enumerate(steps):
                        reps = boot_step_class[step].get(bval_int, [])
                        if not reps:
                            continue
                        arr = np.array(reps)
                        x_line.append(step_i)
                        means_b.append(float(arr.mean()))
                        ses_b.append(float(arr.std()))
                    if x_line:
                        x_dodge = (bi - (len(bhv_vals) - 1) / 2) * 0.15
                        xl = [x + x_dodge for x in x_line]
                        ax.plot(xl, means_b, color=color, lw=1.5, zorder=3)
                        ax.errorbar(
                            xl, means_b,
                            yerr=np.array(ses_b),
                            fmt="none", color=color, capsize=3, zorder=3,
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
# One column per site (sorted by AUC descending), one dot per qualifying ambiguous
# resampled step, colored by step using a diverging colormap.
# Only `split == "ambiguous"` trials are shown.

# %%
if all_rows:
    ambig_df = trial_df[trial_df["split"] == "ambiguous"]

    # Site order: sort by logged_auc descending.
    site_auc = (
        ambig_df.groupby("site_label")["logged_auc"].first()
        .sort_values(ascending=False)
    )
    site_order = list(site_auc.index)
    n_sites = len(site_order)

    # Step values derived from data (typically 2-5 but subject-specific).
    step_values = sorted(ambig_df["resampled"].dropna().unique())
    step_colors = plt.cm.RdBu_r(np.linspace(0.15, 0.85, len(step_values)))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_sites * 0.35), 7), sharex=True)

    # --- Panel A: mean across all trials (pooled behavior) ---
    ax = axes[0]
    for step_i, step in enumerate(step_values):
        x_base = np.arange(n_sites) + step_i * 0.12
        site_means = []
        for sl in site_order:
            vals = ambig_df.loc[
                (ambig_df["site_label"] == sl) & (ambig_df["resampled"] == step),
                "decoder_proba_aligned"
            ].values
            site_means.append(float(vals.mean()) if len(vals) > 0 else np.nan)
        ax.scatter(
            x_base, site_means,
            color=step_colors[step_i], s=20, alpha=0.85, label=f"step {int(step)}",
        )
    ax.set_ylabel("decoder_proba_aligned")
    ax.set_title("Ambiguous trials — pooled behavior")
    ax.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="resampled step", loc="upper right", fontsize=7, ncol=3)

    # --- Panel B: split by behavior (-1 vs +1) ---
    ax = axes[1]
    bhv_styles = {0: dict(marker="v", alpha=0.5), 1: dict(marker="^", alpha=0.5)}
    for step_i, step in enumerate(step_values):
        for bi, bval in enumerate([0, 1]):
            x_base = np.arange(n_sites) + step_i * 0.12 + bi * 0.04
            site_means = []
            for sl in site_order:
                vals = ambig_df.loc[
                    (ambig_df["site_label"] == sl)
                    & (ambig_df["resampled"] == step)
                    & (ambig_df["behavior_categorical_forced"] == bval),
                    "decoder_proba_aligned",
                ].values
                site_means.append(float(vals.mean()) if len(vals) > 0 else np.nan)
            ax.scatter(
                x_base, site_means,
                color=step_colors[step_i], s=15,
                **bhv_styles[bval],
            )
    ax.set_ylabel("decoder_proba_aligned")
    ax.set_title("Ambiguous trials — split by behavior (▲=+1, ▼=−1)")
    ax.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(n_sites))
    ax.set_xticklabels(site_order, rotation=90, fontsize=5)

    plt.tight_layout()
    fig.savefig(outdir_path / "aggregate_figure.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote aggregate_figure.pdf ({n_sites} sites)")
