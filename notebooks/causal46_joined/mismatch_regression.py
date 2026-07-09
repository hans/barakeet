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
# # Single-trial acoustic×percept mismatch regression
#
# For each qualifying cell `(subject, electrode_idx, phoneme_pair, word_end)`,
# fits the single-trial OLS models:
#
#     additive:  HGA ~ step_c + percept_c
#     full:      HGA ~ step_c + percept_c + step_c:percept_c
#
# where `step_c` = resampled centered on mean of included steps and
# `percept_c` = behavior_dummy_forced − 0.5 (class 1 = step-6-endpoint percept).
#
# **β_interaction is the diagnostic**: significant negative interaction → nonlinear
# conflict/surprisal signal beyond additive opponent coding.
#
# **Window selection**: first significant behavioral discriminative window (b_windows,
# window_id=0, ci_excludes_zero=True). Acoustic bootstrap (b4_acoustic_per_window) is
# then looked up at the same time range as a diagnostic — any 5-sample acoustic window
# fully within the behavioral window that has ci_aligned_excludes_zero is flagged.
# Robustness variant uses a fixed a priori window [POD, word_offset].
#
# See: docs/superpowers/plans/2026-07-08-causal46-mismatch-regression.md

# %% tags=["parameters"]
b4_acoustic_per_cell_path = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_per_cell.parquet"
b4_acoustic_per_window_path = "outputs/causal46_joined/acoustic_on_ambiguous/b4_acoustic_per_window.parquet"
b_windows_path = "outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet"
trial_balance_path = "outputs/causal46_joined/trial_balance_index.csv"
epoch_dir = "outputs/epochs_preprocessed"
outdir = "outputs/causal46_joined/mismatch_regression"
K_min_per_step = 4
min_steps = 2
ci_low = 2.5
ci_high = 97.5

# %%
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

import mne

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))

from src.data import add_metadata_features
from src.stimuli import OFFSET_DICT, POD_dict
from src.viz_paper import epoch_sfreq, epoch_tmin
from _within_completion import extract_hga_trials, resolve_behavior_col  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = Path(outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_DIR = Path(epoch_dir)

# t=0 sample index
SAMPLE_T0 = int(round((0.0 - epoch_tmin) * epoch_sfreq))
print(f"epoch_tmin={epoch_tmin}, epoch_sfreq={epoch_sfreq}, SAMPLE_T0={SAMPLE_T0}")

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

# %% [markdown]
# ## §2: Regression helpers

# %%
def _fit_mismatch_regression(trial_df: pd.DataFrame, smin: int, smax: int, hga: np.ndarray):
    """Fit additive and full OLS models for one cell + window.

    Parameters
    ----------
    trial_df : pd.DataFrame with columns step_c, percept_c. The integer index
               must be the original row positions in md_cell (NOT reset-to-0),
               so that hga[trial_df.index.values, :] selects the right trials.
    smin, smax : half-open sample window [smin, smax)
    hga : (n_cell_trials, n_times) array from extract_hga_trials

    Returns dict with regression results or None if not enough data.
    """
    y = hga[trial_df.index.values, smin:smax].mean(axis=1)
    df = trial_df[["step_c", "percept_c"]].copy().reset_index(drop=True)
    df["hga"] = y

    if df["percept_c"].nunique() < 2 or len(df) < 6:
        return None

    mod_add = smf.ols("hga ~ step_c + percept_c", data=df).fit()
    mod_full = smf.ols("hga ~ step_c + percept_c + step_c:percept_c", data=df).fit()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anova = anova_lm(mod_add, mod_full)

    F_int = float(anova["F"].iloc[1]) if not np.isnan(anova["F"].iloc[1]) else float("nan")
    p_int_ftest = float(anova["Pr(>F)"].iloc[1]) if not np.isnan(anova["Pr(>F)"].iloc[1]) else float("nan")

    def _coef(mod, name, kind):
        try:
            if kind == "b":
                return float(mod.params[name])
            elif kind == "se":
                return float(mod.bse[name])
            elif kind == "p":
                return float(mod.pvalues[name])
        except KeyError:
            return float("nan")

    return dict(
        n=len(df),
        beta_step=_coef(mod_full, "step_c", "b"),
        se_step=_coef(mod_full, "step_c", "se"),
        p_step=_coef(mod_full, "step_c", "p"),
        beta_percept=_coef(mod_full, "percept_c", "b"),
        se_percept=_coef(mod_full, "percept_c", "se"),
        p_percept=_coef(mod_full, "percept_c", "p"),
        beta_int=_coef(mod_full, "step_c:percept_c", "b"),
        se_int=_coef(mod_full, "step_c:percept_c", "se"),
        p_int=_coef(mod_full, "step_c:percept_c", "p"),
        F_int=F_int,
        p_int_ftest=p_int_ftest,
        r2_add=float(mod_add.rsquared),
        r2_full=float(mod_full.rsquared),
        delta_r2=float(mod_full.rsquared - mod_add.rsquared),
    )


def _build_trial_df(md_cell: pd.DataFrame, steps: list, behav_col: str) -> pd.DataFrame:
    """Build design matrix for the qualifying steps.

    Filters to `steps` with both percepts present, centers step and percept.
    Returns DataFrame with the ORIGINAL integer positions from md_cell as index
    (not reset), so callers can use df.index.values to index into hga.
    Columns: resampled, percept_raw, step_c, percept_c.
    """
    mask = md_cell[behav_col].isin([0, 1]) & md_cell["resampled"].isin(steps)
    # Keep original index positions so we can slice hga correctly
    df = md_cell.loc[mask, ["resampled", behav_col]].copy()
    df = df.rename(columns={behav_col: "percept_raw"})
    # Keep only steps that have both percepts
    valid_steps = [
        s for s in steps
        if set(df.loc[df["resampled"] == s, "percept_raw"].unique()) >= {0, 1}
    ]
    if len(valid_steps) < min_steps:
        return pd.DataFrame()
    # Check per-step count threshold
    passing_steps = [
        s for s in valid_steps
        if (df.loc[df["resampled"] == s, "percept_raw"] == 0).sum() >= K_min_per_step
        and (df.loc[df["resampled"] == s, "percept_raw"] == 1).sum() >= K_min_per_step
    ]
    if len(passing_steps) < min_steps:
        return pd.DataFrame()
    df = df[df["resampled"].isin(passing_steps)].copy()
    step_mean = df["resampled"].mean()
    df["step_c"] = df["resampled"] - step_mean
    df["percept_c"] = df["percept_raw"] - 0.5
    return df


def _s_to_t(s: int) -> float:
    return float(s) / epoch_sfreq + epoch_tmin


# %% [markdown]
# ## §4: Load data sources

# %%
b4_acoustic_per_cell = pl.read_parquet(b4_acoustic_per_cell_path) if Path(b4_acoustic_per_cell_path).exists() else pl.DataFrame()
b4_acoustic_per_window = pl.read_parquet(b4_acoustic_per_window_path) if Path(b4_acoustic_per_window_path).exists() else pl.DataFrame()
b_windows = pl.read_parquet(b_windows_path) if Path(b_windows_path).exists() else pl.DataFrame()
trial_balance = pl.read_csv(trial_balance_path) if Path(trial_balance_path).exists() else pl.DataFrame()

print(f"b4_acoustic_per_cell: {b4_acoustic_per_cell.height} rows")
print(f"b4_acoustic_per_window: {b4_acoustic_per_window.height} rows")
print(f"b_windows: {b_windows.height} rows")
print(f"trial_balance: {trial_balance.height} rows")

# %% [markdown]
# ## §3: Cell & window selection

# %%
# Behavioral discriminative windows: first significant window (window_id=0,
# ci_excludes_zero=True, not a fallback) from b_windows. These supply smin/smax
# for the regression window and define which cells are behaviorally responsive.
if b_windows.height > 0 and "window_id" in b_windows.columns and "ci_excludes_zero" in b_windows.columns:
    beh_windows = (
        b_windows
        .filter(
            (pl.col("window_id") == 0)
            & pl.col("ci_excludes_zero")
            & ~pl.col("is_fallback")
        )
        .select(CELL_KEYS + ["smin", "smax"])
    )
else:
    beh_windows = pl.DataFrame(schema={k: pl.Utf8 for k in CELL_KEYS} | {"smin": pl.Int64, "smax": pl.Int64})

print(f"Behavioral discriminative windows (first, significant): {beh_windows.height}")

# Acoustic-significant cells: still required for qualifying cells so that
# we only analyse cells where BOTH a behavioral and acoustic effect exist.
if b4_acoustic_per_cell.height > 0 and "best_ci_aligned_excludes_zero" in b4_acoustic_per_cell.columns:
    ac_sig_cells = (
        b4_acoustic_per_cell
        .filter(pl.col("best_ci_aligned_excludes_zero"))
        .select(CELL_KEYS)
    )
else:
    ac_sig_cells = pl.DataFrame(schema={k: pl.Utf8 for k in CELL_KEYS})

print(f"Acoustic-significant cells: {ac_sig_cells.height}")

# Qualifying steps from trial_balance_index (is_ambiguous_step, per-step K threshold)
if trial_balance.height > 0:
    qualifying_steps_df = (
        trial_balance
        .filter(
            pl.col("is_ambiguous_step")
            & (pl.col("min_class") >= K_min_per_step)
        )
        .group_by(CELL_KEYS)
        .agg(pl.col("resampled").sort().alias("qualifying_steps"))
        .filter(pl.col("qualifying_steps").list.len() >= min_steps)
    )
else:
    qualifying_steps_df = pl.DataFrame()

print(f"Cells with ≥{min_steps} qualifying steps (K≥{K_min_per_step}): {qualifying_steps_df.height}")

# Intersection: behavioral-window ∩ acoustic-sig ∩ trial-structure.
# qualifying_cells carries smin/smax from beh_windows as the regression window.
if beh_windows.height > 0 and qualifying_steps_df.height > 0:
    qualifying_cells = beh_windows.join(qualifying_steps_df, on=CELL_KEYS, how="inner")
    if ac_sig_cells.height > 0:
        qualifying_cells = qualifying_cells.join(ac_sig_cells, on=CELL_KEYS, how="inner")
    else:
        print("WARNING: no acoustic significance data; using behavioral ∩ trial-structure only")
else:
    qualifying_cells = pl.DataFrame()

print(f"Qualifying cells for regression: {qualifying_cells.height}")

# Pre-build acoustic per-window lookup as pandas for fast per-cell queries in the loop.
# b4_acoustic_per_window has 5-sample windows; we look up any that fall within the
# behavioral window range and report whether any is ci_aligned_excludes_zero.
if b4_acoustic_per_window.height > 0:
    _baw_pd = b4_acoustic_per_window.to_pandas()
    _baw_pd = _baw_pd.set_index(CELL_KEYS)
else:
    _baw_pd = None

# %% [markdown]
# ## Per-cell regression loop

# %%
# Per-phoneme_pair: assert class 1 = step-6-endpoint percept
# (modal behavior_dummy_forced at step 6 should be 1)
# This is validated per subject from the epoch metadata during the loop.

results_rows = []
cell_table_rows = []

subjects = sorted(qualifying_cells["subject"].unique().to_list()) if qualifying_cells.height > 0 else []

for subject in subjects:
    ep_path = EPOCH_DIR / f"{subject}_epo.fif"
    if not ep_path.exists():
        print(f"  ⚠ {subject}: epoch file missing — skipping")
        continue

    ep_full = mne.read_epochs(str(ep_path), preload=True, verbose="WARNING")
    ep_full.metadata = add_metadata_features(ep_full.metadata.copy())

    behav_col = resolve_behavior_col(ep_full.metadata)

    # Per-subject: assert class 1 = step-6-endpoint percept per phoneme_pair
    for pp in ep_full.metadata["phoneme_pair"].dropna().unique():
        md_pp = ep_full.metadata[ep_full.metadata["phoneme_pair"] == pp]
        md_step6 = md_pp[md_pp["resampled"] == 6]
        if len(md_step6) == 0:
            continue
        modal_class = int(md_step6[behav_col].mode()[0])
        if modal_class != 1:
            print(f"  ⚠ {subject} {pp}: modal class at step 6 = {modal_class} (expected 1) — skipping pair")
            continue

    subj_cells = qualifying_cells.filter(pl.col("subject") == subject)

    for row in subj_cells.iter_rows(named=True):
        elec_idx = int(row["electrode_idx"])
        pp = row["phoneme_pair"]
        we = row["word_end"]
        steps = [int(s) for s in row["qualifying_steps"]]
        smin_beh = int(row["smin"])
        smax_beh = int(row["smax"])

        # Validate class 1 = step-6 percept for this pair
        md_full_pp = ep_full.metadata[ep_full.metadata["phoneme_pair"] == pp]
        md_step6 = md_full_pp[md_full_pp["resampled"] == 6]
        if len(md_step6) == 0 or int(md_step6[behav_col].mode()[0]) != 1:
            continue

        hga, md_cell = extract_hga_trials(ep_full, elec_idx, pp, we)

        if len(md_cell) == 0:
            continue

        # Build design matrix for behavioral window
        trial_df_beh = _build_trial_df(md_cell, steps, behav_col)
        if trial_df_beh.empty:
            continue

        # Regression on behavioral window
        res_beh = _fit_mismatch_regression(trial_df_beh, smin_beh, smax_beh, hga)
        if res_beh is None:
            continue

        tmin_beh = _s_to_t(smin_beh)
        tmax_beh = _s_to_t(smax_beh)
        mirrored = (
            np.sign(res_beh["beta_step"]) != np.sign(res_beh["beta_percept"])
            if res_beh["beta_step"] != 0 and res_beh["beta_percept"] != 0
            else False
        )
        both_main_sig = res_beh["p_step"] < 0.05 and res_beh["p_percept"] < 0.05

        # Acoustic lookup: find 5-sample acoustic bootstrap windows fully within
        # [smin_beh, smax_beh) and report whether any is ci_aligned_excludes_zero.
        ac_sig_in_beh = False
        n_ac_sig_windows = 0
        if _baw_pd is not None:
            try:
                _cell_ac = _baw_pd.loc[(subject, elec_idx, pp, we)]
                if hasattr(_cell_ac, "iterrows"):  # multiple rows
                    _contained = _cell_ac[
                        (_cell_ac["smin"] >= smin_beh) & (_cell_ac["smax"] <= smax_beh)
                    ]
                else:  # single row became a Series
                    _contained = _cell_ac.to_frame().T[
                        (_cell_ac["smin"] >= smin_beh) & (_cell_ac["smax"] <= smax_beh)
                    ]
                n_ac_sig_windows = int(_contained["ci_aligned_excludes_zero"].sum())
                ac_sig_in_beh = n_ac_sig_windows > 0
            except KeyError:
                pass

        # Robustness: fixed a priori window [POD, word_offset]
        pod_t = POD_dict.get(pp, 0.295)
        offset_t = OFFSET_DICT.get(we, 1.0)
        smin_fixed = int(round((pod_t - epoch_tmin) * epoch_sfreq))
        smax_fixed = int(round((offset_t - epoch_tmin) * epoch_sfreq))
        smax_fixed = min(smax_fixed, hga.shape[1])
        smin_fixed = max(smin_fixed, 0)

        res_fixed = None
        if smax_fixed > smin_fixed:
            trial_df_fixed = _build_trial_df(md_cell, steps, behav_col)
            if not trial_df_fixed.empty:
                res_fixed = _fit_mismatch_regression(trial_df_fixed, smin_fixed, smax_fixed, hga)

        row_base = dict(
            subject=subject,
            electrode_idx=elec_idx,
            phoneme_pair=pp,
            word_end=we,
            n_steps=int(trial_df_beh["resampled"].nunique()),
            min_per_step_per_class=K_min_per_step,
            smin=smin_beh,
            smax=smax_beh,
            tmin=round(tmin_beh, 4),
            tmax=round(tmax_beh, 4),
            mirrored_signs=bool(mirrored),
            both_main_sig=bool(both_main_sig),
            ac_sig_in_beh_window=bool(ac_sig_in_beh),
            n_ac_sig_windows=n_ac_sig_windows,
            window_source="behavioral",
            **{k: v for k, v in res_beh.items()},
        )
        results_rows.append(row_base)

        # Robustness variant row
        if res_fixed is not None:
            row_fixed = dict(
                subject=subject,
                electrode_idx=elec_idx,
                phoneme_pair=pp,
                word_end=we,
                n_steps=int(trial_df_fixed["resampled"].nunique()),
                min_per_step_per_class=K_min_per_step,
                smin=smin_fixed,
                smax=smax_fixed,
                tmin=round(_s_to_t(smin_fixed), 4),
                tmax=round(_s_to_t(smax_fixed), 4),
                mirrored_signs=bool(
                    np.sign(res_fixed["beta_step"]) != np.sign(res_fixed["beta_percept"])
                    if res_fixed["beta_step"] != 0 and res_fixed["beta_percept"] != 0
                    else False
                ),
                both_main_sig=res_fixed["p_step"] < 0.05 and res_fixed["p_percept"] < 0.05,
                ac_sig_in_beh_window=False,
                n_ac_sig_windows=0,
                window_source="fixed",
                **{k: v for k, v in res_fixed.items()},
            )
            results_rows.append(row_fixed)

        # Per-(step × percept) cell table for plotting
        # Use all trials in the cell (not just qualifying steps) for the table,
        # but only report qualifying steps so counts are consistent with regression.
        _qualifying_steps_set = set(int(s) for s in trial_df_beh["resampled"].unique())
        y_all = hga[:, smin_beh:smax_beh].mean(axis=1)
        for s in steps:
            if s not in _qualifying_steps_set:
                continue
            step_mask = md_cell["resampled"] == s
            for cls in [0, 1]:
                cls_mask = step_mask & (md_cell[behav_col] == cls)
                n_cls = int(cls_mask.sum())
                mean_hga = float(y_all[cls_mask.values].mean()) if n_cls > 0 else float("nan")
                cell_table_rows.append(dict(
                    subject=subject,
                    electrode_idx=elec_idx,
                    phoneme_pair=pp,
                    word_end=we,
                    resampled=s,
                    percept_class=cls,
                    n=n_cls,
                    mean_hga=mean_hga,
                    smin=smin_beh,
                    smax=smax_beh,
                ))

print(f"\nRegression complete: {len([r for r in results_rows if r['window_source']=='behavioral'])} behavioral-window cells")
print(f"  + {len([r for r in results_rows if r['window_source']=='fixed'])} fixed-window robustness rows")

# %% [markdown]
# ## §9: Validation — check behavioral windows for EC243 example cells
#
# Windows now come from b_windows (behavioral discriminative search):
# - e102 bm mountains: window_id=0 → [80,90) (same as acoustic proof-of-concept)
# - e101 dn necessary: window_id=0 → [100,110) (different from acoustic spec target)

# %% [markdown]
# ### e102 bm mountains: steps [3,4,5], behavioral window [80,90)

# %%
_val_ep_path = EPOCH_DIR / "EC243_epo.fif"
if _val_ep_path.exists():
    _ep_val = mne.read_epochs(str(_val_ep_path), preload=True, verbose="WARNING")
    _ep_val.metadata = add_metadata_features(_ep_val.metadata.copy())
    _behav_col = resolve_behavior_col(_ep_val.metadata)

    print("=== Validation: e102 bm mountains ===")
    _hga102, _md102 = extract_hga_trials(_ep_val, 102, "bm", "mountains")
    print(f"  Trials in cell: {len(_md102)}")

    _steps_102 = [3, 4, 5]
    _smin_102, _smax_102 = 80, 90
    print(f"  Window: [{_smin_102}, {_smax_102}) → t[{_s_to_t(_smin_102):.2f}, {_s_to_t(_smax_102):.2f})")

    _y102 = _hga102[:, _smin_102:_smax_102].mean(axis=1)
    print("  Per-(step, percept) counts and mean HGA:")
    _n_total = 0
    for _s in _steps_102:
        for _cls in [0, 1]:
            _mask = (_md102["resampled"] == _s) & (_md102[_behav_col] == _cls)
            _n = int(_mask.sum())
            _mn = float(_y102[_mask.values].mean()) if _n > 0 else float("nan")
            print(f"    step={_s}, class={_cls}: n={_n}, mean={_mn:.3f}")
            _n_total += _n
    print(f"  Total n: {_n_total}")

    _df102 = _build_trial_df(_md102, _steps_102, _behav_col)
    if not _df102.empty:
        _res102 = _fit_mismatch_regression(_df102, _smin_102, _smax_102, _hga102)
        if _res102:
            print(f"  β_step={_res102['beta_step']:.3f} (p={_res102['p_step']:.3f})")
            print(f"  β_percept={_res102['beta_percept']:.3f} (p={_res102['p_percept']:.3f})")
            print(f"  β_int={_res102['beta_int']:.3f} (p={_res102['p_int']:.3f})")
            print(f"  F_int={_res102['F_int']:.2f}, p_ftest={_res102['p_int_ftest']:.3f}")
            print(f"  R²: {_res102['r2_add']:.3f} → {_res102['r2_full']:.3f} (Δ={_res102['delta_r2']:.3f})")

    print("\n=== Validation: e101 dn necessary ===")
    _hga101, _md101 = extract_hga_trials(_ep_val, 101, "dn", "necessary")
    print(f"  Trials in cell: {len(_md101)}")

    _steps_101 = [2, 3, 4]
    _smin_101, _smax_101 = 100, 110  # behavioral window from b_windows window_id=0
    print(f"  Window: [{_smin_101}, {_smax_101}) → t[{_s_to_t(_smin_101):.2f}, {_s_to_t(_smax_101):.2f})")

    _df101 = _build_trial_df(_md101, _steps_101, _behav_col)
    if not _df101.empty:
        _res101 = _fit_mismatch_regression(_df101, _smin_101, _smax_101, _hga101)
        if _res101:
            print(f"  n={_res101['n']}")
            print(f"  β_step={_res101['beta_step']:.3f} (p={_res101['p_step']:.3f})")
            print(f"  β_percept={_res101['beta_percept']:.3f} (p={_res101['p_percept']:.3f})")
            print(f"  β_int={_res101['beta_int']:.3f} (p={_res101['p_int']:.3f})")
            print(f"  F_int={_res101['F_int']:.2f}, p_ftest={_res101['p_int_ftest']:.3f}")
            print(f"  R²: {_res101['r2_add']:.3f} → {_res101['r2_full']:.3f} (Δ={_res101['delta_r2']:.3f})")
else:
    print("EC243 epoch file not found — skipping validation")

# %% [markdown]
# ## §6: Population statistics

# %%
if results_rows:
    # Separate acoustic-window results from robustness variant
    _results_ac = [r for r in results_rows if r["window_source"] == "behavioral"]
    _results_fixed = [r for r in results_rows if r["window_source"] == "fixed"]

    res_df = pd.DataFrame(_results_ac)
    print(f"Cells in population: {len(res_df)}")

    # BH-FDR on interaction p-values
    if len(res_df) > 0:
        _reject, _q, _, _ = multipletests(res_df["p_int"].fillna(1.0), method="fdr_bh")
        res_df["q_int_fdr"] = _q
        res_df["int_sig_fdr"] = _reject

        # Single-trial mirroring rate
        mirror_rate = float(res_df["mirrored_signs"].mean())
        print(f"\nSingle-trial mirroring rate (sign(β_step) ≠ sign(β_percept)): {mirror_rate:.1%} ({res_df['mirrored_signs'].sum()}/{len(res_df)})")

        both_sig_rate = float(res_df["both_main_sig"].mean())
        print(f"Both main effects significant (p<0.05): {both_sig_rate:.1%} ({res_df['both_main_sig'].sum()}/{len(res_df)})")

        # Interaction distribution
        b_int = res_df["beta_int"].dropna()
        print(f"\nβ_interaction: median={b_int.median():.3f}, mean={b_int.mean():.3f}")
        n_neg = int((b_int < 0).sum())
        print(f"  Negative: {n_neg}/{len(b_int)} ({n_neg/len(b_int):.1%})")

        # Sign test (two-tailed binomial)
        from scipy.stats import wilcoxon, binomtest
        if len(b_int) >= 3:
            bt = binomtest(n_neg, len(b_int), 0.5, alternative="greater")
            print(f"  Sign test (H1: negative dominant): p={bt.pvalue:.4f}")
            if len(b_int) >= 10:
                try:
                    wstat, wp = wilcoxon(b_int, alternative="less")
                    print(f"  Wilcoxon signed-rank (H1: negative): W={wstat:.0f}, p={wp:.4f}")
                except Exception:
                    pass

        n_sig_nominal = int((res_df["p_int"] < 0.05).sum())
        n_sig_fdr = int(res_df["int_sig_fdr"].sum())
        print(f"\nInteraction significance:")
        print(f"  Nominal α=0.05: {n_sig_nominal}/{len(res_df)}")
        print(f"  BH-FDR q<0.05:  {n_sig_fdr}/{len(res_df)}")

        # Per-phoneme_pair breakdown
        print("\nPer-phoneme_pair:")
        for _pp, _grp in res_df.groupby("phoneme_pair"):
            _mr = float(_grp["mirrored_signs"].mean())
            _bi = _grp["beta_int"].dropna()
            _ns = int((_grp["p_int"] < 0.05).sum())
            print(f"  {_pp}: n={len(_grp)}, mirror={_mr:.1%}, median_β_int={_bi.median():.3f}, n_sig={_ns}")

        # Robustness: compare acoustic-window vs fixed-window interaction
        if _results_fixed:
            res_fixed_df = pd.DataFrame(_results_fixed)
            _merge = res_df[["subject", "electrode_idx", "phoneme_pair", "word_end", "beta_int"]].merge(
                res_fixed_df[["subject", "electrode_idx", "phoneme_pair", "word_end", "beta_int"]].rename(
                    columns={"beta_int": "beta_int_fixed"}),
                on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
                how="inner",
            )
            if len(_merge) > 2:
                _corr = _merge["beta_int"].corr(_merge["beta_int_fixed"])
                print(f"\nRobustness: corr(β_int acoustic vs fixed window) = {_corr:.3f} (n={len(_merge)})")
else:
    print("No qualifying cells — population stats skipped")
    res_df = pd.DataFrame()

# %% [markdown]
# ## §7: Outputs

# %%
# Schema for empty-but-typed result
_SCHEMA = {
    "subject": pd.Series(dtype="object"),
    "electrode_idx": pd.Series(dtype="int64"),
    "phoneme_pair": pd.Series(dtype="object"),
    "word_end": pd.Series(dtype="object"),
    "smin": pd.Series(dtype="int64"),
    "smax": pd.Series(dtype="int64"),
    "tmin": pd.Series(dtype="float64"),
    "tmax": pd.Series(dtype="float64"),
    "n": pd.Series(dtype="int64"),
    "n_steps": pd.Series(dtype="int64"),
    "min_per_step_per_class": pd.Series(dtype="int64"),
    "beta_step": pd.Series(dtype="float64"),
    "se_step": pd.Series(dtype="float64"),
    "p_step": pd.Series(dtype="float64"),
    "beta_percept": pd.Series(dtype="float64"),
    "se_percept": pd.Series(dtype="float64"),
    "p_percept": pd.Series(dtype="float64"),
    "beta_int": pd.Series(dtype="float64"),
    "se_int": pd.Series(dtype="float64"),
    "p_int": pd.Series(dtype="float64"),
    "F_int": pd.Series(dtype="float64"),
    "p_int_ftest": pd.Series(dtype="float64"),
    "r2_add": pd.Series(dtype="float64"),
    "r2_full": pd.Series(dtype="float64"),
    "delta_r2": pd.Series(dtype="float64"),
    "mirrored_signs": pd.Series(dtype="bool"),
    "both_main_sig": pd.Series(dtype="bool"),
    "ac_sig_in_beh_window": pd.Series(dtype="bool"),
    "n_ac_sig_windows": pd.Series(dtype="int64"),
    "window_source": pd.Series(dtype="object"),
}

# mismatch_per_cell.parquet — includes both acoustic and fixed-window rows
if results_rows:
    mismatch_per_cell = pd.DataFrame(results_rows)
    if "q_int_fdr" not in mismatch_per_cell.columns:
        # Compute FDR across all acoustic-window rows
        _ac_rows = mismatch_per_cell[mismatch_per_cell["window_source"] == "behavioral"]
        if len(_ac_rows) > 0:
            _reject_all, _q_all, _, _ = multipletests(_ac_rows["p_int"].fillna(1.0), method="fdr_bh")
            mismatch_per_cell.loc[mismatch_per_cell["window_source"] == "behavioral", "q_int_fdr"] = _q_all
            mismatch_per_cell.loc[mismatch_per_cell["window_source"] == "behavioral", "int_sig_fdr"] = _reject_all
else:
    mismatch_per_cell = pd.DataFrame(_SCHEMA)

mismatch_per_cell.to_parquet(OUT_DIR / "mismatch_per_cell.parquet", index=False)
print(f"Written: {OUT_DIR / 'mismatch_per_cell.parquet'} ({len(mismatch_per_cell)} rows)")

# mismatch_cell_table.parquet — step × percept mean HGA per cell
if cell_table_rows:
    mismatch_cell_table = pd.DataFrame(cell_table_rows)
else:
    mismatch_cell_table = pd.DataFrame({
        "subject": pd.Series(dtype="object"),
        "electrode_idx": pd.Series(dtype="int64"),
        "phoneme_pair": pd.Series(dtype="object"),
        "word_end": pd.Series(dtype="object"),
        "resampled": pd.Series(dtype="int64"),
        "percept_class": pd.Series(dtype="int64"),
        "n": pd.Series(dtype="int64"),
        "mean_hga": pd.Series(dtype="float64"),
        "smin": pd.Series(dtype="int64"),
        "smax": pd.Series(dtype="int64"),
    })

mismatch_cell_table.to_parquet(OUT_DIR / "mismatch_cell_table.parquet", index=False)
print(f"Written: {OUT_DIR / 'mismatch_cell_table.parquet'} ({len(mismatch_cell_table)} rows)")

# mismatch_summary.csv
if len(results_rows) > 0:
    _ac_df = mismatch_per_cell[mismatch_per_cell["window_source"] == "behavioral"]
    summary_dict = dict(
        n_qualifying_cells=len(_ac_df),
        mirroring_rate=float(_ac_df["mirrored_signs"].mean()),
        n_mirrored=int(_ac_df["mirrored_signs"].sum()),
        both_main_sig_rate=float(_ac_df["both_main_sig"].mean()),
        median_beta_int=float(_ac_df["beta_int"].median()),
        mean_beta_int=float(_ac_df["beta_int"].mean()),
        frac_beta_int_negative=float((_ac_df["beta_int"] < 0).mean()),
        n_int_sig_nominal=int((_ac_df["p_int"] < 0.05).sum()),
        n_int_sig_fdr=int(_ac_df.get("int_sig_fdr", pd.Series([False] * len(_ac_df))).sum()),
    )
    pd.DataFrame([summary_dict]).to_csv(OUT_DIR / "mismatch_summary.csv", index=False)
    print(f"Written: {OUT_DIR / 'mismatch_summary.csv'}")
    print(pd.DataFrame([summary_dict]).T.to_string())
else:
    pd.DataFrame([{"n_qualifying_cells": 0}]).to_csv(OUT_DIR / "mismatch_summary.csv", index=False)

# %% [markdown]
# ## QC figures

# %%
if len(results_rows) > 0:
    _ac_df = mismatch_per_cell[mismatch_per_cell["window_source"] == "behavioral"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # β_interaction histogram
    ax = axes[0]
    _bi = _ac_df["beta_int"].dropna()
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.hist(_bi, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(float(_bi.median()), color="firebrick", lw=1.5, ls="-", label=f"median={float(_bi.median()):.3f}")
    ax.set_xlabel("β_interaction")
    ax.set_ylabel("Count")
    ax.set_title("Population: interaction coefficient")
    ax.legend(fontsize=8)

    # β_step vs β_percept scatter (mirroring check)
    ax = axes[1]
    _mirror = _ac_df["mirrored_signs"]
    ax.scatter(_ac_df.loc[_mirror, "beta_step"], _ac_df.loc[_mirror, "beta_percept"],
               color="steelblue", s=20, alpha=0.7, label=f"mirrored (n={_mirror.sum()})")
    ax.scatter(_ac_df.loc[~_mirror, "beta_step"], _ac_df.loc[~_mirror, "beta_percept"],
               color="tomato", s=20, alpha=0.7, label=f"matched (n={(~_mirror).sum()})")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("β_step")
    ax.set_ylabel("β_percept")
    ax.set_title("Main effects: step vs percept")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "mismatch_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {OUT_DIR / 'mismatch_summary.pdf'}")

    # Example cells: step × percept heatmap for top-2 cells by |β_int|
    _example_cells = _ac_df.nlargest(2, "beta_int" if len(_ac_df) < 4 else "delta_r2")
    if len(_example_cells) > 0 and cell_table_rows:
        fig2, axes2 = plt.subplots(1, max(1, len(_example_cells)), figsize=(5 * max(1, len(_example_cells)), 4))
        if len(_example_cells) == 1:
            axes2 = [axes2]
        cell_tab = mismatch_cell_table.copy()
        for _ax, (_, _cr) in zip(axes2, _example_cells.iterrows()):
            _ct = cell_tab[
                (cell_tab["subject"] == _cr["subject"])
                & (cell_tab["electrode_idx"] == _cr["electrode_idx"])
                & (cell_tab["phoneme_pair"] == _cr["phoneme_pair"])
                & (cell_tab["word_end"] == _cr["word_end"])
            ]
            if _ct.empty:
                continue
            _pivot = _ct.pivot(index="percept_class", columns="resampled", values="mean_hga")
            _im = _ax.imshow(_pivot.values, aspect="auto", cmap="RdBu_r")
            plt.colorbar(_im, ax=_ax, shrink=0.7)
            _ax.set_xticks(range(len(_pivot.columns)))
            _ax.set_xticklabels(_pivot.columns)
            _ax.set_yticks([0, 1])
            _ax.set_yticklabels(["stop", "nasal"])
            _ax.set_xlabel("step")
            _ax.set_title(
                f"{_cr['subject']} e{_cr['electrode_idx']} {_cr['phoneme_pair']} {_cr['word_end']}\n"
                f"β_int={_cr['beta_int']:.3f} (p={_cr['p_int']:.3f})",
                fontsize=8,
            )
        fig2.tight_layout()
        fig2.savefig(OUT_DIR / "mismatch_examples.pdf", bbox_inches="tight")
        plt.close(fig2)
        print(f"Written: {OUT_DIR / 'mismatch_examples.pdf'}")
else:
    print("No results — skipping figures")
    # Ensure output file always exists
    fig_empty, ax_empty = plt.subplots(1, 1, figsize=(4, 3))
    ax_empty.text(0.5, 0.5, "No qualifying cells", ha="center", va="center", transform=ax_empty.transAxes)
    fig_empty.savefig(OUT_DIR / "mismatch_summary.pdf", bbox_inches="tight")
    plt.close(fig_empty)

# %% [markdown]
# ## §8: Controls & caveats
#
# **Selection bias**: acoustic-window selection inflates β_step only; percept
# and interaction are orthogonal to acoustic step selection under the balanced
# layout — percept/interaction interpretation is unbiased.
#
# **Signed vs unsigned**: a null β_interaction does *not* rule out a signed
# belief−evidence unit (additive opponent) or two additive co-located
# populations — it only fails to detect the *unsigned conflict* signature.
#
# **RT / effort**: if reaction time is available in metadata, it should be
# added as a covariate + interaction to test whether step×percept interaction
# survives. Not included here because RT is not systematically available in
# the epoch metadata.
#
# **Small n**: minority-percept cells at some steps may have n=4–7. Prefer
# the F-test + BH-FDR over pairwise tests; the fixed-window robustness variant
# guards against acoustic-window cherry-picking.
#
# **One window per cell**: no window-level multiple comparisons; fixed-window
# variant is the guard.

print("Notebook complete.")
