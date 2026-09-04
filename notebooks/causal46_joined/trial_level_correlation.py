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
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.formula import api as smf
from tqdm.auto import tqdm, trange

from src.data import add_metadata_features
from src.stimuli import WORD_PHASE_DF


# %%
speech_responsive_path = "outputs/causal6/find_speech_responsive"
lpp_path = "outputs/causal46_joined/late_perceptual_projection/results.csv"

# bootstrap results
b4_per_window_path = "outputs/causal46_joined/t_tests/b4_per_window.parquet"
a_per_window_by_word_end_path = "outputs/causal46_joined/acoustic_bootstrap/a_per_window_by_word_end_all.parquet"

early_acoustic_decoding_summary_path = "outputs/causal46_joined/acoustic_early/acoustic_early_summary.csv"
early_acoustic_decoding_results_path = "outputs/causal46_joined/acoustic_early/acoustic_early_results.csv"

epoch_sfreq = 100
epoch_tmin = -0.4

# %% [markdown]
# ## Load

# %%
epochs_dict = {}
for p in Path("outputs/epochs_preprocessed").glob("*.fif"):
    ep = mne.read_epochs(p, verbose=False)
    ep.metadata = add_metadata_features(ep.metadata)
    epochs_dict[p.stem.rstrip("_epo")] = ep

# %%
speech_responsive_df = pd.concat([
    pd.read_csv(p) for p in Path(speech_responsive_path).glob("*.csv")
])

# %%
lpp = pd.read_csv(lpp_path)

# %%
a_per_window_by_word_end = pl.read_parquet(a_per_window_by_word_end_path)
b4_per_window = pl.read_parquet(b4_per_window_path)

# %%
early_sig_df = pd.read_csv(early_acoustic_decoding_summary_path)
# bring in smin, smax from full results
early_sig_df = pd.merge(
    early_sig_df,
    (
        pd.read_csv(early_acoustic_decoding_results_path)
        [["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]]
    ).drop_duplicates(),
    how="left",
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    validate="m:1",
)

# %% [markdown]
# ## Prepare early acoustic windows

# %%
early_all_acoustic_windows = (
    pd.merge(
        early_sig_df,#.query("significant_target"),
        a_per_window_by_word_end.to_pandas(),
        how="left", on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        suffixes=("_decoder", "_bootstrap"),
    )
    .query("smin_bootstrap >= smin_decoder and smax_bootstrap <= smax_decoder")
)

early_all_acoustic_windows["mean_diff_raw_med_abs"] = early_all_acoustic_windows["mean_diff_raw_med"].abs()

GROUP_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end", "significant_target"]
MAX_GAP = 2  # samples of separation tolerated within a union

early_all_acoustic_windows = (
    early_all_acoustic_windows
    .sort_values(GROUP_KEYS + ["smin_bootstrap"])
    .reset_index(drop=True)
)

early_all_acoustic_windows["contrast_sign"] = np.sign(early_all_acoustic_windows["mean_diff_raw_med"])

# running max of window end within each group
_run_max = (
    early_all_acoustic_windows
    .groupby(GROUP_KEYS, sort=False)["smax_bootstrap"]
    .cummax()
)

_new_group = (
    early_all_acoustic_windows[GROUP_KEYS] != early_all_acoustic_windows[GROUP_KEYS].shift(1)
).any(axis=1)
_gap = early_all_acoustic_windows["smin_bootstrap"] > _run_max.shift(1) + MAX_GAP
_sign_flip = (
    early_all_acoustic_windows["contrast_sign"]
    != early_all_acoustic_windows["contrast_sign"].shift(1)
)

_starts_new = _new_group | _gap | _sign_flip
_starts_new.iloc[0] = True

early_all_acoustic_windows["union_id"] = _starts_new.cumsum()

# collapse each union to a single interval, scored by mean contrast
early_all_acoustic_windows = (
    early_all_acoustic_windows
    .groupby(GROUP_KEYS + ["union_id"])
    .agg(
        smin_bootstrap=("smin_bootstrap", "min"),
        smax_bootstrap=("smax_bootstrap", "max"),
        smin_decoder=("smin_decoder", "first"),
        smax_decoder=("smax_decoder", "first"),
        mean_diff_raw_med_abs=("mean_diff_raw_med_abs", "mean"),
        mean_diff_raw_med=("mean_diff_raw_med", "mean"),
        n_windows=("smin_bootstrap", "size"),
        n_positive=("mean_diff_raw_med", lambda s: (s > 0).sum()),
    )
    .reset_index()
)

# retain union with largest mean contrast
early_all_acoustic_windows = (
    early_all_acoustic_windows
    .sort_values("mean_diff_raw_med_abs")
    .groupby(GROUP_KEYS).last().reset_index()
)

# compare to pod
early_all_acoustic_windows = pd.merge(
    early_all_acoustic_windows,
    WORD_PHASE_DF.set_index("phase").loc["pod"][["word", "start"]].rename(columns={"start": "t_pod"}),
    left_on="word_end", right_on="word",
    how="left", validate="m:1"
)
early_all_acoustic_windows["tcenter_bootstrap"] = (early_all_acoustic_windows["smin_bootstrap"] + early_all_acoustic_windows["smax_bootstrap"]) / 2 / epoch_sfreq + epoch_tmin
early_all_acoustic_windows["tcenter_bootstrap_from_pod"] = early_all_acoustic_windows["tcenter_bootstrap"] - early_all_acoustic_windows["t_pod"]
early_all_acoustic_windows["twidth"] = (early_all_acoustic_windows["smax_bootstrap"] - early_all_acoustic_windows["smin_bootstrap"]) / epoch_sfreq
early_all_acoustic_windows["tmin_bootstrap"] = early_all_acoustic_windows["smin_bootstrap"] / epoch_sfreq + epoch_tmin

early_sig_acoustic_windows = early_all_acoustic_windows.query("significant_target").copy()


# %% [markdown]
# ## Prepare late perceptual windows

# %%
def _sum_window_effects(cell_row):
    """
    Sum the ambiguous (perceptual) and endpoint (acoustic)
    contrasts over [smin, smax] for one cell
    """
    ambig = b4_per_window.filter(
        pl.col("subject") == cell_row.subject,
        pl.col("electrode_idx") == cell_row.electrode_idx,
        pl.col("phoneme_pair") == cell_row.phoneme_pair,
        pl.col("word_end") == cell_row.word_end,
        pl.col("smin") >= cell_row.smin,
        pl.col("smax") <= cell_row.smax,
    )
    ambig_other = b4_per_window.filter(
        pl.col("subject") == cell_row.subject,
        pl.col("electrode_idx") == cell_row.electrode_idx,
        pl.col("phoneme_pair") == cell_row.phoneme_pair,
        pl.col("word_end") != cell_row.word_end,
        pl.col("smin") >= cell_row.smin,
        pl.col("smax") <= cell_row.smax,
    )
    unambig_matched = a_per_window_by_word_end.filter(
        pl.col("subject") == cell_row.subject,
        pl.col("electrode_idx") == cell_row.electrode_idx,
        pl.col("phoneme_pair") == cell_row.phoneme_pair,
        pl.col("word_end") == cell_row.word_end,
        pl.col("smin") >= cell_row.smin,
        pl.col("smax") <= cell_row.smax,
    )
    unambig_mismatched = a_per_window_by_word_end.filter(
        pl.col("subject") == cell_row.subject,
        pl.col("electrode_idx") == cell_row.electrode_idx,
        pl.col("phoneme_pair") == cell_row.phoneme_pair,
        pl.col("word_end") != cell_row.word_end,
        pl.col("smin") >= cell_row.smin,
        pl.col("smax") <= cell_row.smax,
    )

    scalar_trials = pl.concat([
        ambig.select(pl.col("mean_diff_raw_med")),
        ambig_other.select(pl.col("mean_diff_raw_med")),
        unambig_matched.select(pl.col("mean_diff_raw_med")),
        unambig_mismatched.select(pl.col("mean_diff_raw_med")),
    ])

    return {
        "sum_ambig_effect": ambig.select(pl.col("mean_diff_raw_med")).mean().item(),
        "sum_ambig_other_effect": ambig_other.select(pl.col("mean_diff_raw_med")).mean().item(),
        "sum_unambig_matched_effect": unambig_matched.select(pl.col("mean_diff_raw_med")).mean().item(),
        "sum_unambig_mismatched_effect": unambig_mismatched.select(pl.col("mean_diff_raw_med")).mean().item(),
        "scale": scalar_trials.select(pl.col("mean_diff_raw_med").abs().max()).item(),
    }

lpp_specific_results = []
for _, site_row in lpp.query("projection_significant_uncorrected").iterrows():
    lpp_specific_results.append({
        **site_row.to_dict(),
        **_sum_window_effects(site_row)
    })

lpp_specific_results_df = pd.DataFrame(lpp_specific_results)

# %% [markdown]
# ### Estimate perceptual tuning

# %%
lpp_tuning_df = lpp_specific_results_df
lpp_tuning_df["perceptual_tuning"] = (np.sign(lpp_tuning_df.sum_ambig_effect) > 0).astype(int)

# merge in early acoustic information
lpp_tuning_df = pd.merge(
    lpp_tuning_df,
    (
        early_all_acoustic_windows[["subject", "electrode_idx", "phoneme_pair", "word_end", "mean_diff_raw_med"]]
        .rename(columns={"mean_diff_raw_med": "early_acoustic_mean_diff_raw_med"})
        .assign(early_acoustic_tuning=lambda df: (np.sign(df.early_acoustic_mean_diff_raw_med) > 0).astype(int))
    ),
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
)

lpp_tuning_df["lexical_evidence"] = (lpp_tuning_df.word_end.str[0] != lpp_tuning_df.phoneme_pair.str[0]).astype(int)
lpp_tuning_df["perceptual_tuning_matches_we"] = lpp_tuning_df["perceptual_tuning"] == lpp_tuning_df["lexical_evidence"]
lpp_tuning_df[["subject", "electrode_idx", "word_end", "perceptual_tuning", "sum_ambig_effect"]]


# %% [markdown]
# ## Helpers

# %%
def partial_corr(x, y, z):
    """
    Compute the partial correlation between x and y, controlling for z.
    """
    x_resid = sm.OLS(x, sm.add_constant(z)).fit().resid
    y_resid = sm.OLS(y, sm.add_constant(z)).fit().resid
    return stats.pearsonr(x_resid, y_resid)


def _resid(y, Z):
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


def _unit(x):
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / n if n > 0 else None


def _partial_r(early, late, Z):
    ex, lt = _unit(_resid(early, Z)), _unit(_resid(late, Z))
    if ex is None or lt is None:
        return np.nan
    return float(np.clip(ex @ lt, -1.0, 1.0))


def _partial_p(r, n, k):
    df = n - k - 2
    t = r * np.sqrt(df / (1 - r**2))
    return 2 * stats.t.sf(np.abs(t), df)


def _design(xs, cols, steps):
    """Design matrix: constant + continuous controls + optional step dummies.

    Step enters as dummies rather than a linear term because neither response
    is expected to be linear in continuum step; a linear term underfits and
    leaves residual stimulus-driven covariance behind.
    """
    parts = [np.ones((len(xs), 1))]
    if cols:
        parts.append(xs[cols].to_numpy(float))
    if steps:
        D = pd.get_dummies(xs["resampled"].astype("category"),
                           drop_first=True, dtype=float)
        parts.append(D.to_numpy())
    return np.column_stack(parts)


def crossfit_matched_filter_peak(trials, n_folds=5, seed=0):
    """Out-of-fold matched-filter peak amplitude, one value per trial.

    Projects each trial onto the (uncentered) fold-averaged response shape,
    scaled so the projection estimates trial gain under a fixed-shape/
    varying-gain model. This is the least-variance *linear* peak-amplitude
    estimator under that model -- a lower-noise sibling of the flat window
    mean (hga_early/hga_late), weighted by the response's actual temporal
    shape instead of uniformly. Held-out folds avoid template circularity.
    A centered (DC-removed) template was tried and found to discard most of
    the gain signal -- simulation showed it strictly underperforms the
    uncentered version at every noise level tested.
    """
    n = trials.shape[0]
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, n_folds, size=n)
    out = np.full(n, np.nan)
    for f in range(n_folds):
        train = trials[folds != f]
        test_idx = np.flatnonzero(folds == f)
        if len(train) < 2 or len(test_idx) == 0:
            continue
        template = train.mean(axis=0)
        denom = template @ template
        if denom < 1e-8:
            out[test_idx] = trials[test_idx].mean(axis=1)
            continue
        w = template / denom
        out[test_idx] = trials[test_idx] @ w
    return out


def centroid_latency(trials, times):
    """Amplitude-weighted mean time (center of mass) of each trial's window, in seconds.

    Only positive-going signal is weighted (relu), since HGA fluctuates
    around zero and negative weights would make "center of mass" ill-defined.
    Unlike an argmax-based peak time, this integrates over the whole window
    rather than depending on a single noisy sample, giving much lower
    single-trial variance -- simulation showed centroid latency detects a
    true early/late timing coupling ~1.5x more sensitively than argmax or
    half-max rise time, at every noise level tested, with no inflation when
    no true timing coupling is present. Trials with no positive signal in
    the window (rare) return NaN and are dropped downstream.
    """
    pos = np.clip(trials, 0, None)
    denom = pos.sum(axis=1)
    denom = np.where(denom <= 1e-8, np.nan, denom)
    weighted = pos @ times
    return weighted / denom


# %%
# %% Per-trial HGA in each site's early-acoustic and late-perceptual windows
early_late_sites = pd.merge(
    (
        lpp
        [["subject", "electrode_idx", "phoneme_pair", "word_end",
          "projection_significant_uncorrected", "smin", "smax"]]
        .rename(columns={"projection_significant_uncorrected": "late_significant_uncorrected",
                         "smin": "smin_late", "smax": "smax_late"})
    ),
    (
        early_all_acoustic_windows
        [["subject", "electrode_idx", "phoneme_pair", "word_end",
          "significant_target", "mean_diff_raw_med", "smin_bootstrap", "smax_bootstrap"]]
        .rename(columns={"significant_target": "early_significant",
                         "mean_diff_raw_med": "early_mean_diff_raw_med",
                         "smin_bootstrap": "smin_early", "smax_bootstrap": "smax_early"})
    ),
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    how="left",
)
early_late_trial_rows = []

baseline_smin, baseline_smax = 0, 40

for loop_idx, (_, row) in enumerate(tqdm(early_late_sites.iterrows(), total=early_late_sites.shape[0], desc="Extracting early+late HGA")):
    ep_i = epochs_dict[row.subject]
    md_i = ep_i.metadata
    we_mask = (md_i.word_end == row.word_end).values
    if not we_mask.any():
        continue

    # Pick other electrodes from the same subject which are speech-responsive, but not for this word
    alt_sites_i = speech_responsive_df.query("subject == @row.subject and electrode_idx != @row.electrode_idx and speech_responsive")
    alt_sites_i = pd.merge(alt_sites_i, lpp.query("word_end == @row.word_end")[["subject", "electrode_idx"]],
                           on=["subject", "electrode_idx"], how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    if alt_sites_i.empty:
        raise ValueError(f"Could not find any alternative site for {row.subject} {row.electrode_idx} {row.phoneme_pair} {row.word_end}")
    
    # # (ideally also acoustically selective) but not a member of early_late_sites
    # # tier 1: responsive to THIS phoneme pair, but not in lpp
    # alt_sites_i = epp.query("subject == @row.subject and phoneme_pair == @row.phoneme_pair and electrode_idx != @row.electrode_idx")
    # alt_sites_i = pd.merge(alt_sites_i, lpp[["subject", "electrode_idx", "phoneme_pair"]], on=["subject", "electrode_idx", "phoneme_pair"],
    #                        how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    # alt_site_tier = 1
    # if alt_sites_i.empty:
    #     # tier 2: responsive to ANY phoneme pair, but not in lpp
    #     alt_sites_i = epp.query("subject == @row.subject and electrode_idx != @row.electrode_idx")
    #     alt_sites_i = pd.merge(alt_sites_i, lpp[["subject", "electrode_idx", "phoneme_pair"]], on=["subject", "electrode_idx", "phoneme_pair"],
    #                            how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    #     alt_site_tier = 2
    # if alt_sites_i.empty:
    #     # tier 3: speech responsive and not in lpp
    #     alt_sites_i = speech_responsive_df.query("subject == @row.subject and electrode_idx != @row.electrode_idx and speech_responsive")
    #     alt_sites_i = pd.merge(alt_sites_i, lpp[["subject", "electrode_idx"]], on=["subject", "electrode_idx"],
    #                            how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    #     alt_site_tier = 3

    print(f"Found alternative site for {row.subject} {row.electrode_idx} {row.phoneme_pair} {row.word_end}: " +
          (", ".join(str(r.electrode_idx) for _, r in alt_sites_i.iterrows())))

    data_i = ep_i.get_data(picks=row.electrode_idx).squeeze(1)
    alt_data_i = ep_i.get_data(picks=alt_sites_i.electrode_idx)

    baseline_i = data_i[:, baseline_smin:baseline_smax + 1].mean(axis=1)
    alt_baseline_i = alt_data_i[:, :, baseline_smin:baseline_smax + 1].mean(axis=-1).mean(axis=-1)

    early_smin_i, early_smax_i = int(row.smin_early), int(row.smax_early)
    hga_early_i = data_i[:, early_smin_i:early_smax_i + 1].mean(axis=1)

    # mean over both electrodes and time
    hga_alt_early_i = alt_data_i[:, :, early_smin_i:early_smax_i + 1].mean(axis=-1).mean(axis=-1)

    late_smin_i, late_smax_i = int(row.smin_late), int(row.smax_late)
    hga_late_i = data_i[:, late_smin_i:late_smax_i + 1].mean(axis=1)

    # Peak-amplitude (matched filter) and peak-latency (centroid) measures, computed
    # on the raw within-window traces of only this word_end's trials -- see
    # crossfit_matched_filter_peak / centroid_latency docstrings for rationale.
    early_window_we_i = data_i[we_mask][:, early_smin_i:early_smax_i + 1]
    late_window_we_i = data_i[we_mask][:, late_smin_i:late_smax_i + 1]

    mf_early_we_i = crossfit_matched_filter_peak(early_window_we_i, seed=2 * loop_idx)
    mf_late_we_i = crossfit_matched_filter_peak(late_window_we_i, seed=2 * loop_idx + 1)

    times_early_i = epoch_tmin + np.arange(early_smin_i, early_smax_i + 1) / epoch_sfreq
    times_late_i = epoch_tmin + np.arange(late_smin_i, late_smax_i + 1) / epoch_sfreq
    centroid_early_we_i = centroid_latency(early_window_we_i, times_early_i)
    centroid_late_we_i = centroid_latency(late_window_we_i, times_late_i)

    gain_mask = np.ones(data_i.shape[-1], bool)
    gain_mask[baseline_smin:baseline_smax + 1] = False
    gain_mask[early_smin_i:early_smax_i + 1] = False
    gain_mask[late_smin_i:late_smax_i + 1] = False
    hga_gain_i = data_i[:, gain_mask].mean(axis=1)

    md_we = md_i[we_mask]
    early_contrast_sign = np.sign(row.early_mean_diff_raw_med)
    early_late_trial_rows.append(pd.DataFrame({
        "subject": row.subject,
        "electrode_idx": row.electrode_idx,
        "phoneme_pair": row.phoneme_pair,
        "word_end": row.word_end,
        "epoch_idx": md_we.index,
        "resampled": md_we["resampled"].values,
        "behavior_dummy_forced": md_we["behavior_dummy_forced"].values,

        "late_significant_uncorrected": row.late_significant_uncorrected,
        "early_significant": row.early_significant,

        "early_contrast_sign": early_contrast_sign,

        "hga_baseline": baseline_i[we_mask],
        "hga_alt_baseline": alt_baseline_i[we_mask],

        "hga_early": hga_early_i[we_mask],
        "hga_alt_early": hga_alt_early_i[we_mask],
        "hga_late": hga_late_i[we_mask],

        # peak amplitude (out-of-fold matched filter) and peak latency (centroid)
        "mf_early": mf_early_we_i,
        "mf_late": mf_late_we_i,
        "centroid_early": centroid_early_we_i,
        "centroid_late": centroid_late_we_i,

        # local gain control
        "hga_gain": hga_gain_i[we_mask],

        "smin_early": early_smin_i,
        "smax_early": early_smax_i,
        "smin_late": late_smin_i,
        "smax_late": late_smax_i,
    }))

early_late_trial_df = pd.concat(early_late_trial_rows, ignore_index=True)

# %%
early_late_reg_df = early_late_trial_df.query("late_significant_uncorrected")

# get this in the same units as the behavior, on a range from 0 to 1
early_late_reg_df["resampled_centered"] = (early_late_reg_df["resampled"] - 1) / 5
# early_late_reg_df["resampled_centered"] = early_late_reg_df["resampled"] - 3.5
early_late_reg_df["site"] = (
    early_late_reg_df["subject"].astype(str) + "_" + early_late_reg_df["electrode_idx"].astype(str) + "_"
    + early_late_reg_df["phoneme_pair"] + "_" + early_late_reg_df["word_end"]
)
early_late_reg_df["subject_electrode_we"] = (
    early_late_reg_df["subject"].astype(str) + ":" + early_late_reg_df["electrode_idx"].astype(str) + ":" + early_late_reg_df["word_end"]
)
early_late_reg_df["subject_epoch"] = (
    early_late_reg_df["subject"].astype(str) + ":" + early_late_reg_df["electrode_idx"].astype(str) + ":" + early_late_reg_df["epoch_idx"].astype(str)
)
early_late_reg_df["hga_early_aligned"] = early_late_reg_df["hga_early"] * early_late_reg_df["early_contrast_sign"]

early_late_reg_df["early_tuning"] = (early_late_reg_df.early_contrast_sign > 0).astype(int)
early_late_reg_df["lexical_evidence"] = (early_late_reg_df.word_end.str[0] != early_late_reg_df.phoneme_pair.str[0]).astype(int)
early_late_reg_df["congruent"] = (early_late_reg_df.lexical_evidence == early_late_reg_df.early_contrast_sign).astype(int)

# %% [markdown]
# ## Pedagogical check: what do these measures actually compute on single trials?
#
# Before running the coupling analysis: for each trial-level measure, find the
# site where it varies most across trials, pick a handful of trials spanning
# that range, and plot the raw HGA trace with the measure's value made visible
# directly on the trace (flat mean -> horizontal line; matched-filter peak ->
# fitted template shape; centroid -> shaded positive mass + its balance point).
# This is a sanity check on real data, not part of the analysis itself.

# %%
DEMO_N_TRIALS = 4
DEMO_PAD_S = 0.15  # context shown around the analysis window, in seconds


def _demo_pick_cell(measure_col, min_n=40):
    """(subject, electrode_idx, phoneme_pair, word_end) with the largest spread in measure_col."""
    df = early_late_trial_df.query("late_significant_uncorrected")
    spread = (
        df.groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"])[measure_col]
        .apply(lambda s: (s.max() - s.min()) if s.notna().sum() >= min_n else np.nan)
    )
    keys = spread.idxmax()
    return dict(zip(["subject", "electrode_idx", "phoneme_pair", "word_end"], keys))


def _demo_pick_trials(cell_df, measure_col, n_show=DEMO_N_TRIALS):
    """n_show trials spanning measure_col's range within one site."""
    s = cell_df[["epoch_idx", measure_col]].dropna().sort_values(measure_col)
    pick_pos = np.linspace(0, len(s) - 1, n_show).round().astype(int)
    return s.iloc[pick_pos]


def demo_measure(measure_col, window, annotate_fn, title):
    cell = _demo_pick_cell(measure_col)
    cell_mask = (
        (early_late_trial_df.subject == cell["subject"])
        & (early_late_trial_df.electrode_idx == cell["electrode_idx"])
        & (early_late_trial_df.phoneme_pair == cell["phoneme_pair"])
        & (early_late_trial_df.word_end == cell["word_end"])
        & (early_late_trial_df.late_significant_uncorrected)
    )
    cell_df = early_late_trial_df.loc[cell_mask]
    picks = _demo_pick_trials(cell_df, measure_col)

    smin, smax = int(cell_df[f"smin_{window}"].iloc[0]), int(cell_df[f"smax_{window}"].iloc[0])
    win_times = epoch_tmin + np.arange(smin, smax + 1) / epoch_sfreq

    ep_i = epochs_dict[cell["subject"]]
    data_i = ep_i.get_data(picks=cell["electrode_idx"]).squeeze(1)
    pad = int(round(DEMO_PAD_S * epoch_sfreq))
    lo, hi = max(0, smin - pad), min(data_i.shape[1] - 1, smax + pad)
    times = epoch_tmin + np.arange(lo, hi + 1) / epoch_sfreq
    traces = data_i[picks.epoch_idx.to_numpy(), lo:hi + 1]

    # visualization-only template: average window shape over ALL trials at this
    # site (not the out-of-fold, per-fold template actually used in the analysis)
    template = data_i[cell_df.epoch_idx.to_numpy(), smin:smax + 1].mean(axis=0)

    win_lo_idx, win_hi_idx = smin - lo, smax - lo + 1

    n_show = len(picks)
    fig, axes = plt.subplots(1, n_show, figsize=(3.0 * n_show, 2.6), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (_, row), trace in zip(axes, picks.iterrows(), traces):
        win_trace = trace[win_lo_idx:win_hi_idx]
        ax.plot(times, trace, color="0.25", lw=1.1)
        ax.axvspan(win_times[0], win_times[-1], color="C0", alpha=0.10, lw=0)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        annotate_fn(ax, win_times, win_trace, row[measure_col], template)
        ax.set_title(f"epoch {int(row.epoch_idx)}\n{measure_col}={row[measure_col]:.2f}", fontsize=8.5)
        ax.set_xlabel("t (s)", fontsize=8)
    axes[0].set_ylabel("HGA (z)", fontsize=8)
    fig.suptitle(
        f"{title}\n{cell['subject']} e{cell['electrode_idx']} {cell['word_end']} "
        f"({window} window, {n_show} of {len(cell_df)} trials shown, chosen for max spread in {measure_col})",
        fontsize=9.5,
    )
    return fig, cell


def _annotate_mean(ax, win_times, win_trace, value, template):
    ax.hlines(value, win_times[0], win_times[-1], color="crimson", lw=1.8)


def _annotate_mf(ax, win_times, win_trace, value, template):
    ax.plot(win_times, template, color="crimson", lw=1.4, ls="--", alpha=0.8)
    ax.text(0.03, 0.95, f"peak={value:.2f}", transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="crimson")


def _annotate_centroid(ax, win_times, win_trace, value, template):
    pos = np.clip(win_trace, 0, None)
    ax.fill_between(win_times, 0, pos, color="crimson", alpha=0.35)
    ax.axvline(value, color="crimson", lw=1.8)


_ = demo_measure("hga_late", "late", _annotate_mean, "Flat window mean (existing measure)")

# %%
_ = demo_measure("mf_late", "late", _annotate_mf,
                  "Out-of-fold matched-filter peak amplitude\n(dashed = across-trial template shape, for display only)")

# %%
_ = demo_measure("centroid_late", "late", _annotate_centroid,
                  "Amplitude-weighted centroid peak latency\n(shaded = positive mass being weighted; line = its balance point)")

# %%
print("Sanity check: partial correlation between early HGA and resampled step, controlling for baseline HGA")
early_sanity_check = early_late_reg_df.groupby(["subject", "electrode_idx", "word_end"]).apply(
    lambda xs: pd.Series(dict(zip(
        ["rval", "pval"],
        partial_corr(xs["hga_early"], xs["resampled_centered"], xs[["hga_baseline", "epoch_idx"]]))))
)
early_sanity_check.head()

# %%
# %% Trial-level early→late coupling under nested control sets
GK = ["subject", "electrode_idx", "word_end"]

N_BOOT = 5_000
N_PERM = 100_000
ALPHA = 0.05
SEED = 0
CHUNK = 500

# Nested: each adds one class of confound. "primary" gets the bootstrap CIs.
CONTROL_SETS = {
    "uncontrolled": dict(cols=[],                                           steps=False),
    "base":    dict(cols=["hga_baseline", "epoch_idx"],                    steps=False),
    "step":    dict(cols=["hga_baseline", "epoch_idx"],                    steps=True),
    "global":  dict(cols=["hga_baseline", "epoch_idx",
                          "hga_alt_early", "hga_alt_baseline"],            steps=True),
    # "local":   dict(cols=["hga_baseline", "epoch_idx", "hga_gain"],        steps=True),
    "full":    dict(cols=["hga_baseline", "epoch_idx",
                          # "hga_gain",
                          "hga_alt_early", "hga_alt_baseline"],            steps=True),
}
PRIMARY = "full"


def run_spec(df, name, cols, steps, early_col="hga_early", late_col="hga_late",
             n_perm=N_PERM, n_boot=0, seed=SEED):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"spec '{name}' missing columns: {missing}")

    rng = np.random.default_rng(seed)
    cells, keys = {}, []
    n_dropped_total = 0
    for key, xs in df.groupby(GK, sort=True):
        early_raw = xs[early_col].to_numpy(float)
        late_raw = xs[late_col].to_numpy(float)
        # centroid latency can be NaN for trials with no positive signal in the
        # window (rare); drop those trials rather than assert on them.
        finite_mask = np.isfinite(early_raw) & np.isfinite(late_raw)
        n_dropped_total += (~finite_mask).sum()
        xs = xs.loc[finite_mask]
        early = early_raw[finite_mask]
        late = late_raw[finite_mask]
        Z = _design(xs, cols, steps)
        assert np.isfinite(Z).all(), f"non-finite controls {key}"
        k_eff = Z.shape[1] - 1
        assert len(xs) > k_eff + 10, f"cell {key} n={len(xs)} too small for k={k_eff}"
        assert np.linalg.matrix_rank(Z) == Z.shape[1], f"rank-deficient design in {key}"
        cells[key] = (early, late, Z)
        keys.append(key)
    if n_dropped_total:
        print(f"  [{name}] dropped {n_dropped_total} trials with non-finite {early_col}/{late_col}")

    obs_r = np.array([_partial_r(*cells[k]) for k in keys])
    obs_n = np.array([len(cells[k][0]) for k in keys])
    k_eff = np.array([cells[k][2].shape[1] - 1 for k in keys])
    assert np.isfinite(obs_r).all(), f"partial r failed in some cell ({name})"

    # Pivotal statistic equalizes the null across cells differing in n and in
    # the number of controls, so the family max isn't set by the smallest cell.
    scale = np.sqrt(obs_n - k_eff - 3)
    obs_piv = np.arctanh(obs_r) * scale

    resid_pairs = [(_unit(_resid(cells[k][0], cells[k][2])),
                    _unit(_resid(cells[k][1], cells[k][2]))) for k in keys]

    null_max = np.empty(n_perm)
    for start in trange(0, n_perm, CHUNK, desc=f"perm [{name}]", leave=False):
        b = min(CHUNK, n_perm - start)
        chunk = np.empty((b, len(keys)))
        for j, (ex, lt) in enumerate(resid_pairs):
            perm = rng.permuted(np.tile(ex, (b, 1)), axis=1)
            r_null = perm @ lt
            chunk[:, j] = np.abs(np.arctanh(np.clip(r_null, -0.9999, 0.9999)) * scale[j])
        null_max[start:start + b] = chunk.max(axis=1)

    p_fwer = (1 + (null_max[:, None] >= np.abs(obs_piv)[None, :]).sum(0)) / (1 + n_perm)
    crit_piv = np.quantile(null_max, 1 - ALPHA)

    out = pd.DataFrame(
        {"r": obs_r,
         "z": np.arctanh(obs_r),
         "pivotal": obs_piv,
         "p_uncorrected": _partial_p(obs_r, obs_n, k_eff),
         "p_fwer": p_fwer, "n": obs_n, "k": k_eff,
         "crit_r": np.tanh(crit_piv / scale)},
        index=pd.MultiIndex.from_tuples(keys, names=GK),
    )

    if n_boot:
        ci_lo, ci_hi, n_valid = (np.empty(len(keys)) for _ in range(3))
        for j, k in enumerate(tqdm(keys, desc=f"boot [{name}]", leave=False)):
            early, late, Z = cells[k]
            n = len(early)
            z_boot = np.full(n_boot, np.nan)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                if np.linalg.matrix_rank(Z[idx]) < Z.shape[1]:
                    continue                       # degenerate resample
                r_b = _partial_r(early[idx], late[idx], Z[idx])
                if np.isfinite(r_b):
                    z_boot[b] = np.arctanh(np.clip(r_b, -0.9999, 0.9999))
            z_boot = z_boot[np.isfinite(z_boot)]
            n_valid[j] = z_boot.size
            lo, hi = np.percentile(z_boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
            ci_lo[j], ci_hi[j] = np.tanh(lo), np.tanh(hi)
        out["ci_lo"], out["ci_hi"] = ci_lo, ci_hi
        out["z_ci_lo"], out["z_ci_hi"] = np.arctanh(ci_lo), np.arctanh(ci_hi)
        out["n_boot_valid"] = n_valid.astype(int)
        assert (n_valid > 0.9 * n_boot).all(), f"excessive bootstrap failures ({name})"

    out["survives"] = out["p_fwer"] < ALPHA
    out["survives_uncorrected"] = out["p_uncorrected"] < ALPHA
    if n_boot:
        out["ci_excludes_zero"] = np.sign(out["ci_lo"]) == np.sign(out["ci_hi"])
    out.attrs["crit_piv"] = crit_piv
    print(f"[{name:>6}] k={k_eff.min()}–{k_eff.max()}  crit_piv={crit_piv:.3f}  "
          f"{out['survives'].sum()}/{len(out)} survive")
    return out


# ------------------------------------------------------------------------- run
# Three trial-level target measures, all run through the same nested
# control-set / permutation-FWER machinery:
#   amplitude_mean     -- existing flat-window mean (hga_early/hga_late)
#   amplitude_peak     -- out-of-fold matched-filter peak amplitude (mf_early/mf_late)
#   latency_centroid   -- amplitude-weighted peak-time centroid (centroid_early/centroid_late)
MEASURES = {
    "amplitude_mean": dict(early_col="hga_early", late_col="hga_late"),
    "amplitude_peak": dict(early_col="mf_early", late_col="mf_late"),
    "latency_centroid": dict(early_col="centroid_early", late_col="centroid_late"),
}

results_by_measure = {}
for measure_name, measure_cols in MEASURES.items():
    print(f"\n=== measure: {measure_name} ===")
    results_by_measure[measure_name] = {
        name: run_spec(early_late_reg_df, f"{measure_name}/{name}",
                       n_boot=N_BOOT if name == PRIMARY else 0,
                       **measure_cols, **spec)
        for name, spec in CONTROL_SETS.items()
    }

# Downstream cells below were written against the amplitude-mean measure;
# `results` keeps that behavior unchanged. The other two measures are
# summarized separately just below and available via results_by_measure.
results = results_by_measure["amplitude_mean"]

fwer_raw = results[PRIMARY].sort_values("p_fwer")

# ------------------------------------------------------- attrition across specs
comp = pd.concat(
    {name: res[["r", "p_fwer", "survives", "survives_uncorrected"]]
     for name, res in results.items()},
    axis=1,
)
comp = comp.loc[results["base"]["r"].abs().sort_values(ascending=False).index]
print("\nper-cell r and FWER p under each control set:")
print(comp.round(3).to_string())

surv = pd.DataFrame({n: res["survives"] for n, res in results.items()})
print("\nsurvivors by spec:")
for name in CONTROL_SETS:
    hits = surv.index[surv[name]].tolist()
    print(f"  {name:>6}: {len(hits)}  {hits}")

# Shrinkage attributable to each added control, at the cells that ever survive
ever = surv.any(axis=1)
print("\n|r| at ever-surviving cells:")
print(pd.DataFrame({n: results[n].loc[ever, "r"] for n in CONTROL_SETS}).round(3).to_string())

# %% [markdown]
# ### Compare amplitude-mean vs. amplitude-peak vs. latency-centroid coupling
#
# Same nested control sets, same permutation-FWER machinery, evaluated at the
# "full" control spec for each of the three trial-level target measures.

# %%
measure_summary_rows = []
for measure_name in MEASURES:
    res = results_by_measure[measure_name][PRIMARY]
    measure_summary_rows.append({
        "measure": measure_name,
        "n_sites": len(res),
        "n_survive_fwer": int(res["survives"].sum()),
        "n_survive_uncorrected": int(res["survives_uncorrected"].sum()),
        "median_abs_r": res["r"].abs().median(),
        "max_abs_r": res["r"].abs().max(),
    })
measure_summary_df = pd.DataFrame(measure_summary_rows)
print("\ncross-measure comparison (control set = 'full'):")
print(measure_summary_df.round(3).to_string(index=False))

# per-site r values side by side, so individual electrodes can be compared across measures
measure_r_by_site = pd.concat(
    {measure_name: results_by_measure[measure_name][PRIMARY]["r"] for measure_name in MEASURES},
    axis=1,
)
print("\nper-site r under each measure (full control set):")
print(measure_r_by_site.round(3).to_string())

# %%
fwer = (
    pd.merge(results["step"], early_sanity_check.rename(columns=lambda col: f"early_sanity_{col}"),
             left_index=True, right_index=True)
    .sort_values("p_fwer")
)
fwer = fwer.sort_values("p_fwer")

fwer["r_abs"] = fwer["r"].abs()
fwer["early_sanity_rval_abs"] = fwer["early_sanity_rval"].abs()

print("Rank-correlation of early-late HGA partial correlation with early HGA sanity check")
print("If this result is positive, this suggests that the observed early--late correlation may be driven by an SNR confound")
print(stats.spearmanr(fwer["early_sanity_rval_abs"], fwer["r_abs"]))

# Sanity check: the rvalues (estimated at the trial-level) align with the tuning relationship we found averaging over trials
# NB this isn't *necessary* but it'd be hard to interpret if there were a sign mismatch
# for significant sites.
fwer = pd.merge(fwer.reset_index(),
                lpp_tuning_df[["subject", "electrode_idx", "word_end", "sum_ambig_effect", "early_acoustic_mean_diff_raw_med"]],
                on=["subject", "electrode_idx", "word_end"], how="left")
fwer["early_late_sign_mismatch"] = np.where(fwer.sum_ambig_effect.isna(), np.nan,
                                            np.sign(fwer.sum_ambig_effect) != np.sign(fwer.early_acoustic_mean_diff_raw_med))
# do the two analyses agree?
fwer["analyses_consistent"] = (fwer.r < 0) == fwer.early_late_sign_mismatch
# fwer = fwer.drop(columns=["sum_ambig_effect", "early_acoustic_mean_diff_raw_med", "early_late_sign_mismatch"])

fwer[["subject", "electrode_idx", "word_end", "r", "survives", "sum_ambig_effect", "early_acoustic_mean_diff_raw_med"]]

# %%
# check_df = early_late_reg_df.query("subject == 'EC250' and electrode_idx == 185")
check_df = early_late_reg_df.query("subject == 'EC260' and electrode_idx == 204")

# sns.scatterplot(data=check_df, x="hga_early", y="hga_late")
print(stats.pearsonr(check_df.hga_early, check_df.hga_late))
print(partial_corr(check_df.hga_early, check_df.hga_late, check_df[["hga_baseline", "epoch_idx", "resampled_centered"]]))
print(partial_corr(check_df.hga_early, check_df.hga_late, check_df[["hga_baseline", "epoch_idx", "resampled_centered",
                                                                    "hga_alt_baseline", "hga_alt_early"]]))

# compute hga_late | controls, hga_alt_early | controls, then correlate those residuals
Z = sm.add_constant(check_df[["hga_baseline", "epoch_idx", "resampled_centered", "hga_alt_baseline"]])
hga_late_resid = _resid(check_df.hga_late, Z)
hga_alt_early_resid = _resid(check_df.hga_alt_early, Z)
print("late | controls ~ alt_early | controls", stats.pearsonr(hga_late_resid, hga_alt_early_resid))
f, ax = plt.subplots(figsize=(2, 2))
ax.set_title("late vs alt_early")
ax.set_xlabel("alt_early | controls")
ax.set_ylabel("late | controls")
sns.regplot(x=hga_alt_early_resid, y=hga_late_resid, scatter_kws={"s": 10}, ax=ax)

# compute hga_early | controls, hga_late | controls, then correlate those residuals
Z = sm.add_constant(check_df[["hga_baseline", "epoch_idx", "resampled_centered"]])
hga_early_resid = _resid(check_df.hga_early, Z)
hga_late_resid = _resid(check_df.hga_late, Z)
print("late | controls ~ early | controls", stats.pearsonr(hga_late_resid, hga_early_resid))
f, ax = plt.subplots(figsize=(2, 2))
ax.set_title("late | controls ~ early | controls")
ax.set_xlabel("early | controls")
ax.set_ylabel("late | controls")
sns.regplot(x=hga_early_resid, y=hga_late_resid, scatter_kws={"s": 10}, ax=ax)

print(check_df.query("resampled in (1, 6)").groupby("resampled")[["hga_early", "hga_late"]].mean())

# %%
# %% Cross-temporal single-trial coupling matrices: survivors vs. counterfactual sites

DECIM = 2
N_PERM_MAT = 1000
ALPHA_MAT = 0.05
N_COUNTER = None          # None = one counterfactual per survivor

N_RANDOM = 6              # random lpp-negative cells, unmatched
RANDOM_SEED = 42

baseline_smin, baseline_smax = 0, 40


def _hat_resid(Y, Z):
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return Y - Z @ beta


def _unit_cols(R):
    R = R - R.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(R, axis=0, keepdims=True)
    nrm[nrm == 0] = np.inf
    return R / nrm


# ---------------------------------------------------------------- block helpers
MIN_LAG_SAMP = 6          # decimated samples; excludes the autocorr ridge

def _block_means(R, hl, he):
    """B[j, i] = mean of R[j:j+hl, i:i+he] (integral image)."""
    C = np.cumsum(np.cumsum(R, axis=0), axis=1)
    C = np.pad(C, ((1, 0), (1, 0)))
    return (C[hl:, he:] - C[:-hl, he:] - C[hl:, :-he] + C[:-hl, :-he]) / (hl * he)


def _block_summary(R, i0, i1, j0, j1):
    he, hl = i1 - i0, j1 - j0
    d = j0 - i0

    def _invalid(reason):
        return {"r_block": np.nan, "lag_null": np.empty(0), "any_null": np.empty(0),
                    "lag_pct": np.nan, "any_pct": np.nan, "block_shape": (hl, he),
                    "lag_samp": d, "block_valid": False, "block_reason": reason}

    if j0 <= i1:
        return _invalid("late window overlaps or precedes early window")
    if d < MIN_LAG_SAMP:
        return _invalid(f"block lag {d} < MIN_LAG_SAMP ({MIN_LAG_SAMP})")

    B = _block_means(R, hl, he)
    if not (0 <= j0 < B.shape[0] and 0 <= i0 < B.shape[1]):
        return _invalid("block runs off the matrix edge")
    r_block = B[j0, i0]

    idx = np.arange(B.shape[1])
    ok = (idx + d >= 0) & (idx + d < B.shape[0]) & (idx != i0)
    lag_null = B[idx[ok] + d, idx[ok]]
    if lag_null.size <= 3:
        return _invalid(f"matched-lag null too small (n={lag_null.size})")

    jj, ii = np.indices(B.shape)
    any_mask = (jj - ii) >= MIN_LAG_SAMP
    any_mask[j0, i0] = False
    any_null = B[any_mask]

    return {"r_block": r_block, "lag_null": lag_null, "any_null": any_null,
                "lag_pct": 100.0 * (lag_null < r_block).mean(),
                "any_pct": 100.0 * (any_null < r_block).mean(),
                "block_shape": (hl, he), "lag_samp": d,
                "block_valid": True, "block_reason": ""}


def build_tg(subject, electrode_idx, word_end, kind, label_extra="", seed=0):
    """Cross-temporal single-trial correlation matrix for one cell.

    Controls: baseline, trial drift, continuum-step dummies. The local-gain and
    alt-electrode regressors from the partial-r analysis are deliberately omitted
    here — both are broad temporal averages, so residualizing every timepoint on
    them would remove the very structure the matrix is meant to display.
    """
    ep_i = epochs_dict[subject]
    md_i = ep_i.metadata
    we_mask = (md_i.word_end == word_end).values
    assert we_mask.any(), f"no trials for {subject} {word_end}"

    cell = early_late_trial_df.loc[
        (early_late_trial_df.subject == subject)
        & (early_late_trial_df.electrode_idx == electrode_idx)
        & (early_late_trial_df.word_end == word_end)
    ]
    assert len(cell), f"cell not found: {subject} e{electrode_idx} {word_end}"
    cell = cell.iloc[0]

    data_i = ep_i.get_data(picks=electrode_idx).squeeze(1)
    baseline_i = data_i[:, baseline_smin:baseline_smax + 1].mean(axis=1)
    md_we = md_i[we_mask]

    # Build control predictors.
    # these will be partialed out of every timepoint,
    # so the matrix shows the residual early-late correlation
    # after controlling for these confounds.
    step_dummies = pd.get_dummies(md_we["resampled"].astype("category"),
                                  drop_first=True, dtype=float).to_numpy()
    Z = np.column_stack([
        np.ones(we_mask.sum()),
        baseline_i[we_mask],
        md_we.index.to_numpy(float),
        step_dummies,
    ])
    assert np.linalg.matrix_rank(Z) == Z.shape[1], f"rank-deficient design {subject} e{electrode_idx}"

    Y = data_i[we_mask][:, ::DECIM]
    U = _unit_cols(_hat_resid(Y, Z))
    R = U.T @ U

    rng_m = np.random.default_rng(seed)
    n_tr = U.shape[0]
    null_max = np.empty(N_PERM_MAT)
    for b in range(N_PERM_MAT):
        null_max[b] = np.abs(U[rng_m.permutation(n_tr)].T @ U).max()
    crit = np.quantile(null_max, 1 - ALPHA_MAT)

    times = epoch_tmin + np.arange(data_i.shape[1])[::DECIM] / epoch_sfreq
    to_t = lambda s: epoch_tmin + s / epoch_sfreq
    e_lo, e_hi = to_t(cell.smin_early), to_t(cell.smax_early)
    l_lo, l_hi = to_t(cell.smin_late), to_t(cell.smax_late)

    # --- quantitative summary: is the early/late block special, or just pedestal?
    i0 = int(np.searchsorted(times, e_lo))
    i1 = max(i0 + 1, int(np.searchsorted(times, e_hi)))
    j0 = int(np.searchsorted(times, l_lo))
    j1 = max(j0 + 1, int(np.searchsorted(times, l_hi)))
    bs = _block_summary(R, i0, i1, j0, j1)
    if not bs["block_valid"]:
        print(f"  note: {subject} e{electrode_idx} {word_end} — {bs['block_reason']}; "
              "no inset for this panel")

    ti, tj = (i0 + i1) // 2, (j0 + j1) // 2
    r_cell = R[tj, ti]
    ridge = np.mean([R[k, k + 1] for k in range(R.shape[0] - 1)])

    meta = dict(
        kind=kind,
        label=f"{subject} e{electrode_idx}\n{word_end}{label_extra}",
        times=times, crit=crit, n_trials=n_tr,
        early=(e_lo, e_hi), late=(l_lo, l_hi),
        r_cell=r_cell, ridge=ridge, **bs,
        word_offset=WORD_PHASE_DF.query("word == @word_end and phase == 'offset'").iloc[0].start,
        pod=WORD_PHASE_DF.query("word == @word_end and phase == 'pod'").iloc[0].start,
    )
    print(f"[{kind:>7}] {subject} e{electrode_idx} {word_end}: n={n_tr}  "
          f"r_block={bs['r_block']:+.3f}  lag-matched med={np.median(bs['lag_null']):+.3f} "
          f"(pct {bs['lag_pct']:.0f})  any-block pct {bs['any_pct']:.0f}  "
          f"ridge={ridge:.3f}  crit={crit:.3f}")
    return R, meta


# ------------------------------------------------------- pick counterfactual cells
cell_index = (early_late_trial_df
              .groupby(["subject", "electrode_idx", "word_end"])
              .agg(n=("hga_early", "size"),
                   late_sig=("late_significant_uncorrected", "first"))
              .reset_index())

survivors = fwer.query("survives")
null_pool = cell_index.query("~late_sig")
print(f"{len(null_pool)} candidate cells with no late perceptual effect")

# Match each survivor to a same-subject, non-significant cell of similar trial count.
counters, used = [], set()
for _, row in survivors.iterrows():
    n_target = cell_index.query(
        "subject == @row.subject and electrode_idx == @row.electrode_idx "
        "and word_end == @row.word_end"
    ).iloc[0].n
    cand = null_pool.query("subject == @row.subject").copy()
    if cand.empty:
        cand = null_pool.copy()
        print(f"  note: no same-subject counterfactual for {row.subject} e{row.electrode_idx}; "
              "drawing across subjects (global-state comparison is weaker)")
    cand = cand[~cand.apply(lambda r: (r.subject, r.electrode_idx, r.word_end) in used, axis=1)]
    assert len(cand), "counterfactual pool exhausted"
    # prefer same word_end so the acoustic timeline matches, then closest n
    cand["same_we"] = (cand.word_end == row.word_end).astype(int)
    cand["dn"] = (cand.n - n_target).abs()
    print(cand)
    pick = cand.sort_values(["same_we", "dn"], ascending=[False, True]).iloc[0]
    used.add((pick.subject, pick.electrode_idx, pick.word_end))
    counters.append(pick)

if N_COUNTER is not None:
    counters = counters[:N_COUNTER]

# ------------------------------------------------- random (unmatched) null cells
avail = null_pool[~null_pool.apply(
    lambda r: (r.subject, r.electrode_idx, r.word_end) in used, axis=1)]
n_draw = min(N_RANDOM, len(avail))
if n_draw < N_RANDOM:
    print(f"  note: only {n_draw} unused lpp-negative cells available")
randoms = [r for _, r in avail.sample(n=n_draw, random_state=RANDOM_SEED).iterrows()]
print(f"random lpp-negative draw: "
      + ", ".join(f"{r.subject} e{r.electrode_idx} {r.word_end}" for r in randoms))

# ------------------------------------------------------------------------- build
GROUPS = ["late sig", "matched null", "random null"]
mats, metas = [], []

for _, row in survivors.iterrows():
    R, m = build_tg(row.subject, row.electrode_idx, row.word_end,
                    kind="late sig", label_extra=f"  (r={row.r:+.2f})")
    mats.append(R); metas.append(m)

for pick in counters:
    R, m = build_tg(pick.subject, pick.electrode_idx, pick.word_end,
                    kind="matched null", label_extra="  (matched, no late)")
    mats.append(R); metas.append(m)

for pick in randoms:
    R, m = build_tg(pick.subject, pick.electrode_idx, pick.word_end,
                    kind="random null", label_extra="  (random, no late)")
    mats.append(R); metas.append(m)

# ------------------------------------------------------------------------- plot
vmax = max(np.abs(m - np.eye(len(m))).max() for m in mats)
row_of = {g: i for i, g in enumerate(GROUPS)}
ncols = max(sum(m["kind"] == g for m in metas) for g in GROUPS)
fig, axes = plt.subplots(len(GROUPS), ncols,
                         figsize=(3.1 * ncols, 2.8 * len(GROUPS)),
                         constrained_layout=True, squeeze=False)

col_cursor = {g: 0 for g in GROUPS}
for R, meta in zip(mats, metas):
    r_i = row_of[meta["kind"]]
    c_i = col_cursor[meta["kind"]]
    col_cursor[meta["kind"]] += 1
    ax = axes[r_i, c_i]

    t = meta["times"]
    im = ax.imshow(R, origin="lower", extent=[t[0], t[-1], t[0], t[-1]],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")

    ax.plot(t[[0, -1]], t[[0, -1]], color="k", lw=0.5, ls=":", alpha=0.5)
    ax.axvspan(*meta["early"], color="k", alpha=0.10, lw=0)
    ax.axhspan(*meta["late"], color="k", alpha=0.10, lw=0)
    # ax.plot(np.mean(meta["early"]), np.mean(meta["late"]),
    #         marker="+", color="k", ms=9, mew=1.4)

    for v, c in [(0.0, "k"), (meta["pod"], "red"), (meta["word_offset"], "blue")]:
        ax.axvline(v, color=c, lw=0.5, ls="--", alpha=0.5)
        ax.axhline(v, color=c, lw=0.5, ls="--", alpha=0.5)

    # ---- inset: where does the highlighted block sit in the matrix's own distribution?
    if meta["block_valid"]:
        axin = ax.inset_axes([0.55, 0.10, 0.42, 0.26])
        axin.patch.set_facecolor("white"); axin.patch.set_alpha(0.88)

        lo = min(meta["any_null"].min(), meta["lag_null"].min(), meta["r_block"])
        hi = max(meta["any_null"].max(), meta["lag_null"].max(), meta["r_block"])
        bins = np.linspace(lo, hi, 26)
        axin.hist(meta["any_null"], bins=bins, density=True, histtype="step",
                color="0.45", lw=0.8)
        axin.hist(meta["lag_null"], bins=bins, density=True,
                color="0.75", edgecolor="none")
        axin.axvline(meta["r_block"], color="crimson", lw=1.2)
        axin.axvline(0, color="k", lw=0.4, ls=":")

        axin.text(0.98, 0.95, f"{meta['lag_pct']:.0f}%", transform=axin.transAxes,
                ha="right", va="top", fontsize=6, color="crimson")
        axin.set_yticks([])
        axin.tick_params(axis="x", labelsize=5.5, length=2, pad=1)
        axin.set_xticks([round(lo, 1), 0, round(hi, 1)])
        sns.despine(ax=axin, left=True, right=True, top=True)
        axin.spines["bottom"].set_linewidth(0.5)

    ax.set_title(meta["label"], fontsize=7.5)
    ax.set_xlim(-0.1, 1.0); ax.set_ylim(-0.1, 1.0)
    ax.tick_params(labelsize=7)

for ax in axes[-1, :]:
    ax.set_xlabel("$t_1$ (s)", fontsize=8)
for g, lab in zip(GROUPS, ["late perceptual", "no late (matched)", "no late (random)"]):
    axes[row_of[g], 0].set_ylabel(f"{lab}\n$t_2$ (s)", fontsize=8)
for ax in axes.flat:
    if not ax.has_data():
        ax.set_visible(False)

cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("$r$", fontsize=7.5, rotation=0)
cb.ax.tick_params(labelsize=7)

print("\nsummary:")
print(pd.DataFrame([{k: m[k] for k in
                     ["kind", "label", "n_trials", "r_cell", "ridge", "crit",
                     "r_block", "lag_pct", "any_pct"]} for m in metas])
      .assign(label=lambda d: d.label.str.replace("\n", " ")).round(3).to_string(index=False))

# %%
win = (early_late_trial_df
       .groupby(["subject", "electrode_idx", "word_end"])
       [["smin_early", "smax_early", "smin_late", "smax_late"]].first()
       .reset_index())
win["dur_early"] = win.smax_early - win.smin_early
win["dur_late"]  = win.smax_late  - win.smin_late
win["gap_samp"]  = win.smin_late  - win.smax_early
print(win.sort_values("gap_samp").head(15).to_string(index=False))
print(f"\n{(win.gap_samp <= 0).mean():.1%} of cells have overlapping early/late windows")

# %% [markdown]
# ## lmer regression

# %%
el_early_sanity_model = smf.mixedlm(
    "hga_early_aligned ~ resampled_centered + hga_baseline",
    data=early_late_reg_df, groups=early_late_reg_df["subject_electrode_we"],
).fit()

el_model = smf.mixedlm(
    "hga_late ~ hga_early + hga_baseline",
    data=early_late_reg_df, groups=early_late_reg_df["subject_electrode_we"],
).fit()

el_reverse_model = smf.mixedlm(
    "hga_early ~ hga_late + hga_baseline",
    data=early_late_reg_df, groups=early_late_reg_df["subject_electrode_we"],
).fit()

print(el_early_sanity_model.summary())
print(el_model.summary())
print(el_reverse_model.summary())

# %%
el_base_model = smf.mixedlm(
    "hga_late ~ resampled_centered",
    data=early_late_reg_df, groups=early_late_reg_df["subject_electrode_we"],
).fit(reml=False)
el_full_model = smf.mixedlm(
    "hga_late ~ resampled_centered + hga_early",
    data=early_late_reg_df, groups=early_late_reg_df["subject_electrode_we"],
).fit(reml=False)
el_lr_stat = 2 * (el_full_model.llf - el_base_model.llf)
el_lr_p = stats.chi2.sf(el_lr_stat, df=1)
print(f"Likelihood ratio test for early HGA predicting late HGA: LR stat = {el_lr_stat:.3f}, p = {el_lr_p:.3e}")

# %%
el_reverse_base_model = smf.mixedlm(
    "hga_early ~ resampled_centered",
    data=early_late_reg_df, groups=early_late_reg_df["subject_electrode_we"],
).fit(reml=False)
el_reverse_full_model = smf.mixedlm(
    "hga_early ~ resampled_centered + hga_late",
    data=early_late_reg_df, groups=early_late_reg_df["subject_electrode_we"],
).fit(reml=False)
el_reverse_lr_stat = 2 * (el_reverse_full_model.llf - el_reverse_base_model.llf)
el_reverse_lr_p = stats.chi2.sf(el_reverse_lr_stat, df=1)
print(f"Likelihood ratio test for late HGA predicting early HGA: LR stat = {el_reverse_lr_stat:.3f}, p = {el_reverse_lr_p:.3e}")

# %%
import numpy as np, warnings
from tqdm import trange
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def fit_lr(df, yname="y", **kwargs):
    """Return LR stat for adding hga_early, given outcome column `yname`."""
    base = smf.mixedlm(f"{yname} ~ resampled_centered", data=df,
                       groups=df["subject_electrode_we"]).fit(reml=False, **kwargs)
    full = smf.mixedlm(f"{yname} ~ resampled_centered + hga_early", data=df,
                       groups=df["subject_electrode_we"]).fit(reml=False, **kwargs)
    return max(2 * (full.llf - base.llf), 0.0)   # clip tiny negatives from optimizer

df = early_late_reg_df.copy()

# observed
df["y"] = df["hga_late"]
obs_stat = fit_lr(df)

# reduced-model decomposition (Freedman-Lane)
red = smf.mixedlm("hga_late ~ resampled_centered", data=df,
                  groups=df["subject_electrode_we"]).fit(reml=False)
fitted = red.fittedvalues.to_numpy()          # marginal (fixed-effects) fit
resid  = df["hga_late"].to_numpy() - fitted

grp_idx = [np.flatnonzero(df["subject_electrode_we"].to_numpy() == g)
           for g in df["subject_electrode_we"].unique()]

n_perm = 1000
rng = np.random.default_rng(0)
null_stats = np.full(n_perm, np.nan)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    for i in trange(n_perm):
        r = resid.copy()
        for idx in grp_idx:                    # shuffle within electrode only
            r[idx] = rng.permutation(r[idx])
        df["y"] = fitted + r
        try:
            null_stats[i] = fit_lr(df)
        except Exception:
            pass                               # leave as nan, drop below

null_stats = null_stats[~np.isnan(null_stats)]
p_perm = (1 + np.sum(null_stats >= obs_stat)) / (1 + null_stats.size)

print("Permutation test for early HGA predicting late HGA (Freedman-Lane):")
print(f"observed LR = {obs_stat:.3f}, chi2 p = {stats.chi2.sf(obs_stat, 1):.3e}")
print(f"permutation p = {p_perm:.4f}  ({null_stats.size} valid perms, "
      f"null median = {np.median(null_stats):.3f})")

# %%
