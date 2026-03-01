# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# Companion to prepare_neurometrics.py.
#
# Key difference: behavioral site selection uses a sliding t-test
# (find_site_windows) rather than behavioral decoder ROC-AUC improvement.
#
# Pipeline:
# 1. Select phonetically-responsive sites identically (decoder ROC-AUC).
# 2. For each phonetically-selective site × word_end, run find_site_windows()
#    to get the best behavioral window and its t-statistic/p-value.
# 3. Filter by behav_ttest_pvalue_threshold → behav_peaks_ttest_df.
# 4. Snap t-test windows to nearest window in the behavioral decoder searchlight
#    (for compatibility with behav_roc_auc_searchlight_df in A_neurometrics_ttest).
# 5. Build the same PaperData outputs as prepare_neurometrics for downstream use.
#
# Extra outputs vs prepare_neurometrics:
#   ttest_behav_df.parquet        – full t-test results (pre-threshold, all candidate sites)
#   behav_peaks_ttest_df.parquet  – filtered by behav_ttest_pvalue_threshold
#   ttest_snap_report.parquet     – per-site snap offsets for auditing

# %%
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger as L
from tqdm.auto import tqdm

tqdm.pandas()

# %%
from src.data import add_metadata_features
from src.stimuli import (
    OFFSET_DICT,
    WORD_END_TO_PHONEME_PAIR,
    POD_dict,
)
from src.viz_paper import (
    PaperData,
    extract_hga_windows_df,
    find_site_windows,
    phoneme_pair_enum,
    pl_roc_auc,
    subject_enum,
    word_end_enum,
)

# %% tags=["parameters"]
all_epochs = list(Path("outputs/epochs_preprocessed").glob("*_epo.fif"))

A_behav_predictions = list(
    Path("outputs/causal4/behavior_decoding_single_electrode_summarize").glob(
        "*/A-predictions.parquet"
    )
)
A_early_behav_predictions = list(
    Path("outputs/causal4/behavior_decoding_single_electrode_summarize").glob(
        "*/A_early-predictions.parquet"
    )
)

phon_predictions_path = Path(
    "outputs/causal4/A_predictions/behavior_to_phonetic_decoding.parquet"
)

electrode_paths = list(Path("outputs/causal4/find_speech_responsive/").glob("*.csv"))

epoch_tmin = -0.4
epoch_sfreq = 100

phon_response_tmin_min = 0.0
all_response_tmax_max = 1.3

phon_response_peak_threshold = 0.6
ambiguous_response_threshold = 2

# Threshold on the per-site behavioral contrast p-value (two-tailed Welch's t-test).
# Sites with p < this threshold are included in behav_peaks_ttest_df.
behav_ttest_pvalue_threshold = 0.05

outdir = "outputs/causal4/prepare_neurometrics_ttest"

# %%
phon_response_smin_min = (phon_response_tmin_min - epoch_tmin) * epoch_sfreq
all_response_smax_max = int((all_response_tmax_max - epoch_tmin) * epoch_sfreq)

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load raw data

# %%
electrode_df = pl.concat([pl.read_csv(p) for p in electrode_paths]).with_columns(
    pl.col("subject").cast(subject_enum)
)

# %%
epochs = {}
for path in all_epochs:
    subject = re.findall(r"(EC[\d]+)_epo", str(path))[0]
    ep_i = mne.read_epochs(path, verbose=False)
    assert ep_i.metadata is not None
    ep_i.metadata = add_metadata_features(ep_i.metadata)
    epochs[subject] = ep_i

# %%
all_md = pl.from_pandas(
    pd.concat(
        [
            ep.metadata.rename_axis("epoch_idx").assign(subject=subject).reset_index()
            for subject, ep in epochs.items()
        ],
        ignore_index=True,
    ).drop(columns=["TDT Block"])
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)

# %%
word_end_df = pl.from_pandas(
    pd.DataFrame.from_dict(OFFSET_DICT, orient="index", columns=["word_end_offset"])
    .rename_axis("word_end")
    .join(
        pd.DataFrame.from_dict(
            WORD_END_TO_PHONEME_PAIR, orient="index", columns=["phoneme_pair"]
        ).rename_axis("word_end"),
        on="word_end",
    )
    .reset_index()
    .join(
        pd.DataFrame.from_dict(POD_dict, orient="index", columns=["pod"]).rename_axis(
            "phoneme_pair"
        ),
        on="phoneme_pair",
    )
    .reset_index()
).with_columns(
    pl.col("word_end").cast(word_end_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    ((pl.col("word_end_offset") - epoch_tmin) * epoch_sfreq).alias(
        "word_end_offset_sample"
    ),
    ((pl.col("pod") - epoch_tmin) * epoch_sfreq).alias("pod_sample"),
)

# %% [markdown]
# ## Phonetic peaks (identical to prepare_neurometrics)

# %%
phon_pred_df = pl.read_parquet(phon_predictions_path).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    (pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"),
)

# %%
group_cols = ["subject", "electrode_idx", "phoneme_pair", "smin", "smax", "fold"]
phon_roc_auc_searchlight_df = pl_roc_auc(
    df=phon_pred_df.filter(
        (pl.col("smin") >= phon_response_smin_min)
        & (pl.col("smax") <= all_response_smax_max)
    ),
    target_col="decoder_target",
    proba_col="decoder_proba",
    group_cols=group_cols,
    roc_auc_name="phon_roc_auc",
)

# %%
phon_roc_auc_mean_df = phon_roc_auc_searchlight_df.group_by(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
).agg(pl.col("phon_roc_auc").mean())

# %%
phon_peaks_df = (
    phon_roc_auc_mean_df.join(
        word_end_df.group_by(["phoneme_pair"]).agg(pl.max("word_end_offset_sample")),
        on=["phoneme_pair"],
        how="left",
    )
    .filter(
        pl.col("smin") >= phon_response_smin_min,
        pl.col("smax") <= pl.col("word_end_offset_sample"),
        pl.col("phon_roc_auc") >= phon_response_peak_threshold,
    )
    .sort("phon_roc_auc", descending=True)
    .group_by(["subject", "electrode_idx", "phoneme_pair"])
    .first()
)

# %% [markdown]
# ## Behavioral decoder data (kept for downstream functional-dissociation lookup)

# %%
behav_pred_df = pl.concat(
    [pl.read_parquet(f) for f in A_behav_predictions + A_early_behav_predictions]
).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
    (pl.col("decoder_target") == 1).cast(pl.Int8).alias("decoder_target"),
)

# %%
behav_baseline_df = pl_roc_auc(
    df=behav_pred_df.unique(
        subset=[
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "word_end",
            "epoch_idx",
            "fold",
        ],
        keep="first",
    ),
    target_col="decoder_target",
    proba_col="baseline_decoder_proba",
    group_cols=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
    roc_auc_name="behav_roc_auc_baseline",
)

# %%
group_cols = [
    "subject",
    "electrode_idx",
    "phoneme_pair",
    "word_end",
    "smin",
    "smax",
    "fold",
]
behav_roc_auc_searchlight = pl_roc_auc(
    df=behav_pred_df.filter(pl.col("smax") <= all_response_smax_max),
    target_col="decoder_target",
    proba_col="full_decoder_proba",
    group_cols=group_cols,
    roc_auc_name="behav_roc_auc",
)

# %%
behav_roc_auc_searchlight_df = behav_roc_auc_searchlight.join(
    behav_baseline_df,
    on=["subject", "electrode_idx", "phoneme_pair", "word_end", "fold"],
    how="inner",
).with_columns(
    (pl.col("behav_roc_auc") - pl.col("behav_roc_auc_baseline")).alias(
        "behav_roc_auc_improvement"
    )
)

# %% [markdown]
# ## Plot DataFrames at phonetic windows (identical to prepare_neurometrics)

# %%
plot_phon_phon_keys = phon_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
)
plot_phon_phon_df = plot_phon_phon_keys.join(
    phon_pred_df,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="left",
).join(all_md, on=["subject", "epoch_idx", "phoneme_pair"], how="left")

# %%
plot_phon_behav_keys = phon_peaks_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "smin", "smax"]
)
plot_phon_behav_df = plot_phon_behav_keys.join(
    behav_pred_df,
    on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
    how="left",
)

# %% [markdown]
# ## T-test behavioral window search
#
# For each phonetically-selective site × word_end, find the best behavioral contrast
# window using find_site_windows(). Only the ambiguous resampled steps (those where
# the subject made variable choices across repeats) are used for the behavioral contrast.

# %%
# Candidate sites: all phonetically-selective sites × all word_ends for that phoneme_pair
candidate_keys = phon_peaks_df.join(
    word_end_df.select(["phoneme_pair", "word_end"]).unique(),
    on="phoneme_pair",
    how="left",
)
L.info(f"Candidate sites for t-test search: {candidate_keys.height}")

# %%
# Bootstrap PaperData — only the fields used by find_site_windows and
# get_ambiguous_resampled_steps need to be populated.
_empty_pl = pl.DataFrame()
_bootstrap = PaperData(
    electrode_df=electrode_df,
    plot_phon_phon_df=plot_phon_phon_df,
    plot_behav_phon_df=_empty_pl,  # not needed for t-test search
    plot_behav_behav_df=_empty_pl,  # not needed for t-test search
    plot_phon_behav_df=plot_phon_behav_df,
    behav_roc_auc_searchlight_df=_empty_pl,
    phon_roc_auc_searchlight_df=phon_roc_auc_searchlight_df,
    all_md=all_md,
    word_end_df=word_end_df,
    epochs=epochs,
    phon_peaks_df=phon_peaks_df,
    behav_peaks_df=_empty_pl,
    behav_peaks_df_unfiltered=_empty_pl,
    behav_baseline_df=_empty_pl,
    zoomin_keys=candidate_keys,
    early_polarity=None,  # type: ignore
    late_polarity=None,  # type: ignore
)

# %%
# Per-(subject, phoneme_pair, word_end): which resampled steps show ambiguous behavior
ambiguous_resampled_steps = _bootstrap.get_ambiguous_resampled_steps(
    ambiguous_response_threshold=ambiguous_response_threshold
)
L.info(
    f"Items with ambiguous response behavior: "
    f"{sum(1 for v in ambiguous_resampled_steps.values() if v)}"
)

# %%
ttest_rows = []
for row in tqdm(
    candidate_keys.iter_rows(named=True),
    total=candidate_keys.height,
    desc="T-test window search",
):
    subject = row["subject"]
    electrode_idx = row["electrode_idx"]
    phoneme_pair = row["phoneme_pair"]
    word_end = row["word_end"]
    phon_window_smax = row["smax"]

    ambig = ambiguous_resampled_steps.get((subject, phoneme_pair, word_end), ())
    if not ambig:
        continue

    windows = find_site_windows(
        _bootstrap,
        subject,
        electrode_idx,
        phoneme_pair,
        word_end,
        # behavioral response is by construction something that follows the phonetic response
        search_smin=phon_window_smax,
        behavior_resampled_steps=(tuple(ambig),),
        window_size=15,
        window_stride=15,
    )

    bw = windows["behav"].get(tuple(ambig))
    if bw is None:
        continue

    ttest_rows.append(
        {
            "subject": subject,
            "electrode_idx": electrode_idx,
            "phoneme_pair": phoneme_pair,
            "word_end": word_end,
            "behav_t_stat": bw.t_stat,
            "behav_p_value": bw.p_value,
            "behav_n_group1": bw.n_group1,
            "behav_n_group2": bw.n_group2,
            "behav_smin": bw.smin,
            "behav_smax": bw.smax,
            "behav_tmin": bw.tmin,
            "behav_tmax": bw.tmax,
        }
    )

ttest_behav_df = pl.DataFrame(ttest_rows).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)
L.info(f"T-test results: {ttest_behav_df.height} sites with a valid behavioral window")

# %%
# Save full (pre-threshold) results immediately — expensive to recompute
ttest_behav_df.write_parquet(outdir / "ttest_behav_df.parquet")

# %%
# Filter by p-value threshold
behav_peaks_ttest_df = ttest_behav_df.filter(
    pl.col("behav_p_value") < behav_ttest_pvalue_threshold
)
L.info(
    f"Sites passing p < {behav_ttest_pvalue_threshold}: {behav_peaks_ttest_df.height}"
)

# %% [markdown]
# ## Snap t-test windows to nearest decoder searchlight window
#
# find_site_windows uses stride=15 (matching the behavioral decoder searchlight),
# so windows should usually align exactly. We log the offset distribution to verify.

# %%
behav_pred_unique_windows = behav_pred_df.select(
    ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
).unique()

snapped = (
    behav_peaks_ttest_df.join(
        behav_pred_unique_windows,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
        how="left",
    )
    .with_columns(
        (pl.col("smin") - pl.col("behav_smin")).abs().alias("smin_snap_offset")
    )
    .sort("smin_snap_offset")
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .first()
    .rename({"smin": "smin_snapped", "smax": "smax_snapped"})
)

snap_report = snapped.select(
    [
        "subject",
        "electrode_idx",
        "phoneme_pair",
        "word_end",
        "behav_smin",
        "smin_snapped",
        "smin_snap_offset",
    ]
)
L.info(
    f"Snap offset stats (samples):\n"
    f"  median = {snap_report['smin_snap_offset'].median()}\n"
    f"  max    = {snap_report['smin_snap_offset'].max()}\n"
    f"  n_exact = {(snap_report['smin_snap_offset'] == 0).sum()}"
)
snap_report.write_parquet(outdir / "ttest_snap_report.parquet")

# %% [markdown]
# ## Enrich behav_peaks_ttest_df with snapped smin/smax

# %%
# Add snapped smin/smax columns so downstream analysis can join to searchlights
# directly using smin/smax, while behav_smin/smax retain the raw t-test window.
behav_peaks_ttest_df = behav_peaks_ttest_df.join(
    snapped.select(
        [
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "word_end",
            "smin_snapped",
            "smax_snapped",
            "smin_snap_offset",
        ]
    ),
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    how="left",
).rename({"smin_snapped": "smin", "smax_snapped": "smax"})

# same for ttest_behav_df_full (all candidates, pre-threshold)
ttest_behav_df = ttest_behav_df.join(
    snapped.select(
        [
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "word_end",
            "smin_snapped",
            "smax_snapped",
            "smin_snap_offset",
        ]
    ),
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    how="left",
).rename({"smin_snapped": "smin", "smax_snapped": "smax"})

# %% [markdown]
# ## Build plot DataFrames at behavioral (t-test) windows

# %%
# Snapped behavioral window keys
behav_peak_keys_snapped = snapped.select(
    [
        "subject",
        "electrode_idx",
        "phoneme_pair",
        "word_end",
        "smin_snapped",
        "smax_snapped",
    ]
).rename({"smin_snapped": "smin", "smax_snapped": "smax"})

# %%
# plot_behav_phon_df: phoneme decoder predictions at behavioral t-test window
plot_behav_phon_df = (
    behav_peak_keys_snapped.rename({"word_end": "behav_word_end"})
    .join(
        phon_pred_df,
        on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
        how="left",
    )
    .join(
        all_md,
        on=["subject", "epoch_idx", "phoneme_pair"],
        how="left",
    )
    .filter(pl.col("word_end") == pl.col("behav_word_end"))
    .drop("behav_word_end")
)

# %%
# plot_behav_behav_df: behavioral decoder predictions at behavioral t-test window
plot_behav_behav_df = behav_peak_keys_snapped.join(
    behav_pred_df,
    on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
    how="left",
)

# %% [markdown]
# ## Zoomin keys and extract HGA windows

# %%
# zoomin_keys: sites with both phonetic AND behavioral (t-test) peaks
zoomin_keys = phon_peaks_df.join(
    behav_peaks_ttest_df, on=["subject", "electrode_idx", "phoneme_pair"]
).select(
    [
        "subject",
        "electrode_idx",
        "phoneme_pair",
        "word_end",
        "phon_roc_auc",
        "behav_t_stat",
        "behav_p_value",
    ]
)
L.info(f"zoomin_keys sites: {zoomin_keys.height}")

# %% [markdown]
# ## Bootstrap PaperData → compute HGA windows

# %%
# Upgrade bootstrap to include all available DataFrames before extract_hga_windows_df.
# early_polarity/late_polarity depend on hga_df — bootstrap with None, then replace.
_bootstrap2 = PaperData(
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
    phon_peaks_df=phon_peaks_df,
    behav_peaks_df=behav_peaks_ttest_df,
    behav_peaks_df_unfiltered=ttest_behav_df,
    behav_baseline_df=behav_baseline_df,
    zoomin_keys=zoomin_keys,
    early_polarity=None,  # type: ignore
    late_polarity=None,  # type: ignore
)

# %%
hga_df = extract_hga_windows_df(
    _bootstrap2,
    zoomin_keys=zoomin_keys,
    ambiguous_response_threshold=ambiguous_response_threshold,
)

# %% [markdown]
# ## Compute polarities and reg_df

# %%
# early_polarity: direction of acoustic category effect in the early window.
hga_df_unambig = hga_df[hga_df.resampled.isin([1.0, 6.0]) & hga_df.follows_acoustics]
early_polarity = (
    hga_df_unambig.groupby(
        ["subject", "electrode_idx", "phoneme_pair", "word_end", "decoder_target"]
    )
    .hga_early.mean()
    .reset_index()
    .set_index("decoder_target")
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .apply(lambda xs: np.sign(xs.loc[1] - xs.loc[0]))  # type: ignore[union-attr]
    .rename(columns={"hga_early": "early_polarity"})
)

# %%
# late_polarity: direction of behavioral choice effect in the late window.
hga_df_ambig = hga_df[
    hga_df.apply(
        lambda xs: (
            xs.behav_steps_chosen != "None"
            and str(int(xs.resampled)) in xs.behav_steps_chosen
        ),
        axis=1,
    )
]
late_polarity = (
    hga_df_ambig.groupby(
        [
            "subject",
            "electrode_idx",
            "phoneme_pair",
            "word_end",
            "behavior_dummy_forced",
        ]
    )
    .hga_late.mean()
    .reset_index()
    .set_index("behavior_dummy_forced")
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .apply(lambda xs: np.sign(xs.loc[1] - xs.loc[0]))  # type: ignore[union-attr]
    .rename(columns={"hga_late": "late_polarity"})
)

# %%
reg_df = pd.merge(
    hga_df,
    pd.merge(
        early_polarity.reset_index(),
        late_polarity.reset_index(),
        on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
    ),
    on=["subject", "electrode_idx", "phoneme_pair", "word_end"],
)
reg_df["hga_early_signed"] = reg_df["hga_early"] * reg_df["early_polarity"]
reg_df["hga_late_signed"] = reg_df["hga_late"] * reg_df["late_polarity"]
reg_df["is_ambiguous"] = reg_df.apply(
    lambda xs: (
        str(int(xs.resampled)) in xs.behav_steps_chosen
        if xs.behav_steps_chosen is not None
        else np.nan
    ),
    axis=1,
)

# %% [markdown]
# ## Construct final PaperData and save outputs

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
    phon_peaks_df=phon_peaks_df,
    behav_peaks_df=behav_peaks_ttest_df,
    behav_peaks_df_unfiltered=ttest_behav_df,
    behav_baseline_df=behav_baseline_df,
    zoomin_keys=zoomin_keys,
    early_polarity=early_polarity,
    late_polarity=late_polarity,
    hga_df=hga_df,
    reg_df=reg_df,
)

# %%
# Save Polars DataFrames as parquet
paper_data.electrode_df.write_parquet(outdir / "electrode_df.parquet")
paper_data.plot_phon_phon_df.write_parquet(outdir / "plot_phon_phon_df.parquet")
paper_data.plot_behav_phon_df.write_parquet(outdir / "plot_behav_phon_df.parquet")
paper_data.plot_behav_behav_df.write_parquet(outdir / "plot_behav_behav_df.parquet")
paper_data.plot_phon_behav_df.write_parquet(outdir / "plot_phon_behav_df.parquet")
paper_data.behav_roc_auc_searchlight_df.write_parquet(
    outdir / "behav_roc_auc_searchlight_df.parquet"
)
paper_data.phon_roc_auc_searchlight_df.write_parquet(
    outdir / "phon_roc_auc_searchlight_df.parquet"
)
paper_data.all_md.write_parquet(outdir / "all_md.parquet")
paper_data.word_end_df.write_parquet(outdir / "word_end_df.parquet")
paper_data.phon_peaks_df.write_parquet(outdir / "phon_peaks_df.parquet")
paper_data.behav_peaks_df.write_parquet(outdir / "behav_peaks_ttest_df.parquet")
paper_data.behav_peaks_df_unfiltered.write_parquet(
    outdir / "ttest_behav_df_full.parquet"
)
paper_data.behav_baseline_df.write_parquet(outdir / "behav_baseline_df.parquet")
paper_data.zoomin_keys.write_parquet(outdir / "zoomin_keys.parquet")

# Save pandas DataFrames (multi-indexed) as parquet via reset_index
paper_data.early_polarity.reset_index().to_parquet(outdir / "early_polarity.parquet")
paper_data.late_polarity.reset_index().to_parquet(outdir / "late_polarity.parquet")
paper_data.hga_df.to_parquet(outdir / "hga_df.parquet")
paper_data.reg_df.to_parquet(outdir / "reg_df.parquet")

L.success(f"Saved all PaperData fields to {outdir}")
