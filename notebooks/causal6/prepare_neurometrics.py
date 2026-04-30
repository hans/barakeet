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
# # prepare_neurometrics — causal6 edition
#
# Assembles the PaperData parquet bundle that feeds A_neurometrics. Adapted
# from notebooks/causal5/prepare_neurometrics.py with these causal6-specific
# differences:
#
#   - Acoustic and behavior decoders persist coefficients as parquet (one row
#     per (decoder key, fold) with `coef` / `mean` / `scale` list columns).
#     Causal5 used joblib pipeline blobs. We surface the parquet weights via
#     decoder_weights_*.parquet so A_neurometrics can do cross-window transfer
#     (sections #5/#6) without retraining.
#
#   - Behavior decoder is split into two rules: behavior_decoding_single_electrode
#     (full, with control regressors) and behavior_decoding_single_electrode_hga_only
#     (HGA only). HGA-only is the canonical "behaviorally selective" criterion.
#     The control-only baseline lives as `model="baseline"` rows inside the
#     full decoder's scores.parquet.
#
#   - Peak parquets come in multiple flavors per decoder. We load and emit each
#     flavor as a separate parquet, then pick `primary_peak_flavor` (default
#     tstat_maxstat) for the canonical plot_*_df / zoomin_keys / hga_df set.

# %%
import re
import shutil
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
    POD_dict,
    WORD_END_TO_PHONEME_PAIR,
)
from src.viz_paper import (
    PaperData,
    extract_hga_windows_df,
    phoneme_pair_enum,
    pl_roc_auc,
    subject_enum,
    word_end_enum,
)

# %% tags=["parameters"]
# Lists default to globs over the repo so the notebook is runnable interactively;
# Snakemake passes explicit lists.
all_epochs = sorted(str(p) for p in Path("outputs/epochs_preprocessed").glob("*_epo.fif"))
electrode_paths = sorted(
    str(p) for p in Path("outputs/causal5/find_speech_responsive/").glob("*_results.csv")
)

# Acoustic decoder
acoustic_scores = sorted(
    str(p) for p in Path("outputs/causal6/acoustic_decoding_single_electrode").rglob(
        "*/scores.parquet"
    )
)
acoustic_predictions = sorted(
    str(p) for p in Path("outputs/causal6/acoustic_decoding_single_electrode").rglob(
        "*/predictions.parquet"
    )
)
acoustic_coefficients = sorted(
    str(p) for p in Path("outputs/causal6/acoustic_decoding_single_electrode").rglob(
        "*/coefficients.parquet"
    )
)
phon_peaks_foldmean_maxstat = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
phon_peaks_tstat_maxstat = (
    "outputs/causal6/acoustic_decoding_peaks/phon_peaks_tstat_maxstat_all.parquet"
)
phon_roc_auc_searchlight_paths = sorted(
    str(p) for p in Path("outputs/causal6/acoustic_decoding_peaks").rglob(
        "*/phon_roc_auc_searchlight.parquet"
    )
)

# Behavior with-control (full) decoder
behav_full_scores = sorted(
    str(p) for p in Path("outputs/causal6/behavior_decoding_single_electrode").rglob(
        "*/scores.parquet"
    )
)
behav_full_predictions = sorted(
    str(p) for p in Path("outputs/causal6/behavior_decoding_single_electrode").rglob(
        "*/predictions.parquet"
    )
)
behav_full_coefficients = sorted(
    str(p) for p in Path("outputs/causal6/behavior_decoding_single_electrode").rglob(
        "*/coefficients.parquet"
    )
)
behav_full_peaks_foldmean_maxstat = (
    "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_all.parquet"
)
behav_full_peaks_tstat_maxstat = (
    "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_tstat_maxstat_all.parquet"
)
behav_full_peaks_tstat_tfce = (
    "outputs/causal6/behavior_decoding_single_electrode_summarize/peak_summary_tstat_tfce_all.parquet"
)

# Behavior HGA-only decoder
behav_hga_only_scores = sorted(
    str(p) for p in Path("outputs/causal6/behavior_decoding_single_electrode_hga_only").rglob(
        "*/scores.parquet"
    )
)
behav_hga_only_predictions = sorted(
    str(p) for p in Path("outputs/causal6/behavior_decoding_single_electrode_hga_only").rglob(
        "*/predictions.parquet"
    )
)
behav_hga_only_coefficients = sorted(
    str(p) for p in Path("outputs/causal6/behavior_decoding_single_electrode_hga_only").rglob(
        "*/coefficients.parquet"
    )
)
behav_hga_only_peaks_foldmean_maxstat = (
    "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_all.parquet"
)
behav_hga_only_peaks_tstat_maxstat = (
    "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_maxstat_all.parquet"
)
behav_hga_only_peaks_tstat_tfce = (
    "outputs/causal6/behavior_decoding_single_electrode_hga_only_summarize/peak_summary_tstat_tfce_all.parquet"
)

# Constants and thresholds
epoch_tmin = -0.4
epoch_sfreq = 100
ambiguous_response_threshold = 2
primary_peak_flavor = "tstat_maxstat"  # one of {foldmean_maxstat, tstat_maxstat, tstat_tfce}

outdir = "outputs/causal6/prepare_neurometrics"

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Stale-output filter
#
# Local outputs may pre-date the adaptive-K null pipeline (e.g. when running
# this notebook before all subjects have finished re-running). For each
# subject × decoder we check three markers of the current pipeline; if any
# are missing the (subject, decoder) is **dropped** from this run with a
# warning. Other (subject, decoder) pairs continue.
#
#   1. Each null run must have an `escalation_log.parquet` next to its
#      `null_scores.parquet`, and that log must contain at least one row
#      with `K == 9000` (the second adaptive escalation stage). Pre-adaptive
#      runs only emit `null_scores.parquet`.
#
#   2. Each summarize run must emit the t-stat flavors as separate parquets
#      (`peak_summary_tstat_maxstat.parquet`, `peak_summary_tstat_tfce.parquet`)
#      alongside the v1 `peak_summary.parquet`. Pre-flavor runs only emit
#      the v1 file. (Acoustic peaks don't ship TFCE — only t-stat maxstat
#      is required there.)
#
#   3. Each per-subject decoder dir must have a `coefficients.parquet`
#      (cross-window transfer in A_neurometrics needs it). Pre-coefficients
#      runs only emit scores + predictions.
#
# Downstream code filters per-subject path lists and the cross-subject
# aggregator parquets to subjects that pass these checks for the relevant
# decoder. Different decoders can have different fresh-subject sets.

# %%
def _subject_from_path(p) -> str:
    m = re.search(r"(EC\d+)", str(p))
    if not m:
        raise ValueError(f"could not parse subject from {p}")
    return m.group(1)


def _null_dir_for(scores_path: Path) -> Path:
    subject = scores_path.parent.name
    decoder = scores_path.parent.parent.name
    null_dir_name_map = {
        "acoustic_decoding_single_electrode": "acoustic_decoding_null",
        "behavior_decoding_single_electrode": "behavior_decoding_single_electrode_null",
        "behavior_decoding_single_electrode_hga_only":
            "behavior_decoding_single_electrode_hga_only_null",
    }
    return scores_path.parent.parent.parent / null_dir_name_map[decoder] / subject


def _summarize_dir_for(decoder_dir: Path, summarize_rule: str) -> Path:
    return decoder_dir.parent.parent / summarize_rule / decoder_dir.name


def _escalation_ok(null_dir: Path) -> tuple[bool, str]:
    log_path = null_dir / "escalation_log.parquet"
    if not log_path.exists():
        return False, f"missing {log_path.name} (pre-adaptive-K)"
    log = pl.read_parquet(log_path)
    if "K" not in log.columns:
        return False, f"escalation_log has no K column (cols={log.columns})"
    if 9000 not in log["K"].to_list():
        # Soft pass: didn't escalate, but log structure is current. Note it.
        return True, f"never escalated to K=9000 (Ks seen: {sorted(set(log['K'].to_list()))})"
    return True, "OK"


def _tfce_flavors_ok(summarize_dir: Path) -> tuple[bool, str]:
    needed = ["peak_summary_tstat_maxstat.parquet", "peak_summary_tstat_tfce.parquet"]
    missing = [n for n in needed if not (summarize_dir / n).exists()]
    if missing:
        return False, f"summarize_dir missing {missing}"
    return True, "OK"


def _coefficients_ok(per_subject_dir: Path) -> tuple[bool, str]:
    if not (per_subject_dir / "coefficients.parquet").exists():
        return False, "missing coefficients.parquet"
    return True, "OK"


def _filter_fresh_subjects(decoder_dirs, *, decoder_label, summarize_rule, require_tfce=True):
    """Return the set of subjects whose outputs pass all freshness checks for
    this decoder. Warn (but do not raise) on subjects that fail."""
    fresh = set()
    for d in decoder_dirs:
        subj = d.name
        problems = []
        ok, msg = _coefficients_ok(d)
        if not ok:
            problems.append(f"coefficients: {msg}")

        # not worried about escalation; some runs ran with non-adaptive K and that's fine
        # ok, msg = _escalation_ok(_null_dir_for(d / "scores.parquet"))
        # if not ok:
        #     problems.append(f"null: {msg}")
        # elif msg != "OK":
        #     L.warning(f"[stale-filter] {decoder_label}/{subj}: null {msg}")
        
        if summarize_rule is not None and require_tfce:
            ok, msg = _tfce_flavors_ok(_summarize_dir_for(d, summarize_rule))
            if not ok:
                problems.append(f"summarize: {msg}")
        else:
            # Acoustic: just require per-subject phon_peaks_tstat_maxstat.parquet
            peaks_dir = d.parent.parent / "acoustic_decoding_peaks" / subj
            if not (peaks_dir / "phon_peaks_tstat_maxstat.parquet").exists():
                problems.append("peaks: missing phon_peaks_tstat_maxstat.parquet")
        if problems:
            L.warning(
                f"[stale-filter] {decoder_label}/{subj}: EXCLUDED — "
                + "; ".join(problems)
            )
        else:
            fresh.add(subj)
    return fresh


_acoustic_decoder_dirs = sorted({Path(p).parent for p in acoustic_scores})
_behav_full_decoder_dirs = sorted({Path(p).parent for p in behav_full_scores})
_behav_hga_only_decoder_dirs = sorted({Path(p).parent for p in behav_hga_only_scores})

fresh_subjects_acoustic = _filter_fresh_subjects(
    _acoustic_decoder_dirs,
    decoder_label="acoustic",
    summarize_rule=None,
    require_tfce=False,
)
fresh_subjects_behav_full = _filter_fresh_subjects(
    _behav_full_decoder_dirs,
    decoder_label="behav_full",
    summarize_rule="behavior_decoding_single_electrode_summarize",
)
fresh_subjects_behav_hga_only = _filter_fresh_subjects(
    _behav_hga_only_decoder_dirs,
    decoder_label="behav_hga_only",
    summarize_rule="behavior_decoding_single_electrode_hga_only_summarize",
)


def _keep_fresh(paths, fresh_subjects):
    return [p for p in paths if _subject_from_path(p) in fresh_subjects]


# Restrict per-subject path lists to fresh subjects per decoder.
acoustic_scores                = _keep_fresh(acoustic_scores,                fresh_subjects_acoustic)
acoustic_predictions           = _keep_fresh(acoustic_predictions,           fresh_subjects_acoustic)
acoustic_coefficients          = _keep_fresh(acoustic_coefficients,          fresh_subjects_acoustic)
phon_roc_auc_searchlight_paths = _keep_fresh(phon_roc_auc_searchlight_paths, fresh_subjects_acoustic)

behav_full_scores       = _keep_fresh(behav_full_scores,       fresh_subjects_behav_full)
behav_full_predictions  = _keep_fresh(behav_full_predictions,  fresh_subjects_behav_full)
behav_full_coefficients = _keep_fresh(behav_full_coefficients, fresh_subjects_behav_full)

behav_hga_only_scores       = _keep_fresh(behav_hga_only_scores,       fresh_subjects_behav_hga_only)
behav_hga_only_predictions  = _keep_fresh(behav_hga_only_predictions,  fresh_subjects_behav_hga_only)
behav_hga_only_coefficients = _keep_fresh(behav_hga_only_coefficients, fresh_subjects_behav_hga_only)

L.info(
    f"[stale-filter] kept: acoustic={sorted(fresh_subjects_acoustic)}, "
    f"behav_full={sorted(fresh_subjects_behav_full)}, "
    f"behav_hga_only={sorted(fresh_subjects_behav_hga_only)}"
)


def _filter_aggregator_rows(df: pl.DataFrame, fresh_subjects, *, label: str) -> pl.DataFrame:
    """Drop rows whose subject is not in `fresh_subjects`. Aggregator parquets
    (phon_peaks_all, peak_summary_*_all) may include rows from subjects whose
    per-subject outputs are now considered stale; this keeps the cross-subject
    set consistent.

    NOTE: the `q_value` / `significant` columns in the aggregator were
    computed by BH-FDR over the full set of subjects at aggregation time.
    After filtering by fresh subjects here, the FDR threshold is no longer
    guaranteed to be valid for the remaining set — it is conservative (more
    rejections survive than would on the smaller set). Re-run *_aggregate
    after upstream is fully fresh to get canonical FDR.
    """
    n_before = df.height
    df = df.filter(pl.col("subject").is_in(list(fresh_subjects)))
    L.info(f"[stale-filter] {label}: {df.height}/{n_before} rows kept")
    return df

# %% [markdown]
# ## Helpers

# %%
def _read_subject_path(p: str) -> str:
    """Pull the subject ID (e.g. EC248) from a per-subject parquet path."""
    m = re.search(r"(EC\d+)", str(p))
    if not m:
        raise ValueError(f"could not parse subject from {p}")
    return m.group(1)


def _concat_per_subject(paths, *, label: str) -> pl.DataFrame:
    """Read a list of per-subject parquets and concat with subject column injected
    if missing (causal6 parquets already include `subject`, but be defensive).
    """
    frames = []
    for p in paths:
        df = pl.read_parquet(p)
        if "subject" not in df.columns:
            df = df.with_columns(pl.lit(_read_subject_path(p)).alias("subject"))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no {label} parquets found")
    return pl.concat(frames, how="vertical_relaxed")


# %% [markdown]
# ## Electrode metadata

# %%
electrode_df = (
    pl.concat([pl.read_csv(p) for p in electrode_paths], how="vertical_relaxed")
    .with_columns(pl.col("subject").cast(subject_enum))
)

# %% [markdown]
# ## Epochs + per-trial metadata

# %%
epochs = {}
for p in tqdm(all_epochs, desc="loading epochs"):
    m = re.search(r"(EC\d+)_epo", str(p))
    assert m is not None, f"could not parse subject from {p}"
    subj = m.group(1)
    ep = mne.read_epochs(str(p), preload=False, verbose="WARNING")
    ep.metadata = add_metadata_features(ep.metadata.copy())
    epochs[subj] = ep

# %%
all_md_pd = pd.concat(
    {subj: ep.metadata.assign(subject=subj) for subj, ep in epochs.items()},
    ignore_index=True,
)
all_md_pd["TDT Block"] = all_md_pd["TDT Block"].astype(str)
all_md = pl.from_pandas(all_md_pd).with_columns(
    pl.col("subject").cast(subject_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
    pl.col("word_end").cast(word_end_enum),
)

# %% [markdown]
# ## word_end timing lookup

# %%
word_end_records = []
for word_end, offset_s in OFFSET_DICT.items():
    if word_end not in WORD_END_TO_PHONEME_PAIR:
        continue
    pp = WORD_END_TO_PHONEME_PAIR[word_end]
    pod_s = POD_dict[pp]
    word_end_records.append({
        "word_end": word_end,
        "phoneme_pair": pp,
        "word_end_offset_s": offset_s,
        "word_end_offset_sample": int((offset_s - epoch_tmin) * epoch_sfreq),
        "pod_s": pod_s,
        "pod_sample": int((pod_s - epoch_tmin) * epoch_sfreq),
    })
word_end_df = pl.DataFrame(word_end_records).with_columns(
    pl.col("word_end").cast(word_end_enum),
    pl.col("phoneme_pair").cast(phoneme_pair_enum),
)

# %% [markdown]
# ## Acoustic decoder: predictions, scores, coefficients
#
# The causal6 acoustic decoder runs only the `categorical_acoustic_cue` target
# (no `measure` column on the predictions parquet). So we just concat across
# subjects.

# %%
phon_pred_df = (
    _concat_per_subject(acoustic_predictions, label="acoustic predictions")
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
        pl.col("decoder_target").cast(pl.Int8),
    )
)

# %%
phon_scores_df = (
    _concat_per_subject(acoustic_scores, label="acoustic scores")
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
    )
)

# %%
phon_coefs_df = (
    _concat_per_subject(acoustic_coefficients, label="acoustic coefficients")
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
    )
)

# %% [markdown]
# ## Acoustic peaks (two flavors)

# %%
phon_peaks_foldmean_maxstat_df = _filter_aggregator_rows(
    pl.read_parquet(phon_peaks_foldmean_maxstat).with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
    ),
    fresh_subjects_acoustic,
    label="phon_peaks_foldmean_maxstat",
)

phon_peaks_tstat_maxstat_df = _filter_aggregator_rows(
    pl.read_parquet(phon_peaks_tstat_maxstat).with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
    ),
    fresh_subjects_acoustic,
    label="phon_peaks_tstat_maxstat",
)

# %% [markdown]
# ## Acoustic ROC-AUC searchlight (per-subject, per-fold)
#
# Used by figure #3 for cross-window decoding lookup.

# %%
phon_roc_auc_searchlight_df = (
    _concat_per_subject(phon_roc_auc_searchlight_paths, label="phon roc_auc searchlight")
    .with_columns(
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
    )
)

# %% [markdown]
# ## Behavior decoders — predictions / scores / coefficients
#
# Two decoders are loaded:
#   - behavior_full: includes control regressors. Its scores.parquet has
#     `model` ∈ {"full", "baseline"}; baseline rows are the control-only
#     reference fit and are independent of (electrode, smin, smax)
#     (sentinel `electrode_idx = -1`, `smin = -1`, `smax = -1`).
#   - behavior_hga_only: HGA only, no `model` column.

# %%
def _load_behav_bundle(scores_paths, predictions_paths, coefficients_paths, label):
    scores = _concat_per_subject(scores_paths, label=f"{label} scores")
    preds = _concat_per_subject(predictions_paths, label=f"{label} predictions")
    coefs = _concat_per_subject(coefficients_paths, label=f"{label} coefficients")
    cast_cols = [
        pl.col("subject").cast(subject_enum),
        pl.col("phoneme_pair").cast(phoneme_pair_enum),
        pl.col("word_end").cast(word_end_enum),
    ]
    return (
        scores.with_columns(cast_cols),
        preds.with_columns(cast_cols + [pl.col("decoder_target").cast(pl.Int8)]),
        coefs.with_columns(cast_cols),
    )

behav_full_scores_df, behav_full_pred_df, behav_full_coefs_df = _load_behav_bundle(
    behav_full_scores, behav_full_predictions, behav_full_coefficients, "behav_full",
)
behav_hga_only_scores_df, behav_hga_only_pred_df, behav_hga_only_coefs_df = _load_behav_bundle(
    behav_hga_only_scores, behav_hga_only_predictions, behav_hga_only_coefficients, "behav_hga_only",
)

# %% [markdown]
# ## Behavior peak parquets — three flavors per decoder

# %%
def _load_peak_parquet(path, fresh_subjects, label):
    return _filter_aggregator_rows(
        pl.read_parquet(path).with_columns(
            pl.col("subject").cast(subject_enum),
            pl.col("phoneme_pair").cast(phoneme_pair_enum),
            pl.col("word_end").cast(word_end_enum),
        ),
        fresh_subjects,
        label=label,
    )

behav_hga_only_peaks_foldmean_maxstat_df = _load_peak_parquet(
    behav_hga_only_peaks_foldmean_maxstat, fresh_subjects_behav_hga_only,
    "behav_hga_only_peaks_foldmean_maxstat",
)
behav_hga_only_peaks_tstat_maxstat_df = _load_peak_parquet(
    behav_hga_only_peaks_tstat_maxstat, fresh_subjects_behav_hga_only,
    "behav_hga_only_peaks_tstat_maxstat",
)
behav_hga_only_peaks_tstat_tfce_df = _load_peak_parquet(
    behav_hga_only_peaks_tstat_tfce, fresh_subjects_behav_hga_only,
    "behav_hga_only_peaks_tstat_tfce",
)

behav_full_peaks_foldmean_maxstat_df = _load_peak_parquet(
    behav_full_peaks_foldmean_maxstat, fresh_subjects_behav_full,
    "behav_full_peaks_foldmean_maxstat",
)
behav_full_peaks_tstat_maxstat_df = _load_peak_parquet(
    behav_full_peaks_tstat_maxstat, fresh_subjects_behav_full,
    "behav_full_peaks_tstat_maxstat",
)
behav_full_peaks_tstat_tfce_df = _load_peak_parquet(
    behav_full_peaks_tstat_tfce, fresh_subjects_behav_full,
    "behav_full_peaks_tstat_tfce",
)

# %% [markdown]
# ## Behavior baseline (control-only) ROC-AUC
#
# Pulled from the `model="baseline"` rows of behavior_full's scores.parquet.
# These are control-features-only LR test ROC-AUCs, one row per
# (subject, phoneme_pair, word_end, fold).

# %%
behav_baseline_df = (
    behav_full_scores_df
    .filter(pl.col("model") == "baseline")
    .select([
        "subject", "phoneme_pair", "word_end", "fold",
        pl.col("test_roc_auc").alias("baseline_roc_auc"),
    ])
)

# %% [markdown]
# ## Behavior searchlight ROC-AUC + improvement vs baseline (figure #4 input)

# %%
behav_full_searchlight = (
    behav_full_scores_df
    .filter(pl.col("model") == "full")
    .select([
        "subject", "phoneme_pair", "word_end", "electrode_idx", "smin", "smax", "fold",
        pl.col("test_roc_auc").alias("full_roc_auc"),
    ])
)

behav_hga_only_searchlight = behav_hga_only_scores_df.select([
    "subject", "phoneme_pair", "word_end", "electrode_idx", "smin", "smax", "fold",
    pl.col("test_roc_auc").alias("hga_only_roc_auc"),
])

behav_roc_auc_searchlight_df = (
    behav_full_searchlight
    .join(
        behav_hga_only_searchlight,
        on=["subject", "phoneme_pair", "word_end", "electrode_idx", "smin", "smax", "fold"],
        how="full",
        coalesce=True,
    )
    .join(
        behav_baseline_df,
        on=["subject", "phoneme_pair", "word_end", "fold"],
        how="left",
    )
    .with_columns([
        (pl.col("full_roc_auc") - pl.col("baseline_roc_auc")).alias("full_improvement"),
        (pl.col("hga_only_roc_auc") - pl.col("baseline_roc_auc")).alias("hga_only_improvement"),
    ])
)

# %% [markdown]
# ## Pick canonical peak set
#
# `primary_peak_flavor` selects which behavior peak parquet defines the
# "behaviorally selective" criterion downstream. Acoustic peaks always use
# the foldmean_maxstat flavor as canonical (legacy parity); the t-stat flavor
# is loaded for figures that compare flavors.

# %%
phon_peaks_df = phon_peaks_foldmean_maxstat_df  # canonical acoustic
behav_peaks_lookup = {
    "foldmean_maxstat": behav_hga_only_peaks_foldmean_maxstat_df,
    "tstat_maxstat":    behav_hga_only_peaks_tstat_maxstat_df,
    "tstat_tfce":       behav_hga_only_peaks_tstat_tfce_df,
}
if primary_peak_flavor not in behav_peaks_lookup:
    raise ValueError(
        f"primary_peak_flavor={primary_peak_flavor!r} must be one of "
        f"{sorted(behav_peaks_lookup)}"
    )
behav_peaks_df_unfiltered = behav_peaks_lookup[primary_peak_flavor]

# Treat "behaviorally selective" as FDR-significant peaks, dropping rows that
# lack the column (legacy). FDR significance is added by significance_aggregate.
if "significant" in behav_peaks_df_unfiltered.columns:
    behav_peaks_df = behav_peaks_df_unfiltered.filter(pl.col("significant"))
else:
    L.warning(
        "behav peaks parquet has no `significant` column — falling back to "
        "p_value < 0.05 (uncorrected); regenerate with significance_aggregate "
        "for FDR-corrected output."
    )
    behav_peaks_df = behav_peaks_df_unfiltered.filter(pl.col("p_value") < 0.05)

# %% [markdown]
# ## plot_*_df — per-trial decoder predictions at peak windows
#
# Built on the canonical peak set (selected flavor + HGA-only). Mirrors causal5.

# %%
phon_peak_keys = phon_peaks_df.select([
    "subject", "electrode_idx", "phoneme_pair", "smin", "smax",
])
behav_peak_keys = behav_peaks_df.select([
    "subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax",
])

# %%
plot_phon_phon_df = (
    phon_peak_keys
    .join(
        phon_pred_df,
        on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
        how="left",
    )
    .join(
        all_md.select(["subject", "epoch_idx", "phoneme_pair", "word_end",
                       "resampled", "behavior_dummy_forced",
                       "categorical_acoustic_cue"]),
        on=["subject", "epoch_idx", "phoneme_pair"],
        how="left",
    )
)

plot_behav_phon_df = (
    behav_peak_keys
    .rename({"word_end": "behav_word_end", "smin": "smin_behav", "smax": "smax_behav"})
    .join(
        phon_pred_df.rename({"smin": "smin_phon", "smax": "smax_phon"}),
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="left",
    )
    .join(
        all_md.select(["subject", "epoch_idx", "phoneme_pair", "word_end",
                       "resampled", "behavior_dummy_forced",
                       "categorical_acoustic_cue"]),
        on=["subject", "epoch_idx", "phoneme_pair"],
        how="left",
    )
    .filter(pl.col("word_end") == pl.col("behav_word_end"))
)

plot_phon_behav_df = (
    phon_peak_keys
    .join(
        behav_hga_only_pred_df,
        on=["subject", "electrode_idx", "phoneme_pair", "smin", "smax"],
        how="left",
    )
)

plot_behav_behav_df = (
    behav_peak_keys
    .join(
        behav_hga_only_pred_df,
        on=["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"],
        how="left",
    )
)

# %% [markdown]
# ## zoomin_keys — sites with both acoustic + behavioral peaks

# %%
zoomin_keys = (
    phon_peaks_df.select(["subject", "electrode_idx", "phoneme_pair"])
    .unique()
    .join(
        behav_peaks_df.select(["subject", "electrode_idx", "phoneme_pair"]).unique(),
        on=["subject", "electrode_idx", "phoneme_pair"],
        how="inner",
    )
)

# %% [markdown]
# ## Bootstrap PaperData (early/late polarity filled in below)

# %%
_bootstrap = PaperData(
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
    behav_peaks_df=behav_peaks_df,
    behav_peaks_df_unfiltered=behav_peaks_df_unfiltered,
    zoomin_keys=zoomin_keys,
    early_polarity=None,
    late_polarity=None,
    hga_df=None,
    reg_df=None,
)

# %% [markdown]
# ## Extract HGA windows (the slow step)

# %%
hga_df = extract_hga_windows_df(
    _bootstrap,
    zoomin_keys=zoomin_keys,
    ambiguous_response_threshold=ambiguous_response_threshold,
    window_source="decoder",
)

# %% [markdown]
# ## Early polarity (acoustic tuning at acoustic-peak window)

# %%
_early = (
    hga_df[hga_df["resampled"].isin([1.0, 6.0])]
    [hga_df["follows_acoustics"] == True]  # noqa: E712
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end", "decoder_target"])
    ["hga_early"].mean()
    .unstack("decoder_target")
)
early_polarity = np.sign(_early[1] - _early[0]).rename("early_polarity").to_frame()

# %% [markdown]
# ## Late polarity (perceptual tuning at perceptual-peak window, ambiguous trials)

# %%
def _is_ambiguous_row(row):
    chosen = row["behav_steps_chosen"]
    if chosen is None or (isinstance(chosen, float) and np.isnan(chosen)):
        return False
    if not isinstance(chosen, (list, tuple, np.ndarray)):
        return False
    return int(row["resampled"]) in [int(x) for x in chosen]

ambig_mask = hga_df.apply(_is_ambiguous_row, axis=1)
_late = (
    hga_df[ambig_mask]
    .groupby(["subject", "electrode_idx", "phoneme_pair", "word_end", "behavior_dummy_forced"])
    ["hga_late"].mean()
    .unstack("behavior_dummy_forced")
)
late_polarity = np.sign(_late[1] - _late[0]).rename("late_polarity").to_frame()

# %% [markdown]
# ## reg_df — hga_df with polarity-corrected signed columns

# %%
reg_df = (
    hga_df
    .merge(early_polarity.reset_index(), on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="left")
    .merge(late_polarity.reset_index(),  on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="left")
    .assign(
        hga_early_signed=lambda d: d["hga_early"] * d["early_polarity"],
        hga_late_signed=lambda d:  d["hga_late"]  * d["late_polarity"],
        is_ambiguous=lambda d: ambig_mask.values,
    )
)

# %% [markdown]
# ## decoder_weights — surface coefficients.parquet for cross-window transfer
#
# A_neurometrics applies these as `sigmoid(((X - mean) / scale) @ coef + intercept)`
# to evaluate trained decoders at swapped windows (sections #5/#6). Causal6
# parquet columns: `coef`, `mean`, `scale` are List[Float32] of length d
# (per-feature). `intercept` is included if present in the upstream parquet.

# %%
decoder_weights_acoustic = phon_coefs_df  # already keyed by site×fold×window
decoder_weights_behav_hga_only = behav_hga_only_coefs_df

# %% [markdown]
# ## Save parquet bundle

# %%
def _write_pl(df: pl.DataFrame, name: str):
    p = outdir / name
    df.write_parquet(p)
    L.info(f"wrote {name}: {df.height} rows -> {p}")

def _write_pd(df: pd.DataFrame, name: str):
    p = outdir / name
    df.reset_index().to_parquet(p)
    L.info(f"wrote {name}: {len(df)} rows -> {p}")

_write_pl(electrode_df, "electrode_df.parquet")
_write_pl(all_md, "all_md.parquet")
_write_pl(word_end_df, "word_end_df.parquet")

_write_pl(phon_peaks_foldmean_maxstat_df, "phon_peaks_foldmean_maxstat.parquet")
_write_pl(phon_peaks_tstat_maxstat_df, "phon_peaks_tstat_maxstat.parquet")

_write_pl(behav_hga_only_peaks_foldmean_maxstat_df, "behav_hga_only_peaks_foldmean_maxstat.parquet")
_write_pl(behav_hga_only_peaks_tstat_maxstat_df, "behav_hga_only_peaks_tstat_maxstat.parquet")
_write_pl(behav_hga_only_peaks_tstat_tfce_df, "behav_hga_only_peaks_tstat_tfce.parquet")

_write_pl(behav_full_peaks_foldmean_maxstat_df, "behav_full_peaks_foldmean_maxstat.parquet")
_write_pl(behav_full_peaks_tstat_maxstat_df, "behav_full_peaks_tstat_maxstat.parquet")
_write_pl(behav_full_peaks_tstat_tfce_df, "behav_full_peaks_tstat_tfce.parquet")

_write_pl(phon_roc_auc_searchlight_df, "phon_roc_auc_searchlight_df.parquet")
_write_pl(behav_roc_auc_searchlight_df, "behav_roc_auc_searchlight_df.parquet")
_write_pl(behav_baseline_df, "behav_baseline_df.parquet")

_write_pl(plot_phon_phon_df,  "plot_phon_phon_df.parquet")
_write_pl(plot_behav_phon_df, "plot_behav_phon_df.parquet")
_write_pl(plot_phon_behav_df, "plot_phon_behav_df.parquet")
_write_pl(plot_behav_behav_df, "plot_behav_behav_df.parquet")
_write_pl(zoomin_keys, "zoomin_keys.parquet")

# pandas dataframes need reset_index before parquet
hga_df.reset_index(drop=True).to_parquet(outdir / "hga_df.parquet")
_write_pd(early_polarity, "early_polarity.parquet")
_write_pd(late_polarity, "late_polarity.parquet")
reg_df.reset_index(drop=True).to_parquet(outdir / "reg_df.parquet")

_write_pl(decoder_weights_acoustic,        "decoder_weights_acoustic.parquet")
_write_pl(decoder_weights_behav_hga_only,  "decoder_weights_behav_hga_only.parquet")

L.info(f"prepare_neurometrics complete: bundle written to {outdir}")
