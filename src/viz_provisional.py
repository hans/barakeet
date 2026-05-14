"""Provisional-results helpers shared between `notebooks/causal6/view_provisional_results.py`
and `notebooks/causal6/A_neurometrics_provisional.py`.

These helpers operate on raw `outputs/causal6/*_decoding_single_electrode*/scores.parquet`
files and use AUC thresholding instead of FDR/permutation significance. They do
NOT require the `prepare_neurometrics` parquet bundle.

Window-filtering rules match the canonical source in
`src.models.causal6_aggregates` exactly:

  - acoustic: ``smin >= ac_peak_search_smin AND smax <= ac_peak_search_smax``
    (defaults from ``config["analysis"]["decoding"]``: 50 / 75).
  - behavior HGA-only / behavior full: ``smin >= 0 AND smax <= offset_samples[word_end]
    AND smax <= peak_search_smax`` (defaults: 0 / 290, behav_peak_post_offset_s=0.2).
  - Ganong: ``smin >= POD_samples[phoneme_pair] AND smax <= peak_search_smax``.

Nothing in this module reads `config.yaml`; callers pass the constants in.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mne
import numpy as np
import polars as pl

from src.data import (
    add_metadata_features,
    get_ambiguous_resampled_steps,
    get_electrode_df,
)
from src.models.causal6_aggregates import (
    SITE_KEYS_ACOUSTIC,
    SITE_KEYS_BEHAVIOR_HGA_ONLY,
    SITE_KEYS_GANONG_HGA_ONLY,
    _behavior_offset_samples,
    _ganong_pod_samples,
)
from src.stimuli import OFFSET_DICT
from src.viz_paper import add_textgrid

EPOCH_TMIN: float = -0.4
EPOCH_SFREQ: float = 100.0

SITE_KEYS_BEHAV = SITE_KEYS_BEHAVIOR_HGA_ONLY
WINDOW_KEYS_BEHAV = SITE_KEYS_BEHAV + ["smin", "smax"]
_AC_SITE_KEYS = SITE_KEYS_ACOUSTIC
_AC_WINDOW_KEYS = _AC_SITE_KEYS + ["smin", "smax"]
_GANONG_SITE_KEYS = SITE_KEYS_GANONG_HGA_ONLY
_GANONG_WINDOW_KEYS = _GANONG_SITE_KEYS + ["smin", "smax"]
_GANONG_FOLD_KEYS = ["subject", "phoneme_pair", "fold"]
_BEH_FULL_FOLD_KEYS = ["subject", "phoneme_pair", "word_end", "fold"]


def smin_to_ms(s, epoch_tmin: float = EPOCH_TMIN, epoch_sfreq: float = EPOCH_SFREQ) -> np.ndarray:
    """Convert a sample index (or array) into milliseconds post word onset."""
    return (np.asarray(s) / epoch_sfreq + epoch_tmin) * 1000


def _filter_behavior_window_expr(
    offset_samples_lookup: dict[str, int],
    peak_search_smin: int,
    peak_search_smax: int,
) -> tuple[pl.Expr, pl.Expr]:
    """Returns (smax_limit_expr, filter_expr). Apply as:
       df.with_columns(smax_limit_expr.alias("_smax_limit")).filter(filter_expr).drop("_smax_limit")."""
    smax_limit = pl.col("word_end").replace_strict(offset_samples_lookup, default=None)
    filter_expr = (
        (pl.col("smin") >= peak_search_smin)
        & (pl.col("smax") <= pl.col("_smax_limit"))
        & (pl.col("smax") <= peak_search_smax)
    )
    return smax_limit, filter_expr


def _electrode_pl(subject: str) -> pl.DataFrame:
    """Pull (electrode_idx, x, y, z, roi) for a subject as a polars DataFrame."""
    elec_df = get_electrode_df(subject)
    tmp = elec_df.reset_index()[["electrode_idx", "x", "y", "z", "roi"]]
    tmp["roi"] = tmp["roi"].astype(str)
    return pl.from_pandas(tmp)


def _discover_subjects(scores_root: Path, subdir: str, subjects: Iterable[str] | None = None) -> list[Path]:
    paths = sorted(scores_root.glob(f"{subdir}/*/scores.parquet"))
    if subjects is not None:
        keep = set(subjects)
        paths = [p for p in paths if p.parent.name in keep]
    return paths


def build_acoustic_brain_plot(
    scores_root: Path,
    subjects: Iterable[str] | None = None,
    *,
    ac_peak_search_smin: int,
    ac_peak_search_smax: int,
    ac_target: str = "categorical_acoustic_cue",
) -> pl.DataFrame:
    """Build per-(subject, electrode, phoneme_pair) acoustic peak summary.

    Reads ``outputs/causal6/acoustic_decoding_single_electrode/<subject>/scores.parquet``
    for each subject, filters to the canonical acoustic search window
    (``smin >= ac_peak_search_smin`` AND ``smax <= ac_peak_search_smax``) and the
    acoustic target (default ``categorical_acoustic_cue``), and computes the
    per-site peak window by argmax of fold-mean ROC-AUC.

    Returns one row per (subject, electrode_idx, phoneme_pair) with columns:
        subject, electrode_idx, phoneme_pair, peak_auc, peak_smin, peak_smax,
        peak_fold_std, n_folds, fold_tstat, x, y, z, roi.
    """
    frames: list[pl.DataFrame] = []
    for p in _discover_subjects(scores_root, "acoustic_decoding_single_electrode", subjects):
        subject = p.parent.name
        df = pl.read_parquet(p).filter(
            (pl.col("target") == ac_target)
            & (pl.col("smin") >= ac_peak_search_smin)
            & (pl.col("smax") <= ac_peak_search_smax)
        )
        win_stats = df.group_by(_AC_WINDOW_KEYS).agg(
            pl.col("test_roc_auc").mean().alias("fold_mean"),
            pl.col("test_roc_auc").std().alias("fold_std"),
            pl.col("test_roc_auc").len().alias("n_folds"),
        )
        peak = (
            win_stats.group_by(_AC_SITE_KEYS)
            .agg(
                pl.col("fold_mean").max().alias("peak_auc"),
                pl.col("smin").get(pl.col("fold_mean").arg_max()).alias("peak_smin"),
                pl.col("smax").get(pl.col("fold_mean").arg_max()).alias("peak_smax"),
                pl.col("fold_std").get(pl.col("fold_mean").arg_max()).alias("peak_fold_std"),
                pl.col("n_folds").get(pl.col("fold_mean").arg_max()).alias("n_folds"),
            )
            .with_columns(
                (
                    (pl.col("peak_auc") - 0.5)
                    / (
                        pl.max_horizontal(pl.col("peak_fold_std"), pl.lit(1e-6))
                        / pl.col("n_folds").cast(pl.Float64).sqrt()
                    )
                ).alias("fold_tstat")
            )
        )
        frames.append(peak.join(_electrode_pl(subject), on="electrode_idx", how="left"))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def build_behavior_brain_plot(
    scores_root: Path,
    subjects: Iterable[str] | None = None,
    *,
    peak_search_smin: int = 0,
    peak_search_smax: int = 290,
    behav_post_offset_s: float = 0.2,
    epoch_tmin: float = EPOCH_TMIN,
    epoch_sfreq: float = EPOCH_SFREQ,
) -> pl.DataFrame:
    """Build per-(subject, electrode, phoneme_pair, word_end) behavior HGA-only peak summary.

    Reads ``behavior_decoding_single_electrode_hga_only/<subject>/scores.parquet``,
    keeps only the ``model=='full'`` rows (these ARE the HGA-only models for this
    pipeline), applies the per-word_end window cap, and picks per-site peak
    window by argmax of fold-mean ROC-AUC.

    Returns columns: subject, electrode_idx, phoneme_pair, word_end, peak_auc,
    peak_smin, peak_smax, peak_fold_std, n_folds, fold_tstat, x, y, z, roi.
    """
    offset_samples = _behavior_offset_samples(epoch_tmin, epoch_sfreq, behav_post_offset_s)
    smax_limit, filter_expr = _filter_behavior_window_expr(
        offset_samples, peak_search_smin, peak_search_smax,
    )
    frames: list[pl.DataFrame] = []
    for p in _discover_subjects(
        scores_root, "behavior_decoding_single_electrode_hga_only", subjects,
    ):
        subject = p.parent.name
        df = (
            pl.read_parquet(p)
            .filter(pl.col("model") == "full")
            .with_columns(smax_limit.alias("_smax_limit"))
            .filter(filter_expr)
            .drop("_smax_limit")
        )
        win_stats = df.group_by(WINDOW_KEYS_BEHAV).agg(
            pl.col("test_roc_auc").mean().alias("fold_mean"),
            pl.col("test_roc_auc").std().alias("fold_std"),
            pl.col("test_roc_auc").len().alias("n_folds"),
        )
        peak = (
            win_stats.group_by(SITE_KEYS_BEHAV)
            .agg(
                pl.col("fold_mean").max().alias("peak_auc"),
                pl.col("smin").get(pl.col("fold_mean").arg_max()).alias("peak_smin"),
                pl.col("smax").get(pl.col("fold_mean").arg_max()).alias("peak_smax"),
                pl.col("fold_std").get(pl.col("fold_mean").arg_max()).alias("peak_fold_std"),
                pl.col("n_folds").get(pl.col("fold_mean").arg_max()).alias("n_folds"),
            )
            .with_columns(
                (
                    (pl.col("peak_auc") - 0.5)
                    / (
                        pl.max_horizontal(pl.col("peak_fold_std"), pl.lit(1e-6))
                        / pl.col("n_folds").cast(pl.Float64).sqrt()
                    )
                ).alias("fold_tstat")
            )
        )
        peak_pos = peak.join(_electrode_pl(subject), on="electrode_idx", how="left")
        # The original notebook adds an explicit subject literal column even
        # though group_by already preserved it; keep this to remain
        # byte-identical with the existing parquet.
        frames.append(peak_pos.with_columns(pl.lit(subject).alias("subject")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def build_behav_full_brain_plot(
    scores_root: Path,
    subjects: Iterable[str] | None = None,
    *,
    peak_search_smin: int = 0,
    peak_search_smax: int = 290,
    behav_post_offset_s: float = 0.2,
    epoch_tmin: float = EPOCH_TMIN,
    epoch_sfreq: float = EPOCH_SFREQ,
) -> pl.DataFrame | None:
    """Build per-(subject, electrode, phoneme_pair, word_end) behavior full vs baseline
    peak-diff summary.

    Reads ``behavior_decoding_single_electrode/<subject>/scores.parquet`` (full +
    baseline rows), pairs them per fold on (subject, phoneme_pair, word_end, fold),
    computes ``diff = full_roc_auc - baseline_roc_auc``, then peaks by argmax of
    fold-mean diff.

    Returns None when no subject directories exist (e.g. for causal6 today the
    ``behavior_decoding_single_electrode`` directory hasn't been produced).

    Returns columns: subject, electrode_idx, phoneme_pair, word_end, peak_diff,
    peak_smin, peak_smax, peak_fold_std, peak_full_roc_auc, n_folds, fold_tstat,
    x, y, z, roi.
    """
    paths = _discover_subjects(scores_root, "behavior_decoding_single_electrode", subjects)
    if not paths:
        return None
    offset_samples = _behavior_offset_samples(epoch_tmin, epoch_sfreq, behav_post_offset_s)
    smax_limit, filter_expr = _filter_behavior_window_expr(
        offset_samples, peak_search_smin, peak_search_smax,
    )
    frames: list[pl.DataFrame] = []
    for p in paths:
        subject = p.parent.name
        raw = pl.read_parquet(p)
        full = raw.filter(pl.col("model") == "full").drop("model")
        base = (
            raw.filter(pl.col("model") == "baseline")
            .drop("model", "electrode_idx", "smin", "smax")
            .rename({"test_roc_auc": "baseline_roc_auc"})
        )
        paired = (
            full.rename({"test_roc_auc": "full_roc_auc"})
            .join(base, on=_BEH_FULL_FOLD_KEYS, how="left")
            .with_columns((pl.col("full_roc_auc") - pl.col("baseline_roc_auc")).alias("diff"))
        )
        paired = (
            paired.with_columns(smax_limit.alias("_smax_limit"))
            .filter(filter_expr)
            .drop("_smax_limit")
        )
        win_stats = paired.group_by(WINDOW_KEYS_BEHAV).agg(
            pl.col("diff").mean().alias("fold_mean_diff"),
            pl.col("diff").std().alias("fold_std_diff"),
            pl.col("full_roc_auc").mean().alias("fold_mean_full"),
            pl.col("baseline_roc_auc").mean().alias("fold_mean_baseline"),
            pl.col("diff").len().alias("n_folds"),
        )
        peak = (
            win_stats.group_by(SITE_KEYS_BEHAV)
            .agg(
                pl.col("fold_mean_diff").max().alias("peak_diff"),
                pl.col("smin").get(pl.col("fold_mean_diff").arg_max()).alias("peak_smin"),
                pl.col("smax").get(pl.col("fold_mean_diff").arg_max()).alias("peak_smax"),
                pl.col("fold_std_diff").get(pl.col("fold_mean_diff").arg_max()).alias("peak_fold_std"),
                pl.col("fold_mean_full").get(pl.col("fold_mean_diff").arg_max()).alias("peak_full_roc_auc"),
                pl.col("n_folds").get(pl.col("fold_mean_diff").arg_max()).alias("n_folds"),
            )
            .with_columns(
                (
                    pl.col("peak_diff")
                    / (
                        pl.max_horizontal(pl.col("peak_fold_std"), pl.lit(1e-6))
                        / pl.col("n_folds").cast(pl.Float64).sqrt()
                    )
                ).alias("fold_tstat")
            )
        )
        frames.append(peak.join(_electrode_pl(subject), on="electrode_idx", how="left"))
    if not frames:
        return None
    return pl.concat(frames)


def build_ganong_brain_plot(
    scores_root: Path,
    subjects: Iterable[str] | None = None,
    *,
    peak_search_smax: int,
    epoch_tmin: float = EPOCH_TMIN,
    epoch_sfreq: float = EPOCH_SFREQ,
) -> pl.DataFrame:
    """Build per-(subject, electrode, phoneme_pair) Ganong peak summary.

    Reads ``ganong_decoding_single_electrode/<subject>/scores.parquet``, pairs
    full + baseline rows per (subject, phoneme_pair, fold), computes
    ``diff = full - baseline``, applies the POD-floor window filter
    (per-phoneme-pair smin floor + global smax cap), and peaks by argmax of
    fold-mean diff.

    Returns columns: subject, electrode_idx, phoneme_pair, peak_diff,
    peak_smin, peak_smax, peak_fold_std, peak_full_roc_auc, n_folds, fold_tstat,
    x, y, z, roi.
    """
    pod_samples = _ganong_pod_samples(epoch_tmin, epoch_sfreq)
    frames: list[pl.DataFrame] = []
    for p in _discover_subjects(scores_root, "ganong_decoding_single_electrode", subjects):
        subject = p.parent.name
        raw = pl.read_parquet(p)
        full_g = raw.filter(pl.col("model") == "full").drop("model")
        base_g = (
            raw.filter(pl.col("model") == "baseline")
            .drop("model", "electrode_idx", "smin", "smax")
            .rename({"test_roc_auc": "baseline_roc_auc"})
        )
        paired_g = (
            full_g.rename({"test_roc_auc": "full_roc_auc"})
            .join(base_g, on=_GANONG_FOLD_KEYS, how="left")
            .with_columns((pl.col("full_roc_auc") - pl.col("baseline_roc_auc")).alias("diff"))
            .with_columns(
                pl.col("phoneme_pair")
                .replace_strict(pod_samples, default=None)
                .alias("_smin_floor")
            )
            .filter(
                (pl.col("smin") >= pl.col("_smin_floor"))
                & (pl.col("smax") <= peak_search_smax)
            )
            .drop("_smin_floor")
        )
        win_stats_g = paired_g.group_by(_GANONG_WINDOW_KEYS).agg(
            pl.col("diff").mean().alias("fold_mean_diff"),
            pl.col("diff").std().alias("fold_std_diff"),
            pl.col("full_roc_auc").mean().alias("fold_mean_full"),
            pl.col("baseline_roc_auc").mean().alias("fold_mean_baseline"),
            pl.col("diff").len().alias("n_folds"),
        )
        peak_g = (
            win_stats_g.group_by(_GANONG_SITE_KEYS)
            .agg(
                pl.col("fold_mean_diff").max().alias("peak_diff"),
                pl.col("smin").get(pl.col("fold_mean_diff").arg_max()).alias("peak_smin"),
                pl.col("smax").get(pl.col("fold_mean_diff").arg_max()).alias("peak_smax"),
                pl.col("fold_std_diff").get(pl.col("fold_mean_diff").arg_max()).alias("peak_fold_std"),
                pl.col("fold_mean_full").get(pl.col("fold_mean_diff").arg_max()).alias("peak_full_roc_auc"),
                pl.col("n_folds").get(pl.col("fold_mean_diff").arg_max()).alias("n_folds"),
            )
            .with_columns(
                (
                    pl.col("peak_diff")
                    / (
                        pl.max_horizontal(pl.col("peak_fold_std"), pl.lit(1e-6))
                        / pl.col("n_folds").cast(pl.Float64).sqrt()
                    )
                ).alias("fold_tstat")
            )
        )
        frames.append(peak_g.join(_electrode_pl(subject), on="electrode_idx", how="left"))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def load_epochs_dict(epochs_dir: Path) -> dict[str, "mne.Epochs"]:
    """Glob ``<epochs_dir>/*_epo.fif``, parse the ECxxx subject ID, and load
    each Epochs object (lazy: ``preload=False``).

    Metadata is augmented in place via ``src.data.add_metadata_features``.
    """
    out: dict[str, mne.Epochs] = {}
    for p in sorted(Path(epochs_dir).glob("*_epo.fif")):
        m = re.search(r"(EC\d+)_epo", str(p))
        if not m:
            continue
        subj = m.group(1)
        ep = mne.read_epochs(str(p), preload=False, verbose="WARNING")
        ep.metadata = add_metadata_features(ep.metadata.copy())
        out[subj] = ep
    return out


def load_ambig_steps(
    epochs_dict: dict[str, "mne.Epochs"],
    ambiguous_response_threshold: int = 2,
) -> dict[tuple[str, str, str], list[int]]:
    """Build the (subject, phoneme_pair, word_end) -> ambiguous-step-list dict.

    Aliases ``behavior_categorical`` to ``behavior_dummy_forced`` if needed,
    since ``get_ambiguous_resampled_steps`` expects the latter.
    """
    import pandas as pd  # local — keep src.viz_provisional import-cheap

    frames = []
    for s, ep in epochs_dict.items():
        md = ep.metadata.copy()
        md["subject"] = s
        frames.append(md)
    all_md_pd = pd.concat(frames, ignore_index=True)
    if "behavior_dummy_forced" not in all_md_pd.columns:
        all_md_pd = all_md_pd.rename(columns={"behavior_categorical": "behavior_dummy_forced"})
    return get_ambiguous_resampled_steps(
        pl.from_pandas(
            all_md_pd[
                ["subject", "phoneme_pair", "word_end", "resampled", "behavior_dummy_forced"]
            ]
        ),
        ambiguous_response_threshold=ambiguous_response_threshold,
    )


def provisional_star_plot(
    subject: str,
    electrode_idx: int,
    phoneme_pair: str,
    word_end: str,
    epochs_dict: dict[str, "mne.Epochs"],
    ambig_steps: dict[tuple[str, str, str], list[int]],
    *,
    phon_smin: int | None = None,
    phon_smax: int | None = None,
    behav_smin: int | None = None,
    behav_smax: int | None = None,
    textgrid_dir: str = "data/stimuli/textgrid",
    epoch_tmin: float = EPOCH_TMIN,
    epoch_sfreq: float = EPOCH_SFREQ,
    figsize: tuple[float, float] = (6.5, 7.5),
    acoustic_peak_auc: float | None = None,
    acoustic_peak_auc_pct: float | None = None,
    behav_full_peak_diff: float | None = None,
    behav_full_peak_diff_pct: float | None = None,
    behav_hga_peak_auc: float | None = None,
    behav_hga_peak_auc_pct: float | None = None,
) -> "plt.Figure":
    """Three-panel provisional HGA star plot (no prepare_neurometrics required).

    Top: unambiguous trials (resampled 1 & 6); acoustic window shaded if
    phon_smin/phon_smax supplied.
    Middle: within-completion controlled ambiguous trials (behaviorally-defined
    steps from ambig_steps), split by response; behavioral window shaded if
    behav_smin/behav_smax supplied.
    Bottom: all trials within word_end (no step filter), split by response —
    reflects what the decoder actually evaluates on.
    """
    ep = epochs_dict[subject]
    times = ep.times
    md = ep.metadata

    pp_mask = md["phoneme_pair"] == phoneme_pair
    ep_pp = ep[pp_mask.values]
    md_pp = md[pp_mask].reset_index(drop=True)

    hga = (
        ep_pp.copy()
        .apply_baseline((None, 0))
        .get_data(picks=[electrode_idx])
        .squeeze(1)
    )

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    bhv_col = (
        "behavior_dummy_forced"
        if "behavior_dummy_forced" in md_pp.columns
        else "behavior_categorical"
    )
    bhv_colors = ["#762a83", "#1b7837"]

    # ── Top: unambiguous ───────────────────────────────────────────────
    step_colors = {1: "#2166ac", 6: "#d73027"}
    for step, color in step_colors.items():
        mask = md_pp["resampled"] == step
        if not mask.any():
            continue
        tr = hga[mask.values]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        ax_top.plot(times, m, color=color, lw=1.5, label=f"step {step}  (n={mask.sum()})")
        ax_top.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    if phon_smin is not None:
        t_phon = np.array([phon_smin, phon_smax]) / epoch_sfreq + epoch_tmin
        ax_top.axvspan(*t_phon, color="#4dac26", alpha=0.14, label="acoustic window")
    ax_top.axhline(0, color="k", lw=0.5, ls=":")
    ax_top.set_ylabel("HGA (z)")

    def _fmt(label, val, pct, fmt=".3f"):
        if val is None:
            return None
        s = f"{label}={val:{fmt}}"
        if pct is not None:
            s += f" (p{pct:.0f})"
        return s

    metric_parts = [
        s for s in (
            _fmt("ac",       acoustic_peak_auc,    acoustic_peak_auc_pct),
            _fmt("beh_diff", behav_full_peak_diff, behav_full_peak_diff_pct),
            _fmt("beh_hga",  behav_hga_peak_auc,   behav_hga_peak_auc_pct),
        ) if s is not None
    ]
    top_title = f"{subject}  e{electrode_idx}  {phoneme_pair} — unambiguous"
    if metric_parts:
        top_title += "\n" + "  |  ".join(metric_parts)
    ax_top.set_title(top_title, fontsize=9)
    ax_top.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # ── Middle: controlled ambiguous (within-completion) ──────────────
    amb = ambig_steps.get((subject, phoneme_pair, word_end), [3, 4])
    we_amb_mask = (md_pp["word_end"] == word_end) & md_pp["resampled"].isin(amb)

    for i, bhv_val in enumerate(
        sorted(md_pp.loc[we_amb_mask, bhv_col].dropna().unique())
    ):
        mask = we_amb_mask & (md_pp[bhv_col] == bhv_val)
        if not mask.any():
            continue
        tr = hga[mask.values]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        color = bhv_colors[i % len(bhv_colors)]
        ax_mid.plot(times, m, color=color, lw=1.5,
                    label=f"resp={bhv_val}  (n={mask.sum()})")
        ax_mid.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    if behav_smin is not None:
        t_behav = np.array([behav_smin, behav_smax]) / epoch_sfreq + epoch_tmin
        ax_mid.axvspan(*t_behav, color="#f4a582", alpha=0.25, label="behavioral window")
    ax_mid.axhline(0, color="k", lw=0.5, ls=":")
    ax_mid.set_ylabel("HGA (z)")
    ax_mid.set_title(
        f"Controlled ambiguous — {word_end}  (steps {amb})", fontsize=9
    )
    ax_mid.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # ── Bottom: all trials within word_end (decoder view) ─────────────
    we_all_mask = md_pp["word_end"] == word_end

    for i, bhv_val in enumerate(
        sorted(md_pp.loc[we_all_mask, bhv_col].dropna().unique())
    ):
        mask = we_all_mask & (md_pp[bhv_col] == bhv_val)
        if not mask.any():
            continue
        tr = hga[mask.values]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        color = bhv_colors[i % len(bhv_colors)]
        ax_bot.plot(times, m, color=color, lw=1.5,
                    label=f"resp={bhv_val}  (n={mask.sum()})")
        ax_bot.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    if behav_smin is not None:
        t_behav = np.array([behav_smin, behav_smax]) / epoch_sfreq + epoch_tmin
        ax_bot.axvspan(*t_behav, color="#f4a582", alpha=0.25, label="behavioral window")
    ax_bot.axhline(0, color="k", lw=0.5, ls=":")
    ax_bot.set_ylabel("HGA (z)")
    ax_bot.set_xlabel("Time (s, post word onset)")
    ax_bot.set_title(
        f"All trials — {word_end}  (decoder view)", fontsize=9
    )
    ax_bot.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # ── TextGrid ────────────────────────────────────────────────────────
    for ax in (ax_top, ax_mid, ax_bot):
        try:
            add_textgrid(ax, textgrid_dir=textgrid_dir,
                         textgrid_file=f"11_{word_end}_dn_002.TextGrid",
                         vline_extent=1.0)
        except Exception:
            pass

    ax_top.set_xlim(0.0, OFFSET_DICT.get(word_end, 1.0) + 0.1)
    fig.tight_layout()
    return fig


def compute_2x2_contingency(
    sites_a: pl.DataFrame,
    sites_b: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    label_a: str,
    label_b: str,
) -> dict:
    """Build a 2x2 contingency over a universe of sites, partitioning by
    membership in set A vs set B. Returns counts, both conditional probabilities,
    chi-square, and odds ratio with 95% CI (Haldane-Anscombe correction).

    Each input frame must have at least the columns
    ``["subject", "electrode_idx", "phoneme_pair"]``.
    """
    import scipy.stats as stats  # local — viz_provisional should not require scipy at import

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
