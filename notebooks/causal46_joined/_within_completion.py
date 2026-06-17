"""Shared trial-selection / HGA-extraction / bootstrap primitives used by
star_plots.py (JON-43) and t_tests.py (JON-44).

Notebook-local on purpose: src/ stays untouched while JON-41 Group B is in
flux. Promote to src/ only if a third caller appears outside this directory.

The B4 sampling rule on `causal6-speech-responsive-update` is *per-step class
balance*: at each ambiguous step `s`, draw `min_class[s]` trials per class
(both classes — both bootstrapped with replacement for the t-test path; in
the gallery only the majority class is subsampled, with the minority used
in full). Concat across steps. Both classes share identical step composition
by construction, so the class-difference trace is free of within-class
step-acoustic confound.

For B3 (single ambiguous step) this collapses to: draw `min_class[s]` of
each class at that one step, bootstrap both with replacement.

`matched_n_star_plot` (defined here, imported by star_plots.py and t_tests.py)
does single-draw, minority-in-full visualisation. The t-test code in
`t_tests.py` does R-replicate bootstrap of *both* classes with replacement.
The two paths therefore display and test slightly different trial subsets —
the filtered-gallery hook in `t_tests.py` joins per (site, word_end[,
resampled]) key only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stimuli import OFFSET_DICT
from src.viz_paper import add_textgrid, epoch_sfreq, epoch_tmin


BEHAVIOR_COL_PREFERENCE: tuple[str, ...] = (
    "behavior_dummy_forced",
    "behavior_categorical",
)


def resolve_behavior_col(md: pd.DataFrame) -> str:
    """Pick the first column in BEHAVIOR_COL_PREFERENCE present in `md`.

    Mirrors the fallback in `matched_n_star_plot` so the t-tests don't
    explode on subjects whose epoch metadata predates the
    `behavior_dummy_forced` rename.
    """
    for col in BEHAVIOR_COL_PREFERENCE:
        if col in md.columns:
            return col
    raise KeyError(
        f"none of {BEHAVIOR_COL_PREFERENCE} found in metadata "
        f"(columns: {list(md.columns)})"
    )


def per_step_class_counts(
    md_pp: pd.DataFrame,
    *,
    word_end: str,
    qualifying_steps: Sequence[int],
    group_col: str,
) -> dict[int, dict[int, np.ndarray]]:
    """Return {step: {class: trial_indices}} for one cell.

    Caller iterates this to compute `min_class[s] = min(len(v) for v in
    .values())` per step, then draws per-step balanced samples from it.
    Pre-computed once per cell so the inner bootstrap loop only resamples
    indices (cheap), never re-scans metadata.
    """
    we_mask = (md_pp["word_end"] == word_end).values
    groups = sorted(md_pp.loc[we_mask, group_col].dropna().unique())
    out: dict[int, dict[int, np.ndarray]] = {}
    for s in qualifying_steps:
        step_mask = we_mask & (md_pp["resampled"] == s).values
        out[int(s)] = {
            int(g): np.where(step_mask & (md_pp[group_col] == g).values)[0]
            for g in groups
        }
    return out


def select_cell_trials_bootstrap(
    per_step: dict[int, dict[int, np.ndarray]],
    *,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """One bootstrap draw under per-step class balance.

    For each step in `per_step`: n_s = min over classes of len(class indices).
    For each class: rng.choice(class_idxs, size=n_s, replace=True) — both
    classes bootstrapped with replacement. Concat across steps.

    Returns {class_value: concatenated_indices_into_md_pp}, each of length
    sum_s n_s. Length is constant across replicates (per cell).
    """
    drawn: dict[int, list[np.ndarray]] = {}
    for step, by_class in per_step.items():
        n_s = min(len(v) for v in by_class.values())
        if n_s == 0:
            # Step contributes nothing — usually filtered upstream, but be defensive.
            continue
        for cls, idxs in by_class.items():
            drawn.setdefault(cls, []).append(
                rng.choice(idxs, size=n_s, replace=True)
            )
    return {cls: np.concatenate(parts) for cls, parts in drawn.items()}


def n_per_class_from_per_step(per_step: dict[int, dict[int, np.ndarray]]) -> int:
    """Sum of min_class[s] across steps — the per-class sample size per replicate."""
    return int(sum(min(len(v) for v in by_class.values())
                   for by_class in per_step.values()))


def extract_hga(ep, electrode_idx: int) -> np.ndarray:
    """Trials × time HGA for one electrode, baseline-corrected.

    Returned array is indexed by ep.metadata's integer index. Callers slice
    with the indices from select_cell_trials_bootstrap.
    """
    return (
        ep.copy()
        .apply_baseline((None, 0))
        .get_data(picks=[electrode_idx])
        .squeeze(1)
    )


@dataclass
class MeanDiffWindow:
    smin: int
    smax: int
    mean_pos: float       # mean HGA over `pos` group's bootstrap indices
    mean_neg: float       # mean HGA over `neg` group's bootstrap indices
    mean_diff: float      # mean_pos - mean_neg
    n_pos: int
    n_neg: int


def searchlight_mean_diff(
    hga: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    *,
    search_smin: int,
    search_smax: int,
    window_size: int = 15,
    stride: int = 15,
) -> list[MeanDiffWindow]:
    """Per-window mean(HGA[pos, win]) - mean(HGA[neg, win]).

    Returns one MeanDiffWindow per window start in
    `range(search_smin, search_smax - window_size + 1, stride)`. Empty list
    if the search interval is shorter than `window_size`.
    """
    out: list[MeanDiffWindow] = []
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    for start in range(search_smin, search_smax - window_size + 1, stride):
        end = start + window_size
        mp = float(hga[pos_idx, start:end].mean())
        mn = float(hga[neg_idx, start:end].mean())
        out.append(MeanDiffWindow(
            smin=start, smax=end,
            mean_pos=mp, mean_neg=mn, mean_diff=mp - mn,
            n_pos=n_pos, n_neg=n_neg,
        ))
    return out


def acoustic_preferred_class(
    hga: np.ndarray,
    md_pp: pd.DataFrame,
    *,
    group_col: str,
    word_end: str,
    acoustic_smin: int,
    acoustic_smax: int,
    endpoints: tuple[int, int] = (1, 6),
) -> int | None:
    """Per cell, identify the behavior class aligned with the cell's
    acoustic tuning.

    Determined on *endpoint trials only* (resampled=1, =6) — never on the
    ambiguous trials the bootstrap t-test runs on. Mechanic:

      1. Compare mean HGA in the acoustic window between the two
         endpoint steps. Pick the endpoint with higher mean HGA — this
         is the acoustically-preferred endpoint.
      2. Map the preferred endpoint to a behavior class via the modal
         `behavior_dummy_forced` value at that endpoint (at endpoints the
         participant almost always reports the corresponding percept).
      3. Return that class. `mean_diff_aligned > 0` then means
         class-matching-acoustic-tuning has higher HGA in the behavioral
         window — a non-circular signed contrast.

    Returns None when either endpoint has no trials, the two endpoint
    means are tied, or the modal behavior at the preferred endpoint is
    tied. Caller emits `mean_diff_aligned = NaN` for those cells.

    `endpoints` is the (low_step, high_step) pair. `word_end` filters the
    pool — the same word_end the bootstrap cell uses.
    """
    we_mask = (md_pp["word_end"] == word_end).values
    low, high = endpoints
    low_mask = we_mask & (md_pp["resampled"] == low).values
    high_mask = we_mask & (md_pp["resampled"] == high).values
    if not low_mask.any() or not high_mask.any():
        return None
    m_low = float(hga[low_mask, acoustic_smin:acoustic_smax].mean())
    m_high = float(hga[high_mask, acoustic_smin:acoustic_smax].mean())
    if m_low == m_high:
        return None
    pref_step_mask = low_mask if m_low > m_high else high_mask
    modal = md_pp.loc[pref_step_mask, group_col].mode()
    if len(modal) != 1:
        return None
    return int(modal.iloc[0])


def bootstrap_A_site(
    hga: np.ndarray,
    md_pp: pd.DataFrame,
    *,
    search_smin: int,
    search_smax: int,
    window_size: int = 10,
    stride: int = 10,
    endpoints: tuple[int, int] = (1, 6),
    min_n: int = 4,
    R: int = 1000,
    base_seed: int = 0,
) -> tuple[list[dict], int, int] | None:
    """Endpoint-balanced A (acoustic) bootstrap for one (electrode × phoneme_pair) site.

    Pools ALL endpoint trials for the pair (both word_ends). Balances between
    step `lo` and step `hi` across replicates (draws min(n_lo, n_hi) per class).

    pos = step hi (default 6 — high-numbered endpoint, e.g. /n/ for dn)
    neg = step lo (default 1 — low-numbered endpoint, e.g. /d/ for dn)
    mean_diff_raw = mean(HGA[pos]) − mean(HGA[neg])
    acoustic_sign = sign(median(mean_diff_raw)) at best significant window

    Returns (rows, n_lo, n_hi) or None if either endpoint has < min_n trials.
    rows: list of dicts with keys replicate, smin, smax, tmin, tmax,
          mean_diff_raw, n_per_class (= min(n_lo, n_hi) for that replicate).
    """
    lo, hi = endpoints
    lo_idx = np.where((md_pp["resampled"] == lo).values)[0]
    hi_idx = np.where((md_pp["resampled"] == hi).values)[0]
    n_lo, n_hi = len(lo_idx), len(hi_idx)
    if n_lo < min_n or n_hi < min_n:
        return None
    n_bal = min(n_lo, n_hi)
    rows: list[dict] = []
    for r in range(R):
        rng = np.random.default_rng(base_seed + r)
        pos_drawn = rng.choice(hi_idx, size=n_bal, replace=True)
        neg_drawn = rng.choice(lo_idx, size=n_bal, replace=True)
        windows = searchlight_mean_diff(
            hga, pos_drawn, neg_drawn,
            search_smin=search_smin, search_smax=search_smax,
            window_size=window_size, stride=stride,
        )
        for w in windows:
            rows.append({
                "replicate": r,
                "smin": w.smin, "smax": w.smax,
                "tmin": w.smin / epoch_sfreq + epoch_tmin,
                "tmax": w.smax / epoch_sfreq + epoch_tmin,
                "mean_diff_raw": w.mean_diff,
                "n_per_class": n_bal,
            })
    return rows, n_lo, n_hi


def load_behav_decoding_scores(full_path, hga_path=None):
    """Load and join behavioral decoding window_mean_scores from explicit file paths.

    full_path : path to behavior_decoding_single_electrode_summarize/.../window_mean_scores.parquet
    hga_path  : path to behavior_decoding_single_electrode_hga_only_summarize/.../window_mean_scores.parquet

    Returns a polars DataFrame with per-(site × window) columns:
      diff (full−baseline), full_roc_auc, baseline_roc_auc, test_roc_auc (HGA-only).
    Returns None if neither file exists.
    """
    import polars as pl
    from pathlib import Path

    full_path = Path(full_path) if full_path is not None else None
    hga_path = Path(hga_path) if hga_path is not None else None

    on_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end", "smin", "smax"]
    full_df = pl.read_parquet(full_path) if full_path is not None and full_path.exists() else None
    hga_df = (
        pl.read_parquet(hga_path).select(on_keys + ["test_roc_auc"])
        if hga_path is not None and hga_path.exists() else None
    )
    if full_df is not None and hga_df is not None:
        return full_df.join(hga_df, on=on_keys, how="left")
    if full_df is not None:
        return full_df.with_columns(pl.lit(None).cast(pl.Float32).alias("test_roc_auc"))
    if hga_df is not None:
        return hga_df.with_columns(
            pl.lit(None).cast(pl.Float32).alias("diff"),
            pl.lit(None).cast(pl.Float32).alias("full_roc_auc"),
            pl.lit(None).cast(pl.Float32).alias("baseline_roc_auc"),
        )
    return None


def _mark_decoding_peaks(ax, t_centers, vals, t_split, color, FS):
    """Vertical dotted lines + value labels at the early and late decoding peaks."""
    for mask in (t_centers <= t_split, t_centers > t_split):
        if not mask.any():
            continue
        t_sub, v_sub = t_centers[mask], vals[mask]
        i_pk = int(np.argmax(v_sub))
        peak_t, peak_v = float(t_sub[i_pk]), float(v_sub[i_pk])
        ax.axvline(peak_t, color=color, lw=0.7, ls=":", alpha=0.65)
        va = "bottom" if peak_v >= 0 else "top"
        ax.text(peak_t, peak_v, f" {peak_v:.3f}", ha="left", va=va,
                fontsize=max(FS - 2, 5), color=color, clip_on=True)


def _draw_behav_decoding_panel(ax, site_df, *, early_smax_s: int, FS: int = 7):
    """Plot full−baseline and HGA-only behavioral decoding advantage curves.

    site_df: polars DataFrame for one (electrode_idx × phoneme_pair × word_end) cell,
             columns smin, smax, diff (full−baseline), test_roc_auc (HGA-only).
    early_smax_s: sample index marking the early / late window boundary.
    Both metrics are expressed as "advantage" referenced to zero:
      full−baseline is already centered; HGA-only AUC is shifted by −0.5.
    """
    import polars as pl

    ax.tick_params(labelsize=FS - 1)
    ax.set_ylabel("dec.\nadv.", fontsize=FS)
    ax.axhline(0, color="k", lw=0.5, ls=":")

    if site_df is None or site_df.height == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FS, color="gray")
        return

    COLOR_FULL = "#7b3294"   # purple: full−baseline
    COLOR_HGA = "#1b7837"    # dark green: HGA-only
    t_early = early_smax_s / epoch_sfreq + epoch_tmin

    smin_arr = site_df["smin"].to_numpy()
    smax_arr = site_df["smax"].to_numpy()
    order = np.argsort(smin_arr)
    t_centers = (smin_arr + smax_arr)[order] / 2 / epoch_sfreq + epoch_tmin

    ax.axvline(t_early, color="k", lw=0.6, ls="--", alpha=0.35)

    if "diff" in site_df.columns:
        diff_vals = site_df["diff"].to_numpy()[order]
        finite = np.isfinite(diff_vals)
        if finite.any():
            ax.plot(t_centers[finite], diff_vals[finite], color=COLOR_FULL,
                    lw=1.1, ls="-", label="full−base")
            _mark_decoding_peaks(ax, t_centers[finite], diff_vals[finite],
                                 t_early, COLOR_FULL, FS)

    if "test_roc_auc" in site_df.columns:
        hga_vals = site_df["test_roc_auc"].to_numpy()[order] - 0.5
        finite = np.isfinite(hga_vals)
        if finite.any():
            ax.plot(t_centers[finite], hga_vals[finite], color=COLOR_HGA,
                    lw=1.1, ls="--", label="HGA−only")
            _mark_decoding_peaks(ax, t_centers[finite], hga_vals[finite],
                                 t_early, COLOR_HGA, FS)

    ax.legend(fontsize=max(FS - 2, 5), loc="upper right", framealpha=0.6,
              handlelength=1.2, ncol=2)


def matched_n_star_plot(
    subject,
    electrode_idx,
    phoneme_pair,
    word_end,
    qualifying_steps,
    *,
    epochs_dict,
    n_per_class,
    phon_smin=None,
    phon_smax=None,
    phon_search_smin=None,
    phon_search_smax=None,
    textgrid_dir="textgrids",
    figsize=(6.5, 5.5),
    acoustic_peak_auc=None,
    R_plot=200,
    sig_windows=None,
    mean_diff_arrays=None,
    xlim=None,
    behav_decoding_df=None,
    early_smax_s=None,
):
    """Two-panel B4 star plot.

    Top panel: unambiguous steps 1 & 6 (acoustic anchor).
    Bottom panel: per-step class-balanced behavioral contrast shown as
    bootstrap mean ± percentile CI (R_plot replicates, same trial-selection
    rule as the main t-test bootstrap). Optionally overlays bootstrap mean
    aligned diff + CI band, and significance bars.

    Parameters
    ----------
    R_plot : int
        Number of bootstrap replicates for the bottom-panel class curves.
    ci_low, ci_high : float
        Percentile bounds for the CI bands (default 2.5 / 97.5).
    sig_windows : list of (tmin, tmax) float tuples, optional
        Windows where the bootstrap CI excludes zero. Drawn as gray bars at
        the top of ax_bot.
    mean_diff_arrays : dict, optional
        Pre-computed bootstrap mean-diff overlay for ax_bot. Expected keys:
        ``tcenter``, ``mean``, ``ci_lo``, ``ci_hi`` (all float arrays).
    """
    ep = epochs_dict[subject]
    md = ep.metadata
    bhv_col = resolve_behavior_col(md)

    pp_mask = (md["phoneme_pair"] == phoneme_pair).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = extract_hga(ep_pp, electrode_idx)
    times = ep.times

    we_mask = (md_pp["word_end"] == word_end).values

    _add_dec = behav_decoding_df is not None
    if _add_dec:
        fig, (ax_top, ax_bot, ax_dec) = plt.subplots(
            3, 1,
            figsize=(figsize[0], figsize[1] + 1.5),
            gridspec_kw={"height_ratios": [1, 1, 0.45]},
            sharex=True,
        )
    else:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        ax_dec = None

    # Top: unambiguous step 1 & 6, restricted to this word_end.
    step_colors = {1: "#2166ac", 6: "#d73027"}
    for step, color in step_colors.items():
        mask = we_mask & (md_pp["resampled"] == step).values
        if not mask.any():
            continue
        tr = hga[mask]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        ax_top.plot(times, m, color=color, lw=1.5,
                    label=f"step {step}  (n={mask.sum()})")
        ax_top.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    if phon_search_smin is not None and phon_search_smax is not None:
        for s in (phon_search_smin, phon_search_smax):
            ax_top.axvline(s / epoch_sfreq + epoch_tmin,
                           color="k", lw=0.6, ls="--", alpha=0.5)
    if phon_smin is not None:
        t_phon = np.array([phon_smin, phon_smax]) / epoch_sfreq + epoch_tmin
        ax_top.axvspan(*t_phon, color="#4dac26", alpha=0.20, label="acoustic peak")
    ax_top.axhline(0, color="k", lw=0.5, ls=":")
    ax_top.set_ylabel("HGA (z)")
    top_title = (
        f"{subject} e{electrode_idx} {phoneme_pair} · {word_end} — unambiguous"
    )
    if acoustic_peak_auc is not None:
        top_title += f"  (ac={acoustic_peak_auc:.3f})"
    ax_top.set_title(top_title, fontsize=9, pad=20)
    ax_top.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # Bottom: bootstrap-estimated class mean HGA timecourses.
    # R_plot replicates of per-step balanced sampling (same protocol as the
    # main t-test bootstrap: both classes drawn with replacement to min_class[s]
    # per step, concatenated across steps).
    bhv_colors = ["#2166ac", "#d73027"]
    bhv_vals = sorted(md_pp.loc[we_mask, bhv_col].dropna().unique())

    per_step = per_step_class_counts(
        md_pp, word_end=word_end,
        qualifying_steps=list(qualifying_steps),
        group_col=bhv_col,
    )
    boot_traces: dict[int, list[np.ndarray]] = {bhv: [] for bhv in bhv_vals}
    for r in range(R_plot):
        draws = select_cell_trials_bootstrap(per_step, rng=np.random.default_rng(r))
        for bhv in bhv_vals:
            if bhv in draws:
                boot_traces[bhv].append(hga[draws[bhv]].mean(0))

    for i, bhv in enumerate(bhv_vals):
        if not boot_traces[bhv]:
            continue
        arr = np.array(boot_traces[bhv])   # (R_plot, n_times)
        m = arr.mean(0)
        se = arr.std(0)   # bootstrap SE ≈ sample SEM
        color = bhv_colors[i % len(bhv_colors)]
        ax_bot.plot(times, m, color=color, lw=1.5,
                    label=f"resp={bhv}  (n≈{n_per_class}/rep)")
        ax_bot.fill_between(times, m - se, m + se, color=color, alpha=0.18)

    # Bootstrap mean aligned diff overlay (dashed line + CI band).
    if mean_diff_arrays is not None:
        tc = mean_diff_arrays["tcenter"]
        mv = mean_diff_arrays["mean"]
        cl = mean_diff_arrays["ci_lo"]
        ch = mean_diff_arrays["ci_hi"]
        valid = np.isfinite(mv)
        if valid.any():
            ax_bot.plot(tc[valid], mv[valid], color="#4d4d4d", lw=1.3, ls="--",
                        label="bootstrap mean diff (aligned)", zorder=4)
            ax_bot.fill_between(tc[valid], cl[valid], ch[valid],
                                color="#4d4d4d", alpha=0.12, zorder=3)

    ax_bot.axhline(0, color="k", lw=0.5, ls=":")
    ax_bot.set_ylabel("HGA (z)")
    ax_bot.set_xlabel("" if _add_dec else "Time (s, post word onset)")
    ax_bot.set_title(
        f"Per-step class-balanced — steps {list(qualifying_steps)}  "
        f"({n_per_class} per class)",
        fontsize=9,
        pad=20,
    )
    ax_bot.legend(fontsize=7, loc="upper left", framealpha=0.7)

    textgrid_file = next(iter(
        Path(textgrid_dir).glob(f"*_{word_end}_{phoneme_pair}_*.TextGrid")
    ))
    for ax in (ax_top, ax_bot):
        add_textgrid(
            ax,
            textgrid_dir=textgrid_dir,
            textgrid_file=textgrid_file.name,
            vline_extent=1.0,
        )

    if xlim is None:
        xlim = OFFSET_DICT.get(word_end, 1.0) + 0.1
    ax_top.set_xlim(0.0, xlim)

    # Significance bars: gray horizontal bars at top of ax_bot for sig windows.
    if sig_windows:
        ymin, ymax = ax_bot.get_ylim()
        bar_h = (ymax - ymin) * 0.04
        bar_y = ymin + (ymax - ymin) * 0.95
        for tmin_s, tmax_s in sig_windows:
            ax_bot.barh(y=bar_y, width=tmax_s - tmin_s, left=tmin_s,
                        height=bar_h, color="gray", alpha=0.6,
                        edgecolor="none", zorder=5)

    # Behavioral decoding panel (thin, below ax_bot).
    if _add_dec:
        import polars as pl
        _site_dec = behav_decoding_df.filter(
            (pl.col("electrode_idx") == electrode_idx)
            & (pl.col("phoneme_pair") == phoneme_pair)
            & (pl.col("word_end") == word_end)
        )
        _early_s = early_smax_s if early_smax_s is not None else phon_search_smax
        _draw_behav_decoding_panel(ax_dec, _site_dec, early_smax_s=_early_s)
        ax_dec.set_xlabel("Time (s, post word onset)")

    fig._ax_behav = ax_bot
    fig.tight_layout()
    return fig


def early_window_star_plot(
    subject: str,
    electrode_idx: int,
    phoneme_pair: str,
    *,
    ep,
    bhv_col: str,
    a_per_window,
    b1_per_window,
    b2_per_window,
    we0: str,
    we1: str,
    b1_qualifying_steps: list,
    b2_qualifying_steps: list,
    b1_n_per_class: int,
    b2_n_per_class: int,
    a_n_step1: int,
    a_n_step6: int,
    acoustic_sign: float,
    site_type: str,
    manifest_tuning: str = "",
    acoustic_peak_auc: float | None = None,
    search_smin: int = 40,
    search_smax: int = 130,
    b_search_smin: int | None = None,
    b_search_smax: int | None = None,
    figsize: tuple = (8.5, 10.0),
    R_plot: int = 200,
    behav_decoding_df=None,
) -> "plt.Figure":
    """Three-panel early-window star plot for one (subject × electrode × phoneme_pair) site.

    Top: A (acoustic) — endpoint steps 1 vs 6, pooled across word_ends.
    Middle: B₁ (WE0 behavioral) — ambiguous qualifying steps for we0.
    Bottom: B₂ (WE1 behavioral) — ambiguous qualifying steps for we1.

    A panel shading: [search_smin, search_smax] (full acoustic search range).
    B panel shading: [b_search_smin, b_search_smax] if provided, else same as A.
    Green bars on A panel: A bootstrap CI-excluding-zero windows.
    Gray bars on B panels: aligned CI-excluding-zero (▲); red bars: anti-aligned (▼).

    Parameters
    ----------
    ep : mne.Epochs
        Full subject epochs (not pre-filtered to phoneme_pair).
    a_per_window, b1_per_window, b2_per_window : polars DataFrame
        Pre-filtered to this site (and word_end for B). Empty DataFrame = no data.
    b_search_smin, b_search_smax : int, optional
        B bootstrap search range in samples. If None, uses search_smin/smax.
    """
    import polars as pl

    md = ep.metadata
    pp_mask = (md["phoneme_pair"] == phoneme_pair).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = extract_hga(ep_pp, electrode_idx)
    times = ep.times

    t_search_lo = search_smin / epoch_sfreq + epoch_tmin
    t_search_hi = search_smax / epoch_sfreq + epoch_tmin
    _bsmin = b_search_smin if b_search_smin is not None else search_smin
    _bsmax = b_search_smax if b_search_smax is not None else search_smax
    t_b_lo = _bsmin / epoch_sfreq + epoch_tmin
    t_b_hi = _bsmax / epoch_sfreq + epoch_tmin

    _add_dec = behav_decoding_df is not None
    if _add_dec:
        fig, (ax_top, ax_mid, ax_mid_dec, ax_bot, ax_bot_dec) = plt.subplots(
            5, 1,
            figsize=(figsize[0], figsize[1] + 2.5),
            gridspec_kw={"height_ratios": [1, 1, 0.4, 1, 0.4]},
            sharex=True,
        )
    else:
        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        ax_mid_dec = ax_bot_dec = None

    # ── Top: A (endpoint steps 1 vs 6, pooled word_ends) ────────────────────
    ax_top.axvspan(t_search_lo, t_search_hi, color="gray", alpha=0.08, zorder=0)
    step_colors = {1: "#2166ac", 6: "#d73027"}
    for step, color in step_colors.items():
        mask = (md_pp["resampled"] == step).values
        if not mask.any():
            continue
        tr = hga[mask]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        ax_top.plot(times, m, color=color, lw=1.5,
                    label=f"step {step}  (n={mask.sum()})")
        ax_top.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    ax_top.axhline(0, color="k", lw=0.5, ls=":")
    ax_top.set_ylabel("HGA (z)")
    ax_top.legend(fontsize=7, loc="upper left", framealpha=0.7)

    # Green significance bars on A panel
    if a_per_window is not None and a_per_window.height > 0:
        a_sig = a_per_window.filter(pl.col("ci_excludes_zero"))
        if a_sig.height > 0:
            ax_top.autoscale_view()
            ymin_a, ymax_a = ax_top.get_ylim()
            bar_h_a = (ymax_a - ymin_a) * 0.04
            bar_y_a = ymin_a + (ymax_a - ymin_a) * 0.94
            for r in a_sig.iter_rows(named=True):
                ax_top.barh(y=bar_y_a, width=r["tmax"] - r["tmin"],
                            left=r["tmin"], height=bar_h_a,
                            color="#4dac26", alpha=0.7, edgecolor="none", zorder=5)
            ax_top.set_ylim(ymin_a, ymax_a)

    auc_str = f"  AUC={acoustic_peak_auc:.3f}" if acoustic_peak_auc is not None else ""
    sign_str = f"  ac_sign={int(acoustic_sign):+d}" if np.isfinite(acoustic_sign) else "  ac_sign=?"
    ax_top.set_title(
        f"{subject} / e{electrode_idx} / {phoneme_pair} / tuning={manifest_tuning!r} / {site_type}\n"
        f"A: n_step1={a_n_step1}  n_step6={a_n_step6}{auc_str}{sign_str}",
        fontsize=8, loc="left", pad=3,
    )

    # ── Helper: draw one behavioral panel ────────────────────────────────────
    def _draw_b_panel(ax, we, qualifying_steps, n_per_class_b, b_pw, panel_label):
        ax.axvspan(t_b_lo, t_b_hi, color="gray", alpha=0.12, zorder=0)
        if not qualifying_steps:
            ax.text(0.5, 0.5,
                    f"B: {we}  — underpowered / no qualifying steps",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_ylabel("HGA (z)")
            return

        per_step_b = per_step_class_counts(
            md_pp, word_end=we,
            qualifying_steps=list(qualifying_steps),
            group_col=bhv_col,
        )
        we_mask_b = (md_pp["word_end"] == we).values
        bhv_vals = sorted(md_pp.loc[we_mask_b, bhv_col].dropna().unique())
        boot_traces: dict = {bhv: [] for bhv in bhv_vals}
        for r in range(R_plot):
            draws = select_cell_trials_bootstrap(per_step_b, rng=np.random.default_rng(r))
            for bhv in bhv_vals:
                if bhv in draws:
                    boot_traces[bhv].append(hga[draws[bhv]].mean(0))

        bhv_colors_b = ["#2166ac", "#d73027"]
        for i, bhv in enumerate(bhv_vals):
            if not boot_traces[bhv]:
                continue
            arr = np.array(boot_traces[bhv])
            m = arr.mean(0)
            se = arr.std(0)
            color = bhv_colors_b[i % len(bhv_colors_b)]
            ax.plot(times, m, color=color, lw=1.5,
                    label=f"resp={bhv}  (n≈{n_per_class_b}/rep)")
            ax.fill_between(times, m - se, m + se, color=color, alpha=0.18)

        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_ylabel("HGA (z)")
        ax.set_title(
            f"{panel_label}: {we}  steps={qualifying_steps}  n_per_class={n_per_class_b}",
            fontsize=8, loc="left", pad=3,
        )
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

        # Significance bars + ▲/▼ glyphs
        if b_pw is not None and b_pw.height > 0:
            b_sig_rows = b_pw.filter(pl.col("ci_aligned_excludes_zero"))
            if b_sig_rows.height > 0:
                ax.autoscale_view()
                ymin_b, ymax_b = ax.get_ylim()
                bar_h_b = (ymax_b - ymin_b) * 0.04
                bar_y_al = ymin_b + (ymax_b - ymin_b) * 0.94
                bar_y_an = ymin_b + (ymax_b - ymin_b) * 0.88
                for brow in b_sig_rows.iter_rows(named=True):
                    med = brow.get("mean_diff_aligned_med") or 0.0
                    is_aligned = med > 0
                    ax.barh(
                        y=bar_y_al if is_aligned else bar_y_an,
                        width=brow["tmax"] - brow["tmin"],
                        left=brow["tmin"],
                        height=bar_h_b,
                        color="gray" if is_aligned else "#d73027",
                        alpha=0.65, edgecolor="none", zorder=5,
                    )
                    ax.text(
                        (brow["tmin"] + brow["tmax"]) / 2,
                        bar_y_al if is_aligned else bar_y_an,
                        "▲" if is_aligned else "▼",
                        ha="center", va="center", fontsize=5, zorder=6,
                    )
                ax.set_ylim(ymin_b, ymax_b)

    _draw_b_panel(ax_mid, we0, b1_qualifying_steps, b1_n_per_class, b1_per_window, "B₁")
    _draw_b_panel(ax_bot, we1, b2_qualifying_steps, b2_n_per_class, b2_per_window, "B₂")

    # Behavioral decoding panels (thin rows below each B panel).
    if _add_dec:
        _dec_b1 = behav_decoding_df.filter(
            (pl.col("electrode_idx") == electrode_idx)
            & (pl.col("phoneme_pair") == phoneme_pair)
            & (pl.col("word_end") == we0)
        ) if we0 else pl.DataFrame()
        _dec_b2 = behav_decoding_df.filter(
            (pl.col("electrode_idx") == electrode_idx)
            & (pl.col("phoneme_pair") == phoneme_pair)
            & (pl.col("word_end") == we1)
        ) if we1 else pl.DataFrame()
        _draw_behav_decoding_panel(ax_mid_dec, _dec_b1 if _dec_b1.height > 0 else None,
                                   early_smax_s=_bsmax)
        _draw_behav_decoding_panel(ax_bot_dec, _dec_b2 if _dec_b2.height > 0 else None,
                                   early_smax_s=_bsmax)
        ax_bot_dec.set_xlabel("Time (s, post word onset)")
    else:
        ax_bot.set_xlabel("Time (s, post word onset)")

    xlim = max(
        OFFSET_DICT.get(we0, 1.0) if we0 else 1.0,
        OFFSET_DICT.get(we1, 1.0) if we1 else 1.0,
    ) + 0.15
    ax_top.set_xlim(0.0, xlim)

    fig._ax_b2 = ax_bot
    fig.tight_layout()
    return fig


def early_window_star_plot_compact(
    subject: str,
    electrode_idx: int,
    phoneme_pair: str,
    *,
    ep,
    bhv_col: str,
    a_per_window,
    b1_per_window,
    b2_per_window,
    we0: str,
    we1: str,
    b1_qualifying_steps: list,
    b2_qualifying_steps: list,
    b1_n_per_class: int,
    b2_n_per_class: int,
    a_n_step1: int,
    a_n_step6: int,
    acoustic_sign: float,
    site_type: str,
    manifest_tuning: str = "",
    acoustic_peak_auc: float | None = None,
    search_smin: int = 40,
    search_smax: int = 130,
    b_search_smin: int | None = None,
    b_search_smax: int | None = None,
    figsize: tuple = (3.5, 8.5),
    R_plot: int = 200,
    behav_decoding_df=None,
) -> "plt.Figure":
    """Compact vertical version of early_window_star_plot for slide tiling.

    Same 3×1 vertical layout (A on top, B₁ middle, B₂ bottom) with a narrow
    default figsize so several can be placed side-by-side on a slide.
    Reduced font sizes and tighter margins; all significance markers retained.
    """
    import polars as pl

    md = ep.metadata
    pp_mask = (md["phoneme_pair"] == phoneme_pair).values
    ep_pp = ep[pp_mask]
    md_pp = md[pp_mask].reset_index(drop=True)
    hga = extract_hga(ep_pp, electrode_idx)
    times = ep.times

    t_search_lo = search_smin / epoch_sfreq + epoch_tmin
    t_search_hi = search_smax / epoch_sfreq + epoch_tmin
    _bsmin = b_search_smin if b_search_smin is not None else search_smin
    _bsmax = b_search_smax if b_search_smax is not None else search_smax
    t_b_lo = _bsmin / epoch_sfreq + epoch_tmin
    t_b_hi = _bsmax / epoch_sfreq + epoch_tmin

    _add_dec = behav_decoding_df is not None
    if _add_dec:
        fig, (ax_top, ax_mid, ax_mid_dec, ax_bot, ax_bot_dec) = plt.subplots(
            5, 1,
            figsize=(figsize[0], figsize[1] + 2.0),
            gridspec_kw={"height_ratios": [1, 1, 0.35, 1, 0.35]},
            sharex=True,
        )
    else:
        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        ax_mid_dec = ax_bot_dec = None

    FS = 6   # base font size for compact layout

    # ── Top: A (acoustic) ────────────────────────────────────────────────────
    ax_top.axvspan(t_search_lo, t_search_hi, color="gray", alpha=0.08, zorder=0)
    step_colors = {1: "#2166ac", 6: "#d73027"}
    for step, color in step_colors.items():
        mask = (md_pp["resampled"] == step).values
        if not mask.any():
            continue
        tr = hga[mask]
        m = tr.mean(0)
        se = tr.std(0) / np.sqrt(mask.sum())
        ax_top.plot(times, m, color=color, lw=1.0,
                    label=f"s{step} (n={mask.sum()})")
        ax_top.fill_between(times, m - se, m + se, color=color, alpha=0.18)
    ax_top.axhline(0, color="k", lw=0.4, ls=":")
    ax_top.set_ylabel("HGA (z)", fontsize=FS)
    ax_top.tick_params(labelsize=FS - 1)
    ax_top.legend(fontsize=FS - 1, loc="upper left", framealpha=0.6, handlelength=1)

    if a_per_window is not None and a_per_window.height > 0:
        a_sig = a_per_window.filter(pl.col("ci_excludes_zero"))
        if a_sig.height > 0:
            ax_top.autoscale_view()
            ymin_a, ymax_a = ax_top.get_ylim()
            bar_h_a = (ymax_a - ymin_a) * 0.04
            bar_y_a = ymin_a + (ymax_a - ymin_a) * 0.94
            for r in a_sig.iter_rows(named=True):
                ax_top.barh(y=bar_y_a, width=r["tmax"] - r["tmin"],
                            left=r["tmin"], height=bar_h_a,
                            color="#4dac26", alpha=0.7, edgecolor="none", zorder=5)
            ax_top.set_ylim(ymin_a, ymax_a)

    auc_str = f" AUC={acoustic_peak_auc:.2f}" if acoustic_peak_auc is not None else ""
    sign_str = f" sgn={int(acoustic_sign):+d}" if np.isfinite(acoustic_sign) else ""
    ax_top.set_title(
        f"{subject} e{electrode_idx} {phoneme_pair} [{site_type}]\n"
        f"n1={a_n_step1} n6={a_n_step6}{auc_str}{sign_str}",
        fontsize=FS, loc="left", pad=2,
    )

    # ── Helper: compact behavioral panel ─────────────────────────────────────
    def _draw_b_compact(ax, we, qualifying_steps, n_per_class_b, b_pw, label):
        ax.axvspan(t_b_lo, t_b_hi, color="gray", alpha=0.12, zorder=0)
        ax.tick_params(labelsize=FS - 1)
        ax.set_ylabel("HGA (z)", fontsize=FS)
        if not qualifying_steps:
            ax.text(0.5, 0.5, f"{label}: {we}\nunderpowered",
                    ha="center", va="center", transform=ax.transAxes, fontsize=FS)
            return

        per_step_b = per_step_class_counts(
            md_pp, word_end=we,
            qualifying_steps=list(qualifying_steps),
            group_col=bhv_col,
        )
        we_mask_b = (md_pp["word_end"] == we).values
        bhv_vals = sorted(md_pp.loc[we_mask_b, bhv_col].dropna().unique())
        boot_traces: dict = {bhv: [] for bhv in bhv_vals}
        for r in range(R_plot):
            draws = select_cell_trials_bootstrap(per_step_b, rng=np.random.default_rng(r))
            for bhv in bhv_vals:
                if bhv in draws:
                    boot_traces[bhv].append(hga[draws[bhv]].mean(0))

        bhv_colors_b = ["#2166ac", "#d73027"]
        for i, bhv in enumerate(bhv_vals):
            if not boot_traces[bhv]:
                continue
            arr = np.array(boot_traces[bhv])
            m = arr.mean(0)
            se = arr.std(0)
            color = bhv_colors_b[i % len(bhv_colors_b)]
            ax.plot(times, m, color=color, lw=1.0,
                    label=f"r={bhv} (n≈{n_per_class_b})")
            ax.fill_between(times, m - se, m + se, color=color, alpha=0.18)

        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.set_title(f"{label}: {we}  steps={qualifying_steps}",
                     fontsize=FS, loc="left", pad=2)
        ax.legend(fontsize=FS - 1, loc="upper left", framealpha=0.6, handlelength=1)

        if b_pw is not None and b_pw.height > 0:
            b_sig_rows = b_pw.filter(pl.col("ci_aligned_excludes_zero"))
            if b_sig_rows.height > 0:
                ax.autoscale_view()
                ymin_b, ymax_b = ax.get_ylim()
                bar_h_b = (ymax_b - ymin_b) * 0.04
                bar_y_al = ymin_b + (ymax_b - ymin_b) * 0.94
                bar_y_an = ymin_b + (ymax_b - ymin_b) * 0.88
                for brow in b_sig_rows.iter_rows(named=True):
                    med = brow.get("mean_diff_aligned_med") or 0.0
                    is_aligned = med > 0
                    ax.barh(
                        y=bar_y_al if is_aligned else bar_y_an,
                        width=brow["tmax"] - brow["tmin"],
                        left=brow["tmin"],
                        height=bar_h_b,
                        color="gray" if is_aligned else "#d73027",
                        alpha=0.65, edgecolor="none", zorder=5,
                    )
                    ax.text(
                        (brow["tmin"] + brow["tmax"]) / 2,
                        bar_y_al if is_aligned else bar_y_an,
                        "▲" if is_aligned else "▼",
                        ha="center", va="center", fontsize=FS - 2, zorder=6,
                    )
                ax.set_ylim(ymin_b, ymax_b)

    _draw_b_compact(ax_mid, we0, b1_qualifying_steps, b1_n_per_class, b1_per_window, "B₁")
    _draw_b_compact(ax_bot, we1, b2_qualifying_steps, b2_n_per_class, b2_per_window, "B₂")

    # Behavioral decoding panels (thin rows below each B panel).
    if _add_dec:
        _dec_b1 = behav_decoding_df.filter(
            (pl.col("electrode_idx") == electrode_idx)
            & (pl.col("phoneme_pair") == phoneme_pair)
            & (pl.col("word_end") == we0)
        ) if we0 else pl.DataFrame()
        _dec_b2 = behav_decoding_df.filter(
            (pl.col("electrode_idx") == electrode_idx)
            & (pl.col("phoneme_pair") == phoneme_pair)
            & (pl.col("word_end") == we1)
        ) if we1 else pl.DataFrame()
        _draw_behav_decoding_panel(ax_mid_dec, _dec_b1 if _dec_b1.height > 0 else None,
                                   early_smax_s=_bsmax, FS=FS)
        _draw_behav_decoding_panel(ax_bot_dec, _dec_b2 if _dec_b2.height > 0 else None,
                                   early_smax_s=_bsmax, FS=FS)
        ax_bot_dec.set_xlabel("Time (s)", fontsize=FS)
    else:
        ax_bot.set_xlabel("Time (s)", fontsize=FS)

    xlim = max(
        OFFSET_DICT.get(we0, 1.0) if we0 else 1.0,
        OFFSET_DICT.get(we1, 1.0) if we1 else 1.0,
    ) + 0.15
    ax_top.set_xlim(0.0, xlim)

    fig._ax_b2 = ax_bot
    fig.tight_layout(pad=0.5, h_pad=0.6)
    return fig
