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

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True)

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
    ax_bot.set_xlabel("Time (s, post word onset)")
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
    figsize: tuple = (8.5, 10.0),
    R_plot: int = 200,
) -> "plt.Figure":
    """Three-panel early-window star plot for one (subject × electrode × phoneme_pair) site.

    Top: A (acoustic) — endpoint steps 1 vs 6, pooled across word_ends.
    Middle: B₁ (WE0 behavioral) — ambiguous qualifying steps for we0.
    Bottom: B₂ (WE1 behavioral) — ambiguous qualifying steps for we1.

    Gray shading: acoustic search range.
    Green bars on A panel: A bootstrap CI-excluding-zero windows.
    Gray bars on B panels: aligned CI-excluding-zero (▲); red bars: anti-aligned (▼).

    Parameters
    ----------
    ep : mne.Epochs
        Full subject epochs (not pre-filtered to phoneme_pair).
    a_per_window, b1_per_window, b2_per_window : polars DataFrame
        Pre-filtered to this site (and word_end for B). Empty DataFrame = no data.
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

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=figsize, sharex=True)

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
        ax.axvspan(t_search_lo, t_search_hi, color="gray", alpha=0.08, zorder=0)
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

    ax_bot.set_xlabel("Time (s, post word onset)")
    xlim = max(
        OFFSET_DICT.get(we0, 1.0) if we0 else 1.0,
        OFFSET_DICT.get(we1, 1.0) if we1 else 1.0,
    ) + 0.15
    ax_top.set_xlim(0.0, xlim)

    fig.tight_layout()
    return fig
