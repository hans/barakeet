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
