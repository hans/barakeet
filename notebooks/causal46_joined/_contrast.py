"""Shared continuous-time HGA contrast-plot primitives (causal46_joined).

Extracted from ``contrast_plot.py`` so that both the behavior-driven contrast
plot and the per-site-type acoustic contrast plot
(``contrast_plot_by_site_type.py``) share one implementation of trajectory
aggregation, sliding-window significance testing, the per-site contrast
computations, and the two-line + significance-bar plotting style.

Notebook-local on purpose (same tier as ``_within_completion.py``, which this
imports from). Promote to ``src/`` only if a caller appears outside this
directory.

Sign convention is deliberately NOT baked in here: ``acoustic_endpoint_means``
returns the two endpoint mean trajectories and ``behavioral_bootstrap_meandiff``
returns the bootstrap mean of a *documented* subtraction order. Each caller
forms its own oriented contrast explicitly, so there is no hidden polarity.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

from _within_completion import (
    extract_hga,
    per_step_class_counts,
    resolve_behavior_col,
    select_cell_trials_bootstrap,
)

# Default line colors (from viz_paper.py style)
ACOUSTIC_COLOR = "#2166ac"
BEHAVIORAL_COLOR = "#d73027"

# p-threshold height multipliers (from viz_paper.py style)
P_THRESHOLD_MULTS = [1.0, 0.5, 0.25]


# --------------------------------------------------------------------------- #
# Aggregation + significance testing
# --------------------------------------------------------------------------- #
def aggregate_trajectories(trajectories):
    """Return (matrix, grand_mean, sem) from a list of 1D trajectory arrays.

    SEM uses ddof=1, so it is NaN for a single trajectory; callers should guard
    on ``matrix.shape[0] >= 2`` before drawing a ribbon.
    """
    if not trajectories:
        return None, None, None
    matrix = np.stack(trajectories, axis=0)  # (n_sites, n_times)
    grand_mean = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    return matrix, grand_mean, sem


def sliding_ttest(matrix, times, window_size, window_stride):
    """One-sample t-test on sliding windows. Returns list of (t_start, t_end, p_val).

    Returns an empty list when ``matrix`` has fewer than 2 rows (t-test
    undefined).
    """
    if matrix is None or matrix.shape[0] < 2:
        return []
    n_times = matrix.shape[1]
    results = []
    for start in range(0, n_times - window_size + 1, window_stride):
        window_means = matrix[:, start:start + window_size].mean(axis=1)
        t_stat, p_val = scipy.stats.ttest_1samp(window_means, 0)
        end = min(start + window_size, n_times - 1)
        results.append((times[start], times[end], p_val))
    return results


# --------------------------------------------------------------------------- #
# Per-site contrast computations
# --------------------------------------------------------------------------- #
def acoustic_endpoint_means(ep_pp, electrode_idx, *, endpoints=(1, 6)):
    """Mean HGA trajectory at each unambiguous endpoint step for one site.

    ``ep_pp`` must already be restricted to the relevant phoneme_pair. Returns
    ``(mean_low, mean_high)`` — the trial-averaged HGA for the low endpoint
    (default step 1 = clear first phoneme) and high endpoint (default step 6 =
    clear second phoneme), each a 1D array over time. Returns ``None`` if either
    endpoint has no trials.

    No sign is applied: the caller forms its own oriented contrast, e.g.
    ``sign * (mean_high - mean_low)``.
    """
    md_pp = ep_pp.metadata.reset_index(drop=True)
    hga = extract_hga(ep_pp, electrode_idx)
    lo, hi = endpoints
    lo_mask = (md_pp["resampled"] == lo).values
    hi_mask = (md_pp["resampled"] == hi).values
    if not lo_mask.any() or not hi_mask.any():
        return None
    return hga[lo_mask].mean(0), hga[hi_mask].mean(0)


def behavioral_bootstrap_meandiff(
    ep_pp,
    electrode_idx,
    word_end,
    *,
    min_class_k: int = 4,
    bootstrap_r: int = 1000,
    bootstrap_seed: int = 42,
    candidate_steps: Sequence[int] = (2, 3, 4, 5),
):
    """Within-completion behavioral contrast for one (site × word_end) cell.

    Per-step class-balanced bootstrap on ambiguous steps, streamed to avoid
    memory accumulation. Returns ``(mean_diff, status)`` where ``mean_diff`` is
    the bootstrap mean of ``HGA[class 0].mean(0) - HGA[class 1].mean(0)``
    (class 0 = first phoneme, class 1 = second phoneme) as a 1D array, or
    ``None``. ``status`` ∈ {"ok", "no_qualifying", "skipped"}.

    No sign is applied: the caller orients via its own tuning convention, e.g.
    ``behav_sign * mean_diff`` or ``-acoustic_sign * mean_diff``.
    """
    md_pp = ep_pp.metadata.reset_index(drop=True)
    hga = extract_hga(ep_pp, electrode_idx)

    bhv_col = resolve_behavior_col(md_pp)
    we_mask = (md_pp["word_end"] == word_end).values
    candidate = [
        s for s in candidate_steps
        if (we_mask & (md_pp["resampled"] == s).values).any()
    ]
    per_step = per_step_class_counts(
        md_pp, word_end=word_end, qualifying_steps=candidate, group_col=bhv_col,
    )
    qualifying = [
        s for s, by_class in per_step.items()
        if len(by_class) == 2 and min(len(v) for v in by_class.values()) >= min_class_k
    ]
    if not qualifying:
        return None, "no_qualifying"

    per_step_q = {s: per_step[s] for s in qualifying}
    running_sum = np.zeros(hga.shape[1])
    valid_reps = 0
    for r in range(bootstrap_r):
        draws = select_cell_trials_bootstrap(
            per_step_q, rng=np.random.default_rng(bootstrap_seed + r)
        )
        if 0 not in draws or 1 not in draws:
            continue
        # class 0 = first phoneme, class 1 = second phoneme (documented order)
        diff_r = hga[draws[0]].mean(0) - hga[draws[1]].mean(0)
        running_sum += diff_r
        valid_reps += 1

    if valid_reps == 0:
        return None, "skipped"
    return running_sum / valid_reps, "ok"


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
class _HandlerRect(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        rect = Rectangle([xdescent, ydescent], width, height,
                         facecolor=orig_handle.get_facecolor(),
                         alpha=orig_handle.get_alpha(),
                         edgecolor="none")
        return [rect]


def plot_contrast_axis(
    ax,
    times,
    *,
    ac_mean=None,
    ac_sem=None,
    ac_ttest=(),
    n_acoustic=0,
    bh_mean=None,
    bh_sem=None,
    bh_ttest=(),
    n_behav=0,
    pval_thresholds=(0.00001, 0.0001, 0.001),
    acoustic_label=None,
    behav_label=None,
    pod_vline=None,
    pod_band=None,
    site_traces=None,
    site_complex_color="#762a83",
    site_trace_color="#9aacc4",
    site_trace_alpha=0.3,
    site_trace_lw=0.5,
    acoustic_color=ACOUSTIC_COLOR,
    behavioral_color=BEHAVIORAL_COLOR,
    draw_legend=True,
    legend_title="Sig bars: acoustic (blue), behavioral (red)\n[different site populations]",
    legend_loc="lower left",
    legend_bbox_to_anchor=None,
    legend_framealpha=None,
):
    """Draw the acoustic + behavioral contrast lines, SEM ribbons, significance
    bars, and legend onto ``ax``.

    Means/SEMs/ttests are precomputed by the caller (via
    ``aggregate_trajectories`` + ``sliding_ttest``). The caller sets title,
    axis labels, and xlim afterwards.

    Ribbons are drawn only when the corresponding population has ≥2 members
    (SEM is otherwise NaN). ``site_traces`` (new in the per-type plot) is an
    optional list of ``(trajectory, is_complex)`` tuples drawn faintly behind
    the acoustic mean; ``None`` reproduces the original single-line style.
    """
    if acoustic_label is None:
        acoustic_label = f"Acoustic (n={n_acoustic} sites)"
    if behav_label is None:
        behav_label = f"Behavioral (n={n_behav} cells)"

    # -- Faint per-site acoustic traces (behind the mean)
    if site_traces:
        for traj, is_complex in site_traces:
            ax.plot(times, traj, lw=site_trace_lw, alpha=site_trace_alpha, zorder=1,
                    color=site_complex_color if is_complex else site_trace_color)

    # -- Acoustic mean + SEM ribbon
    if ac_mean is not None and n_acoustic >= 1:
        ax.plot(times, ac_mean, color=acoustic_color, lw=2, label=acoustic_label, zorder=3)
        if n_acoustic >= 2 and ac_sem is not None and np.all(np.isfinite(ac_sem)):
            ax.fill_between(times, ac_mean - ac_sem, ac_mean + ac_sem,
                            color=acoustic_color, alpha=0.18, zorder=2)

    # -- Behavioral mean + SEM ribbon
    if bh_mean is not None and n_behav >= 1:
        ax.plot(times, bh_mean, color=behavioral_color, lw=2, label=behav_label, zorder=3)
        if n_behav >= 2 and bh_sem is not None and np.all(np.isfinite(bh_sem)):
            ax.fill_between(times, bh_mean - bh_sem, bh_mean + bh_sem,
                            color=behavioral_color, alpha=0.18, zorder=2)

    ax.axhline(0, color="k", lw=0.5, ls=":")

    # -- POD reference (per-pair vline or pooled band)
    if pod_band is not None:
        ax.axvspan(pod_band[0], pod_band[1], color="gray", alpha=0.10,
                   lw=0, label="POD range")
    if pod_vline is not None:
        ax.axvline(pod_vline, color="gray", lw=1.2, ls="--", alpha=0.7, label="POD")

    # -- Significance bars: acoustic (upper row), behavioral (lower row)
    ymin, ymax = ax.get_ylim()
    base_bar_h = (ymax - ymin) * 0.04
    bar_row_gap = base_bar_h * 1.5
    bar_y_ac = ymin + (ymax - ymin) * 0.95
    bar_y_bh = bar_y_ac - bar_row_gap

    p_thresholds_sorted = sorted(pval_thresholds)  # ascending (darkest first)

    def _draw_bars(ttest, bar_y, color):
        for (t_start, t_end, p_val) in ttest:
            for i, p_thresh in enumerate(p_thresholds_sorted):
                if p_val < p_thresh:
                    ax.barh(y=bar_y, width=t_end - t_start, left=t_start,
                            height=base_bar_h * P_THRESHOLD_MULTS[i],
                            color=color, alpha=0.5, edgecolor="none")
                    break

    _draw_bars(ac_ttest, bar_y_ac, acoustic_color)
    _draw_bars(bh_ttest, bar_y_bh, behavioral_color)

    if not draw_legend:
        return

    # -- Legend (lines + p-threshold swatches)
    p_handles = [
        Rectangle((0, 0), 1, mult, facecolor="gray", alpha=0.5,
                  label=f"p < {p_thresh:g}".replace("-0", "-"))
        for p_thresh, mult in zip(p_thresholds_sorted, P_THRESHOLD_MULTS)
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + p_handles,
        labels=labels + [h.get_label() for h in p_handles],
        handler_map={Rectangle: _HandlerRect()},
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        framealpha=legend_framealpha,
        fontsize=8,
        title=legend_title,
        title_fontsize=7,
    )
