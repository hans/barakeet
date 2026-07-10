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

from typing import Optional, Sequence

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


def _permute_per_step(
    per_step_q: dict,
    rng: np.random.Generator,
) -> dict:
    """Shuffle class labels within each step, preserving per-step class sizes.

    Canonical within-step permutation (matches t_tests.bootstrap_cell's null
    path): for each step, pool all class trial indices, shuffle, re-split with
    the same class sizes. Per-step class counts are invariant — only the
    percept↔HGA link is destroyed.
    """
    perm: dict = {}
    for s, by_class in per_step_q.items():
        class_keys = sorted(by_class.keys())
        sizes = [len(by_class[k]) for k in class_keys]
        pool = np.concatenate([np.asarray(by_class[k]) for k in class_keys])
        rng.shuffle(pool)
        start = 0
        perm[s] = {}
        for k, sz in zip(class_keys, sizes):
            perm[s][k] = pool[start : start + sz]
            start += sz
    return perm


def _prepare_cell_data(
    ep_pp,
    electrode_idx: int,
    word_end: str,
    *,
    min_class_k: int,
    candidate_steps: Sequence[int],
):
    """Extract HGA and qualifying per-step trial indices for one cell.

    Returns ``(hga, per_step_q)`` or ``(None, None)`` if there are no
    qualifying steps. Separating extraction from the bootstrap loop lets
    callers (e.g. ``oriented_group_band``) call ``extract_hga`` once and
    reuse the array across the observed path and all null replicates.

    hga : (n_trials, n_times) float array
    per_step_q : {step: {class_val: np.ndarray of trial indices}}
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
        return None, None

    return hga, {s: per_step[s] for s in qualifying}


def _run_bootstrap_meandiff(
    hga: np.ndarray,
    per_step_q: dict,
    *,
    bootstrap_r: int,
    bootstrap_seed: int,
    perm_rng: Optional[np.random.Generator] = None,
):
    """Bootstrap mean diff given pre-extracted HGA and per-step trial indices.

    If ``perm_rng`` is not None, class labels are permuted within each step
    before bootstrapping (null path). Returns ``(mean_diff, status)`` where
    status ∈ {"ok", "skipped"}.

    All ``bootstrap_r`` draws are generated in one vectorized call per
    (step, class) rather than a Python loop, so runtime is dominated by
    numpy array operations rather than interpreter overhead.
    """
    if perm_rng is not None:
        per_step_q = _permute_per_step(per_step_q, perm_rng)

    rng = np.random.default_rng(bootstrap_seed)
    drawn: dict = {}
    for step, by_class in per_step_q.items():
        n_s = min(len(v) for v in by_class.values())
        if n_s == 0:
            continue
        for cls, idxs in by_class.items():
            # (bootstrap_r, n_s) — all replicates in one call
            drawn.setdefault(cls, []).append(
                rng.choice(idxs, size=(bootstrap_r, n_s), replace=True)
            )

    if 0 not in drawn or 1 not in drawn:
        return None, "skipped"

    # Concatenate steps → (bootstrap_r, K) per class; index hga → (bootstrap_r, K, n_times)
    idx0 = np.concatenate(drawn[0], axis=1)  # (bootstrap_r, K0)
    idx1 = np.concatenate(drawn[1], axis=1)  # (bootstrap_r, K1)
    # class 0 = first phoneme, class 1 = second phoneme (documented order)
    diff = hga[idx0].mean(axis=1) - hga[idx1].mean(axis=1)  # (bootstrap_r, n_times)
    return diff.mean(axis=0), "ok"


def _analytic_meandiff(
    hga: np.ndarray,
    per_step_q: dict,
    *,
    perm_rng: Optional[np.random.Generator] = None,
):
    """Per-step-balanced class mean difference — analytic (no bootstrap).

    Equivalent to the bootstrap_r → ∞ limit of ``_run_bootstrap_meandiff``:
    the bootstrap mean converges to ``(1/N) Σ_s n_s (μ_0s - μ_1s)`` where
    n_s = min class size at step s, μ_cs = empirical mean of hga for class c
    at step s, and N = Σ_s n_s.  Computing this directly eliminates the
    bootstrap loop and all associated random-number overhead.

    If ``perm_rng`` is not None, class labels are permuted within each step
    first (null path). The permutation is the sole source of randomness.

    Returns ``(mean_diff, status)`` where status ∈ {"ok", "skipped"}.
    """
    if perm_rng is not None:
        per_step_q = _permute_per_step(per_step_q, perm_rng)

    weighted_sum = np.zeros(hga.shape[1])
    total_n = 0
    for step, by_class in per_step_q.items():
        n_s = min(len(v) for v in by_class.values())
        if n_s == 0 or 0 not in by_class or 1 not in by_class:
            continue
        # class 0 = first phoneme, class 1 = second phoneme (documented order)
        weighted_sum += n_s * (hga[by_class[0]].mean(0) - hga[by_class[1]].mean(0))
        total_n += n_s

    if total_n == 0:
        return None, "skipped"
    return weighted_sum / total_n, "ok"


def behavioral_bootstrap_meandiff(
    ep_pp,
    electrode_idx,
    word_end,
    *,
    min_class_k: int = 4,
    bootstrap_r: int = 1000,
    bootstrap_seed: int = 42,
    candidate_steps: Sequence[int] = (2, 3, 4, 5),
    perm_rng: Optional[np.random.Generator] = None,
):
    """Within-completion behavioral contrast for one (site × word_end) cell.

    Per-step class-balanced bootstrap on ambiguous steps, streamed to avoid
    memory accumulation. Returns ``(mean_diff, status)`` where ``mean_diff`` is
    the bootstrap mean of ``HGA[class 0].mean(0) - HGA[class 1].mean(0)``
    (class 0 = first phoneme, class 1 = second phoneme) as a 1D array, or
    ``None``. ``status`` ∈ {"ok", "no_qualifying", "skipped"}.

    No sign is applied: the caller orients via its own tuning convention, e.g.
    ``behav_sign * mean_diff`` or ``-acoustic_sign * mean_diff``.

    ``perm_rng``: when provided, class labels are permuted within each step
    before bootstrapping (within-step shuffle preserves per-step class counts).
    This is the null path for ``oriented_group_band``; the observed path leaves
    ``perm_rng`` as None.
    """
    hga, per_step_q = _prepare_cell_data(
        ep_pp, electrode_idx, word_end,
        min_class_k=min_class_k,
        candidate_steps=candidate_steps,
    )
    if hga is None:
        return None, "no_qualifying"
    return _run_bootstrap_meandiff(
        hga, per_step_q,
        bootstrap_r=bootstrap_r,
        bootstrap_seed=bootstrap_seed,
        perm_rng=perm_rng,
    )


def oriented_group_band(
    cells: Sequence[dict],
    epochs_dict: dict,
    *,
    n_perm: int = 1000,
    seed: int = 0,
    min_class_k: int = 4,
    bootstrap_r: int = 1000,
    bootstrap_seed: int = 42,
    candidate_steps: Sequence[int] = (2, 3, 4, 5),
):
    """Observed oriented grand-mean trajectory and matched-permutation null band.

    For each cell in ``cells`` (a dict with keys subject, electrode_idx,
    phoneme_pair, word_end, smin, smax):

    Per cell, HGA is extracted once via ``_prepare_cell_data`` and reused
    across the observed path and all ``n_perm`` null replicates — avoiding
    the ``n_perm`` redundant ``ep.copy().get_data()`` calls that made the
    previous implementation slow.

    1. Extract HGA + qualifying per-step trial indices (once per cell).
    2. Bootstrap observed mean diff; in-window sign = ``sign(diff[smin:smax])``.
    3. Oriented observed trajectory = sign × mean_diff.

    The null repeats the bootstrap ``n_perm`` times with within-step label
    permutation, destroying the percept↔HGA link while preserving per-step
    trial counts. The sign is recomputed from each permuted replicate — this
    captures the rectification floor (orientation bias from selecting a sign
    from the same data being averaged). Reusing the observed sign would
    collapse the null to zero mean, hiding the floor.

    Cells failing the observed path (no qualifying steps) are excluded from
    both observed and null. No-window cells (``smin`` or ``smax`` undefined)
    must be excluded by the caller.

    Runtime: O(n_perm × bootstrap_r × n_cells) bootstrap iterations — but
    ``get_data()`` is called once per cell, not once per replicate.

    Parameters
    ----------
    cells : list of dicts
        Each dict: {subject, electrode_idx, phoneme_pair, word_end, smin, smax}.
    epochs_dict : dict
        {subject: mne.Epochs} mapping (as returned by load_epochs_dict).
    n_perm : int
        Number of permutation replicates for the null band.
    seed : int
        Master seed for permutation RNGs. Each (cell_idx, perm_idx) pair gets
        an independent RNG seeded with ``[seed, cell_idx, perm_idx]``.

    Returns
    -------
    observed_mean : (n_times,) array or None
        Grand mean of sign-oriented cell trajectories.
    null_matrix : (n_perm, n_times) array or None
        Null-distribution trajectories; row p is the grand mean under
        permutation replicate p.
    n_valid : int
        Number of cells that contributed (skipped cells excluded).
    """
    kw_prep = dict(min_class_k=min_class_k, candidate_steps=candidate_steps)
    n_times: Optional[int] = None
    obs_sum: Optional[np.ndarray] = None
    null_matrix: Optional[np.ndarray] = None
    n_valid = 0

    for cell_idx, cell in enumerate(cells):
        subject = cell["subject"]
        electrode_idx = int(cell["electrode_idx"])
        phoneme_pair = cell["phoneme_pair"]
        word_end = cell["word_end"]
        smin = int(cell["smin"])
        smax = int(cell["smax"])

        ep = epochs_dict[subject]
        ep_pp = ep[ep.metadata["phoneme_pair"].values == phoneme_pair]

        # Extract HGA and per-step indices once; reuse across observed + all
        # null replicates (avoids n_perm repeated ep.copy().get_data() calls).
        hga, per_step_q = _prepare_cell_data(
            ep_pp, electrode_idx, word_end, **kw_prep
        )
        if hga is None:
            continue

        # Observed: analytic per-step-balanced class mean (= bootstrap_r→∞ limit).
        obs_diff, status = _analytic_meandiff(hga, per_step_q)
        if status != "ok":
            continue

        if n_times is None:
            n_times = len(obs_diff)
            obs_sum = np.zeros(n_times)
            null_matrix = np.zeros((n_perm, n_times))

        obs_sign = float(np.sign(obs_diff[smin:smax].mean()) or 1.0)
        obs_sum += obs_sign * obs_diff
        n_valid += 1

        # Null: one RNG per cell drives all n_perm label permutations.
        # Analytic mean replaces the bootstrap loop — permutation is the
        # sole source of randomness, so bootstrap_r is not needed here.
        cell_perm_rng = np.random.default_rng([seed, cell_idx])
        for p in range(n_perm):
            perm_diff, perm_status = _analytic_meandiff(
                hga, per_step_q, perm_rng=cell_perm_rng
            )
            if perm_status != "ok":
                continue
            perm_sign = float(np.sign(perm_diff[smin:smax].mean()) or 1.0)
            null_matrix[p] += perm_sign * perm_diff

    if n_valid == 0:
        return None, None, 0

    observed_mean = obs_sum / n_valid
    null_matrix = null_matrix / n_valid
    return observed_mean, null_matrix, n_valid


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
