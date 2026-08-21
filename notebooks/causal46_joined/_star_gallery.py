"""Star-plot gallery helpers shared by t_tests.py and t_tests_by_early_window.py."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from tqdm.auto import tqdm

from src.stimuli import OFFSET_DICT

from _within_completion import matched_n_star_plot  # noqa: E402

try:
    from pypdf import PdfReader, PdfWriter
    HAS_PYPDF = True
except ImportError:
    PdfReader = PdfWriter = None  # type: ignore[assignment]
    HAS_PYPDF = False


def site_effect_fig(
    row: dict,
    site_per_window: pl.DataFrame,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
) -> plt.Figure:
    """CI-trace figure for one cell: bootstrap mean ± CI band across windows."""
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    if site_per_window.height > 0:
        pw = site_per_window.sort("tmin")
        tcenter = ((pw["tmin"] + pw["tmax"]) / 2).to_numpy().astype(float)
        mn = pw["mean_diff_aligned_mean"].to_numpy().astype(float)
        ci_lo = pw["mean_diff_aligned_ci_lo"].to_numpy().astype(float)
        ci_hi = pw["mean_diff_aligned_ci_hi"].to_numpy().astype(float)
        ax.plot(tcenter, mn, color="#2166ac", lw=1.5,
                label="bootstrap mean aligned diff")
        ax.fill_between(tcenter, ci_lo, ci_hi, color="#2166ac", alpha=0.22,
                        label=f"{ci_low}–{ci_high}% bootstrap CI")
        if row.get("best_tmin") is not None and row.get("best_tmax") is not None:
            ax.axvspan(float(row["best_tmin"]), float(row["best_tmax"]),
                       color="#fdae61", alpha=0.45, label="best window", zorder=0)
    ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("Window center (s, post word onset)")
    ax.set_ylabel("aligned mean_diff (HGA)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    med_val = row.get("best_mean_diff_aligned_med")
    ci_lo_v = row.get("best_mean_diff_aligned_ci_lo")
    ci_hi_v = row.get("best_mean_diff_aligned_ci_hi")
    p_val = row.get("best_emp_p_aligned")
    sig_str = "CI excludes 0" if row.get("best_ci_aligned_excludes_zero") else "CI includes 0"
    id_str = (f"{row['subject']} e{row['electrode_idx']} "
              f"{row['phoneme_pair']} · {row['word_end']}")
    if row.get("resampled") is not None:
        id_str += f" step {row['resampled']}"
    stat_str = ""
    if med_val is not None:
        stat_str = (f"  |  effect = {med_val:.3f} [{ci_lo_v:.3f}, {ci_hi_v:.3f}]"
                    f"  p = {p_val:.3f}  {sig_str}")
    ax.set_title(id_str + stat_str, fontsize=9)
    fig.tight_layout()
    return fig


def write_annotated_pdfs(
    entries: list[dict],
    per_window: pl.DataFrame,
    cell_keys: list[str],
    out_path: Path,
    epochs_dict: dict | None = None,
    pair_lookup: dict | None = None,
    ac_search_smin: int = 45,
    ac_search_smax: int = 68,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
    behav_dec_by_subject: dict | None = None,
    acoustic_per_window: pl.DataFrame | None = None,
    acoustic_R_plot: int = 200,
    acoustic_site_sig_windows: dict | None = None,
    step_tuning_df: pl.DataFrame | None = None,
) -> int:
    """Filtered-gallery PDF: regenerated star plot per cell.

    pair_lookup: optional dict keyed by (subject, electrode_idx, phoneme_pair) →
    b4_per_pair row dict. When provided, a colored bar is added showing the
    cross-WE pooled test result for that site/pair.

    acoustic_per_window: optional per-window summary for the acoustic-step contrast
    (b4_acoustic_per_window.parquet). When provided, adds an acoustic panel to
    each star plot (via matched_n_star_plot acoustic_* params).

    step_tuning_df: optional per-(cell, step, window_kind) summary
    (b4_step_tuning.parquet, from step_tuning_pass). When provided, adds
    step-tuning panel(s) (windowed mean HGA vs. acoustic step, all
    qualifying steps) to each star plot: one for window_kind=="global_best"
    rows and, if present, a second for window_kind=="late_excl_phon" rows.
    """
    if not entries or not HAS_PYPDF:
        return 0
    group_xlim: dict[tuple, float] = {}
    for row in entries:
        key = (row["subject"], row["electrode_idx"], row["phoneme_pair"])
        we_xlim = OFFSET_DICT.get(row["word_end"], 1.0) + 0.1
        group_xlim[key] = max(group_xlim.get(key, 0.0), we_xlim)
    writer = PdfWriter()
    n = 0
    for row in tqdm(entries):
        filt = (
            (pl.col("subject") == row["subject"])
            & (pl.col("electrode_idx") == row["electrode_idx"])
            & (pl.col("phoneme_pair") == row["phoneme_pair"])
            & (pl.col("word_end") == row["word_end"])
        )
        if "resampled" in cell_keys and row.get("resampled") is not None:
            filt = filt & (pl.col("resampled") == row["resampled"])
        site_pw = per_window.filter(filt) if per_window.height else pl.DataFrame()

        sig_wins = None
        mda = None
        if site_pw.height:
            pw_s = site_pw.sort("tmin")
            sig_list = [
                (float(r["tmin"]), float(r["tmax"]))
                for r in pw_s.filter(pl.col("ci_aligned_excludes_zero")).iter_rows(named=True)
            ]
            sig_wins = sig_list or None
            mda = {
                "tcenter": ((pw_s["tmin"] + pw_s["tmax"]) / 2).to_numpy().astype(float),
                "mean": pw_s["mean_diff_aligned_mean"].to_numpy().astype(float),
                "ci_lo": pw_s["mean_diff_aligned_ci_lo"].to_numpy().astype(float),
                "ci_hi": pw_s["mean_diff_aligned_ci_hi"].to_numpy().astype(float),
            }

        # Extract acoustic panel data when acoustic_per_window is available.
        ac_sig_wins = None
        ac_mda = None
        ac_extreme_steps = None
        if acoustic_per_window is not None and acoustic_per_window.height > 0:
            ac_filt = (
                (pl.col("subject") == row["subject"])
                & (pl.col("electrode_idx") == row["electrode_idx"])
                & (pl.col("phoneme_pair") == row["phoneme_pair"])
                & (pl.col("word_end") == row["word_end"])
            )
            ac_pw = acoustic_per_window.filter(ac_filt)
            if ac_pw.height > 0:
                ac_pw_s = ac_pw.sort("tmin")
                ac_sig_list = [
                    (float(r["tmin"]), float(r["tmax"]))
                    for r in ac_pw_s.filter(
                        pl.col("ci_aligned_excludes_zero")
                    ).iter_rows(named=True)
                ]
                ac_sig_wins = ac_sig_list or None
                ac_mda = {
                    "tcenter": ((ac_pw_s["tmin"] + ac_pw_s["tmax"]) / 2).to_numpy().astype(float),
                    "mean": ac_pw_s["mean_diff_aligned_mean"].to_numpy().astype(float),
                    "ci_lo": ac_pw_s["mean_diff_aligned_ci_lo"].to_numpy().astype(float),
                    "ci_hi": ac_pw_s["mean_diff_aligned_ci_hi"].to_numpy().astype(float),
                }
                # Extreme steps are a per-cell constant. Prefer per_window columns
                # if present, else fall back to the entry row (the acoustic cell
                # manifest carries s_lo/s_hi; per_window_summary drops them).
                if "s_lo" in ac_pw_s.columns and "s_hi" in ac_pw_s.columns:
                    ac_extreme_steps = (
                        int(ac_pw_s["s_lo"][0]),
                        int(ac_pw_s["s_hi"][0]),
                    )
                elif row.get("s_lo") is not None and row.get("s_hi") is not None:
                    ac_extreme_steps = (int(row["s_lo"]), int(row["s_hi"]))

        # Extract step-tuning panel data when step_tuning_df is available.
        # Two variants, disambiguated by `window_kind`: the unrestricted
        # global-best window, and the late window excluding overlap with the
        # site's acoustic-peak window (see exclude_overlapping_windows).
        step_tuning_rows = None
        step_tuning_window = None
        step_tuning_late_rows = None
        step_tuning_late_window = None
        if step_tuning_df is not None and step_tuning_df.height > 0:
            st_base_filt = (
                (pl.col("subject") == row["subject"])
                & (pl.col("electrode_idx") == row["electrode_idx"])
                & (pl.col("phoneme_pair") == row["phoneme_pair"])
                & (pl.col("word_end") == row["word_end"])
            )
            st_rows = step_tuning_df.filter(
                st_base_filt & (pl.col("window_kind") == "global_best")
            ).sort("step")
            if st_rows.height > 0:
                step_tuning_rows = st_rows.to_dicts()
                step_tuning_window = (
                    int(st_rows["best_smin"][0]), int(st_rows["best_smax"][0])
                )
            st_late_rows = step_tuning_df.filter(
                st_base_filt & (pl.col("window_kind") == "late_excl_phon")
            ).sort("step")
            if st_late_rows.height > 0:
                step_tuning_late_rows = st_late_rows.to_dicts()
                step_tuning_late_window = (
                    int(st_late_rows["best_smin"][0]), int(st_late_rows["best_smax"][0])
                )

        qs = row.get("qualifying_steps")
        can_regen = (
            epochs_dict is not None
            and row.get("subject") in epochs_dict
            and qs is not None
            and row.get("phon_smin") is not None
        )
        if not can_regen:
            print(f"  ⚠ skipping {row['subject']} e{row['electrode_idx']}: "
                  "cannot regenerate star plot (missing epochs or qualifying_steps)")
            continue
        if isinstance(qs, str):
            qs = [int(s) for s in qs.split(",") if s]
        key = (row["subject"], row["electrode_idx"], row["phoneme_pair"])
        try:
            behav_df = (behav_dec_by_subject or {}).get(row["subject"])
            top_sig = (acoustic_site_sig_windows or {}).get(
                (row["subject"], int(row["electrode_idx"]), row["phoneme_pair"])
            )
            fig2 = matched_n_star_plot(
                subject=row["subject"],
                electrode_idx=int(row["electrode_idx"]),
                phoneme_pair=row["phoneme_pair"],
                word_end=row["word_end"],
                qualifying_steps=list(qs),
                epochs_dict=epochs_dict,
                n_per_class=int(row["n_per_class"]),
                phon_smin=int(row["phon_smin"]),
                phon_smax=int(row["phon_smax"]),
                phon_search_smin=ac_search_smin,
                phon_search_smax=ac_search_smax,
                acoustic_peak_auc=row.get("acoustic_peak_auc"),
                sig_windows=sig_wins,
                mean_diff_arrays=mda,
                xlim=group_xlim[key],
                behav_decoding_df=behav_df,
                early_smax_s=ac_search_smax,
                top_sig_windows=top_sig,
                acoustic_mean_diff_arrays=ac_mda,
                acoustic_sig_windows=ac_sig_wins,
                acoustic_extreme_steps=ac_extreme_steps,
                acoustic_R_plot=acoustic_R_plot if ac_extreme_steps is not None else None,
                step_tuning=step_tuning_rows,
                step_tuning_window=step_tuning_window,
                step_tuning_extreme_steps=ac_extreme_steps,
                step_tuning_late=step_tuning_late_rows,
                step_tuning_late_window=step_tuning_late_window,
            )
            if pair_lookup is not None:
                pair_key_lut = (
                    row["subject"], int(row["electrode_idx"]), row["phoneme_pair"]
                )
                pr = pair_lookup.get(pair_key_lut)
                if pr is not None:
                    ax_bot = getattr(fig2, "_ax_behav", fig2.axes[1])
                    ymin, ymax = ax_bot.get_ylim()
                    bar_h = (ymax - ymin) * 0.04
                    bar_y = ymin + (ymax - ymin) * 0.90
                    pair_tmin = pr.get("pair_tmin")
                    pair_tmax = pr.get("pair_tmax")
                    pair_sig = bool(pr.get("pair_ci_excludes_zero", False))
                    emp_p = pr.get("pair_emp_p")
                    emp_p_str = f"{float(emp_p):.3f}" if emp_p is not None else "?"
                    if pair_tmin is not None and pair_tmax is not None:
                        bar_color = "#01665e" if pair_sig else "#c7eae5"
                        ax_bot.barh(
                            y=bar_y,
                            width=float(pair_tmax) - float(pair_tmin),
                            left=float(pair_tmin),
                            height=bar_h,
                            color=bar_color, alpha=0.85,
                            edgecolor="none", zorder=5,
                            label=f"cross-WE pooled (p={emp_p_str})",
                        )
                        ax_bot.legend(fontsize=7, loc="lower left", framealpha=0.7)
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="pdf", bbox_inches="tight")
            plt.close(fig2)
            buf2.seek(0)
            for page in PdfReader(buf2).pages:
                writer.add_page(page)
            n += 1
        except Exception as exc:
            print(f"  ⚠ star plot regen failed for {row['subject']} "
                  f"e{row['electrode_idx']}: {exc}")
    if n:
        with out_path.open("wb") as fh:
            writer.write(fh)
    return n
