"""Compare three speech-responsive screens on the preprocessed epochs.

Three flags per electrode:
  * `amp_flag`        — causal4 criterion: max|baselined evoked| in [0, 0.9s] > 0.3
  * `full_flag`       — causal5/6 production: paired t-test, post-window=[0, tmax],
                        threshold t > 7
  * `refined_pos`     — same paired t-test, post-window=[0, POST_TMAX_S], t > +7
  * `refined_abs`     — same paired t-test, post-window=[0, POST_TMAX_S], |t| > 7

The point is to attribute the causal4-vs-causal5/6 asymmetry to (a) post-window
length and (b) one-sidedness independently.

Run via the local venv:
    .venv/bin/python scripts/refined_speech_responsive.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import polars as pl
from scipy.stats import ttest_rel

EPOCHS_DIR = Path("outputs/epochs_preprocessed")
OUTDIR = Path("outputs/causal6/refined_speech_responsive")
OUTDIR.mkdir(parents=True, exist_ok=True)

POST_TMAX_S = 0.6
T_THRESHOLD = 7.0
AMP_THRESHOLD = 0.3
AMP_TMAX_S = 0.9

ALL_SUBJECTS = [
    "EC243", "EC248", "EC250", "EC253", "EC260",
    "EC270", "EC278", "EC279", "EC282", "EC287",
]
PLOT_SUBJECTS = ["EC270", "EC279"]
N_SHOW_PER_SUBJECT = 10


def screens_for_subject(subj: str) -> pl.DataFrame:
    p = EPOCHS_DIR / f"{subj}_epo.fif"
    if not p.exists():
        print(f"  [skip] {subj}: no epochs at {p}")
        return pl.DataFrame()

    epochs = mne.read_epochs(p, verbose="error")
    n_chans = len(epochs.info["ch_names"])

    # --- causal4 amplitude criterion (matches notebooks/causal4/find_speech_responsive.py)
    evoked = (
        epochs.copy()
        .apply_baseline((-0.1, 0))
        .average()
        .crop(tmin=0, tmax=AMP_TMAX_S)
        .get_data()
    )
    amp_value = np.abs(evoked).max(axis=1)
    amp_flag = amp_value > AMP_THRESHOLD

    # --- causal5/6 t-test criterion (matches notebooks/causal5/find_speech_responsive.py)
    data = epochs.get_data()  # (n_epochs, n_chans, n_times)
    s_pre_lo, s_pre_hi = epochs.time_as_index([epochs.tmin, 0])
    s_full_lo, s_full_hi = epochs.time_as_index([0, epochs.tmax])
    pre_means = data[:, :, s_pre_lo:s_pre_hi].mean(axis=2)
    post_means_full = data[:, :, s_full_lo:s_full_hi].mean(axis=2)
    t_full, _ = ttest_rel(post_means_full, pre_means, axis=0)
    full_flag = t_full > T_THRESHOLD

    # --- refined: short post-window, two-sided
    s_short_lo, s_short_hi = epochs.time_as_index([0, POST_TMAX_S])
    post_means_short = data[:, :, s_short_lo:s_short_hi].mean(axis=2)
    t_short, _ = ttest_rel(post_means_short, pre_means, axis=0)
    refined_pos = t_short > T_THRESHOLD
    refined_abs = np.abs(t_short) > T_THRESHOLD

    return pl.DataFrame({
        "subject": [subj] * n_chans,
        "electrode_idx": np.arange(n_chans, dtype=np.int64),
        "amp_value": amp_value[:n_chans].astype(np.float64),
        "amp_flag": amp_flag[:n_chans].astype(bool),
        "t_full": t_full[:n_chans].astype(np.float64),
        "full_flag": full_flag[:n_chans].astype(bool),
        "t_short": t_short[:n_chans].astype(np.float64),
        "refined_pos_flag": refined_pos[:n_chans].astype(bool),
        "refined_abs_flag": refined_abs[:n_chans].astype(bool),
    })


def build_table() -> pl.DataFrame:
    frames = []
    for s in ALL_SUBJECTS:
        df = screens_for_subject(s)
        if not df.is_empty():
            frames.append(df)
            print(f"  {s}: {df.height} channels")
    return pl.concat(frames) if frames else pl.DataFrame()


def print_overall_counts(df: pl.DataFrame) -> None:
    total = df.height
    print("\n=== Overall counts ===")
    for col in ["amp_flag", "full_flag", "refined_pos_flag", "refined_abs_flag"]:
        n = df.filter(pl.col(col)).height
        print(f"  {col:>18}: {n:5d} / {total} = {n/total:.1%}")


def print_pairwise(df: pl.DataFrame) -> None:
    print("\n=== Pairwise overlap (amp_flag vs each other criterion) ===")
    print(f"  total channels = {df.height}")
    for col in ["full_flag", "refined_pos_flag", "refined_abs_flag"]:
        both    = df.filter( pl.col("amp_flag") &  pl.col(col)).height
        only_a  = df.filter( pl.col("amp_flag") & ~pl.col(col)).height
        only_b  = df.filter(~pl.col("amp_flag") &  pl.col(col)).height
        neither = df.filter(~pl.col("amp_flag") & ~pl.col(col)).height
        print(f"  amp vs {col:<18}: both={both:4d}  only_amp={only_a:4d}  "
              f"only_other={only_b:4d}  neither={neither:5d}")


def print_recovery(df: pl.DataFrame) -> None:
    """How many of the 'only-causal4' (amp & ¬full) sites get rescued by the refinements?"""
    only_amp = df.filter(pl.col("amp_flag") & ~pl.col("full_flag"))
    only_amp_n = only_amp.height
    print(f"\n=== Refinement-recovery from the only-causal4 set "
          f"(amp_flag & ¬full_flag, n={only_amp_n}) ===")
    if only_amp_n == 0:
        print("  (no such sites)")
        return
    rec_pos = only_amp.filter(pl.col("refined_pos_flag")).height
    rec_abs = only_amp.filter(pl.col("refined_abs_flag")).height
    print(f"  recovered by refined_pos (post=[0,{POST_TMAX_S}s], t>+{T_THRESHOLD}): "
          f"{rec_pos:4d}  ({rec_pos/only_amp_n:.1%})")
    print(f"  recovered by refined_abs (post=[0,{POST_TMAX_S}s], |t|>{T_THRESHOLD}): "
          f"{rec_abs:4d}  ({rec_abs/only_amp_n:.1%})")
    # The added-by-suppression set — sites refined_abs catches that refined_pos doesn't.
    sup = only_amp.filter(pl.col("refined_abs_flag") & ~pl.col("refined_pos_flag"))
    print(f"  added by two-sidedness only (suppression-like): {sup.height:4d}")


def print_per_subject(df: pl.DataFrame) -> None:
    print("\n=== Per-subject (counts + recovery within only-c4) ===")
    rows = []
    for subj_tup, grp in df.group_by("subject", maintain_order=True):
        subj = subj_tup[0]
        a = grp.filter(pl.col("amp_flag")).height
        f = grp.filter(pl.col("full_flag")).height
        rp = grp.filter(pl.col("refined_pos_flag")).height
        ra = grp.filter(pl.col("refined_abs_flag")).height
        only_amp = grp.filter(pl.col("amp_flag") & ~pl.col("full_flag")).height
        rec_abs = grp.filter(
            pl.col("amp_flag") & ~pl.col("full_flag") & pl.col("refined_abs_flag")
        ).height
        rec_pos = grp.filter(
            pl.col("amp_flag") & ~pl.col("full_flag") & pl.col("refined_pos_flag")
        ).height
        rows.append((subj, grp.height, a, f, rp, ra, only_amp, rec_pos, rec_abs))
    hdr = (f"  {'subj':>6}  {'n':>4}  {'amp':>4}  {'full':>4}  "
           f"{'rfP':>4}  {'rfA':>4}  {'only_c4':>7}  {'recP':>4}  {'recA':>4}")
    print(hdr)
    for row in rows:
        print(f"  {row[0]:>6}  {row[1]:4d}  {row[2]:4d}  {row[3]:4d}  "
              f"{row[4]:4d}  {row[5]:4d}  {row[6]:7d}  {row[7]:4d}  {row[8]:4d}")


def plot_only_amp_sites(df: pl.DataFrame, subj: str, n_show: int) -> None:
    """For top-amplitude only-c4 sites: trial-stack evoked, per-trial pre/post scatter,
    per-trial Δ histograms (full vs short post-window)."""
    p = EPOCHS_DIR / f"{subj}_epo.fif"
    if not p.exists():
        print(f"  [skip plot] {subj}: no epochs")
        return
    only_c4 = df.filter(
        (pl.col("subject") == subj) & pl.col("amp_flag") & ~pl.col("full_flag")
    ).sort("amp_value", descending=True).head(n_show)
    if only_c4.is_empty():
        print(f"  [skip plot] {subj}: no only-c4 sites")
        return

    epochs = mne.read_epochs(p, verbose="error")
    times = epochs.times
    data = epochs.get_data()
    evoked_b = epochs.copy().apply_baseline((-0.1, 0)).average().get_data()

    s_pre_lo, s_pre_hi = epochs.time_as_index([epochs.tmin, 0])
    s_full_lo, s_full_hi = epochs.time_as_index([0, epochs.tmax])
    s_short_lo, s_short_hi = epochs.time_as_index([0, POST_TMAX_S])

    n = only_c4.height
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.2 * n), squeeze=False)
    n_trial_overlay = min(data.shape[0], 200)

    for row, rec in enumerate(only_c4.iter_rows(named=True)):
        ei = rec["electrode_idx"]
        ax_e, ax_s, ax_h = axes[row]

        for tr in range(n_trial_overlay):
            ax_e.plot(times, data[tr, ei, :], color="grey", alpha=0.04, lw=0.4)
        ax_e.plot(times, evoked_b[ei], color="firebrick", lw=1.4,
                  label="baselined evoked mean")
        ax_e.axvline(0, color="k", lw=0.5)
        ax_e.axvline(POST_TMAX_S, color="b", ls="--", lw=0.6,
                     label=f"refined post tmax={POST_TMAX_S}s")
        ax_e.axvline(AMP_TMAX_S, color="r", ls=":", lw=0.6,
                     label=f"amp post tmax={AMP_TMAX_S}s")
        ax_e.axhline(AMP_THRESHOLD, color="r", ls=":", lw=0.4)
        ax_e.axhline(-AMP_THRESHOLD, color="r", ls=":", lw=0.4)
        ax_e.set_xlim(times.min(), times.max())
        ax_e.set_title(
            f"{subj} e{ei}  amp={rec['amp_value']:.2f}  "
            f"t_full={rec['t_full']:+.2f}  t_short={rec['t_short']:+.2f}",
            fontsize=11,
        )
        if row == 0:
            ax_e.legend(fontsize=8, loc="upper right")

        pre = data[:, ei, s_pre_lo:s_pre_hi].mean(axis=1)
        post_short = data[:, ei, s_short_lo:s_short_hi].mean(axis=1)
        post_full = data[:, ei, s_full_lo:s_full_hi].mean(axis=1)
        ax_s.scatter(pre, post_short, s=6, alpha=0.4, label="post[0,0.6s]")
        ax_s.scatter(pre, post_full, s=6, alpha=0.4, label="post[0,tmax]")
        lo = float(np.nanmin([pre.min(), post_short.min(), post_full.min()]))
        hi = float(np.nanmax([pre.max(), post_short.max(), post_full.max()]))
        ax_s.plot([lo, hi], [lo, hi], "k--", lw=0.5)
        ax_s.set_xlabel("pre-mean per trial")
        ax_s.set_ylabel("post-mean per trial")
        ax_s.set_title(
            f"per-trial pre vs post  short t={rec['t_short']:+.2f}", fontsize=10
        )
        if row == 0:
            ax_s.legend(fontsize=8)

        delta_short = post_short - pre
        delta_full = post_full - pre
        bins = 30
        ax_h.hist(delta_full, bins=bins, alpha=0.55,
                  label=f"Δ full  med={np.median(delta_full):+.3f}")
        ax_h.hist(delta_short, bins=bins, alpha=0.55,
                  label=f"Δ short med={np.median(delta_short):+.3f}")
        ax_h.axvline(0, color="k", lw=0.5)
        ax_h.set_xlabel("post − pre per trial")
        ax_h.set_title("per-trial Δ (full vs short post-window)", fontsize=10)
        if row == 0:
            ax_h.legend(fontsize=8)

    fig.suptitle(
        f"{subj}: top-{n} only-c4 sites (amp & ¬full t>7) — per-trial diagnostics",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUTDIR / f"{subj}_only_amp_sites.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    print(f"Building screens table for {len(ALL_SUBJECTS)} subjects...")
    df = build_table()
    if df.is_empty():
        print("No subjects loaded — nothing to do.")
        return
    out_pq = OUTDIR / "all_screens.parquet"
    out_csv = OUTDIR / "all_screens.csv"
    df.write_parquet(out_pq)
    df.write_csv(out_csv)
    print(f"\nWrote {out_pq} and {out_csv}")

    print_overall_counts(df)
    print_pairwise(df)
    print_recovery(df)
    print_per_subject(df)

    print(f"\nDiagnostic plots for: {PLOT_SUBJECTS}")
    for subj in PLOT_SUBJECTS:
        plot_only_amp_sites(df, subj, N_SHOW_PER_SUBJECT)


if __name__ == "__main__":
    main()
