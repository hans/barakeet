# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Strong-generator SCAN: same-word-end perceptual-reactivation diagnostic
#
# **Interactive/diagnostic, not part of the Snakemake pipeline. Runs on prod**
# (`b4_bootstrap.parquet` + `outputs/epochs_preprocessed/` are prod-only).
#
# The population version of `strong_generator_demo.py`: instead of one hand-picked
# `(subject, electrode_idx, phoneme_pair, word_end)` cell, it scans **every B4
# cell** in `b4_bootstrap.parquet` and, per cell, on the **same word end**, reports
# the two slopes that decide whether the late (integration) percept response
# re-expresses that word end's own unambiguous /d/–/n/ tuning:
#
# - **β_amb** (`p`) — the ambiguous within-completion percept slope, read straight
#   from `b4_bootstrap` (`mean_diff_raw`), fixed reference heard-/n/ − heard-/d/.
# - **β_unamb** (`â`) — the *same-window* endpoint step1-vs-step6 difference on the
#   **unambiguous** trials **of this same word end** (`bootstrap_endpoint_beta`),
#   same fixed /n/−/d/ reference. This IS the late acoustic template â, per word end.
# - **same_sign** — sign(β_amb) == sign(β_unamb): the **projection-pass proxy**.
#   A projection π = ⟨â, p⟩ > 0 is a *sign* alignment (magnitude-independent), so
#   it is STRICTLY WEAKER than the demo's strong-generator "CONSISTENT" verdict
#   (which additionally needs β_amb ≈ β_unamb in magnitude). Count same-sign cells,
#   not β_amb≈β_unamb cells, when asking "would this pass the projection test".
#
# This feeds the go/no-go (issue #17): does the same-word-end-â subset — Jon's
# predicted plurality/majority — actually exist, and does it clear a population
# null? The pooled TFCE gate (PR #14) came out near-null (6/187, 0 FDR) precisely
# because it pools across word ends; this scan is strictly per-word-end, so a
# cell that reactivates on only one completion (the interaction signature) is kept.
#
# ## The one design knob — per-cell window selection
#
# Each cell has a searchlight grid of windows in `b4_bootstrap`. "The integration
# response" needs one window per cell. `WINDOW_MODE`:
#
# - `"peak_beta_amb"` (default) — the grid window maximizing |median β_amb|.
#   ⚠️ **Circularity caveat:** selecting on β_amb inflates the count of
#   `beta_amb_reliable` cells (the window is chosen where the percept slope is
#   biggest). It does **NOT** bias the `same_sign` readout, because β_unamb is
#   computed independently from endpoint trials — so same-sign agreement at the
#   β_amb-peak window is a legitimate, non-circular statistic. Read "does â exist
#   and align" from `same_sign` / `beta_unamb_reliable`, and treat the
#   `beta_amb_reliable` count as optimistic.
# - `"all_windows"` — emit one row per (cell × grid window); no per-cell selection,
#   so nothing is circular, but multiple-comparison load is on the reader. Use this
#   for the eventual sliding-window projection with a permutation null (the proper
#   CPO version, deferred to the method spec).
#
# `LATE_ONLY` optionally restricts candidate windows to late/post-POD windows
# (center ≥ `LATE_MIN_S`) so "integration" isn't contaminated by early windows.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.viz_paper import epoch_sfreq, epoch_tmin

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))
from _within_completion import (  # noqa: E402
    beta_summary,
    bootstrap_endpoint_beta,
    extract_hga,
    resolve_behavior_col,
)

# %% tags=["parameters"]
# ── Data sources (prod-canonical outputs/; the raw bootstrap is not synced) ──
b4_bootstrap_path = "outputs/causal46_joined/t_tests/b4_bootstrap.parquet"
epoch_dir = "outputs/epochs_preprocessed"
out_path = "outputs/causal46_joined/strong_generator_scan/cell_reactivation.parquet"

# ── Window selection ─────────────────────────────────────────────────────────
WINDOW_MODE = "peak_beta_amb"   # {"peak_beta_amb", "all_windows"}
LATE_ONLY = True                # restrict candidate windows to center ≥ LATE_MIN_S
LATE_MIN_S = 0.30               # seconds post word onset (≈ POD for dn=0.295)

# ── Bootstrap / CI settings (match strong_generator_demo defaults) ───────────
R_UNAMB = 1000
CI_LOW, CI_HIGH = 2.5, 97.5
MIN_ENDPOINT_N = 3


# %%
def s_to_t(s) -> float:
    return s / epoch_sfreq + epoch_tmin


# %% [markdown]
# ## Load the raw per-replicate bootstrap and enumerate cells

# %%
if not Path(b4_bootstrap_path).exists():
    raise SystemExit(
        f"{b4_bootstrap_path} not found. This scan reads the RAW per-replicate "
        f"bootstrap (prod-only) — run on prod, or point b4_bootstrap_path at a "
        f"location where the pipeline output is present."
    )
b4 = pl.read_parquet(b4_bootstrap_path)

cell_keys = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
cells = b4.select(cell_keys).unique().sort(cell_keys)
print(f"{cells.height} B4 cells across {cells['subject'].n_unique()} subjects")

# Candidate late windows shared by the whole file (single searchlight grid).
grid = b4.select(["smin", "smax"]).unique().sort("smin")
grid_centers_s = np.asarray(
    [s_to_t(0.5 * (r["smin"] + r["smax"])) for r in grid.iter_rows(named=True)]
)
if LATE_ONLY:
    keep = grid_centers_s >= LATE_MIN_S
    grid = grid.filter(pl.Series(keep))
    print(f"LATE_ONLY: {int(keep.sum())}/{keep.size} grid windows kept "
          f"(center ≥ {LATE_MIN_S}s)")
grid_windows = [(int(r["smin"]), int(r["smax"])) for r in grid.iter_rows(named=True)]


# %% [markdown]
# ## Scan: per cell, β_amb (from bootstrap) + β_unamb (same-window endpoints)
#
# Epochs are loaded once per subject (the expensive step) and reused for every
# electrode/word_end of that subject.

# %%
import mne  # noqa: E402
from src.data import add_metadata_features  # noqa: E402

rows: list[dict] = []
subjects = cells["subject"].unique().to_list()

for subj in subjects:
    ep_path = Path(epoch_dir) / f"{subj}_epo.fif"
    if not ep_path.exists():
        print(f"  [skip] {subj}: no epochs at {ep_path}")
        continue
    ep_full = mne.read_epochs(str(ep_path), preload=True, verbose="WARNING")
    ep_full.metadata = add_metadata_features(ep_full.metadata.copy())
    md = ep_full.metadata
    _ = resolve_behavior_col(md)  # validate the metadata has a usable behavior col

    subj_cells = cells.filter(pl.col("subject") == subj)
    for pp in subj_cells["phoneme_pair"].unique().to_list():
        pp_mask = (md["phoneme_pair"] == pp).values
        ep_pp = ep_full[pp_mask]
        md_pp = md[pp_mask].reset_index(drop=True)

        pp_cells = subj_cells.filter(pl.col("phoneme_pair") == pp)
        for elec in pp_cells["electrode_idx"].unique().to_list():
            hga = extract_hga(ep_pp, int(elec))
            elec_cells = pp_cells.filter(pl.col("electrode_idx") == elec)
            for we in elec_cells["word_end"].unique().to_list():
                cell_boot = b4.filter(
                    (pl.col("subject") == subj)
                    & (pl.col("electrode_idx") == elec)
                    & (pl.col("phoneme_pair") == pp)
                    & (pl.col("word_end") == we)
                )
                # endpoint trial counts within this word end (incl. incongruent
                # endpoints, e.g. step1+ecessary / step6+esolate) — is â estimable?
                we_mask = (md_pp["word_end"] == we).values
                n_lo = int((we_mask & (md_pp["resampled"] == 1).values).sum())
                n_hi = int((we_mask & (md_pp["resampled"] == 6).values).sum())

                # candidate windows present for this cell in the (late) grid
                cell_windows = [
                    (smin, smax) for (smin, smax) in grid_windows
                    if cell_boot.filter(
                        (pl.col("smin") == smin) & (pl.col("smax") == smax)
                    ).height > 0
                ]
                if not cell_windows:
                    continue

                if WINDOW_MODE == "peak_beta_amb":
                    # pick window with max |median β_amb|
                    best = None
                    for (smin, smax) in cell_windows:
                        arr = cell_boot.filter(
                            (pl.col("smin") == smin) & (pl.col("smax") == smax)
                        )["mean_diff_raw"].to_numpy()
                        arr = arr[np.isfinite(arr)]
                        if arr.size == 0:
                            continue
                        med = abs(float(np.median(arr)))
                        if best is None or med > best[0]:
                            best = (med, smin, smax)
                    if best is None:
                        continue
                    chosen = [(best[1], best[2])]
                elif WINDOW_MODE == "all_windows":
                    chosen = cell_windows
                else:
                    raise ValueError(f"unknown WINDOW_MODE={WINDOW_MODE!r}")

                for (smin, smax) in chosen:
                    amb_arr = cell_boot.filter(
                        (pl.col("smin") == smin) & (pl.col("smax") == smax)
                    )["mean_diff_raw"].to_numpy()
                    b_amb = beta_summary(amb_arr, CI_LOW, CI_HIGH)

                    unamb_arr = bootstrap_endpoint_beta(
                        hga, md_pp, word_end=we, smin=smin, smax=smax,
                        R=R_UNAMB, min_n=MIN_ENDPOINT_N,
                    )
                    b_unamb = (beta_summary(unamb_arr, CI_LOW, CI_HIGH)
                               if unamb_arr is not None
                               else dict(med=np.nan, ci_low=np.nan, ci_high=np.nan,
                                         reliable=False, sign=0, n=0))

                    same_sign = (
                        b_amb["sign"] != 0 and b_amb["sign"] == b_unamb["sign"]
                    )
                    rows.append(dict(
                        subject=subj, electrode_idx=int(elec), phoneme_pair=pp,
                        word_end=we, smin=int(smin), smax=int(smax),
                        tmin_s=float(s_to_t(smin)), tmax_s=float(s_to_t(smax)),
                        n_endpoint_step1=n_lo, n_endpoint_step6=n_hi,
                        beta_amb=b_amb["med"], beta_amb_reliable=b_amb["reliable"],
                        beta_amb_sign=b_amb["sign"],
                        beta_unamb=b_unamb["med"],
                        beta_unamb_reliable=b_unamb["reliable"],
                        beta_unamb_sign=b_unamb["sign"],
                        beta_unamb_estimable=(unamb_arr is not None),
                        same_sign=bool(same_sign),
                    ))

scan = pl.DataFrame(rows)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
scan.write_parquet(out_path)
print(f"\nwrote {scan.height} rows → {out_path}")


# %% [markdown]
# ## Population readout — the go/no-go numbers (issue #17)
#
# In `peak_beta_amb` mode there is exactly one row per cell.

# %%
from math import comb  # noqa: E402


def binom_upper_p(k: int, n: int, p: float = 0.5) -> float:
    """P(X ≥ k) for X ~ Binomial(n, p)."""
    if n <= 0:
        return float("nan")
    return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


if scan.height == 0:
    print("no rows — check WINDOW_MODE / LATE_ONLY / data availability")
else:
    n_cells = scan.height
    n_estimable = int(scan["beta_unamb_estimable"].sum())     # enough endpoint trials to estimate â
    n_a_hat = int(scan["beta_unamb_reliable"].sum())          # â exists (late, per word end)
    n_p = int(scan["beta_amb_reliable"].sum())                # reliable late percept slope

    # ── HEADLINE: sign agreement within the INTEGRATION-RESPONSE population ──
    # The go/no-go (issue #17) is about the cells that HAVE a late integration
    # response, not all 187 cells. A binomial over every cell is diluted by the
    # ~majority of noise cells that sign-agree ~50% by construction and pins the
    # test near-null regardless of the real effect — the wrong population.
    #
    # Non-circular conditioning: `beta_amb_reliable` is the ENTRY criterion
    # ("is there an integration response here"), `beta_unamb_estimable` is a
    # data-quantity gate, and the TESTED quantity is sign agreement — β_unamb is
    # computed independently of β_amb, so its sign isn't set by the entry rule.
    # ⚠ Caveat for #17: in WINDOW_MODE="peak_beta_amb" the window was picked by
    # |β_amb|, so `beta_amb_reliable` as the SUBSET SELECTOR is optimistic (the
    # clean version selects windows independently, e.g. from b_windows.parquet).
    # The sign-agreement WITHIN the subset does not inherit that bias.
    subset = scan.filter(pl.col("beta_amb_reliable") & pl.col("beta_unamb_estimable"))
    n_sub = subset.height
    n_same_sub = int(subset["same_sign"].sum())
    p_sub = binom_upper_p(n_same_sub, n_sub)
    n_pass = int((subset["beta_unamb_reliable"] & subset["same_sign"]).sum())

    # diluted whole-population number, kept only for reference (NOT the go/no-go)
    n_same_all = int(scan["same_sign"].sum())
    base_all = int((scan["beta_amb_sign"] != 0).sum())
    p_all = binom_upper_p(n_same_all, base_all)

    print(f"cells scanned                                       : {n_cells}")
    print(f"â estimable (≥{MIN_ENDPOINT_N} endpoint trials/step)             : {n_estimable}")
    print(f"â exists  (β_unamb reliable)                        : {n_a_hat}")
    print(f"integration responses (β_amb reliable & â estimable): {n_sub}   <- go/no-go population")
    print("")
    print(f"HEADLINE  same-sign within integration responses    : {n_same_sub} / {n_sub}")
    print(f"          Binomial({n_sub}, 0.5)  P(≥{n_same_sub})               : {p_sub:.4g}")
    print(f"          reactivation pass (+ β_unamb reliable)     : {n_pass} / {n_sub}")
    print("")
    print(f"(reference, diluted by noise cells — NOT the test)  : "
          f"same-sign {n_same_all}/{base_all}, binom p {p_all:.4g}")
    print("NOTE: Binomial(0.5) is a first-cut null; the principled CPO null is a "
          "within-step label-permutation of the percept split (deferred to #17).")
