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
# # Calibrate max_iter for permutation-null runs
#
# Validates Tier-1 #1 from the perf review: the production setting
# `max_iter=50` is oversized for permutation refits because under permuted
# labels with `reg_lambda=1.0` Newton converges in well under 15 iters.
# The kernel deliberately avoids early-exit syncs, so it pays for every
# unused iteration.
#
# Two diagnostics:
#   A. **Convergence histogram.** Call `fit_batched_l2_logreg` directly on
#      a representative slice (one phoneme_pair, one fold, K perms) with
#      `max_iter=50`. Capture the per-problem `n_iter` and `converged`
#      flags. Report the cdf — what fraction has converged by each
#      candidate cutoff.
#   B. **AUC stability + wall-time sweep.** Run the full
#      `run_acoustic_searchlight_permutations` with a small K at several
#      `max_iter` values. Compare per-problem AUC distributions to the
#      max_iter=50 reference; report max abs deviation and timing.
#
# Recommended cutoff = smallest value where AUCs match the reference to
# tolerance (default 1e-4) AND >=99% of problems converged in diagnostic A.
#
# Run (one subject, ~30-60 min on a GPU node depending on K):
#   conda activate /scratch/jgauthier/transformers3
#   jupytext --to notebook --execute notebooks/causal6/calibrate_max_iter_for_null.py

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %%
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.model_selection import StratifiedKFold

# %%
from src.data import add_metadata_features
from src.models.causal6 import (
    _resolve_target,
    make_windows,
    run_acoustic_searchlight_permutations,
)
from src.models.decoding_gpu import (
    compute_balanced_sample_weight,
    fit_batched_l2_logreg,
    standardise_per_batch,
)

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
outdir = "outputs/causal6/calibrate_max_iter_for_null"

min_sample = 1
window_size = 15
stride = 2

target = "categorical_acoustic_cue"
reg_lambda = 1.0
n_folds = 5
cv_random_state = 42
device = "cuda"
tol = 1e-6

# Diagnostic A (direct kernel) — small K, single fold, capture n_iter.
diag_K = 200

# Diagnostic B (full pipeline AUC sweep) — K traded off vs total runtime.
# At K=200 each run is ~ (200/10000) * (production wallclock) ≈ 24min/subject;
# four max_iter values ≈ 90-120min total.
sweep_K = 200
sweep_max_iters = [10, 15, 25, 50]
sweep_reference_max_iter = 50  # AUCs from this run are the ground truth
permutation_chunk_size = 6

# Acceptance thresholds for the recommendation.
auc_tol = 1e-4         # max abs AUC deviation vs reference
converged_frac_min = 0.99  # min fraction of problems that hit `tol` at the cutoff

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# %%
electrode_df = pd.read_csv(electrodes_path)
speech_responsive_idxs = sorted(
    electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"].unique().astype(int)
)

epochs = mne.read_epochs(epochs_path, verbose=False)
assert epochs.metadata is not None
epochs.metadata = add_metadata_features(epochs.metadata)

max_sample = epochs.times.shape[0]
windows = make_windows(min_sample, max_sample, window_size, stride)
n_electrodes = len(speech_responsive_idxs)
n_windows = windows.shape[0]
print(f"[{subject}] {n_electrodes} electrodes × {n_windows} windows = "
      f"{n_electrodes * n_windows} problems per phoneme_pair")

# %% [markdown]
# ## Diagnostic A — convergence histogram
#
# Replicates the inner loop of `_fit_batched_cv_permutations` for one
# phoneme_pair and one fold, with `max_iter=50`, capturing the kernel's
# returned `n_iter` and `converged` per (perm, problem). The kernel only
# increments `n_iter` while `active`, so it equals the iteration at which
# convergence was detected (or 50 if never).

# %%
md = epochs.metadata
phoneme_pairs = sorted(md.phoneme_pair.dropna().unique())
resampled_mask = md.resampled.isin((1, 6)).values

# First phoneme_pair with enough data — same selection logic as the
# production permutation entry point.
diag_pp = None
for pp in phoneme_pairs:
    if ((md.phoneme_pair == pp).values & resampled_mask).sum() > 0:
        diag_pp = pp
        break
assert diag_pp is not None
print(f"Diagnostic A using phoneme_pair={diag_pp}")

selection = (md.phoneme_pair == diag_pp).values & resampled_mask
y_real = _resolve_target(md, target, diag_pp, selection)
X_full = epochs.get_data(picks=list(speech_responsive_idxs))
X_sel = X_full[selection]
n_trials = X_sel.shape[0]

# Build the (n_trials, B, win_size) batch the same way the production code does.
B = n_electrodes * n_windows
win_size = int(windows[0, 1] - windows[0, 0])
X_batch = np.empty((n_trials, B, win_size), dtype=np.float64)
b = 0
for e_idx in range(n_electrodes):
    for smin, smax in windows:
        X_batch[:, b, :] = X_sel[:, e_idx, smin:smax]
        b += 1

# One representative training fold from the real labels.
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)
train_idx, test_idx = next(iter(skf.split(np.zeros(n_trials), y_real)))
n_tr = len(train_idx)

# Permuted labels (same generation scheme as the production code).
rng_seeds = list(range(diag_K))
y_perms = np.empty((diag_K, n_trials), dtype=np.int64)
for k, seed in enumerate(rng_seeds):
    y_perms[k] = np.random.default_rng(int(seed)).permutation(y_real.astype(np.int64))

dtype = torch.float32
X_gpu = torch.tensor(X_batch.transpose(1, 0, 2).copy(), dtype=dtype, device=device)
train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
mask_tr_b = torch.ones(B, n_tr, dtype=dtype, device=device)
X_train_t = X_gpu.index_select(1, train_idx_t)
X_train_std, _, _, _ = standardise_per_batch(X_train_t, mask_tr_b, X_train_t)

n_iter_records = []
converged_records = []
for k in range(diag_K):
    y_train_k = torch.as_tensor(
        y_perms[k, train_idx], dtype=dtype, device=device
    ).unsqueeze(0).expand(B, n_tr).contiguous()
    sw_k = compute_balanced_sample_weight(y_train_k, mask_tr_b)
    _, n_iter_b, conv_b = fit_batched_l2_logreg(
        X_train_std, y_train_k, mask_tr_b, sw_k,
        reg_lambda=reg_lambda, tol=tol, max_iter=50,
    )
    n_iter_records.append(n_iter_b.cpu().numpy())
    converged_records.append(conv_b.cpu().numpy())

n_iter_arr = np.stack(n_iter_records)        # (diag_K, B)
converged_arr = np.stack(converged_records)  # (diag_K, B)

# Per-cutoff convergence fraction.
cutoffs = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
print("\nConvergence-by-cutoff (fraction of perm × problem fits that hit "
      f"|g|/|g_0| < tol={tol:g} by iteration k):")
print(f"  total fits: {n_iter_arr.size}")
for c in cutoffs:
    frac_done = float(((n_iter_arr <= c) & converged_arr).mean())
    print(f"    k={c:>3d}: {frac_done:.4f}")

print(f"\nn_iter summary: mean={n_iter_arr.mean():.2f}, "
      f"median={np.median(n_iter_arr):.0f}, "
      f"p95={np.percentile(n_iter_arr, 95):.0f}, "
      f"p99={np.percentile(n_iter_arr, 99):.0f}, "
      f"max={n_iter_arr.max():d}")
print(f"final converged fraction: {converged_arr.mean():.4f}")

# Persist for later inspection.
diag_a_path = outdir / "diagnostic_a_n_iter.parquet"
pl.DataFrame({
    "permutation_idx": np.repeat(np.arange(diag_K, dtype=np.int64), B),
    "problem_idx": np.tile(np.arange(B, dtype=np.int64), diag_K),
    "n_iter": n_iter_arr.reshape(-1).astype(np.int32),
    "converged": converged_arr.reshape(-1),
}).write_parquet(diag_a_path)
print(f"Wrote {diag_a_path}")

# %% [markdown]
# ## Diagnostic B — AUC stability across max_iter
#
# Runs the full `run_acoustic_searchlight_permutations` (same code path
# the production rule uses) at multiple `max_iter` values, with a small K
# for tractable wall time. The max_iter=50 run is the reference; for
# every other value we report:
#   - `max_abs_diff`: max |auc - auc_ref| across all (perm × fold ×
#     problem) tuples, and the same per-problem max
#   - `mean_abs_diff`
#   - `corr`: Pearson correlation of auc to auc_ref
#   - `wallclock_s` and speedup vs reference.
#
# Speedup at this K extrapolates linearly to production K=10000 because
# the GPU kernel time per chunk scales with `max_iter` (Newton iters ×
# (1 + ls_max_halvings) forward passes per iter).

# %%
permute_seeds = list(range(0, sweep_K))
sweep_results = {}
for mi in sweep_max_iters:
    print(f"\n--- max_iter = {mi} (K={sweep_K}) ---")
    t0 = time.time()
    null_scores = run_acoustic_searchlight_permutations(
        epochs, subject=subject,
        electrode_idxs=speech_responsive_idxs,
        windows=windows,
        reg_lambda=reg_lambda,
        permute_seeds=permute_seeds,
        permutation_chunk_size=permutation_chunk_size,
        target=target,
        n_folds=n_folds, cv_random_state=cv_random_state,
        device=device, dtype=torch.float32,
        tol=tol, max_iter=mi,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    sweep_results[mi] = {
        "scores": null_scores,
        "wallclock_s": elapsed,
    }
    print(f"  wallclock: {elapsed:.1f}s, rows: {null_scores.height}")
    null_scores.write_parquet(outdir / f"null_scores_max_iter_{mi}.parquet")

# %%
ref_mi = sweep_reference_max_iter
ref = sweep_results[ref_mi]["scores"].rename({"test_roc_auc": "auc_ref"}).select(
    ["phoneme_pair", "electrode_idx", "smin", "smax",
     "fold", "permutation_idx", "auc_ref"]
)

summary_rows = []
for mi, r in sweep_results.items():
    if mi == ref_mi:
        summary_rows.append({
            "max_iter": mi,
            "wallclock_s": r["wallclock_s"],
            "speedup_vs_ref": 1.0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "corr": 1.0,
            "max_per_problem_abs_diff_p99": 0.0,
            "n_compared": ref.height,
        })
        continue
    joined = r["scores"].join(
        ref,
        on=["phoneme_pair", "electrode_idx", "smin", "smax", "fold", "permutation_idx"],
        how="inner",
    ).with_columns(
        (pl.col("test_roc_auc") - pl.col("auc_ref")).abs().alias("absdiff")
    )
    a = joined["test_roc_auc"].to_numpy()
    b = joined["auc_ref"].to_numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    a_f, b_f = a[finite], b[finite]
    absdiff = np.abs(a_f - b_f)
    # Per-problem worst-case absdiff (which problems are most sensitive?).
    problem_keys = ["phoneme_pair", "electrode_idx", "smin", "smax"]
    per_problem_max = (
        joined.filter(pl.col("test_roc_auc").is_finite() & pl.col("auc_ref").is_finite())
        .group_by(problem_keys)
        .agg(pl.col("absdiff").max().alias("max_absdiff"))["max_absdiff"]
        .to_numpy()
    )
    summary_rows.append({
        "max_iter": mi,
        "wallclock_s": r["wallclock_s"],
        "speedup_vs_ref": sweep_results[ref_mi]["wallclock_s"] / r["wallclock_s"],
        "max_abs_diff": float(absdiff.max()) if absdiff.size else float("nan"),
        "mean_abs_diff": float(absdiff.mean()) if absdiff.size else float("nan"),
        "corr": float(np.corrcoef(a_f, b_f)[0, 1]) if a_f.size > 1 else float("nan"),
        "max_per_problem_abs_diff_p99": float(np.percentile(per_problem_max, 99))
            if per_problem_max.size else float("nan"),
        "n_compared": int(finite.sum()),
    })

summary = pl.DataFrame(summary_rows).sort("max_iter")
print("\n=== Diagnostic B summary ===")
print(summary)
summary.write_parquet(outdir / "diagnostic_b_summary.parquet")

# %% [markdown]
# ## Recommendation

# %%
# Smallest max_iter that satisfies BOTH:
#   (i) AUC matches reference within `auc_tol`
#   (ii) Diagnostic A shows >= `converged_frac_min` of fits converged by k
diag_a_frac = {
    c: float(((n_iter_arr <= c) & converged_arr).mean()) for c in cutoffs
}

candidates = []
for row in summary.iter_rows(named=True):
    mi = row["max_iter"]
    if mi == ref_mi:
        continue
    auc_ok = (row["max_abs_diff"] <= auc_tol)
    # Use the closest cutoff <= mi from the diagnostic A grid.
    le_mi = [c for c in cutoffs if c <= mi]
    diag_cutoff = max(le_mi) if le_mi else min(cutoffs)
    converged_ok = diag_a_frac[diag_cutoff] >= converged_frac_min
    candidates.append({
        "max_iter": mi,
        "auc_ok": auc_ok,
        "converged_ok": converged_ok,
        "speedup": row["speedup_vs_ref"],
    })

print("Candidate cutoffs (auc_ok=AUC matches ref within tol, "
      "converged_ok=>=99% of fits hit tol by then):")
for c in candidates:
    print(f"  max_iter={c['max_iter']:>3d}  auc_ok={c['auc_ok']}  "
          f"converged_ok={c['converged_ok']}  speedup={c['speedup']:.2f}x")

passing = [c for c in candidates if c["auc_ok"] and c["converged_ok"]]
if passing:
    rec = min(passing, key=lambda c: c["max_iter"])
    print(f"\nRECOMMENDATION: max_iter = {rec['max_iter']}  "
          f"(estimated speedup vs production max_iter=50: {rec['speedup']:.2f}x)")
else:
    auc_passing = [c for c in candidates if c["auc_ok"]]
    if auc_passing:
        rec = min(auc_passing, key=lambda c: c["max_iter"])
        print(f"\nNo cutoff hit both criteria. AUC matches at max_iter="
              f"{rec['max_iter']} but convergence criterion not met — consider "
              f"loosening auc_tol or sampling more cutoffs in Diagnostic A.")
    else:
        print("\nNo cutoff matched the AUC tolerance — extend sweep upward.")

print(f"\nArtifacts written under {outdir}/")
