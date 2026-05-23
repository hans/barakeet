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
# causal6: behavior-with-control permutation-null refits with two-stage
# adaptive K.
#
# Stage 1 runs `n_permutations_stage1` shuffles for both `model='full'`
# and `model='baseline'`. The stage-1 nulls are aggregated via
# `aggregate_behavior_with_control` (paired full−baseline diff) and
# gated by `stage1_gate` over the four behavior flavors (fold_mean +
# t_stat, raw + TFCE). Borderline sites get K2 more shuffles (with
# non-overlapping seeds for determinism) restricted to the borderline
# electrodes; results are filtered down to the exact borderline
# (electrode_idx, phoneme_pair, word_end) tuples and merged with stage 1.
#
# Outputs:
#   null_scores.parquet     — merged stage1 + filtered stage2 null.
#   escalation_log.parquet  — per-site corrected_p, peak window/
#                             flavor, escalated bool, final per-site K.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %%
from functools import partial
from pathlib import Path
import re

import mne
import pandas as pd
import polars as pl
import torch

# %%
from src.data import add_metadata_features
from src.models.causal6 import (
    make_windows,
    run_behavior_with_control_permutations,
)
from src.models.causal6_adaptive_null import (
    filter_null_to_borderline,
    log_stage1_gate,
    log_stage2_done,
    log_stage2_skipped,
    stage1_gate,
    stage2_spill_dir,
    stage3_boost,
)
from src.models.causal6_aggregates import (
    FLAVORS_BEHAVIOR_WITH_CONTROL,
    SITE_KEYS_BEHAVIOR_WITH_CONTROL,
    aggregate_behavior_with_control,
    behavior_null_smax,
    preagg_behavior_with_control_null,
)

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
scores_path = f"outputs/causal6/behavior_decoding_single_electrode/{subject}/scores.parquet"
outdir = "."

min_sample = 1
window_size = 15
stride = 2

epoch_tmin = -0.4
epoch_sfreq = 100
behav_peak_post_offset_s = 0.2
peak_search_smin = 0
peak_search_smax = 290

reg_lambda = 1.0
reg_lambda_baseline = None
n_folds = 5
cv_random_state = 42
device = "cuda"
tol = 1e-6
max_iter = 15

n_permutations_stage1 = 1000
n_permutations_stage2 = 9000
escalate_corrected_p_max = 0.20
permutation_seed = 0
permutation_chunk_size = 6

# Stage-3 boost — refit sites in the BH rejection neighborhood at K3 perms.
n_permutations_stage3 = 90000
stage3_k_gate = 200
fdr_alpha = 0.05
fdr_rois = []                       # populated from config.analysis.fdr_rois
electrode_dfs_paths = []            # all subjects' find_speech_responsive CSVs

# %%
subject = re.findall(r"(EC[\d]+)_epo", str(epochs_path))[0]
outdir = Path(outdir)

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
_null_smax = behavior_null_smax(epoch_tmin, epoch_sfreq, behav_peak_post_offset_s, peak_search_smax)
windows = windows[(windows[:, 0] >= peak_search_smin) & (windows[:, 1] <= _null_smax)]

# %% [markdown]
# ## Stage 1 — permutations across all speech-responsive electrodes.

# %%
stage1_seeds = list(range(permutation_seed, permutation_seed + n_permutations_stage1))
null_stage1_raw = run_behavior_with_control_permutations(
    epochs, subject=subject,
    electrode_idxs=speech_responsive_idxs,
    windows=windows,
    reg_lambda=reg_lambda,
    reg_lambda_baseline=reg_lambda_baseline,
    permute_seeds=stage1_seeds,
    permutation_chunk_size=permutation_chunk_size,
    n_folds=n_folds, cv_random_state=cv_random_state,
    device=device, dtype=torch.float32,
    tol=tol, max_iter=max_iter,
)
assert null_stage1_raw.height > 0, (
    f"[{subject}] behavior with-control stage-1 produced no rows"
)

# %% [markdown]
# ## Gate — aggregate, compute per-flavor corrected p, decide escalation.

# %%
real_scores = pl.read_parquet(scores_path)
null_stage1 = preagg_behavior_with_control_null(null_stage1_raw, real_scores)
del null_stage1_raw

real_agg, null_agg_stage1 = aggregate_behavior_with_control(
    real_scores, null_stage1,
    epoch_tmin=epoch_tmin,
    epoch_sfreq=epoch_sfreq,
    behav_peak_post_offset_s=behav_peak_post_offset_s,
    peak_search_smin=peak_search_smin,
    peak_search_smax=peak_search_smax,
)

borderline_keys, gate_log = stage1_gate(
    real_agg, null_agg_stage1,
    site_keys=SITE_KEYS_BEHAVIOR_WITH_CONTROL,
    flavors=FLAVORS_BEHAVIOR_WITH_CONTROL,
    p_max=escalate_corrected_p_max,
)

n_esc = len(borderline_keys)
log_stage1_gate(
    subject,
    n_permutations_stage1=n_permutations_stage1,
    p_max=escalate_corrected_p_max,
    gate_log=gate_log,
    n_borderline=n_esc,
)

# %% [markdown]
# ## Stage 2 — additional permutations on the borderline electrodes only.

# %%
if borderline_keys and n_permutations_stage2 > 0:
    stage2_seeds = list(range(
        permutation_seed + n_permutations_stage1,
        permutation_seed + n_permutations_stage1 + n_permutations_stage2,
    ))
    eidx_pos = SITE_KEYS_BEHAVIOR_WITH_CONTROL.index("electrode_idx")
    borderline_electrode_idxs = sorted({k[eidx_pos] for k in borderline_keys})

    with stage2_spill_dir(outdir) as spill_dir:
        run_behavior_with_control_permutations(
            epochs, subject=subject,
            electrode_idxs=borderline_electrode_idxs,
            windows=windows,
            reg_lambda=reg_lambda,
            reg_lambda_baseline=reg_lambda_baseline,
            permute_seeds=stage2_seeds,
            permutation_chunk_size=permutation_chunk_size,
            n_folds=n_folds, cv_random_state=cv_random_state,
            device=device, dtype=torch.float32,
            tol=tol, max_iter=max_iter,
            spill_dir=spill_dir,
        )
        null_stage2_raw = filter_null_to_borderline(
            pl.scan_parquet(spill_dir / "*.parquet"),
            borderline_keys,
            site_keys=SITE_KEYS_BEHAVIOR_WITH_CONTROL,
            baseline_site_keys=["subject", "phoneme_pair", "word_end"],
        ).collect()
    null_stage2 = preagg_behavior_with_control_null(null_stage2_raw, real_scores)
    del null_stage2_raw

    null_scores = pl.concat([null_stage1, null_stage2])
    log_stage2_done(
        subject,
        n_permutations_stage2=n_permutations_stage2,
        n_borderline_electrodes=len(borderline_electrode_idxs),
        n_borderline_sites=n_esc,
        null_height=null_scores.height,
    )
else:
    null_scores = null_stage1
    log_stage2_skipped(
        subject,
        n_borderline=n_esc,
        n_permutations_stage2=n_permutations_stage2,
    )

# %% [markdown]
# ## Stage 3 — boost K for sites in the BH rejection neighborhood.

# %%
if fdr_rois and electrode_dfs_paths and n_permutations_stage3 > 0:
    electrode_dfs = [
        pl.from_pandas(pd.read_csv(p)) for p in electrode_dfs_paths
    ]
    stage3_seeds = list(range(
        permutation_seed + n_permutations_stage1 + n_permutations_stage2,
        permutation_seed + n_permutations_stage1 + n_permutations_stage2
            + n_permutations_stage3,
    ))
    null_scores, gate_log = stage3_boost(
        subject=subject, outdir=outdir,
        real_scores=real_scores,
        real_agg=real_agg,
        null_scores=null_scores,
        gate_log=gate_log,
        site_keys=SITE_KEYS_BEHAVIOR_WITH_CONTROL,
        flavors=FLAVORS_BEHAVIOR_WITH_CONTROL,
        aggregate_fn=partial(
            aggregate_behavior_with_control,
            epoch_tmin=epoch_tmin,
            epoch_sfreq=epoch_sfreq,
            behav_peak_post_offset_s=behav_peak_post_offset_s,
            peak_search_smin=peak_search_smin,
            peak_search_smax=peak_search_smax,
        ),
        preagg_fn=preagg_behavior_with_control_null,
        run_permutations_fn=partial(
            run_behavior_with_control_permutations,
            epochs, subject=subject,
            windows=windows,
            reg_lambda=reg_lambda,
            reg_lambda_baseline=reg_lambda_baseline,
            permutation_chunk_size=permutation_chunk_size,
            n_folds=n_folds, cv_random_state=cv_random_state,
            device=device, dtype=torch.float32,
            tol=tol, max_iter=max_iter,
        ),
        baseline_site_keys=["subject", "phoneme_pair", "word_end"],
        electrode_dfs=electrode_dfs,
        fdr_rois=fdr_rois,
        k_gate=stage3_k_gate,
        fdr_alpha=fdr_alpha,
        permutation_seeds=stage3_seeds,
        n_permutations_pre=n_permutations_stage1 + n_permutations_stage2,
    )
else:
    print(
        f"[{subject}] stage3 skipped "
        f"(rois={bool(fdr_rois)}, paths={bool(electrode_dfs_paths)}, K3={n_permutations_stage3})",
        flush=True,
    )
    gate_log = gate_log.with_columns(pl.lit(False).alias("stage3_refit"))

# %% [markdown]
# ## Outputs.

# %%
gate_log = gate_log.with_columns(
    pl.when(pl.col("stage3_refit"))
    .then(n_permutations_stage1 + n_permutations_stage2 + n_permutations_stage3)
    .when(pl.col("escalated"))
    .then(n_permutations_stage1 + n_permutations_stage2)
    .otherwise(n_permutations_stage1)
    .alias("n_permutations")
)

null_scores.write_parquet(outdir / "null_scores.parquet")
gate_log.write_parquet(outdir / "escalation_log.parquet")
print(
    f"Wrote null_scores.parquet ({null_scores.height} rows) and "
    f"escalation_log.parquet ({gate_log.height} rows) to {outdir}",
    flush=True,
)
