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
# causal6: acoustic decoder permutation-null refits with two-stage adaptive K.
#
# Stage 1 runs `n_permutations_stage1` shuffles across all speech-responsive
# electrodes. The stage-1 nulls are aggregated via
# `src.models.causal6_aggregates.aggregate_acoustic` and gated by
# `stage1_gate` from `src.models.causal6_adaptive_null`: any site whose
# global-min K1 max-stat-corrected p (over the same flavors as
# `acoustic_decoding_peaks` produces) is ≤ `escalate_corrected_p_max` is
# flagged for stage 2. Stage 2 runs `n_permutations_stage2` more shuffles
# on the borderline electrodes only, with non-overlapping seeds so the
# merged null is bit-identical to a flat-K=K1+K2 reference.
#
# Outputs:
#   null_scores.parquet     — merged (stage1 + stage2-filtered) null.
#   escalation_log.parquet  — one row per site with per-flavor
#                             corrected_p, peak window/flavor,
#                             escalated bool, and final per-site K.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

# %%
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
    run_acoustic_searchlight_permutations,
)
from src.models.causal6_adaptive_null import (
    filter_null_to_borderline,
    log_stage1_gate,
    log_stage2_done,
    log_stage2_skipped,
    stage1_gate,
    stage2_spill_dir,
)
from src.models.causal6_aggregates import (
    FLAVORS_ACOUSTIC,
    SITE_KEYS_ACOUSTIC,
    aggregate_acoustic,
)

# %% tags=["parameters"]
subject = "EC282"
epochs_path = f"outputs/epochs_preprocessed/{subject}_epo.fif"
electrodes_path = f"outputs/causal5/find_speech_responsive/{subject}_results.csv"
scores_path = f"outputs/causal6/acoustic_decoding_single_electrode/{subject}/scores.parquet"
outdir = "."

min_sample = 1
window_size = 15
stride = 2

target = "categorical_acoustic_cue"
peak_search_smin = 50
peak_search_smax = 75

reg_lambda = 1.0
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

# %% [markdown]
# ## Stage 1 — permutations across all speech-responsive electrodes.

# %%
stage1_seeds = list(range(permutation_seed, permutation_seed + n_permutations_stage1))
null_stage1 = run_acoustic_searchlight_permutations(
    epochs, subject=subject,
    electrode_idxs=speech_responsive_idxs,
    windows=windows,
    reg_lambda=reg_lambda,
    permute_seeds=stage1_seeds,
    permutation_chunk_size=permutation_chunk_size,
    target=target,
    n_folds=n_folds, cv_random_state=cv_random_state,
    device=device, dtype=torch.float32,
    tol=tol, max_iter=max_iter,
)
assert null_stage1.height > 0, f"[{subject}] acoustic stage-1 produced no rows"

# %% [markdown]
# ## Gate — aggregate, compute per-flavor corrected p, decide escalation.

# %%
real_scores = pl.read_parquet(scores_path)
real_agg, null_agg_stage1 = aggregate_acoustic(
    real_scores, null_stage1,
    target=target,
    peak_search_smin=peak_search_smin,
    peak_search_smax=peak_search_smax,
)

borderline_keys, gate_log = stage1_gate(
    real_agg, null_agg_stage1,
    site_keys=SITE_KEYS_ACOUSTIC,
    flavors=FLAVORS_ACOUSTIC,
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
    eidx_pos = SITE_KEYS_ACOUSTIC.index("electrode_idx")
    borderline_electrode_idxs = sorted({k[eidx_pos] for k in borderline_keys})

    # Stream stage-2 chunks to parquet shards instead of materializing the
    # full raw null in RAM; scan + filter lazily so peak memory stays
    # bounded by the (much smaller) borderline-filtered result.
    with stage2_spill_dir(outdir) as spill_dir:
        run_acoustic_searchlight_permutations(
            epochs, subject=subject,
            electrode_idxs=borderline_electrode_idxs,
            windows=windows,
            reg_lambda=reg_lambda,
            permute_seeds=stage2_seeds,
            permutation_chunk_size=permutation_chunk_size,
            target=target,
            n_folds=n_folds, cv_random_state=cv_random_state,
            device=device, dtype=torch.float32,
            tol=tol, max_iter=max_iter,
            spill_dir=spill_dir,
        )
        null_stage2 = filter_null_to_borderline(
            pl.scan_parquet(spill_dir / "*.parquet"),
            borderline_keys,
            site_keys=SITE_KEYS_ACOUSTIC,
        ).collect()

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
# ## Outputs.

# %%
gate_log = gate_log.with_columns(
    pl.when(pl.col("escalated"))
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
