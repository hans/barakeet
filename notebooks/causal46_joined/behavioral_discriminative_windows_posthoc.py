# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet (3.12.13)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
import sys

import mne
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from src.data import add_metadata_features

sys.path.insert(0, str(Path(".").resolve() / "notebooks" / "causal46_joined"))

from _within_completion import select_cell_trials_bootstrap, searchlight_mean_diff, \
    per_step_class_counts, resolve_behavior_col, extract_hga

# %%
min_per_class_k = 3
R = 10000
base_seed = 42

beh_windows = pd.read_parquet("outputs/causal46_joined/behavioral_discriminative_windows/b_windows.parquet")
trial_balance = pl.read_csv("outputs/causal46_joined/trial_balance_index.csv")

# %%
b4_qualified = (
    trial_balance
    .filter(pl.col("is_ambiguous_step"))
    .group_by(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .agg(
        pl.col("resampled").sort().alias("qualifying_steps"),
        pl.col("min_class").sum().alias("n_per_class"),
        pl.len().alias("n_qualifying"),
    )
    .filter((pl.col("n_qualifying") >= 1) & (pl.col("n_per_class") >= min_per_class_k))
    .sort(["subject", "electrode_idx", "phoneme_pair", "word_end"])
    .to_pandas()
)

# %%
epochs_dict = {}
for p in Path("outputs/epochs_preprocessed").glob("*.fif"):
    ep = mne.read_epochs(p, verbose=False)
    ep.metadata = add_metadata_features(ep.metadata)
    epochs_dict[p.stem.rstrip("_epo")] = ep

# %%
bhv_col = resolve_behavior_col(next(iter(epochs_dict.values())).metadata)
bhv_col

# %%
# DEV
row = beh_windows.iloc[0]
row


# %%
def estimate_contrast_pvalue(row, R=R, base_seed=base_seed):
    ep_i = epochs_dict[row.subject]
    md_i = ep_i.metadata
    ep_pp = ep_i[ep_i.metadata.phoneme_pair == row.phoneme_pair]
    md_pp = md_i.query(f"phoneme_pair == '{row.phoneme_pair}'")

    qualifying_steps = b4_qualified.query(
        f"subject == '{row.subject}' & electrode_idx == {row.electrode_idx} & phoneme_pair == '{row.phoneme_pair}' & word_end == '{row.word_end}'"
    ).iloc[0].qualifying_steps
    qualifying_steps = [int(x) for x in qualifying_steps]

    per_step = per_step_class_counts(
        md_pp, word_end=row.word_end, qualifying_steps=qualifying_steps,
        group_col=bhv_col,
    )

    hga = extract_hga(ep_pp, int(row.electrode_idx))

    W = row.smax - row.smin
    obs, null = np.empty(R), np.empty(R)
    for r in range(R):                                   # R can now be 1e5+
        rng   = np.random.default_rng(base_seed + r)
        draws = select_cell_trials_bootstrap(per_step, rng=rng)
        klo, khi = sorted(draws)                          # neg, pos
        obs[r] = searchlight_mean_diff(hga, draws[khi], draws[klo],
                    search_smin=row.smin, search_smax=row.smax, window_size=W, stride=W)[0].mean_diff
        # --- inline the same per-step pool-and-resplit permutation as bootstrap_cell ---
        npos, nneg, off = [], [], 0
        for by in per_step.values():
            ns = min(len(v) for v in by.values())
            if ns == 0: continue
            pool = np.concatenate([draws[khi][off:off+ns], draws[klo][off:off+ns]]); rng.shuffle(pool)
            npos.append(pool[:ns]); nneg.append(pool[ns:]); off += ns
        null[r] = searchlight_mean_diff(hga, np.concatenate(npos), np.concatenate(nneg),
                    search_smin=row.smin, search_smax=row.smax, window_size=W, stride=W)[0].mean_diff

    # per-cell two-tailed permutation p at the fixed window, well-resolved at large R:
    p = (np.sum(np.abs(null) >= abs(obs.mean())) + 1) / (R + 1)

    return obs, null, p


# %%
obs, null, p = estimate_contrast_pvalue(beh_windows.iloc[1], R=10, base_seed=42)

# %%
contrast_results = []
for _, row in tqdm(beh_windows.iterrows(), total=len(beh_windows)):
    obs, null, p = estimate_contrast_pvalue(row, R=R, base_seed=base_seed)
    contrast_results.append({
        "subject": row.subject,
        "electrode_idx": row.electrode_idx,
        "phoneme_pair": row.phoneme_pair,
        "word_end": row.word_end,
        "smin": row.smin,
        "smax": row.smax,
        "obs_mean_diff": obs.mean(),
        "null_mean_diff": null.mean(),
        "p_value": p
    })

contrast_results_df = pd.DataFrame(contrast_results)
