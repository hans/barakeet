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
# Select the subject to use for causal6 reg_lambda tuning.
#
# Rank subjects by the count of (phoneme_pair, word_end) tuples that contain at
# least 2 ambiguous resampled steps — i.e. stimulus steps that elicited variable
# behavioral responses across repeats. Break ties by total ambiguous-trial count.
#
# This identifies a subject where the behavior decoder has the most signal to
# operate on, making the reg_lambda sweep maximally informative.

# %%
from pathlib import Path

import mne
import pandas as pd
import polars as pl
from loguru import logger as L
from tqdm.auto import tqdm

# %%
from src.data import add_metadata_features, get_ambiguous_resampled_steps

# %% tags=["parameters"]
all_epochs_dir = "outputs/epochs_preprocessed"
outdir = "."
ambiguous_response_threshold = 2

# %%
epoch_paths = sorted(Path(all_epochs_dir).glob("*_epo.fif"))
L.info(f"Loading {len(epoch_paths)} subject epoch files...")

# %%
all_md_frames = []
for epoch_path in tqdm(epoch_paths, desc="subjects"):
    subject = epoch_path.stem.replace("_epo", "")
    ep = mne.read_epochs(str(epoch_path), verbose=False)
    assert ep.metadata is not None
    md = add_metadata_features(ep.metadata).copy()
    md["subject"] = subject
    all_md_frames.append(md)

all_md_pd = pd.concat(all_md_frames, ignore_index=True)

# %%
all_md = pl.from_pandas(
    all_md_pd[
        ["subject", "phoneme_pair", "word_end", "resampled", "behavior_dummy_forced"]
    ].dropna()
)

amb = get_ambiguous_resampled_steps(
    all_md, ambiguous_response_threshold=ambiguous_response_threshold
)

# %%
# Rank subjects
rank_rows = []
for (subject, phoneme_pair, word_end), steps in amb.items():
    rank_rows.append(dict(
        subject=subject, phoneme_pair=phoneme_pair, word_end=word_end,
        n_ambiguous_steps=len(steps), ambiguous_steps=tuple(steps),
    ))
rank_df = pd.DataFrame(rank_rows)

# Trial counts for tie-breaking
trial_counts = (
    all_md_pd.groupby(["subject", "phoneme_pair", "word_end", "resampled"])
    .size().reset_index(name="n_trials")
)

# Sum trials in ambiguous steps per subject
amb_steps_flat = rank_df.explode("ambiguous_steps").rename(
    columns={"ambiguous_steps": "resampled"}
)
amb_steps_flat["resampled"] = amb_steps_flat["resampled"].astype(int)

amb_trial_counts = amb_steps_flat.merge(
    trial_counts, on=["subject", "phoneme_pair", "word_end", "resampled"], how="left",
)

per_subject = (
    rank_df.groupby("subject")
    .agg(n_ambiguous_tuples=("phoneme_pair", "count"),
         total_ambiguous_steps=("n_ambiguous_steps", "sum"))
    .merge(
        amb_trial_counts.groupby("subject")["n_trials"].sum().rename("ambiguous_trial_count"),
        on="subject", how="left",
    )
    .sort_values(
        by=["n_ambiguous_tuples", "total_ambiguous_steps", "ambiguous_trial_count"],
        ascending=False,
    )
    .reset_index()
)

print(per_subject.to_string(index=False))

winner = per_subject.iloc[0]["subject"]
L.info(f"Selected tuning subject: {winner}")

# %%
out = Path(outdir) / "tuning_subject_ranking.csv"
per_subject.to_csv(out, index=False)
L.info(f"Wrote {out}")

winner_path = Path(outdir) / "tuning_subject.txt"
winner_path.write_text(str(winner))
L.info(f"Wrote {winner_path}")
