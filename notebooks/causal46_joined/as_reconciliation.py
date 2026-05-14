# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # causal4/causal6 AS-site reconciliation
#
# Classifies every (subject, electrode_idx, phoneme_pair) tuple evaluated by
# either pipeline into one of five buckets, then renders summary stats and
# four star-plot PDFs (losses, gains-eligible, gains-newly-eligible, both) for
# visual inspection. Final canonical AS-site list is written to
# `outputs/causal46_joined/canonical_AS_sites.csv`.
#
# See `docs/superpowers/plans/2026-05-14-causal46-as-reconciliation.md` and
# Linear JON-42.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

# %%
HOME = Path(os.path.expanduser("~"))
# Resolve REPO from this notebook's location so it works in any worktree.
REPO = Path(__file__).resolve().parents[2]
CAUSAL4_DIR = HOME / "freesurfer_subjects/barakeet/causal4_pipeline/prepare_neurometrics"
CAUSAL6_DIR = HOME / "freesurfer_subjects/barakeet/causal6_speech_responsive_pipeline/acoustic_decoding_peaks"
OUT_DIR = REPO / "outputs/causal46_joined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSAL4_AUC_THRESHOLD = 0.65
CAUSAL6_P_THRESHOLD = 0.05

print(f"REPO:        {REPO}")
print(f"CAUSAL4_DIR: {CAUSAL4_DIR}")
print(f"CAUSAL6_DIR: {CAUSAL6_DIR}")
print(f"OUT_DIR:     {OUT_DIR}")
