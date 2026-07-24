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
# # Late integration: no individual cell survives FDR (pre-TFCE)
#
# Report claim: using **just the behavioral discriminative windows** and a plain
# fixed-contrast two-tailed test — **before any TFCE** — no individual integration
# (late/perceptual) cell is significant after FDR.
#
# Contrast is fixed and signed: `mean_diff_raw` = chose /n/ − chose /d/ (NOT aligned
# to acoustic tuning). Within each cell the ~10-window post-acoustic search is collapsed
# by a **max-|z| permutation correction** → one p per cell; then **BH-FDR across cells**.
# Pooling all (cell × window) tests into one flat BH family would ignore this nesting.
#
# Pure post-processing of `b4_bootstrap.parquet` (the B4 within-completion bootstrap +
# label-permutation null). No epochs needed.

# %% tags=["parameters"]
# Defaults read the synced prod state (outputs_prod). On prod, point at outputs/.
b4_bootstrap_path = "outputs_prod/causal46_joined/t_tests/b4_bootstrap.parquet"
b4_per_cell_path = "outputs_prod/causal46_joined/t_tests/b4_per_cell.parquet"
search_tail_s = 0.1       # behavioral search range = [phon_smax, word_offset + tail]
alpha = 0.05

# %%
import numpy as np
import polars as pl
from statsmodels.stats.multitest import multipletests

from src.stimuli import OFFSET_DICT
from src.viz_paper import epoch_sfreq, epoch_tmin

CELL = ["subject", "electrode_idx", "phoneme_pair", "word_end"]

# %% [markdown]
# ## Load fixed-contrast bootstrap replicates + per-cell acoustic boundary

# %%
boot = pl.read_parquet(
    b4_bootstrap_path,
    columns=CELL + ["smin", "smax", "replicate", "mean_diff_raw", "mean_diff_aligned_null"],
)
phon = pl.read_parquet(b4_per_cell_path).select(CELL + ["phon_smax"]).unique()
print(f"{boot.height:,} replicate rows across {phon.height} cells")

# %% [markdown]
# ## Restrict to post-acoustic candidate windows
#
# The behavioral search range (matching `t_tests.py::behav_search_range` /
# `behavioral_discriminative_windows.py`): `smin >= phon_smax` and
# `smax <= word_offset + search_tail_s`.

# %%
b = boot.join(phon, on=CELL).with_columns(
    pl.col("word_end").replace_strict(OFFSET_DICT, default=None).alias("offset")
)
b = b.with_columns(
    ((pl.col("offset") + search_tail_s - epoch_tmin) * epoch_sfreq)
    .round().cast(pl.Int64).alias("search_smax")
)
b = b.filter((pl.col("smin") >= pl.col("phon_smax")) & (pl.col("smax") <= pl.col("search_smax")))
n_win = b.select(CELL + ["smin"]).unique().height
n_cell = b.select(CELL).unique().height
print(f"{n_win} candidate (cell, window) tests over {n_cell} cells")

# %% [markdown]
# ## Per-window standardized statistic
#
# `z_obs = |mean(mean_diff_raw)| / SD(null)` per window — standardized by the
# permutation-null SD so the across-window max isn't dominated by high-variance windows.

# %%
win = (
    b.group_by(CELL + ["smin"])
    .agg(
        pl.col("mean_diff_raw").mean().abs().alias("obs_abs"),
        pl.col("mean_diff_aligned_null").std().alias("sd_null"),
    )
    .filter(pl.col("sd_null") > 0)
    .with_columns((pl.col("obs_abs") / pl.col("sd_null")).alias("z_obs"))
)

# %% [markdown]
# ## Max-|z| permutation correction → one p per cell
#
# Observed statistic = max over windows of `z_obs`. Null = for each permutation
# replicate, the max over windows of `|null| / SD(null)`. Per-cell p = fraction of
# null replicates whose max ≥ observed max (`+1` unbiased permutation p).

# %%
obs = win.group_by(CELL).agg(pl.col("z_obs").max().alias("obs_maxz"))

nb = b.join(win.select(CELL + ["smin", "sd_null"]), on=CELL + ["smin"]).with_columns(
    (pl.col("mean_diff_aligned_null").abs() / pl.col("sd_null")).alias("z_null")
)
null_max = nb.group_by(CELL + ["replicate"]).agg(pl.col("z_null").max().alias("null_maxz"))

obs_lookup = {tuple(r[:-1]): r[-1] for r in obs.iter_rows()}
rows = []
for keys, g in null_max.group_by(CELL):
    o = obs_lookup[keys]
    nd = g["null_maxz"].to_numpy()
    R = len(nd)
    rows.append({
        **dict(zip(CELL, keys)),
        "p": (np.sum(nd >= o) + 1) / (R + 1),
        "R": R,
        "at_floor": bool(np.sum(nd >= o) == 0),
    })
res = pl.DataFrame(rows)

# %% [markdown]
# ## BH-FDR across cells

# %%
p = res["p"].to_numpy()
reject, q, _, _ = multipletests(p, method="fdr_bh", alpha=alpha)
res = res.with_columns(pl.Series("q", q), pl.Series("reject", reject))

print(f"cells:               {len(p)}")
print(f"uncorrected p<{alpha}:   {int((p < alpha).sum())}")
print(f"BH-FDR survivors:    {int(reject.sum())}   (min q = {q.min():.3f})")

# %% [markdown]
# ## Floor check — is the null real, or permutation-censored?
#
# The permutation p can resolve no finer than `1/(R+1)`. If the smallest per-cell p is
# well above that floor and no cell is pinned at it, R is sufficient and the null is
# genuine (not a resolution artifact).

# %%
floor = 1 / (int(res["R"][0]) + 1)
print(f"permutation floor 1/(R+1) = {floor:.2e}")
print(f"min per-cell p            = {p.min():.4f}")
print(f"cells pinned at floor     = {int(res['at_floor'].sum())}")
print(
    "=> min p is well above the floor and 0 cells are pinned: R is sufficient, null is real."
    if p.min() > 2 * floor and int(res["at_floor"].sum()) == 0
    else "=> near the floor: more permutations needed for a rigorous corrected p."
)

# %% [markdown]
# **Result.** 0/187 cells survive BH-FDR (only ~2 reach uncorrected p<0.05, chance
# level; min q≈1.0). The min per-cell p (≈0.019) sits an order of magnitude above the
# 10⁻³ permutation floor with no cell pinned there, so R=1000 is ample — the absence of
# individual integration effects is a genuine null, not permutation censoring. For
# contrast, taking each cell's *self-selected* peak window as given (no search
# correction) leaves ~48/187 surviving FDR; that number is circular and must not be
# reported as a test.
