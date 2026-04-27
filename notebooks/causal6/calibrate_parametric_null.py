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
# # Calibrate parametric / GPD nulls against an existing K=10000 permutation run
#
# Question: can we replace K=10000 permutation refits (~20 h/subject) with a
# cheaper approximation, and still call the same sites significant?
#
# This notebook treats one already-computed `null_scores.parquet` as ground
# truth and asks how well two cheaper alternatives reproduce its tail
# p-values:
#
# * **GPD-tail extrapolation** — fit a generalized Pareto distribution to
#   the upper tail of a `K_sub`-sized subsample of the perms, extrapolate
#   to small p (Knijnenburg et al. 2009). Cost: K_sub refits per subject.
# * **Analytic Mann-Whitney null** — closed-form variance of fold-mean AUC
#   under H0 (no perms at all). Cost: free. Approximate because it ignores
#   the variance contributed by *refitting* the L2 logreg on shuffled
#   labels.
#
# Two calibration sections:
#   1. **Per-(site, window) p-values**: most granular, easy to reason about.
#   2. **Per-site peak (max-stat) p-values**: family-wise corrected over the
#      searchlight window — what downstream code actually consumes. Analytic
#      is skipped here because the max-over-correlated-windows distribution
#      doesn't have a simple closed form; only GPD vs empirical is compared.
#
# Both sections use a leave-one-out bootstrap: take perm `k` as a synthetic
# "observed" value, treat the other K-1 perms as the gold-standard empirical
# null. Repeat for many k. Under H0 each method's p-values should be
# uniform on [0, 1]; deviations are miscalibration.
#
# Run on the server (where the K=10000 nulls live):
#   conda activate /scratch/jgauthier/transformers3
#   jupytext --to notebook --execute notebooks/causal6/calibrate_parametric_null.py

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["POLARS_MAX_THREADS"] = "1"
# Polars prints query stages, streaming chunk sizes, and group-by progress
# to stderr when verbose=1. Helpful when watching long-running aggregations.
os.environ.setdefault("POLARS_VERBOSE", "1")

# %%
import resource
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import genpareto, norm
from tqdm.auto import tqdm


def _rss_gb() -> float:
    """Resident set size in GB. macOS reports bytes, Linux kilobytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)


_step_t0 = time.perf_counter()


def _step(msg: str) -> None:
    """Print a step marker with cumulative wallclock + peak RSS."""
    elapsed = time.perf_counter() - _step_t0
    print(f"[{elapsed:7.1f}s | rss={_rss_gb():5.2f}GB] {msg}", flush=True)

# %% tags=["parameters"]
# Defaults: an acoustic null run. Override to point at any decoder's parquet.
subject = "EC248"
null_scores_path = (
    f"outputs/causal6/acoustic_decoding_null/{subject}/null_scores.parquet"
)
outdir = "outputs/causal6/calibrate_parametric_null"

# Statistic to validate. fold_mean and t_stat are both used downstream
# (acoustic_decoding_peaks). t_stat is variance-normalized — we expect
# better calibration of the analytic null on it.
statistic = "fold_mean"  # one of {"fold_mean", "t_stat"}

# Calibration sweep: K_sub permutations represent the "production" budget
# we're considering replacing K=10000 with. Smaller K_sub = bigger speedup.
K_subs = [200, 500, 1000]

# Number of held-out "observed" candidates per (site, window) to use in
# the bootstrap calibration. Higher = tighter tail estimate but more compute.
K_test = 500

# Optional: cap the perms loaded from the parquet to control memory.
# Pushed down through the lazy scan as `permutation_idx < cap`.
#
# Tradeoff: gold-standard empirical CDF has p-floor 1/(K_train+1) where
# K_train = cap - K_test. To validate GPD at p=1e-3 you want K_train >=
# ~2000 (so cap >= 2500). For p=1e-4 you want K_train >= ~10000. Set to
# None to use every perm in the parquet (default for full validation).
n_permutations_cap = None

# Random seed for the leave-one-out bootstrap subsample selection.
rng_seed = 0

# GPD threshold quantile: GPD is fit to exceedances above this quantile of
# the K_sub subsample. 0.90 is the Knijnenburg default; 0.85 widens the
# fitting set and helps when K_sub is small.
gpd_threshold_q = 0.90

# %%
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(rng_seed)

null_path = Path(null_scores_path)
assert null_path.exists(), f"missing null parquet: {null_path}"
size_gb = null_path.stat().st_size / (1024 ** 3)
_step(f"lazy-scanning {null_path} (file size {size_gb:.2f} GB on disk)")
# Lazy scan + push-down filter + aggregation. Loading the long-format K=10000
# parquet eagerly would peak at ~45 GB for acoustic / ~90 GB for behavior;
# streaming aggregation collapses 5x at IO time and keeps peak memory bounded.
raw_lazy = pl.scan_parquet(null_path)
schema_cols = raw_lazy.collect_schema().names()
_step(f"  schema: {schema_cols}")

# %% [markdown]
# ## Aggregate folds → per-(site, window, perm) statistic
#
# Inlines `fold_tstat_aggregate`'s logic so we can run it on a LazyFrame
# (push-down to streaming) and fold the n_test mean into the same pass.

# %%
# Detect which group keys are present (acoustic has phoneme_pair, behavior
# adds word_end and model). Filter to model="full" if present (baseline is
# a separate test).
has_word_end = "word_end" in schema_cols
has_model = "model" in schema_cols
if has_model:
    raw_lazy = raw_lazy.filter(pl.col("model") == "full")
    print("  filter pushed: model='full'")
if n_permutations_cap is not None:
    raw_lazy = raw_lazy.filter(pl.col("permutation_idx") < n_permutations_cap)
    print(f"  filter pushed: permutation_idx < {n_permutations_cap}")

site_keys = ["subject", "phoneme_pair", "electrode_idx"]
if has_word_end:
    site_keys = site_keys + ["word_end"]
window_keys = site_keys + ["smin", "smax"]
group_keys = window_keys + ["permutation_idx"]
print(f"  site_keys = {site_keys}")
print(f"  window_keys = {window_keys}")

# Single streaming aggregation: collapse folds + carry n_test_mean through.
std_floor = 0.01
agg_lazy = (
    raw_lazy
    .group_by(group_keys)
    .agg(
        pl.col("test_roc_auc").mean().alias("fold_mean"),
        pl.col("test_roc_auc").std().alias("fold_std"),
        pl.col("test_roc_auc").len().alias("n_folds"),
        pl.col("n_test").mean().alias("n_test_mean"),
    )
)
# Show the optimized plan so it's clear what polars is about to do.
print("--- query plan ---")
try:
    print(agg_lazy.explain(streaming=True))
except TypeError:
    print(agg_lazy.explain())
print("------------------", flush=True)

# Streaming aggregation. Polars 1.24 takes `streaming=True`; newer
# versions take `engine="streaming"`. Try both, fall back to non-streaming.
_step("collecting fold-aggregation (this is the big one) ...")
try:
    agg = agg_lazy.collect(streaming=True)
    _collect_mode = "streaming=True"
except (TypeError, ValueError):
    try:
        agg = agg_lazy.collect(engine="streaming")
        _collect_mode = "engine=streaming"
    except (TypeError, ValueError):
        agg = agg_lazy.collect()
        _collect_mode = "non-streaming"
_step(f"  agg collected ({_collect_mode}) → {agg.height} rows, "
      f"{agg.estimated_size('mb'):.0f} MB")
sem = (
    pl.max_horizontal(pl.col("fold_std"), pl.lit(std_floor))
    / pl.col("n_folds").cast(pl.Float64).sqrt()
)
agg = agg.with_columns(((pl.col("fold_mean") - 0.5) / sem).alias("t_stat"))
_step(f"  added t_stat → {agg.height} rows ({statistic} available)")

# Build a (n_units, K) numpy block by sorting on (window_keys, permutation_idx)
# and reshaping. Avoids polars pivot-version pitfalls.
permutation_ids = sorted(agg["permutation_idx"].unique().to_list())
K = len(permutation_ids)
print(f"  K = {K} permutations")
assert K >= max(K_subs) + K_test, (
    f"need at least {max(K_subs) + K_test} perms for K_sub={max(K_subs)} "
    f"and K_test={K_test}; got {K}"
)
# Defensive: confirm every (site, window) has all K permutation rows.
counts = agg.group_by(window_keys).agg(pl.len().alias("n"))
assert (counts["n"] == K).all(), (
    f"some (site, window) groups have != {K} perms; "
    f"min={counts['n'].min()}, max={counts['n'].max()}"
)

# Sort: stable lexicographic sort by (window_keys, permutation_idx) puts
# every (site, window)'s K rows in contiguous memory in perm-id order.
_step(f"sorting agg ({agg.height} rows) ...")
sorted_agg = agg.sort(window_keys + ["permutation_idx"])
del agg
_step("  sort done")
n_units = sorted_agg.height // K
stats_mat = sorted_agg[statistic].to_numpy().reshape(n_units, K)
_step(f"  extracted stats_mat ({stats_mat.nbytes / 1024**2:.0f} MB)")
# (Site, window) metadata in row order; n_test_mean is constant within
# a (site, window), so just take the first occurrence.
wide = sorted_agg.gather_every(K).select(window_keys + ["n_test_mean"])
del sorted_agg  # ~all of the post-aggregation polars memory; freed once arrays are extracted
n_test_mean = wide["n_test_mean"].to_numpy()
assert wide.height == n_units, f"row alignment broken: {wide.height} vs {n_units}"
_step(f"  n_units (site × window) = {n_units}")
_step(f"  stats_mat shape = {stats_mat.shape}, "
      f"finite fraction = {np.isfinite(stats_mat).mean():.4f}")


# %% [markdown]
# ## Helper functions

# %%
def empirical_pvalue(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """
    Right-tail empirical p-value for each observed against shared null.
    `observed` is (M,), `null` is (N,). Returns (M,) p-values in [1/(N+1), 1].
    Uses the (>=) convention with +1 numerator/denominator (Phipson & Smyth
    2010) so p is never exactly zero — important for log-scale plotting.
    """
    null_sorted = np.sort(null[np.isfinite(null)])
    if null_sorted.size == 0:
        return np.full_like(observed, np.nan)
    # Number of null values >= observed = N - searchsorted(null, observed)
    rank_ge = null_sorted.size - np.searchsorted(null_sorted, observed, side="left")
    return (rank_ge + 1) / (null_sorted.size + 1)


def gpd_tail_pvalue(
    observed: np.ndarray, null: np.ndarray, threshold_q: float = 0.90,
    min_exceedances: int = 20,
) -> np.ndarray:
    """
    Generalized-Pareto-tail p-value (Knijnenburg et al. 2009).

    For observed values above the `threshold_q` quantile of `null`, fit a
    GPD to the threshold exceedances and extrapolate the tail. For
    observed values at or below the threshold, fall back to the empirical
    p-value (no extrapolation needed).
    """
    null = null[np.isfinite(null)]
    if null.size == 0:
        return np.full_like(observed, np.nan)
    threshold = np.quantile(null, threshold_q)
    exceedances = null[null > threshold] - threshold

    if exceedances.size < min_exceedances:
        # Not enough tail data to fit GPD — fall back to empirical.
        return empirical_pvalue(observed, null)

    # Fit GPD with location fixed at 0 (we've already shifted by threshold).
    try:
        shape, _, scale = genpareto.fit(exceedances, floc=0)
    except Exception:
        return empirical_pvalue(observed, null)
    if not (np.isfinite(shape) and np.isfinite(scale) and scale > 0):
        return empirical_pvalue(observed, null)

    out = empirical_pvalue(observed, null)  # default: empirical for <= threshold
    above = observed > threshold
    if above.any():
        # P(X > x) = (1 - threshold_q) * P(GPD > x - threshold)
        sf = genpareto.sf(observed[above] - threshold, shape, loc=0, scale=scale)
        out[above] = (1.0 - threshold_q) * sf
    # Floor at 1/(2 * len(null)) so log-plots don't blow up to zero.
    return np.clip(out, 1.0 / (2 * null.size), 1.0)


def analytic_foldmean_pvalue(
    observed_foldmean: np.ndarray,
    n_test_total: np.ndarray,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Right-tail p-value of fold-mean AUC under H0, assuming
    Mann-Whitney AUC variance per fold and approximate fold independence.

    Per fold (assuming approximately balanced classes per stratified
    split): n_pos = n_neg = n_test/2, so Var(AUC|H0) = (n_test+1)/(3*n_test^2).
    Var(fold-mean) ≈ Var(AUC|H0) / n_folds (independent folds).

    This null IGNORES the variance contributed by refitting the model on
    shuffled labels, so it's an approximation — its accuracy is exactly
    what this notebook is calibrating.
    """
    # n_test = n_pos + n_neg; with balanced classes n_pos = n_neg = n_test/2.
    # Var per fold = (n_pos + n_neg + 1) / (12 * n_pos * n_neg)
    #              = (n_test + 1) / (12 * (n_test/2)^2)
    #              = (n_test + 1) / (3 * n_test^2)
    var_fold = (n_test_total + 1.0) / (3.0 * n_test_total ** 2)
    var_foldmean = var_fold / n_folds
    sd_foldmean = np.sqrt(var_foldmean)
    z = (observed_foldmean - 0.5) / sd_foldmean
    return norm.sf(z)


# %% [markdown]
# ## Sanity checks on the helpers

# %%
# empirical_pvalue: under H0, the bootstrap p-value distribution is uniform.
_x = rng.standard_normal(20000)
_obs = rng.standard_normal(5000)
_p = empirical_pvalue(_obs, _x)
ks_dev = float(np.abs(np.sort(_p) - np.linspace(0, 1, len(_p), endpoint=False)).max())
print(f"empirical_pvalue under H0 (gaussian): max KS deviation = {ks_dev:.4f}")
assert ks_dev < 0.03, "empirical_pvalue helper is broken"

# GPD: under a gaussian null, the GPD tail should match empirical in [0, 0.1]
# region and slightly extrapolate beyond.
_p_gpd = gpd_tail_pvalue(_obs, _x, threshold_q=0.90)
print(f"gpd vs empirical at p<0.05: median ratio = "
      f"{np.median((_p_gpd / _p)[_p < 0.05]):.3f}")


# %% [markdown]
# ## Section 1 — per-(site, window) calibration
#
# For each (site, window):
#   - Hold out K_test perms as "observed" candidates.
#   - Use the remaining K_train = K - K_test as the gold-standard empirical null.
#   - Sub-sample K_sub of those K_train as the "production budget" null.
#   - Compute three p-values per observed candidate:
#       * `p_gold`     — empirical CDF on the K_train null (truth)
#       * `p_sub_emp`  — empirical CDF on the K_sub null (baseline: just fewer perms)
#       * `p_sub_gpd`  — GPD-tail using the K_sub null
#       * `p_analytic` — Mann-Whitney closed form (no perms)
#
# Compare {p_sub_emp, p_sub_gpd, p_analytic} against p_gold across all units.

# %%
def calibrate_per_window(K_sub: int) -> pl.DataFrame:
    """Run the leave-one-out bootstrap calibration with the given K_sub budget."""
    rng_local = np.random.default_rng(rng_seed)
    K_train = K - K_test
    cap = n_units * K_test
    # Pre-allocated columns; trim to actual write count at the end. Avoids
    # building a list of millions of Python dicts.
    out_unit = np.empty(cap, dtype=np.int32)
    out_p_gold = np.empty(cap, dtype=np.float64)
    out_p_sub_emp = np.empty(cap, dtype=np.float64)
    out_p_sub_gpd = np.empty(cap, dtype=np.float64)
    out_p_analytic = np.empty(cap, dtype=np.float64)
    w = 0

    for u in tqdm(range(n_units), desc=f"calibrate_per_window K_sub={K_sub}",
                  unit="unit", leave=False):
        nulls_full = stats_mat[u]                                # (K,)
        n_test_u = float(n_test_mean[u])
        if not np.isfinite(nulls_full).any():
            continue
        idx = rng_local.permutation(K)
        train_idx = idx[:K_train]
        test_idx = idx[K_train:K_train + K_test]
        null_train = nulls_full[train_idx]
        observed = nulls_full[test_idx]

        sub_idx = rng_local.choice(K_train, size=K_sub, replace=False)
        null_sub = null_train[sub_idx]

        p_gold = empirical_pvalue(observed, null_train)
        p_sub_emp = empirical_pvalue(observed, null_sub)
        p_sub_gpd = gpd_tail_pvalue(observed, null_sub, threshold_q=gpd_threshold_q)
        if statistic == "fold_mean":
            p_analytic = analytic_foldmean_pvalue(
                observed, np.full_like(observed, n_test_u),
            )
        else:
            p_analytic = np.full(observed.shape, np.nan)

        n = observed.size
        out_unit[w:w + n] = u
        out_p_gold[w:w + n] = p_gold
        out_p_sub_emp[w:w + n] = p_sub_emp
        out_p_sub_gpd[w:w + n] = p_sub_gpd
        out_p_analytic[w:w + n] = p_analytic
        w += n

    return pl.DataFrame({
        "unit_idx": out_unit[:w],
        "K_sub": np.full(w, K_sub, dtype=np.int32),
        "p_gold": out_p_gold[:w],
        "p_sub_emp": out_p_sub_emp[:w],
        "p_sub_gpd": out_p_sub_gpd[:w],
        "p_analytic": out_p_analytic[:w],
    })


# %%
_step(f"=== Section 1: per-(site, window) calibration on `{statistic}` ===")
section1_parts = []
for K_sub in K_subs:
    _step(f"  computing K_sub={K_sub} ...")
    section1_parts.append(calibrate_per_window(K_sub))
    _step(f"  K_sub={K_sub} done")
section1 = pl.concat(section1_parts)
section1.write_parquet(outdir / f"section1_{statistic}.parquet")
print(f"  → {outdir / f'section1_{statistic}.parquet'} ({section1.height} rows)")


# %%
def summarize(df: pl.DataFrame, *, label: str) -> pl.DataFrame:
    """For each (K_sub, p_gold threshold), compare each method's behavior."""
    rows: list[dict] = []
    thresholds = [0.05, 0.01, 1e-3, 1e-4]
    for K_sub in df["K_sub"].unique().sort().to_list():
        d = df.filter(pl.col("K_sub") == K_sub)
        for thr in thresholds:
            below = d.filter(pl.col("p_gold") <= thr)
            if below.height == 0:
                continue
            for method in ("p_sub_emp", "p_sub_gpd", "p_analytic"):
                vals = below[method].drop_nulls().drop_nans().to_numpy()
                if vals.size == 0:
                    continue
                # Of the things p_gold called significant at `thr`, how many
                # does the alternative also call significant at `thr`?
                tpr = float(np.mean(vals <= thr))
                # Median log10 ratio: how much does the method shift p?
                gold_vals = below["p_gold"].to_numpy()[: vals.size]
                med_log_ratio = float(
                    np.median(np.log10(np.clip(vals, 1e-12, 1.0))
                              - np.log10(np.clip(gold_vals, 1e-12, 1.0)))
                )
                rows.append({
                    "section": label,
                    "K_sub": K_sub,
                    "p_gold_threshold": thr,
                    "method": method,
                    "agreement_at_threshold": tpr,
                    "median_log10_p_offset": med_log_ratio,
                    "n_units_below": below.height,
                })
    return pl.DataFrame(rows)


s1_summary = summarize(section1, label="per_window")
print("\n=== Section 1 summary ===")
with pl.Config(tbl_rows=200, tbl_width_chars=200):
    print(s1_summary)
s1_summary.write_parquet(outdir / f"section1_summary_{statistic}.parquet")


# %% [markdown]
# ## Section 2 — peak (max-stat) calibration
#
# For each site, take the max statistic across windows per perm. This is
# the family-wise null distribution that downstream `null_standardized_peak_test`
# uses. Bootstrap-calibrate K_sub-budget GPD against the K_train empirical
# truth. Analytic is skipped — no clean closed form for the maximum of
# correlated AUCs across overlapping windows.

# %%
# Per-site, per-perm max over windows. Group by site_keys (no smin/smax)
# and reduce the perm-wise stat over windows. stats_mat is (n_units, K)
# where each row of `wide` is one (site, window) pairing.
site_key_rows = [
    tuple(row[k] for k in site_keys) for row in wide.select(site_keys).iter_rows(named=True)
]
site_to_units: dict[tuple, list[int]] = {}
for u, key in enumerate(site_key_rows):
    site_to_units.setdefault(key, []).append(u)
site_keys_ordered = list(site_to_units.keys())
site_unit_indices = [np.asarray(site_to_units[k]) for k in site_keys_ordered]
print(f"  n_sites = {len(site_keys_ordered)}")

# Max-over-windows stat per (site, perm): (n_sites, K)
maxstat_mat = np.full((len(site_keys_ordered), K), np.nan)
for s_idx, units in enumerate(site_unit_indices):
    block = stats_mat[units]  # (n_windows, K)
    # nanmax — some windows may have NaN (degenerate fold); ignore them.
    with np.errstate(invalid="ignore"):
        maxstat_mat[s_idx] = np.nanmax(block, axis=0)
print(f"  maxstat_mat shape = {maxstat_mat.shape}, "
      f"finite frac = {np.isfinite(maxstat_mat).mean():.4f}")


# %%
def calibrate_per_site_peak(K_sub: int) -> pl.DataFrame:
    rng_local = np.random.default_rng(rng_seed + 1)
    K_train = K - K_test
    n_sites = maxstat_mat.shape[0]
    cap = n_sites * K_test
    out_site = np.empty(cap, dtype=np.int32)
    out_p_gold = np.empty(cap, dtype=np.float64)
    out_p_sub_emp = np.empty(cap, dtype=np.float64)
    out_p_sub_gpd = np.empty(cap, dtype=np.float64)
    w = 0
    for s in tqdm(range(n_sites), desc=f"calibrate_per_site_peak K_sub={K_sub}",
                  unit="site", leave=False):
        nulls_full = maxstat_mat[s]
        if not np.isfinite(nulls_full).any():
            continue
        idx = rng_local.permutation(K)
        train_idx = idx[:K_train]
        test_idx = idx[K_train:K_train + K_test]
        null_train = nulls_full[train_idx]
        observed = nulls_full[test_idx]

        sub_idx = rng_local.choice(K_train, size=K_sub, replace=False)
        null_sub = null_train[sub_idx]

        p_gold = empirical_pvalue(observed, null_train)
        p_sub_emp = empirical_pvalue(observed, null_sub)
        p_sub_gpd = gpd_tail_pvalue(observed, null_sub, threshold_q=gpd_threshold_q)

        n = observed.size
        out_site[w:w + n] = s
        out_p_gold[w:w + n] = p_gold
        out_p_sub_emp[w:w + n] = p_sub_emp
        out_p_sub_gpd[w:w + n] = p_sub_gpd
        w += n

    return pl.DataFrame({
        "site_idx": out_site[:w],
        "K_sub": np.full(w, K_sub, dtype=np.int32),
        "p_gold": out_p_gold[:w],
        "p_sub_emp": out_p_sub_emp[:w],
        "p_sub_gpd": out_p_sub_gpd[:w],
        "p_analytic": np.full(w, np.nan),  # not defined for max-stat
    })


_step(f"=== Section 2: per-site peak (max-stat) calibration on `{statistic}` ===")
section2_parts = []
for K_sub in K_subs:
    _step(f"  computing K_sub={K_sub} ...")
    section2_parts.append(calibrate_per_site_peak(K_sub))
    _step(f"  K_sub={K_sub} done")
section2 = pl.concat(section2_parts)
section2.write_parquet(outdir / f"section2_{statistic}.parquet")
print(f"  → {outdir / f'section2_{statistic}.parquet'} ({section2.height} rows)")

s2_summary = summarize(section2, label="per_site_peak")
print("\n=== Section 2 summary ===")
with pl.Config(tbl_rows=200, tbl_width_chars=200):
    print(s2_summary)
s2_summary.write_parquet(outdir / f"section2_summary_{statistic}.parquet")


# %% [markdown]
# ## Recommendation
#
# Picks the smallest K_sub for which:
#   * **GPD vs empirical (Section 2 max-stat)** has agreement >= 0.95 at
#     `p_gold = 1e-3` AND median p-offset within ±0.5 log10 (factor of ~3).
#   * On Section 1 fold_mean, analytic null also has agreement >= 0.95 at
#     `p_gold = 1e-3` if you wanted to drop perms entirely.
#
# Both criteria are conservative — adjust if the application demands a
# tighter or looser tolerance.

# %%
def check_recommendation(summary: pl.DataFrame, label: str, method: str,
                         p_threshold: float = 1e-3,
                         agreement_floor: float = 0.95,
                         max_log_offset: float = 0.5):
    rows = summary.filter(
        (pl.col("section") == label)
        & (pl.col("method") == method)
        & (pl.col("p_gold_threshold") == p_threshold)
    ).sort("K_sub")
    if rows.height == 0:
        print(f"  no {label}/{method} rows at p_gold={p_threshold}")
        return None
    for r in rows.iter_rows(named=True):
        ok = (r["agreement_at_threshold"] >= agreement_floor
              and abs(r["median_log10_p_offset"]) <= max_log_offset)
        marker = "OK " if ok else "no "
        print(f"  {marker}{label}/{method}  K_sub={r['K_sub']:>5d}  "
              f"agreement={r['agreement_at_threshold']:.3f}  "
              f"median_log10_p_offset={r['median_log10_p_offset']:+.2f}")
        if ok:
            return r["K_sub"]
    return None


print("\n=== Recommendation ===")
print("Per-site peak (max-stat) — GPD vs empirical:")
peak_K = check_recommendation(s2_summary, "per_site_peak", "p_sub_gpd")
print("Per-(site, window) — GPD vs empirical:")
window_K = check_recommendation(s1_summary, "per_window", "p_sub_gpd")
print("Per-(site, window) — analytic Mann-Whitney vs empirical:")
analytic_K = check_recommendation(s1_summary, "per_window", "p_analytic")

print()
if peak_K is not None:
    print(f"GPD-tail with K_sub={peak_K} reproduces the K=10000 max-stat "
          f"empirical p<1e-3 calls at >=95% agreement.")
    print(f"  → Estimated speedup vs K=10000: {10000 / peak_K:.1f}x")
else:
    print("GPD never hit the agreement floor at p<1e-3 within the tested "
          f"K_subs ({K_subs}); try larger K_sub or relax thresholds.")

if analytic_K is not None:
    print("\nAnalytic Mann-Whitney null is well-calibrated — could go fully "
          "parametric (zero permutation refits).")
else:
    print("\nAnalytic null is NOT well-calibrated at extreme p — model-refit "
          "variance is non-negligible. Stick with permutations + GPD.")

print(f"\nArtifacts under {outdir}/")
