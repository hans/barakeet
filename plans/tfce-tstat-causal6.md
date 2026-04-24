# Plan: TFCE + t-stat extensions for causal6 peak-finding

## Context

The null-standardized max-stat permutation test is in place for all three
causal6 decoders (see earlier commits — `null_standardized_peak_test` in
[src/models/significance.py](../src/models/significance.py), feeding
`peak_summary.parquet` / `phon_peaks.parquet` via the three
peak/summarize rules).

Two known limitations drive this follow-up:

1. **Per-site saturation at max-stat.** At K=500 permutations with
   max-stat over W≈21 effective windows, the per-site p-value floor is
   `W_eff / (K+1) ≈ 0.042` on the behavior decoders. Real signal spans
   multiple adjacent windows (typical for smooth ECoG responses) and
   gets no credit for that breadth — max-stat only cares about the
   single hottest window. This hits the behavior decoders hard; the
   acoustic decoder is already constrained to a narrow
   pre-registered peak-search window (W≈3-4) so max-stat there is fine.

2. **Fold variance thrown away.** Current statistic is `fold_mean`. We
   have five fold-AUCs per (site, window) in both real and null. Real
   signal is typically consistent across folds (tight) while null runs
   scatter more, so a variance-normalized t-like statistic can beat
   fold-mean at discrimination. A demo notebook
   ([notebooks/causal6/demo_fold_variance_tstat.py](../notebooks/causal6/demo_fold_variance_tstat.py))
   is in flight to empirically validate the assumption — assume the
   user has sanity-checked this before you begin implementation, but
   the upgrade is cheap enough to include regardless.

This plan layers both fixes **alongside** the existing
max-stat-on-fold-mean output. Downstream consumers (neurometrics, FDR
aggregate) pick whichever method they want.

| Decoder(s)          | Emitted flavors per subject |
|---------------------|-----------------------------|
| acoustic            | `foldmean_maxstat` (existing), `tstat_maxstat` (new) |
| behavior (×2)       | `foldmean_maxstat` (existing), `tstat_maxstat` (new), `foldmean_tfce` (new), `tstat_tfce` (new) |

Acoustic picks up the t-stat upgrade because fold-variance discrimination
is a per-site win independent of the peak-search width. It doesn't need
TFCE — the peak_search window there is already narrow, so the max-stat
penalty is small and cluster structure isn't the bottleneck.

## Approach

### 1. Two new utilities in [src/models/significance.py](../src/models/significance.py)

**`fold_tstat_aggregate`** — aggregate per-fold scores to per-(site,
window[, perm]) with both `fold_mean` and `t_stat`. Decoder-agnostic;
caller supplies the centering value (0.5 for AUC, 0 for `diff`).

```python
def fold_tstat_aggregate(
    scores: pl.DataFrame,
    *,
    group_keys: Sequence[str],         # site_keys + window_keys[, perm_key]
    stat_col: str = "test_roc_auc",
    center: float = 0.5,
    std_floor: float = 0.01,
) -> pl.DataFrame:
    """Aggregate fold-wise scores to (fold_mean, fold_std, t_stat).

    t_stat = (fold_mean - center) / (max(fold_std, std_floor) / sqrt(n_folds))
    """
```

**`tfce_1d_per_site`** — apply 1D TFCE enhancement to a long-format
per-(site, window) statistic DataFrame, per site, per permutation
(pass `perm_key=None` for real, or the perm-index column for null).

```python
def tfce_1d_per_site(
    stats: pl.DataFrame,               # site_keys + window_keys + [perm_key?, stat_col]
    *,
    site_keys: Sequence[str],
    window_keys: Sequence[str] = ("smin", "smax"),
    perm_key: Optional[str] = None,
    stat_col: str = "statistic",
    E: float = 0.5,
    H: float = 2.0,
    dh: Optional[float] = None,        # default: (max_stat - 0) / 100 per group
    threshold: float = 0.0,            # stat values below this don't contribute
) -> pl.DataFrame:
    """
    Per-site (and per-permutation, if perm_key is given) 1D TFCE
    enhancement along window_keys. Windows are ordered by smin within
    each group. Returns long-format with stat_col replaced by the
    TFCE-enhanced value.

    Implementation: for each group, iterate thresholds h from dh to
    max(stat) in steps of dh; label contiguous runs above h via
    scipy.ndimage.label; add (extent^E * h^H * dh) to each window in
    the run. One-tailed (positive side only) — values below `threshold`
    are zeroed before TFCE.
    """
```

Implementation is a small Python loop per group using numpy +
`scipy.ndimage.label`. Cheap: W≈100 windows × ~100 threshold steps per
group × ~50 sites × (1 real + 500 perms) ≈ seconds. No GPU needed.

### 2. Cross-validate the custom TFCE against MNE (one-off sanity test)

In [tests/test_significance.py](../tests/test_significance.py), add a
test that constructs a fake 1D map that fits MNE's
`permutation_cluster_1samp_test(..., threshold=dict(start=0, step=0.2))`
API, runs both, and asserts per-window TFCE values match to ~1e-6. This
guards against silent bugs in our implementation and documents the
equivalence. The pipeline code never calls MNE — it's an awkward fit
because our null is already computed externally.

### 3. Rewire the three peak/summarize notebooks

**Acoustic** ([notebooks/causal6/acoustic_decoding_peaks.py](../notebooks/causal6/acoustic_decoding_peaks.py)):

1. Filter by `peak_search_smin/smax` as today (narrow acoustic window).
2. `fold_tstat_aggregate` on real and null (center = 0.5; statistic = AUC).
3. Call `null_standardized_peak_test` twice — once for
   `(fold_mean, maxstat)`, once for `(t_stat, maxstat)`.
4. Emit two parquets per subject:
   - `phon_peaks.parquet` (unchanged schema & contract; foldmean_maxstat).
   - `phon_peaks_tstat_maxstat.parquet` (same schema, computed on t_stat).
5. `phon_roc_auc_searchlight.parquet` unchanged.

**Behavior** ([notebooks/causal6/behavior_decoding_single_electrode_summarize.py](../notebooks/causal6/behavior_decoding_single_electrode_summarize.py),
[notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py](../notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py)):

1. Pair full+baseline (with-control only) as today.
2. Apply `_window_filter` / `_pair_and_filter` as today.
3. `fold_tstat_aggregate` on real and null (centers: 0 for
   with-control `diff`; 0.5 for hga_only AUC).
4. For each of four `(statistic, method)` combinations, build the
   real/null stats DataFrames and call the appropriate test:
   - `(fold_mean, maxstat)` → `null_standardized_peak_test` (current)
   - `(t_stat, maxstat)`    → `null_standardized_peak_test`
   - `(fold_mean, tfce)`    → `tfce_1d_per_site` on both, then
     `null_standardized_peak_test` on enhanced values
   - `(t_stat, tfce)`       → same pipeline with t_stat input
5. Write four parquets per subject:
   - `peak_summary.parquet` (unchanged — foldmean_maxstat)
   - `peak_summary_tstat_maxstat.parquet`
   - `peak_summary_foldmean_tfce.parquet`
   - `peak_summary_tstat_tfce.parquet`

   Each new file uses the same schema as the existing file — so
   downstream consumers (neurometrics, aggregate FDR) can read any of
   them with zero code change.

6. `peak_predictions.parquet` continues to be derived from
   `peak_summary.parquet` (the foldmean_maxstat peak windows) to keep
   existing peak-HGA-extraction consumers unchanged. Trial predictions
   at TFCE peaks are a follow-up.

### 4. Snakefile wiring — seven new per-decoder aggregate rules

Each peak/summarize per-subject rule gets additional outputs declared
(the notebook writes all of them). Then for each new per-subject file
type, add one aggregate rule per decoder, mirroring the existing
aggregate rule but reading the appropriate parquet:

```
acoustic_decoding_peaks_aggregate_tstat_maxstat
behavior_decoding_single_electrode_summarize_aggregate_tstat_maxstat
behavior_decoding_single_electrode_summarize_aggregate_foldmean_tfce
behavior_decoding_single_electrode_summarize_aggregate_tstat_tfce
behavior_decoding_single_electrode_hga_only_summarize_aggregate_tstat_maxstat
behavior_decoding_single_electrode_hga_only_summarize_aggregate_foldmean_tfce
behavior_decoding_single_electrode_hga_only_summarize_aggregate_tstat_tfce
```

The aggregate notebook
([notebooks/causal6/significance_aggregate.py](../notebooks/causal6/significance_aggregate.py))
is already parametrized on `result_paths` + `output_name` — no change
needed.

`causal6_all` target adds the seven new aggregate outputs as dependencies.

### 5. Config — no changes

`E=0.5`, `H=2` are the standard TFCE defaults; hard-coded as utility
kwarg defaults with overrides available per call. `dh` defaults to
`max(stat)/100` per group. `std_floor=0.01` is a small constant for the
t-stat denominator (rationale documented in utility docstring).

If any of these need to become tunable per-run, add them to config.yaml
as a follow-up — not needed for v1.

## Files to touch

### New
- `src/models/significance.py` — append `fold_tstat_aggregate`,
  `tfce_1d_per_site` to the existing file.
- Tests in `tests/test_significance.py` covering:
  - `fold_tstat_aggregate` correctness on synthetic data (mean/std/t
    values match hand computation; center and std_floor behave as
    documented).
  - `tfce_1d_per_site` matches MNE's TFCE on one representative 1D
    stat map to ~1e-6.
  - Monotonicity: TFCE preserves rank of isolated peaks and elevates
    broad clusters above narrow peaks of the same height.
  - NaN handling: NaN null entries at a window are dropped, not
    propagated through TFCE.

### Modified
- [notebooks/causal6/acoustic_decoding_peaks.py](../notebooks/causal6/acoustic_decoding_peaks.py) —
  add `t_stat` path and a second `phon_peaks_tstat_maxstat.parquet`
  output. No TFCE.
- [notebooks/causal6/behavior_decoding_single_electrode_summarize.py](../notebooks/causal6/behavior_decoding_single_electrode_summarize.py) —
  four flavors.
- [notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py](../notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py) —
  four flavors.
- [workflows/causal6.Snakefile](../workflows/causal6.Snakefile) —
  extra per-subject output paths on the three peak/summarize rules,
  seven new aggregate rules, updated `causal6_all` target.

### Unchanged
- `null_standardized_peak_test` — reused as-is.
- [notebooks/causal6/significance_aggregate.py](../notebooks/causal6/significance_aggregate.py) —
  reused as-is (already parametrized on `output_name`).
- Null-production rules (`*_null`) — out of scope; TFCE and t-stat are
  post-processing on their output.

## Verification

1. **Unit tests** as listed above, especially the MNE cross-validation
   of TFCE values.
2. **Smoke run** on one subject: confirm all four behavior parquets
   (and two acoustic parquets) appear and schemas match; confirm
   aggregate rules produce `*_all.parquet` per flavor; confirm
   p-value columns in all are in `[1/(K+1), 1]`.
3. **Comparison summary** on one real subject: tabulate per-site best
   p across methods. Expected pattern if the demo notebook confirms
   the fold-variance hypothesis: t-stat < fold-mean (more power);
   TFCE < max-stat at sites with broad smooth signal (more power);
   TFCE ≈ max-stat at sites with isolated sharp peaks.
4. **BH-FDR aggregate outputs**: count significant sites under each
   method. If TFCE + t-stat combined recovers meaningfully more
   behavior sites than foldmean_maxstat, the upgrade pays off and
   downstream analyses can opt in.
5. **Snakemake dry-run**:
   `snakemake -s workflows/causal6.Snakefile causal6_all --dry-run`
   — confirms the DAG is consistent with the new rules.

## Context to hand to the implementation agent

Before starting, read these in order to understand what already exists:

1. [src/models/significance.py](../src/models/significance.py) —
   existing `null_standardized_peak_test` is the workhorse the new
   utilities feed into.
2. [notebooks/causal6/acoustic_decoding_peaks.py](../notebooks/causal6/acoustic_decoding_peaks.py),
   [notebooks/causal6/behavior_decoding_single_electrode_summarize.py](../notebooks/causal6/behavior_decoding_single_electrode_summarize.py),
   [notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py](../notebooks/causal6/behavior_decoding_single_electrode_hga_only_summarize.py)
   — the three notebooks being extended, all with similar shape.
3. [workflows/causal6.Snakefile](../workflows/causal6.Snakefile) —
   current rule wiring for reference; the three peak/summarize rules
   and their aggregates sit near the bottom.
4. [notebooks/causal6/demo_fold_variance_tstat.py](../notebooks/causal6/demo_fold_variance_tstat.py)
   — demonstrates the t-stat rationale on real data; keep the same
   centering and std_floor conventions in `fold_tstat_aggregate`.
5. Relevant project memory:
   [feedback_no_local_runtime.md](/Users/jon/.claude/projects/-Users-jon-Projects-barakeet/memory/feedback_no_local_runtime.md)
   and
   [feedback_commit_messages.md](/Users/jon/.claude/projects/-Users-jon-Projects-barakeet/memory/feedback_commit_messages.md)
   — local uv env for smoke tests, brief commit messages preferred.

The demo notebook's result (whether fold-variance beats fold-mean at
discriminating real from null) should be checked **before** coding —
if the assumption fails, the t-stat flavor adds no value and can be
dropped from the plan, leaving just TFCE for behavior decoders. The
utility itself is still worth implementing.
