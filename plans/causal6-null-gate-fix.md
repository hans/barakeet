# causal6 `*_null` two-stage gate: OOM root-cause + fix spec

## Symptom

`behavior_decoding_single_electrode_hga_only_null` SIGKILLs (OOM) on
EC248 (56 speech-resp. electrodes), EC250 (58), and EC253 (80) on a
1 TB-RAM box. Smaller subjects (e.g. EC270 with 42 speech-resp.
electrodes) complete. The kill happens after stage 1 finishes — the
papermill log shows ~12/16 cells done at the time of death, putting it
mid-stage-2 (or at the post-stage-2 `pl.concat` / `write_parquet`).

`journalctl` confirmed kernel OOM (`sisamddaemon` is just the alloc
caller at the moment the killer fired, not the victim — the victim is
the python3.11 papermill process).

Stage 1 alone uses 15-30% of system RAM (~150-300 GB). The blow-up
happens during/after stage 2.

## Root cause: stage-1 → stage-2 escalation gate is non-selective

The gate at `src/models/causal6_adaptive_null.py:168-286` (`stage1_gate`)
is supposed to filter sites whose K1 evidence rules out significance,
so stage 2 (with K2 = 9× K1) only refits the borderline minority. In
practice it lets through ~100% of sites:

- EC270 escalation log: **42/42 (100%)** sites escalated.
- All failing subjects also escalate ~100%, but at higher electrode
  counts the all-sites stage-2 work no longer fits in RAM.

### Why the current gate is too loose

The implemented condition is:

> escalate if `min over (windows × flavors) of pointwise_p ≤ p_max`,
> with `p_max = 0.20`, K1 = 1000.

Two compounding problems:

1. **The bound the docstring claims doesn't match what the code
   computes.** The docstring (`causal6_adaptive_null.py:24-28`)
   justifies the gate via
   `corrected_maxstat_p ≥ pointwise_p[real_peak_window]`, so it'd
   escalate when `pointwise_p[peak] ≤ p_max`. But the code uses
   `min over all windows of pointwise_p`, which is strictly ≤
   `pointwise_p[peak]`. Looser than even the stated bound.

2. **Even the stated bound is permissive given window count.** With
   `peak_search_smin=0`, `peak_search_smax=290`, `window_size=15`,
   `stride=2`: ~138 windows per (site, group). For a true null site,
   the peak window's pointwise_p concentrates near `1/W ≈ 0.007`
   (the peak window is, by definition, where real ranks high among
   the per-window null) — so `pointwise_p[peak] ≤ 0.20` is true with
   ~probability 1.

Result: stage-2 effectively re-runs at K2 = 9000 perms × all
electrodes × 5 folds × all (phoneme_pair, word_end) groups. That's
roughly 9× the stage-1 null held in RAM (because spilled stage-2
shards are joined+collected into a single `null_stage2` frame), then
`pl.concat([null_stage1, null_stage2])` materialises the merged frame
on top of stage-1's still-resident allocation. EC253 at 80 electrodes
hits ~1 TB peak; smaller subjects squeak through.

## Proposed fix (gate logic)

Replace `min over windows of pointwise_p` with the **K1-derived
corrected p** that the production summarize step would compute, then
escalate only sites whose K1-corrected p sits in an ambiguous band
around the eventual decision threshold.

For the behavior-HGA-only flavor list
(`FLAVORS_BEHAVIOR_HGA_ONLY` in `src/models/causal6_aggregates.py:91-97`):

| stat_col   | apply_tfce | tfce_threshold |
|------------|-----------:|---------------:|
| fold_mean  | False      | n/a            |
| t_stat     | False      | n/a            |
| fold_mean  | True       | 0.5            |
| t_stat     | True       | 0.0            |

Per flavor, per site, at K1:

- max-stat flavors (`apply_tfce=False`):
  - `null_max[k] = max_w stat[k, w]` for k = 0..K1-1
  - `real_max = max_w real_stat[w]`
  - `corrected_p = (#{null_max ≥ real_max} + 1) / (K1 + 1)`
- TFCE flavors: same, but on the output of
  `tfce_1d_per_site(...)` along windows (the gate already applies
  this for the existing pointwise path; reuse).

Per site: `min_corrected_p_K1 = min over flavors of corrected_p`.

Escalate if `min_corrected_p_K1 ≤ p_max` (default still `0.20`,
re-interpreted as a corrected-p threshold).

Optional: also escalate if `≤ p_max / k` for some `k ~ 3` to fence-sit
*highly* significant sites whose K1 estimate is already past the
decision threshold but whose SE at K1 is wide enough to warrant K2.
Personally I'd start without this lower band — the very-significant
sites are cheap to skip, and stage 2 is for resolving the borderline.

### Public-knob choice

Two options; pick one before implementation:

- **A. Rename** `escalate_pointwise_p_max` →
  `escalate_corrected_p_max` everywhere (`config.yaml`,
  `causal6.Snakefile`, all five `*_null.py` notebooks). Cleanest;
  forces a re-think of the value.
- **B. Keep the name, redefine semantics.** No config churn; the
  value 0.20 just means something different now.

## Files involved

- `src/models/causal6_adaptive_null.py` — `stage1_gate` (rewrite),
  optional helper for K1 max-stat correction. Docstring at top of
  file also needs updating (section "Why `pointwise_p` (not the
  corrected p-value) at the gate" no longer applies).
- `src/models/causal6_aggregates.py` — no schema change; reuse
  `FLAVORS_*` and `tfce_1d_per_site`.
- Five `*_null.py` notebooks under `notebooks/causal6/` — the
  `borderline_keys, gate_log = stage1_gate(...)` call signature
  stays the same; only the parameter name changes if option A.
- `config.yaml` — `causal6.escalate_pointwise_p_max` → new name (A).
- `workflows/causal6.Snakefile` — config-key references (A).

## Smoke test brief (what the other agent should write)

The goal is to **measure peak RSS of the gate + concat path on the
remote, with realistic data shapes, before vs after the fix**, without
spending GPU time on actual permutation refits.

### Approach

Mock-generate the two parquet inputs that the `*_null.py` notebook
consumes mid-pipeline, then run the same code path the notebook runs
(stage-1 aggregate → gate → stage-2 simulated null materialisation →
final concat → write).

Skipping the GPU loops is what makes this cheap. The OOM happens in
polars / numpy land, not on the GPU, so a faithful reproduction only
needs to mock the *score frames* the GPU loop produces.

### Schemas to mock

For a target subject (e.g. EC253), generate two parquet frames matching
what `_run_behavior_core_permutations` (real run) and the GPU
permutation loop produce:

**`real_scores`** (one row per `(electrode, phoneme_pair, word_end,
window, fold)`):

| column         | dtype  | notes |
|----------------|--------|-------|
| subject        | Utf8   | constant |
| phoneme_pair   | Utf8   | from epochs metadata; per-subject set |
| word_end       | Utf8   | per phoneme_pair, from metadata |
| electrode_idx  | Int64  | from `find_speech_responsive` csv |
| smin, smax     | Int64  | window indices (see `make_windows`) |
| model          | Utf8   | always `"full"` for HGA-only |
| fold           | Int32  | 0..n_folds-1 |
| test_roc_auc   | Float64 | uniform[0.4, 0.6] is fine |
| n_train, n_test, n_iter | Int64/Int32 | doesn't matter |
| converged      | Bool   | doesn't matter |

**`null_scores_stage1`** (one row per
`(electrode, phoneme_pair, word_end, window, perm, fold)`):

Same columns + `permutation_idx: Int64` (0..K1-1). `test_roc_auc` from
uniform[0.4, 0.6] under the null. K1 = 1000.

### Dimensions to use

Pull from the actual subject so shapes match production exactly:

```python
import mne
from src.data import add_metadata_features
from src.models.causal6 import make_windows
import pandas as pd

subject = "EC253"
epochs = mne.read_epochs(f"outputs/epochs_preprocessed/{subject}_epo.fif", verbose=False)
epochs.metadata = add_metadata_features(epochs.metadata)
md = epochs.metadata
electrode_df = pd.read_csv(f"outputs/causal5/find_speech_responsive/{subject}_results.csv")
electrode_idxs = sorted(
    electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"].unique().astype(int)
)
windows = make_windows(min_sample=1, max_sample=epochs.times.shape[0],
                       window_size=15, stride=2)
groups = []
for pp in sorted(md.phoneme_pair.dropna().unique()):
    for we in sorted(md.word_end[md.phoneme_pair == pp].dropna().unique()):
        groups.append((pp, we))
```

For EC253 expect roughly: ~80 electrodes × ~138 windows × |groups| ×
5 folds rows in `real_scores`, and that × K1=1000 in
`null_scores_stage1`. The test should verify the row count it produces
matches what the GPU loop would have produced (a sanity check before
proceeding).

### What to invoke

Mirror the `*_null.py` notebook flow in `notebooks/causal6/behavior_decoding_single_electrode_hga_only_null.py:122-209`:

```python
from src.models.causal6_aggregates import (
    FLAVORS_BEHAVIOR_HGA_ONLY,
    SITE_KEYS_BEHAVIOR_HGA_ONLY,
    aggregate_behavior_hga_only,
)
from src.models.causal6_adaptive_null import (
    filter_null_to_borderline,
    stage1_gate,
)
import polars as pl

real_agg, null_agg_stage1 = aggregate_behavior_hga_only(
    real_scores, null_stage1,
    epoch_tmin=-0.4, epoch_sfreq=100, behav_peak_post_offset_s=0.2,
    peak_search_smin=0, peak_search_smax=290,
)
borderline_keys, gate_log = stage1_gate(
    real_agg, null_agg_stage1,
    site_keys=SITE_KEYS_BEHAVIOR_HGA_ONLY,
    flavors=FLAVORS_BEHAVIOR_HGA_ONLY,
    p_max=0.20,
)
print(f"borderline: {len(borderline_keys)} / {gate_log.height}")
```

Then **simulate stage 2's null frame** without GPU work — generate a
mock `null_stage2` of the size that stage 2 *would* produce given
`borderline_keys`:

```python
# Per the notebook's stage-2 logic, stage 2 runs perms across the
# borderline electrodes' (any phoneme_pair × word_end) and is then
# filtered down to the exact borderline (electrode, phoneme_pair,
# word_end) keys. Mock that by directly generating null rows for the
# borderline keys × n_windows × K2 × n_folds.
borderline_subset = (
    real_scores.join(
        pl.DataFrame(
            list(borderline_keys),
            schema=SITE_KEYS_BEHAVIOR_HGA_ONLY,
            orient="row",
        ),
        on=SITE_KEYS_BEHAVIOR_HGA_ONLY, how="semi",
    )
    .select(["subject", "phoneme_pair", "word_end", "electrode_idx",
             "smin", "smax", "model", "fold", "n_train", "n_test"])
    .unique()
)
K2 = 9000
null_stage2 = (
    borderline_subset.join(
        pl.DataFrame({"permutation_idx": pl.arange(0, K2, eager=True)}),
        how="cross",
    )
    .with_columns(test_roc_auc=pl.lit(0.5) + pl.lit(0.05) * pl.col("permutation_idx").cast(pl.Float64) / K2)  # any deterministic fill
)
null_scores = pl.concat([null_stage1, null_stage2])
null_scores.write_parquet("/tmp/smoke_null_scores.parquet")
```

The smoke run should also exercise `filter_null_to_borderline` over a
`pl.scan_parquet` view of stage-2 shards on disk — that's the actual
notebook path, and polars' lazy collect over many small parquets is
itself a memory suspect. To do that faithfully, write `null_stage2` as
multiple small shards (e.g. 4500 shards per group, mirroring K2 / 10
chunks × 5 folds × |groups|), then `pl.scan_parquet(...).collect()`.

### What to measure

Track peak RSS at these checkpoints (use `psutil.Process().memory_info().rss`
at each line, plus `/usr/bin/time -v` on the wrapping process for the
overall peak):

1. After loading mocked `real_scores` + `null_scores_stage1`.
2. After `aggregate_behavior_hga_only`.
3. After `stage1_gate` — also report `len(borderline_keys)` /
   `gate_log.height`.
4. After mocking + writing stage-2 shards.
5. After `pl.scan_parquet(...).join(...semi...).collect()`.
6. After `pl.concat([null_stage1, null_stage2])`.
7. After `null_scores.write_parquet(...)`.

Report each subject's results as a single-line summary:

```
EC253 borderline=N/M peak_rss_GB={loaded, agg, gate, mock2, collect, concat, write}
```

### Targets / exit criteria

- **Before the fix**: expect `borderline=M/M` (100%), peak RSS approaches
  or exceeds 1 TB on EC253. Optional: cap K2 at e.g. 1000 in the smoke
  test to keep the *before* run from itself OOMing — adjust the
  numbers proportionally; what matters is that we observe the same
  scaling pattern.
- **After the fix**: expect `borderline ≪ M`, peak RSS falls
  ~proportionally with the borderline count.

### Subjects to include

Run the smoke at least on EC270 (current pass), EC253 (current fail),
and ideally EC248 + EC250 + EC282. The before/after comparison should
hold across all of them; the gate fix should drop the borderline count
on every subject, but the memory drop will be most visible on the
biggest electrode sets.

### Non-goals for this smoke

- No GPU. No real refitting. No actual statistical correctness check
  on the gate output — that comes in a separate test (synthetic null
  data with known signal/no-signal sites, asserting escalation rates
  match `p_max`).
- No need to run the full Snakemake rule. Direct python script that
  imports the gate functions is fine.
