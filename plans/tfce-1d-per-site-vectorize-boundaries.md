# Vectorize group-boundary scan in `tfce_1d_per_site`

## Symptom

`stage1_gate` in the causal6 `*_null` notebooks is dominated by
`src/models/significance.py:tfce_1d_per_site` when the input null table
is large. With production K1=1000 + stride=2 the smoke script's gate
phase takes 20–40 min/subject; profiling shows the time is in the
pure-Python row scan at lines 408–417 (current code), not in
`_tfce_1d` itself.

```
for i in range(1, n + 1):
    at_end = i == n
    if not at_end:
        changed = any(col[i] != col[i - 1] for col in group_cols)
    if at_end or changed:
        enhanced[start:i] = _tfce_1d(stat_np[start:i], ...)
        start = i
```

This walks every one of N≈40M rows in Python, doing a per-step
`any(col[i] != col[i-1] for col in group_cols)` over 4–5 group-key
columns (some of which are object/Utf8 arrays). The number of TFCE
calls (one per group) is fine — the boundary detection is the cost.

## Goal

Replace the per-row Python boundary scan with a vectorized one. **No
change to numerical output**: the per-group `_tfce_1d(...)` call,
its arguments, and the slice it sees must be identical to the current
implementation. The cross-validation test in
`tests/test_significance.py` (asserting equivalence with MNE's
`_find_clusters` per the `_tfce_1d` docstring) and any other existing
TFCE tests must continue to pass without modification.

## Fix

After the existing sort, derive group-start indices in O(N) numpy:

1. Compute a per-row integer group id with polars' run-length-id
   primitive over the sort key:
   ```python
   group_id = (
       sorted_df
       .select(pl.struct(group_keys).rle_id().alias("_gid"))["_gid"]
       .to_numpy()
   )
   ```
   Because `sorted_df` is sorted by `group_keys + [order_col]`,
   identical group tuples are contiguous and `rle_id()` returns
   0,0,…,1,1,…,2,…  — exactly the boundaries the current loop
   recovers.

2. Derive group boundaries vectorially:
   ```python
   if n == 0:
       boundaries = np.empty(0, dtype=np.int64)
   else:
       boundaries = np.concatenate((
           [0],
           np.flatnonzero(np.diff(group_id)) + 1,
           [n],
       ))
   ```
   `boundaries[k]` is the start index of group k; `boundaries[-1]==n`
   is the sentinel end. `len(boundaries) - 1 == n_groups`.

3. Loop only over groups (not rows):
   ```python
   enhanced = np.empty(n, dtype=np.float64)
   for k in range(len(boundaries) - 1):
       s, e = boundaries[k], boundaries[k + 1]
       enhanced[s:e] = _tfce_1d(
           stat_np[s:e], E=E, H=H, dh=dh, threshold=threshold,
       )
   ```

4. The `else` branch (no `group_keys`) is unchanged.

For the EC250-shaped null (≈40M rows, ≈336k groups) this collapses a
~40M-iteration Python loop to ≈336k calls — same as today's `_tfce_1d`
call count, just without the per-row boundary check overhead. Expected
gate runtime: under a minute (numpy + polars `rle_id` + the existing
`_tfce_1d` work, which dominates after the fix).

## Invariants to preserve

- **Sort order**: still `group_keys + [order_col]`. Do not add or remove
  sort keys; do not change `descending`.
- **Per-group slice**: `_tfce_1d` must receive `stat_np[s:e]` for the
  exact same `[s, e)` ranges the current loop produces. (Equivalent by
  construction: `rle_id` increments precisely where the current
  `any(col[i] != col[i-1])` would fire.)
- **`_tfce_1d` arguments**: same `E, H, dh, threshold`. Adaptive
  `dh=None` continues to be resolved per-group inside `_tfce_1d`
  using that slice's `max_stat/100`.
- **Output DataFrame**: same schema, same row order (the sorted order),
  same `stat_col` values.
- **NaN/empty-group handling**: `_tfce_1d` already handles empty +
  all-NaN slices (`out = np.zeros_like(stat)`); the fix doesn't change
  what slices are passed in.

## Files

- `src/models/significance.py` — replace lines ≈403–417 of
  `tfce_1d_per_site` (the `if group_keys: …` branch). Keep the
  surrounding `sorted_df` setup, `enhanced` allocation, and final
  `with_columns(...)` return as-is.
- `tests/test_significance.py` — add a regression test:
  - Build a small fixture with 3 sites × 4 perms × 8 windows, mixed
    NaN windows in one site, deterministic random `statistic`.
  - Run the *new* `tfce_1d_per_site` and an inlined copy of the *old*
    Python-loop variant on the same input.
  - Assert `np.array_equal(new[stat_col].to_numpy(),
    old[stat_col].to_numpy())` (exact equality — same input, same
    `_tfce_1d` calls).
  - Existing MNE-equivalence test stays untouched.

## Out of scope

- Vectorizing `_tfce_1d` itself across groups. Per-group call count is
  unchanged after this fix; if `_tfce_1d` becomes the bottleneck later,
  that's a separate change with its own correctness story
  (per-group adaptive `dh` makes batched threshold sweeps non-trivial).
- Changing TFCE math (E, H, threshold semantics). The fix is a pure
  refactor of group-boundary detection.

## Verification checklist

- `pytest tests/test_significance.py` passes (existing + new
  regression test).
- Re-run `scripts/smoke_causal6_null_oom.py` on EC248 with K1=1000,
  stride=2 — gate phase completes in <2 min (vs. current 20–40 min)
  with identical `borderline=N/M` numbers.
- `gate_log` parquet from one production `*_null` rule is byte-equal
  to a pre-fix run on the same subject (same seeds) — strongest
  end-to-end equivalence check.
