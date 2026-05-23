# causal46_joined — Behavioral & Ganong Decoders Restricted to AS Sites

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Linear:** TODO — create as sibling of the causal6 stage-3 task.

**Goal:** Re-run the behavior and ganong decoders (full + HGA-only, decode + null + summarize + aggregate) restricted to **acoustic-significant (AS) electrodes only**, so the per-electrode FDR family is small enough that the cross-pair behavioral tests at each AS site (all 3 phoneme_pairs, not just the one that was acoustic-sig) have headroom to survive correction.

**AS definition:** uncorrected `p_value < 0.05` in `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet` (v1 foldmean_maxstat flavor — **not** tstat or TFCE), collapsed to electrode level via OR over `phoneme_pair`.

**Output root:** `outputs/causal46_joined/`. Everything below that root is new and parallel to the existing `outputs/causal6/` tree.

**Snakefile composition:** new `workflows/causal46_joined.Snakefile` that `include:`s the causal6 file and defines 20 new `joined_*` rules (decode + null + summarize + aggregate × full + hga_only × behavior + ganong) plus 1 checkpoint to assemble the AS-filter electrode CSVs.

**Tech Stack:** Python, polars, torch (GPU), Snakemake checkpoints.

---

## Context

The current causal6 pipeline runs behavioral + ganong decoders on all ~80 speech-responsive electrodes per subject. After hierarchical Simes + BH + Holm in the aggregate, the family is large enough that mid-strength behavioral effects don't clear correction.

If we already know which sites carry an acoustic response, the interesting test is: at those same sites, what does the behavioral decoder reveal — not only for the matching phoneme_pair, but for all 3? "AS site for `dn` that also shows behavioral significance for `bm`" is a different scientific claim than the current per-site-per-pair test.

This pass keeps the same decoder code and config; only the electrode filter and output directory change. The acoustic pipeline itself (decode, null, peaks, aggregate) is **reused unchanged** — its `phon_peaks_all.parquet` is the input that defines AS.

### Existing causal46_joined work (out of scope here)

`notebooks/causal46_joined/{as_reconciliation,star_plots,trial_balance_index}.py` already exist as manual / one-off analyses. They are **not** consumed by this pipeline. `canonical_AS_sites.csv` is not used either — this pipeline takes its AS definition straight from `phon_peaks_all.parquet` to keep the source deterministic and Snakemake-trackable.

---

## File Structure

**Create:**
- `workflows/causal46_joined.Snakefile` — `include:`s `causal6.Snakefile`; defines 1 checkpoint + 20 new `joined_*` rules + a default `causal46_joined_all` target.
- `notebooks/causal46_joined/prepare_as_electrode_filter.py` — checkpoint notebook: read `phon_peaks_all.parquet` + per-subject `find_speech_responsive/{subject}_results.csv`; emit per-subject CSVs with a new `acoustic_significant` boolean column alongside the existing `speech_responsive`; emit a manifest `subjects_with_as.txt` listing subjects with ≥1 AS electrode.
- **8 copied decoder/null notebooks** under `notebooks/causal46_joined/`, one per causal6 source:
  - `behavior_decoding_single_electrode.py`
  - `behavior_decoding_single_electrode_hga_only.py`
  - `behavior_decoding_single_electrode_null.py`
  - `behavior_decoding_single_electrode_hga_only_null.py`
  - `ganong_decoding_single_electrode.py`
  - `ganong_decoding_single_electrode_hga_only.py`
  - `ganong_decoding_null.py`
  - `ganong_decoding_hga_only_null.py`

  Each copy is byte-identical to its causal6 source **except** the single line `electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"]` becomes `electrode_df.loc[electrode_df.acoustic_significant & electrode_df.speech_responsive, "electrode_idx"]` (the AND keeps the AS criterion conservative — see "AS definition" in the Goal). No parameter; the AS gate is hardcoded so the file is self-documenting via grep.

  > **Sync discipline:** if a causal6 decoder notebook gets touched in the future (e.g. a new model knob), the corresponding `causal46_joined/` copy must be re-synced by hand. Flag this in the joined notebook headers with a short "# Sync source: notebooks/causal6/<filename>" comment so the relationship is grep-able. Keep the copies otherwise verbatim — no per-pipeline divergence.

**Not modified:**
- The `summarize`/`aggregate` notebooks. They don't read the electrode CSV directly (summarize joins on whatever sites are in the scores parquet; aggregate uses electrode CSVs only for the ROI lookup, which is decoder-input-blind). Joined rules point them at the joined per-subject parquets and let them work unchanged.
- Any causal6 notebook. The existing pipeline is untouched by this plan.

---

## Design

### Checkpoint: `prepare_as_electrode_filter`

```python
checkpoint prepare_as_electrode_filter:
    input:
        phon_peaks_all   = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        electrode_csvs   = expand("outputs/causal6/find_speech_responsive/{subject}_results.csv",
                                  subject=config["data"]["subjects"]),
        notebook         = "notebooks/causal46_joined/prepare_as_electrode_filter.py",
    output:
        notebook         = "outputs/causal46_joined/electrodes_as_filtered/notebook.ipynb",
        manifest         = "outputs/causal46_joined/electrodes_as_filtered/subjects_with_as.txt",
        csvs             = expand("outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv",
                                  subject=config["data"]["subjects"]),
    parameters:
        as_p_threshold   = 0.05,
        ...
```

Notebook body sketch:

```python
phon = pl.read_parquet(phon_peaks_path).filter(pl.col("p_value") < as_p_threshold)
as_electrodes_per_subj = (
    phon.group_by(["subject", "electrode_idx"])
        .agg(pl.lit(True).alias("acoustic_significant"))
)
manifest = []
for csv_path in electrode_csv_paths:
    sr = pd.read_csv(csv_path)
    subj = sr.subject.iloc[0]
    as_set = as_electrodes_per_subj.filter(pl.col("subject") == subj)["electrode_idx"].to_list()
    sr["acoustic_significant"] = sr.electrode_idx.isin(as_set) & sr.speech_responsive
    sr.to_csv(out_dir / f"{subj}_results.csv", index=False)
    if sr.acoustic_significant.any():
        manifest.append(subj)
write_text(out_dir / "subjects_with_as.txt", "\n".join(manifest))
```

Notes:
- `acoustic_significant` is AND'd with `speech_responsive` so we never include an electrode that wasn't even speech-responsive (defensive).
- The OR over `phoneme_pair` is implicit in the `group_by(["subject", "electrode_idx"])` collapse — any sig row in any pair flips the electrode to True.
- Manifest is the list of subjects with ≥1 AS electrode; consumed by aggregate-rule input functions to skip empty subjects.

### Empty-subject handling

Decoder/null/summarize rules will **crash** if invoked on a subject with zero AS electrodes (existing notebooks don't guard `electrode_idxs = []`). Per the user's choice ("skip silently in aggregates only"), aggregate-rule input lists are built via a Snakemake-checkpoint-driven function:

```python
def get_joined_summarize_paths(template):
    """Build aggregate input list from the AS-subjects manifest."""
    def _fn(wildcards):
        ckpt = checkpoints.prepare_as_electrode_filter.get().output.manifest
        with open(ckpt) as f:
            subjects = [s.strip() for s in f if s.strip()]
        return [template.format(subject=s) for s in subjects]
    return _fn
```

So joined aggregates only request per-subject parquets for AS-positive subjects — decoder/null/summarize rules are never invoked for empty subjects in the canonical DAG. If a user manually targets a joined decode rule for a zero-AS subject, it crashes visibly (acceptable per user spec).

### Joined rule pattern (copy-paste with `joined_` prefix)

Each joined rule is a near-mechanical clone of its causal6 counterpart, with three changes:
1. `input.notebook` → the `notebooks/causal46_joined/<basename>.py` copy (with the hardcoded AS gate).
2. `input.electrodes` → `outputs/causal46_joined/electrodes_as_filtered/{subject}_results.csv` (output of the checkpoint).
3. All `output` paths route under `outputs/causal46_joined/{decoder_name}/...`.

Aggregate-rule changes:
4. `input.result_paths` uses `get_joined_summarize_paths(...)` rather than `expand(...)` on the full subjects list.
5. `input.all_electrode_dfs` uses the AS-filtered CSVs (so the ROI lookup happens on AS-only electrodes — but as noted in the Aggregate FDR section below, the actual family-size restriction comes from the per-subject parquets, which only contain AS rows anyway).

### Aggregate FDR semantic (clarification)

`significance_aggregate.py`'s hierarchical branch already does the right thing: it Simes-collapses across phoneme_pair × word_end per (subject, electrode_idx), then BH across electrodes. The family size is `len(elec_simes)` — the number of unique (subject, electrode_idx) in the in-family rows. In the joined pipeline, the per-subject summarize parquet **only contains AS electrodes** (since decode only ran there), so the family naturally equals AS ∩ ROIs after `restrict_to_rois`. No changes to `significance_aggregate.py`.

### Rule inventory (20 new rules + 1 checkpoint)

| Decoder | decode | null | summarize | aggregate flavors |
|---|---|---|---|---|
| behavior_full | joined_behavior_decoding_single_electrode | joined_behavior_decoding_single_electrode_null | joined_behavior_decoding_single_electrode_summarize | joined_behavior_decoding_single_electrode_summarize_aggregate{,_tstat_maxstat,_foldmean_tfce,_tstat_tfce} |
| behavior_hga_only | joined_behavior_decoding_single_electrode_hga_only | joined_behavior_decoding_single_electrode_hga_only_null | joined_behavior_decoding_single_electrode_hga_only_summarize | joined_behavior_decoding_single_electrode_hga_only_summarize_aggregate{,_tstat_maxstat,_foldmean_tfce,_tstat_tfce} |
| ganong_full | joined_ganong_decoding_single_electrode | joined_ganong_decoding_null | joined_ganong_decoding_summarize | joined_ganong_decoding_summarize_aggregate |
| ganong_hga_only | joined_ganong_decoding_single_electrode_hga_only | joined_ganong_decoding_hga_only_null | joined_ganong_decoding_hga_only_summarize | joined_ganong_decoding_hga_only_summarize_aggregate |

Counts: 4 decoder + 4 null + 4 summarize + (4+4+1+1) = 10 aggregate → 22. Plus 1 checkpoint. Plus 1 `causal46_joined_all` target.

### `causal46_joined_all` target

```python
rule causal46_joined_all:
    """Default target: run AS-filter + all joined aggregates."""
    input:
        "outputs/causal46_joined/electrodes_as_filtered/subjects_with_as.txt",
        # 10 aggregate _all.parquets...
```

---

## Tasks

### A. Checkpoint + AS-filter notebook
1. Write `notebooks/causal46_joined/prepare_as_electrode_filter.py` per the sketch above. Param: `as_p_threshold=0.05`. Inputs: `phon_peaks_all.parquet` + per-subject `find_speech_responsive/{subject}_results.csv`. Outputs: per-subject `*_results.csv` (full schema + new `acoustic_significant` column) + `subjects_with_as.txt`.
2. Add the `checkpoint prepare_as_electrode_filter` rule to `workflows/causal46_joined.Snakefile`.
3. Test (CPU-light): hand-fabricate a tiny phon_peaks_all parquet + 2-subject find_speech_responsive CSVs; assert the output CSVs have `acoustic_significant=True` only for sites with at least one phon_pair below threshold AND `speech_responsive=True`, and that the manifest contains only AS-positive subjects.

### B. Copy 8 decoder/null notebooks into `notebooks/causal46_joined/`
For each of the 8 source files in `notebooks/causal6/` listed in **File Structure**:

1. Copy the file verbatim into `notebooks/causal46_joined/`.
2. Insert a `# Sync source: notebooks/causal6/<filename>` comment in the top-of-file markdown cell so the relationship is grep-able.
3. Change the one line `electrode_df.loc[electrode_df.speech_responsive, "electrode_idx"]` → `electrode_df.loc[electrode_df.acoustic_significant & electrode_df.speech_responsive, "electrode_idx"]`.

No notebook parameters are added. The AS gate is hardcoded so consumers reading the file see exactly what the decoder runs on. Causal6 notebooks are untouched.

Acceptance: a `diff -u notebooks/causal6/X.py notebooks/causal46_joined/X.py` shows exactly two changes per file: the sync-source comment and the electrode filter line.

### C. New `workflows/causal46_joined.Snakefile`
1. Header: `include: "causal6.Snakefile"`. Re-import any helpers the joined rules need (`run_notebook`, `run_notebook_with_gpu`, `_load_reg_lambda`, `C6`).
2. Define the `prepare_as_electrode_filter` checkpoint.
3. Define a top-level helper:
   ```python
   def joined_aggregate_paths(template):
       def _fn(wildcards):
           ckpt = checkpoints.prepare_as_electrode_filter.get().output.manifest
           ...
       return _fn
   ```
4. Copy-paste the 22 rules from causal6.Snakefile with the 5 prefix/path/parameter changes listed under "Joined rule pattern" above. Each rule is mechanical; no logic changes.
5. Define `rule causal46_joined_all`.

### D. Smoke test on one subject
1. Run the checkpoint on production data: `uv run snakemake --snakefile workflows/causal46_joined.Snakefile outputs/causal46_joined/electrodes_as_filtered/subjects_with_as.txt`.
2. Sanity-check: per-subject CSV `acoustic_significant` count matches the count of unique AS electrodes from `phon_peaks_all.parquet`.
3. Run `joined_behavior_decoding_single_electrode_null` for one AS-positive subject (e.g., EC282) at smoke K (`--config causal6.n_permutations_stage1=100 causal6.n_permutations_stage2=100 causal6.n_permutations_stage3=100`).
4. Run the corresponding aggregate; confirm `electrode_q_value`/`electrode_significant` columns appear and family size matches AS-in-ROIs from the manifest.

### E. Production run
After D passes:
1. `uv run snakemake --snakefile workflows/causal46_joined.Snakefile causal46_joined_all`.
2. Burns the same GPU budget per AS electrode as causal6 per speech-responsive electrode, but at ~1/4 the electrode count (rough estimate; depends on AS rate). Expected wall time ≈ 25-30% of the causal6 behavior/ganong null re-run.

---

## Acceptance Criteria

- `prepare_as_electrode_filter` test passes; per-subject CSV `acoustic_significant` count == unique AS electrodes per subject.
- Existing causal6 behavior/ganong rules produce bit-identical outputs after the `electrode_column` parameter is added (defaults preserve behavior).
- `outputs/causal46_joined/electrodes_as_filtered/subjects_with_as.txt` exists and lists 8-10 of the 10 subjects (typical AS yields).
- Joined per-subject summarize parquets only contain AS-electrode rows.
- Joined aggregate parquets have populated `electrode_q_value`/`electrode_significant` for AS-in-ROI sites; the BH family size equals the AS-in-ROI count from the manifest.
- For at least one subject: count of `significant=True` rows in the joined behavior_with_control aggregate ≥ count in the causal6 aggregate, scaled by family-size ratio (rough sanity).

---

## Followups / Out of Scope

- prepare_neurometrics / A_neurometrics fork — deferred.
- canonical_AS_sites.csv integration / manual-override workflow — deferred. If you want manual curation in the loop, replace the checkpoint's input with `outputs/causal46_joined/canonical_AS_sites.csv` and collapse to electrode level there.
- TFCE / tstat AS definitions — out of scope. Switch via a config flag if you want a sensitivity sweep.
- AS threshold sweep (p<0.01, p<0.10) — out of scope; add as a parametrized `as_p_threshold` to the checkpoint and fan out if useful.
- A causal4-style "AUC ≥ 0.65" AS definition path is available via `notebooks/causal46_joined/as_reconciliation.py` but not used here.
- The joined pipeline assumes the causal6 acoustic pipeline has run to completion (`phon_peaks_all.parquet` exists). Document this in `causal46_joined.Snakefile`'s header comment.
