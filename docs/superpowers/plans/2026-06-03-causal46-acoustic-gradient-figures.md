# Plan: Acoustic gradient figures for causal46_joined

## Problem

The causal46_joined pipeline has three populated aggregate parquets with no
downstream visualization:

| Parquet | Rule |
|---|---|
| `outputs/causal46_joined/acoustic_ax_discrimination/ax_discrimination_df_all.parquet` | `joined_acoustic_ax_discrimination_aggregate` |
| `outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet` | `joined_acoustic_univariate_gradient_aggregate` |
| `outputs/causal46_joined/acoustic_univariate_gradient/model_comparison_df_all.parquet` | `joined_acoustic_univariate_gradient_aggregate` |

The causal5 analog is `notebooks/causal5/acoustic_morphology_on_ambiguous.py`,
which does both computation and visualization in one monolith. For causal46_joined
the computation is already split across upstream rules; what's needed is a
population visualization notebook that loads the three aggregates and produces
figures.

## Analog: causal5 figures produced

The following figures are produced by `acoustic_morphology_on_ambiguous.py`:

| causal5 figure | File | Description |
|---|---|---|
| Fig 1 | `confidence_by_step.pdf` | Boxplot of `\|hga_norm − 0.5\|` per morph step |
| Fig 2 | `confidence_scatter.pdf` | Per-site endpoint vs. ambiguous confidence scatter, colored by phon_roc_auc |
| Fig 3 | `behavior_agreement.pdf` | HGA ROC-AUC predicting behavior on ambiguous/endpoint trials |
| Fig 4 | `site_example.pdf` | Single exemplar site |
| Fig 5 | `catplots_sample.pdf` | 24 sample catplots with mean neurometric + AX AUC on secondary y-axis |
| Fig 5b | `ax_discrimination_population.pdf` | Population AX curve: mean ± SEM per adjacent step pair |
| Fig 6 | `hga_timecourse_sample.pdf` | HGA timecourse by step (requires epoch reload) |
| Sigmoid Fig 1 | `ideal_model_shapes.pdf` | Ideal sigmoid shapes at k = 0.1 → 2 |
| Sigmoid Fig 2 | `sigmoid_parameter_distributions.pdf` | k, x0, R² histograms |
| Sigmoid Fig 2b | `pse_by_subject_phoneme.pdf` | PSE strip+box by subject and phoneme pair |
| Sigmoid Fig 2c | `pse_overlay_candidates.pdf` | Multi-electrode PSE overlay for 3+ categorical sites |
| Sigmoid Fig 3 | `sigmoid_vs_auc.pdf` | Steepness k and PSE x0 vs. phon_roc_auc scatter |
| Sigmoid Fig 4 | `catplots_sigmoid_fits.pdf` | Same 24 catplots with sigmoid overlaid |
| Cluster | `cluster_stats.pdf`, `cluster_curves.pdf` | KMeans on AX curves, cluster mean curves |

## What to include / skip

**Include (direct ports):**
- Fig 1 — HGA confidence by morph step: computable from `trial_df_all` (`hga_norm` per trial)
- Fig 2 — Endpoint vs. ambiguous confidence scatter: computable from `trial_df_all`
- Fig 5 — Sample catplots with AX overlay: `trial_df_all` + `ax_discrimination_df_all`
- Fig 5b — Population AX discrimination curve: `ax_discrimination_df_all` alone — **headline figure**
- Sigmoid Fig 1 — Ideal shapes: pure computation, no data needed
- Sigmoid Fig 2 — Parameter distributions: `model_comparison_df_all`
- Sigmoid Fig 2b — PSE by subject/phoneme pair: `model_comparison_df_all`
- Sigmoid Fig 2c — Multi-electrode PSE overlay: `trial_df_all` + `model_comparison_df_all`
- Sigmoid Fig 3 — Steepness vs. AUC scatter: `model_comparison_df_all` + `phon_peaks_all` join
- Sigmoid Fig 4 — Catplots with sigmoid fits: `trial_df_all` + `model_comparison_df_all`

**Skip (reasons noted):**
- Fig 3 — Behavior agreement: requires `behavior_categorical_forced`, which is not
  stored in `trial_df_all`. Would need epoch-metadata join. Out of scope here.
- Fig 4 — Single exemplar: low priority; superseded by catplots.
- Fig 6 — HGA timecourse: requires epoch reload for all subjects. Too heavy for a
  visualization-only notebook.
- Clustering analysis: exploratory; depends on AX curve shapes which may be noisy
  with the manifest-restricted pool. Defer.

## Key differences vs. causal5

1. **Site pool**: manifest-restricted (already QC'd via `filtered_manifest.csv`).
   No `phon_roc_auc >= 0.65` threshold filter needed — those sites are already
   included by virtue of being in the manifest.
2. **Peak window**: from causal6 null-standardized phon_peaks (per-subject files
   previously merged into `phon_peaks_all.parquet`). Not the causal5 searchlight.
3. **`phon_roc_auc` not stored in gradient outputs**: join it from
   `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet` using
   `(subject, electrode_idx, phoneme_pair)` key. The visualization notebook
   takes `phon_peaks_all` as an explicit input.
4. **No `site_stats` pre-computed**: causal5 built a `site_stats` frame
   (confidence, behavior AUC) during epoch loading. Here we compute just the
   confidence columns from `trial_df_all`; behavior AUC is skipped.

## Implementation

### New notebook

**Path:** `notebooks/causal46_joined/acoustic_gradient_figures.py`

Jupytext percent-format. Parameters cell:

```python
trial_df_path            = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet"
model_comparison_df_path = "outputs/causal46_joined/acoustic_univariate_gradient/model_comparison_df_all.parquet"
ax_discrimination_path   = "outputs/causal46_joined/acoustic_ax_discrimination/ax_discrimination_df_all.parquet"
phon_peaks_path          = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"
manifest_path            = "outputs/causal46_joined/filtered_manifest.csv"
outdir                   = "outputs/causal46_joined/acoustic_gradient_figures"
n_sample                 = 24        # sites for catplots
```

### Notebook structure

**Section 0 — Loads**
- Load the four parquets.
- Join `phon_roc_auc` from `phon_peaks_all` into `trial_df_all` and
  `model_comparison_df_all` on `(subject, electrode_idx, phoneme_pair)`.
  Keep only the best-AUC row per site from `phon_peaks_all` in case of
  duplicate rows (it already has one row per site but verify).
- Report N sites, N subjects, pair counts.

**Section 1 — HGA confidence**

Compute per-trial `confidence = |hga_norm - 0.5|` from `trial_df_all`.

Compute per-site summary (matching causal5 `site_stats` structure):
```
mean_endpoint_confidence = mean confidence on resampled ∈ {1, 6}
mean_ambig_confidence    = mean confidence on resampled ∈ {2, 3, 4, 5}
```

- **Fig 1** (`confidence_by_step.pdf`): boxplot of confidence per step (endpoint boxes steelblue, ambiguous boxes lightyellow). Port from causal5 lines 359–387.
- **Fig 2** (`confidence_scatter.pdf`): per-site endpoint vs. ambiguous confidence scatter, colored by `phon_roc_auc`. Port from causal5 lines 393–424.

**Section 2 — Population AX discrimination curve** *(headline)*

- **Fig AX** (`ax_discrimination_population.pdf`): mean ± SEM AUC per step pair.
  Port from causal5 lines 633–673. Use `ax_discrimination_df_all` directly.

**Section 3 — Sample catplots with AX overlay**

Sample `n_sample=24` sites from `model_comparison_df_all` sorted by `phon_roc_auc`
(linspace across range, matching causal5). Build label map. Filter `trial_df_all`
to sample sites.

- **Fig 5** (`catplots_sample.pdf`): per-behavior jittered scatter + mean neurometric
  line + AX AUC on twin y-axis. Port from causal5 lines 529–623.
  Coloring: `behavior_dummy_forced` is not available in `trial_df_all`;
  color by `resampled` step instead (blue→red gradient), omitting the
  behavior split. The mean neurometric line is over all trials.

**Section 4 — Sigmoid figures**

- **Sigmoid Fig 1** (`ideal_model_shapes.pdf`): pure illustration. Port from causal5 lines 877–902.
- **Sigmoid Fig 2** (`sigmoid_parameter_distributions.pdf`): k / x0 / R² histograms, using
  `EFFECTIVELY_LINEAR_K` from `src.models.sigmoid`. Port from causal5 lines 908–955.
- **Sigmoid Fig 2b** (`pse_by_subject_phoneme.pdf`): PSE strip+box by subject and
  phoneme pair using seaborn. Port from causal5 lines 965–1012.
- **Sigmoid Fig 2c** (`pse_overlay_candidates.pdf`): find (subject × phoneme_pair) combos
  with ≥3 categorical electrodes (k < 1, R² > 0.05, PSE ∈ [1.5, 5.5]) with spread ≥ 0.5.
  For each combo overlay sigmoid curves + trial scatter. Port from causal5 lines 1024–1147.
  The per-trial scatter comes from `trial_df_all` (subset by subject + phoneme_pair + electrode_idx).
- **Sigmoid Fig 3** (`sigmoid_vs_auc.pdf`): steepness k and PSE x0 vs. `phon_roc_auc`
  scatter, filtered to PSE ∈ [2, 5]. Port from causal5 lines 1157–1224.
- **Sigmoid Fig 4** (`catplots_sigmoid_fits.pdf`): same 24 sample sites with sigmoid
  overlay in tomato. Port from causal5 lines 1320–1412. Skip behavior split (see Fig 5 note above).

### Snakemake rule

Add to `workflows/causal46_joined.Snakefile`, after `joined_acoustic_univariate_gradient_aggregate`:

```python
rule joined_acoustic_gradient_figures:
    """Population visualization: AX discrimination + sigmoid figures — manifest pool."""
    input:
        trial_df_all            = "outputs/causal46_joined/acoustic_univariate_gradient/trial_df_all.parquet",
        model_comparison_df_all = "outputs/causal46_joined/acoustic_univariate_gradient/model_comparison_df_all.parquet",
        ax_discrimination_all   = "outputs/causal46_joined/acoustic_ax_discrimination/ax_discrimination_df_all.parquet",
        phon_peaks_all          = "outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet",
        manifest                = "outputs/causal46_joined/filtered_manifest.csv",
        notebook                = "notebooks/causal46_joined/acoustic_gradient_figures.py",

    output:
        notebook   = "outputs/causal46_joined/acoustic_gradient_figures/notebook.ipynb",
        figures    = directory("outputs/causal46_joined/acoustic_gradient_figures/figures"),

    run:
        outdir = Path(output.notebook).parent
        run_notebook(
            str(input.notebook),
            str(output.notebook),
            parameters=dict(
                trial_df_path=str(input.trial_df_all),
                model_comparison_df_path=str(input.model_comparison_df_all),
                ax_discrimination_path=str(input.ax_discrimination_all),
                phon_peaks_path=str(input.phon_peaks_all),
                manifest_path=str(input.manifest),
                outdir=str(outdir / "figures"),
            ),
        )
```

### Fig 5 / Fig 4 note: no behavior split

`trial_df_all` stores `resampled` (morph step) but not `behavior_categorical_forced`
(subject's reported percept). The causal5 catplots color by percept (blue/red).
For causal46_joined, color per-trial scatter by `resampled` step (blue→red gradient,
matching Fig 6) rather than by behavior. This is less interpretable for perceptual
claims but still shows the neurometric function shape clearly. Mark clearly in figure
subtitles: "colored by morph step (percept label not available)."

If behavior coloring is later needed, `epoch_idx` is in `trial_df_all`, enabling
a join to epoch metadata loaded from `.fif` files — but that requires epoch reload
and is out of scope here.

## Outputs

All figures written to `outputs/causal46_joined/acoustic_gradient_figures/figures/`:

| File | Figure |
|---|---|
| `confidence_by_step.pdf` | HGA confidence by morph step |
| `confidence_scatter.pdf` | Endpoint vs. ambiguous confidence scatter |
| `ax_discrimination_population.pdf` | Population AX discrimination curve |
| `catplots_sample.pdf` | 24 sample catplots + AX overlay |
| `ideal_model_shapes.pdf` | Ideal sigmoid shapes |
| `sigmoid_parameter_distributions.pdf` | k / x0 / R² histograms |
| `pse_by_subject_phoneme.pdf` | PSE by subject × phoneme pair |
| `pse_overlay_candidates.pdf` | Multi-electrode PSE overlay |
| `sigmoid_vs_auc.pdf` | Steepness vs. AUC scatter |
| `ax_per_site_gallery.pdf` | Per-site gallery: neurometric + AX curve (one page per site) |
| `catplots_sigmoid_fits.pdf` | 24 sample catplots + sigmoid fits |

## Open questions

1. **`phon_roc_auc` join**: `phon_peaks_all.parquet` has one row per
   `(subject, electrode_idx, phoneme_pair)` (confirm — it's the best-peak row
   from each subject's `phon_peaks.parquet` after per-subject threshold filtering).
   If there are duplicate rows per site, take the first after sorting by
   `phon_roc_auc` descending.

2. **`n_sample=24` site selection**: sort `model_comparison_df_all` by `phon_roc_auc`
   ascending (after the join), take evenly-spaced linspace indices. Sites without
   a `phon_roc_auc` match are dropped before sampling.

3. **`EFFECTIVELY_LINEAR_K`** is already imported in `acoustic_univariate_gradient.py`
   from `src.models.sigmoid`. Same import in the new notebook.

4. **`outdir / "figures"` vs. `outdir`**: the Snakemake rule declares `figures` as a
   `directory()` output to force re-execution when figure files change. All `fig.savefig`
   calls in the notebook write to `outdir` (which is set to `outdir/figures` via
   the parameter). This matches how other figure-producing rules work in the Snakefile.
