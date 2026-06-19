# Plan: type1 early acoustic decoder output on ambiguous trials

**Date:** 2026-06-19
**Notebook to create:** `notebooks/causal46_joined/type1_early_decoder_on_ambiguous.py`
**Outputs:** `outputs/causal46_joined/type1_early_decoder_on_ambiguous/`

---

## Goal

At sites classified as **type1 (acoustic only)** — no detectable perceptual response in the
early window — ask what the early acoustic decoder "thinks" when it sees an ambiguous stimulus.
Two competing predictions:

- **Gradient encoding**: decoder output tracks subphonemic acoustic variation; output drifts
  smoothly from ~0 at step 1 through intermediate values at steps 2–5 to ~1 at step 6.
- **Categorical encoding**: decoder output is committed even on ambiguous stimuli; the
  distribution at steps 2–5 is bimodal near 0 and 1, not centred.

An additional secondary question: does the decoder output on ambiguous steps predict
the participant's reported percept (hue split by behavioral response)?

---

## Site selection

Load `outputs/causal46_joined/manual_annotations/early_acoustic_window.csv` and filter to
`site_type_relabel == 'type1_acoustic_only'`.

There are **56 type1 (subject, electrode_idx, phoneme_pair) sites** across 10 subjects.
8 of the 56 have `status == 'unclassifiable_B_power'` (vs 48 with `status == 'ok'`). This
refers to the behavioral contrast panel being unclassifiable due to power — it does NOT
affect the acoustic decoder, which is estimated on unambiguous endpoint trials. Include all
56 sites but log a warning next to sites with non-ok status.

---

## Data sources

| Source | Path | Use |
|---|---|---|
| Manual annotations | `outputs/causal46_joined/manual_annotations/early_acoustic_window.csv` | Site list |
| Peak acoustic windows | `outputs_prod/causal6/acoustic_decoding_peaks/{subject}/phon_peaks.parquet` | `(smin, smax)` and logged `test_roc_auc` per site |
| Epoch files | `outputs/epochs_preprocessed/{subject}_epo.fif` | HGA + metadata |
| Decoder hyperparams | `config.yaml` (`causal6.*`) + `outputs_prod/causal6/reg_lambda_sweep/reg_lambda_winners.json` | Reproducible refit |

---

## Decoder refit procedure

For each type1 site `(subject, electrode_idx, phoneme_pair, smin, smax)`:

1. **Load epochs** for the subject (once per subject, cache). Apply baseline. Enrich metadata
   with `add_metadata_features`.

2. **Refit acoustic decoder** using the canonical hyperparams:

   ```python
   import json, yaml
   cfg = yaml.safe_load(open("config.yaml"))
   reg_lambda = json.loads(open("outputs_prod/causal6/reg_lambda_sweep/reg_lambda_winners.json").read())["reg_lambda_acoustic"]
   n_folds        = cfg["causal6"]["n_folds"]          # 5
   cv_random_state = cfg["causal6"]["cv_random_state"] # 42
   tol            = cfg["causal6"]["tol"]              # 1e-6
   max_iter       = cfg["causal6"]["max_iter"]         # 50
   ```

   Call `run_acoustic_searchlight` with default `resampled_steps=(1, 6)` at the single
   peak window:

   ```python
   from src.models.causal6 import run_acoustic_searchlight
   scores, preds, coefs = run_acoustic_searchlight(
       ep, subject=subject, electrode_idxs=[electrode_idx],
       windows=np.array([[smin, smax]]),
       target="categorical_acoustic_cue",
       reg_lambda=reg_lambda,
       n_folds=n_folds, cv_random_state=cv_random_state,
       device="cpu",    # CPU sufficient for 1 electrode × 1 window
       tol=tol, max_iter=max_iter,
   )
   ```

3. **Sanity check — two tiers**:

   **Tier A: per-trial proba match against canonical predictions.parquet** (primary check;
   validates the reconstruction formula, not just overall AUC).

   For the first type1 site of the first subject where canonical predictions exist, load
   `outputs_prod/causal6/acoustic_decoding_single_electrode/{subject}/predictions.parquet`,
   filter to (electrode_idx, phoneme_pair, smin, smax). For each fold, reconstruct probas
   using our formula and compare per-trial to canonical:

   ```python
   canonical = pl.read_parquet(f"outputs_prod/causal6/acoustic_decoding_single_electrode/{subject}/predictions.parquet")
   canonical_site = canonical.filter(
       (pl.col("electrode_idx") == electrode_idx)
       & (pl.col("phoneme_pair") == phoneme_pair)
       & (pl.col("smin") == smin) & (pl.col("smax") == smax)
       & (pl.col("target") == "categorical_acoustic_cue")
   )
   # Apply our reconstruction for fold 0:
   fold0_row = coefs_pp.filter(pl.col("fold") == 0).row(0, named=True)
   w = np.array(fold0_row["coef"], dtype=np.float64)   # shape (win_size,) — NO INTERCEPT
   mu = np.array(fold0_row["mean"], dtype=np.float64)
   sigma = np.array(fold0_row["scale"], dtype=np.float64)
   fold0_canonical = canonical_site.filter(pl.col("fold") == 0).sort("epoch_idx")
   ep_idxs = fold0_canonical["epoch_idx"].to_numpy()
   epoch_data = ep.get_data(picks=[electrode_idx]).squeeze(1)  # (N, n_times)
   X = epoch_data[ep_idxs, smin:smax].astype(np.float64)
   X_std = (X - mu) / sigma
   reconstructed = 1.0 / (1.0 + np.exp(-(X_std @ w)))
   canonical_probas = fold0_canonical["decoder_proba"].to_numpy().astype(np.float64)
   max_diff = np.abs(reconstructed - canonical_probas).max()
   assert max_diff < 0.02, f"Per-trial proba mismatch: max_diff={max_diff:.5f}"
   ```

   This check validates the complete reconstruction pipeline (standardization + coef math).
   Small differences (<0.01) are expected from CPU vs CUDA float precision.

   **Tier B: fold-mean AUC consistency** (secondary check):

   ```python
   refitted_auc = float(scores.filter(pl.col("phoneme_pair") == phoneme_pair)["test_roc_auc"].mean())
   logged_auc   = float(phon_peaks.filter(...)["test_roc_auc"][0])
   assert abs(refitted_auc - logged_auc) < 0.02, (
       f"AUC mismatch: refitted={refitted_auc:.4f} logged={logged_auc:.4f} "
       f"for {subject} e{electrode_idx} {phoneme_pair}"
   )
   ```

   Run Tier A once (on the first site) to validate the formula. Run Tier B on every site.
   Tolerance of 0.02 is appropriate because original runs used CUDA and refits here use CPU.

4. **Endpoint predictions** (steps 1 and 6): already in `preds` as test-fold predictions.
   Each epoch appears exactly once (the held-out fold). Keep as-is — these are the reference
   anchors.

5. **Score ambiguous trials** (steps 2–5) using the stored coefficients. The `coefs` DataFrame
   has one row per fold and columns `coef` (list of floats, shape `win_size` — **no intercept**;
   see `src/models/causal6.py` line 25 and 397), `mean` (per-feature scaler mean), `scale`
   (per-feature scaler std). Apply each fold's model to all ambiguous trials, then average:

   ```python
   import numpy as np

   md = ep.metadata
   pp_ambig_mask = (
       (md["phoneme_pair"] == phoneme_pair).values
       & md["resampled"].isin([2, 3, 4, 5]).values
   )
   epoch_data = ep.get_data(picks=[electrode_idx]).squeeze(1)  # (n_total, n_times)
   X_ambig = epoch_data[pp_ambig_mask, smin:smax].astype(np.float64)  # (n_ambig, win_size)

   coefs_pp = coefs.filter(
       (pl.col("phoneme_pair") == phoneme_pair)
       & (pl.col("electrode_idx") == electrode_idx)
       & (pl.col("smin") == smin) & (pl.col("smax") == smax)
   )

   fold_probas = []
   for row in coefs_pp.iter_rows(named=True):
       w = np.array(row["coef"], dtype=np.float64)    # shape (win_size,) — NO intercept
       mu = np.array(row["mean"], dtype=np.float64)   # shape (win_size,)
       sigma = np.array(row["scale"], dtype=np.float64)  # shape (win_size,)
       X_std = (X_ambig - mu) / sigma                  # standardise same as training
       z = X_std @ w                                    # logit (no intercept term)
       proba = 1.0 / (1.0 + np.exp(-z))
       fold_probas.append(proba)

   proba_ambig = np.mean(fold_probas, axis=0)  # average across folds
   ```

   Collect into a long-form DataFrame with columns:
   `subject, electrode_idx, phoneme_pair, epoch_idx, resampled, word_end,
    behavior_categorical_forced, decoder_proba, split` (split = "ambiguous").

   Endpoint predictions from `preds` get `split = "endpoint"`.

   Join with epoch metadata on `(epoch_idx, phoneme_pair)` to get `resampled`, `word_end`,
   `behavior_categorical_forced`.

---

## Trial DataFrame schema

After collecting all sites and concatenating, the master trial DF has:

| Column | Type | Notes |
|---|---|---|
| `subject` | str | |
| `electrode_idx` | int | |
| `phoneme_pair` | str | |
| `epoch_idx` | int | |
| `resampled` | float | 1–6 |
| `word_end` | str | e.g. "desolate", "necessary" |
| `behavior_categorical_forced` | float | −1 or +1 |
| `decoder_proba` | float | 0–1; averaged across folds for ambiguous |
| `split` | str | "endpoint" or "ambiguous" |
| `acoustic_sign` | float | from annotations: −1 or +1 (polarity of acoustic tuning) |
| `site_label` | str | f"{subject} e{electrode_idx} {phoneme_pair}" |

`acoustic_sign` is `early_acoustic_window.csv:acoustic_sign`. It's used to optionally
flip `decoder_proba` so that a high value always means "heard the acoustically preferred
phoneme." This makes pooled plots interpretable without knowing each site's polarity.

---

## Output files

Write to `outputs/causal46_joined/type1_early_decoder_on_ambiguous/`:

- `trial_df.parquet` — the master trial DataFrame described above
- `per_site_catplots.pdf` — one page per site (see Figure 1 below)
- `aggregate_figure.pdf` — population-level summary (see Figure 2 below)

---

## Figure 1 — Per-site catplots

One page per site. A single seaborn-style strip/box plot:

- **x-axis**: `resampled` step (1, 2, 3, 4, 5, 6), treated as categorical
- **y-axis**: `decoder_proba` (0–1)
- **hue**: behavioral response. Map `behavior_categorical_forced` to a human-readable label.
  For the `dn` pair: −1 → "heard /d/", +1 → "heard /n/"; for `bm`: −1 → "heard /b/",
  +1 → "heard /m/"; for `pb`: −1 → "heard /p/", +1 → "heard /b/". Use `PHONEME_PAIR_TO_WORD_ENDS`
  or similar to resolve which end is which, or just label −1 / +1 generically.
- **Points**: individual trials (stripplot with alpha ~0.3 and jitter)
- **Overlay**: mean ± 95% bootstrap CI per (step, behavior) cell, plotted as a
  connected line over the strip
- **Facet by word_end**: use two columns per site page (one per word_end) so any
  completion-acoustic confound in the percept split is visible. The within-completion
  constraint is not applied here (decoder is acoustic, not perceptual) but splitting by
  word_end makes confounds detectable at a glance. For the `dn` pair the acoustic window
  is pre-POD (~0.10–0.28s vs POD=0.295s), so completion acoustics are absent anyway —
  note this in the figure caption. For `bm`/`pb` pairs, verify the site's window timing.

Steps 1 and 6 have nearly deterministic behavior (all trials heard one phoneme), so the
hue split degenerates there — that's expected and informative. Show them anyway.

**Title per page**: `f"{subject} e{electrode_idx} {phoneme_pair}  (site_type=type1)  AUC={logged_auc:.3f}"`

Use `matplotlib.backends.backend_pdf.PdfPages` to write all pages.

**Layout**: one axes per page, figure size ~(6, 4).

---

## Figure 2 — Aggregate summary

Designed to let you see the population-level pattern across all 56 type1 sites at once.

**Structure**: a single figure with sites "stacked" horizontally (one per column).

Concretely, create a grid where:
- Each **column** is one site (sorted by logged `test_roc_auc` descending, or by subject)
- Each **row of markers** shows the mean `decoder_proba` per resampled step at that site
- **Color** encodes resampled step using a diverging colormap (e.g. `RdBu_r`: step 1 =
  deep blue, step 6 = deep red, steps 2–5 = intermediate)
- Show mean across all trials (pooled across behavioral responses) as a dot, and optionally
  show ±1 SEM as a thin error bar

This is effectively a seaborn `pointplot` or a hand-rolled scatter:
```python
fig, ax = plt.subplots(figsize=(14, 4))
step_colors = plt.cm.RdBu_r(np.linspace(0.05, 0.95, 6))  # steps 1–6
for step_i, step in enumerate([1, 2, 3, 4, 5, 6]):
    site_means = (
        trial_df[trial_df["resampled"] == step]
        .groupby("site_label")["decoder_proba"].mean()
        .reindex(site_order)  # site_order = sorted site labels
    )
    ax.scatter(
        np.arange(len(site_order)) + step_i * 0.12,  # slight x-dodge per step
        site_means.values,
        color=step_colors[step_i], s=20, alpha=0.85, label=f"step {step}",
    )
ax.set_xticks(np.arange(len(site_order)))
ax.set_xticklabels(site_order, rotation=90, fontsize=6)
ax.set_ylabel("decoder_proba")
ax.legend(title="resampled", loc="upper right", fontsize=8)
```

**Optional second panel**: same structure but split by behavioral response (two sub-rows:
"heard phoneme 1" vs "heard phoneme 2"), useful to see whether the decoder output predicts
behavior even at mid-steps.

---

## Notebook structure (cells)

```
# %% [markdown]
# # Type-1 early acoustic decoder output on ambiguous trials
# ...goal text...

# %% tags=["parameters"]
annotations_path = "outputs/causal46_joined/manual_annotations/early_acoustic_window.csv"
phon_peaks_root  = "outputs_prod/causal6/acoustic_decoding_peaks"
epoch_dir        = "outputs/epochs_preprocessed"
config_path      = "config.yaml"
reg_lambda_winners_path = "outputs_prod/causal6/reg_lambda_sweep/reg_lambda_winners.json"
outdir           = "outputs/causal46_joined/type1_early_decoder_on_ambiguous"
device           = "cpu"

# %% — imports

# %% — load hyperparams from config + reg_lambda_winners

# %% — load type1 site list from annotations

# %% — main loop: per subject → per site
#       (cache epoch load per subject; iterate type1 sites within subject)
#       yields trial_df rows for endpoints and ambiguous trials

# %% — concatenate, validate, write trial_df.parquet

# %% — per-site catplots → per_site_catplots.pdf

# %% — aggregate figure → aggregate_figure.pdf
```

---

## Snakefile rule (optional, add later)

This notebook is initially run interactively (no Snakemake rule needed for the first pass).
If it becomes a pipeline step, it takes no per-subject parameterisation (reads all 10 subjects
in one notebook run) and its sole file output is `trial_df.parquet` (the PDFs are side effects).

---

## Key implementation notes

1. **Epoch caching**: load each subject's `.fif` once, extract all of that subject's type1
   sites in the inner loop. Do not reload per site.

2. **`acoustic_sign` flip**: some sites have `acoustic_sign == -1` meaning step 6 → the
   lower HGA. For interpretable pooled plots, define
   `decoder_proba_aligned = decoder_proba if acoustic_sign == 1 else 1 - decoder_proba`.
   Store both in `trial_df`; use `decoder_proba_aligned` for the aggregate figure.

3. **Fold averaging for ambiguous**: the 5-fold models each score all ambiguous trials
   (none of the ambiguous trials appear in any fold's training set). Average `decoder_proba`
   across the 5 folds per trial to reduce variance.

4. **Endpoint fold assignment**: for steps 1 and 6 use test-fold held-out predictions from
   `preds` (each endpoint trial scored by exactly 1 fold). This matches the convention used
   in `predictions.parquet`.

5. **Missing epoch files**: 4 of 10 subjects may not have `.fif` files locally (only 6 are
   in `outputs/epochs_preprocessed/`). For missing subjects, log a warning and skip. The
   plan assumes all subjects are available on the full run (prod has all files).

6. **Reconstruction formula — no intercept**: `coef` in the coefficients DataFrame is shape
   `(win_size,)` with NO intercept appended (see `causal6.py` module docstring line 25 and
   `_acoustic_window_inspect.py:compute_transfer_auc` for the canonical application).
   Application: `z = X_std @ w; proba = sigmoid(z)`. Do NOT split off `w[-1]` as intercept.

7. **Tolerance on sanity check**: use `atol=0.02` for fold-mean AUC (Tier B). Per-trial
   proba (Tier A) is expected to match within ~0.01 due to CPU vs CUDA float precision.
   If max_diff exceeds 0.02 on Tier A, the formula is wrong — halt and diagnose.
