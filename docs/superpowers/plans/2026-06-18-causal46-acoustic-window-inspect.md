# Plan: interactive acoustic-decoding window inspector (causal46_joined)

**Date:** 2026-06-18
**Branch:** causal6-speech-responsive-update
**Goal:** An interactive notebook to manually inspect acoustic decoding at a
hand-picked *subsequent* time window on a single electrode — visualize the
causal46 star plot with the window highlighted, then fit a fresh acoustic
decoder on that window and transfer the electrode's original (peak-window)
acoustic decoder onto it, reporting both performances.

---

## User decisions (grilled 2026-06-18)

1. **Window spec:** fixed width = the electrode's peak acoustic window width
   (read from `phon_peaks_all.parquet`, not hard-coded). User supplies only the
   **onset**. Same width is required because (a) `run_acoustic_searchlight`
   asserts all windows share one width and (b) transferring the source decoder
   needs matching feature dimensionality.
2. **Source decoder:** re-fit in-notebook via a single `run_acoustic_searchlight`
   call containing *both* windows (peak + new), so they share trials and the CV
   split. With matched trials/seed this reproduces the on-disk causal6 decoder
   deterministically.
3. **Trial set:** pooled across completions (standard causal6 acoustic decoding,
   steps 1 vs 6, all word-ends). `word_end` only selects which star plot is
   shown. Acoustic 1-vs-6 has no completion confound, so the within-completion
   constraint does not apply here.
4. **No-peak fallback:** default source = peak acoustic window; allow an explicit
   `source_smin`/`source_smax` override in the params cell for sites with no
   recorded/significant peak.

---

## Deliverables

- `notebooks/causal46_joined/acoustic_decoding_single_electrode_inspect.py`
  (jupytext percent-format, parameterized, interactive — no Snakemake rule).
- `notebooks/causal46_joined/_acoustic_window_inspect.py` — small importable
  helper holding the reusable logic (window construction/validation + transfer
  AUC), per the "extract shared logic into a helper" convention. Keeps the
  notebook thin and lets the transfer math be unit-checkable.

No files written by the notebook (optional: a `savefig` line, commented out).

---

## Key facts established from the code

- Star plot: `matched_n_star_plot()` in
  `notebooks/causal46_joined/_within_completion.py`.
  - `ax_top`: unambiguous steps 1 & 6 for the word_end (acoustic anchor); green
    `axvspan` for the acoustic peak (`phon_smin/phon_smax`); dashed search bounds.
  - `ax_bot` (`fig._ax_behav`): per-step class-balanced behavioral bootstrap.
  - optional `ax_dec`: behavioral decoding overlay.
  - Time axis = seconds post word onset; `epoch_tmin=-0.4`, `epoch_sfreq=100`
    (import `epoch_tmin, epoch_sfreq` from `src.viz_paper`).
- Acoustic fit: `run_acoustic_searchlight()` in `src/models/causal6.py`.
  - Returns polars `scores`, `predictions`, `coefficients`.
  - `coefficients` carries per-(window×fold) `coef`, `mean`, `scale` arrays;
    **no intercept** (`z = einsum("bnd,bd->bn", X_std, beta)`).
  - `predictions` gives per-fold held-out `epoch_idx`, `decoder_target`,
    `decoder_proba`.
  - Defaults match pipeline: `resampled_steps=(1,6)`, pooled over word-ends.
- Decoder hyperparams are **loaded at runtime**, never hard-coded — the notebook
  mirrors the `acoustic_decoding_single_electrode` Snakemake rule's wiring exactly:
  - `reg_lambda` ← `_load_reg_lambda(winners_path, "acoustic")` reading
    `reg_lambda_acoustic` from
    `outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json` (empirically tuned;
    currently 1.0, but read it, don't assume).
  - `min_sample, window_size, stride` ← `config["analysis"]["decoding"][...]`
    (currently 1 / 15 / 2).
  - `n_folds, cv_random_state, device, tol, max_iter` ← `config["causal6"][...]`
    (currently 5 / 42 / cuda / 1e-6 / 50).
  - `device` overridable in params (default `cpu` interactively — the 1-electrode×
    2-window fit is trivial).
- Peaks: `outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet`
  with `subject, electrode_idx, phoneme_pair, smin, smax, test_roc_auc, p_value`.
- Transfer convention: `evaluate_phonetic_transfer()` in `src/viz_paper.py` —
  standardize the *target* window with its *own* scaler, apply the *source*
  decoder's coefficients.

---

## Notebook structure (cells)

1. **Params** (`tags=["parameters"]`): only user-facing knobs + paths, never
   frozen hyperparams.
   - Site: `subject, electrode_idx, phoneme_pair, word_end`.
   - Window onset: `new_window_onset_s` (seconds post word onset; **takes
     priority**) and `new_window_onset_sample` (fallback if seconds is None).
   - Optional source override: `source_smin/source_smax`.
   - Paths: `config_path="config.yaml"`,
     `reg_lambda_winners_path="outputs/causal6/reg_lambda_sweep/reg_lambda_winners.json"`,
     `phon_peaks_path="outputs/causal6/acoustic_decoding_peaks/phon_peaks_all.parquet"`,
     `epoch_dir`, behavioral-decoding roots for the star-plot overlay.
   - `device="cpu"` (override of the config default for interactive use).

2. **Load config + epochs:** read `config.yaml` and the winners JSON; derive
   `reg_lambda/min_sample/window_size/stride/n_folds/cv_random_state/tol/max_iter`
   from them (same sources as the Snakefile rule, echoed in a print so the user
   sees the empirically-tuned values in play). Load epochs
   (`load_epochs_dict`), peaks parquet, behavioral-decoding scores for overlay.

3. **Resolve windows:** look up this electrode×pair peak row →
   `(peak_smin, peak_smax)`, `width = peak_smax - peak_smin`,
   `acoustic_peak_auc`. Build new window from onset:
   `new_smin = round((onset_s - epoch_tmin)*sfreq)`, `new_smax = new_smin + width`.
   Validate bounds (`>= min_sample`, `<= n_times`); if no peak row and no
   override → clear error. Source window = override if given else peak window.
   Print both windows in samples and seconds.

4. **Star plot:** call `matched_n_star_plot(...)` with the peak window + behav
   overlay (mirror the `t_tests.py` call site), then post-hoc `axvspan` the new
   window `[new_tmin, new_tmax]` on every panel (`ax_top`, `ax_bot=fig._ax_behav`,
   `ax_dec` if present) in a distinct color (e.g. `#8856a7` purple) with a label,
   so it's visually separable from the green acoustic-peak shade. Do **not**
   modify `matched_n_star_plot`.

5. **Fit both decoders (one call):**
   `windows = np.array([[source_smin, source_smax], [new_smin, new_smax]])`
   (dedup if override == new), `electrode_idxs=[electrode_idx]`,
   `target="categorical_acoustic_cue"`, hyperparams from config.
   Filter outputs to `phoneme_pair`. Sanity: assert both windows present.

6. **Transfer (helper `compute_transfer_auc`):** for each fold f —
   - held-out test epochs + labels from `predictions` (new window, fold f);
   - `X_new` = electrode HGA at `[new_smin:new_smax]` for those epochs;
   - `(mean_f, scale_f)` from `coefficients` (new window, fold f);
   - `coef_f` from `coefficients` (source window, fold f);
   - `proba = sigmoid(((X_new - mean_f)/scale_f) @ coef_f)`;
   - `auc_f = roc_auc(labels, proba)`.
   Average folds → **transfer AUC**. (Faithful to `evaluate_phonetic_transfer`;
   evaluated on identical held-out trials as the retrained decoder.)

7. **Report:** small table / bar chart of
   - **Retrained-on-new** = `scores` rows for the new window (CV fold-mean AUC ± sd),
   - **Transfer (original → new)** = step 6,
   - context: **Original-on-peak** (its home-window AUC) and the searchlight
     time-course (`test_roc_auc` vs window center) with the peak and new windows
     marked, so the user sees where the new window sits.
   Both headline AUCs land on identical held-out new-window trials per fold → directly comparable.

---

## Verification (no GPU/GL needed locally)

- `uv run python -m py_compile` (or jupytext round-trip) on the new notebook +
  helper.
- Optionally `uv run jupytext --to ipynb` and run on a node with the env where
  epochs/peaks exist. Fit is 1 electrode × 2 windows → trivially fast on CPU.

## Open / minor
- Default `device="cpu"` in params (tiny fit); user can flip to `cuda`.
- If `width != 15` for a site (peaks are argmax over 15-sample searchlight
  windows, so width is always 15) — still read it dynamically, don't assume.
