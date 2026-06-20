# Type-1 sites: what HGA encodes on ambiguous input (graded vs committed)

Date: 2026-06-20
Branch: causal6-speech-responsive-update

## Question

At **type1 (acoustic-only)** sites — canonical early acoustic response, no report
split in the early window on matched ambiguous trials — the report-blocked mean HGA
curves overlap. The mean alone cannot tell us *how* the site represents ambiguous
input. Three hypotheses, on the early-window HGA per acoustic step among the
already-selected ambiguous steps:

- **O1 — graded intermediate**: unimodal, centered between the endpoint values.
- **O2a — committed, fixed**: unimodal, sitting at an *endpoint*, not the middle.
- **O2b — committed, trial-varying**: bimodal at the two endpoint values trial-by-trial.

Read together (per the methods discussion):
- O1 → mid-axis location, endpoint-like spread
- O2a → endpoint location, endpoint-like spread
- O2b → both ends, **inflated** spread

## Synthesis being merged

- **Site set + qualifying-steps logic** from `type1_early_decoder_on_ambiguous.py`
  (annotations → type1 sites; `trial_balance_index.csv` → ambiguous steps).
- **HGA-d′ axis + sigmoid + AX machinery** from `acoustic_gradient_figures.py`.
- **Drop the decoder refit** — the user wants HGA d′, not `decoder_proba`. The whole
  `run_acoustic_searchlight` / coefficient-reconstruction path is unnecessary here.

## Inputs and coverage (verified against `outputs_prod`)

| Input | Path | Note |
|---|---|---|
| type1 site list | `manual_annotations/early_acoustic_window.csv`, `site_type_relabel=="type1_acoustic_only"` | 56 sites |
| per-trial HGA at peak acoustic window | `acoustic_univariate_gradient/trial_df_all.parquet` | has `hga_raw`, `resampled`, `epoch_idx`, `hga_norm`, `hga_polarity`, `smin/smax`; **no** `word_end`/behavior |
| qualifying ambiguous steps | `trial_balance_index.csv`, `is_ambiguous_step` | per (subject, phoneme_pair, word_end) |
| AX adjacent-step AUC | `acoustic_ax_discrimination/ax_discrimination_df_all.parquet` | `step_a, step_b, roc_auc, roc_auc_std` |
| epoch metadata (behavior panel only) | `epochs_preprocessed/{subject}_epo.fif` | read metadata only, merge on `epoch_idx` |

**Coverage**: 50/56 type1 sites present in `trial_df_all`; 6 dropped by the
separation floor (weak gradient → low d′). Endpoint counts are healthy (min 24,
median 48 per endpoint), so the d′ ruler is stable. **Decision (user): epoch-fallback
to include all 56.** For the 6 missing sites, read their peak window (`smin/smax` from
`phon_peaks`) from epochs and extract `hga_raw = data[epoch, eidx, smin:smax].mean(1)`,
then run the same endpoint-normalization as the parquet sites. Flag them as
`coverage="epoch_fallback"` and keep the d′-stratified power caveat (these are
weak-gradient sites, so likely still uninformative for the spread test — labelled, not
hidden). Epoch reads are needed for the behavior panel anyway, so this composes.

## The normalization fix (the crux)

The methods require the d′ axis scaled by the **pooled within-condition** endpoint SD.
The existing pipeline (`acoustic_univariate_gradient.py:165–170`) computes
`hga_endpoint_std` as `.std()` over the *combined* step-1+step-6 pool — which absorbs
the between-endpoint mean gap and inflates the denominator (verified: combined/within
ratio median 1.05, max 1.49 across these sites). That inflation compresses spread in
exactly the direction that would mask O2b.

Scope of the fix (per advisor): **only the spread/O2b statistic is affected.**
- **Location (O1 vs O2a)** lives in `hga_norm` (∈[0,1], polarity-canonicalized
  /d/→0,/n/→1). It is SD-invariant — reuse it as-is.
- **Spread (O2a vs O2b)** must use a recomputed within-condition pooled SD.

Recompute per site from endpoint trials in `trial_df_all`:
```
sigma_pooled = sqrt( ((n1-1)*var(step1) + (n6-1)*var(step6)) / (n1+n6-2) )   # ddof=1
d_prime_corr = |mean(step6) - mean(step1)| / sigma_pooled
hga_dprime   = (hga_raw - midpoint) / sigma_pooled,  midpoint=(mean1+mean6)/2,
               sign-flipped by hga_polarity so /d/<0, /n/>0
```
Honor the two cautions with columns already present:
- report `sigma1/sigma6` ratio per site; if far from 1, note unequal variance (pooled
  SD already handles it, but flag it).
- ruler stability is fine here (n≥24/endpoint); still record `A_n_step1/A_n_step6`.

**Do NOT re-apply `acoustic_sign`** (advisor #5): `hga_norm`/`hga_polarity` already
canonicalize polarity. `acoustic_sign` only existed to flip `decoder_proba` on the
dropped decoder path; applying both would flip half the sites.

## Statistic decomposition

The O1/O2a/O2b assignment is a **within-step** question and must be answered by
within-step statistics. The across-step sigmoid `k` and AX characterize *tuning
categoricity* and only **corroborate** — they do not adjudicate. (Why `k` cannot do the
O1/O2a work: it is an across-step shape parameter and conflates "categorical tuning
curve" with "commits at the ambiguous step" — a site that commits to the nearer
endpoint at every step produces a steep `k` indistinguishable from sharpened-but-
midpoint-respecting tuning. Fitting one `k` across steps also reintroduces the
continuum-sampling dependence that within-step conditioning removes.)

### Primary — within-step LOCATION test (O1 vs O2a)

Per site, define the response axis from the unambiguous endpoints: /d/-endpoint
mean = 0, /n/-endpoint = the site's endpoint d′, scaled by the **pooled within-condition
endpoint SD** (same `sigma_pooled` as below). Then **separately for each ambiguous
step**, the location as a fraction of the endpoint separation:
```
loc(step) = (mean_ambig(step) - mean_/d/) / (mean_/n/ - mean_/d/)
```
`loc = 0.5` → midpoint (graded); `loc → 0 or 1` → committed to an endpoint.
(Note: after polarity canonicalization this equals the per-step mean of `hga_norm`, so
`loc` is SD-invariant — the d′ scaling matters only for the spread test and for plotting
in d′ units.)
- **Quant test**: per step, test `loc` against 0.5 (midpoint) and against the nearer
  endpoint, with CIs from **trial-level bootstrap**. Aggregate across steps/sites:
  O1 → distribution of `loc` concentrates near 0.5; O2a → concentrates near 0/1.
- Stays strictly within-step → inherits none of the across-step sampling confound.

### Secondary — within-step SPREAD test (O2a vs O2b), variance ratio only

On the *same* within-step distributions: per qualifying ambiguous step, the **variance
ratio** `var(hga_dprime at step) / sigma_pooled^2` (≈1 → committed-fixed O2a or graded
O1; ≫1 → trial-varying O2b). Scaled by endpoint SD, **never** the ambiguous SD (that
would divide out the very inflation that defines O2b). No explicit modality test — see
the O2b note below.

### Corroboration only — tuning categoricity (keep, but relabel)

- **Sigmoid `k`/`x0`/`r2`** (`src.models.sigmoid.fit_sigmoid` on per-site step-mean
  `hga_norm`): characterizes whether the *tuning curve* is shallow (graded) or steep
  (categorical). Complementary to, not a substitute for, the location test.
- **AX adjacent-step discrimination**: restrict `ax_discrimination_df_all` to type1
  sites and ambiguous step pairs (2v3,3v4,4v5). Population mean ± SEM + per-site overlay.

Read together (real corroboration, done correctly): a coherent graded encoder is
**O1** (loc ≈ 0.5, endpoint-like spread) *and* shallow `k` *and* discriminates middle
steps under AX. A categorical site is **O2a/O2b** *and* steep `k` *and* shows an AX
plateau. Agreement is meaningful **because the within-step location test does the
O1/O2a adjudication**; `k` and AX corroborate, they do not decide.

## O2b note — variance ratio only (user decision)

No explicit modality test (GMM/diptest/bimodality coefficient) — per-step-per-site
counts are too thin to power one, and the variance ratio captures the inflation that
defines O2b. (The user mentioned dip/GMM when restating the O2a/O2b split; the standing
explicit decision is variance-ratio-only — flagging the minor tension, not silently
flipping it. Easy to add a population-pooled GMM/dip later if wanted.) The
location×spread scatter and per-step violins surface any visible bimodality descriptively.
- **Power caveat to print**: O2b's expected spread ratio scales with d′/2, so low-d′
  sites cannot distinguish the three hypotheses. Stratify/restrict the spread test
  to high-d′ sites and say so (this is where the 6 epoch-fallback sites land).

## Confound handling

- **Primary acoustic test pools across completions** — at a fixed step the continuum
  token is physically identical across word_ends and the early window precedes the POD,
  so there is zero acoustic confound. Assert this in the notebook header.
- **Qualifying steps for the acoustic test**: union of `is_ambiguous_step` across
  word_ends per (subject, phoneme_pair) (parquet lacks `word_end`).
- **Secondary behavior-split panel is a perceptual claim** → CLAUDE.md within-completion
  constraint is non-negotiable. Recover `word_end`+`behavior_categorical_forced` by
  reading epoch metadata and merging on `(subject, epoch_idx)`; reuse
  `_within_completion.py` (`per_step_class_counts`, `select_cell_trials_bootstrap`)
  with per-cell balanced bootstrap, exactly as the type1 notebook does.

## Outputs (new notebook `notebooks/causal46_joined/type1_ambiguous_hga_coding.py`)

Parameters cell (plain assignments, no annotations — ploomber): the five input paths
above + `outdir`.

1. `site_dprime.parquet` — per type1 site: `d_prime_corr`, `d_prime_existing`,
   `sigma_pooled`, `sigma1/sigma6`, endpoint n, sigmoid `k/x0/r2` (corroboration),
   coverage flag.
2. `ambiguous_step_stats.parquet` — per site × qualifying step: **`loc`** + bootstrap
   CI, p(loc vs 0.5), p(loc vs nearer endpoint); mean & variance of `hga_dprime`,
   variance ratio, n.
3. **Fig — within-step location, per site** *(primary O1/O2a)*: endpoint-normalized
   axis (0–1) with each ambiguous step's `loc` as a point + bootstrap CI; midpoint line
   at 0.5, endpoint lines at 0/1. Type1 sites, sorted by d′.
4. **Fig — `loc` histogram across sites** *(primary O1/O2a, the headline read)*: `loc`
   pooled over steps/sites — peaked-at-0.5 (O1) vs bimodal-at-edges (O2a) reads the
   split directly.
5. **Fig — location × spread scatter** (O1/O2a/O2b map): x = `loc`, y = variance ratio;
   quadrants annotated (O1 = 0.5/low-var; O2a = edge/low-var; O2b = edge/high-var).
   Colored/sized by d′.
6. **Fig — per-step spread**: violin/strip of `hga_dprime` at each ambiguous step with
   the endpoint σ band overlaid (descriptive view of the variance ratio); high-d′ subset.
7. **Fig — tuning categoricity (corroboration)**: per-site neurometric
   (`hga_dprime` mean±SEM + sigmoid overlay) + AX bottom panel; population AX on
   ambiguous step pairs (type1 only). **Labelled as corroboration, not adjudication.**
8. **Fig — behavior split (within-completion)**: ambiguous `hga_dprime` vs report,
   per completion — the secondary "does position predict percept" panel.

## Implementation notes

- New standalone notebook (don't edit either source); factor the d′-recompute into a
  small helper importable by both if reuse is wanted later (per repo convention).
- Wire a `joined_type1_ambiguous_hga_coding` rule into `causal46_joined.Snakefile`
  mirroring `joined_acoustic_gradient_figures`.
- `uv run` for all execution; never concurrent `uv run`.
- Validate with `jupytext --to ipynb` + `py_compile`; run against `outputs_prod` inputs
  locally where present, full run in prod (epoch files).

## Cautions / open items

- 6/56 sites covered via epoch-fallback (weak-gradient, separation-floor drops) —
  flagged `coverage="epoch_fallback"`, kept in the d′-stratified view.
- Variance test underpowered at low d′ — stratify and label.
- Confirm the early acoustic window (`smin/smax`) precedes the POD for each phoneme pair
  before asserting the cross-completion no-confound claim (check `POD_dict` vs window
  times); expected true but verify.
