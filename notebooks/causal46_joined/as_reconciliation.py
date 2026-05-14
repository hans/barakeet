# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # causal4/causal6 AS-site reconciliation
#
# Classifies every (subject, electrode_idx, phoneme_pair) tuple evaluated by
# either pipeline into one of five buckets, then renders summary stats and
# four star-plot PDFs (losses, gains-eligible, gains-newly-eligible, both) for
# visual inspection. Final canonical AS-site list is written to
# `outputs/causal46_joined/canonical_AS_sites.csv`.
#
# See `docs/superpowers/plans/2026-05-14-causal46-as-reconciliation.md` and
# Linear JON-42.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

# %%
HOME = Path(os.path.expanduser("~"))
# Resolve REPO from this notebook's location so it works in any worktree.
REPO = Path(".").resolve()
CAUSAL4_DIR = HOME / "u/projects/barakeet/outputs/causal4/prepare_neurometrics"
CAUSAL6_DIR = HOME / "u/projects/barakeet-speech-responsive/outputs/causal6/acoustic_decoding_peaks"
OUT_DIR = REPO / "outputs/causal46_joined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAUSAL4_AUC_THRESHOLD = 0.65
CAUSAL6_P_THRESHOLD = 0.05

print(f"REPO:        {REPO}")
print(f"CAUSAL4_DIR: {CAUSAL4_DIR}")
print(f"CAUSAL6_DIR: {CAUSAL6_DIR}")
print(f"OUT_DIR:     {OUT_DIR}")

# %% [markdown]
# ## Load causal4 outputs

# %%
# phon_peaks_df.parquet contains the peak window per (subject, electrode_idx,
# phoneme_pair) over the full causal4 search range -- it is NOT pre-filtered
# to AUC>=0.65. Apply the causal4 AS criterion explicitly here.
c4_peaks = pl.read_parquet(CAUSAL4_DIR / "phon_peaks_df.parquet")
c4_peaks = c4_peaks.with_columns(
    pl.col("subject").cast(pl.Utf8),
    pl.col("phoneme_pair").cast(pl.Utf8),
).rename({
    "phon_roc_auc": "causal4_peak_auc",
    "smin": "causal4_smin",
    "smax": "causal4_smax",
}).drop("word_end_offset_sample")
print(f"causal4 phon_peaks_df rows (unfiltered): {c4_peaks.shape[0]}")
c4_AS = c4_peaks.filter(pl.col("causal4_peak_auc") >= CAUSAL4_AUC_THRESHOLD)

c4_eligible = (
    pl.read_parquet(CAUSAL4_DIR / "phon_roc_auc_searchlight_df.parquet")
    .with_columns(
        pl.col("subject").cast(pl.Utf8),
        pl.col("phoneme_pair").cast(pl.Utf8),
    )
    .select(["subject", "electrode_idx", "phoneme_pair"])
    .unique()
)

print(f"causal4 AS sites: {c4_AS.shape[0]}")
print(f"causal4 evaluated tuples: {c4_eligible.shape[0]}")
print(f"causal4 subjects (in AS): {sorted(c4_AS['subject'].unique().to_list())}")

# %% [markdown]
# ## Load causal6 outputs

# %%
c6_paths = sorted(CAUSAL6_DIR.glob("*/phon_peaks.parquet"))
c6_subjects_present = [p.parent.name for p in c6_paths]
print(f"causal6 subjects in prod: {c6_subjects_present}")

c6_all = pl.concat([pl.read_parquet(p) for p in c6_paths])
c6_all = c6_all.rename({
    "test_roc_auc": "causal6_test_roc_auc",
    "p_value": "causal6_p_value",
    "n_permutations": "causal6_n_perm",
    "smin": "causal6_smin",
    "smax": "causal6_smax",
}).select([
    "subject", "electrode_idx", "phoneme_pair",
    "causal6_test_roc_auc", "causal6_p_value", "causal6_n_perm",
    "causal6_smin", "causal6_smax",
])
print(f"causal6 evaluated tuples: {c6_all.shape[0]}")
print(f"causal6 significant (p<0.05): {int((c6_all['causal6_p_value'] < CAUSAL6_P_THRESHOLD).sum())}")

# %% [markdown]
# ## Subject coverage warning

# %%
c4_subj = set(c4_AS["subject"].unique().to_list())
c6_subj = set(c6_subjects_present)
missing_in_c6 = sorted(c4_subj - c6_subj)
if missing_in_c6:
    print(
        f"WARNING: {len(missing_in_c6)} causal4 subjects absent from causal6 prod: "
        f"{missing_in_c6}. Their sites are excluded from reconciliation."
    )
    c4_AS = c4_AS.filter(~pl.col("subject").is_in(missing_in_c6))
    c4_eligible = c4_eligible.filter(~pl.col("subject").is_in(missing_in_c6))

# %% [markdown]
# ## Build the reconciliation table

# %%
KEYS = ["subject", "electrode_idx", "phoneme_pair"]

# Universe = union of every tuple either pipeline evaluated.
universe = pl.concat([
    c4_eligible.select(KEYS),
    c6_all.select(KEYS),
]).unique()

recon = (
    universe
    .join(
        c4_eligible.with_columns(pl.lit(True).alias("causal4_eligible")),
        on=KEYS, how="left",
    )
    .with_columns(pl.col("causal4_eligible").fill_null(False))
    .join(
        c4_AS.with_columns(pl.lit(True).alias("causal4_AS")),
        on=KEYS, how="left",
    )
    .with_columns(pl.col("causal4_AS").fill_null(False))
    .join(c6_all, on=KEYS, how="left")
    .with_columns(
        (pl.col("causal6_p_value") < CAUSAL6_P_THRESHOLD)
            .fill_null(False)
            .alias("causal6_AS"),
    )
)


def assign_bucket(c4_elig: bool, c4_AS_: bool, c6_AS_: bool) -> str:
    if c4_AS_ and c6_AS_:
        return "both"
    if c4_AS_ and not c6_AS_:
        return "causal4_only"
    if c6_AS_ and c4_elig:
        return "causal6_only_eligible"
    if c6_AS_ and not c4_elig:
        return "causal6_only_newly_eligible"
    return "neither_AS"


recon = recon.with_columns(
    pl.struct(["causal4_eligible", "causal4_AS", "causal6_AS"])
      .map_elements(
          lambda s: assign_bucket(s["causal4_eligible"], s["causal4_AS"], s["causal6_AS"]),
          return_dtype=pl.Utf8,
      )
      .alias("bucket")
)

print("Bucket counts:")
print(recon.group_by("bucket").len().sort("len", descending=True))

recon.write_parquet(OUT_DIR / "reconciliation.parquet")
print(f"Written: {OUT_DIR / 'reconciliation.parquet'}  ({recon.shape[0]} rows)")

# %% [markdown]
# ## Summary panels
#
# - Bucket counts per subject and phoneme_pair
# - Loss audit: distribution of causal4 peak AUC for `causal4_only` sites
# - Gain audit: distribution of causal6 corrected p-values for gain buckets
# - Joint scatter: causal4 peak AUC × causal6 p-value, coloured by bucket

# %%
breakdown = (
    recon
    .group_by(["bucket", "subject", "phoneme_pair"])
    .len()
    .pivot(values="len", index=["subject", "phoneme_pair"], on="bucket")
    .fill_null(0)
    .sort(["subject", "phoneme_pair"])
)
print("Per-subject / phoneme_pair bucket breakdown:")
print(breakdown)
breakdown.write_csv(OUT_DIR / "bucket_breakdown.csv")

# %%
losses = recon.filter(pl.col("bucket") == "causal4_only")
print(f"Losses: {losses.shape[0]} sites")
print(f"  causal4_peak_auc:  min={losses['causal4_peak_auc'].min():.3f}  "
      f"median={losses['causal4_peak_auc'].median():.3f}  "
      f"max={losses['causal4_peak_auc'].max():.3f}")
print(f"  count >= 0.70: {int((losses['causal4_peak_auc'] >= 0.70).sum())}  "
      f">= 0.75: {int((losses['causal4_peak_auc'] >= 0.75).sum())}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(
    losses["causal4_peak_auc"].to_numpy(),
    bins=20, color="tomato", alpha=0.8, edgecolor="k",
)
axes[0].axvline(0.65, color="k", lw=0.8, ls="--", label="causal4 threshold")
axes[0].set_xlabel("causal4 peak AUC")
axes[0].set_ylabel("losses (count)")
axes[0].set_title(f"Loss audit — {losses.shape[0]} sites")
axes[0].legend()

gains = recon.filter(pl.col("bucket").is_in(
    ["causal6_only_eligible", "causal6_only_newly_eligible"]
))
print(f"Gains: {gains.shape[0]} sites")
print(f"  eligible:        {(gains['bucket'] == 'causal6_only_eligible').sum()}")
print(f"  newly_eligible:  {(gains['bucket'] == 'causal6_only_newly_eligible').sum()}")

for bucket, color in [
    ("causal6_only_eligible", "steelblue"),
    ("causal6_only_newly_eligible", "seagreen"),
]:
    vals = gains.filter(pl.col("bucket") == bucket)["causal6_p_value"].to_numpy()
    axes[1].hist(vals, bins=np.linspace(0, 0.05, 21), alpha=0.6,
                 color=color, edgecolor="k", label=bucket)
axes[1].set_xlabel("causal6 corrected p-value")
axes[1].set_ylabel("gains (count)")
axes[1].set_title(f"Gain audit — {gains.shape[0]} sites")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "summary_audit.png", dpi=150)
plt.close(fig)

# %%
plot_df = recon.filter(
    pl.col("causal4_peak_auc").is_not_null()
    | pl.col("causal6_p_value").is_not_null()
)

fig, ax = plt.subplots(figsize=(7, 5))
bucket_colors = {
    "both": "#3a823a",
    "causal4_only": "#c44e4e",
    "causal6_only_eligible": "#4a78b8",
    "causal6_only_newly_eligible": "#2d8b8b",
    "neither_AS": "#999999",
}
for bucket, color in bucket_colors.items():
    sub = plot_df.filter(pl.col("bucket") == bucket)
    if sub.shape[0] == 0:
        continue
    x = sub["causal4_peak_auc"].fill_null(np.nan).to_numpy()
    y = sub["causal6_p_value"].fill_null(np.nan).to_numpy()
    ax.scatter(x, y, c=color, s=18, alpha=0.55,
               label=f"{bucket} (n={sub.shape[0]})", edgecolors="none")

ax.axvline(0.65, color="k", lw=0.6, ls="--", label="causal4 AUC threshold")
ax.axhline(0.05, color="k", lw=0.6, ls=":", label="causal6 p threshold")
ax.set_xlabel("causal4 peak AUC")
ax.set_ylabel("causal6 corrected p-value")
ax.set_yscale("log")
ax.set_title("causal4 vs causal6 — agreement & disagreement")
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(OUT_DIR / "summary_scatter.png", dpi=150)
plt.close(fig)
print(f"Wrote: {OUT_DIR / 'summary_audit.png'}, {OUT_DIR / 'summary_scatter.png'}")

# %% [markdown]
# ## Star plot galleries — visual inspection
#
# Four PDFs:
#   - losses.pdf:                bucket == "causal4_only", sort by causal4 peak AUC desc
#   - gains_eligible.pdf:        bucket == "causal6_only_eligible", sort by causal6 AUC desc
#   - gains_newly_eligible.pdf:  bucket == "causal6_only_newly_eligible", sort by causal6 AUC desc
#   - both.pdf:                  random sample of 10 from "both" for sanity
#
# Star plot helper imported from src.viz_provisional.

# %%
import mne

from src.data import add_metadata_features
from src.stimuli import PHONEME_PAIR_TO_WORD_ENDS
from src.viz_provisional import (
    load_ambig_steps,
    provisional_star_plot,
)

# Epoch files live wherever the preprocessing pipeline output them. Default is
# <REPO>/outputs/epochs_preprocessed; override with the BARAKEET_EPOCH_DIR env
# var if they live elsewhere on this machine.
EPOCH_DIR = Path(os.environ.get(
    "BARAKEET_EPOCH_DIR",
    str(REPO / "outputs/epochs_preprocessed"),
))
print(f"EPOCH_DIR: {EPOCH_DIR}  (exists: {EPOCH_DIR.exists()})")

needed_subjects = sorted(
    recon.filter(pl.col("bucket").is_in([
        "causal4_only", "causal6_only_eligible",
        "causal6_only_newly_eligible", "both",
    ]))["subject"].unique().to_list()
)
print(f"Loading epochs for {len(needed_subjects)} subjects: {needed_subjects}")

epochs_dict: dict = {}
for s in needed_subjects:
    path = EPOCH_DIR / f"{s}_epo.fif"
    if not path.exists():
        print(f"  (skip {s}: {path} missing)")
        continue
    ep = mne.read_epochs(str(path), preload=False, verbose="WARNING")
    ep.metadata = add_metadata_features(ep.metadata.copy())
    epochs_dict[s] = ep
print(f"Loaded epochs for {len(epochs_dict)} subjects: {sorted(epochs_dict)}")

ambig_steps = load_ambig_steps(epochs_dict) if epochs_dict else {}
print(f"ambig_steps: {len(ambig_steps)} (subject, phoneme_pair, word_end) keys")

# %%
def render_gallery(rows: pl.DataFrame, out_path: Path, title_prefix: str):
    """Render one PDF: one page per (site, word_end)."""
    if rows.shape[0] == 0:
        print(f"  (no sites for {out_path.name})")
        return
    n_pages = 0
    n_skipped = 0
    with PdfPages(out_path) as pdf:
        for row in rows.iter_rows(named=True):
            if row["subject"] not in epochs_dict:
                n_skipped += 1
                continue
            for we in PHONEME_PAIR_TO_WORD_ENDS.get(row["phoneme_pair"], []):
                try:
                    phon_smin = (
                        int(row["causal6_smin"]) if row["causal6_smin"] is not None
                        else (int(row["causal4_smin"]) if row["causal4_smin"] is not None else None)
                    )
                    phon_smax = (
                        int(row["causal6_smax"]) if row["causal6_smax"] is not None
                        else (int(row["causal4_smax"]) if row["causal4_smax"] is not None else None)
                    )
                    ac_auc = (
                        float(row["causal6_test_roc_auc"]) if row["causal6_test_roc_auc"] is not None
                        else (float(row["causal4_peak_auc"]) if row["causal4_peak_auc"] is not None else None)
                    )
                    fig = provisional_star_plot(
                        subject=row["subject"],
                        electrode_idx=int(row["electrode_idx"]),
                        phoneme_pair=row["phoneme_pair"],
                        word_end=we,
                        epochs_dict=epochs_dict,
                        ambig_steps=ambig_steps,
                        phon_smin=phon_smin,
                        phon_smax=phon_smax,
                        acoustic_peak_auc=ac_auc,
                    )
                    fig.suptitle(
                        f"{title_prefix}  |  {row['subject']} e{row['electrode_idx']} "
                        f"{row['phoneme_pair']} -> {we}",
                        y=1.02, fontsize=10,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    n_pages += 1
                except Exception as ex:
                    print(f"  star_plot failed for {row['subject']} e{row['electrode_idx']} "
                          f"{row['phoneme_pair']} {we}: {ex}")
                    plt.close("all")
    print(f"Wrote {out_path.name}: {n_pages} pages  ({n_skipped} sites skipped: no epochs)")


# %%
losses_sorted = (
    recon.filter(pl.col("bucket") == "causal4_only")
         .sort("causal4_peak_auc", descending=True)
)
render_gallery(losses_sorted, OUT_DIR / "losses.pdf", title_prefix="LOSS")

# %%
ge_sorted = (
    recon.filter(pl.col("bucket") == "causal6_only_eligible")
         .sort("causal6_test_roc_auc", descending=True)
)
render_gallery(ge_sorted, OUT_DIR / "gains_eligible.pdf", title_prefix="GAIN(elig)")

# %%
gne_sorted = (
    recon.filter(pl.col("bucket") == "causal6_only_newly_eligible")
         .sort("causal6_test_roc_auc", descending=True)
)
render_gallery(gne_sorted, OUT_DIR / "gains_newly_eligible.pdf", title_prefix="GAIN(new)")

# %%
_both_n = recon.filter(pl.col("bucket") == "both").shape[0]
both_sample = (
    recon.filter(pl.col("bucket") == "both")
         .sample(min(10, _both_n), seed=0)
         .sort("causal6_test_roc_auc", descending=True)
)
render_gallery(both_sample, OUT_DIR / "both.pdf", title_prefix="BOTH")

# %% [markdown]
# ## Canonical AS-site list
#
# Initial canonical list = every site with `causal6_AS == True` (union of `both`,
# `causal6_only_eligible`, `causal6_only_newly_eligible`).
#
# The user may overwrite `canonical_AS_sites.csv` manually after reviewing the
# PDFs (e.g., to add back high-AUC `causal4_only` losses, or remove borderline
# gains). Downstream notebooks (Group B/C) MUST read from this CSV.

# %%
canonical = (
    recon.filter(pl.col("causal6_AS"))
         .select([
             "subject", "electrode_idx", "phoneme_pair",
             pl.col("causal6_smin").alias("smin"),
             pl.col("causal6_smax").alias("smax"),
             pl.col("causal6_test_roc_auc").alias("peak_auc"),
             pl.col("causal6_p_value").alias("p_value"),
             "bucket",
         ])
         .sort(["subject", "electrode_idx", "phoneme_pair"])
)
canonical.write_csv(OUT_DIR / "canonical_AS_sites.csv")
print(f"Canonical AS sites: {canonical.shape[0]}")
print(f"Written: {OUT_DIR / 'canonical_AS_sites.csv'}")
print(canonical.group_by("bucket").len().sort("len", descending=True))

# %% [markdown]
# ## Review checklist for the user
#
# 1. Open `outputs/causal46_joined/losses.pdf` — are any of the highest-AUC
#    losses visually compelling (clear divergence between step 1 and step 6
#    HGA in the top panel within the shaded acoustic window)? If yes, causal6
#    NHST may be over-conservative; consider relaxing the p threshold or
#    keeping selected sites manually.
# 2. Open `outputs/causal46_joined/gains_eligible.pdf` — do the gains look
#    real? If most are noisy, causal6 NHST may have inflated power (e.g.,
#    insufficient permutations).
# 3. Open `outputs/causal46_joined/gains_newly_eligible.pdf` — these sites
#    were rejected by causal4's speech-responsive pre-screen. Validating
#    these supports dropping the pre-screen.
# 4. Edit `outputs/causal46_joined/canonical_AS_sites.csv` manually if you
#    want to override the default (causal6_AS = True) selection.
# 5. The 3 absent subjects (EC248, EC250, EC253) are NOT in the canonical
#    list. When causal6 prod is re-synced with them, re-run this notebook.
# 6. The star-plot galleries skip silently if epoch files are not available
#    under `<REPO>/outputs/epochs_preprocessed/`. Set `BARAKEET_EPOCH_DIR`
#    to point at the actual epoch directory, then re-run.
