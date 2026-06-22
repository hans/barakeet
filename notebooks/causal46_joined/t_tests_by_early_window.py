# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytextVersion: 1.19.1
#   kernelspec:
#     display_name: barakeet
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Star plots grouped by early-window EP status
#
# For each powered B4 AS-site cell (from `t_tests.py`), classify the parent
# site+pair by how many of its word-ends appear in `ep_windows.parquet` (i.e.,
# have a significant perceptual contrast in the early acoustic window).
# Then write three separate star-plot galleries:
#
# | Group label | Criterion |
# |---|---|
# | `both_WEs_EP` | All WEs tested in `early_perceptual_windows` are EP-significant. Includes 2-of-2 and 1-of-1 (only WE tested). |
# | `one_WE_EP` | Exactly one of two tested WEs is EP-significant. |
# | `neither` | No WEs EP-significant — includes sites tested-but-not-significant AND sites never tested (not annotated `behav @ac`). |
#
# Note: `ep_windows.parquet` is only populated for cells annotated `behav @ac`
# in `filtered_manifest.csv`. For all other cells the early window was never
# searched, so they land in `neither` by absence.
#
# Outputs (relative to `outdir`):
# - `star_plots_by_ep_group/both_WEs_EP.pdf`
# - `star_plots_by_ep_group/one_WE_EP.pdf`
# - `star_plots_by_ep_group/neither.pdf`
# - `star_plots_by_ep_group/ep_group_manifest.csv`

# %% tags=["parameters"]
ep_windows_path = "outputs/causal46_joined/early_perceptual_windows/ep_windows.parquet"
b4_per_cell_path = "outputs/causal46_joined/t_tests/b4_per_cell.parquet"
b4_per_window_path = "outputs/causal46_joined/t_tests/b4_per_window.parquet"
b4_per_pair_path = "outputs/causal46_joined/t_tests/b4_per_pair.parquet"
filtered_manifest_path = "outputs/causal46_joined/manual_annotations/filtered_manifest.csv"
epoch_dir = "outputs/epochs_preprocessed"
behav_dec_full_root = "outputs_prod/causal46_joined/behavior_decoding_single_electrode"
behav_dec_hga_only_root = "outputs_prod/causal46_joined/behavior_decoding_single_electrode_hga_only"
outdir = "outputs/causal46_joined/t_tests"

# %%
import sys
from pathlib import Path

import polars as pl
import yaml

from src.viz_provisional import load_epochs_dict

REPO = Path(".").resolve()
sys.path.insert(0, str(REPO / "notebooks" / "causal46_joined"))
from _within_completion import load_behav_decoding_scores  # noqa: E402
from _star_gallery import write_annotated_pdfs  # noqa: E402

_cfg = yaml.safe_load((REPO / "config.yaml").read_text())
AC_SEARCH_SMIN = int(_cfg["analysis"]["decoding"].get("acoustic_peak_search_smin", 0))
AC_SEARCH_SMAX = int(_cfg["analysis"]["decoding"].get("acoustic_peak_search_smax", 50))

OUT_DIR = Path(outdir)
GALLERY_DIR = OUT_DIR / "star_plots_by_ep_group"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

CELL_KEYS = ["subject", "electrode_idx", "phoneme_pair", "word_end"]
SITE_KEYS = ["subject", "electrode_idx", "phoneme_pair"]

# %% [markdown]
# ## Load inputs

# %%
ep_windows = pl.read_parquet(ep_windows_path)
b4_per_cell = pl.read_parquet(b4_per_cell_path)
b4_per_window = pl.read_parquet(b4_per_window_path)
b4_per_pair = pl.read_parquet(b4_per_pair_path)
filtered_manifest = pl.read_csv(filtered_manifest_path)

print(f"ep_windows:      {ep_windows.height} rows")
print(f"b4_per_cell:     {b4_per_cell.height} rows")
print(f"b4_per_window:   {b4_per_window.height} rows")
print(f"b4_per_pair:     {b4_per_pair.height} rows")
print(f"filtered_manifest: {filtered_manifest.height} rows")

# %%
epochs_dict = load_epochs_dict(Path(epoch_dir))
print(f"epochs loaded: {sorted(epochs_dict)}")

behav_dec_by_subject: dict = {}
for subj in sorted(epochs_dict):
    try:
        df = load_behav_decoding_scores(
            f"{behav_dec_full_root}/{subj}/scores.parquet",
            f"{behav_dec_hga_only_root}/{subj}/scores.parquet",
        )
        behav_dec_by_subject[subj] = df
    except FileNotFoundError:
        pass
print(f"behavioral decoding scores loaded for: {sorted(behav_dec_by_subject)}")

# %% [markdown]
# ## Classify site+pairs by EP status
#
# For each `(subject, electrode_idx, phoneme_pair)`:
# - `ep_we_count` = number of distinct `word_end` values in `ep_windows` for
#   that site+pair (presence ≡ significance, since `ep_windows` only records
#   significant union windows and emits zero rows for non-significant cells).
# - `universe_we_count` = number of distinct `word_end` values in the `behav @ac`
#   rows of `filtered_manifest` for that site+pair (the denominator for early-EP
#   testing).

# %%
# Count EP-significant WEs per site+pair
ep_sig_per_pair = (
    ep_windows
    .group_by(SITE_KEYS)
    .agg(pl.col("word_end").n_unique().alias("ep_we_count"))
)

# Count universe WEs per site+pair from filtered_manifest (behav @ac rows)
behav_ac_fm = filtered_manifest.filter(pl.col("behav @ac").is_not_null())
universe_per_pair = (
    behav_ac_fm
    .group_by(SITE_KEYS)
    .agg(pl.col("word_end").n_unique().alias("universe_we_count"))
)

# Join: all b4_per_cell site+pairs; fill 0 for those absent from ep_windows / universe
site_classification = (
    b4_per_cell
    .select(SITE_KEYS)
    .unique()
    .join(ep_sig_per_pair, on=SITE_KEYS, how="left")
    .join(universe_per_pair, on=SITE_KEYS, how="left")
    .with_columns([
        pl.col("ep_we_count").fill_null(0),
        pl.col("universe_we_count").fill_null(0),
    ])
    .with_columns(
        pl.when(
            (pl.col("ep_we_count") > 0)
            & (pl.col("ep_we_count") == pl.col("universe_we_count"))
        ).then(pl.lit("both_WEs_EP"))
        .when(
            (pl.col("ep_we_count") > 0)
            & (pl.col("ep_we_count") < pl.col("universe_we_count"))
        ).then(pl.lit("one_WE_EP"))
        .otherwise(pl.lit("neither"))
        .alias("ep_group")
    )
)

print("Site+pair EP groups:")
print(site_classification.group_by("ep_group").len().sort("ep_group"))
print(site_classification)

# %%
# Tag each b4_per_cell row with its site's ep_group
b4_tagged = b4_per_cell.join(
    site_classification.select(SITE_KEYS + ["ep_group", "ep_we_count", "universe_we_count"]),
    on=SITE_KEYS, how="left",
)
print("Cell counts per ep_group:")
print(b4_tagged.group_by("ep_group").len().sort("ep_group"))

# %% [markdown]
# ## Build pair lookup for cross-WE pooled test bars

# %%
pair_lut = {
    (r["subject"], int(r["electrode_idx"]), r["phoneme_pair"]): r
    for r in b4_per_pair.iter_rows(named=True)
} if b4_per_pair.height else None

# %% [markdown]
# ## Generate star-plot galleries per EP group

# %%
GROUP_ORDER = ["both_WEs_EP", "one_WE_EP", "neither"]
manifest_rows = []

for group in GROUP_ORDER:
    group_df = b4_tagged.filter(pl.col("ep_group") == group)
    entries = group_df.iter_rows(named=True)
    entries_list = list(entries)
    out_pdf = GALLERY_DIR / f"{group}.pdf"
    print(f"\n--- {group}: {len(entries_list)} cells ---")
    n = write_annotated_pdfs(
        entries_list,
        b4_per_window,
        CELL_KEYS,
        out_pdf,
        epochs_dict=epochs_dict,
        pair_lookup=pair_lut,
        ac_search_smin=AC_SEARCH_SMIN,
        ac_search_smax=AC_SEARCH_SMAX,
        behav_dec_by_subject=behav_dec_by_subject,
    )
    print(f"  wrote {n} pages → {out_pdf}")
    for row in entries_list:
        manifest_rows.append({
            "subject": row["subject"],
            "electrode_idx": row["electrode_idx"],
            "phoneme_pair": row["phoneme_pair"],
            "word_end": row["word_end"],
            "ep_group": group,
            "ep_we_count": row.get("ep_we_count"),
            "universe_we_count": row.get("universe_we_count"),
            "best_ci_aligned_excludes_zero": row.get("best_ci_aligned_excludes_zero"),
            "best_emp_p_aligned": row.get("best_emp_p_aligned"),
            "best_mean_diff_aligned_med": row.get("best_mean_diff_aligned_med"),
            "pair_ci_excludes_zero": row.get("pair_ci_excludes_zero"),
            "pair_emp_p": row.get("pair_emp_p"),
        })

manifest_df = pl.DataFrame(manifest_rows)
manifest_path = GALLERY_DIR / "ep_group_manifest.csv"
manifest_df.write_csv(manifest_path)
print(f"\nwrote {manifest_path}  ({manifest_df.height} rows)")
print(manifest_df.group_by("ep_group").len().sort("ep_group"))

# %% [markdown]
# ## Done
