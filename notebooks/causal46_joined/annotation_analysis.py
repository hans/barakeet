# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: barakeet
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Annotation manifest — scratch analyses
#
# Explores relationships in the hand-annotated b4 star-plot review at
# `outputs/causal46_joined/manual_annotations/filtered_manifest.csv`.
#
# Schema doc: `notebooks/causal46_joined/manual_annotation_schema.md`.
#
# Three hypotheses, framed from the annotator's running notes:
#
# 1. **Mirrored-polarity spatial distribution.** For site×pair groups labeled
#    `mirrored…`, where does the "positive" sign live and is it predicted by
#    site acoustic tuning + the WE's intended onset?
# 2. **Cross-WE morphology vs per-cell significance.** Does pair-consistent
#    morphology (matched/mirrored) systematically include cells that fail
#    individual significance — quantifying the lift the cross-WE pooled test
#    is going after?
# 3. **Ambiguity-independence by (WE × acoustic-tuning) consistency.** When
#    the WE's intended onset matches the site's acoustic tuning, does the
#    late behavioral contrast also appear in unambiguous trials more often
#    than when they're inconsistent?

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.stimuli import WORD_END_TO_PHONEME_PAIR

ROI_DIR = Path("/Users/jon/freesurfer_subjects/barakeet/causal6_speech_responsive_pipeline/find_speech_responsive")

CSV = Path("outputs/causal46_joined/manual_annotations/filtered_manifest.csv")
df = pl.read_csv(CSV)
print(f"loaded {df.height} rows")

# %% [markdown]
# ## Normalize the manual annotation columns
#
# - Rename to regex-friendly names.
# - Parse `matched/mirrored` into `match_polarity` + `match_bins`.
# - Add WE intended onset and (WE × acoustic-tuning) consistency flag.

# %%
WE_ONSET = {
    "desolate": "d", "necessary": "n",
    "bountiful": "b", "mountains": "m",
    "penecillin": "p", "beneficial": "b",
}

# Letter set used to distinguish single-letter tunings from {both, complex, two peaks}.
SINGLE_LETTERS = {"b", "m", "d", "n", "p"}


def parse_match(s: str | None) -> tuple[str, frozenset[str]]:
    """Returns (polarity, bins). polarity ∈ {matched, mirrored, unspecified, n, none}."""
    if s is None:
        return ("none", frozenset())
    s = s.strip()
    if s == "n":
        return ("n", frozenset())
    if s == "y":
        return ("unspecified", frozenset())
    s_low = s.lower()
    if "matched" in s_low or "yes, long sustained" in s_low:
        polarity = "matched"
    elif "mirrored" in s_low:
        polarity = "mirrored"
    else:
        polarity = "unspecified"
    bins: set[str] = set()
    for tag, key in [
        ("@ac slightly late", "slightly_late"),
        ("@slightly late", "slightly_late"),
        ("@ac", "ac"),
        ("@late", "late"),
        ("sustain", "sustain"),
        ("yes, long sustained", "sustain"),
        ("all", "all"),
    ]:
        if tag in s_low:
            bins.add(key)
    if not bins and polarity in ("matched", "mirrored"):
        bins.add("all")
    return (polarity, frozenset(bins))


_parsed = [parse_match(s) for s in df["matched/mirrored morphology across WE?"].to_list()]
df = df.with_columns([
    pl.Series("match_polarity", [p for p, _ in _parsed]),
    pl.Series("match_bins", [",".join(sorted(b)) if b else "" for _, b in _parsed]),
    pl.col("word_end").replace(WE_ONSET).alias("we_onset"),
])

# (WE × acoustic-tuning) consistency: only meaningful when acoustic_tuning is single-letter
df = df.with_columns([
    pl.when(pl.col("acoustic tuning").is_in(list(SINGLE_LETTERS)))
        .then(pl.col("acoustic tuning") == pl.col("we_onset"))
        .otherwise(None)
        .alias("we_ac_consistent"),
])

# Derived sus: behav @ac single-letter & acoustic_tuning single-letter & tunings differ.
df = df.with_columns([
    (pl.col("behav @ac").is_in(list(SINGLE_LETTERS))
     & pl.col("acoustic tuning").is_in(list(SINGLE_LETTERS))
     & (pl.col("behav @ac") != pl.col("acoustic tuning"))).alias("sus_derived"),
])

# Has-any-behav flag
df = df.with_columns([
    (pl.col("behav @ac").is_not_null()
     | pl.col("behav @ac slightly late").is_not_null()
     | pl.col("behav @late").is_not_null()).alias("any_behav"),
])

print("polarity counts:")
print(df["match_polarity"].value_counts(sort=True))
print("\nmatch_bins counts:")
print(df["match_bins"].value_counts(sort=True))
print("\nwe_ac_consistent (single-letter acoustic only):")
print(df["we_ac_consistent"].value_counts(sort=True))

# %% [markdown]
# ## Attach ROI / anatomy

# %%
roi_frames = []
for subj in sorted(df["subject"].unique().to_list()):
    p = ROI_DIR / f"{subj}_results.csv"
    if not p.exists():
        print(f"⚠ no ROI file for {subj}: {p}")
        continue
    sub = pl.read_csv(p, columns=["electrode_idx", "roi", "x", "y", "z", "subject"])
    roi_frames.append(sub)

if roi_frames:
    electrode_meta = pl.concat(roi_frames, how="diagonal_relaxed")
else:
    electrode_meta = pl.DataFrame(schema={"subject": pl.Utf8, "electrode_idx": pl.Int64,
                                          "roi": pl.Utf8})

df = df.join(electrode_meta, on=["subject", "electrode_idx"], how="left") \
       .with_columns(pl.col("roi").fill_null("unknown"))
print(f"\nROI coverage: {df.filter(pl.col('roi') != 'unknown').height}/{df.height}")
print(df["roi"].value_counts(sort=True).head(10))

# %% [markdown]
# ## H1 — Mirrored-polarity sites: where, and which WE gets the "positive" sign?
#
# **Sign convention.** Use the bootstrap's aligned mean-diff direction
# (`best_mean_diff_aligned_med` > 0 ⇔ the acoustically-preferred class shows
# higher HGA in the best behavioral window). For mirrored cells the two WEs'
# signs differ by construction; we tabulate which WE gets the positive sign
# and check whether `we_ac_consistent` predicts it.

# %%
mirrored = df.filter(pl.col("match_polarity") == "mirrored")
print(f"mirrored cells: {mirrored.height}  "
      f"(in {mirrored.select(['subject','electrode_idx','phoneme_pair']).unique().height} groups)")

# Per cell: is the bootstrap sign positive (aligned > 0)?
mirrored = mirrored.with_columns(
    (pl.col("best_mean_diff_aligned_med") > 0).alias("boot_positive")
)

# For each mirrored cell, cross-tab (we_ac_consistent, boot_positive).
print("\n(we_ac_consistent × bootstrap-positive-sign) for mirrored cells")
print(
    mirrored
    .filter(pl.col("we_ac_consistent").is_not_null())
    .group_by(["we_ac_consistent", "boot_positive"])
    .len()
    .sort(["we_ac_consistent", "boot_positive"])
)

print("\nROI distribution of mirrored cells (recomputed after ROI join):")
mirrored_with_roi = df.filter(pl.col("match_polarity") == "mirrored")
print(mirrored_with_roi.group_by("roi").len().sort("len", descending=True))

print("\nMNI-x of mirrored vs matched vs none (median, IQR):")
print(
    df.filter(pl.col("x").is_not_null())
      .group_by("match_polarity")
      .agg(
          pl.len().alias("n"),
          pl.col("x").median().alias("x_med"),
          pl.col("x").quantile(0.25).alias("x_q25"),
          pl.col("x").quantile(0.75).alias("x_q75"),
      )
      .sort("match_polarity")
)

# Per-cell manual tuning vs WE onset — does the WE whose behav tuning == we_onset
# get the bootstrap-positive sign?
def _first_letter(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip().split(",")[0].split(" then ")[0].strip()
    return s if s in SINGLE_LETTERS else None


# Pick best representative behav tuning per row: prefer @late, then @ac, then @slightly_late
def _rep(row):
    for col in ("behav @late", "behav @ac", "behav @ac slightly late"):
        v = _first_letter(row[col])
        if v is not None:
            return v
    return None


mirrored_pd = mirrored.to_pandas()
mirrored_pd["behav_rep"] = mirrored_pd.apply(_rep, axis=1)
mirrored_pd["behav_matches_we_onset"] = (
    mirrored_pd["behav_rep"] == mirrored_pd["we_onset"]
)
print("\n(behav tuning == WE onset) × (bootstrap-positive) for mirrored cells")
print(mirrored_pd.groupby(["behav_matches_we_onset", "boot_positive"], dropna=False).size())

# %% [markdown]
# ## H2 — Cross-WE consistency vs per-cell significance
#
# Mirrors the cross-WE plan's motivation. For each site×pair group, ask:
# of cells where the annotator says morphology matches/mirrors across WEs,
# how many would be missed by per-cell CI tests?

# %%
# Resolve `y` (unspecified) → matched/mirrored by inspecting per-WE rep tunings
# in this site×pair group. Per schema doc: same tuning across the two WE rows
# → matched; opposite → mirrored. Fall back to unspecified.
df_pd = df.to_pandas()
df_pd["behav_rep"] = df_pd.apply(_rep, axis=1)
unsp_groups = df_pd[df_pd["match_polarity"] == "unspecified"].groupby(
    ["subject", "electrode_idx", "phoneme_pair"]
)["behav_rep"].apply(lambda s: list(s.dropna()))
resolved_groups: dict[tuple, str] = {}
for key, vals in unsp_groups.items():
    if len(vals) < 2:
        continue
    resolved_groups[key] = "matched" if vals[0] == vals[1] else "mirrored"
print(f"resolved {len(resolved_groups)} 'y' groups: "
      f"{sum(v=='matched' for v in resolved_groups.values())} matched, "
      f"{sum(v=='mirrored' for v in resolved_groups.values())} mirrored")

def _resolved_polarity(row):
    if row["match_polarity"] != "unspecified":
        return row["match_polarity"]
    key = (row["subject"], row["electrode_idx"], row["phoneme_pair"])
    return resolved_groups.get(key, "unspecified")

df_pd["match_polarity_resolved"] = df_pd.apply(_resolved_polarity, axis=1)
df = pl.from_pandas(df_pd[["subject", "electrode_idx", "phoneme_pair", "word_end",
                            "match_polarity_resolved"]]).join(
    df, on=["subject", "electrode_idx", "phoneme_pair", "word_end"], how="right"
)

df = df.with_columns(
    pl.col("match_polarity_resolved").is_in(["matched", "mirrored"]).alias("morph_match"),
)
group_keys = ["subject", "electrode_idx", "phoneme_pair"]
group = (
    df.group_by(group_keys)
      .agg(
          pl.col("morph_match").any().alias("any_match"),
          pl.col("match_polarity_resolved").first().alias("match_polarity"),
          pl.col("best_ci_aligned_excludes_zero").cast(pl.Int64).sum().alias("n_cells_sig"),
          pl.col("any_behav").cast(pl.Int64).sum().alias("n_cells_with_behav"),
          pl.col("best_mean_diff_aligned_med").abs().median().alias("median_abs_effect"),
          pl.len().alias("n_we"),
      )
)
print("group totals by match_polarity (with both WEs present):")
print(
    group.filter(pl.col("n_we") == 2)
         .group_by("match_polarity")
         .agg(
             pl.len().alias("n_groups"),
             pl.col("n_cells_sig").sum().alias("total_cells_sig"),
             (pl.col("n_cells_sig") == 0).sum().alias("groups_0_sig"),
             (pl.col("n_cells_sig") == 1).sum().alias("groups_1_sig"),
             (pl.col("n_cells_sig") == 2).sum().alias("groups_2_sig"),
             pl.col("median_abs_effect").median().alias("median_abs_effect"),
         )
         .sort("n_groups", descending=True)
)

# Lift target: matched-or-mirrored groups where neither cell is individually sig.
match_or_mirror = group.filter(
    (pl.col("n_we") == 2)
    & pl.col("match_polarity").is_in(["matched", "mirrored"])
)
print(f"\nmatched-or-mirrored 2-WE groups: {match_or_mirror.height}")
print(f"  both cells individually sig: "
      f"{match_or_mirror.filter(pl.col('n_cells_sig') == 2).height}")
print(f"  exactly 1 sig:               "
      f"{match_or_mirror.filter(pl.col('n_cells_sig') == 1).height}")
print(f"  neither sig (cross-WE LIFT target): "
      f"{match_or_mirror.filter(pl.col('n_cells_sig') == 0).height}")

# %% [markdown]
# ## H3 — Ambiguity-independence by (WE × acoustic-tuning) consistency
#
# Rows where the WE's intended onset matches the site's acoustic tuning vs.
# rows where it doesn't. Within each cohort, what fraction of cells with a
# late behav response also have the late contrast visible on unambiguous
# trials?

# %%
late_only = df.filter(pl.col("behav @late").is_not_null()).with_columns(
    pl.col("later contrast also present in unambiguous?").is_not_null().alias("unambig_match"),
)

print(f"rows with behav @late filled: {late_only.height}")
print(
    late_only
    .filter(pl.col("we_ac_consistent").is_not_null())
    .group_by("we_ac_consistent")
    .agg(
        pl.len().alias("n"),
        pl.col("unambig_match").cast(pl.Int64).sum().alias("n_unambig"),
        pl.col("unambig_match").cast(pl.Float64).mean().alias("frac_unambig"),
    )
    .sort("we_ac_consistent")
)

# Fisher's exact on the 2x2 to put a CI around the H3 trend.
try:
    from scipy.stats import fisher_exact
    tab = (
        late_only.filter(pl.col("we_ac_consistent").is_not_null())
                 .group_by(["we_ac_consistent", "unambig_match"])
                 .len().sort(["we_ac_consistent", "unambig_match"])
    )
    print("\n2x2 cell counts (we_ac_consistent × unambig_match):")
    print(tab)
    # Build the 2x2 in a fixed order: [[consistent=F & match=F, consistent=F & match=T],
    #                                  [consistent=T & match=F, consistent=T & match=T]]
    cnt = {(r["we_ac_consistent"], r["unambig_match"]): r["len"] for r in tab.iter_rows(named=True)}
    table = [[cnt.get((False, False), 0), cnt.get((False, True), 0)],
             [cnt.get((True, False), 0),  cnt.get((True, True), 0)]]
    odds, p = fisher_exact(table, alternative="two-sided")
    print(f"Fisher's exact: odds(consistent vs not, of having unambig match) = {odds:.3f}, p = {p:.3f}")
    print("H3 prediction (consistent→more unambig) would expect odds > 1; observed ≈", odds)
except ImportError:
    print("scipy not available, skipping Fisher test")

# Same but stratified by phoneme_pair
print("\nstratified by phoneme_pair:")
print(
    late_only
    .filter(pl.col("we_ac_consistent").is_not_null())
    .group_by(["phoneme_pair", "we_ac_consistent"])
    .agg(
        pl.len().alias("n"),
        pl.col("unambig_match").cast(pl.Float64).mean().alias("frac_unambig"),
    )
    .sort(["phoneme_pair", "we_ac_consistent"])
)

# %% [markdown]
# ## H1b — Is the WE-onset / acoustic-tuning relationship informative even on
# matched (same-polarity) sites?
#
# If matched sites tend to have behav tuning == acoustic tuning, that
# corroborates a shared population; if matched but flipped (behav tuning ==
# opposite of acoustic), something else is going on.

# %%
matched = df.filter(pl.col("match_polarity") == "matched")
matched_pd = matched.to_pandas()
matched_pd["behav_rep"] = matched_pd.apply(_rep, axis=1)
matched_pd["behav_matches_ac"] = (
    matched_pd["behav_rep"] == matched_pd["acoustic tuning"]
)
matched_pd["behav_matches_we_onset"] = (
    matched_pd["behav_rep"] == matched_pd["we_onset"]
)
print("matched cells: tuning vs acoustic tuning")
print(matched_pd.groupby(
    ["behav_matches_ac", "behav_matches_we_onset"], dropna=False
).size())

# %% [markdown]
# ## H4 — `sus` flag rate by ROI and acoustic tuning
#
# Quick characterization of "behav in ac window with wrong tuning" sites.

# %%
print("derived sus rate by ROI:")
print(
    df.filter(pl.col("behav @ac").is_in(list(SINGLE_LETTERS)))
      .group_by("roi")
      .agg(
          pl.len().alias("n_with_behav_ac"),
          pl.col("sus_derived").cast(pl.Int64).sum().alias("n_sus"),
      )
      .with_columns((pl.col("n_sus") / pl.col("n_with_behav_ac")).alias("frac_sus"))
      .sort("n_sus", descending=True)
)
