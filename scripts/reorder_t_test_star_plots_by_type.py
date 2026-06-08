"""Re-order T-Test star-plot gallery by (early_type × late_type).

Source PDF: outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered.pdf
  Pages are in the same order as powered=True rows in the T-Tests
  filtered_manifest (which corresponds to b4_per_cell cell-key order).

Group key: Cartesian product of
  early_type  — human-readable label from early_acoustic_window.csv
                (site_type_relabel mapped through SITE_TYPE_LABELS;
                 ETC_SITE_TYPES and all non-canonical values → "Other")
  late_type   — whether the site×pair has a late behavioral response in
                manual_annotations/filtered_manifest.csv:
                  absent    = 0 non-null 'behav @late' rows
                  one-sided = 1 non-null 'behav @late' row
                  two-sided = 2 non-null 'behav @late' rows

Pages are sorted by (early_sort_key, late_sort_key, subject, electrode_idx,
phoneme_pair, word_end).  A title page is inserted before each group and
registered as a PDF outline bookmark.

Pages whose site×pair is absent from early_acoustic_window.csv are placed in
an "unknown early" group rather than silently dropped.

Usage
-----
    uv run python scripts/reorder_t_test_star_plots_by_type.py \\
        --t-tests-manifest outputs/causal46_joined/t_tests/star_plots_filtered/filtered_manifest.csv \\
        --early            outputs/causal46_joined/manual_annotations/early_acoustic_window.csv \\
        --late             outputs/causal46_joined/manual_annotations/filtered_manifest.csv \\
        --src              outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered.pdf \\
        --out              outputs/causal46_joined/t_tests/star_plots_filtered/b4_powered_by_type.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal46_joined import (  # noqa: E402
    SITE_TYPE_ORDER,
    make_title_page,
    site_type_display_label,
)

# -----------------------------------------------------------------------
# Early-type: human-readable label used as the group key.
# Sort order follows SITE_TYPE_ORDER (first canonical raw type that maps
# to each label wins the position).
# -----------------------------------------------------------------------
_seen: set[str] = set()
_EARLY_LABEL_ORDER: list[str] = []
for _raw in SITE_TYPE_ORDER:
    _lbl = site_type_display_label(_raw)
    if _lbl not in _seen:
        _seen.add(_lbl)
        _EARLY_LABEL_ORDER.append(_lbl)
_EARLY_LABEL_SORT: dict[str, int] = {lbl: i for i, lbl in enumerate(_EARLY_LABEL_ORDER)}

UNKNOWN_EARLY_LABEL = "Unknown early"

# -----------------------------------------------------------------------
# Late-window categories (same logic as sankey_early_late.py)
# -----------------------------------------------------------------------
LATE_ORDER = ["absent", "one-sided", "two-sided"]
LATE_LABELS = {
    "absent":    "Late window\nabsent",
    "one-sided": "Late window\n(one-sided)",
    "two-sided": "Late window\n(two-sided)",
}
_LATE_SORT: dict[str, int] = {t: i for i, t in enumerate(LATE_ORDER)}


def _late_category(n_nonnull: int) -> str:
    if n_nonnull == 0:
        return "absent"
    elif n_nonnull == 1:
        return "one-sided"
    return "two-sided"


def _group_label(early_label: str, late: str) -> str:
    return f"{early_label}\n×\n{LATE_LABELS.get(late, late)}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--t-tests-manifest", required=True,
        help="outputs/causal46_joined/t_tests/star_plots_filtered/filtered_manifest.csv",
    )
    p.add_argument(
        "--early", required=True,
        help="outputs/causal46_joined/manual_annotations/early_acoustic_window.csv",
    )
    p.add_argument(
        "--late", required=True,
        help="outputs/causal46_joined/manual_annotations/filtered_manifest.csv",
    )
    p.add_argument(
        "--src", required=True,
        help="Source PDF (b4_powered.pdf or b4_powered_significant.pdf)",
    )
    p.add_argument("--out", required=True, help="Output PDF path")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        sys.exit("pypdf is required.  Install with:  pip install pypdf")

    src_path = Path(args.src)
    out_path = Path(args.out)

    # ------------------------------------------------------------------
    # 1. Load T-Tests filtered_manifest: powered=True rows give page order
    # ------------------------------------------------------------------
    ttm = pl.read_csv(args.t_tests_manifest)
    powered = ttm.filter(pl.col("powered").cast(pl.Boolean))
    n_powered = powered.height

    src_pdf = PdfReader(str(src_path))
    n_pages = len(src_pdf.pages)
    if n_pages != n_powered:
        sys.exit(
            f"Page count mismatch: {src_path.name} has {n_pages} pages but "
            f"filtered_manifest has {n_powered} powered rows.\n"
            "Re-run the t_tests notebook to regenerate the source PDF."
        )

    # ------------------------------------------------------------------
    # 2. Load early_acoustic_window.csv → human-readable label as group key
    # ------------------------------------------------------------------
    early_ann = pl.read_csv(args.early).select(
        ["subject", "electrode_idx", "phoneme_pair", "site_type_relabel"]
    ).with_columns(
        pl.col("site_type_relabel")
        .str.strip_chars()
        .map_elements(site_type_display_label, return_dtype=pl.String)
        .alias("early_label")
    )

    # ------------------------------------------------------------------
    # 3. Load manual filtered_manifest → late_category per site×pair
    # ------------------------------------------------------------------
    late_ann = pl.read_csv(args.late)
    # 'behav @late' is non-null when the cell has a late behavioral response
    late_per_pair = (
        late_ann
        .with_columns(
            (
                pl.col("behav @late").is_not_null()
                & (pl.col("behav @late").cast(pl.String).str.strip_chars() != "")
            ).alias("_has_late")
        )
        .group_by(["subject", "electrode_idx", "phoneme_pair"])
        .agg(pl.col("_has_late").sum().alias("n_late"))
        .with_columns(
            pl.col("n_late")
            .map_elements(_late_category, return_dtype=pl.String)
            .alias("late_category")
        )
    )

    # ------------------------------------------------------------------
    # 4. Verification: reproduce Sankey cross-tab at site×pair level
    # ------------------------------------------------------------------
    site_pair_keys = ["subject", "electrode_idx", "phoneme_pair"]
    early_for_xcheck = (
        early_ann.select(site_pair_keys + ["early_label"])
        .unique(subset=site_pair_keys)
    )
    sankey_check = (
        early_for_xcheck
        .join(late_per_pair, on=site_pair_keys, how="left")
        .with_columns(pl.col("late_category").fill_null("absent"))
    )
    ct = (
        sankey_check
        .group_by(["early_label", "late_category"])
        .len()
        .sort(["early_label", "late_category"])
    )
    print("\nSankey cross-tab verification (site×pair level):")
    print(ct.to_pandas().pivot(index="early_label", columns="late_category", values="len")
          .fillna(0).astype(int).to_string())
    print()

    # ------------------------------------------------------------------
    # 5. Join categories onto the T-Tests page list
    # ------------------------------------------------------------------
    page_list = (
        powered
        .join(early_ann.select(site_pair_keys + ["early_label"]),
              on=site_pair_keys, how="left")
        .join(late_per_pair.select(site_pair_keys + ["late_category"]),
              on=site_pair_keys, how="left")
        .with_columns([
            pl.col("early_label").fill_null(UNKNOWN_EARLY_LABEL),
            pl.col("late_category").fill_null("absent"),
        ])
    )

    n_unknown = (page_list["early_label"] == UNKNOWN_EARLY_LABEL).sum()
    if n_unknown:
        print(f"⚠ {n_unknown} page(s) with no early annotation → placed in 'Unknown early' group")

    # ------------------------------------------------------------------
    # 6. Compute sort keys and sort
    # ------------------------------------------------------------------
    page_list = page_list.with_columns([
        pl.col("early_label")
        .map_elements(
            lambda lbl: _EARLY_LABEL_SORT.get(lbl, len(_EARLY_LABEL_ORDER)),
            return_dtype=pl.Int32,
        )
        .alias("_early_sort"),
        pl.col("late_category")
        .map_elements(lambda c: _LATE_SORT.get(c, len(LATE_ORDER)), return_dtype=pl.Int32)
        .alias("_late_sort"),
    ])

    # Attach original 0-based page index before sorting
    page_list = page_list.with_row_index("_page_idx")

    page_list = page_list.sort(
        ["_early_sort", "_late_sort", "subject", "electrode_idx", "phoneme_pair", "word_end"]
    )

    # ------------------------------------------------------------------
    # 7. Write output PDF with title pages and outline
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    output_page_num = 0

    # Pre-count pages per group for the title-page subtitle
    group_counts: dict[tuple[str, str], int] = {}
    for row in page_list.iter_rows(named=True):
        key = (row["early_label"], row["late_category"])
        group_counts[key] = group_counts.get(key, 0) + 1

    current_group: tuple[str, str] | None = None

    for row in page_list.iter_rows(named=True):
        group_key = (row["early_label"], row["late_category"])
        if group_key != current_group:
            current_group = group_key
            label = _group_label(*group_key)
            title_pg = make_title_page(label, group_counts[group_key])
            writer.add_page(title_pg)
            writer.add_outline_item(label.replace("\n", " "), output_page_num)
            output_page_num += 1

        writer.add_page(src_pdf.pages[int(row["_page_idx"])])
        output_page_num += 1

    with out_path.open("wb") as fh:
        writer.write(fh)

    n_groups = len(group_counts)
    n_content = output_page_num - n_groups
    print(f"wrote {out_path}")
    print(f"  {n_content} star-plot pages + {n_groups} title pages  "
          f"({n_groups} groups)")
    print("\nPages by group:")
    for (early_lbl, late), cnt in sorted(
        group_counts.items(),
        key=lambda kv: (_EARLY_LABEL_SORT.get(kv[0][0], len(_EARLY_LABEL_ORDER)),
                        _LATE_SORT.get(kv[0][1], 99)),
    ):
        print(f"  {early_lbl.replace(chr(10), ' '):45s}  ×  "
              f"{LATE_LABELS.get(late, late).replace(chr(10), ' '):30s}  n={cnt}")


if __name__ == "__main__":
    main()
