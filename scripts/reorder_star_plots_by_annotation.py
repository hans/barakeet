"""Re-order compact star-plot pages by manual annotation.

Reads ``manual_annotations/early_acoustic_window.csv`` and uses the
``site_type_relabel`` column (the hand-annotated type) to determine the
ordering of pages.  Pages are sorted by:

    (site_type_sort_key, subject, electrode_idx, phoneme_pair)

A title page is inserted before each type group and registered as a PDF
outline (bookmark) entry so PDF viewers show a navigable section list.

Usage
-----
    uv run python scripts/reorder_star_plots_by_annotation.py \\
        --relabel  outputs/causal46_joined/manual_annotations/early_acoustic_window.csv \\
        --star-dir outputs/causal46_joined/early_window_site_types \\
        --out      outputs/causal46_joined/early_window_site_types/star_plots_by_annotation.pdf

The script expects one compact PDF per subject at
``{star_dir}/{subject}/star_plots_early_compact.pdf``.  The page order within
each compact PDF must match the row order of that subject's
``site_type_assignments.parquet`` (sorted by electrode_idx, phoneme_pair),
which is exactly what the per-subject notebook produces.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import polars as pl

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causal46_joined import (  # noqa: E402
    ETC_SITE_TYPES,
    site_type_display_label,
    site_type_sort_key,
)


def _make_title_page(label: str, count: int,
                     page_width: float = 612, page_height: float = 792):
    """Return a pypdf PageObject for a title page with centered text.

    Uses only the standard built-in Helvetica-Bold font so no font embedding
    is required.  ``label`` may contain literal newlines to split across lines.
    """
    from pypdf import PdfReader

    lines = label.split("\n")
    font_size = 48
    line_height = font_size * 1.4
    count_size = 24

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    # Approximate Helvetica-Bold char width (0.55 × font_size)
    char_w = font_size * 0.55
    count_char_w = count_size * 0.55
    count_label = f"n = {count}"

    total_h = len(lines) * line_height
    y_top = page_height / 2 + total_h / 2

    parts = ["BT", "/F1 48 Tf"]
    for i, line in enumerate(lines):
        x = (page_width - len(line) * char_w) / 2
        y = y_top - (i + 1) * line_height
        parts.append(f"1 0 0 1 {x:.1f} {y:.1f} Tm")
        parts.append(f"({_esc(line)}) Tj")

    # Count subtitle
    cx = (page_width - len(count_label) * count_char_w) / 2
    cy = y_top - len(lines) * line_height - count_size * 1.2
    parts += ["/F1 24 Tf", f"1 0 0 1 {cx:.1f} {cy:.1f} Tm", f"({_esc(count_label)}) Tj"]
    parts.append("ET")

    content = "\n".join(parts).encode("latin-1")

    buf = io.BytesIO()

    def w(s: str | bytes) -> None:
        buf.write(s.encode("latin-1") if isinstance(s, str) else s)

    offsets: dict[int, int] = {}

    w(b"%PDF-1.4\n")

    offsets[1] = buf.tell()
    w("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n")

    offsets[2] = buf.tell()
    w("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n")

    offsets[3] = buf.tell()
    w(f"3 0 obj\n<< /Type /Page /Parent 2 0 R "
      f"/MediaBox [0 0 {page_width:.0f} {page_height:.0f}] "
      f"/Contents 4 0 R "
      f"/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\n")

    offsets[4] = buf.tell()
    w(f"4 0 obj\n<< /Length {len(content)} >>\nstream\n")
    buf.write(content)
    w("\nendstream\nendobj\n\n")

    offsets[5] = buf.tell()
    w("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n\n")

    xref_pos = buf.tell()
    w("xref\n0 6\n")
    w(f"0000000000 65535 f \n")
    for i in range(1, 6):
        w(f"{offsets[i]:010d} 00000 n \n")
    w(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n")

    buf.seek(0)
    return PdfReader(buf).pages[0]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--relabel", required=True,
                   help="Path to manual_annotations/early_acoustic_window.csv")
    p.add_argument("--star-dir", required=True,
                   help="Directory containing per-subject subdirs with "
                        "star_plots_early_compact.pdf files")
    p.add_argument("--out", required=True,
                   help="Output PDF path")
    p.add_argument("--full", action="store_true",
                   help="Use star_plots_early.pdf (full-size) instead of compact")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        sys.exit("pypdf is required. Install with: pip install pypdf")

    relabel_path = Path(args.relabel)
    star_dir = Path(args.star_dir)
    out_path = Path(args.out)
    pdf_name = "star_plots_early.pdf" if args.full else "star_plots_early_compact.pdf"

    # ------------------------------------------------------------------
    # Load annotation table
    # ------------------------------------------------------------------
    ann = pl.read_csv(relabel_path)
    required = {"subject", "electrode_idx", "phoneme_pair", "site_type_relabel"}
    missing = required - set(ann.columns)
    if missing:
        sys.exit(f"early_acoustic_window.csv is missing columns: {missing}")

    # Use the manual site_type_relabel column directly as the effective type.
    # Collapse ETC_SITE_TYPES (A_unsigned, problematic, interesting) → "etc"
    # so they form a single "Other" section rather than three separate ones.
    _etc_set = set(ETC_SITE_TYPES)
    ann = ann.with_columns(
        pl.col("site_type_relabel")
        .str.strip_chars()
        .map_elements(lambda t: "etc" if t in _etc_set else t, return_dtype=pl.String)
        .alias("effective_type")
    )

    # Attach sort key
    ann = ann.with_columns(
        pl.col("effective_type")
        .map_elements(site_type_sort_key, return_dtype=pl.Int32)
        .alias("_sort_key")
    )

    # ------------------------------------------------------------------
    # Build per-subject page index
    # For each subject, load its parquet to get the exact per-page row order
    # (electrode_idx, phoneme_pair sorted), then map annotation rows to pages.
    # ------------------------------------------------------------------
    subjects = sorted(ann["subject"].unique().to_list())
    # page_records: list of (sort_key, subject, electrode_idx, phoneme_pair, pdf_path, page_idx, effective_type)
    page_records: list[tuple] = []

    for subj in subjects:
        pdf_path = star_dir / subj / pdf_name
        if not pdf_path.exists():
            print(f"  ⚠ {subj}: {pdf_name} not found, skipping")
            continue

        # Load site_type_assignments to get the exact page order
        assign_path = star_dir / subj / "site_type_assignments.parquet"
        if not assign_path.exists():
            print(f"  ⚠ {subj}: site_type_assignments.parquet not found, skipping")
            continue

        assign = (
            pl.read_parquet(assign_path)
            .select(["electrode_idx", "phoneme_pair"])
            .sort(["electrode_idx", "phoneme_pair"])
        )

        # Build lookup: (electrode_idx, phoneme_pair) → page index (0-based)
        page_map: dict[tuple, int] = {
            (int(r["electrode_idx"]), str(r["phoneme_pair"])): i
            for i, r in enumerate(assign.iter_rows(named=True))
        }

        n_pdf_pages = len(PdfReader(str(pdf_path)).pages)
        if n_pdf_pages != assign.height:
            print(f"  ⚠ {subj}: page count mismatch "
                  f"(PDF={n_pdf_pages}, parquet={assign.height}) — using parquet order")

        subj_ann = ann.filter(pl.col("subject") == subj)
        for row in subj_ann.iter_rows(named=True):
            ei = int(row["electrode_idx"])
            pp = str(row["phoneme_pair"])
            page_idx = page_map.get((ei, pp))
            if page_idx is None:
                print(f"  ⚠ {subj} e{ei} {pp}: not in parquet, skipping")
                continue
            if page_idx >= n_pdf_pages:
                print(f"  ⚠ {subj} e{ei} {pp}: page {page_idx} out of range ({n_pdf_pages}), skipping")
                continue
            page_records.append((
                int(row["_sort_key"]),
                subj,
                ei,
                pp,
                str(pdf_path),
                page_idx,
                str(row["effective_type"]),
            ))

    if not page_records:
        sys.exit("No pages to write — check that annotation and PDF paths match.")

    # Sort by (site_type_sort_key, subject, electrode_idx, phoneme_pair)
    page_records.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    # ------------------------------------------------------------------
    # Write output PDF with per-type title pages and outline entries
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache open readers to avoid reopening the same PDF repeatedly
    _readers: dict[str, PdfReader] = {}

    def _reader(path: str) -> PdfReader:
        if path not in _readers:
            _readers[path] = PdfReader(path)
        return _readers[path]

    # Pre-count sites per type for the title page subtitle
    type_counts: dict[str, int] = {}
    for rec in page_records:
        type_counts[rec[6]] = type_counts.get(rec[6], 0) + 1

    writer = PdfWriter()
    current_sort_key: int | None = None
    output_page_num = 0  # 0-based index of the next page to be added

    for sort_key, subj, ei, pp, pdf_path, page_idx, eff_type in page_records:
        if sort_key != current_sort_key:
            # Insert title page and outline entry for this group
            current_sort_key = sort_key
            label = site_type_display_label(eff_type)
            title_page = _make_title_page(label, type_counts[eff_type])
            writer.add_page(title_page)
            # Outline title replaces newlines with space for single-line display
            outline_title = label.replace("\n", " ")
            writer.add_outline_item(outline_title, output_page_num)
            output_page_num += 1

        writer.add_page(_reader(pdf_path).pages[page_idx])
        output_page_num += 1

    with out_path.open("wb") as fh:
        writer.write(fh)

    total_content = output_page_num - len(type_counts)
    print(f"wrote {out_path}  ({total_content} star-plot pages + {len(type_counts)} title pages)")

    print("\nPages by effective type:")
    for eff_type, count in sorted(type_counts.items(),
                                  key=lambda kv: site_type_sort_key(kv[0])):
        label = site_type_display_label(eff_type).replace("\n", " ")
        print(f"  {label:40s}  {count}")


if __name__ == "__main__":
    main()
