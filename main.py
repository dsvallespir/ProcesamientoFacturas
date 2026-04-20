#!/usr/bin/env python3
"""
main.py – Command-line interface for the Invoice Processing system.

Usage examples
--------------
Process a single invoice and export to JSON::

    python main.py invoice.png

Specify output directory and format::

    python main.py invoice.pdf --output-dir ./results --format csv

Disable skew correction::

    python main.py invoice.png --no-deskew
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invoice-processor",
        description="Automatic invoice data extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", help="Path to the invoice image file.")
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        metavar="DIR",
        help="Directory where extracted data will be saved.",
    )
    parser.add_argument(
        "--format", "-f",
        default="json",
        choices=["json", "csv", "excel"],
        dest="fmt",
        help="Export format.",
    )
    parser.add_argument(
        "--lang", "-l",
        default="spa+eng",
        help="Tesseract language string.",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        default=False,
        help="Disable automatic skew correction.",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="Up-scale image to 300 DPI before OCR.",
    )
    parser.add_argument(
        "--current-dpi",
        type=int,
        default=72,
        metavar="DPI",
        help="Assumed DPI of the source image (used with --scale).",
    )
    parser.add_argument(
        "--print-entities",
        action="store_true",
        default=False,
        help="Print extracted entities to stdout as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}", file=sys.stderr)
        return 1

    # Import here so the CLI is importable even without all deps installed
    try:
        from invoice_processor import InvoicePipeline
    except ImportError as exc:
        print(f"ERROR: Could not import invoice_processor: {exc}", file=sys.stderr)
        return 1

    pipe = InvoicePipeline(
        lang=args.lang,
        do_deskew=not args.no_deskew,
        do_scale=args.scale,
        current_dpi=args.current_dpi,
    )

    print(f"Processing: {image_path}")
    try:
        result = pipe.process(
            image_path,
            output_dir=args.output_dir,
            fmt=args.fmt,
        )
    except Exception as exc:
        print(f"ERROR during processing: {exc}", file=sys.stderr)
        return 1

    entities = result["entities"]
    layout_count = len(result["layout"])
    words_count = len(result["ocr"].get("words", []))

    print(f"  Words detected : {words_count}")
    print(f"  Layout regions : {layout_count}")
    print(f"  Supplier       : {entities.get('supplier')}")
    print(f"  Invoice No.    : {entities.get('invoice_number')}")
    print(f"  Date           : {entities.get('date')}")
    print(f"  Total          : {entities.get('total')} {entities.get('currency') or ''}")
    print(f"  Line items     : {len(entities.get('line_items', []))}")

    if result["output_file"]:
        print(f"  Saved to       : {result['output_file']}")

    if args.print_entities:
        print("\n--- Entities (JSON) ---")
        print(json.dumps(entities, ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
