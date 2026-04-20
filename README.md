# ProcesamientoFacturas

Automatic invoice data extraction pipeline for Spanish and English invoices.

## Overview

This project implements a complete invoice-processing pipeline:

| Step | Module | Description |
|------|--------|-------------|
| 1. Preprocessing | `invoice_processor.preprocessor` | Image cleaning – grayscale, deskew, noise removal, binarisation (OpenCV) |
| 2. OCR | `invoice_processor.ocr` | Text + bounding-box extraction (pytesseract / Tesseract-OCR) |
| 3. Layout Analysis | `invoice_processor.layout_analyzer` | Table, header and logo region detection |
| 4. Entity Extraction | `invoice_processor.entity_extractor` | Key Information Extraction – supplier, date, totals, line items |
| 5. Export | `invoice_processor.exporter` | Serialise to JSON, CSV or Excel (pandas/openpyxl) |

## Requirements

* Python ≥ 3.9
* [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract) installed and on `PATH`
  * Language packs: `eng` (English) and `spa` (Spanish) recommended

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr tesseract-ocr-spa

# macOS (Homebrew)
brew install tesseract tesseract-lang
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Command-line interface

```bash
python main.py invoice.png
```

```bash
python main.py invoice.png --output-dir ./results --format csv
python main.py invoice.png --output-dir ./results --format excel
python main.py invoice.png --print-entities
python main.py invoice.png --no-deskew --scale --current-dpi 96
```

### Python API

```python
from invoice_processor import InvoicePipeline

pipe = InvoicePipeline(lang="spa+eng", do_deskew=True)

result = pipe.process("invoice.png", output_dir="output/", fmt="json")

entities = result["entities"]
print(entities["supplier"])       # e.g. "Tecnología Global S.A. de C.V."
print(entities["date"])           # e.g. "2024-03-15"
print(entities["total"])          # e.g. 9280.0
print(entities["currency"])       # e.g. "MXN"
print(entities["line_items"])     # list of {description, quantity, unit_price, total}
```

The `result` dictionary also contains:

| Key | Type | Description |
|-----|------|-------------|
| `binary` | `np.ndarray` | Preprocessed image |
| `ocr` | `dict` | OCR text + word-level bounding boxes |
| `layout` | `list[dict]` | Detected layout regions (header, logos, tables) |
| `entities` | `dict` | Extracted invoice fields |
| `output_file` | `Path \| None` | Path to the exported file |

### Individual modules

```python
from invoice_processor.preprocessor import preprocess
from invoice_processor.ocr import run_ocr
from invoice_processor.layout_analyzer import analyse_layout
from invoice_processor.entity_extractor import extract_entities
from invoice_processor.exporter import export

binary = preprocess("invoice.png")
ocr    = run_ocr(binary)
layout = analyse_layout(binary)
data   = extract_entities(ocr["text"])
export(data, "invoice.json", fmt="json")
```

## Extracted Fields

| Field | Type | Example |
|-------|------|---------|
| `supplier` | `str \| None` | `"Acme Corp LLC"` |
| `invoice_number` | `str \| None` | `"INV-2024-042"` |
| `date` | `str \| None` (ISO 8601) | `"2024-03-05"` |
| `due_date` | `str \| None` (ISO 8601) | `"2024-04-04"` |
| `subtotal` | `float \| None` | `1000.0` |
| `tax` | `float \| None` | `160.0` |
| `total` | `float \| None` | `1160.0` |
| `currency` | `str \| None` | `"USD"` |
| `line_items` | `list[dict]` | `[{description, quantity, unit_price, total}]` |

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
invoice_processor/
├── __init__.py          # Package entry point (exports InvoicePipeline)
├── preprocessor.py      # Image cleaning with OpenCV
├── ocr.py               # Text/coordinate extraction via pytesseract
├── layout_analyzer.py   # Table, header, logo detection
├── entity_extractor.py  # Regex + heuristic KIE
├── exporter.py          # JSON / CSV / Excel export
└── pipeline.py          # Full end-to-end orchestrator
tests/
├── test_preprocessor.py
├── test_entity_extractor.py
├── test_exporter.py
└── test_layout_analyzer.py
main.py                  # CLI entry point
requirements.txt
```
