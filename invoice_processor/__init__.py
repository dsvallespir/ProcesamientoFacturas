"""
Invoice Processor – automatic invoice data extraction pipeline.

Modules:
    preprocessor     – image cleaning (deskew, denoising, binarisation)
    ocr              – text + bounding-box extraction (pytesseract)
    layout_analyzer  – table / header / logo region detection
    entity_extractor – key-information extraction (supplier, date, total …)
    exporter         – export results to JSON / CSV / Excel
    pipeline         – orchestrates the full end-to-end flow
"""

from .pipeline import InvoicePipeline

__all__ = ["InvoicePipeline"]
__version__ = "0.1.0"
