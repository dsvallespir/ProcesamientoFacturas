"""
Pipeline module.

Orchestrates the full invoice-processing workflow:

    1. Preprocess  – clean and normalise the image (OpenCV)
    2. OCR         – extract text + word coordinates (Tesseract)
    3. Layout      – identify tables, headers, logos
    4. Extract     – key information extraction (supplier, date, totals …)
    5. Export      – serialise to JSON / CSV / Excel

Usage
-----
::

    from invoice_processor import InvoicePipeline

    pipe = InvoicePipeline()
    result = pipe.process("invoice.png", output_dir="output/", fmt="json")
    print(result["entities"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .entity_extractor import extract_entities
from .exporter import export
from .layout_analyzer import analyse_layout
from .ocr import run_ocr
from .preprocessor import preprocess


class InvoicePipeline:
    """
    End-to-end invoice processing pipeline.

    Parameters
    ----------
    lang:
        Tesseract language string (default: ``"spa+eng"``).
    do_deskew:
        Apply skew correction during preprocessing (default: ``True``).
    do_scale:
        Up-scale images to 300 DPI before OCR (default: ``False``).
    current_dpi:
        Assumed DPI of the source image – used when *do_scale* is ``True``.
    """

    def __init__(
        self,
        lang: str = "spa+eng",
        *,
        do_deskew: bool = True,
        do_scale: bool = False,
        current_dpi: int = 72,
    ) -> None:
        self.lang = lang
        self.do_deskew = do_deskew
        self.do_scale = do_scale
        self.current_dpi = current_dpi

    # ------------------------------------------------------------------
    # Step runners (can be overridden or called individually)
    # ------------------------------------------------------------------

    def run_preprocessing(self, source: Any) -> np.ndarray:
        """Return the preprocessed binary image."""
        return preprocess(
            source,
            do_deskew=self.do_deskew,
            do_scale=self.do_scale,
            current_dpi=self.current_dpi,
        )

    def run_ocr(self, binary: np.ndarray) -> dict[str, Any]:
        """Return OCR results (text + words with coordinates)."""
        return run_ocr(binary, lang=self.lang)

    def run_layout_analysis(self, binary: np.ndarray) -> list[dict[str, Any]]:
        """Return detected layout regions (header, logo, tables)."""
        return analyse_layout(binary)

    def run_entity_extraction(self, text: str) -> dict[str, Any]:
        """Return extracted invoice entities."""
        return extract_entities(text)

    def run_export(
        self,
        entities: dict[str, Any],
        output_path: str | Path,
        fmt: str,
    ) -> Path:
        """Serialise entities to the requested format and return the output path."""
        return export(entities, output_path, fmt=fmt)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        source: Any,
        output_dir: str | Path | None = None,
        fmt: str = "json",
        stem: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full pipeline on a single invoice image.

        Parameters
        ----------
        source:
            File path (str / Path) or numpy BGR array of the invoice image.
        output_dir:
            Directory where the export file will be written.  If ``None``
            the file is written to the same directory as *source* (or the
            current working directory for array inputs).
        fmt:
            Export format: ``"json"`` (default), ``"csv"``, or ``"excel"``/
            ``"xlsx"``.
        stem:
            Base name (without extension) for the output file.  Derived
            from the source filename when not supplied.

        Returns
        -------
        Dictionary with the following keys:

        * ``binary``       – preprocessed image array
        * ``ocr``          – OCR output dict (text, words)
        * ``layout``       – list of detected region dicts
        * ``entities``     – extracted invoice entity dict
        * ``output_file``  – :class:`~pathlib.Path` of the exported file
                             (or ``None`` if *output_dir* is ``None`` and
                             *source* is a numpy array with no path context)
        """
        # ---- 1. Preprocess ----
        binary = self.run_preprocessing(source)

        # ---- 2. OCR ----
        ocr_result = self.run_ocr(binary)

        # ---- 3. Layout analysis ----
        layout_regions = self.run_layout_analysis(binary)

        # ---- 4. Entity extraction ----
        entities = self.run_entity_extraction(ocr_result["text"])

        # ---- 5. Export ----
        output_file: Path | None = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            if stem is None:
                if isinstance(source, (str, Path)):
                    stem = Path(source).stem
                else:
                    stem = "invoice"
            ext = "xlsx" if fmt in ("excel", "xlsx") else fmt
            output_path = output_dir / f"{stem}.{ext}"
            output_file = self.run_export(entities, output_path, fmt=fmt)

        return {
            "binary": binary,
            "ocr": ocr_result,
            "layout": layout_regions,
            "entities": entities,
            "output_file": output_file,
        }
