"""
OCR module.

Extracts text and word-level bounding boxes from preprocessed invoice images
using Tesseract (via *pytesseract*).

Each word is returned as a dictionary::

    {
        "text": "Total",
        "x": 120,
        "y": 340,
        "width": 45,
        "height": 15,
        "confidence": 92.4,
        "page": 1,
        "block": 2,
        "paragraph": 1,
        "line": 3,
        "word": 1,
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TESSERACT_AVAILABLE = False


ImageLike = Union[str, Path, np.ndarray]


def _check_tesseract() -> None:
    if not _TESSERACT_AVAILABLE:
        raise ImportError(
            "pytesseract is not installed. "
            "Run `pip install pytesseract` and ensure Tesseract-OCR is available on PATH."
        )


def extract_text(image: np.ndarray, lang: str = "spa+eng") -> str:
    """
    Return the full plain-text content of *image*.

    Parameters
    ----------
    image:
        Preprocessed (grayscale or binary) numpy array.
    lang:
        Tesseract language string.  Defaults to Spanish + English.
    """
    _check_tesseract()
    return pytesseract.image_to_string(image, lang=lang, config="--psm 6")


def extract_words(image: np.ndarray, lang: str = "spa+eng") -> list[dict[str, Any]]:
    """
    Return a list of word-level records including bounding-box coordinates.

    Each record contains keys: text, x, y, width, height, confidence,
    page, block, paragraph, line, word.
    """
    _check_tesseract()
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )

    words: list[dict[str, Any]] = []
    n = len(data["text"])
    for i in range(n):
        word_text = str(data["text"][i]).strip()
        if not word_text:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf < 0:
            continue  # Tesseract marks non-word rows as -1

        words.append(
            {
                "text": word_text,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
                "confidence": round(conf, 2),
                "page": int(data["page_num"][i]),
                "block": int(data["block_num"][i]),
                "paragraph": int(data["par_num"][i]),
                "line": int(data["line_num"][i]),
                "word": int(data["word_num"][i]),
            }
        )
    return words


def run_ocr(
    image: np.ndarray,
    lang: str = "spa+eng",
) -> dict[str, Any]:
    """
    Run OCR and return both the plain text and the structured word list.

    Returns
    -------
    dict with keys:
        ``text``  – full document string
        ``words`` – list of word records (see :func:`extract_words`)
    """
    text = extract_text(image, lang=lang)
    words = extract_words(image, lang=lang)
    return {"text": text, "words": words}
