"""
Layout Analyser module.

Identifies structural regions in an invoice image:

* **Tables**   – detected via contour / line analysis.
* **Headers**  – top portion of the page that typically contains the
                 company name, logo and invoice number.
* **Logos**    – compact high-contrast rectangular blobs in the header
                 zone that are likely to be image-based logos.

All regions are returned as dictionaries::

    {
        "type":   "table" | "header" | "logo",
        "x":      int,   # left edge
        "y":      int,   # top edge
        "width":  int,
        "height": int,
    }
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


RegionDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rect(region_type: str, x: int, y: int, w: int, h: int) -> RegionDict:
    return {"type": region_type, "x": x, "y": y, "width": w, "height": h}


def _detect_horizontal_lines(binary: np.ndarray, min_width_ratio: float = 0.4) -> np.ndarray:
    """Return a mask containing only long horizontal line segments."""
    h, w = binary.shape[:2]
    min_len = int(w * min_width_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    inverted = cv2.bitwise_not(binary)
    return cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)


def _detect_vertical_lines(binary: np.ndarray, min_height_ratio: float = 0.05) -> np.ndarray:
    """Return a mask containing only vertical line segments."""
    h, w = binary.shape[:2]
    min_len = int(h * min_height_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    inverted = cv2.bitwise_not(binary)
    return cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_tables(binary: np.ndarray) -> list[RegionDict]:
    """
    Detect table regions by finding areas with intersecting horizontal and
    vertical line segments.

    Parameters
    ----------
    binary:
        Binary (thresholded) grayscale image.

    Returns
    -------
    List of region dicts with ``type == "table"``.
    """
    h_mask = _detect_horizontal_lines(binary)
    v_mask = _detect_vertical_lines(binary)

    # Combine both masks – their intersection indicates a table grid
    grid_mask = cv2.add(h_mask, v_mask)

    # Dilate to merge nearby segments into a single table block
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(grid_mask, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables: list[RegionDict] = []
    img_area = binary.shape[0] * binary.shape[1]
    for cnt in contours:
        x, y, w, wb = cv2.boundingRect(cnt)
        area = w * wb
        # Filter out very small spurious regions (< 1 % of image)
        if area / img_area < 0.01:
            continue
        tables.append(_rect("table", x, y, w, wb))

    return tables


def detect_header(binary: np.ndarray, header_ratio: float = 0.25) -> list[RegionDict]:
    """
    Mark the top *header_ratio* fraction of the page as the header region.

    Parameters
    ----------
    binary:
        Binary grayscale image.
    header_ratio:
        Fraction of page height to treat as the header (default 25 %).

    Returns
    -------
    List with a single region dict of type ``"header"``.
    """
    h, w = binary.shape[:2]
    header_height = int(h * header_ratio)
    return [_rect("header", 0, 0, w, header_height)]


def detect_logos(binary: np.ndarray, header_ratio: float = 0.25) -> list[RegionDict]:
    """
    Detect logo candidates in the header zone.

    A logo is assumed to be a compact rectangular region with significant
    non-white content in the upper portion of the page.

    Returns
    -------
    List of region dicts with ``type == "logo"``.
    """
    h, w = binary.shape[:2]
    header_h = int(h * header_ratio)
    header_zone = binary[:header_h, :]

    # Invert so non-white pixels are foreground
    inverted = cv2.bitwise_not(header_zone)

    # Morphological close to merge nearby ink blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logos: list[RegionDict] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch else 0
        area = cw * ch
        # Logo heuristic: square-ish, at least 30×30 px, not too wide
        if area > 900 and 0.3 < aspect < 5.0 and cw < w * 0.5:
            logos.append(_rect("logo", x, y, cw, ch))

    return logos


def analyse_layout(binary: np.ndarray) -> list[RegionDict]:
    """
    Run full layout analysis and return all detected regions.

    Parameters
    ----------
    binary:
        Binary (thresholded) grayscale image (output of the preprocessor).

    Returns
    -------
    Combined list of header, logo, and table region dicts.
    """
    regions: list[RegionDict] = []
    regions.extend(detect_header(binary))
    regions.extend(detect_logos(binary))
    regions.extend(detect_tables(binary))
    return regions
