"""
Preprocessor module.

Cleans and normalises invoice images before OCR:
    - Converts to grayscale
    - Deskews / corrects rotation
    - Reduces noise (Gaussian blur + morphological operations)
    - Binarises with Otsu's threshold
    - Optionally scales the image to improve OCR accuracy
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import cv2
import numpy as np


ImageLike = Union[str, Path, np.ndarray]


def load_image(source: ImageLike) -> np.ndarray:
    """Load an image from a file path or return the array unchanged."""
    if isinstance(source, np.ndarray):
        return source.copy()
    img = cv2.imread(str(source))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale (no-op if already single-channel)."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(gray: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur and morphological closing to reduce noise."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Binarise with Otsu's threshold for maximum contrast."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Correct skew angle using the Hough line-based method.

    Returns the corrected (rotated) grayscale image.  If no dominant
    skew is detected the original array is returned unchanged.
    """
    # Invert so text pixels are white on black background
    inverted = cv2.bitwise_not(gray)
    # Detect edges
    edges = cv2.Canny(inverted, 50, 150, apertureSize=3)
    # Probabilistic Hough transform to find line segments
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:  # avoid division by zero
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (±45°)
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:  # already straight enough
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def scale_image(image: np.ndarray, target_dpi: int = 300, current_dpi: int = 72) -> np.ndarray:
    """
    Up-scale a low-resolution image to *target_dpi* equivalent.

    Typical screen scans are 72–96 DPI; Tesseract performs best at 300+ DPI.
    """
    scale = target_dpi / current_dpi
    if scale <= 1.0:
        return image
    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def preprocess(
    source: ImageLike,
    *,
    do_deskew: bool = True,
    do_scale: bool = False,
    target_dpi: int = 300,
    current_dpi: int = 72,
) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    source:
        File path or BGR numpy array.
    do_deskew:
        Whether to attempt skew correction.
    do_scale:
        Whether to up-scale the image before OCR.
    target_dpi / current_dpi:
        Used only when *do_scale* is True.

    Returns
    -------
    Preprocessed single-channel (grayscale) binary image ready for OCR.
    """
    image = load_image(source)
    gray = to_grayscale(image)
    if do_deskew:
        gray = deskew(gray)
    denoised = remove_noise(gray)
    binary = binarize(denoised)
    if do_scale:
        binary = scale_image(binary, target_dpi=target_dpi, current_dpi=current_dpi)
    return binary
