"""
Tests for invoice_processor.preprocessor
"""

from __future__ import annotations

import numpy as np
import pytest

from invoice_processor.preprocessor import (
    binarize,
    deskew,
    load_image,
    preprocess,
    remove_noise,
    scale_image,
    to_grayscale,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_image() -> np.ndarray:
    """A simple 100×100 white BGR image."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def gray_image() -> np.ndarray:
    """A simple 100×100 grayscale image."""
    return np.full((100, 100), 200, dtype=np.uint8)


@pytest.fixture
def simple_text_image() -> np.ndarray:
    """A grayscale image with a horizontal black line (simulates text)."""
    img = np.full((100, 200), 255, dtype=np.uint8)
    img[50, 20:180] = 0  # horizontal black line
    return img


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_returns_copy_of_ndarray(self, white_image):
        result = load_image(white_image)
        assert isinstance(result, np.ndarray)
        assert result is not white_image  # must be a copy

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nonexistent.png")

    def test_loads_real_file(self, tmp_path):
        import cv2
        path = tmp_path / "test.png"
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(path), img)
        loaded = load_image(path)
        assert loaded.shape == (50, 50, 3)


# ---------------------------------------------------------------------------
# to_grayscale
# ---------------------------------------------------------------------------

class TestToGrayscale:
    def test_bgr_to_gray(self, white_image):
        gray = to_grayscale(white_image)
        assert gray.ndim == 2
        assert gray.shape == (100, 100)

    def test_already_gray_is_unchanged_shape(self, gray_image):
        result = to_grayscale(gray_image)
        assert result.shape == gray_image.shape


# ---------------------------------------------------------------------------
# remove_noise
# ---------------------------------------------------------------------------

class TestRemoveNoise:
    def test_output_same_shape(self, gray_image):
        result = remove_noise(gray_image)
        assert result.shape == gray_image.shape

    def test_output_dtype(self, gray_image):
        result = remove_noise(gray_image)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# binarize
# ---------------------------------------------------------------------------

class TestBinarize:
    def test_output_only_0_and_255(self, simple_text_image):
        binary = binarize(simple_text_image)
        unique = set(np.unique(binary).tolist())
        assert unique.issubset({0, 255})

    def test_output_same_shape(self, simple_text_image):
        binary = binarize(simple_text_image)
        assert binary.shape == simple_text_image.shape


# ---------------------------------------------------------------------------
# scale_image
# ---------------------------------------------------------------------------

class TestScaleImage:
    def test_scale_increases_size(self, gray_image):
        scaled = scale_image(gray_image, target_dpi=300, current_dpi=72)
        assert scaled.shape[0] > gray_image.shape[0]
        assert scaled.shape[1] > gray_image.shape[1]

    def test_no_scale_when_equal_dpi(self, gray_image):
        result = scale_image(gray_image, target_dpi=150, current_dpi=150)
        assert result.shape == gray_image.shape

    def test_no_scale_when_target_less_than_current(self, gray_image):
        result = scale_image(gray_image, target_dpi=72, current_dpi=300)
        assert result.shape == gray_image.shape


# ---------------------------------------------------------------------------
# deskew
# ---------------------------------------------------------------------------

class TestDeskew:
    def test_returns_same_shape(self, simple_text_image):
        result = deskew(simple_text_image)
        assert result.shape == simple_text_image.shape

    def test_no_crash_on_blank_image(self):
        blank = np.full((100, 200), 255, dtype=np.uint8)
        result = deskew(blank)
        assert result.shape == blank.shape


# ---------------------------------------------------------------------------
# preprocess (integration)
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_returns_binary_array(self, white_image):
        result = preprocess(white_image, do_deskew=False)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2

    def test_output_only_0_and_255(self, white_image):
        result = preprocess(white_image, do_deskew=False)
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255})

    def test_preprocess_with_deskew(self, simple_text_image):
        result = preprocess(simple_text_image, do_deskew=True)
        assert result.ndim == 2

    def test_preprocess_file(self, tmp_path):
        import cv2
        path = tmp_path / "invoice.png"
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(path), img)
        result = preprocess(path, do_deskew=False)
        assert result.ndim == 2
