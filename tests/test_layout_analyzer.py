"""
Tests for invoice_processor.layout_analyzer
"""

from __future__ import annotations

import numpy as np
import pytest

from invoice_processor.layout_analyzer import (
    analyse_layout,
    detect_header,
    detect_logos,
    detect_tables,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blank_binary() -> np.ndarray:
    """All-white binary image (no content)."""
    return np.full((400, 600), 255, dtype=np.uint8)


@pytest.fixture
def table_image() -> np.ndarray:
    """
    A binary image with a simple grid of horizontal and vertical lines
    that should be detected as a table.
    """
    img = np.full((400, 600), 255, dtype=np.uint8)
    # Horizontal lines
    for y in (100, 150, 200, 250):
        img[y, 50:550] = 0
    # Vertical lines
    for x in (50, 200, 350, 550):
        img[100:250, x] = 0
    return img


@pytest.fixture
def logo_image() -> np.ndarray:
    """A binary image with a dark rectangle in the upper-left (logo candidate)."""
    img = np.full((400, 600), 255, dtype=np.uint8)
    img[20:80, 20:120] = 0  # compact dark blob in header zone
    return img


# ---------------------------------------------------------------------------
# detect_header
# ---------------------------------------------------------------------------

class TestDetectHeader:
    def test_returns_one_region(self, blank_binary):
        regions = detect_header(blank_binary)
        assert len(regions) == 1

    def test_region_type(self, blank_binary):
        regions = detect_header(blank_binary)
        assert regions[0]["type"] == "header"

    def test_region_starts_at_top(self, blank_binary):
        regions = detect_header(blank_binary)
        assert regions[0]["x"] == 0
        assert regions[0]["y"] == 0

    def test_region_spans_full_width(self, blank_binary):
        regions = detect_header(blank_binary)
        assert regions[0]["width"] == blank_binary.shape[1]

    def test_header_height_is_ratio(self, blank_binary):
        ratio = 0.25
        regions = detect_header(blank_binary, header_ratio=ratio)
        expected_h = int(blank_binary.shape[0] * ratio)
        assert regions[0]["height"] == expected_h


# ---------------------------------------------------------------------------
# detect_tables
# ---------------------------------------------------------------------------

class TestDetectTables:
    def test_no_tables_on_blank(self, blank_binary):
        tables = detect_tables(blank_binary)
        assert tables == []

    def test_detects_table(self, table_image):
        tables = detect_tables(table_image)
        assert len(tables) >= 1

    def test_table_type(self, table_image):
        tables = detect_tables(table_image)
        for t in tables:
            assert t["type"] == "table"

    def test_table_has_required_keys(self, table_image):
        tables = detect_tables(table_image)
        for t in tables:
            assert {"type", "x", "y", "width", "height"}.issubset(t.keys())


# ---------------------------------------------------------------------------
# detect_logos
# ---------------------------------------------------------------------------

class TestDetectLogos:
    def test_no_logos_on_blank(self, blank_binary):
        logos = detect_logos(blank_binary)
        assert logos == []

    def test_detects_logo(self, logo_image):
        logos = detect_logos(logo_image)
        assert len(logos) >= 1

    def test_logo_type(self, logo_image):
        logos = detect_logos(logo_image)
        for lg in logos:
            assert lg["type"] == "logo"


# ---------------------------------------------------------------------------
# analyse_layout (integration)
# ---------------------------------------------------------------------------

class TestAnalyseLayout:
    def test_always_has_header(self, blank_binary):
        regions = analyse_layout(blank_binary)
        types = [r["type"] for r in regions]
        assert "header" in types

    def test_table_detected_in_table_image(self, table_image):
        regions = analyse_layout(table_image)
        types = [r["type"] for r in regions]
        assert "table" in types

    def test_logo_detected_in_logo_image(self, logo_image):
        regions = analyse_layout(logo_image)
        types = [r["type"] for r in regions]
        assert "logo" in types

    def test_all_regions_have_required_keys(self, table_image):
        regions = analyse_layout(table_image)
        for r in regions:
            assert {"type", "x", "y", "width", "height"}.issubset(r.keys())
