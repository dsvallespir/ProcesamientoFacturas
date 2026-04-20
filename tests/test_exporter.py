"""
Tests for invoice_processor.exporter
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from invoice_processor.exporter import export, to_csv, to_json


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_DATA = {
    "supplier": "Acme Corp",
    "invoice_number": "INV-001",
    "date": "2024-01-15",
    "due_date": "2024-02-15",
    "subtotal": 1000.0,
    "tax": 160.0,
    "total": 1160.0,
    "currency": "MXN",
    "line_items": [
        {"description": "Widget A", "quantity": 2.0, "unit_price": 300.0, "total": 600.0},
        {"description": "Widget B", "quantity": 2.0, "unit_price": 200.0, "total": 400.0},
    ],
}

EMPTY_DATA = {
    "supplier": None,
    "invoice_number": None,
    "date": None,
    "due_date": None,
    "subtotal": None,
    "tax": None,
    "total": None,
    "currency": None,
    "line_items": [],
}


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

class TestToJson:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "invoice.json"
        result = to_json(SAMPLE_DATA, out)
        assert result.exists()

    def test_content_is_valid_json(self, tmp_path):
        out = tmp_path / "invoice.json"
        to_json(SAMPLE_DATA, out)
        with out.open() as fh:
            data = json.load(fh)
        assert data["supplier"] == "Acme Corp"
        assert data["total"] == 1160.0
        assert len(data["line_items"]) == 2

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "invoice.json"
        to_json(SAMPLE_DATA, out)
        assert out.exists()

    def test_returns_resolved_path(self, tmp_path):
        out = tmp_path / "invoice.json"
        result = to_json(SAMPLE_DATA, out)
        assert isinstance(result, Path)

    def test_empty_data(self, tmp_path):
        out = tmp_path / "empty.json"
        to_json(EMPTY_DATA, out)
        with out.open() as fh:
            data = json.load(fh)
        assert data["supplier"] is None


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------

class TestToCsv:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "invoice.csv"
        result = to_csv(SAMPLE_DATA, out)
        assert result.exists()

    def test_header_fields_present(self, tmp_path):
        out = tmp_path / "invoice.csv"
        to_csv(SAMPLE_DATA, out)
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        # First row is the column header
        assert rows[0] == ["field", "value"]
        # Check some values are present
        fields_in_csv = {row[0]: row[1] for row in rows[1:] if len(row) == 2}
        assert fields_in_csv.get("supplier") == "Acme Corp"
        assert fields_in_csv.get("total") == "1160.0"

    def test_line_items_section(self, tmp_path):
        out = tmp_path / "invoice.csv"
        to_csv(SAMPLE_DATA, out)
        content = out.read_text(encoding="utf-8")
        assert "description" in content
        assert "Widget A" in content
        assert "Widget B" in content

    def test_empty_line_items(self, tmp_path):
        out = tmp_path / "invoice.csv"
        to_csv(EMPTY_DATA, out)
        content = out.read_text(encoding="utf-8")
        # No item section when line_items is empty
        assert "Widget" not in content


# ---------------------------------------------------------------------------
# to_excel
# ---------------------------------------------------------------------------

class TestToExcel:
    def test_creates_file(self, tmp_path):
        pytest.importorskip("pandas")
        from invoice_processor.exporter import to_excel
        out = tmp_path / "invoice.xlsx"
        result = to_excel(SAMPLE_DATA, out)
        assert result.exists()

    def test_content_readable(self, tmp_path):
        pd = pytest.importorskip("pandas")
        from invoice_processor.exporter import to_excel
        out = tmp_path / "invoice.xlsx"
        to_excel(SAMPLE_DATA, out)

        summary = pd.read_excel(out, sheet_name="Summary")
        assert "Field" in summary.columns
        assert "Value" in summary.columns
        fields = summary["Field"].tolist()
        assert "supplier" in fields

        items = pd.read_excel(out, sheet_name="Line Items")
        assert len(items) == 2


# ---------------------------------------------------------------------------
# export (dispatch)
# ---------------------------------------------------------------------------

class TestExport:
    def test_dispatch_json(self, tmp_path):
        out = tmp_path / "inv.json"
        result = export(SAMPLE_DATA, out, fmt="json")
        assert result.exists()

    def test_dispatch_csv(self, tmp_path):
        out = tmp_path / "inv.csv"
        result = export(SAMPLE_DATA, out, fmt="csv")
        assert result.exists()

    def test_dispatch_excel(self, tmp_path):
        pytest.importorskip("pandas")
        out = tmp_path / "inv.xlsx"
        result = export(SAMPLE_DATA, out, fmt="excel")
        assert result.exists()

    def test_invalid_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported export format"):
            export(SAMPLE_DATA, tmp_path / "inv.pdf", fmt="pdf")
