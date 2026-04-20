"""
Tests for invoice_processor.entity_extractor
"""

from __future__ import annotations

import pytest

from invoice_processor.entity_extractor import (
    _normalise_amount,
    _parse_date,
    extract_entities,
)


# ---------------------------------------------------------------------------
# _normalise_amount
# ---------------------------------------------------------------------------

class TestNormaliseAmount:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1234.56", 1234.56),
            ("1,234.56", 1234.56),
            ("1.234,56", 1234.56),
            ("1,000", 1000.0),
            ("0.99", 0.99),
            ("", None),
            ("abc", None),
        ],
    )
    def test_various_formats(self, raw, expected):
        result = _normalise_amount(raw)
        assert result == expected


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Fecha: 2024-01-31", "2024-01-31"),
            ("Fecha: 31/01/2024", "2024-01-31"),
            ("Fecha: 31.01.2024", "2024-01-31"),
            ("Fecha: 2024/01/31", "2024-01-31"),
            ("Fecha: 31 de enero de 2024", "2024-01-31"),
            ("Date: January 31, 2024", "2024-01-31"),
            ("No date here", None),
            ("invalid 99/99/9999 date", None),
        ],
    )
    def test_formats(self, text, expected):
        result = _parse_date(text)
        assert result == expected


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------

SAMPLE_INVOICE_ES = """
FACTURA
Proveedor: Tecnología Global S.A. de C.V.
Factura No.: INV-2024-001
Fecha: 15 de marzo de 2024
Fecha de vencimiento: 14/04/2024
Moneda: MXN

Descripción                  Cantidad   Precio    Total
Servicios de consultoría        1       5,000.00  5,000.00
Licencia de software            2       1,500.00  3,000.00

Subtotal:    8,000.00
IVA (16%):   1,280.00
Total:       9,280.00
"""

SAMPLE_INVOICE_EN = """
INVOICE
Supplier: Acme Corp LLC
Invoice Number: INV-2024-042
Date: March 5, 2024
Due Date: 04/04/2024
Currency: USD

1 Consulting services   500.00   500.00
2 Software license      250.00   500.00

Subtotal: 1,000.00
VAT: 160.00
Total: 1,160.00
"""


class TestExtractEntities:
    # --- Spanish invoice ---

    def test_supplier_es(self):
        result = extract_entities(SAMPLE_INVOICE_ES)
        assert result["supplier"] == "Tecnología Global S.A. de C.V."

    def test_invoice_number_es(self):
        result = extract_entities(SAMPLE_INVOICE_ES)
        assert result["invoice_number"] == "INV-2024-001"

    def test_date_es(self):
        result = extract_entities(SAMPLE_INVOICE_ES)
        assert result["date"] == "2024-03-15"

    def test_due_date_es(self):
        result = extract_entities(SAMPLE_INVOICE_ES)
        assert result["due_date"] == "2024-04-14"

    def test_currency_es(self):
        result = extract_entities(SAMPLE_INVOICE_ES)
        assert result["currency"] == "MXN"

    def test_subtotal_es(self):
        result = extract_entities(SAMPLE_INVOICE_ES)
        assert result["subtotal"] == 8000.0

    def test_tax_es(self):
        result = extract_entities(SAMPLE_INVOICE_ES)
        assert result["tax"] == 1280.0

    def test_total_es(self):
        result = extract_entities(SAMPLE_INVOICE_ES)
        assert result["total"] == 9280.0

    # --- English invoice ---

    def test_supplier_en(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert result["supplier"] == "Acme Corp LLC"

    def test_invoice_number_en(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert result["invoice_number"] == "INV-2024-042"

    def test_date_en(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert result["date"] == "2024-03-05"

    def test_due_date_en(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert result["due_date"] == "2024-04-04"

    def test_currency_en(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert result["currency"] == "USD"

    def test_subtotal_en(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert result["subtotal"] == 1000.0

    def test_tax_en(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert result["tax"] == 160.0

    def test_total_en(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert result["total"] == 1160.0

    # --- Edge cases ---

    def test_empty_text_returns_nones(self):
        result = extract_entities("")
        assert result["supplier"] is None
        assert result["invoice_number"] is None
        assert result["date"] is None
        assert result["total"] is None

    def test_result_has_all_expected_keys(self):
        result = extract_entities("")
        expected_keys = {
            "supplier", "invoice_number", "date", "due_date",
            "subtotal", "tax", "total", "currency", "line_items",
        }
        assert expected_keys == set(result.keys())

    def test_line_items_is_list(self):
        result = extract_entities(SAMPLE_INVOICE_EN)
        assert isinstance(result["line_items"], list)

    def test_fallback_total_from_largest_amount(self):
        text = "Total $5,000.00\nOther $100.00"
        result = extract_entities(text)
        assert result["total"] == 5000.0
