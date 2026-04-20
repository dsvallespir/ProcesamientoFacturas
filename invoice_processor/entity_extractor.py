"""
Entity Extractor module (Key Information Extraction – KIE).

Extracts structured invoice fields from the raw OCR text:

* **supplier**       – company / vendor name
* **invoice_number** – invoice / folio identifier
* **date**           – invoice issue date
* **due_date**       – payment due date
* **subtotal**       – amount before taxes
* **tax**            – tax amount (IVA / VAT)
* **total**          – total amount payable
* **currency**       – currency code (MXN, USD, EUR …)
* **line_items**     – individual product / service rows

All amounts are returned as ``float`` (or ``None`` if not found).
Dates are returned as ISO-8601 strings (``YYYY-MM-DD``) or ``None``.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# --- Dates ---
# Matches: 01/01/2024, 01-01-2024, 01.01.2024,
#          2024/01/01, 2024-01-01, 2024.01.01,
#          31 de enero de 2024, January 31, 2024
_MONTHS_ES = (
    r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    r"septiembre|octubre|noviembre|diciembre"
)
_MONTHS_EN = (
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december"
)
_MONTH_NAMES = f"(?:{_MONTHS_ES}|{_MONTHS_EN})"

_DATE_PATTERNS: list[re.Pattern[str]] = [
    # ISO-like: 2024-01-31 or 2024/01/31
    re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b"),
    # European: 31/01/2024 or 31-01-2024 (only when groups 0 and 1 are purely numeric)
    re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b"),
    # Spanish long: "31 de enero de 2024"
    re.compile(
        rf"\b(\d{{1,2}})\s+de\s+({_MONTHS_ES})\s+de\s+(\d{{4}})\b",
        re.IGNORECASE,
    ),
    # English long: "January 31, 2024"
    re.compile(
        rf"\b({_MONTHS_EN})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
        re.IGNORECASE,
    ),
]

_MONTH_MAP: dict[str, int] = {
    "enero": 1, "january": 1,
    "febrero": 2, "february": 2,
    "marzo": 3, "march": 3,
    "abril": 4, "april": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "june": 6,
    "julio": 7, "july": 7,
    "agosto": 8, "august": 8,
    "septiembre": 9, "september": 9,
    "octubre": 10, "october": 10,
    "noviembre": 11, "november": 11,
    "diciembre": 12, "december": 12,
}

# --- Amounts ---
# Matches: $1,234.56  1.234,56  1234.56  1,234
_AMOUNT_RE = re.compile(
    r"(?:[\$€£])\s*([\d]{1,3}(?:[,.][\d]{3})*(?:[.,]\d{1,2})?)"
    r"|"
    r"\b([\d]{1,3}(?:[,.][\d]{3})*(?:[.,]\d{1,2}))\b",
)

# --- Invoice number ---
# Require the keyword and value to appear on the same line (no newlines in between).
# Put "number" before the short "n[oº°]?.?" to avoid partial matching.
_INV_NUM_RE = re.compile(
    r"(?:"
    r"factura\s+(?:n[uú]mero\b|n[oº°]\.?)"     # Factura Número / Factura No.
    r"|invoice\s+(?:number\b|n[oº°]?\.?\b)"     # Invoice Number / Invoice No.
    r"|folio"
    r"|n[uú]mero\s+(?:de\s+)?factura"
    r"|#"
    r")"
    r"[^\S\n]*[:\-]?[^\S\n]*"
    r"([A-Z0-9][A-Z0-9\-/]{2,19})",
    re.IGNORECASE,
)

# --- Currency ---
_CURRENCY_RE = re.compile(r"\b(MXN|USD|EUR|GBP|CAD|JPY)\b", re.IGNORECASE)

# --- Label-value patterns for totals ---
# Allows optional parenthetical qualifier between label and colon, e.g. "IVA (16%):".
# Uses [^\S\n]* instead of \s* to prevent cross-line matches (e.g. table column header
# "Total" grabbing the first price on the next line).
# Also uses [ \t\w]* (no newline) in the label suffix for the same reason.
_LABEL_AMOUNT_RE = re.compile(
    r"(?P<label>subtotal|sub[\s\-]?total|iva|i\.v\.a\.|vat|tax|"
    r"impuesto|total[ \t\w]*|importe[ \t\w]*|monto[ \t\w]*)"
    r"[^\S\n]*(?:\([^)]*\))?[^\S\n]*"  # optional "(16%)" qualifier, same line
    r"[:\-]?[^\S\n]*"
    r"(?:[\$€£])?[^\S\n]*"
    r"(?P<amount>[\d]{1,3}(?:[,.][\d]{3})*(?:[.,]\d{1,2})?)",
    re.IGNORECASE,
)

# --- Supplier keywords (lines that likely contain the company name) ---
_SUPPLIER_LABELS = re.compile(
    r"(?:proveedor|vendedor|emisor|empresa|compañ[ií]a|supplier|vendor|"
    r"sold[\s\-]?by|from|razón[\s\-]?social)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)

# --- Due date keywords ---
_DUE_DATE_LABELS = re.compile(
    r"(?:fecha[\s\w]*vencimiento|fecha[\s\w]*pago|due[\s\-]?date|payment[\s\w]*date"
    r"|vencimiento|plazo)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)

# --- Invoice date keywords ---
# Match lines whose label specifically refers to the emission/issue date,
# NOT to due-date / payment-date lines.
_INVOICE_DATE_LABELS = re.compile(
    r"^[ \t]*(?:fecha\s*(?:de\s*)?(?:factura|emis[ií][oó]n|expedici[oó]n)|"
    r"fecha\s*:|date\s*:|issued|emitida?)\s*[:\-]?\s*(.+)",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalise_amount(raw: str) -> float | None:
    """Convert a raw amount string to float, handling comma/dot separators."""
    raw = raw.strip()
    if not raw:
        return None
    # Detect European format (1.234,56) vs US format (1,234.56)
    if re.search(r"\d,\d{2}$", raw):
        # European: last separator is comma → decimal
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_date(text: str) -> str | None:
    """
    Extract the first parseable date from *text* and return it as
    ``YYYY-MM-DD``, or ``None`` if no date is found.
    """
    for pattern in _DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        groups = m.groups()
        try:
            # ISO-like: group0=year, group1=month, group2=day
            if re.match(r"^\d{4}$", groups[0]):
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
            # Spanish text: group0=day, group1=month_name, group2=year
            elif groups[1].lower() in _MONTH_MAP:
                day = int(groups[0])
                month = _MONTH_MAP[groups[1].lower()]
                year = int(groups[2])
            # English text: group0=month_name, group1=day, group2=year
            elif groups[0].lower() in _MONTH_MAP:
                month = _MONTH_MAP[groups[0].lower()]
                day = int(groups[1])
                year = int(groups[2])
            # European numeric: group0=day, group1=month, group2=year
            elif re.match(r"^\d{4}$", groups[2]) and re.match(r"^\d{1,2}$", groups[1]):
                day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
            else:
                continue
            if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                return f"{year:04d}-{month:02d}-{day:02d}"
        except (ValueError, IndexError):
            continue
    return None


def _find_amounts_by_label(text: str) -> dict[str, float | None]:
    """Return a dict of {label_key: amount} from labelled lines."""
    results: dict[str, float | None] = {}
    for m in _LABEL_AMOUNT_RE.finditer(text):
        label = m.group("label").strip().lower()
        amount = _normalise_amount(m.group("amount"))
        if "subtotal" in label or "sub total" in label or "sub-total" in label:
            results.setdefault("subtotal", amount)
        elif re.search(r"iva|vat|tax|impuesto", label):
            results.setdefault("tax", amount)
        elif re.search(r"total|importe|monto", label):
            results.setdefault("total", amount)
    return results


def _extract_line_items(text: str) -> list[dict[str, Any]]:
    """
    Heuristically extract product / service rows from the OCR text.

    A line item is identified as a line that contains:
      - at least one word (description)
      - a numeric quantity
      - a unit price and/or total price

    Returns a list of dicts with keys: description, quantity, unit_price, total.
    """
    items: list[dict[str, Any]] = []
    lines = text.splitlines()

    # Pattern: optional quantity, description words, price(s)
    item_re = re.compile(
        r"^(?P<qty>\d+(?:[.,]\d+)?)\s+"          # quantity (optional but common first)
        r"(?P<desc>.+?)\s+"
        r"(?P<unit>[\d]{1,3}(?:[,.][\d]{3})*(?:[.,]\d{1,2})?)\s*"  # unit price
        r"(?P<total>[\d]{1,3}(?:[,.][\d]{3})*(?:[.,]\d{1,2}))?",   # line total
        re.IGNORECASE,
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = item_re.match(line)
        if m:
            items.append(
                {
                    "description": m.group("desc").strip(),
                    "quantity": _normalise_amount(m.group("qty")),
                    "unit_price": _normalise_amount(m.group("unit")),
                    "total": _normalise_amount(m.group("total")) if m.group("total") else None,
                }
            )
    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> dict[str, Any]:
    """
    Extract key invoice entities from raw OCR text.

    Parameters
    ----------
    text:
        Full plain-text output from the OCR engine.

    Returns
    -------
    Dictionary with the following keys (any may be ``None`` if not found):

    * ``supplier``       – vendor / company name (str)
    * ``invoice_number`` – invoice identifier (str)
    * ``date``           – issue date (``YYYY-MM-DD``)
    * ``due_date``       – payment due date (``YYYY-MM-DD``)
    * ``subtotal``       – pre-tax amount (float)
    * ``tax``            – tax / IVA amount (float)
    * ``total``          – total payable amount (float)
    * ``currency``       – currency code (str)
    * ``line_items``     – list of line-item dicts
    """
    result: dict[str, Any] = {
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

    # --- Supplier ---
    m = _SUPPLIER_LABELS.search(text)
    if m:
        result["supplier"] = m.group(1).strip()

    # --- Invoice number ---
    m = _INV_NUM_RE.search(text)
    if m:
        result["invoice_number"] = m.group(1).strip()

    # --- Currency ---
    m = _CURRENCY_RE.search(text)
    if m:
        result["currency"] = m.group(1).upper()

    # --- Dates ---
    # Try labelled date first, then generic
    due_m = _DUE_DATE_LABELS.search(text)
    if due_m:
        result["due_date"] = _parse_date(due_m.group(1))

    date_m = _INVOICE_DATE_LABELS.search(text)
    if date_m:
        candidate = _parse_date(date_m.group(1))
        if candidate and candidate != result["due_date"]:
            result["date"] = candidate

    # Fallback: grab first date anywhere in text
    if result["date"] is None:
        result["date"] = _parse_date(text)

    # --- Amounts ---
    amounts = _find_amounts_by_label(text)
    result["subtotal"] = amounts.get("subtotal")
    result["tax"] = amounts.get("tax")
    result["total"] = amounts.get("total")

    # If only total is present but not subtotal/tax, try to derive
    if result["total"] is None:
        # Fallback: pick the largest monetary amount in the document
        all_amounts = [
            _normalise_amount(m.group(1) or m.group(2))
            for m in _AMOUNT_RE.finditer(text)
        ]
        all_amounts = [a for a in all_amounts if a is not None]
        if all_amounts:
            result["total"] = max(all_amounts)

    # --- Line items ---
    result["line_items"] = _extract_line_items(text)

    return result
