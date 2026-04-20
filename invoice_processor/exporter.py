"""
Exporter module.

Serialises extracted invoice data to JSON, CSV and Excel formats.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def to_json(data: dict[str, Any], output_path: str | Path, indent: int = 2) -> Path:
    """
    Write *data* to a JSON file.

    Parameters
    ----------
    data:
        Extracted invoice entities dict.
    output_path:
        Destination file path (.json).
    indent:
        JSON indentation level.

    Returns
    -------
    Resolved :class:`~pathlib.Path` of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent, default=str)
    return output_path.resolve()


def to_csv(data: dict[str, Any], output_path: str | Path) -> Path:
    """
    Write the scalar invoice fields (header) and line items to a CSV file.

    Two sections are written:
    1. Header rows – one row per scalar field (field, value).
    2. Line items  – one row per item with columns: description, quantity,
                     unit_price, total.

    Parameters
    ----------
    data:
        Extracted invoice entities dict.
    output_path:
        Destination file path (.csv).

    Returns
    -------
    Resolved :class:`~pathlib.Path` of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scalar_fields = [k for k in data if k != "line_items"]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)

        # Header section
        writer.writerow(["field", "value"])
        for field in scalar_fields:
            writer.writerow([field, data.get(field, "")])

        # Line items section
        line_items: list[dict[str, Any]] = data.get("line_items", [])
        if line_items:
            writer.writerow([])  # blank separator
            writer.writerow(["description", "quantity", "unit_price", "total"])
            for item in line_items:
                writer.writerow(
                    [
                        item.get("description", ""),
                        item.get("quantity", ""),
                        item.get("unit_price", ""),
                        item.get("total", ""),
                    ]
                )

    return output_path.resolve()


def to_excel(data: dict[str, Any], output_path: str | Path) -> Path:
    """
    Write the invoice data to an Excel workbook (.xlsx).

    Two worksheets are created:
    * **Summary** – scalar fields.
    * **Line Items** – product / service rows.

    Requires *openpyxl* (installed as part of pandas' optional dependencies).

    Parameters
    ----------
    data:
        Extracted invoice entities dict.
    output_path:
        Destination file path (.xlsx).

    Returns
    -------
    Resolved :class:`~pathlib.Path` of the written file.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pandas is required for Excel export. Install it with `pip install pandas openpyxl`."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scalar_fields = {k: v for k, v in data.items() if k != "line_items"}
    summary_df = pd.DataFrame(
        list(scalar_fields.items()), columns=["Field", "Value"]
    )

    line_items: list[dict[str, Any]] = data.get("line_items", [])
    if line_items:
        items_df = pd.DataFrame(line_items)
    else:
        items_df = pd.DataFrame(columns=["description", "quantity", "unit_price", "total"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        items_df.to_excel(writer, sheet_name="Line Items", index=False)

    return output_path.resolve()


def export(
    data: dict[str, Any],
    output_path: str | Path,
    fmt: str = "json",
) -> Path:
    """
    Convenience wrapper – dispatch to the appropriate export function.

    Parameters
    ----------
    data:
        Extracted invoice entities dict.
    output_path:
        Destination file path.
    fmt:
        One of ``"json"``, ``"csv"``, ``"excel"`` / ``"xlsx"``.

    Returns
    -------
    Resolved :class:`~pathlib.Path` of the written file.

    Raises
    ------
    ValueError
        If *fmt* is not recognised.
    """
    fmt = fmt.lower().strip()
    if fmt == "json":
        return to_json(data, output_path)
    elif fmt == "csv":
        return to_csv(data, output_path)
    elif fmt in ("excel", "xlsx"):
        return to_excel(data, output_path)
    else:
        raise ValueError(f"Unsupported export format: {fmt!r}. Choose json, csv or excel.")
