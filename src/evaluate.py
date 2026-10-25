"""Exact-match, per-field recall reporting."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .schemas import CSV_COLUMNS, FIELDS


def evaluate(rows: Iterable[dict[str, str]], prediction_by_file: dict[str, dict[str, str | None]]) -> tuple[dict[str, float], list[dict[str, str]]]:
    rows = list(rows)
    totals = {field: 0 for field in FIELDS}
    matches = {field: 0 for field in FIELDS}
    mismatches: list[dict[str, str]] = []
    for row in rows:
        filename = row["File Name"]
        prediction = prediction_by_file.get(filename, {})
        for field in FIELDS:
            expected = (row.get(CSV_COLUMNS[field]) or "").strip()
            actual = str(prediction.get(field) or "").strip()
            totals[field] += 1
            if expected == actual:
                matches[field] += 1
            else:
                mismatches.append({"File Name": filename, "field": field, "expected": expected, "predicted": actual})
    recall = {field: (matches[field] / totals[field] if totals[field] else 0.0) for field in FIELDS}
    return recall, mismatches


def write_mismatches(path: Path, mismatches: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["File Name", "field", "expected", "predicted"])
        writer.writeheader()
        writer.writerows(mismatches)
