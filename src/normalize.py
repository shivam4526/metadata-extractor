"""Minimal formatting of values already selected by the LLM; never extracts from text."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from dateutil import parser

from .schemas import ExtractionResult


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _date(value: Optional[str]) -> Optional[str]:
    value = _clean(value)
    if not value:
        return None
    try:
        return parser.parse(value, dayfirst=True, fuzzy=False).strftime("%d.%m.%Y")
    except (ValueError, OverflowError, TypeError):
        return value


def _number(value: Optional[str]) -> Optional[str]:
    value = _clean(value)
    if not value:
        return None
    # Only formatting the model's selected numeric value, not searching source text.
    cleaned = value.replace(",", "").replace("₹", "").replace("$", "").replace("Rs.", "").replace("Rs", "").replace("/-", "").strip()
    try:
        number = float(cleaned)
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        return value


def normalize(result: ExtractionResult) -> ExtractionResult:
    return ExtractionResult(
        agreement_value=_number(result.agreement_value),
        agreement_start_date=_date(result.agreement_start_date),
        agreement_end_date=_date(result.agreement_end_date),
        renewal_notice_days=_number(result.renewal_notice_days),
        party_one=_clean(result.party_one),
        party_two=_clean(result.party_two),
    )
