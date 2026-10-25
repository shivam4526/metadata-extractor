"""Shared extraction result schema and CSV column mapping."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

FIELDS = (
    "agreement_value",
    "agreement_start_date",
    "agreement_end_date",
    "renewal_notice_days",
    "party_one",
    "party_two",
)

CSV_COLUMNS = {
    "agreement_value": "Aggrement Value",
    "agreement_start_date": "Aggrement Start Date",
    "agreement_end_date": "Aggrement End Date",
    "renewal_notice_days": "Renewal Notice (Days)",
    "party_one": "Party One",
    "party_two": "Party Two",
}


class ExtractionResult(BaseModel):
    """The only structured contract between the LLM and the application."""

    model_config = ConfigDict(extra="forbid")
    agreement_value: Optional[str] = Field(default=None)
    agreement_start_date: Optional[str] = Field(default=None)
    agreement_end_date: Optional[str] = Field(default=None)
    renewal_notice_days: Optional[str] = Field(default=None)
    party_one: Optional[str] = Field(default=None)
    party_two: Optional[str] = Field(default=None)
