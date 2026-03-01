from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    interview_id: str
    quote: str
    interpretation: str
    factor: str
    mechanism: str
    outcome: str
    tags: list[str] = Field(default_factory=list)
    language: str = "en"
    quote_english: str | None = None
    translation_status: str = "pending"
    timestamp: datetime = Field(default_factory=utc_now)
