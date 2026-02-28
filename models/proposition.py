from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


PropositionStatus = Literal[
    "untested",
    "exploring",
    "confirmed",
    "challenged",
    "saturated",
    "weak",
    "merged",
]


class Proposition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    factor: str
    mechanism: str
    outcome: str
    confidence: float = 0.0
    status: PropositionStatus = "untested"
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    first_seen_interview: int = 0
    last_updated_interview: int = 0
    interviews_without_new_evidence: int = 0
    merged_into: str | None = None
