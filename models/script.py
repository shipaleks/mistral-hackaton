from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SectionPriority = Literal["high", "medium", "low"]
SectionInstruction = Literal["EXPLORE", "VERIFY", "CHALLENGE", "SATURATED"]
ModeType = Literal["divergent", "convergent"]


class ScriptSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    proposition_id: str
    priority: SectionPriority
    instruction: SectionInstruction
    main_question: str
    probes: list[str] = Field(default_factory=list)
    context: str = ""


class InterviewScript(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int
    generated_after_interview: str | None = None
    research_question: str
    opening_question: str
    sections: list[ScriptSection] = Field(default_factory=list)
    closing_question: str
    wildcard: str
    mode: ModeType = "divergent"
    convergence_score: float = 0.0
    novelty_rate: float = 1.0
    changes_summary: str = ""
