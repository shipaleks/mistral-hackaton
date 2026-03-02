from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from models.evidence import Evidence
from models.interview import Interview
from models.proposition import Proposition
from models.script import InterviewScript


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


ProjectStatus = Literal["draft", "running", "reporting", "done"]
PromptSafetyStatus = Literal["ok", "sanitized", "fallback"]


class ProjectMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    convergence_score: float = 0.0
    novelty_rate: float = 1.0
    mode: str = "divergent"


class ProjectState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    research_question: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    initial_angles: list[str] = Field(default_factory=list)
    language: str = "en"
    elevenlabs_agent_id: str | None = None

    status: ProjectStatus = "draft"
    talk_to_link: str | None = None

    report_markdown: str | None = None
    report_generated_at: datetime | None = None
    report_stale: bool = False
    report_generation_mode: str = "none"
    report_fallback_reason: str | None = None
    finished_at: datetime | None = None

    evidence_store: list[Evidence] = Field(default_factory=list)
    proposition_store: list[Proposition] = Field(default_factory=list)
    interview_store: list[Interview] = Field(default_factory=list)
    script_versions: list[InterviewScript] = Field(default_factory=list)

    processed_conversation_ids: list[str] = Field(default_factory=list)
    metrics: ProjectMetrics = Field(default_factory=ProjectMetrics)

    sync_pending: bool = False
    sync_pending_script_version: int | None = None
    prompt_safety_status: PromptSafetyStatus = "ok"
    prompt_safety_violations_count: int = 0
    last_prompt_update_at: datetime | None = None

    @property
    def current_script(self) -> InterviewScript | None:
        return self.script_versions[-1] if self.script_versions else None
