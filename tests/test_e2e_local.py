from __future__ import annotations

import pytest

from models.analysis import AnalysisMetrics, AnalysisResult, PropositionUpdate
from models.evidence import Evidence
from models.proposition import Proposition
from models.script import InterviewScript, ScriptSection
from services.pipeline import Pipeline
from services.project_service import ProjectService
from services.sse_manager import SSEManager


class RotatingAnalyst:
    def __init__(self) -> None:
        self.counter = 0

    async def analyze_interview(self, *args, **kwargs):
        self.counter += 1
        evidence_id = f"E{self.counter:03d}"
        confidence = min(0.3 * self.counter, 0.9)
        return AnalysisResult(
            new_evidence=[
                Evidence(
                    id=evidence_id,
                    interview_id=kwargs["interview_id"],
                    quote=f"Interview {self.counter} quote",
                    interpretation="Iterative learning",
                    factor="team dynamics",
                    mechanism="faster alignment",
                    outcome="better execution",
                    tags=["team"],
                    language="en",
                )
            ],
            proposition_updates=[
                PropositionUpdate(
                    id="P001",
                    new_confidence=confidence,
                    new_status="exploring" if confidence < 0.7 else "confirmed",
                )
            ],
            metrics=AnalysisMetrics(
                convergence_score=confidence,
                novelty_rate=max(0.1, 0.9 - confidence),
                mode="convergent" if confidence >= 0.7 else "divergent",
            ),
        )


class ScriptBumpingDesigner:
    async def update_script(self, *args, **kwargs):
        prev = kwargs["previous_script"]
        return InterviewScript(
            version=prev.version + 1,
            generated_after_interview=kwargs["evidence"][-1].interview_id,
            research_question=prev.research_question,
            opening_question=prev.opening_question,
            sections=prev.sections,
            closing_question=prev.closing_question,
            wildcard=prev.wildcard,
            mode="convergent" if prev.version >= 2 else "divergent",
            convergence_score=min(1.0, prev.convergence_score + 0.2),
            novelty_rate=max(0.1, prev.novelty_rate - 0.2),
            changes_summary=f"Version {prev.version + 1}",
        )

    async def generate_minimal_script(self, *args, **kwargs):
        return InterviewScript(
            version=kwargs["version"],
            research_question=kwargs["research_question"],
            opening_question="Start",
            sections=[],
            closing_question="End",
            wildcard="Wildcard",
            mode="divergent",
            convergence_score=0.0,
            novelty_rate=1.0,
            changes_summary="Init",
        )

    def build_interviewer_prompt(self, script):
        return f"PROMPT {script.version}"


class NoopElevenLabs:
    async def update_agent_prompt(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
async def test_local_e2e_three_interviews(tmp_path):
    project_service = ProjectService(tmp_path)
    project = project_service.create_project("hackathon-demo", "What is your experience?")
    project.elevenlabs_agent_id = "agent"
    project.proposition_store = [
        Proposition(
            id="P001",
            factor="team dynamics",
            mechanism="faster alignment",
            outcome="better execution",
            confidence=0.1,
            status="exploring",
        )
    ]
    project.script_versions = [
        InterviewScript(
            version=1,
            research_question=project.research_question,
            opening_question="Open",
            sections=[
                ScriptSection(
                    proposition_id="P001",
                    priority="high",
                    instruction="EXPLORE",
                    main_question="How is your team working together?",
                    probes=["Any examples?"],
                    context="Initial",
                )
            ],
            closing_question="Close",
            wildcard="Anything else?",
            mode="divergent",
            convergence_score=0.1,
            novelty_rate=0.9,
            changes_summary="v1",
        )
    ]
    project_service.save_project(project)

    pipeline = Pipeline(
        project_service=project_service,
        analyst=RotatingAnalyst(),
        designer=ScriptBumpingDesigner(),
        elevenlabs=NoopElevenLabs(),
        sse=SSEManager(),
    )

    for idx in range(1, 4):
        result = await pipeline.process_interview(
            project_id="hackathon-demo",
            transcript=f"User interview {idx}",
            conversation_id=f"conv-{idx}",
        )
        assert result["status"] == "processed"

    saved = project_service.load_project("hackathon-demo")
    assert len(saved.interview_store) == 3
    assert len(saved.script_versions) >= 4
    assert saved.script_versions[0].version == 1
    assert saved.script_versions[-1].version >= 3
