from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.routes_projects import (
    StartProjectRequest,
    _generate_report_task,
    get_qrcode,
    start_project,
)
from models.analysis import AnalysisMetrics, AnalysisResult
from models.evidence import Evidence
from models.proposition import Proposition
from models.script import InterviewScript, ScriptSection
from services.pipeline import Pipeline
from services.project_service import ProjectService
from services.script_safety import ScriptSafetyGuard
from services.sse_manager import SSEManager


class FakeSynthesizer:
    async def synthesize(self, project):
        return "## Executive Summary\n\nDone."


class NoopElevenLabs:
    async def update_agent_prompt(self, *args, **kwargs):
        return None

    @staticmethod
    def get_talk_to_link(agent_id: str) -> str:
        return f"https://elevenlabs.io/app/talk-to/{agent_id}"


class SimpleAnalyst:
    async def analyze_interview(self, *args, **kwargs):
        return AnalysisResult(
            new_evidence=[
                Evidence(
                    id="",
                    interview_id=kwargs["interview_id"],
                    quote="Fresh signal",
                    interpretation="New input after finish",
                    factor="team dynamics",
                    mechanism="alignment",
                    outcome="progress",
                    tags=["team"],
                    language="en",
                )
            ],
            metrics=AnalysisMetrics(convergence_score=0.7, novelty_rate=0.2, mode="convergent"),
        )


class SimpleDesigner:
    async def update_script(self, *args, **kwargs):
        prev = kwargs["previous_script"]
        return InterviewScript(
            version=prev.version + 1,
            generated_after_interview="INT_001",
            research_question=prev.research_question,
            opening_question=prev.opening_question,
            sections=prev.sections,
            closing_question=prev.closing_question,
            wildcard=prev.wildcard,
            mode="convergent",
            convergence_score=0.7,
            novelty_rate=0.2,
            changes_summary="Updated",
        )

    async def generate_minimal_script(self, *args, **kwargs):
        return InterviewScript(
            version=1,
            research_question=kwargs["research_question"],
            opening_question="Open",
            sections=[],
            closing_question="Close",
            wildcard="Wildcard",
            mode="divergent",
            convergence_score=0.0,
            novelty_rate=1.0,
            changes_summary="Init",
        )

    def build_interviewer_prompt(self, script):
        return f"PROMPT {script.version}"

    async def generate_initial_script(self, *args, **kwargs):
        propositions = [
            Proposition(
                id="P001",
                factor="team dynamics",
                mechanism="alignment quality",
                outcome="delivery pace",
                confidence=0.2,
                status="exploring",
            )
        ]
        script = InterviewScript(
            version=1,
            research_question=kwargs["research_question"],
            opening_question="Open",
            sections=[
                ScriptSection(
                    proposition_id="P001",
                    priority="high",
                    instruction="EXPLORE",
                    main_question="How is your team collaboration?",
                    probes=["Example"],
                    context="",
                )
            ],
            closing_question="Close",
            wildcard="Anything else?",
            mode="divergent",
            convergence_score=0.1,
            novelty_rate=0.9,
            changes_summary="Init",
        )
        return propositions, script


@pytest.mark.asyncio
async def test_generate_report_task_sets_done_and_report_fields(tmp_path):
    project_service = ProjectService(tmp_path)
    project = project_service.create_project("demo", "RQ")
    project.status = "reporting"
    project_service.save_project(project)

    sse = SSEManager()
    queue = sse.subscribe("demo")

    await _generate_report_task("demo", project_service, FakeSynthesizer(), sse)

    saved = project_service.load_project("demo")
    assert saved.status == "done"
    assert saved.report_markdown is not None
    assert saved.report_generated_at is not None
    assert saved.finished_at is not None
    assert saved.report_stale is False

    events = []
    while not queue.empty():
        events.append(await queue.get())
    names = [e["event"] for e in events]
    assert "report_ready" in names
    assert "project_status" in names


def test_qrcode_endpoint_returns_png(tmp_path):
    project_service = ProjectService(tmp_path)
    project = project_service.create_project("demo", "RQ")
    project.talk_to_link = "https://example.com/talk"
    project_service.save_project(project)

    response = get_qrcode(
        "demo",
        project_service=project_service,
        elevenlabs=NoopElevenLabs(),
    )

    assert response.media_type == "image/png"
    assert isinstance(response.body, (bytes, bytearray))
    assert len(response.body) > 100


@pytest.mark.asyncio
async def test_pipeline_marks_report_stale_after_done(tmp_path):
    project_service = ProjectService(tmp_path)
    project = project_service.create_project("demo", "RQ")
    project.status = "done"
    project.report_stale = False
    project.elevenlabs_agent_id = "agent"
    project.proposition_store = [
        Proposition(
            id="P001",
            factor="team dynamics",
            mechanism="alignment",
            outcome="progress",
            confidence=0.6,
            status="confirmed",
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
                    main_question="How is teamwork?",
                    probes=["Example"],
                    context="",
                )
            ],
            closing_question="Close",
            wildcard="Any more?",
            mode="convergent",
            convergence_score=0.6,
            novelty_rate=0.2,
            changes_summary="Init",
        )
    ]
    project_service.save_project(project)

    sse = SSEManager()
    queue = sse.subscribe("demo")

    pipeline = Pipeline(
        project_service=project_service,
        analyst=SimpleAnalyst(),
        designer=SimpleDesigner(),
        elevenlabs=NoopElevenLabs(),
        sse=sse,
    )

    result = await pipeline.process_interview(
        project_id="demo",
        transcript="User: new interview",
        conversation_id="conv-stale",
    )

    saved = project_service.load_project("demo")
    assert result["status"] == "processed"
    assert saved.report_stale is True
    assert saved.status == "done"

    events = []
    while not queue.empty():
        events.append(await queue.get())
    names = [e["event"] for e in events]
    assert "report_stale" in names
    assert "project_stats" in names


@pytest.mark.asyncio
async def test_start_project_rejects_agent_used_by_active_project(tmp_path):
    project_service = ProjectService(tmp_path)
    active = project_service.create_project("active", "RQ1")
    active.status = "running"
    active.elevenlabs_agent_id = "agent_busy"
    project_service.save_project(active)

    target = project_service.create_project("target", "RQ2")
    project_service.save_project(target)

    with pytest.raises(HTTPException) as exc:
        await start_project(
            project_id="target",
            payload=StartProjectRequest(elevenlabs_agent_id="agent_busy"),
            project_service=project_service,
            designer=SimpleDesigner(),
            elevenlabs=NoopElevenLabs(),
            script_safety=ScriptSafetyGuard(),
            settings=type("Settings", (), {"max_propositions_in_script": 8, "elevenlabs_agent_id": ""})(),
            sse=SSEManager(),
        )

    assert exc.value.status_code == 409
    assert "already used by active project" in str(exc.value.detail)
