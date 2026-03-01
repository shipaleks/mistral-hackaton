from __future__ import annotations

import pytest

from models.analysis import AnalysisMetrics, AnalysisResult, EvidenceMapping, PropositionUpdate
from models.evidence import Evidence
from models.proposition import Proposition
from models.script import InterviewScript, ScriptSection
from services.pipeline import Pipeline
from services.project_service import ProjectService
from services.sse_manager import SSEManager


class FakeAnalyst:
    async def analyze_interview(self, *args, **kwargs):
        return AnalysisResult(
            new_evidence=[
                Evidence(
                    id="",
                    interview_id=kwargs["interview_id"],
                    quote="We learned fast",
                    interpretation="Rapid iteration",
                    factor="time pressure",
                    mechanism="forced tradeoffs",
                    outcome="faster iteration",
                    tags=["time"],
                    language="en",
                )
            ],
            evidence_mappings=[
                EvidenceMapping(
                    evidence_id="E001",
                    proposition_id="P001",
                    relationship="supports",
                )
            ],
            proposition_updates=[
                PropositionUpdate(id="P001", new_confidence=0.8, new_status="confirmed")
            ],
            metrics=AnalysisMetrics(convergence_score=0.8, novelty_rate=0.2, mode="convergent"),
        )


class FakeDesigner:
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
            convergence_score=0.8,
            novelty_rate=0.2,
            changes_summary="Updated",
        )

    async def generate_minimal_script(self, *args, **kwargs):
        return InterviewScript(
            version=1,
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
        return f"SCRIPT {script.version}"


class PersonalizingDesigner(FakeDesigner):
    async def update_script(self, *args, **kwargs):
        prev = kwargs["previous_script"]
        return InterviewScript(
            version=prev.version + 1,
            generated_after_interview="INT_001",
            research_question=prev.research_question,
            opening_question="Earlier you mentioned issues. Can we continue?",
            sections=[
                ScriptSection(
                    proposition_id="P001",
                    priority="high",
                    instruction="CHALLENGE",
                    main_question="Earlier, you mentioned working alone. How did that feel?",
                    probes=["You said your project implementation changed. Why?"],
                    context="Earlier you mentioned details",
                )
            ],
            closing_question="From what you said before, what else?",
            wildcard="Anything else you told me earlier?",
            mode="convergent",
            convergence_score=0.8,
            novelty_rate=0.2,
            changes_summary="Updated with memory",
        )

    def build_interviewer_prompt(self, script):
        return (
            f"OPEN={script.opening_question}\n"
            f"Q={script.sections[0].main_question}\n"
            f"P={script.sections[0].probes[0]}\n"
            f"CLOSE={script.closing_question}"
        )


class FakeElevenLabs:
    def __init__(self):
        self.calls = []

    async def update_agent_prompt(self, agent_id, new_prompt):
        self.calls.append((agent_id, new_prompt))


class AnalystNoMapping:
    async def analyze_interview(self, *args, **kwargs):
        return AnalysisResult(
            new_evidence=[
                Evidence(
                    id="",
                    interview_id=kwargs["interview_id"],
                    quote="We lost focus because of short deadlines",
                    quote_english="We lost focus because of short deadlines",
                    translation_status="translated",
                    interpretation="Deadline pressure reduced focus",
                    factor="deadline pressure",
                    mechanism="sleep loss",
                    outcome="low focus",
                    tags=["deadline", "focus"],
                    language="en",
                )
            ],
            evidence_mappings=[],
            proposition_updates=[],
            metrics=AnalysisMetrics(convergence_score=0.3, novelty_rate=0.6, mode="divergent"),
        )


@pytest.mark.asyncio
async def test_pipeline_idempotency_and_updates(tmp_path):
    project_service = ProjectService(tmp_path)
    project = project_service.create_project("demo", "What is your experience?")
    project.elevenlabs_agent_id = "agent_123"
    project.proposition_store = [
        Proposition(
            id="P001",
            factor="time pressure",
            mechanism="forced tradeoffs",
            outcome="faster iteration",
            confidence=0.2,
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
                    main_question="How was time pressure?",
                    probes=["Example?"],
                    context="",
                )
            ],
            closing_question="Close",
            wildcard="Anything else?",
            mode="divergent",
            convergence_score=0.2,
            novelty_rate=0.7,
            changes_summary="Init",
        )
    ]
    project_service.save_project(project)

    sse = SSEManager()
    queue = sse.subscribe("demo")

    pipeline = Pipeline(
        project_service=project_service,
        analyst=FakeAnalyst(),
        designer=FakeDesigner(),
        elevenlabs=FakeElevenLabs(),
        sse=sse,
    )

    result1 = await pipeline.process_interview(
        project_id="demo",
        transcript="User: We learned fast",
        conversation_id="conv-1",
    )
    result2 = await pipeline.process_interview(
        project_id="demo",
        transcript="User: duplicate",
        conversation_id="conv-1",
    )

    saved = project_service.load_project("demo")

    assert result1["status"] == "processed"
    assert result2["status"] == "duplicate"
    assert len(saved.interview_store) == 1
    assert len(saved.evidence_store) == 1
    assert saved.proposition_store[0].confidence == 0.8
    assert saved.script_versions[-1].version == 2

    events = []
    while not queue.empty():
        events.append(await queue.get())
    event_names = [item["event"] for item in events]
    assert "new_evidence" in event_names
    assert "proposition_updated" in event_names
    assert "script_updated" in event_names


@pytest.mark.asyncio
async def test_pipeline_sanitizes_personalized_prompt_before_sync(tmp_path):
    project_service = ProjectService(tmp_path)
    project = project_service.create_project("demo", "What is your experience?")
    project.elevenlabs_agent_id = "agent_123"
    project.proposition_store = [
        Proposition(
            id="P001",
            factor="team pressure",
            mechanism="coordination constraints",
            outcome="stress",
            confidence=0.3,
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
                    main_question="How was teamwork?",
                    probes=["Example?"],
                    context="",
                )
            ],
            closing_question="Close",
            wildcard="Anything else?",
            mode="divergent",
            convergence_score=0.2,
            novelty_rate=0.7,
            changes_summary="Init",
        )
    ]
    project_service.save_project(project)

    sse = SSEManager()
    queue = sse.subscribe("demo")
    elevenlabs = FakeElevenLabs()

    pipeline = Pipeline(
        project_service=project_service,
        analyst=FakeAnalyst(),
        designer=PersonalizingDesigner(),
        elevenlabs=elevenlabs,
        sse=sse,
    )

    result = await pipeline.process_interview(
        project_id="demo",
        transcript="User: We learned fast",
        conversation_id="conv-2",
    )

    assert result["status"] == "processed"
    assert elevenlabs.calls
    _, prompt = elevenlabs.calls[-1]
    assert "earlier you mentioned" not in prompt.lower()
    assert "you said" not in prompt.lower()

    saved = project_service.load_project("demo")
    assert saved.prompt_safety_status in {"sanitized", "fallback"}
    assert saved.prompt_safety_violations_count > 0

    events = []
    while not queue.empty():
        events.append(await queue.get())
    names = [item["event"] for item in events]
    assert "prompt_sanitized" in names


@pytest.mark.asyncio
async def test_pipeline_adds_heuristic_links_without_touching_confirmed(tmp_path):
    project_service = ProjectService(tmp_path)
    project = project_service.create_project("demo", "What affects participant focus?")
    project.elevenlabs_agent_id = "agent_123"
    project.proposition_store = [
        Proposition(
            id="P001",
            factor="deadline pressure",
            mechanism="sleep loss",
            outcome="low focus",
            confidence=0.2,
            status="exploring",
            supporting_evidence=[],
            contradicting_evidence=[],
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
                    main_question="How do deadlines affect your focus?",
                    probes=["Example?"],
                    context="",
                )
            ],
            closing_question="Close",
            wildcard="Any more?",
            mode="divergent",
            convergence_score=0.2,
            novelty_rate=0.7,
            changes_summary="Init",
        )
    ]
    project_service.save_project(project)

    sse = SSEManager()
    queue = sse.subscribe("demo")

    pipeline = Pipeline(
        project_service=project_service,
        analyst=AnalystNoMapping(),
        designer=FakeDesigner(),
        elevenlabs=FakeElevenLabs(),
        sse=sse,
    )

    result = await pipeline.process_interview(
        project_id="demo",
        transcript="User: deadlines destroyed focus",
        conversation_id="conv-h1",
    )

    assert result["status"] == "processed"
    saved = project_service.load_project("demo")
    prop = saved.proposition_store[0]
    assert prop.supporting_evidence == []
    assert len(prop.heuristic_supporting_evidence) >= 1

    events = []
    while not queue.empty():
        events.append(await queue.get())
    names = [item["event"] for item in events]
    assert "heuristic_links_updated" in names
