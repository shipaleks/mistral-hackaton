from __future__ import annotations

import pytest

from agents.synthesizer import SynthesizerAgent
from models.evidence import Evidence
from models.interview import Interview
from models.project import ProjectState
from models.proposition import Proposition


class DummyLLM:
    def __init__(
        self,
        response: str | None = None,
        fail: bool = False,
        json_response: dict | None = None,
    ):
        self.response = response
        self.fail = fail
        self.json_response = json_response or {}

    async def chat(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("llm failure")
        return self.response or ""

    async def chat_json(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("llm failure")
        return self.json_response


def _sample_project() -> ProjectState:
    project = ProjectState(
        id="demo",
        research_question="RQ",
        status="running",
    )
    project.interview_store = []
    project.evidence_store = []
    project.proposition_store = []
    return project


@pytest.mark.asyncio
async def test_synthesizer_no_data_report_when_empty():
    project = _sample_project()
    agent = SynthesizerAgent(DummyLLM(response="ignored"))

    report = await agent.synthesize(project)

    assert "Report Not Ready" in report
    assert "No interview evidence is currently stored" in report


@pytest.mark.asyncio
async def test_synthesizer_rejects_hallucinated_quotes_and_falls_back():
    project = _sample_project()
    project.interview_store = [
        Interview(
            id="INT_001",
            conversation_id="conv-1",
            transcript="text",
            language="en",
            metadata={},
        )
    ]
    project.evidence_store = [
        Evidence(
            id="E001",
            interview_id="INT_001",
            quote="I only slept two hours",
            interpretation="Fatigue impacted performance",
            factor="time pressure",
            mechanism="sleep deprivation",
            outcome="lower focus",
            tags=["fatigue"],
            language="en",
        )
    ]
    project.proposition_store = [
        Proposition(
            id="P001",
            factor="time pressure",
            mechanism="sleep deprivation",
            outcome="lower focus",
            confidence=0.9,
            status="confirmed",
            supporting_evidence=["E001"],
        )
    ]

    hallucinated = '## Executive Summary\n\n"Participant A said she skipped meals"'
    agent = SynthesizerAgent(DummyLLM(response=hallucinated))

    report = await agent.synthesize(project)

    assert "Grounded Fallback" in report
    assert "Participant A" not in report
    assert "I only slept two hours" in report


@pytest.mark.asyncio
async def test_synthesizer_uses_llm_when_grounded():
    project = _sample_project()
    project.interview_store = [
        Interview(
            id="INT_001",
            conversation_id="conv-1",
            transcript="text",
            language="en",
            metadata={},
        )
    ]
    project.evidence_store = [
        Evidence(
            id="E001",
            interview_id="INT_001",
            quote="I only slept two hours",
            interpretation="Fatigue impacted performance",
            factor="time pressure",
            mechanism="sleep deprivation",
            outcome="lower focus",
            tags=["fatigue"],
            language="en",
        )
    ]

    grounded = '## Executive Summary\n\n"I only slept two hours"'
    agent = SynthesizerAgent(DummyLLM(response=grounded))

    report = await agent.synthesize(project)

    assert report == grounded


@pytest.mark.asyncio
async def test_synthesizer_accepts_translated_quotes_with_original_marker():
    project = _sample_project()
    project.interview_store = [
        Interview(
            id="INT_001",
            conversation_id="conv-1",
            transcript="text",
            language="ru",
            metadata={},
        )
    ]
    project.evidence_store = [
        Evidence(
            id="E001",
            interview_id="INT_001",
            quote="Я почти не спал",
            interpretation="Fatigue impacted performance",
            factor="time pressure",
            mechanism="sleep deprivation",
            outcome="lower focus",
            tags=["fatigue"],
            language="ru",
        )
    ]

    translated_report = (
        "## Core Findings\n\n"
        'The participant described exhaustion: "I barely slept." '
        '[original: "Я почти не спал"]'
    )
    agent = SynthesizerAgent(
        DummyLLM(
            response=translated_report,
            json_response={"translations": [{"id": "E001", "english": "I barely slept."}]},
        )
    )

    report = await agent.synthesize(project)

    assert report == translated_report


@pytest.mark.asyncio
async def test_synthesizer_rejects_non_english_quotes_outside_original_marker():
    project = _sample_project()
    project.interview_store = [
        Interview(
            id="INT_001",
            conversation_id="conv-1",
            transcript="text",
            language="ru",
            metadata={},
        )
    ]
    project.evidence_store = [
        Evidence(
            id="E001",
            interview_id="INT_001",
            quote="Я почти не спал",
            interpretation="Fatigue impacted performance",
            factor="time pressure",
            mechanism="sleep deprivation",
            outcome="lower focus",
            tags=["fatigue"],
            language="ru",
        )
    ]
    project.proposition_store = [
        Proposition(
            id="P001",
            factor="time pressure",
            mechanism="sleep deprivation",
            outcome="lower focus",
            confidence=0.6,
            status="exploring",
            supporting_evidence=["E001"],
        )
    ]

    llm_output = '## Core Findings\n\n"Я почти не спал"'
    agent = SynthesizerAgent(
        DummyLLM(
            response=llm_output,
            json_response={"translations": [{"id": "E001", "english": "I barely slept."}]},
        )
    )

    report = await agent.synthesize(project)

    assert "Grounded Fallback" in report
    assert 'Quote (EN): "I barely slept."' in report
