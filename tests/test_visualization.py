from __future__ import annotations

import pytest

from api.routes_projects import get_hypothesis_map
from agents.synthesizer import SynthesizerAgent
from models.evidence import Evidence
from models.proposition import Proposition
from services.project_service import ProjectService
from services.visualization import build_hypothesis_map
from services.sse_manager import SSEManager


class DummyLLM:
    async def chat(self, *args, **kwargs):
        return ""

    async def chat_json(self, *args, **kwargs):
        return {}


def test_build_hypothesis_map_supports_and_contradicts(tmp_path) -> None:
    service = ProjectService(tmp_path)
    project = service.create_project("demo", "RQ")

    project.evidence_store = [
        Evidence(
            id="E001",
            interview_id="INT_001",
            quote="The team rules were confusing.",
            interpretation="Rules blocked collaboration",
            factor="Team rules",
            mechanism="Constraint mismatch",
            outcome="Low coordination",
            tags=["rules", "team"],
            language="en",
        ),
        Evidence(
            id="E002",
            interview_id="INT_001",
            quote="Mentors helped us a lot.",
            interpretation="Mentorship reduced friction",
            factor="Mentorship access",
            mechanism="Timely support",
            outcome="Higher velocity",
            tags=["mentor"],
            language="en",
        ),
    ]
    project.proposition_store = [
        Proposition(
            id="P001",
            factor="Team formation dynamics",
            mechanism="Rules and support quality",
            outcome="Execution quality",
            confidence=0.6,
            status="exploring",
            supporting_evidence=["E002"],
            contradicting_evidence=["E001"],
        )
    ]
    service.save_project(project)

    model = build_hypothesis_map(project)

    assert model["stats"]["hypotheses"] == 1
    assert model["stats"]["evidence"] == 2
    assert model["stats"]["supports_edges"] == 1
    assert model["stats"]["contradicts_edges"] == 1
    assert any(edge["relation"] == "supports" for edge in model["edges"])
    assert any(edge["relation"] == "contradicts" for edge in model["edges"])
    assert "status_legend" in model
    assert "unvalidated_hypotheses" in model
    assert all("source_type" in edge for edge in model["edges"])
    assert all("explanation" in edge for edge in model["edges"])


@pytest.mark.asyncio
async def test_hypothesis_map_route_returns_payload(tmp_path) -> None:
    service = ProjectService(tmp_path)
    project = service.create_project("demo", "RQ")
    service.save_project(project)

    payload = await get_hypothesis_map(
        "demo",
        project_service=service,
        synthesizer=SynthesizerAgent(DummyLLM()),
        sse=SSEManager(),
    )

    assert payload["project_id"] == "demo"
    assert "nodes" in payload
    assert "edges" in payload
    assert "stats" in payload
    assert "progress_snapshot" in payload
