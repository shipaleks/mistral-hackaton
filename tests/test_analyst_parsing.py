from __future__ import annotations

import pytest

from agents.analyst import AnalystAgent


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    async def chat_json(self, *args, **kwargs):
        return self.payload


@pytest.mark.asyncio
async def test_analyst_coerces_partially_invalid_payload() -> None:
    payload = {
        "new_evidence": [
            {
                "id": "E001",
                "quote": "It was intense",
                "interpretation": "Time pressure was high",
                "factor": "time pressure",
                "mechanism": "forced prioritization",
                "outcome": "faster decisions",
                "tags": ["time", "decision"],
                "language": "en",
            },
            {"id": "E_BAD"},
            "bad-item",
        ],
        "evidence_mappings": [
            {"evidence_id": "E001", "proposition_id": "P001", "relationship": "supports"},
            {"evidence_id": "E001", "proposition_id": "P001", "relationship": "irrelevant"},
        ],
        "new_propositions": [
            {
                "id": "P002",
                "factor": "mentor support",
                "mechanism": "faster unblock",
                "outcome": "higher confidence",
                "confidence": 0.7,
                "status": "exploring",
            },
            {"id": "broken"},
        ],
        "proposition_updates": [
            {"id": "P001", "new_confidence": 0.8, "new_status": "confirmed"},
            {"id": "", "new_confidence": 0.2, "new_status": "exploring"},
        ],
        "metrics": {"convergence_score": 0.5, "novelty_rate": 0.2, "mode": "divergent"},
        "prunes": ["P999"],
    }
    analyst = AnalystAgent(FakeLLM(payload))

    result = await analyst.analyze_interview(
        transcript="User: It was intense",
        existing_evidence=[],
        existing_propositions=[],
        interview_id="INT_001",
        interview_index=1,
    )

    assert len(result.new_evidence) == 1
    assert result.new_evidence[0].id == "E001"
    assert len(result.evidence_mappings) == 1
    assert len(result.new_propositions) == 1
    assert len(result.proposition_updates) == 1
    assert result.metrics.convergence_score == 0.5
    assert result.prunes == ["P999"]
