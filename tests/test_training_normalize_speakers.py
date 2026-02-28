from __future__ import annotations

from training.normalize_speakers import (
    ROLE_INTERVIEWEE,
    ROLE_MODERATOR,
    gate_two_roles,
    heuristic_mapping,
    normalize_transcript_entry,
    question_ratio,
)


class FakeClient:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def chat_json(self, **_: object) -> dict[str, object]:
        return {"mapping": self.mapping}


def _segments() -> list[dict[str, object]]:
    return [
        {"speaker_id": "A", "text": "Hello everyone?", "start": 0.0, "end": 1.0},
        {"speaker_id": "B", "text": "I joined the hackathon", "start": 1.0, "end": 2.0},
        {"speaker_id": "C", "text": "It was intense", "start": 2.0, "end": 3.0},
        {"speaker_id": "A", "text": "Can you give an example?", "start": 3.0, "end": 4.0},
    ]


def test_question_ratio() -> None:
    assert question_ratio(["one?", "two"]) == 0.5


def test_heuristic_mapping_picks_questioning_speaker() -> None:
    mapping = heuristic_mapping(_segments())
    assert mapping["A"] == ROLE_MODERATOR
    assert mapping["B"] == ROLE_INTERVIEWEE


def test_gate_two_roles() -> None:
    segments = [
        {"role": ROLE_MODERATOR, "text": "Why?"},
        {"role": ROLE_INTERVIEWEE, "text": "Because of time pressure"},
    ]
    ok, reasons, _ = gate_two_roles(segments)
    assert ok is True
    assert reasons == []


def test_normalize_transcript_entry_auto_merge_to_two_roles() -> None:
    transcript = {"language": "en", "segments": _segments()}
    normalized, manifest = normalize_transcript_entry(
        transcript=transcript,
        client=FakeClient({"A": ROLE_MODERATOR, "B": ROLE_INTERVIEWEE, "C": ROLE_INTERVIEWEE}),
        model="mistral-large-latest",
        max_segments_for_llm=50,
    )
    assert manifest["status"] in {"auto_merged", "success"}
    roles = {seg["role"] for seg in normalized["segments"]}
    assert ROLE_MODERATOR in roles
    assert ROLE_INTERVIEWEE in roles
