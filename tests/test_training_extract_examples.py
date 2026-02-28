from __future__ import annotations

from training.extract_examples import _find_turn_index, _sanitize_example, _to_training_turns


def test_to_training_turns_maps_roles() -> None:
    segments = [
        {"role": "MODERATOR", "text": "How was it?"},
        {"role": "INTERVIEWEE", "text": "Quite hard"},
    ]
    turns = _to_training_turns(segments, max_turns=10)
    assert turns == [
        {"role": "assistant", "content": "How was it?"},
        {"role": "user", "content": "Quite hard"},
    ]


def test_sanitize_example_quality_gate() -> None:
    raw = {
        "context": [{"role": "user", "content": "answer"}, {"role": "assistant", "content": "question"}],
        "good_question": "What happened next?",
        "improved_question": "Could you share a specific example?",
        "quality_score": 4,
        "technique": "example",
    }
    clean = _sanitize_example(raw, min_quality=4)
    assert clean is not None
    assert clean["quality_score"] == 4


def test_find_turn_index() -> None:
    turns = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "What happened next?"},
    ]
    assert _find_turn_index(turns, "What happened next?") == 1
