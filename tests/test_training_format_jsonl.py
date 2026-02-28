from __future__ import annotations

from training.format_jsonl import _estimate_tokens, _normalize_record


def test_estimate_tokens_non_zero() -> None:
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("abcdefgh") == 2


def test_normalize_record_builds_messages() -> None:
    row = {
        "context": [
            {"role": "user", "content": "I had problems with onboarding"},
            {"role": "assistant", "content": "Could you explain?"},
        ],
        "good_question": "What part was hardest?",
        "improved_question": "Which exact step blocked you most?",
        "quality_score": 4,
    }
    record = _normalize_record(row, target_field="good_question", min_quality=4)
    assert record is not None
    assert record["messages"][0]["role"] == "user"
    assert "Conversation context" in record["messages"][0]["content"]
    assert record["messages"][1]["content"] == "What part was hardest?"
