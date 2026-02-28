from __future__ import annotations

from training.anonymize import detect_leaks, regex_cleanup_example


def test_regex_cleanup_removes_contacts_and_yandex() -> None:
    sample = {
        "context": [{"role": "user", "content": "My email is me@example.com, Yandex Maps helped"}],
        "good_question": "Call me at +1 (555) 111-2222?",
        "improved_question": "Visit https://yandex.ru and tell me",
        "technique": "why",
    }
    cleaned = regex_cleanup_example(sample)
    assert detect_leaks(cleaned) == []
    assert "[CONTACT]" in cleaned["good_question"]
    assert "[URL]" in cleaned["improved_question"]


def test_detect_leaks_flags_remaining_pii() -> None:
    sample = {
        "context": [{"role": "user", "content": "Reach me at john@corp.com"}],
        "good_question": "What happened?",
        "improved_question": "What happened next?",
    }
    leaks = detect_leaks(sample)
    assert "email" in leaks
