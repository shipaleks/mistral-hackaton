from __future__ import annotations

import pytest

from agents.prompt_loader import load_prompt


def test_load_prompt_english_explicit() -> None:
    text = load_prompt("analyst_system.txt", language="en")
    assert len(text) > 50
    assert "evidence" in text.lower() or "interview" in text.lower()


def test_load_prompt_russian() -> None:
    text = load_prompt("analyst_system.txt", language="ru")
    assert len(text) > 50
    assert any(ord(c) > 0x0400 for c in text), "Russian prompt should contain Cyrillic characters"


def test_load_prompt_fallback_to_root() -> None:
    text = load_prompt("training_extraction.txt", language="ru")
    assert len(text) > 0


def test_load_prompt_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent_file_xyz.txt", language="en")


def test_load_prompt_default_language_is_english() -> None:
    text_default = load_prompt("analyst_system.txt")
    text_en = load_prompt("analyst_system.txt", language="en")
    assert text_default == text_en
