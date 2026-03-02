from __future__ import annotations

from config import SUPPORTED_LANGUAGES
from services.project_service import ProjectService


def test_create_project_with_russian(tmp_path) -> None:
    service = ProjectService(tmp_path)
    project = service.create_project("ru-test", "Исследовательский вопрос", language="ru")
    assert project.language == "ru"


def test_create_project_default_english(tmp_path) -> None:
    service = ProjectService(tmp_path)
    project = service.create_project("en-test", "Research question")
    assert project.language == "en"


def test_project_language_persists_through_save_load(tmp_path) -> None:
    service = ProjectService(tmp_path)
    project = service.create_project("persist-test", "Вопрос", language="ru")
    service.save_project(project)

    loaded = service.load_project("persist-test")
    assert loaded.language == "ru"


def test_backward_compatibility_missing_language(tmp_path) -> None:
    service = ProjectService(tmp_path)
    project = service.create_project("compat-test", "RQ")
    service.save_project(project)

    import json
    project_path = tmp_path / "compat-test" / "project.json"
    data = json.loads(project_path.read_text())
    del data["language"]
    project_path.write_text(json.dumps(data))

    loaded = service.load_project("compat-test")
    assert loaded.language == "en"


def test_project_card_includes_language(tmp_path) -> None:
    service = ProjectService(tmp_path)
    service.create_project("card-test", "RQ", language="ru")

    cards = service.list_project_cards()
    card = next(c for c in cards if c["id"] == "card-test")
    assert card["language"] == "ru"


def test_supported_languages() -> None:
    assert "en" in SUPPORTED_LANGUAGES
    assert "ru" in SUPPORTED_LANGUAGES
