from __future__ import annotations

from services.project_service import ProjectService


def test_find_project_for_agent_prefers_running_and_latest(tmp_path) -> None:
    service = ProjectService(tmp_path)

    done_project = service.create_project("p-done", "Done project")
    done_project.elevenlabs_agent_id = "agent_same"
    done_project.status = "done"
    service.save_project(done_project)

    running_old = service.create_project("p-running-old", "Running old")
    running_old.elevenlabs_agent_id = "agent_same"
    running_old.status = "running"
    service.save_project(running_old)

    running_latest = service.create_project("p-running-latest", "Running latest")
    running_latest.elevenlabs_agent_id = "agent_same"
    running_latest.status = "running"
    service.save_project(running_latest)

    selected = service.find_project_for_agent("agent_same")

    assert selected == "p-running-latest"


def test_find_project_for_agent_returns_none_when_missing(tmp_path) -> None:
    service = ProjectService(tmp_path)
    service.create_project("p1", "RQ")

    assert service.find_project_for_agent("agent_missing") is None
    assert service.find_project_for_agent("  ") is None

