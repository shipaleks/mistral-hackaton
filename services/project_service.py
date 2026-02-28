from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.interview import Interview
from models.project import ProjectState
from models.script import InterviewScript


class ProjectNotFoundError(FileNotFoundError):
    pass


class ProjectAlreadyExistsError(FileExistsError):
    pass


class ProjectService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        return self.data_dir / project_id

    def _project_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _interviews_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "interviews"

    def _scripts_dir(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "scripts"

    def exists(self, project_id: str) -> bool:
        return self._project_file(project_id).exists()

    def create_project(
        self, project_id: str, research_question: str, initial_angles: list[str] | None = None
    ) -> ProjectState:
        if self.exists(project_id):
            raise ProjectAlreadyExistsError(f"Project '{project_id}' already exists")

        project = ProjectState(
            id=project_id,
            research_question=research_question,
            initial_angles=initial_angles or [],
        )
        self._project_dir(project_id).mkdir(parents=True, exist_ok=True)
        self._interviews_dir(project_id).mkdir(parents=True, exist_ok=True)
        self._scripts_dir(project_id).mkdir(parents=True, exist_ok=True)
        self.save_project(project)
        return project

    def load_project(self, project_id: str) -> ProjectState:
        project_file = self._project_file(project_id)
        if not project_file.exists():
            raise ProjectNotFoundError(f"Project '{project_id}' not found")
        with project_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return ProjectState.model_validate(payload)

    def save_project(self, project: ProjectState) -> None:
        project_dir = self._project_dir(project.id)
        project_dir.mkdir(parents=True, exist_ok=True)
        self._interviews_dir(project.id).mkdir(parents=True, exist_ok=True)
        self._scripts_dir(project.id).mkdir(parents=True, exist_ok=True)

        with self._project_file(project.id).open("w", encoding="utf-8") as f:
            json.dump(project.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    def delete_project(self, project_id: str) -> None:
        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            raise ProjectNotFoundError(f"Project '{project_id}' not found")
        for path in sorted(project_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        project_dir.rmdir()

    def next_interview_id(self, project: ProjectState) -> str:
        return f"INT_{len(project.interview_store) + 1:03d}"

    def next_evidence_id(self, project: ProjectState) -> str:
        return f"E{len(project.evidence_store) + 1:03d}"

    def next_proposition_id(self, project: ProjectState) -> str:
        return f"P{len(project.proposition_store) + 1:03d}"

    def add_interview(self, project: ProjectState, interview: Interview) -> None:
        project.interview_store.append(interview)
        filename = self._interviews_dir(project.id) / f"{interview.id}.json"
        with filename.open("w", encoding="utf-8") as f:
            json.dump(interview.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    def add_script(self, project: ProjectState, script: InterviewScript) -> None:
        project.script_versions.append(script)
        filename = self._scripts_dir(project.id) / f"script_v{script.version}.json"
        with filename.open("w", encoding="utf-8") as f:
            json.dump(script.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    def list_projects(self) -> list[str]:
        return sorted([p.name for p in self.data_dir.iterdir() if p.is_dir()])

    def project_summary(self, project_id: str) -> dict[str, Any]:
        project = self.load_project(project_id)
        return {
            "id": project.id,
            "research_question": project.research_question,
            "interviews": len(project.interview_store),
            "evidence": len(project.evidence_store),
            "propositions": len(project.proposition_store),
            "scripts": len(project.script_versions),
            "mode": project.metrics.mode,
        }
