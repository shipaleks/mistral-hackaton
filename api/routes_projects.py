from __future__ import annotations

from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from agents.designer import DesignerAgent
from agents.synthesizer import SynthesizerAgent
from api.deps import (
    get_designer_agent,
    get_elevenlabs_service,
    get_pipeline,
    get_project_service,
    get_settings,
    get_synthesizer_agent,
)
from config import Settings
from models.project import ProjectState
from models.proposition import Proposition
from models.script import ScriptSection
from services.elevenlabs_service import ElevenLabsService
from services.pipeline import Pipeline
from services.project_service import (
    ProjectAlreadyExistsError,
    ProjectNotFoundError,
    ProjectService,
)


router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    research_question: str = Field(min_length=1)
    initial_angles: list[str] = Field(default_factory=list)


class StartProjectRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    elevenlabs_agent_id: str | None = None


class SimulateInterviewRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    transcript: str = Field(min_length=1)
    conversation_id: str | None = None


class SynthesizeResponse(BaseModel):
    report: str
    report_path: str


@router.post("", status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreateRequest,
    project_service: ProjectService = Depends(get_project_service),
) -> dict[str, str]:
    try:
        project = project_service.create_project(
            project_id=payload.id,
            research_question=payload.research_question,
            initial_angles=payload.initial_angles,
        )
    except ProjectAlreadyExistsError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err

    return {"project_id": project.id, "status": "created"}


@router.get("", status_code=status.HTTP_200_OK)
def list_projects(
    project_service: ProjectService = Depends(get_project_service),
) -> dict[str, list[str]]:
    return {"projects": project_service.list_projects()}


@router.get("/{project_id}", response_model=ProjectState)
def get_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectState:
    try:
        return project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@router.get("/{project_id}/evidence", status_code=status.HTTP_200_OK)
def get_evidence(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> list[dict]:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return [item.model_dump(mode="json") for item in project.evidence_store]


@router.get("/{project_id}/propositions", status_code=status.HTTP_200_OK)
def get_propositions(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> list[dict]:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return [item.model_dump(mode="json") for item in project.proposition_store]


@router.get("/{project_id}/scripts", status_code=status.HTTP_200_OK)
def get_scripts(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> list[dict]:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return [item.model_dump(mode="json") for item in project.script_versions]


@router.post("/{project_id}/start", status_code=status.HTTP_200_OK)
async def start_project(
    project_id: str,
    payload: StartProjectRequest,
    project_service: ProjectService = Depends(get_project_service),
    designer: DesignerAgent = Depends(get_designer_agent),
    elevenlabs: ElevenLabsService = Depends(get_elevenlabs_service),
    settings: Settings = Depends(get_settings),
) -> dict:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    try:
        propositions, script = await designer.generate_initial_script(
            research_question=project.research_question,
            initial_angles=project.initial_angles,
        )
    except Exception:
        propositions = [
            Proposition(
                id="P001",
                factor="Overall hackathon experience",
                mechanism="Personal perception of events and constraints",
                outcome="Positive or negative sentiment during participation",
                confidence=0.0,
                status="untested",
            )
        ]
        script = await designer.generate_minimal_script(
            research_question=project.research_question,
            propositions=propositions,
            metrics=project.metrics.model_dump(),
            version=1,
        )

    # Ensure proposition IDs are stable.
    proposition_ids = set()
    fixed: list[Proposition] = []
    next_idx = len(project.proposition_store) + 1
    for prop in propositions:
        if not prop.id or prop.id in proposition_ids:
            while True:
                candidate = f"P{next_idx:03d}"
                next_idx += 1
                if candidate not in proposition_ids:
                    prop.id = candidate
                    break
        proposition_ids.add(prop.id)
        fixed.append(prop)

    project.proposition_store = fixed
    if not project.proposition_store:
        project.proposition_store = [
            Proposition(
                id=project_service.next_proposition_id(project),
                factor="Overall hackathon experience",
                mechanism="Personal perception of events and constraints",
                outcome="Positive or negative sentiment during participation",
                confidence=0.0,
                status="untested",
            )
        ]
    project.metrics.mode = script.mode
    project.metrics.convergence_score = script.convergence_score
    project.metrics.novelty_rate = script.novelty_rate

    if not script.sections and project.proposition_store:
        script.sections = []
        for prop in project.proposition_store[: settings.max_propositions_in_script]:
            script.sections.append(
                ScriptSection(
                    proposition_id=prop.id,
                    priority="high",
                    instruction="EXPLORE",
                    main_question=f"Could you tell me more about {prop.factor.lower()}?",
                    probes=["Can you give an example?", "What happened next?"],
                    context="Bootstrap section",
                )
            )

    project.script_versions = []
    project_service.add_script(project, script)

    agent_id = payload.elevenlabs_agent_id or project.elevenlabs_agent_id or settings.elevenlabs_agent_id
    if agent_id:
        project.elevenlabs_agent_id = agent_id

    project.sync_pending = False
    project.sync_pending_script_version = None

    if project.elevenlabs_agent_id:
        prompt = designer.build_interviewer_prompt(script)
        try:
            await elevenlabs.update_agent_prompt(project.elevenlabs_agent_id, prompt)
        except Exception:
            project.sync_pending = True
            project.sync_pending_script_version = script.version

    project_service.save_project(project)

    response = {
        "project_id": project.id,
        "status": "started",
        "script_version": script.version,
        "propositions": len(project.proposition_store),
        "sync_pending": project.sync_pending,
    }
    if project.elevenlabs_agent_id:
        response["talk_to_link"] = elevenlabs.get_talk_to_link(project.elevenlabs_agent_id)
    return response


@router.post("/{project_id}/simulate", status_code=status.HTTP_200_OK)
async def simulate_interview(
    project_id: str,
    payload: SimulateInterviewRequest,
    pipeline: Pipeline = Depends(get_pipeline),
):
    conversation_id = payload.conversation_id or f"sim-{project_id}-{uuid4().hex[:12]}"
    return await pipeline.process_interview(
        project_id=project_id,
        transcript=payload.transcript,
        conversation_id=conversation_id,
        metadata={"source": "simulate"},
    )


@router.post("/{project_id}/synthesize", response_model=SynthesizeResponse)
async def synthesize(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    synthesizer: SynthesizerAgent = Depends(get_synthesizer_agent),
) -> SynthesizeResponse:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    report = await synthesizer.synthesize(project)
    report_path = Path(project_service.data_dir / project_id / "report.md")
    report_path.write_text(report, encoding="utf-8")

    return SynthesizeResponse(report=report, report_path=str(report_path))


@router.delete("/{project_id}", status_code=status.HTTP_200_OK)
def delete_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> dict[str, str]:
    try:
        project_service.delete_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return {"project_id": project_id, "status": "deleted"}
