from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import qrcode
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from agents.designer import DesignerAgent
from agents.synthesizer import SynthesizerAgent
from api.deps import (
    get_designer_agent,
    get_elevenlabs_service,
    get_pipeline,
    get_project_service,
    get_script_safety,
    get_settings,
    get_sse_manager,
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
from services.script_safety import ScriptSafetyGuard
from services.sse_manager import SSEManager
from services.visualization import build_hypothesis_map


router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
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


class ProjectReportResponse(BaseModel):
    status: str
    report_markdown: str | None
    report_generated_at: str | None
    report_stale: bool


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value[:40] or "research"


def _generate_project_id(question: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{_slugify(question)}-{ts}"


def _project_stats(project: ProjectState) -> dict:
    return {
        "project_id": project.id,
        "status": project.status,
        "participants": len(project.interview_store),
        "interviews_count": len(project.interview_store),
        "evidence_count": len(project.evidence_store),
        "propositions_count": len(project.proposition_store),
        "active_propositions_count": len(
            [p for p in project.proposition_store if p.status not in {"weak", "merged"}]
        ),
        "convergence_score": project.metrics.convergence_score,
        "novelty_rate": project.metrics.novelty_rate,
        "mode": project.metrics.mode,
        "report_stale": project.report_stale,
        "prompt_safety_status": project.prompt_safety_status,
        "prompt_safety_violations_count": project.prompt_safety_violations_count,
    }


async def _generate_report_task(
    project_id: str,
    project_service: ProjectService,
    synthesizer: SynthesizerAgent,
    sse: SSEManager,
) -> None:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError:
        return

    try:
        report = await synthesizer.synthesize(project)
    except Exception as err:
        project.status = "running"
        project_service.save_project(project)
        await sse.emit(
            project_id,
            "project_status",
            {
                "project_id": project.id,
                "status": project.status,
                "report_stale": project.report_stale,
                "error": str(err),
            },
        )
        return

    report_path = Path(project_service.data_dir / project_id / "report.md")
    report_path.write_text(report, encoding="utf-8")

    project.report_markdown = report
    project.report_generated_at = datetime.now(timezone.utc)
    project.report_stale = False
    if project.finished_at is None:
        project.finished_at = datetime.now(timezone.utc)
    project.status = "done"
    project_service.save_project(project)

    await sse.emit(
        project_id,
        "report_ready",
        {
            "project_id": project.id,
            "status": project.status,
            "report_generated_at": project.report_generated_at.isoformat()
            if project.report_generated_at
            else None,
            "report_stale": project.report_stale,
            "report_path": str(report_path),
        },
    )
    await sse.emit(
        project_id,
        "project_status",
        {
            "project_id": project.id,
            "status": project.status,
            "report_stale": project.report_stale,
        },
    )
    await sse.emit(project_id, "project_stats", _project_stats(project))


@router.post("", status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreateRequest,
    project_service: ProjectService = Depends(get_project_service),
) -> dict[str, str]:
    requested_id = payload.id.strip() if isinstance(payload.id, str) else ""
    project_id = requested_id or _generate_project_id(payload.research_question)

    if project_service.exists(project_id):
        project_id = f"{project_id}-{uuid4().hex[:6]}"

    try:
        project = project_service.create_project(
            project_id=project_id,
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


@router.get("/cards", status_code=status.HTTP_200_OK)
def list_project_cards(
    project_service: ProjectService = Depends(get_project_service),
) -> list[dict]:
    return project_service.list_project_cards()


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


@router.get("/{project_id}/report", response_model=ProjectReportResponse)
def get_report(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> ProjectReportResponse:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    return ProjectReportResponse(
        status=project.status,
        report_markdown=project.report_markdown,
        report_generated_at=(
            project.report_generated_at.isoformat() if project.report_generated_at else None
        ),
        report_stale=project.report_stale,
    )


@router.post("/{project_id}/start", status_code=status.HTTP_200_OK)
async def start_project(
    project_id: str,
    payload: StartProjectRequest,
    project_service: ProjectService = Depends(get_project_service),
    designer: DesignerAgent = Depends(get_designer_agent),
    elevenlabs: ElevenLabsService = Depends(get_elevenlabs_service),
    script_safety: ScriptSafetyGuard = Depends(get_script_safety),
    settings: Settings = Depends(get_settings),
    sse: SSEManager = Depends(get_sse_manager),
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

    agent_id = payload.elevenlabs_agent_id or project.elevenlabs_agent_id or settings.elevenlabs_agent_id
    if agent_id:
        active_project = project_service.find_active_project_for_agent(
            agent_id=agent_id,
            exclude_project_id=project.id,
        )
        if active_project:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Agent '{agent_id}' is already used by active project '{active_project}'",
            )
        project.elevenlabs_agent_id = agent_id
        project.talk_to_link = elevenlabs.get_talk_to_link(agent_id)

    safety_result = script_safety.enforce(
        script=script,
        research_question=project.research_question,
        propositions=project.proposition_store,
    )
    script = safety_result.script
    project.prompt_safety_status = safety_result.status
    project.prompt_safety_violations_count = safety_result.violations_count
    if safety_result.status in {"sanitized", "fallback"}:
        marker = f"safety_guard={safety_result.status} violations={safety_result.violations_count}"
        summary = script.changes_summary.strip() or "Script initialized"
        if marker not in summary:
            script.changes_summary = f"{summary} [{marker}]"

    project.script_versions = []
    project_service.add_script(project, script)

    project.sync_pending = False
    project.sync_pending_script_version = None
    project.status = "running"

    if project.elevenlabs_agent_id:
        prompt = designer.build_interviewer_prompt(script)
        try:
            await elevenlabs.update_agent_prompt(project.elevenlabs_agent_id, prompt)
            project.last_prompt_update_at = datetime.now(timezone.utc)
        except Exception:
            project.sync_pending = True
            project.sync_pending_script_version = script.version

    project_service.save_project(project)

    await sse.emit(
        project.id,
        "project_status",
        {
            "project_id": project.id,
            "status": project.status,
            "report_stale": project.report_stale,
            "sync_pending": project.sync_pending,
            "prompt_safety_status": project.prompt_safety_status,
            "prompt_safety_violations_count": project.prompt_safety_violations_count,
        },
    )
    await sse.emit(project.id, "project_stats", _project_stats(project))
    if safety_result.status in {"sanitized", "fallback"}:
        await sse.emit(
            project.id,
            "prompt_sanitized",
            {
                "project_id": project.id,
                "script_version": script.version,
                "status": safety_result.status,
                "violations_count": safety_result.violations_count,
            },
        )
    if safety_result.topic_redirect_applied:
        await sse.emit(
            project.id,
            "topic_redirect_applied",
            {"project_id": project.id, "script_version": script.version},
        )
    await sse.emit(
        project.id,
        "visualization_model_ready",
        {"project_id": project.id, "reason": "project_started"},
    )

    response = {
        "project_id": project.id,
        "status": "started",
        "script_version": script.version,
        "propositions": len(project.proposition_store),
        "sync_pending": project.sync_pending,
        "project_status": project.status,
        "prompt_safety_status": project.prompt_safety_status,
        "prompt_safety_violations_count": project.prompt_safety_violations_count,
    }
    if project.talk_to_link:
        response["talk_to_link"] = project.talk_to_link
    return response


@router.post("/{project_id}/finish", status_code=status.HTTP_202_ACCEPTED)
async def finish_project(
    project_id: str,
    background_tasks: BackgroundTasks,
    project_service: ProjectService = Depends(get_project_service),
    synthesizer: SynthesizerAgent = Depends(get_synthesizer_agent),
    sse: SSEManager = Depends(get_sse_manager),
) -> dict:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    project.status = "reporting"
    if project.finished_at is None:
        project.finished_at = datetime.now(timezone.utc)
    project_service.save_project(project)

    await sse.emit(
        project.id,
        "project_status",
        {
            "project_id": project.id,
            "status": project.status,
            "report_stale": project.report_stale,
        },
    )

    background_tasks.add_task(_generate_report_task, project.id, project_service, synthesizer, sse)

    return {
        "project_id": project.id,
        "status": "reporting",
        "message": "Report generation started",
    }


@router.get("/{project_id}/qrcode", status_code=status.HTTP_200_OK)
def get_qrcode(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    elevenlabs: ElevenLabsService = Depends(get_elevenlabs_service),
) -> Response:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    talk_to_link = project.talk_to_link
    if not talk_to_link and project.elevenlabs_agent_id:
        talk_to_link = elevenlabs.get_talk_to_link(project.elevenlabs_agent_id)

    if not talk_to_link:
        raise HTTPException(status_code=400, detail="talk_to_link is not available for this project")

    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data(talk_to_link)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@router.get("/{project_id}/visualization/hypothesis-map", status_code=status.HTTP_200_OK)
def get_hypothesis_map(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
) -> dict:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return build_hypothesis_map(project)


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
    sse: SSEManager = Depends(get_sse_manager),
) -> SynthesizeResponse:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    report = await synthesizer.synthesize(project)
    report_path = Path(project_service.data_dir / project_id / "report.md")
    report_path.write_text(report, encoding="utf-8")

    project.report_markdown = report
    project.report_generated_at = datetime.now(timezone.utc)
    project.report_stale = False
    if project.finished_at is None:
        project.finished_at = datetime.now(timezone.utc)
    project.status = "done"
    project_service.save_project(project)

    await sse.emit(
        project.id,
        "report_ready",
        {
            "project_id": project.id,
            "status": project.status,
            "report_generated_at": project.report_generated_at.isoformat(),
            "report_stale": project.report_stale,
            "report_path": str(report_path),
        },
    )

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
