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
from config import SUPPORTED_LANGUAGES, Settings
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
    language: str = "en"


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
    report_generation_mode: str
    report_fallback_reason: str | None


def _slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\s-]", "", text).strip().lower()
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
        "report_generation_mode": project.report_generation_mode,
        "report_fallback_reason": project.report_fallback_reason,
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
        if hasattr(synthesizer, "synthesize_with_meta"):
            synthesis = await synthesizer.synthesize_with_meta(project)
            report = synthesis["report"]
        else:
            report = await synthesizer.synthesize(project)
            synthesis = {"is_fallback": False, "fallback_reason": None}
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
    project.report_generation_mode = "fallback" if synthesis.get("is_fallback") else "llm"
    project.report_fallback_reason = synthesis.get("fallback_reason")
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
            "report_generation_mode": project.report_generation_mode,
            "report_fallback_reason": project.report_fallback_reason,
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
    language = payload.language or "en"
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}",
        )

    requested_id = payload.id.strip() if isinstance(payload.id, str) else ""
    project_id = requested_id or _generate_project_id(payload.research_question)

    if project_service.exists(project_id):
        project_id = f"{project_id}-{uuid4().hex[:6]}"

    try:
        project = project_service.create_project(
            project_id=project_id,
            research_question=payload.research_question,
            initial_angles=payload.initial_angles,
            language=language,
        )
    except ProjectAlreadyExistsError as err:
        raise HTTPException(status_code=409, detail=str(err)) from err

    return {"project_id": project.id, "status": "created", "language": language}


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
        report_generation_mode=project.report_generation_mode,
        report_fallback_reason=project.report_fallback_reason,
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

    language = getattr(project, "language", "en") or "en"

    try:
        propositions, script = await designer.generate_initial_script(
            research_question=project.research_question,
            initial_angles=project.initial_angles,
            language=language,
        )
    except Exception:
        if language == "ru":
            propositions = [
                Proposition(
                    id="P001",
                    factor="Общий опыт респондента",
                    mechanism="Личное восприятие ключевых ограничений и возможностей",
                    outcome="Позитивное или негативное отношение к исследовательскому вопросу",
                    confidence=0.0,
                    status="untested",
                )
            ]
        else:
            propositions = [
                Proposition(
                    id="P001",
                    factor="Overall respondent experience",
                    mechanism="Personal perception of key constraints and enablers",
                    outcome="Positive or negative sentiment related to the research question",
                    confidence=0.0,
                    status="untested",
                )
            ]
        script = await designer.generate_minimal_script(
            research_question=project.research_question,
            propositions=propositions,
            metrics=project.metrics.model_dump(),
            version=1,
            language=language,
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
        if language == "ru":
            project.proposition_store = [
                Proposition(
                    id=project_service.next_proposition_id(project),
                    factor="Общий опыт респондента",
                    mechanism="Личное восприятие ключевых ограничений и возможностей",
                    outcome="Позитивное или негативное отношение к исследовательскому вопросу",
                    confidence=0.0,
                    status="untested",
                )
            ]
        else:
            project.proposition_store = [
                Proposition(
                    id=project_service.next_proposition_id(project),
                    factor="Overall respondent experience",
                    mechanism="Personal perception of key constraints and enablers",
                    outcome="Positive or negative sentiment related to the research question",
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
            if language == "ru":
                script.sections.append(
                    ScriptSection(
                        proposition_id=prop.id,
                        priority="high",
                        instruction="EXPLORE",
                        main_question=f"Расскажите подробнее о {prop.factor.lower()}?",
                        probes=["Можете привести конкретный пример?", "Что произошло дальше?"],
                        context="Начальная секция",
                    )
                )
            else:
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
        language=language,
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
        prompt = designer.build_interviewer_prompt(script, language=language)
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
        "language": language,
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


@router.post("/{project_id}/stop", status_code=status.HTTP_200_OK)
async def stop_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    sse: SSEManager = Depends(get_sse_manager),
) -> dict:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    if project.status in {"running", "reporting"}:
        project.status = "done"
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
                "sync_pending": project.sync_pending,
                "prompt_safety_status": project.prompt_safety_status,
                "prompt_safety_violations_count": project.prompt_safety_violations_count,
            },
        )
        await sse.emit(project.id, "project_stats", _project_stats(project))

    return {
        "project_id": project.id,
        "status": project.status,
        "message": "Project stopped",
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
async def get_hypothesis_map(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
    synthesizer: SynthesizerAgent = Depends(get_synthesizer_agent),
    sse: SSEManager = Depends(get_sse_manager),
) -> dict:
    try:
        project = project_service.load_project(project_id)
    except ProjectNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    missing_english = [item for item in project.evidence_store if not str(item.quote_english or "").strip()]
    translated_count = 0
    translation_touched = False
    if missing_english:
        translation_map = await synthesizer.translate_evidence_quotes(project.evidence_store)
        for evidence in project.evidence_store:
            if str(evidence.quote_english or "").strip():
                continue
            quote_en = str(translation_map.get(evidence.id, "")).strip()
            language = str(evidence.language or "").strip().lower()
            if quote_en:
                evidence.quote_english = quote_en
                evidence.translation_status = "native_en" if language.startswith("en") else "translated"
                translated_count += 1
                translation_touched = True
            elif language.startswith("en"):
                evidence.quote_english = evidence.quote
                evidence.translation_status = "native_en"
                translated_count += 1
                translation_touched = True
            else:
                evidence.translation_status = "failed"
                translation_touched = True

    if translation_touched:
        project_service.save_project(project)
        if translated_count:
            await sse.emit(
                project.id,
                "translations_backfilled",
                {"project_id": project.id, "translated_count": translated_count},
            )

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

    if hasattr(synthesizer, "synthesize_with_meta"):
        synthesis = await synthesizer.synthesize_with_meta(project)
        report = synthesis["report"]
    else:
        report = await synthesizer.synthesize(project)
        synthesis = {"is_fallback": False, "fallback_reason": None}
    report_path = Path(project_service.data_dir / project_id / "report.md")
    report_path.write_text(report, encoding="utf-8")

    project.report_markdown = report
    project.report_generated_at = datetime.now(timezone.utc)
    project.report_stale = False
    project.report_generation_mode = "fallback" if synthesis.get("is_fallback") else "llm"
    project.report_fallback_reason = synthesis.get("fallback_reason")
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
            "report_generation_mode": project.report_generation_mode,
            "report_fallback_reason": project.report_fallback_reason,
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
