from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from agents.analyst import AnalystAgent
from agents.designer import DesignerAgent
from agents.llm_client import LLMClient
from agents.synthesizer import SynthesizerAgent
from api.routes_projects import router as projects_router
from api.routes_stream import router as stream_router
from api.routes_webhook import router as webhook_router
from config import get_settings
from services.elevenlabs_service import ElevenLabsService
from services.pipeline import Pipeline
from services.project_service import ProjectService
from services.script_safety import ScriptSafetyGuard
from services.sse_manager import SSEManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    project_service = ProjectService(settings.data_dir)
    sse_manager = SSEManager()
    script_safety = ScriptSafetyGuard()

    def build_llm(model: str) -> LLMClient:
        return LLMClient(
            api_key=settings.mistral_api_key,
            model=model,
            api_base=settings.mistral_api_base,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            backoff_seconds=settings.llm_retry_backoff_seconds,
        )

    designer_agent = DesignerAgent(
        llm=build_llm(settings.designer_model),
        max_sections=settings.max_propositions_in_script,
    )
    analyst_agent = AnalystAgent(llm=build_llm(settings.analyst_model))
    synthesizer_agent = SynthesizerAgent(llm=build_llm(settings.synthesizer_model))

    elevenlabs_service = ElevenLabsService(
        api_key=settings.elevenlabs_api_key,
        timeout_seconds=20.0,
        max_retries=settings.llm_max_retries,
        backoff_seconds=settings.llm_retry_backoff_seconds,
    )

    pipeline = Pipeline(
        project_service=project_service,
        analyst=analyst_agent,
        designer=designer_agent,
        elevenlabs=elevenlabs_service,
        sse=sse_manager,
        script_safety=script_safety,
    )

    app.state.settings = settings
    app.state.project_service = project_service
    app.state.sse_manager = sse_manager
    app.state.designer_agent = designer_agent
    app.state.analyst_agent = analyst_agent
    app.state.synthesizer_agent = synthesizer_agent
    app.state.elevenlabs_service = elevenlabs_service
    app.state.script_safety = script_safety
    app.state.pipeline = pipeline

    yield


app = FastAPI(title="Eidetic API", version="0.1.0", lifespan=lifespan)

app.include_router(projects_router, prefix="/api")
app.include_router(webhook_router, prefix="/api")
app.include_router(stream_router, prefix="/api")

UI_FILE = Path(__file__).resolve().parent / "dashboard" / "projects.html"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root():
    if UI_FILE.exists():
        return FileResponse(UI_FILE)
    return {"service": "eidetic", "docs": "/docs"}


@app.get("/ui")
def ui():
    if UI_FILE.exists():
        return FileResponse(UI_FILE)
    return {"service": "eidetic", "docs": "/docs"}
