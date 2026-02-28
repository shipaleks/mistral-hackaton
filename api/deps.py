from __future__ import annotations

from fastapi import Request

from agents.analyst import AnalystAgent
from agents.designer import DesignerAgent
from agents.synthesizer import SynthesizerAgent
from config import Settings
from services.elevenlabs_service import ElevenLabsService
from services.pipeline import Pipeline
from services.project_service import ProjectService
from services.sse_manager import SSEManager


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_project_service(request: Request) -> ProjectService:
    return request.app.state.project_service


def get_sse_manager(request: Request) -> SSEManager:
    return request.app.state.sse_manager


def get_designer_agent(request: Request) -> DesignerAgent:
    return request.app.state.designer_agent


def get_analyst_agent(request: Request) -> AnalystAgent:
    return request.app.state.analyst_agent


def get_synthesizer_agent(request: Request) -> SynthesizerAgent:
    return request.app.state.synthesizer_agent


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def get_elevenlabs_service(request: Request) -> ElevenLabsService:
    return request.app.state.elevenlabs_service
