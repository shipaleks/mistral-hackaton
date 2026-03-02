from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

SUPPORTED_LANGUAGES = {"en", "ru"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_base_url: str
    data_dir: Path
    default_project_id: str
    default_language: str

    mistral_api_key: str
    mistral_api_base: str
    mistral_model: str

    elevenlabs_api_key: str
    elevenlabs_agent_id: str
    elevenlabs_webhook_secret: str
    elevenlabs_signature_tolerance_seconds: int

    designer_model: str
    analyst_model: str
    synthesizer_model: str

    convergence_score_threshold: float
    novelty_rate_threshold: float
    merge_overlap_threshold: float
    prune_confidence_threshold: float
    prune_min_interviews: int

    max_interview_duration_minutes: int
    max_propositions_in_script: int

    llm_timeout_seconds: float
    synthesizer_timeout_seconds: float
    llm_max_retries: int
    llm_retry_backoff_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    mistral_model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    llm_timeout_seconds = _env_float("LLM_TIMEOUT_SECONDS", 45.0)
    return Settings(
        app_base_url=os.getenv("APP_BASE_URL", "http://localhost:8000"),
        data_dir=Path(os.getenv("DATA_DIR", "./data/projects")),
        default_project_id=os.getenv("DEFAULT_PROJECT_ID", "hackathon-demo"),
        default_language=os.getenv("DEFAULT_LANGUAGE", "en"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
        mistral_api_base=os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
        mistral_model=mistral_model,
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        elevenlabs_agent_id=os.getenv("ELEVENLABS_AGENT_ID", ""),
        elevenlabs_webhook_secret=os.getenv("ELEVENLABS_WEBHOOK_SECRET", ""),
        elevenlabs_signature_tolerance_seconds=_env_int(
            "ELEVENLABS_SIGNATURE_TOLERANCE_SECONDS", 300
        ),
        designer_model=os.getenv("DESIGNER_MODEL", mistral_model) or mistral_model,
        analyst_model=os.getenv("ANALYST_MODEL", mistral_model) or mistral_model,
        synthesizer_model=os.getenv("SYNTHESIZER_MODEL", mistral_model) or mistral_model,
        convergence_score_threshold=_env_float("CONVERGENCE_SCORE_THRESHOLD", 0.6),
        novelty_rate_threshold=_env_float("NOVELTY_RATE_THRESHOLD", 0.15),
        merge_overlap_threshold=_env_float("MERGE_OVERLAP_THRESHOLD", 0.6),
        prune_confidence_threshold=_env_float("PRUNE_CONFIDENCE_THRESHOLD", 0.15),
        prune_min_interviews=_env_int("PRUNE_MIN_INTERVIEWS", 3),
        max_interview_duration_minutes=_env_int("MAX_INTERVIEW_DURATION_MINUTES", 10),
        max_propositions_in_script=_env_int("MAX_PROPOSITIONS_IN_SCRIPT", 8),
        llm_timeout_seconds=llm_timeout_seconds,
        synthesizer_timeout_seconds=_env_float(
            "SYNTHESIZER_TIMEOUT_SECONDS",
            max(90.0, llm_timeout_seconds),
        ),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 3),
        llm_retry_backoff_seconds=_env_float("LLM_RETRY_BACKOFF_SECONDS", 0.8),
    )
