from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.deps import get_pipeline, get_project_service, get_settings
from config import Settings
from services.pipeline import Pipeline
from services.project_service import ProjectService
from services.webhook_security import verify_elevenlabs_signature


router = APIRouter(prefix="/webhook", tags=["webhook"])


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _extract_transcript_line(item: dict[str, Any]) -> str:
    speaker = _first_non_empty(
        item.get("speaker"),
        item.get("role"),
        item.get("author"),
        item.get("name"),
    )

    text = _first_non_empty(
        _extract_text(item.get("text")),
        _extract_text(item.get("message")),
        _extract_text(item.get("content")),
        _extract_text(item.get("utterance")),
        _extract_text(item.get("transcript_text")),
    )
    if not text:
        return ""
    if speaker:
        return f"{speaker}: {text}"
    return text


def _extract_text(transcript_payload: Any) -> str:
    if transcript_payload is None:
        return ""

    if isinstance(transcript_payload, str):
        return transcript_payload

    if isinstance(transcript_payload, (int, float, bool)):
        return str(transcript_payload)

    if isinstance(transcript_payload, list):
        lines: list[str] = []
        for item in transcript_payload:
            text = _extract_text(item).strip()
            if text:
                lines.append(text)
        return "\n".join(lines)

    if isinstance(transcript_payload, dict):
        line = _extract_transcript_line(transcript_payload)
        if line:
            return line

        for key in ("text", "message", "content", "utterance", "transcript_text"):
            if key not in transcript_payload:
                continue
            text = _extract_text(transcript_payload.get(key)).strip()
            if text:
                return text

        for key in ("segments", "entries", "items", "messages", "turns", "transcript"):
            if key not in transcript_payload:
                continue
            text = _extract_text(transcript_payload.get(key)).strip()
            if text:
                return text

    return ""


def _extract_conversation_payload(payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    data = _as_dict(payload.get("data"))
    payload_metadata = _as_dict(payload.get("metadata"))
    data_metadata = _as_dict(data.get("metadata"))
    client_data = _as_dict(data.get("conversation_initiation_client_data"))
    dynamic_variables = _as_dict(client_data.get("dynamic_variables"))

    event_type = _first_non_empty(payload.get("type"), payload.get("event"), data.get("event"))
    agent_id = _first_non_empty(
        payload.get("agent_id"),
        data.get("agent_id"),
        payload_metadata.get("agent_id"),
        data_metadata.get("agent_id"),
        client_data.get("agent_id"),
        _as_dict(data.get("agent")).get("id"),
    )

    conversation_id = _first_non_empty(
        payload.get("conversation_id")
        or payload.get("conversationId")
        or data.get("conversation_id")
        or data.get("conversationId")
        or data.get("id")
        or payload.get("id")
    )

    transcript_raw = (
        payload.get("transcript")
        or data.get("transcript")
        or data.get("transcript_text")
        or payload.get("transcript_text")
        or data.get("messages")
        or data.get("turns")
        or _as_dict(data.get("analysis")).get("transcript")
        or ""
    )
    transcript = _extract_text(transcript_raw).strip()

    project_id_hint = _first_non_empty(
        payload.get("project_id"),
        data.get("project_id"),
        payload_metadata.get("project_id"),
        data_metadata.get("project_id"),
        dynamic_variables.get("project_id"),
        dynamic_variables.get("projectId"),
    )

    metadata = {
        "event": event_type,
        "agent_id": agent_id,
        "project_id_hint": project_id_hint,
        "has_audio": payload.get("has_audio"),
        "has_user_audio": payload.get("has_user_audio"),
        "has_response_audio": payload.get("has_response_audio"),
        "conversation_end_reason": data.get("conversation_end_reason"),
        "start_time_unix_secs": data.get("start_time_unix_secs"),
        "call_duration_secs": data.get("call_duration_secs"),
        "raw_data": data,
    }

    return conversation_id, transcript, metadata


@router.post("/elevenlabs", status_code=status.HTTP_200_OK)
async def elevenlabs_webhook(
    request: Request,
    settings: Settings = Depends(get_settings),
    project_service: ProjectService = Depends(get_project_service),
    pipeline: Pipeline = Depends(get_pipeline),
) -> dict:
    raw_body = await request.body()
    signature_header = request.headers.get("ElevenLabs-Signature") or request.headers.get(
        "x-elevenlabs-signature"
    )

    if not verify_elevenlabs_signature(
        raw_body=raw_body,
        signature_header=signature_header,
        secret=settings.elevenlabs_webhook_secret,
        tolerance_seconds=settings.elevenlabs_signature_tolerance_seconds,
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as err:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from err

    conversation_id, transcript, metadata = _extract_conversation_payload(payload)
    if not conversation_id or not transcript:
        return {
            "status": "ignored",
            "reason": "missing_conversation_or_transcript",
            "event": metadata.get("event"),
            "conversation_id": conversation_id or None,
        }

    project_id = _first_non_empty(metadata.get("project_id_hint"))
    if not project_id:
        project_id = project_service.find_project_for_agent(
            _first_non_empty(metadata.get("agent_id"))
        ) or ""
    if not project_id:
        project_id = _first_non_empty(settings.default_project_id)

    if not project_service.exists(project_id):
        fallback_project_id = project_service.find_project_for_agent(
            _first_non_empty(metadata.get("agent_id"))
        )
        if fallback_project_id and project_service.exists(fallback_project_id):
            project_id = fallback_project_id
        else:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    if not project_id:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    return await pipeline.process_interview(
        project_id=project_id,
        transcript=transcript,
        conversation_id=conversation_id,
        metadata=metadata,
    )
