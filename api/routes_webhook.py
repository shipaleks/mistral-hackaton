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


def _extract_text(transcript_payload: Any) -> str:
    if isinstance(transcript_payload, str):
        return transcript_payload

    if isinstance(transcript_payload, list):
        lines: list[str] = []
        for item in transcript_payload:
            if isinstance(item, str):
                lines.append(item)
            elif isinstance(item, dict):
                speaker = str(item.get("speaker", "")).strip()
                text = str(item.get("text", "")).strip()
                if text:
                    prefix = f"{speaker}: " if speaker else ""
                    lines.append(f"{prefix}{text}")
        return "\n".join(lines)

    if isinstance(transcript_payload, dict):
        if "text" in transcript_payload:
            return str(transcript_payload.get("text", ""))
        if "segments" in transcript_payload:
            return _extract_text(transcript_payload["segments"])

    return ""


def _extract_conversation_payload(payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}

    conversation_id = (
        payload.get("conversation_id")
        or data.get("conversation_id")
        or data.get("id")
        or payload.get("id")
    )
    conversation_id = str(conversation_id or "").strip()

    transcript_raw = (
        payload.get("transcript")
        or data.get("transcript")
        or data.get("transcript_text")
        or payload.get("transcript_text")
        or ""
    )
    transcript = _extract_text(transcript_raw).strip()

    metadata = {
        "event": payload.get("event"),
        "has_audio": payload.get("has_audio"),
        "has_user_audio": payload.get("has_user_audio"),
        "has_response_audio": payload.get("has_response_audio"),
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
    if not conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id missing")
    if not transcript:
        raise HTTPException(status_code=400, detail="transcript missing")

    project_id = (
        payload.get("project_id")
        or (
            payload.get("metadata", {}).get("project_id")
            if isinstance(payload.get("metadata"), dict)
            else None
        )
        or settings.default_project_id
    )
    project_id = str(project_id)
    if not project_service.exists(project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    return await pipeline.process_interview(
        project_id=project_id,
        transcript=transcript,
        conversation_id=conversation_id,
        metadata=metadata,
    )
