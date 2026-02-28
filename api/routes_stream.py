from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from api.deps import get_sse_manager
from services.sse_manager import SSEManager


router = APIRouter(prefix="/projects", tags=["stream"])


@router.get("/{project_id}/stream")
async def project_stream(
    project_id: str,
    sse_manager: SSEManager = Depends(get_sse_manager),
) -> EventSourceResponse:
    queue = sse_manager.subscribe(project_id)

    async def event_generator():
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {
                        "event": message["event"],
                        "data": json.dumps(message["data"], ensure_ascii=False),
                    }
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            sse_manager.unsubscribe(project_id, queue)

    return EventSourceResponse(event_generator())
