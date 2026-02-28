from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class SSEManager:
    def __init__(self) -> None:
        self.subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, project_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers[project_id].add(queue)
        return queue

    def unsubscribe(self, project_id: str, queue: asyncio.Queue) -> None:
        self.subscribers[project_id].discard(queue)
        if not self.subscribers[project_id]:
            self.subscribers.pop(project_id, None)

    async def emit(self, project_id: str, event_type: str, data: dict[str, Any]) -> None:
        if project_id not in self.subscribers:
            return
        message = {"event": event_type, "data": data}
        for queue in list(self.subscribers[project_id]):
            await queue.put(message)
