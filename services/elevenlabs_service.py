from __future__ import annotations

import asyncio

import httpx


class ElevenLabsService:
    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.8,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    @property
    def headers(self) -> dict[str, str]:
        return {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def update_agent_prompt(self, agent_id: str, new_prompt: str) -> None:
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not configured")
        if not agent_id:
            raise RuntimeError("ELEVENLABS agent_id is missing")

        payload = {
            "conversation_config": {
                "agent": {
                    "prompt": {
                        "prompt": new_prompt,
                    }
                }
            }
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.patch(
                        f"{self.BASE_URL}/convai/agents/{agent_id}",
                        headers=self.headers,
                        json=payload,
                    )

                if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        "Transient ElevenLabs error",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()
                return
            except (httpx.RequestError, httpx.HTTPStatusError) as err:
                last_error = err
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        raise RuntimeError("Failed to update ElevenLabs prompt") from last_error

    @staticmethod
    def get_talk_to_link(agent_id: str) -> str:
        return f"https://elevenlabs.io/app/talk-to/{agent_id}"
