from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = "https://api.mistral.ai/v1",
        timeout_seconds: float = 45.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.8,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        data = await self._post_json("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM returned no choices")

        message = choices[0].get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)

        if isinstance(content, dict):
            return json.dumps(content)

        raise RuntimeError("Unable to parse LLM content")

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        content = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as err:
            raise RuntimeError("LLM JSON decode error") from err

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not configured")

        url = f"{self.api_base}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(url, headers=headers, json=payload)

                if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        "Transient LLM error",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()
                return response.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as err:
                last_error = err
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        raise RuntimeError("Failed to call LLM") from last_error
