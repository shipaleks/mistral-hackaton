from __future__ import annotations

import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
DEFAULT_AUDIO_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".webm",
    ".wav",
    ".ogg",
    ".flac",
    ".mp4",
    ".mpeg",
}


class MistralAPIError(RuntimeError):
    """Raised for non-retriable API errors."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    ensure_dir(path.parent)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, item: Any) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False))
        fh.write("\n")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def collect_audio_files(directory: Path) -> list[Path]:
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_AUDIO_EXTENSIONS
    ]
    return sorted(files)


def get_mistral_config() -> tuple[str, str]:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    api_base = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1").strip()
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not configured")
    if not api_base:
        raise RuntimeError("MISTRAL_API_BASE is empty")
    return api_key, api_base.rstrip("/")


def json_from_text(raw_text: str) -> Any:
    raw_text = raw_text.strip()
    if not raw_text:
        raise ValueError("Empty response")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, raw_text)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON payload found in text")


class MistralClient:
    def __init__(
        self,
        api_key: str,
        api_base: str,
        *,
        timeout_seconds: float = 180.0,
        max_retries: int = 4,
        backoff_seconds: float = 0.8,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        data: Any = None,
        files: Any = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if json_payload is not None:
            headers["Content-Type"] = "application/json"
        url = f"{self.api_base}{path}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.request(
                        method,
                        url,
                        headers=headers,
                        json=json_payload,
                        data=data,
                        files=files,
                    )

                if response.status_code in TRANSIENT_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"Transient error status={response.status_code}",
                        request=response.request,
                        response=response,
                    )

                if response.status_code >= 400:
                    raise MistralAPIError(
                        f"Mistral API error status={response.status_code}: {response.text}"
                    )
                return response.json()
            except (httpx.RequestError, httpx.HTTPStatusError) as err:
                last_error = err
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        raise RuntimeError("Failed to call Mistral API after retries") from last_error

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        data = self._request("POST", "/chat/completions", json_payload=payload)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by chat completions")
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return json.dumps(content)

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        text = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        parsed = json_from_text(text)
        if not isinstance(parsed, dict):
            raise RuntimeError("Expected JSON object in response")
        return parsed

    def transcribe(
        self,
        *,
        model: str,
        audio_file: Path,
        diarize: bool,
        timestamp_granularities: list[str] | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        payload: list[tuple[str, str]] = [("model", model), ("diarize", str(diarize).lower())]
        if language:
            payload.append(("language", language))
        for granularity in timestamp_granularities or []:
            payload.append(("timestamp_granularities", granularity))

        with audio_file.open("rb") as fh:
            files = {"file": (audio_file.name, fh, "application/octet-stream")}
            return self._request("POST", "/audio/transcriptions", data=payload, files=files)

    def upload_file(self, *, file_path: Path, purpose: str = "fine-tune") -> dict[str, Any]:
        payload = [("purpose", purpose)]
        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh, "application/jsonl")}
            return self._request("POST", "/files", data=payload, files=files)

    def create_fine_tuning_job(
        self, *, payload: dict[str, Any], dry_run: bool | None = None
    ) -> dict[str, Any]:
        path = "/fine_tuning/jobs"
        if dry_run is not None:
            path += f"?dry_run={'true' if dry_run else 'false'}"
        return self._request("POST", path, json_payload=payload)

    def start_fine_tuning_job(self, *, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/fine_tuning/jobs/{job_id}/start")

    def get_fine_tuning_job(self, *, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/fine_tuning/jobs/{job_id}")

