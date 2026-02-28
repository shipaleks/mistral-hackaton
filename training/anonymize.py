from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .common import MistralClient, append_jsonl, get_mistral_config, iter_jsonl, now_iso, write_json
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        MistralClient,
        append_jsonl,
        get_mistral_config,
        iter_jsonl,
        now_iso,
        write_json,
    )


EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\-\s()]{6,}\d)")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[a-zA-Z0-9_]{3,}")
LONG_ID_RE = re.compile(r"\b\d{6,}\b")

YANDEX_TERMS = re.compile(
    r"(?:\b(?:yandex|yango|ya\.ru|yandex\.ru|yandexgo|alice)\b|яндекс|y[aа]ndex)",
    re.IGNORECASE,
)


def _record_key(record: dict[str, Any]) -> str:
    source = str(record.get("source_file", "")).strip()
    turn_index = str(record.get("turn_index", "")).strip()
    question = str(record.get("good_question", "")).strip()
    return f"{source}|{turn_index}|{question}"


def _apply_regex_cleanup(text: str) -> str:
    updated = text
    updated = EMAIL_RE.sub("[CONTACT]", updated)
    updated = PHONE_RE.sub("[CONTACT]", updated)
    updated = URL_RE.sub("[URL]", updated)
    updated = HANDLE_RE.sub("[CONTACT]", updated)
    updated = LONG_ID_RE.sub("[ID]", updated)
    updated = YANDEX_TERMS.sub("[REDACTED_ORG]", updated)
    return updated.strip()


def regex_cleanup_example(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    context = cleaned.get("context")
    if isinstance(context, list):
        clean_context: list[dict[str, str]] = []
        for item in context:
            if not isinstance(item, dict):
                continue
            clean_context.append(
                {
                    "role": str(item.get("role", "")).strip().lower(),
                    "content": _apply_regex_cleanup(str(item.get("content", ""))),
                }
            )
        cleaned["context"] = clean_context

    for field in ("good_question", "improved_question", "technique"):
        cleaned[field] = _apply_regex_cleanup(str(cleaned.get(field, "")))
    return cleaned


def detect_leaks(record: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for field in ("good_question", "improved_question"):
        texts.append(str(record.get(field, "")))
    context = record.get("context")
    if isinstance(context, list):
        for item in context:
            if isinstance(item, dict):
                texts.append(str(item.get("content", "")))

    leaks: set[str] = set()
    for text in texts:
        if not text:
            continue
        if EMAIL_RE.search(text):
            leaks.add("email")
        if PHONE_RE.search(text):
            leaks.add("phone")
        if URL_RE.search(text):
            leaks.add("url")
        if HANDLE_RE.search(text):
            leaks.add("handle")
        if LONG_ID_RE.search(text):
            leaks.add("long_id")
        if YANDEX_TERMS.search(text):
            leaks.add("yandex")
    return sorted(leaks)


def _sanitize_example_shape(raw: dict[str, Any]) -> dict[str, Any] | None:
    context_raw = raw.get("context")
    if not isinstance(context_raw, list):
        return None

    context: list[dict[str, str]] = []
    for item in context_raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        context.append({"role": role, "content": content})
    if not context:
        return None

    good_question = str(raw.get("good_question", "")).strip()
    if not good_question:
        return None
    improved_question = str(raw.get("improved_question", "")).strip() or good_question
    try:
        quality_score = int(raw.get("quality_score", 0))
    except (TypeError, ValueError):
        quality_score = 0
    return {
        "context": context,
        "good_question": good_question,
        "improved_question": improved_question,
        "quality_score": quality_score,
        "technique": str(raw.get("technique", "")).strip().lower(),
    }


def anonymize_with_llm(
    *,
    client: MistralClient,
    model: str,
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    system = (
        "You anonymize interview training examples. Remove personal data and all Yandex references. "
        "Keep semantic meaning. Return only JSON object."
    )
    user_payload = {
        "task": "Sanitize one training example",
        "rules": [
            "Replace personal names with [NAME]",
            "Replace organizations with [ORG]",
            "Replace products/services with [PRODUCT]",
            "Replace locations with [LOCATION]",
            "Remove contact details and IDs",
            "Remove all Yandex-related references",
            "If impossible to sanitize safely, set keep=false",
        ],
        "input_example": record,
        "output_schema": {
            "keep": True,
            "drop_reason": "string if keep=false",
            "example": {
                "context": [{"role": "user|assistant", "content": "text"}],
                "good_question": "text",
                "improved_question": "text",
                "quality_score": 4,
                "technique": "echo|why|example|contrast|impact|reframe|silence",
            },
        },
    }
    payload = client.chat_json(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    keep = bool(payload.get("keep", True))
    if not keep:
        return None, str(payload.get("drop_reason", "llm_drop")).strip() or "llm_drop"

    example_raw = payload.get("example")
    if isinstance(example_raw, dict):
        clean = _sanitize_example_shape(example_raw)
    else:
        clean = _sanitize_example_shape(payload)
    if clean is None:
        return None, "invalid_shape"

    return clean, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anonymize extracted training examples")
    parser.add_argument("--input", required=True, help="Input JSONL with raw examples")
    parser.add_argument("--output", required=True, help="Output JSONL with anonymized examples")
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        help="Model used for anonymization",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Worker count")
    parser.add_argument("--resume", action="store_true", help="Skip examples already in output")
    parser.add_argument(
        "--manifest",
        default="training/artifacts/anonymize_manifest.jsonl",
        help="JSONL manifest path",
    )
    parser.add_argument(
        "--report-out",
        default="training/artifacts/anonymize_report.json",
        help="Summary report JSON path",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional record cap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    report_path = Path(args.report_out).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    records = iter_jsonl(input_path)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise SystemExit("Input JSONL has no records")

    processed_keys: set[str] = set()
    if args.resume and output_path.exists():
        for item in iter_jsonl(output_path):
            processed_keys.add(_record_key(item))

    api_key, api_base = get_mistral_config()
    workers = max(1, int(args.concurrency))
    print(
        f"[anonymize] records={len(records)} model={args.model} workers={workers} resume={args.resume}"
    )

    counters = {"kept": 0, "dropped": 0, "failed": 0, "skipped_existing": 0}
    dropped_reasons: Counter[str] = Counter()
    leak_counter: Counter[str] = Counter()

    def run(record: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        key = _record_key(record)
        if key in processed_keys:
            return None, {
                "stage": "anonymize",
                "status": "skipped_existing",
                "key": key,
                "updated_at": now_iso(),
            }

        client = MistralClient(api_key=api_key, api_base=api_base)
        sanitized, status = anonymize_with_llm(client=client, model=args.model, record=record)
        if sanitized is None:
            return None, {
                "stage": "anonymize",
                "status": "dropped",
                "key": key,
                "reason": status,
                "updated_at": now_iso(),
            }

        merged = dict(record)
        merged.update(sanitized)
        merged = regex_cleanup_example(merged)
        leaks = detect_leaks(merged)
        if leaks:
            return None, {
                "stage": "anonymize",
                "status": "dropped",
                "key": key,
                "reason": "leak_after_cleanup",
                "leaks": leaks,
                "updated_at": now_iso(),
            }

        merged["anonymized_at"] = now_iso()
        return merged, {
            "stage": "anonymize",
            "status": "kept",
            "key": key,
            "updated_at": now_iso(),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run, record): record for record in records}
        for future in as_completed(future_map):
            try:
                clean_record, manifest = future.result()
            except Exception as err:  # noqa: BLE001
                counters["failed"] += 1
                manifest = {
                    "stage": "anonymize",
                    "status": "failed",
                    "error": str(err),
                    "updated_at": now_iso(),
                }
                clean_record = None
            else:
                status = str(manifest.get("status", "failed"))
                if status in counters:
                    counters[status] += 1
                if status == "dropped":
                    reason = str(manifest.get("reason", "unknown_drop"))
                    dropped_reasons[reason] += 1
                    for leak in manifest.get("leaks", []) if isinstance(manifest.get("leaks"), list) else []:
                        leak_counter[str(leak)] += 1

            if clean_record is not None:
                append_jsonl(output_path, clean_record)
            append_jsonl(manifest_path, manifest)

    report = {
        "created_at": now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "stats": counters,
        "drop_reasons": dict(dropped_reasons),
        "leaks": dict(leak_counter),
    }
    write_json(report_path, report)

    print(
        "[anonymize] done "
        f"kept={counters['kept']} dropped={counters['dropped']} "
        f"failed={counters['failed']} skipped={counters['skipped_existing']}"
    )
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
