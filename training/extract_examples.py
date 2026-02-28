from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .common import (
        MistralClient,
        append_jsonl,
        get_mistral_config,
        iter_jsonl,
        json_from_text,
        now_iso,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        MistralClient,
        append_jsonl,
        get_mistral_config,
        iter_jsonl,
        json_from_text,
        now_iso,
    )


def _iter_transcripts(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.rglob("*.json") if path.is_file()]
    return sorted(path for path in files if "/raw/" not in str(path))


def _to_training_turns(segments: list[dict[str, Any]], max_turns: int) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for segment in segments:
        role = str(segment.get("role", "")).upper()
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        if role == "MODERATOR":
            turns.append({"role": "assistant", "content": text})
        elif role == "INTERVIEWEE":
            turns.append({"role": "user", "content": text})
    if max_turns > 0:
        return turns[:max_turns]
    return turns


def _sanitize_example(raw: dict[str, Any], *, min_quality: int) -> dict[str, Any] | None:
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
    improved_question = str(raw.get("improved_question", "")).strip() or good_question
    if not good_question:
        return None

    try:
        quality_score = int(raw.get("quality_score", 0))
    except (TypeError, ValueError):
        return None
    if quality_score < min_quality:
        return None

    technique = str(raw.get("technique", "")).strip().lower()
    return {
        "context": context,
        "good_question": good_question,
        "improved_question": improved_question,
        "quality_score": quality_score,
        "technique": technique,
    }


def _find_turn_index(turns: list[dict[str, str]], question: str) -> int:
    normalized = question.strip().lower()
    if not normalized:
        return -1
    for index, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        content = turn["content"].strip().lower()
        if content == normalized:
            return index
    for index, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        content = turn["content"].strip().lower()
        if normalized in content or content in normalized:
            return index
    return -1


def _extract_with_model(
    *,
    client: MistralClient,
    model: str,
    system_prompt: str,
    source_file: Path,
    transcript: dict[str, Any],
    min_quality: int,
    max_turns: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_segments = transcript.get("segments")
    segments = [item for item in raw_segments if isinstance(item, dict)] if isinstance(raw_segments, list) else []
    turns = _to_training_turns(segments, max_turns=max_turns)

    user_payload = {
        "source_file": str(source_file),
        "language": transcript.get("language"),
        "conversation_turns": turns,
        "instructions": {
            "output": {"examples": "array of extracted training examples"},
            "min_quality_score": min_quality,
        },
    }

    raw_text = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.1,
        max_tokens=8192,
    )
    parsed = json_from_text(raw_text)
    if isinstance(parsed, dict):
        items = parsed.get("examples", [])
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    clean_examples: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        clean = _sanitize_example(item, min_quality=min_quality)
        if clean is None:
            continue

        clean["source_file"] = str(source_file.resolve())
        clean["language"] = transcript.get("language")
        clean["speaker_role"] = "MODERATOR"
        clean["turn_index"] = _find_turn_index(turns, clean["good_question"])
        clean["extracted_at"] = now_iso()
        clean_examples.append(clean)

    manifest = {
        "stage": "extract_examples",
        "status": "success",
        "source_file": str(source_file.resolve()),
        "turns": len(turns),
        "examples": len(clean_examples),
        "updated_at": now_iso(),
    }
    return clean_examples, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract high-quality interviewer examples from transcripts")
    parser.add_argument("--input-dir", required=True, help="Directory with normalized transcript JSON files")
    parser.add_argument("--output", required=True, help="Output JSONL path for extracted examples")
    parser.add_argument(
        "--prompt-file",
        default="training_extraction.txt",
        help="Prompt filename from prompts/ (default: training_extraction.txt)",
    )
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        help="Model used for extraction",
    )
    parser.add_argument("--min-quality", type=int, default=4, help="Minimum quality score threshold")
    parser.add_argument("--concurrency", type=int, default=4, help="Worker count")
    parser.add_argument("--resume", action="store_true", help="Skip files already present in output JSONL")
    parser.add_argument(
        "--manifest",
        default="training/artifacts/extract_examples_manifest.jsonl",
        help="JSONL manifest path",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional file cap")
    parser.add_argument("--max-turns", type=int, default=500, help="Max turns sent to model per transcript")
    return parser.parse_args()


def _load_prompt(prompt_arg: str) -> str:
    candidate = Path(prompt_arg)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    fallback = prompts_dir / prompt_arg
    if fallback.exists():
        return fallback.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"Prompt file not found: {prompt_arg}")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    transcripts = _iter_transcripts(input_dir)
    if args.limit > 0:
        transcripts = transcripts[: args.limit]
    if not transcripts:
        raise SystemExit("No transcript files found")

    prompt = _load_prompt(args.prompt_file)
    api_key, api_base = get_mistral_config()
    workers = max(1, int(args.concurrency))

    skip_sources: set[str] = set()
    if args.resume and output_path.exists():
        for row in iter_jsonl(output_path):
            source = str(row.get("source_file", "")).strip()
            if source:
                skip_sources.add(source)

    print(
        f"[extract_examples] transcripts={len(transcripts)} model={args.model} "
        f"workers={workers} min_quality={args.min_quality} resume={args.resume}"
    )

    counters = {"success": 0, "failed": 0, "skipped_existing": 0, "examples": 0}

    def run(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if str(path.resolve()) in skip_sources:
            return [], {
                "stage": "extract_examples",
                "status": "skipped_existing",
                "source_file": str(path.resolve()),
                "updated_at": now_iso(),
            }
        transcript = json.loads(path.read_text(encoding="utf-8"))
        client = MistralClient(api_key=api_key, api_base=api_base)
        return _extract_with_model(
            client=client,
            model=args.model,
            system_prompt=prompt,
            source_file=path,
            transcript=transcript,
            min_quality=max(1, int(args.min_quality)),
            max_turns=max(50, int(args.max_turns)),
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run, path): path for path in transcripts}
        for future in as_completed(future_map):
            path = future_map[future]
            try:
                examples, record = future.result()
            except Exception as err:  # noqa: BLE001
                counters["failed"] += 1
                record = {
                    "stage": "extract_examples",
                    "status": "failed",
                    "source_file": str(path.resolve()),
                    "error": str(err),
                    "updated_at": now_iso(),
                }
                print(f"[extract_examples] failed: {path.name} -> {err}")
                examples = []
            else:
                status = str(record.get("status", "success"))
                if status == "skipped_existing":
                    counters["skipped_existing"] += 1
                else:
                    counters["success"] += 1
                print(f"[extract_examples] {status}: {path.name} examples={len(examples)}")

            for item in examples:
                append_jsonl(output_path, item)
            counters["examples"] += len(examples)
            append_jsonl(manifest_path, record)

    print(
        "[extract_examples] done "
        f"success={counters['success']} skipped={counters['skipped_existing']} "
        f"failed={counters['failed']} examples={counters['examples']}"
    )
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
