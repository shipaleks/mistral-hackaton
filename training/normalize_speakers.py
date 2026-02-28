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
        ensure_parent,
        get_mistral_config,
        now_iso,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        MistralClient,
        append_jsonl,
        ensure_parent,
        get_mistral_config,
        now_iso,
        write_json,
    )


ROLE_MODERATOR = "MODERATOR"
ROLE_INTERVIEWEE = "INTERVIEWEE"
VALID_ROLES = {ROLE_MODERATOR, ROLE_INTERVIEWEE}


def question_ratio(texts: list[str]) -> float:
    if not texts:
        return 0.0
    question_count = sum(1 for text in texts if "?" in text)
    return question_count / max(1, len(texts))


def _speaker_stats(segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for index, segment in enumerate(segments):
        speaker = str(segment.get("speaker_id") or "unknown")
        text = str(segment.get("text", "")).strip()
        entry = stats.setdefault(
            speaker,
            {"speaker_id": speaker, "turns": 0, "chars": 0, "question_turns": 0, "first_index": index},
        )
        entry["turns"] += 1
        entry["chars"] += len(text)
        if "?" in text:
            entry["question_turns"] += 1
    for value in stats.values():
        turns = max(1, int(value["turns"]))
        value["question_ratio"] = float(value["question_turns"]) / turns
    return stats


def heuristic_mapping(segments: list[dict[str, Any]]) -> dict[str, str]:
    stats = _speaker_stats(segments)
    speakers = list(stats)
    if not speakers:
        return {}

    moderator = max(
        speakers,
        key=lambda sid: (
            stats[sid]["question_ratio"],
            stats[sid]["question_turns"],
            -int(stats[sid]["first_index"]),
        ),
    )
    mapping = {speaker: ROLE_INTERVIEWEE for speaker in speakers}
    mapping[moderator] = ROLE_MODERATOR
    return mapping


def apply_role_mapping(
    segments: list[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        speaker_id = str(segment.get("speaker_id") or "unknown")
        role = mapping.get(speaker_id, ROLE_INTERVIEWEE)
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        try:
            start = float(segment.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        try:
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            end = start

        normalized.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "speaker_id": speaker_id,
                "role": role,
                "text": text,
            }
        )
    return normalized


def role_stats(segments: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[str]] = {ROLE_MODERATOR: [], ROLE_INTERVIEWEE: []}
    for segment in segments:
        role = str(segment.get("role", ROLE_INTERVIEWEE))
        if role not in grouped:
            continue
        grouped[role].append(str(segment.get("text", "")))

    stats: dict[str, dict[str, float | int]] = {}
    for role, texts in grouped.items():
        turns = len(texts)
        chars = sum(len(text) for text in texts)
        questions = sum(1 for text in texts if "?" in text)
        stats[role] = {
            "turns": turns,
            "chars": chars,
            "question_turns": questions,
            "question_ratio": (questions / turns) if turns else 0.0,
        }
    return stats


def gate_two_roles(segments: list[dict[str, Any]]) -> tuple[bool, list[str], dict[str, dict[str, float | int]]]:
    stats = role_stats(segments)
    reasons: list[str] = []
    for role in (ROLE_MODERATOR, ROLE_INTERVIEWEE):
        turns = int(stats[role]["turns"])
        chars = int(stats[role]["chars"])
        if turns == 0:
            reasons.append(f"{role.lower()}_has_no_turns")
        if chars == 0:
            reasons.append(f"{role.lower()}_has_no_chars")

    moderator_q = float(stats[ROLE_MODERATOR]["question_ratio"])
    interviewee_q = float(stats[ROLE_INTERVIEWEE]["question_ratio"])
    if moderator_q <= interviewee_q:
        reasons.append("moderator_question_ratio_not_higher")

    return len(reasons) == 0, reasons, stats


def _llm_mapping_prompt_payload(
    *, segments: list[dict[str, Any]], max_segments: int
) -> dict[str, Any]:
    stats = _speaker_stats(segments)
    sample = [
        {
            "speaker_id": str(item.get("speaker_id") or "unknown"),
            "text": str(item.get("text", ""))[:220],
            "start": item.get("start"),
            "end": item.get("end"),
        }
        for item in segments[:max_segments]
    ]
    return {"speaker_stats": stats, "sample_segments": sample}


def llm_mapping(
    *,
    client: MistralClient,
    model: str,
    segments: list[dict[str, Any]],
    max_segments: int,
) -> tuple[dict[str, str], str]:
    payload = _llm_mapping_prompt_payload(segments=segments, max_segments=max_segments)
    system_prompt = (
        "You map diarized speakers in an interview to exactly two roles: "
        "MODERATOR and INTERVIEWEE. Return only JSON object."
    )
    user_prompt = {
        "task": "Map every speaker_id to one of MODERATOR or INTERVIEWEE",
        "constraint": "Exactly one speaker should be MODERATOR when possible.",
        "data": payload,
        "output_schema": {"mapping": {"speaker_id": "MODERATOR|INTERVIEWEE"}},
    }

    response = client.chat_json(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    mapping_raw = response.get("mapping")
    mapping: dict[str, str] = {}
    if isinstance(mapping_raw, dict):
        for speaker_id, role in mapping_raw.items():
            speaker_key = str(speaker_id)
            role_name = str(role).upper()
            if role_name not in VALID_ROLES:
                continue
            mapping[speaker_key] = role_name
    return mapping, "llm"


def _sanitize_mapping(mapping: dict[str, str], segments: list[dict[str, Any]]) -> dict[str, str]:
    speakers = sorted({str(seg.get("speaker_id") or "unknown") for seg in segments})
    if not speakers:
        return {}

    sanitized = {speaker: mapping.get(speaker, ROLE_INTERVIEWEE) for speaker in speakers}
    moderators = [speaker for speaker, role in sanitized.items() if role == ROLE_MODERATOR]
    if len(moderators) == 1:
        return sanitized
    fallback = heuristic_mapping(segments)
    return {speaker: fallback.get(speaker, ROLE_INTERVIEWEE) for speaker in speakers}


def normalize_transcript_entry(
    *,
    transcript: dict[str, Any],
    client: MistralClient,
    model: str,
    max_segments_for_llm: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_segments = transcript.get("segments")
    segments: list[dict[str, Any]]
    if isinstance(raw_segments, list):
        segments = [item for item in raw_segments if isinstance(item, dict)]
    else:
        segments = []

    speaker_ids = sorted({str(seg.get("speaker_id") or "unknown") for seg in segments})
    method = "heuristic"

    if len(speaker_ids) <= 2:
        mapping = heuristic_mapping(segments)
    else:
        try:
            mapping_llm, method = llm_mapping(
                client=client,
                model=model,
                segments=segments,
                max_segments=max_segments_for_llm,
            )
            mapping = _sanitize_mapping(mapping_llm, segments)
        except Exception:  # noqa: BLE001
            method = "heuristic_fallback"
            mapping = heuristic_mapping(segments)

    mapped_segments = apply_role_mapping(segments, mapping)
    gate_ok, gate_reasons, stats = gate_two_roles(mapped_segments)
    if len(speaker_ids) > 2 and gate_ok:
        status = "auto_merged"
    elif len(speaker_ids) > 2 and not gate_ok:
        status = "auto_merged_low_confidence"
    elif not gate_ok:
        status = "low_confidence"
    else:
        status = "success"

    normalized_payload = dict(transcript)
    normalized_payload["segments"] = mapped_segments
    normalized_payload["speaker_role_mapping"] = mapping
    normalized_payload["role_stats"] = stats
    normalized_payload["normalization"] = {
        "input_speaker_ids": speaker_ids,
        "method": method,
        "gate_passed": gate_ok,
        "gate_reasons": gate_reasons,
        "status": status,
        "normalized_at": now_iso(),
    }

    manifest = {
        "stage": "normalize_speakers",
        "status": status,
        "input_speakers": len(speaker_ids),
        "output_roles": len([role for role in stats if int(stats[role]["turns"]) > 0]),
        "gate_passed": gate_ok,
        "gate_reasons": gate_reasons,
        "method": method,
        "updated_at": now_iso(),
    }
    return normalized_payload, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize diarized speakers to two interview roles")
    parser.add_argument("--input-dir", required=True, help="Directory with transcription JSON files")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for normalized two-role transcripts",
    )
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        help="Model for auto-merge mapping when >2 speakers",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Worker count")
    parser.add_argument("--resume", action="store_true", help="Skip existing outputs")
    parser.add_argument(
        "--manifest",
        default="training/artifacts/normalize_speakers_manifest.jsonl",
        help="JSONL manifest path",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional file cap")
    parser.add_argument(
        "--max-segments-for-llm",
        type=int,
        default=120,
        help="Max segment count sent to LLM for mapping",
    )
    return parser.parse_args()


def _iter_transcript_files(input_dir: Path) -> list[Path]:
    files = [path for path in input_dir.rglob("*.json") if path.is_file()]
    return sorted(path for path in files if "/raw/" not in str(path))


def _output_path(input_dir: Path, output_dir: Path, input_path: Path) -> Path:
    return output_dir / input_path.relative_to(input_dir)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    transcript_files = _iter_transcript_files(input_dir)
    if args.limit > 0:
        transcript_files = transcript_files[: args.limit]
    if not transcript_files:
        raise SystemExit("No transcription JSON files found")

    api_key, api_base = get_mistral_config()
    workers = max(1, int(args.concurrency))
    counters = {
        "success": 0,
        "auto_merged": 0,
        "auto_merged_low_confidence": 0,
        "low_confidence": 0,
        "failed": 0,
        "skipped_existing": 0,
    }

    print(
        f"[normalize_speakers] files={len(transcript_files)} model={args.model} "
        f"workers={workers} resume={args.resume}"
    )

    def run(path: Path) -> dict[str, Any]:
        out_path = _output_path(input_dir, output_dir, path)
        if args.resume and out_path.exists():
            return {
                "stage": "normalize_speakers",
                "status": "skipped_existing",
                "input_path": str(path.resolve()),
                "output_path": str(out_path.resolve()),
                "updated_at": now_iso(),
            }

        transcript = json.loads(path.read_text(encoding="utf-8"))
        client = MistralClient(api_key=api_key, api_base=api_base)
        normalized, record = normalize_transcript_entry(
            transcript=transcript,
            client=client,
            model=args.model,
            max_segments_for_llm=max(20, int(args.max_segments_for_llm)),
        )
        ensure_parent(out_path)
        write_json(out_path, normalized)
        record["input_path"] = str(path.resolve())
        record["output_path"] = str(out_path.resolve())
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run, path): path for path in transcript_files}
        for future in as_completed(future_map):
            path = future_map[future]
            try:
                record = future.result()
            except Exception as err:  # noqa: BLE001
                record = {
                    "stage": "normalize_speakers",
                    "status": "failed",
                    "input_path": str(path.resolve()),
                    "error": str(err),
                    "updated_at": now_iso(),
                }
                counters["failed"] += 1
                print(f"[normalize_speakers] failed: {path.name} -> {err}")
            else:
                status = str(record.get("status", "failed"))
                if status not in counters:
                    counters[status] = 0
                counters[status] += 1
                print(f"[normalize_speakers] {status}: {path.name}")
            append_jsonl(manifest_path, record)

    print(
        "[normalize_speakers] done "
        f"success={counters.get('success', 0)} "
        f"auto_merged={counters.get('auto_merged', 0)} "
        f"low_conf={counters.get('auto_merged_low_confidence', 0) + counters.get('low_confidence', 0)} "
        f"failed={counters.get('failed', 0)} "
        f"skipped={counters.get('skipped_existing', 0)}"
    )
    return 0 if counters.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

