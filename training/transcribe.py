from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from .common import (
        MistralClient,
        append_jsonl,
        collect_audio_files,
        ensure_parent,
        get_mistral_config,
        now_iso,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (  # type: ignore
        MistralClient,
        append_jsonl,
        collect_audio_files,
        ensure_parent,
        get_mistral_config,
        now_iso,
        write_json,
    )


def normalize_transcription_response(
    *, raw: dict[str, Any], audio_path: Path, model: str, diarize: bool
) -> dict[str, Any]:
    segments_payload = raw.get("segments")
    segments: list[dict[str, Any]] = []

    if isinstance(segments_payload, list):
        for index, item in enumerate(segments_payload):
            if not isinstance(item, dict):
                continue
            try:
                start = float(item.get("start", 0.0))
            except (TypeError, ValueError):
                start = 0.0
            try:
                end = float(item.get("end", start))
            except (TypeError, ValueError):
                end = start

            speaker = item.get("speaker_id")
            speaker_id = None if speaker is None else str(speaker).strip() or None
            segments.append(
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "text": str(item.get("text", "")).strip(),
                    "speaker_id": speaker_id,
                    "score": item.get("score"),
                }
            )

    text = str(raw.get("text", "")).strip()
    if not segments and text:
        segments = [
            {
                "index": 0,
                "start": 0.0,
                "end": 0.0,
                "text": text,
                "speaker_id": None,
                "score": None,
            }
        ]

    usage = raw.get("usage")
    usage_payload = usage if isinstance(usage, dict) else {}

    speaker_ids = sorted({seg["speaker_id"] for seg in segments if seg.get("speaker_id")})
    language = str(raw.get("language") or raw.get("audio_language") or "").strip() or None

    return {
        "audio_path": str(audio_path.resolve()),
        "audio_filename": audio_path.name,
        "model": model,
        "diarize": diarize,
        "language": language,
        "text": text,
        "segments": segments,
        "speaker_ids": speaker_ids,
        "usage": usage_payload,
        "transcribed_at": now_iso(),
    }


def _output_paths(input_dir: Path, output_dir: Path, audio_path: Path) -> tuple[Path, Path]:
    relative = audio_path.relative_to(input_dir)
    normalized_path = output_dir / relative.with_suffix(".json")
    raw_path = output_dir / "raw" / relative.with_suffix(".raw.json")
    return normalized_path, raw_path


def _transcribe_one(
    *,
    client: MistralClient,
    input_dir: Path,
    output_dir: Path,
    audio_path: Path,
    model: str,
    diarize: bool,
    language: str | None,
    resume: bool,
    timestamp_granularities: list[str],
) -> dict[str, Any]:
    normalized_path, raw_path = _output_paths(input_dir, output_dir, audio_path)
    if resume and normalized_path.exists() and raw_path.exists():
        return {
            "stage": "transcribe",
            "status": "skipped_existing",
            "audio_path": str(audio_path.resolve()),
            "output_path": str(normalized_path.resolve()),
            "raw_output_path": str(raw_path.resolve()),
            "model": model,
            "updated_at": now_iso(),
        }

    raw = client.transcribe(
        model=model,
        audio_file=audio_path,
        diarize=diarize,
        timestamp_granularities=timestamp_granularities,
        language=language,
    )
    normalized = normalize_transcription_response(
        raw=raw,
        audio_path=audio_path,
        model=model,
        diarize=diarize,
    )

    ensure_parent(normalized_path)
    ensure_parent(raw_path)
    write_json(normalized_path, normalized)
    write_json(raw_path, raw)
    return {
        "stage": "transcribe",
        "status": "success",
        "audio_path": str(audio_path.resolve()),
        "output_path": str(normalized_path.resolve()),
        "raw_output_path": str(raw_path.resolve()),
        "model": model,
        "language": normalized.get("language"),
        "segments": len(normalized.get("segments", [])),
        "updated_at": now_iso(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe interview audio with Voxtral")
    parser.add_argument("--input-dir", required=True, help="Directory with audio files")
    parser.add_argument("--output-dir", required=True, help="Directory for transcription outputs")
    parser.add_argument(
        "--model",
        default="voxtral-mini-2602",
        help="Audio transcription model (default: voxtral-mini-2602)",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Worker count (default: 4)")
    parser.add_argument("--resume", action="store_true", help="Skip already transcribed files")
    parser.add_argument(
        "--manifest",
        default="training/artifacts/transcribe_manifest.jsonl",
        help="Path to JSONL manifest",
    )
    parser.add_argument("--language", default="", help="Optional fixed language (e.g. en)")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of files")
    parser.add_argument(
        "--timestamp-granularity",
        action="append",
        dest="timestamp_granularities",
        default=[],
        help="Timestamp granularity values (repeatable, default: segment)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable diarization (enabled by default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    files = collect_audio_files(input_dir)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No audio files found in input directory")

    diarize = not args.no_diarize
    timestamp_granularities = args.timestamp_granularities or ["segment"]
    language = args.language.strip() or None

    api_key, api_base = get_mistral_config()
    workers = max(1, int(args.concurrency))
    print(
        f"[transcribe] files={len(files)} model={args.model} diarize={diarize} "
        f"workers={workers} resume={args.resume}"
    )

    counters = {"success": 0, "skipped_existing": 0, "failed": 0}

    def run(audio_path: Path) -> dict[str, Any]:
        client = MistralClient(api_key=api_key, api_base=api_base)
        return _transcribe_one(
            client=client,
            input_dir=input_dir,
            output_dir=output_dir,
            audio_path=audio_path,
            model=args.model,
            diarize=diarize,
            language=language,
            resume=args.resume,
            timestamp_granularities=timestamp_granularities,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run, audio_path): audio_path for audio_path in files}
        for future in as_completed(future_map):
            audio_path = future_map[future]
            try:
                record = future.result()
            except Exception as err:  # noqa: BLE001
                counters["failed"] += 1
                record = {
                    "stage": "transcribe",
                    "status": "failed",
                    "audio_path": str(audio_path.resolve()),
                    "model": args.model,
                    "error": str(err),
                    "updated_at": now_iso(),
                }
                print(f"[transcribe] failed: {audio_path.name} -> {err}")
            else:
                status = str(record.get("status", "unknown"))
                if status in counters:
                    counters[status] += 1
                print(f"[transcribe] {status}: {audio_path.name}")

            append_jsonl(manifest_path, record)

    print(
        "[transcribe] done "
        f"success={counters['success']} skipped={counters['skipped_existing']} failed={counters['failed']}"
    )
    return 0 if counters["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
