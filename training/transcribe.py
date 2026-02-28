from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
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


def _probe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(audio_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip() or "0")


def _split_audio_chunks(audio_path: Path, chunk_seconds: int) -> tuple[list[tuple[Path, float]], Path | None]:
    duration = _probe_duration_seconds(audio_path)
    if duration <= 0:
        return [(audio_path, 0.0)], None

    chunk_paths: list[tuple[Path, float]] = []
    temp_dir = tempfile.mkdtemp(prefix="voxtral_chunks_")
    start = 0.0
    index = 0
    while start < duration:
        out_path = Path(temp_dir) / f"{audio_path.stem}.chunk{index:03d}{audio_path.suffix}"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(chunk_seconds),
            "-i",
            str(audio_path),
            "-c",
            "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        chunk_paths.append((out_path, start))
        start += chunk_seconds
        index += 1
    return chunk_paths, Path(temp_dir)


def _merge_chunked_transcriptions(
    *,
    chunks: list[tuple[dict[str, Any], float, int]],
    audio_path: Path,
    model: str,
    diarize: bool,
) -> dict[str, Any]:
    merged_segments: list[dict[str, Any]] = []
    texts: list[str] = []
    speaker_ids: set[str] = set()
    language: str | None = None
    prompt_audio_seconds = 0
    segment_index = 0

    for normalized, offset, chunk_idx in chunks:
        chunk_text = str(normalized.get("text", "")).strip()
        if chunk_text:
            texts.append(chunk_text)
        if language is None:
            language = normalized.get("language")

        usage = normalized.get("usage")
        if isinstance(usage, dict):
            value = usage.get("prompt_audio_seconds")
            if isinstance(value, int):
                prompt_audio_seconds += value

        for segment in normalized.get("segments", []):
            if not isinstance(segment, dict):
                continue
            try:
                start = float(segment.get("start", 0.0)) + offset
            except (TypeError, ValueError):
                start = offset
            try:
                end = float(segment.get("end", start)) + offset
            except (TypeError, ValueError):
                end = start

            source_speaker = segment.get("speaker_id")
            if source_speaker is None:
                speaker = None
            else:
                speaker = f"chunk{chunk_idx}:{str(source_speaker).strip()}"
                speaker_ids.add(speaker)

            merged_segments.append(
                {
                    "index": segment_index,
                    "start": start,
                    "end": end,
                    "text": str(segment.get("text", "")).strip(),
                    "speaker_id": speaker,
                    "score": segment.get("score"),
                }
            )
            segment_index += 1

    return {
        "audio_path": str(audio_path.resolve()),
        "audio_filename": audio_path.name,
        "model": model,
        "diarize": diarize,
        "language": language,
        "text": "\n".join(texts).strip(),
        "segments": merged_segments,
        "speaker_ids": sorted(speaker_ids),
        "usage": {"prompt_audio_seconds": prompt_audio_seconds},
        "chunked": True,
        "chunk_count": len(chunks),
        "transcribed_at": now_iso(),
    }


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
    max_upload_mb: int,
    chunk_seconds: int,
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

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    force_chunk = file_size_mb > max_upload_mb

    normalized: dict[str, Any]
    raw: dict[str, Any] | list[dict[str, Any]]
    used_chunking = False
    chunk_temp_dir: Path | None = None

    try:
        if force_chunk:
            raise RuntimeError(
                f"File size {file_size_mb:.1f}MB exceeds --max-upload-mb={max_upload_mb}, switching to chunked mode"
            )
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
    except Exception:  # noqa: BLE001
        used_chunking = True
        raw_chunks: list[dict[str, Any]] = []
        normalized_chunks: list[tuple[dict[str, Any], float, int]] = []
        try:
            chunk_paths, chunk_temp_dir = _split_audio_chunks(audio_path, chunk_seconds)
            for chunk_idx, (chunk_path, offset) in enumerate(chunk_paths):
                chunk_raw = client.transcribe(
                    model=model,
                    audio_file=chunk_path,
                    diarize=diarize,
                    timestamp_granularities=timestamp_granularities,
                    language=language,
                )
                raw_chunks.append(
                    {
                        "chunk_index": chunk_idx,
                        "offset_seconds": offset,
                        "raw": chunk_raw,
                    }
                )
                chunk_norm = normalize_transcription_response(
                    raw=chunk_raw,
                    audio_path=chunk_path,
                    model=model,
                    diarize=diarize,
                )
                normalized_chunks.append((chunk_norm, offset, chunk_idx))
        finally:
            if chunk_temp_dir is not None and chunk_temp_dir.exists():
                shutil.rmtree(chunk_temp_dir, ignore_errors=True)

        raw = {"chunked": True, "chunks": raw_chunks}
        normalized = _merge_chunked_transcriptions(
            chunks=normalized_chunks,
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
        "chunked": used_chunking,
        "file_size_mb": round(file_size_mb, 2),
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
        "--max-upload-mb",
        type=int,
        default=20,
        help="If file exceeds this size, use chunked transcription fallback (default: 20)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=900,
        help="Chunk size in seconds for fallback mode (default: 900)",
    )
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
            max_upload_mb=max(1, int(args.max_upload_mb)),
            chunk_seconds=max(60, int(args.chunk_seconds)),
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
