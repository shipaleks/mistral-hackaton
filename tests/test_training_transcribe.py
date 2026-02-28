from __future__ import annotations

from pathlib import Path

from training.transcribe import normalize_transcription_response


def test_normalize_transcription_response_with_segments() -> None:
    raw = {
        "text": "Hello there",
        "language": "en",
        "usage": {"prompt_audio_seconds": 12},
        "segments": [
            {"start": 0.0, "end": 1.2, "text": "Hello", "speaker_id": "spk_0"},
            {"start": 1.3, "end": 2.4, "text": "Hi?", "speaker_id": "spk_1"},
        ],
    }
    result = normalize_transcription_response(
        raw=raw,
        audio_path=Path("/tmp/sample.m4a"),
        model="voxtral-mini-2602",
        diarize=True,
    )
    assert result["model"] == "voxtral-mini-2602"
    assert result["diarize"] is True
    assert result["language"] == "en"
    assert len(result["segments"]) == 2
    assert result["speaker_ids"] == ["spk_0", "spk_1"]


def test_normalize_transcription_response_without_segments() -> None:
    raw = {"text": "Fallback text"}
    result = normalize_transcription_response(
        raw=raw,
        audio_path=Path("/tmp/sample2.m4a"),
        model="voxtral-mini-2602",
        diarize=True,
    )
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "Fallback text"
    assert result["segments"][0]["speaker_id"] is None
