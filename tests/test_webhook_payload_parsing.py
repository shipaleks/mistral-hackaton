from __future__ import annotations

from api.routes_webhook import _extract_conversation_payload, _extract_text


def test_extract_text_handles_role_message_items() -> None:
    transcript_payload = [
        {"role": "agent", "message": "Hello!"},
        {"role": "user", "message": "Hi, I loved the mentoring."},
    ]

    text = _extract_text(transcript_payload)

    assert "agent: Hello!" in text
    assert "user: Hi, I loved the mentoring." in text


def test_extract_conversation_payload_reads_post_call_shape() -> None:
    payload = {
        "type": "post_call_transcription",
        "data": {
            "conversation_id": "conv_123",
            "agent_id": "agent_abc",
            "conversation_end_reason": "call_ended_by_assistant",
            "transcript": [
                {"role": "agent", "message": "How was the hackathon?"},
                {"role": "user", "message": "Very intense, but useful."},
            ],
            "conversation_initiation_client_data": {
                "dynamic_variables": {"project_id": "hackathon-20260228"}
            },
        },
    }

    conversation_id, transcript, metadata = _extract_conversation_payload(payload)

    assert conversation_id == "conv_123"
    assert "How was the hackathon?" in transcript
    assert "Very intense, but useful." in transcript
    assert metadata["event"] == "post_call_transcription"
    assert metadata["agent_id"] == "agent_abc"
    assert metadata["project_id_hint"] == "hackathon-20260228"
    assert metadata["conversation_end_reason"] == "call_ended_by_assistant"

