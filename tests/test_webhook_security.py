from __future__ import annotations

import hashlib
import hmac

from services.webhook_security import verify_elevenlabs_signature


def _sign(secret: str, timestamp: int, raw: bytes) -> str:
    payload = f"{timestamp}.{raw.decode('utf-8')}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"t={timestamp},v0={digest}"


def test_verify_signature_valid() -> None:
    body = b'{"hello":"world"}'
    secret = "top-secret"
    timestamp = 1_700_000_000
    header = _sign(secret, timestamp, body)

    assert verify_elevenlabs_signature(
        raw_body=body,
        signature_header=header,
        secret=secret,
        tolerance_seconds=300,
        now_ts=timestamp + 60,
    )


def test_verify_signature_invalid_digest() -> None:
    body = b'{"hello":"world"}'
    secret = "top-secret"
    timestamp = 1_700_000_000
    header = f"t={timestamp},v0=deadbeef"

    assert not verify_elevenlabs_signature(
        raw_body=body,
        signature_header=header,
        secret=secret,
        tolerance_seconds=300,
        now_ts=timestamp + 60,
    )


def test_verify_signature_rejects_old_timestamp() -> None:
    body = b'{"hello":"world"}'
    secret = "top-secret"
    timestamp = 1_700_000_000
    header = _sign(secret, timestamp, body)

    assert not verify_elevenlabs_signature(
        raw_body=body,
        signature_header=header,
        secret=secret,
        tolerance_seconds=300,
        now_ts=timestamp + 301,
    )
