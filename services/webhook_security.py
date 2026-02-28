from __future__ import annotations

import hashlib
import hmac
import time


def _parse_signature_header(signature_header: str) -> tuple[str | None, str | None]:
    timestamp: str | None = None
    signature: str | None = None
    for part in signature_header.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "t":
            timestamp = value
        elif key == "v0":
            signature = value
    return timestamp, signature


def verify_elevenlabs_signature(
    raw_body: bytes,
    signature_header: str | None,
    secret: str,
    tolerance_seconds: int = 300,
    now_ts: int | None = None,
) -> bool:
    if not secret:
        return True
    if not signature_header:
        return False

    timestamp, signature = _parse_signature_header(signature_header)
    if not timestamp or not signature:
        return False

    try:
        ts_int = int(timestamp)
    except ValueError:
        return False

    current = now_ts if now_ts is not None else int(time.time())
    if abs(current - ts_int) > tolerance_seconds:
        return False

    payload = f"{timestamp}.{raw_body.decode('utf-8')}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)
