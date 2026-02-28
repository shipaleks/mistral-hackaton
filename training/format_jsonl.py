from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .common import iter_jsonl, now_iso, write_json
except ImportError:  # pragma: no cover - direct script execution
    from common import iter_jsonl, now_iso, write_json  # type: ignore


TRAINING_PRICE_PER_MILLION_USD = {
    "mistral-large-latest": 8.0,
    "mistral-medium-latest": 2.0,
}


def _estimate_tokens(text: str) -> int:
    # Coarse approximation for budgeting.
    return max(1, math.ceil(len(text) / 4))


def _context_to_user_message(context: list[dict[str, str]]) -> str:
    lines = [
        "Conversation context from a qualitative interview:",
        "",
    ]
    for turn in context:
        role = turn["role"]
        prefix = "Respondent" if role == "user" else "Interviewer"
        lines.append(f"{prefix}: {turn['content']}")
    lines.extend(
        [
            "",
            "Task: produce the next best open-ended probing interview question.",
            "Do not add explanations.",
        ]
    )
    return "\n".join(lines)


def _normalize_record(
    row: dict[str, Any],
    *,
    target_field: str,
    min_quality: int,
) -> dict[str, Any] | None:
    context_raw = row.get("context")
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

    try:
        quality_score = int(row.get("quality_score", 0))
    except (TypeError, ValueError):
        return None
    if quality_score < min_quality:
        return None

    target = str(row.get(target_field, "")).strip()
    if not target:
        return None

    user_text = _context_to_user_message(context)
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": target},
        ]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format anonymized examples for Mistral fine-tuning")
    parser.add_argument("--input", required=True, help="Input JSONL with anonymized examples")
    parser.add_argument("--output", required=True, help="Output JSONL for fine-tuning API")
    parser.add_argument(
        "--target-field",
        default="good_question",
        choices=["good_question", "improved_question"],
        help="Field used as assistant target",
    )
    parser.add_argument("--min-quality", type=int, default=4, help="Minimum quality score to include")
    parser.add_argument(
        "--stats-out",
        default="training/artifacts/dataset_stats.json",
        help="Output path for dataset statistics",
    )
    parser.add_argument(
        "--estimate-model",
        default="mistral-large-latest",
        help="Model used for rough training cost estimate",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional input record cap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    stats_path = Path(args.stats_out).expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    rows = iter_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("Input JSONL has no records")

    formatted: list[dict[str, Any]] = []
    dropped = 0
    unique_fingerprints: set[str] = set()

    total_context_chars = 0
    total_target_chars = 0
    total_estimated_tokens = 0

    for row in rows:
        sample = _normalize_record(
            row,
            target_field=args.target_field,
            min_quality=max(1, int(args.min_quality)),
        )
        if sample is None:
            dropped += 1
            continue

        user_text = str(sample["messages"][0]["content"])
        target_text = str(sample["messages"][1]["content"])
        fingerprint = f"{user_text}\n---\n{target_text}"
        if fingerprint in unique_fingerprints:
            continue
        unique_fingerprints.add(fingerprint)

        total_context_chars += len(user_text)
        total_target_chars += len(target_text)
        total_estimated_tokens += _estimate_tokens(user_text) + _estimate_tokens(target_text)
        formatted.append(sample)

    if not formatted:
        raise SystemExit("No records passed formatting filters")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in formatted:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")

    training_cost_per_million = TRAINING_PRICE_PER_MILLION_USD.get(args.estimate_model)
    estimated_cost = None
    if training_cost_per_million is not None:
        estimated_cost = (total_estimated_tokens / 1_000_000) * training_cost_per_million

    stats = {
        "created_at": now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "target_field": args.target_field,
        "estimate_model": args.estimate_model,
        "records_input": len(rows),
        "records_output": len(formatted),
        "records_dropped": dropped,
        "records_deduplicated": len(rows) - dropped - len(formatted),
        "chars_context_total": total_context_chars,
        "chars_target_total": total_target_chars,
        "estimated_tokens_total": total_estimated_tokens,
        "estimated_training_cost_usd": estimated_cost,
        "training_price_per_million_usd": training_cost_per_million,
        "token_estimation_method": "ceil(chars/4)",
    }
    write_json(stats_path, stats)
    print(
        "[format_jsonl] done "
        f"input={len(rows)} output={len(formatted)} dropped={dropped} "
        f"estimated_tokens={total_estimated_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
