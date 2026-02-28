from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

try:
    from .common import MistralClient, get_mistral_config, now_iso, write_json
except ImportError:  # pragma: no cover - direct script execution
    from common import MistralClient, get_mistral_config, now_iso, write_json  # type: ignore


FINAL_STATUSES = {"SUCCESS", "FAILED", "CANCELLED", "FAILED_VALIDATION"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Mistral fine-tuning job")
    parser.add_argument("--data", required=True, help="Path to fine-tune JSONL data file")
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        help="Base model for fine-tuning (default: mistral-large-latest)",
    )
    parser.add_argument("--name", default="", help="Optional suffix for fine-tuned model")
    parser.add_argument("--training-steps", type=int, default=100, help="Training steps")
    parser.add_argument("--learning-rate", type=float, default=0.0, help="Optional learning rate")
    parser.add_argument("--epochs", type=float, default=0.0, help="Optional epochs")
    parser.add_argument("--seq-len", type=int, default=0, help="Optional sequence length")
    parser.add_argument("--fim-ratio", type=float, default=-1.0, help="Optional FIM ratio")
    parser.add_argument("--dry-run", action="store_true", help="Only validate job metadata")
    parser.add_argument("--skip-start", action="store_true", help="Do not call /start after create")
    parser.add_argument("--poll", action="store_true", help="Poll job status until terminal state")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval seconds")
    parser.add_argument("--timeout-minutes", type=int, default=240, help="Polling timeout in minutes")
    parser.add_argument(
        "--output",
        default="training/artifacts/finetune_job.json",
        help="Output JSON artifact path",
    )
    return parser.parse_args()


def _build_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {"training_steps": max(1, int(args.training_steps))}
    if args.learning_rate > 0:
        params["learning_rate"] = float(args.learning_rate)
    if args.epochs > 0:
        params["epochs"] = float(args.epochs)
    if args.seq_len > 0:
        params["seq_len"] = int(args.seq_len)
    if args.fim_ratio >= 0:
        params["fim_ratio"] = float(args.fim_ratio)
    return params


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not data_path.exists():
        raise SystemExit(f"Data file does not exist: {data_path}")

    api_key, api_base = get_mistral_config()
    client = MistralClient(api_key=api_key, api_base=api_base, timeout_seconds=180.0)
    artifact: dict[str, Any] = {
        "created_at": now_iso(),
        "data_path": str(data_path),
        "base_model": args.model,
    }

    print(f"[finetune] uploading data file: {data_path.name}")
    upload = client.upload_file(file_path=data_path, purpose="fine-tune")
    file_id = str(upload.get("id", "")).strip()
    if not file_id:
        raise SystemExit(f"Upload response missing file id: {upload}")
    artifact["upload"] = upload

    payload: dict[str, Any] = {
        "model": args.model,
        "training_files": [{"file_id": file_id}],
        "hyperparameters": _build_hyperparameters(args),
        "job_type": "completion",
        "auto_start": False,
    }
    if args.name.strip():
        payload["suffix"] = args.name.strip()

    print(f"[finetune] creating job for model={args.model} dry_run={args.dry_run}")
    create = client.create_fine_tuning_job(payload=payload, dry_run=args.dry_run)
    artifact["create"] = create
    artifact["request_payload"] = payload

    job_id = str(create.get("id", "")).strip()
    artifact["job_id"] = job_id or None
    if args.dry_run:
        artifact["status"] = "DRY_RUN_ONLY"
        write_json(output_path, artifact)
        print("[finetune] dry run completed")
        return 0

    if not job_id:
        raise SystemExit(f"Create job response missing id: {create}")

    if args.skip_start:
        print(f"[finetune] skip_start=true; job created id={job_id}")
        artifact["status"] = str(create.get("status", "CREATED"))
        write_json(output_path, artifact)
        return 0

    print(f"[finetune] starting job id={job_id}")
    started = client.start_fine_tuning_job(job_id=job_id)
    artifact["start"] = started
    artifact["status"] = str(started.get("status", "STARTED"))

    if args.poll:
        interval = max(5, int(args.poll_interval))
        timeout_seconds = max(1, int(args.timeout_minutes)) * 60
        deadline = time.time() + timeout_seconds
        polls: list[dict[str, Any]] = []
        print(f"[finetune] polling every {interval}s up to {args.timeout_minutes}m")
        while True:
            snapshot = client.get_fine_tuning_job(job_id=job_id)
            status = str(snapshot.get("status", "")).upper()
            polls.append(
                {
                    "at": now_iso(),
                    "status": status,
                    "fine_tuned_model": snapshot.get("fine_tuned_model"),
                    "trained_tokens": snapshot.get("trained_tokens"),
                    "metadata": snapshot.get("metadata"),
                }
            )
            if status in FINAL_STATUSES:
                artifact["final_job"] = snapshot
                artifact["status"] = status
                break
            if time.time() >= deadline:
                artifact["status"] = "TIMEOUT"
                artifact["final_job"] = snapshot
                break
            time.sleep(interval)
        artifact["polls"] = polls

    write_json(output_path, artifact)
    print(
        f"[finetune] done job_id={job_id} status={artifact.get('status')} "
        f"fine_tuned_model={artifact.get('final_job', {}).get('fine_tuned_model') if isinstance(artifact.get('final_job'), dict) else None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
