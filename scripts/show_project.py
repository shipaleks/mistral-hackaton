#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Show Eidetic project state")
    parser.add_argument("project_id", help="Project identifier")
    parser.add_argument(
        "--data-dir",
        default="./data/projects",
        help="Projects data directory (default: ./data/projects)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full JSON instead of summary",
    )
    args = parser.parse_args()

    project_path = Path(args.data_dir) / args.project_id / "project.json"
    if not project_path.exists():
        raise SystemExit(f"Project file not found: {project_path}")

    payload = json.loads(project_path.read_text(encoding="utf-8"))
    if args.full:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    summary = {
        "id": payload.get("id"),
        "research_question": payload.get("research_question"),
        "interviews": len(payload.get("interview_store", [])),
        "evidence": len(payload.get("evidence_store", [])),
        "propositions": len(payload.get("proposition_store", [])),
        "script_versions": len(payload.get("script_versions", [])),
        "mode": payload.get("metrics", {}).get("mode"),
        "sync_pending": payload.get("sync_pending"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
