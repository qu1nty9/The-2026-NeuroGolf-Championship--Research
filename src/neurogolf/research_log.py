from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PATHS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("timestamp_utc", utc_now())
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        f.write("\n")


def append_experiment(record: dict[str, Any], log_dir: str | Path = PATHS.experiments) -> None:
    append_jsonl(Path(log_dir) / "experiments.jsonl", record)


def append_markdown_section(path: str | Path, title: str, body: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{body.strip()}\n")
