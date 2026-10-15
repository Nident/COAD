from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Trajectory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def action_hash(tool: str, arguments: dict[str, Any]) -> str:
        raw = json.dumps([tool, arguments], sort_keys=True, default=str, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def append(self, event: str, payload: dict[str, Any]) -> None:
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

