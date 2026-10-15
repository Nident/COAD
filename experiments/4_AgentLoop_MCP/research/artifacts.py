from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactStore:
    def __init__(self, root: Path, run_id: str, dataset_version: str = "unknown") -> None:
        self.root = root / run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_version = dataset_version

    def _id(self, tool: str, arguments: dict[str, Any]) -> str:
        raw = json.dumps([tool, arguments], sort_keys=True, default=str).encode()
        return "artifact_" + hashlib.sha256(raw).hexdigest()[:18]

    def table(self, tool: str, frame: pd.DataFrame, arguments: dict[str, Any], split_scope: str, *, artifact_id: str | None = None) -> str:
        artifact_id = artifact_id or self._id(tool, arguments)
        path = self.root / f"{artifact_id}.parquet"
        if not path.exists():
            frame.to_parquet(path, index=False)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        meta = {
            "artifact_id": artifact_id, "path": str(path), "schema": {str(k): str(v) for k, v in frame.dtypes.items()},
            "row_count": len(frame), "creating_tool": tool, "arguments": arguments,
            "split_scope": split_scope, "dataset_version": self.dataset_version, "checksum": checksum, "created_at": _now(),
        }
        meta_path = self.root / f"{artifact_id}.json"
        if not meta_path.exists():
            tmp = meta_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, meta_path)
        return artifact_id

    def json(self, tool: str, value: Any, arguments: dict[str, Any], split_scope: str, *, artifact_id: str | None = None) -> str:
        artifact_id = artifact_id or self._id(tool, arguments)
        path = self.root / f"{artifact_id}.json"
        payload = {"artifact_id": artifact_id, "creating_tool": tool, "arguments": arguments, "split_scope": split_scope, "dataset_version": self.dataset_version, "created_at": _now(), "value": value}
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return artifact_id

    def metadata(self, artifact_id: str) -> dict[str, Any]:
        path = next(self.root.glob(f"{artifact_id}.json"), None)
        if path is None:
            raise FileNotFoundError(artifact_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def preview(self, artifact_id: str, columns: list[str] | None = None, limit: int = 20) -> list[dict[str, Any]]:
        meta = self.metadata(artifact_id)
        path = Path(meta.get("path", ""))
        if path.exists():
            frame = pd.read_parquet(path, columns=columns)
            records = frame.head(max(0, min(limit, 100))).to_dict(orient="records")
            return [cast(dict[str, Any], record) for record in records]
        value = meta.get("value")
        if isinstance(value, list):
            return [cast(dict[str, Any], record) for record in value if isinstance(record, dict)]
        return []

