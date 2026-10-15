from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def source_manifest(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    data = config["data"]
    root = (config_path.parent / data["root"]).resolve()
    files: dict[str, Any] = {}
    for key in ("crispr", "expression", "mutations", "cnv", "metadata"):
        path = root / data[key]
        stat = path.stat()
        header = pd.read_csv(path, nrows=0).columns.tolist()
        files[key] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "columns": header,
        }
    return {"cache_version": str(data.get("cache_version", "1")), "created_at": utc_now(), "files": files}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)

