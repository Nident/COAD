from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .manifests import sha256_json, utc_now, write_json


class AccessDenied(PermissionError):
    """Raised when a tool attempts to cross an immutable split boundary."""


@dataclass(frozen=True)
class SplitManifest:
    manifest_id: str
    run_id: str
    seed: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    patient_grouped: bool
    created_at: str

    def scope(self, name: str) -> tuple[str, ...]:
        values = {"TRAIN": self.train, "VALIDATION": self.validation, "TEST": self.test}
        if name not in values:
            raise AccessDenied(f"Unknown scope: {name}")
        return values[name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "patient_grouped": self.patient_grouped,
            "created_at": self.created_at,
        }


def build_split_manifest(
    run_id: str,
    model_ids: list[str],
    metadata: pd.DataFrame,
    config: dict[str, Any],
    test_model_id: str,
    path: Path,
) -> SplitManifest:
    split = config["split"]
    seed = int(split["seed"])
    if test_model_id not in model_ids:
        raise ValueError(f"Test ModelID is not in CRISPR data: {test_model_id}")
    remaining = [m for m in sorted(set(model_ids)) if m != test_model_id]
    grouped = bool(split.get("patient_grouped", True)) and "PatientID" in metadata.columns
    if grouped:
        groups: dict[str, list[str]] = {}
        for model_id in remaining:
            patient = str(metadata.loc[model_id, "PatientID"]) if model_id in metadata.index else model_id
            groups.setdefault(patient if patient and patient != "nan" else model_id, []).append(model_id)
        group_keys = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
        validation_target = max(1, round(len(remaining) * float(split["validation_fraction"])))
        validation: list[str] = []
        for group in group_keys:
            if len(validation) >= validation_target:
                break
            validation.extend(groups[group])
    else:
        validation = sorted(remaining, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())[: max(1, round(len(remaining) * float(split["validation_fraction"]))) ]
    validation_set = set(validation)
    train = [m for m in remaining if m not in validation_set]
    payload = {"run_id": run_id, "seed": seed, "train": train, "validation": validation, "test": [test_model_id], "patient_grouped": grouped}
    manifest_id = "split_" + sha256_json(payload)[:16]
    manifest = SplitManifest(manifest_id, run_id, seed, tuple(train), tuple(validation), (test_model_id,), grouped, utc_now())
    write_json(path, manifest.as_dict())
    return manifest


def load_split_manifest(path: Path) -> SplitManifest:
    raw = __import__("json").loads(path.read_text(encoding="utf-8"))
    return SplitManifest(
        manifest_id=raw["manifest_id"], run_id=raw["run_id"], seed=int(raw["seed"]),
        train=tuple(raw["train"]), validation=tuple(raw["validation"]), test=tuple(raw["test"]),
        patient_grouped=bool(raw["patient_grouped"]), created_at=raw["created_at"],
    )


def require_scope(manifest: SplitManifest, model_ids: list[str], scope: str) -> None:
    allowed = set(manifest.scope(scope))
    forbidden = sorted(set(model_ids) - allowed)
    if forbidden:
        raise AccessDenied(f"ModelIDs outside {scope} scope: {forbidden[:5]}")

