from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_model_ids(settings_path: Path, batch: dict[str, Any]) -> list[str]:
    configured = [str(value) for value in batch.get("test_model_ids", [])]
    if configured:
        return configured[: int(batch.get("max_test_models", len(configured)))]

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    data = settings["data"]
    data_root = (settings_path.parent / data["root"]).resolve()
    source = data_root / data["crispr"]
    with source.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        next(reader)
        model_ids = [str(row[0]) for row in reader if row]
    return sorted(set(model_ids))[: int(batch.get("max_test_models", 500))]


def run_batch(batch_path: Path, settings_path: Path) -> dict[str, Any]:
    batch = yaml.safe_load(batch_path.read_text(encoding="utf-8")) or {}
    model_ids = load_model_ids(settings_path, batch)
    output = (batch_path.parent / batch["output"]["directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / batch["output"].get("predictions_file", "predictions.jsonl")
    summary_path = output / batch["output"].get("summary_file", "summary.json")
    batch_size = max(1, int(batch.get("batch_size", 50)))
    target_gene = str(batch["target_gene"])
    completed: list[dict[str, Any]] = []
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.append(json.loads(line))
    # A failed attempt must be retried after credentials or configuration are
    # fixed; only successful predictions are considered completed.
    completed_ids = {str(item.get("test_model_id")) for item in completed if item.get("status") == "OK"}

    for start in range(0, len(model_ids), batch_size):
        chunk = model_ids[start : start + batch_size]
        for model_id in chunk:
            if model_id in completed_ids:
                continue
            command = [
                sys.executable,
                "main.py",
                "--gene",
                target_gene,
                "--test-model-id",
                model_id,
                "--config",
                str(settings_path),
            ]
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            record: dict[str, Any] = {"test_model_id": model_id, "returncode": process.returncode}
            if process.returncode == 0:
                try:
                    record.update(json.loads(process.stdout))
                    record["status"] = "OK"
                except json.JSONDecodeError:
                    record.update({"status": "ERROR", "error": "Invalid JSON from main.py", "stdout": process.stdout[-2000:]})
            else:
                record.update({"status": "ERROR", "stderr": process.stderr[-4000:]})
            completed.append(record)
            completed_ids.add(model_id)
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        summary = {
            "batch_name": batch.get("batch_name", batch_path.stem),
            "target_gene": target_gene,
            "batch_size": batch_size,
            "requested_models": len(model_ids),
            "completed_models": len(completed),
            "successful_models": sum(item.get("status") == "OK" for item in completed),
            "failed_models": sum(item.get("status") == "ERROR" for item in completed),
            "last_checkpoint": min(start + batch_size, len(model_ids)),
            "predictions_file": str(predictions_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))

    return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCP prediction in batches over many test models")
    parser.add_argument("--batch-config", type=Path, default=Path("config/batch_settings.yaml"))
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    args = parser.parse_args()
    run_batch(args.batch_config.resolve(), args.config.resolve())


if __name__ == "__main__":
    main()
