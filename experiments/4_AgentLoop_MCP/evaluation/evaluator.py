from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import re

from data.manifests import load_config, utc_now


def locate_artifact(root: Path, artifact_id: str) -> tuple[Path, dict[str, Any]]:
    for path in root.glob(f"*/{artifact_id}.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("artifact_id") == artifact_id and raw.get("creating_tool") == "predict_test_gene_effect":
            return path.parent, raw["value"]
    raise FileNotFoundError(f"Immutable prediction artifact not found: {artifact_id}")


def evaluate(artifact_id: str, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    results_root = (config_path.parent / config["artifacts"]["root"]).resolve()
    run_dir, prediction = locate_artifact(results_root, artifact_id)
    if prediction.get("fallback_used") is None or not prediction.get("frozen_plan_id"):
        raise PermissionError("Only immutable prediction artifacts may be evaluated")
    data_root = (config_path.parent / config["data"]["root"]).resolve()
    effect_path = data_root / config["data"]["crispr"]
    headers = pd.read_csv(effect_path, nrows=0).columns.tolist()
    symbol = prediction["target_gene"]
    column = next((value for value in headers[1:] if re.sub(r"\s+\(\d+\)$", "", str(value)).strip() == symbol), None)
    if column is None:
        raise KeyError(f"Target gene not found in CRISPR header: {symbol}")
    effect = pd.read_csv(effect_path, usecols=[headers[0], column])
    actual_row = effect.loc[effect.iloc[:, 0].astype(str) == prediction["test_model_id"]]
    if actual_row.empty:
        raise KeyError(prediction["test_model_id"])
    actual = float(actual_row.iloc[0][column])
    predicted = float(prediction["predicted_gene_effect"])
    output = {"evaluation_id": f"evaluation_{artifact_id}", "prediction_artifact_id": artifact_id, "target_gene": prediction["target_gene"], "test_model_id": prediction["test_model_id"], "actual_gene_effect": actual, "predicted_gene_effect": predicted, "absolute_error": abs(actual - predicted), "squared_error": (actual - predicted) ** 2, "evaluated_at": utc_now(), "scope": "EVALUATION"}
    path = run_dir / f"{output['evaluation_id']}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation_path": str(path), **output}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-artifact", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    args = parser.parse_args()
    evaluate(args.prediction_artifact, args.config.resolve())


if __name__ == "__main__":
    main()
