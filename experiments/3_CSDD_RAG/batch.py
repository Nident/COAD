import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def metrics(table: pd.DataFrame) -> dict[str, object]:
    predicted = cast(
        np.ndarray,
        table["PredictedGeneEffect"].to_numpy(dtype=float),
    )
    actual = cast(
        np.ndarray,
        table["ActualGeneEffect"].to_numpy(dtype=float),
    )
    baseline = cast(
        np.ndarray,
        table["KNNBaselineGeneEffect"].to_numpy(dtype=float),
    )
    adjustments = cast(
        np.ndarray,
        table["LLMAdjustment"].to_numpy(dtype=float),
    )
    confidence = cast(
        np.ndarray,
        table["Confidence"].to_numpy(dtype=float),
    )
    uncertainty = cast(
        np.ndarray,
        table["Uncertainty"].to_numpy(dtype=float),
    )
    categories_match = cast(
        np.ndarray,
        (
            table["PredictedCategory"] == table["ActualCategory"]
        ).to_numpy(dtype=bool),
    )
    absolute_error = np.abs(predicted - actual)
    baseline_absolute_error = np.abs(baseline - actual)
    predicted_ranks = pd.Series(predicted).rank().to_numpy(dtype=float)
    baseline_ranks = pd.Series(baseline).rank().to_numpy(dtype=float)
    actual_ranks = pd.Series(actual).rank().to_numpy(dtype=float)
    confusion = pd.crosstab(
        table["ActualCategory"],
        table["PredictedCategory"],
    )
    result: dict[str, object] = {
        "n": len(table),
        "MAE": float(np.mean(absolute_error)),
        "RMSE": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "median_absolute_error": float(np.median(absolute_error)),
        "classification_accuracy": float(np.mean(categories_match)),
        "mean_confidence": float(np.mean(confidence)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "KNNBaseline": {
            "MAE": float(np.mean(baseline_absolute_error)),
            "RMSE": float(np.sqrt(np.mean(np.square(baseline - actual)))),
            "PearsonCorrelation": correlation(baseline, actual),
            "SpearmanCorrelation": correlation(baseline_ranks, actual_ranks),
        },
        "LLMCorrection": {
            "nonzero_adjustments": int(np.count_nonzero(adjustments)),
            "mean_adjustment": float(np.mean(adjustments)),
            "minimum_adjustment": float(np.min(adjustments)),
            "maximum_adjustment": float(np.max(adjustments)),
            "MAE_change_vs_KNN": float(
                np.mean(absolute_error) - np.mean(baseline_absolute_error)
            ),
        },
        "diagnostics": {
            "uncertainty_absolute_error_correlation": correlation(
                uncertainty,
                absolute_error,
            ),
            "confidence_absolute_error_correlation": correlation(
                confidence,
                absolute_error,
            ),
            "category_confusion_matrix": {
                str(actual_category): {
                    str(predicted_category): int(count)
                    for predicted_category, count in row.items()
                }
                for actual_category, row in confusion.iterrows()
            },
        },
    }
    if len(table) >= 2:
        result["PearsonCorrelation"] = correlation(predicted, actual)
        result["SpearmanCorrelation"] = correlation(
            predicted_ranks,
            actual_ranks,
        )
    else:
        result["PearsonCorrelation"] = None
        result["SpearmanCorrelation"] = None
    return result


def agent_diagnostics(
    runs_dir: Path,
    target_gene: str,
    test_model_ids: list[str],
) -> dict[str, object]:
    roles = ["expression", "mutation", "cnv", "metadata", "judge"]
    token_counts: dict[str, list[int]] = {role: [] for role in roles}
    directions: dict[str, dict[str, int]] = {
        role: {} for role in ["expression", "mutation", "cnv"]
    }
    transferability: dict[str, int] = {}
    for test_model_id in test_model_ids:
        run_dir = runs_dir / (
            f"{safe_name(target_gene)}_{safe_name(test_model_id)}"
        )
        prompt_data = cast(
            dict[str, dict[str, object]],
            json.loads(
                (run_dir / "prompts" / "prompt_metrics.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        for role in roles:
            token_counts[role].append(
                cast(int, prompt_data[role]["estimated_tokens"])
            )
        for role in directions:
            evidence = cast(
                dict[str, object],
                json.loads(
                    (run_dir / "agent_outputs" / f"{role}.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
            direction = cast(str, evidence["direction"])
            directions[role][direction] = (
                directions[role].get(direction, 0) + 1
            )
        metadata = cast(
            dict[str, object],
            json.loads(
                (run_dir / "agent_outputs" / "metadata.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        transfer = cast(str, metadata["transferability"])
        transferability[transfer] = transferability.get(transfer, 0) + 1

    return {
        "completed_model_count": len(test_model_ids),
        "completed_llm_call_count": len(test_model_ids) * len(roles),
        "directions": directions,
        "metadata_transferability": transferability,
        "prompt_estimated_tokens": {
            role: {
                "total": sum(values),
                "mean": sum(values) / len(values),
                "maximum": max(values),
            }
            for role, values in token_counts.items()
        },
        "total_estimated_input_tokens": sum(
            sum(values) for values in token_counts.values()
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RAG pipeline for configured test ModelIDs."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config/batch_settings.yaml",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_dir = Path(__file__).parent
    config_path = (project_dir / arguments.config).resolve()
    batch_config = cast(
        dict[str, Any],
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
    )
    settings = cast(
        dict[str, Any],
        yaml.safe_load(
            (project_dir / "config" / "settings.yaml").read_text(
                encoding="utf-8"
            )
        ),
    )
    target_gene = cast(str, batch_config["target_gene"])
    test_model_ids = cast(list[str], batch_config["test_model_ids"])
    use_llm = cast(bool, batch_config["use_llm"])
    run_ablations = cast(bool, batch_config["run_ablations"])
    runs_dir = (
        project_dir / "config" / settings["output"]["runs"]
    ).resolve()
    output_root = (
        config_path.parent / batch_config["output"]["directory"]
    ).resolve()
    output_dir = output_root / safe_name(cast(str, batch_config["batch_name"]))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "batch_settings.yaml")

    evaluations: list[dict[str, object]] = []
    for index, test_model_id in enumerate(test_model_ids, start=1):
        command = [
            sys.executable,
            str(project_dir / "main.py"),
            target_gene,
            test_model_id,
        ]
        if not use_llm:
            command.append("--no-llm")
        if not run_ablations:
            command.append("--no-ablations")
        print(f"[{index}/{len(test_model_ids)}] {target_gene} {test_model_id}")
        subprocess.run(command, cwd=project_dir, check=True)
        run_name = f"{safe_name(target_gene)}_{safe_name(test_model_id)}"
        evaluation_path = runs_dir / run_name / "evaluation.json"
        evaluations.append(cast(
            dict[str, object],
            json.loads(evaluation_path.read_text(encoding="utf-8")),
        ))

    table = pd.DataFrame(evaluations)
    table.to_csv(output_dir / "predictions.csv", index=False)
    summary = metrics(table)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if use_llm:
        diagnostics = agent_diagnostics(
            runs_dir,
            target_gene,
            test_model_ids,
        )
        (output_dir / "agent_diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_dir)


if __name__ == "__main__":
    main()
