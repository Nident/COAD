import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from batch import agent_diagnostics, metrics, safe_name
from Data import DataRepository
from Graph import DependencyRAGGraph
from main import run_case
from tools import MultiOmicsToolbox


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def prepare_split(
    config_path: Path,
    split_config: dict[str, Any],
) -> tuple[Path, Path]:
    source_path = resolve(config_path, cast(str, split_config["source_dataset"]))
    train_path = resolve(config_path, cast(str, split_config["train_dataset"]))
    test_path = resolve(config_path, cast(str, split_config["test_dataset"]))
    train_rows = int(split_config["train_rows"])
    test_rows = int(split_config["test_rows"])
    source = pd.read_csv(source_path)
    train = source.iloc[:train_rows].copy()
    test = source.iloc[train_rows:train_rows + test_rows].copy()
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def save_gene_results(
    output_dir: Path,
    target_gene: str,
    evaluations: list[dict[str, object]],
) -> None:
    gene_dir = output_dir / "genes" / safe_name(target_gene)
    gene_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(evaluations)
    table.to_csv(gene_dir / "predictions.csv", index=False)
    summary = metrics(table)
    save_json(gene_dir / "metrics.json", summary)
    classification = cast(
        dict[str, object],
        summary["classification_metrics"],
    )
    per_class = cast(
        dict[str, dict[str, float | int]],
        classification["per_class_one_vs_rest"],
    )
    pd.DataFrame.from_dict(per_class, orient="index").rename_axis(
        "category"
    ).reset_index().to_csv(
        gene_dir / "classification_metrics.csv",
        index=False,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed train/test RAG evaluation for multiple genes."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config/big_batch_settings.yaml",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_dir = Path(__file__).parent
    config_path = (project_dir / arguments.config).resolve()
    config = cast(
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
    split_config = cast(dict[str, Any], config["split"])
    train_path, test_path = prepare_split(config_path, split_config)
    train_model_ids = cast(
        list[str],
        pd.read_csv(train_path, usecols=["ModelID"])["ModelID"].tolist(),
    )
    test_model_ids = cast(
        list[str],
        pd.read_csv(test_path, usecols=["ModelID"])["ModelID"].tolist(),
    )
    target_genes = cast(list[str], config["target_genes"])
    use_llm = bool(config["use_llm"])
    run_ablations = bool(config["run_ablations"])
    resume = bool(config["resume"])
    runs_dir = (
        project_dir / "config" / settings["output"]["runs"]
    ).resolve()
    data_config = cast(dict[str, str], settings["data"])
    data_dir = (
        project_dir / "config" / data_config["data_path"]
    ).resolve()
    expression_model_ids = set(cast(
        list[str],
        pd.read_csv(
            data_dir / data_config["expression"],
            usecols=["ModelID"],
        )["ModelID"].tolist(),
    ))
    available_train_model_ids = [
        model_id for model_id in train_model_ids
        if model_id in expression_model_ids
    ]
    runnable_test_model_ids = [
        model_id for model_id in test_model_ids
        if model_id in expression_model_ids
    ]
    excluded_test_model_ids = [
        model_id for model_id in test_model_ids
        if model_id not in expression_model_ids
    ]
    output_root = resolve(
        config_path,
        cast(dict[str, str], config["output"])["directory"],
    )
    output_dir = output_root / safe_name(cast(str, config["batch_name"]))
    if output_dir.exists() and not resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "big_batch_settings.yaml")

    reference_ids_path = output_dir / "train_model_ids.csv"
    pd.DataFrame({"ModelID": train_model_ids}).to_csv(
        reference_ids_path,
        index=False,
    )
    pd.DataFrame({"ModelID": test_model_ids}).to_csv(
        output_dir / "test_model_ids.csv",
        index=False,
    )
    pd.DataFrame({"ModelID": available_train_model_ids}).to_csv(
        output_dir / "available_train_model_ids.csv",
        index=False,
    )
    pd.DataFrame({"ModelID": runnable_test_model_ids}).to_csv(
        output_dir / "runnable_test_model_ids.csv",
        index=False,
    )
    pd.DataFrame({"ModelID": excluded_test_model_ids}).to_csv(
        output_dir / "excluded_test_model_ids.csv",
        index=False,
    )
    save_json(output_dir / "split_summary.json", {
        "source_dataset": str(resolve(
            config_path,
            cast(str, split_config["source_dataset"]),
        )),
        "train_dataset": str(train_path),
        "test_dataset": str(test_path),
        "train_rows": len(train_model_ids),
        "test_rows": len(test_model_ids),
        "available_expression_train_rows": len(available_train_model_ids),
        "runnable_expression_test_rows": len(runnable_test_model_ids),
        "excluded_test_rows_without_expression": len(excluded_test_model_ids),
        "unused_source_rows": (
            len(pd.read_csv(
                resolve(
                    config_path,
                    cast(str, split_config["source_dataset"]),
                ),
                usecols=["ModelID"],
            ))
            - len(train_model_ids)
            - len(test_model_ids)
        ),
        "target_genes": target_genes,
        "total_predictions": len(target_genes) * len(runnable_test_model_ids),
    })
    if arguments.prepare_only:
        print(output_dir)
        return

    all_evaluations: list[dict[str, object]] = []
    total_runs = len(target_genes) * len(runnable_test_model_ids)
    completed = 0
    repository = DataRepository(project_dir / "config" / "settings.yaml")
    graph = (
        DependencyRAGGraph(project_dir / "config")
        if use_llm
        else None
    )
    toolbox = MultiOmicsToolbox(cast(dict[str, Any], settings["analysis"]))
    for target_gene in target_genes:
        base_data = repository.load(
            target_gene,
            runnable_test_model_ids[0],
            train_model_ids,
        )
        gene_evaluations: list[dict[str, object]] = []
        for test_model_id in runnable_test_model_ids:
            completed += 1
            evaluation_copy = (
                output_dir
                / "evaluations"
                / safe_name(target_gene)
                / f"{safe_name(test_model_id)}.json"
            )
            if resume and evaluation_copy.exists():
                evaluation = cast(
                    dict[str, object],
                    json.loads(evaluation_copy.read_text(encoding="utf-8")),
                )
            else:
                print(
                    f"[{completed}/{total_runs}] "
                    f"{target_gene} {test_model_id}",
                    flush=True,
                )
                data = replace(base_data, test_model_id=test_model_id)
                evaluation, _ = run_case(
                    project_dir=project_dir,
                    config=settings,
                    repository=repository,
                    data=data,
                    run_target_gene=data.target_gene,
                    use_llm=use_llm,
                    execute_ablations=run_ablations,
                    configured_reference_model_count=len(train_model_ids),
                    graph=graph,
                    toolbox=toolbox,
                    update_global_summary=False,
                )
                save_json(evaluation_copy, evaluation)
            gene_evaluations.append(evaluation)
            all_evaluations.append(evaluation)
            pd.DataFrame(all_evaluations).to_csv(
                output_dir / "predictions_in_progress.csv",
                index=False,
            )

        save_gene_results(output_dir, target_gene, gene_evaluations)
        if use_llm:
            save_json(
                output_dir
                / "genes"
                / safe_name(target_gene)
                / "agent_diagnostics.json",
                agent_diagnostics(
                    runs_dir,
                    base_data.target_gene,
                    runnable_test_model_ids,
                ),
            )

    all_table = pd.DataFrame(all_evaluations)
    all_table.to_csv(output_dir / "predictions.csv", index=False)
    save_json(output_dir / "metrics.json", metrics(all_table))
    print(output_dir)


if __name__ == "__main__":
    main()
