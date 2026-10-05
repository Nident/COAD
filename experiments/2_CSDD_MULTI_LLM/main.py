import json
import re
from pathlib import Path
from typing import cast

import pandas as pd
from pydantic import BaseModel

from Data import DataPreparator, DataReader, GeneData
from Graph import DependencyGraph
from Schemas import CompletedGraphState, EffectCategory, GraphInput


def table_json(table: pd.DataFrame) -> str:
    clean = table.astype(object).where(table.notna(), None)
    data = clean.to_dict(orient="split")
    data.pop("index")
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def model_rows(table: pd.DataFrame, model_id: str) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        table.loc[table["ModelID"] == model_id].copy(),
    )


def safe_name(value: str) -> str:
    return re.sub(r'[/\\:*?"<>|]+', "_", value).strip()


def save_json(path: Path, value: BaseModel | dict[str, object]) -> None:
    serializable = value.model_dump() if isinstance(value, BaseModel) else value
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_state(data: GeneData, test_model_id: str) -> GraphInput:
    test_mutation = model_rows(data.test_mutation, test_model_id)
    return {
        "gene": data.gene,
        "test_model_id": test_model_id,
        "train_crispr": table_json(data.train_crispr),
        "train_donor": table_json(data.train_donor),
        "train_expression": table_json(data.train_expression),
        "train_mutation": table_json(data.train_mutation),
        "test_donor": table_json(model_rows(data.test_donor, test_model_id)),
        "test_expression": table_json(
            model_rows(data.test_expression, test_model_id)
        ),
        "test_mutation": table_json(test_mutation),
        "test_has_mutation": not test_mutation.empty,
    }


def save_run(
    run_dir: Path,
    state: CompletedGraphState,
    actual_category: EffectCategory,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, object] = {
        "gene": state["gene"],
        "test_model_id": state["test_model_id"],
        "train_crispr": json.loads(state["train_crispr"]),
        "train_donor": json.loads(state["train_donor"]),
        "train_expression": json.loads(state["train_expression"]),
        "train_mutation": json.loads(state["train_mutation"]),
        "test_donor": json.loads(state["test_donor"]),
        "test_expression": json.loads(state["test_expression"]),
        "test_mutation": json.loads(state["test_mutation"]),
        "test_has_mutation": state["test_has_mutation"],
    }
    save_json(run_dir / "inputs.json", inputs)
    (run_dir / "donor_prompt.md").write_text(
        state["donor_prompt"], encoding="utf-8"
    )
    (run_dir / "expression_prompt.md").write_text(
        state["expression_prompt"], encoding="utf-8"
    )
    (run_dir / "mutation_prompt.md").write_text(
        state["mutation_prompt"], encoding="utf-8"
    )
    (run_dir / "judge_prompt.md").write_text(
        state["judge_prompt"], encoding="utf-8"
    )
    save_json(run_dir / "donor_hypothesis.json", state["donor_hypothesis"])
    save_json(
        run_dir / "expression_hypothesis.json",
        state["expression_hypothesis"],
    )
    save_json(run_dir / "mutation_hypothesis.json", state["mutation_hypothesis"])
    save_json(run_dir / "verdict.json", state["verdict"])
    save_json(
        run_dir / "evaluation.json",
        {
            "model_id": state["test_model_id"],
            "predicted_category": state["verdict"].predicted_category,
            "actual_category": actual_category,
            "correct": state["verdict"].predicted_category == actual_category,
        },
    )


def main() -> None:
    project_dir = Path(__file__).parent
    reader = DataReader(project_dir / "config" / "settings.yaml")
    preparator = DataPreparator(
        reader.read(),
        reader.read_gene_relations(),
    )
    analysis = reader.config["analysis"]
    graph = DependencyGraph(project_dir / "config")
    output_dir = project_dir / "runs"

    genes = preparator.genes(
        analysis["gene_start_index"],
        analysis["gene_end_index"],
    )
    for gene in genes:
        data = preparator.prepare(
            gene=gene,
            train_size=analysis["train_model_limit"],
            test_size=analysis["test_model_limit"],
            random_state=analysis["random_state"],
        )
        model_ids = cast(list[str], data.ground_truth["ModelID"].tolist())
        categories = cast(
            list[EffectCategory],
            data.ground_truth["EffectCategory"].tolist(),
        )
        truth = dict(zip(model_ids, categories, strict=True))
        train_count = len(data.train_crispr)

        for test_model_id in data.test_ids:
            state = graph.invoke(build_state(data, test_model_id))
            run_name = (
                f"{safe_name(gene)}_train_{train_count}_"
                f"test_{safe_name(test_model_id)}"
            )
            run_dir = output_dir / run_name
            save_run(run_dir, state, truth[test_model_id])
            print(run_dir)


if __name__ == "__main__":
    main()
