import json
import re
import traceback
from pathlib import Path
from typing import cast

import pandas as pd
import tiktoken
from pydantic import BaseModel

from Data import DataPreparator, DataReader, GeneData
from Graph import DependencyGraph
from Schemas import CompletedGraphState, EffectCategory, GraphInput


TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


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


def text_metrics(
    text: str,
    llm_invoked: bool | None = None,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "bytes_utf8": len(text.encode("utf-8")),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "estimated_tokens": len(TOKEN_ENCODING.encode(text)),
        "tokenizer": "cl100k_base",
        "token_count_is_estimate": True,
    }
    if llm_invoked is not None:
        metrics["llm_invoked"] = llm_invoked
    return metrics


def save_file_metrics(output_path: Path, files: list[Path]) -> None:
    metrics: dict[str, object] = {
        path.name: text_metrics(path.read_text(encoding="utf-8"))
        for path in files
    }
    save_json(output_path, metrics)


def save_prompt_bundle(
    output_dir: Path,
    prompts: dict[str, str],
    mutation_llm_invoked: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {}
    for role, prompt in prompts.items():
        path = output_dir / f"{role}_prompt.md"
        path.write_text(prompt, encoding="utf-8")
        invoked = role != "mutation" or mutation_llm_invoked
        metrics[role] = text_metrics(prompt, invoked)
    save_json(output_dir / "prompt_metrics.json", metrics)


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


def decoded_table(value: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(value))


def save_prepared_data(
    output_dir: Path,
    state: GraphInput,
    related_genes: list[str],
    relation_depth: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_crispr = decoded_table(state["train_crispr"])

    metadata_path = output_dir / "metadata.json"
    donor_path = output_dir / "donor_agent_input.json"
    expression_path = output_dir / "expression_agent_input.json"
    mutation_path = output_dir / "mutation_agent_input.json"

    save_json(
        metadata_path,
        {
            "target_gene": state["gene"],
            "test_model_id": state["test_model_id"],
            "relation_depth": relation_depth,
            "related_genes": related_genes,
        },
    )
    save_json(
        donor_path,
        {
            "train_crispr": train_crispr,
            "train_donor": decoded_table(state["train_donor"]),
            "test_donor": decoded_table(state["test_donor"]),
        },
    )
    save_json(
        expression_path,
        {
            "train_crispr": train_crispr,
            "train_expression": decoded_table(state["train_expression"]),
            "test_expression": decoded_table(state["test_expression"]),
        },
    )
    save_json(
        mutation_path,
        {
            "train_crispr": train_crispr,
            "train_mutation": decoded_table(state["train_mutation"]),
            "test_mutation": decoded_table(state["test_mutation"]),
            "test_has_mutation": state["test_has_mutation"],
        },
    )
    save_file_metrics(
        output_dir / "json_metrics.json",
        [metadata_path, donor_path, expression_path, mutation_path],
    )


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
    save_prompt_bundle(
        run_dir,
        {
            "donor": state["donor_prompt"],
            "expression": state["expression_prompt"],
            "mutation": state["mutation_prompt"],
            "judge": state["judge_prompt"],
        },
        state["test_has_mutation"],
    )
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
        reader.config["analysis"]["relation_depth"],
    )
    analysis = reader.config["analysis"]
    graph = DependencyGraph(project_dir / "config")
    output_dir = project_dir / "runs"
    prepared_data_dir = (
        project_dir / "config" / reader.config["output"]["prepared_data"]
    ).resolve()
    error_dir = (
        project_dir / "config" / reader.config["output"]["errors"]
    ).resolve()

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
            split_mode=analysis["split_mode"],
        )
        model_ids = cast(list[str], data.ground_truth["ModelID"].tolist())
        categories = cast(
            list[EffectCategory],
            data.ground_truth["EffectCategory"].tolist(),
        )
        truth = dict(zip(model_ids, categories, strict=True))
        train_count = len(data.train_crispr)
        related_genes = preparator.related_genes(gene)

        for test_model_id in data.test_ids:
            run_name = (
                f"{safe_name(gene)}_train_{train_count}_"
                f"test_{safe_name(test_model_id)}"
            )
            run_dir = output_dir / run_name
            if (
                analysis["resume_completed_runs"]
                and (run_dir / "evaluation.json").exists()
            ):
                print(f"Skipped completed run: {run_dir}")
                continue

            graph_input = build_state(data, test_model_id)
            prepared_run_dir = prepared_data_dir / run_name
            save_prepared_data(
                prepared_run_dir,
                graph_input,
                related_genes,
                analysis["relation_depth"],
            )
            save_prompt_bundle(
                prepared_run_dir / "prompts",
                graph.analyst_prompts(graph_input),
                graph_input["test_has_mutation"],
            )
            print(prepared_run_dir)

            try:
                state = graph.invoke(graph_input)
            except Exception as error:
                run_error_dir = error_dir / run_name
                run_error_dir.mkdir(parents=True, exist_ok=True)
                save_json(
                    run_error_dir / "error.json",
                    {
                        "gene": gene,
                        "test_model_id": test_model_id,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "prepared_data": str(prepared_run_dir),
                    },
                )
                print(run_error_dir / "error.json")
                raise
            save_run(run_dir, state, truth[test_model_id])
            print(run_dir)


if __name__ == "__main__":
    main()
