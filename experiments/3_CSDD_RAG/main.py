import argparse
import json
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import tiktoken
import yaml
from pydantic import BaseModel

from Data import DataRepository, TrainingData
from Graph import DependencyRAGGraph
from Retrieval import MultiOmicsRetriever, RetrievalResult
from Schemas import CompletedGraphState, EffectCategory, GraphInput


TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


def safe_name(value: str) -> str:
    return re.sub(r'[/\\:*?"<>|]+', "_", value).strip()


def save_json(path: Path, value: object) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )


def frame_json(table: pd.DataFrame) -> str:
    return cast(str, table.to_json(orient="records", force_ascii=False))


def dict_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def prompt_metrics(text: str) -> dict[str, object]:
    return {
        "bytes_utf8": len(text.encode("utf-8")),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "estimated_tokens": len(TOKEN_ENCODING.encode(text)),
        "tokenizer": "cl100k_base",
        "token_count_is_estimate": True,
    }


def effect_category(
    effect: float,
    thresholds: dict[str, float],
) -> EffectCategory:
    if effect <= thresholds["strong_effect"]:
        return "strong_effect"
    if effect <= thresholds["weak_effect"]:
        return "weak_effect"
    return "no_effect"


def graph_input(
    data: TrainingData,
    result: RetrievalResult,
) -> GraphInput:
    return {
        "gene": data.target_gene,
        "test_model_id": data.test_model_id,
        "baseline": dict_json(result.baseline),
        "neighbours": frame_json(result.top_k),
        "expression_data": frame_json(result.expression_context),
        "mutation_data": dict_json(result.mutation_context),
        "cnv_data": frame_json(result.cnv_context),
        "metadata_data": frame_json(result.metadata_context),
    }


def save_retrieval(run_dir: Path, result: RetrievalResult) -> None:
    retrieval_dir = run_dir / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    result.expression_top.to_csv(
        retrieval_dir / "expression_top_30.csv",
        index=False,
    )
    result.reranked_top.to_csv(
        retrieval_dir / "reranked_top_30.csv",
        index=False,
    )
    result.top_k.to_csv(retrieval_dir / "final_top_k.csv", index=False)
    save_json(retrieval_dir / "knn_baseline.json", result.baseline)
    save_json(retrieval_dir / "feature_manifest.json", result.feature_manifest)

    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result.expression_context.to_csv(
        evidence_dir / "expression_context.csv",
        index=False,
    )
    result.cnv_context.to_csv(evidence_dir / "cnv_context.csv", index=False)
    result.metadata_context.to_csv(
        evidence_dir / "metadata_context.csv",
        index=False,
    )
    save_json(evidence_dir / "mutation_context.json", result.mutation_context)


def save_prepared_prompts(
    run_dir: Path,
    graph: DependencyRAGGraph,
    state: GraphInput,
) -> None:
    prompt_dir = run_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, object] = {}
    for role, text in graph.analyst_prompts(state).items():
        (prompt_dir / f"{role}_prompt.md").write_text(text, encoding="utf-8")
        metrics[role] = prompt_metrics(text)
    save_json(prompt_dir / "prompt_metrics.json", metrics)


def save_graph_result(run_dir: Path, state: CompletedGraphState) -> None:
    prompt_dir = run_dir / "prompts"
    evidence_dir = run_dir / "agent_outputs"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    prompts = {
        "expression": state["expression_prompt"],
        "mutation": state["mutation_prompt"],
        "cnv": state["cnv_prompt"],
        "metadata": state["metadata_prompt"],
        "judge": state["judge_prompt"],
    }
    for role, text in prompts.items():
        (prompt_dir / f"{role}_prompt.md").write_text(text, encoding="utf-8")
    save_json(
        prompt_dir / "prompt_metrics.json",
        {role: prompt_metrics(text) for role, text in prompts.items()},
    )
    save_json(evidence_dir / "expression.json", state["expression_evidence"])
    save_json(evidence_dir / "mutation.json", state["mutation_evidence"])
    save_json(evidence_dir / "cnv.json", state["cnv_evidence"])
    save_json(evidence_dir / "metadata.json", state["metadata_evidence"])
    save_json(evidence_dir / "judge.json", state["judge_decision"])


def deterministic_confidence(baseline: dict[str, Any]) -> float:
    similarity = float(baseline["mean_similarity"])
    deviation = float(baseline["standard_deviation"])
    return float(np.clip(similarity / (1.0 + deviation), 0.0, 1.0))


def run_ablations(
    retriever: MultiOmicsRetriever,
    data: TrainingData,
) -> list[dict[str, object]]:
    variants = [
        ("expression_only_knn", False, False, False),
        ("expression_mutation", True, False, False),
        ("expression_cnv", False, True, False),
        ("expression_mutation_cnv", True, True, False),
        ("expression_mutation_cnv_metadata", True, True, True),
        ("full_rag_without_llm", True, True, True),
    ]
    records: list[dict[str, object]] = []
    for name, mutation, cnv, metadata in variants:
        result = retriever.retrieve(data, mutation, cnv, metadata)
        records.append({
            "variant": name,
            "use_mutation_rerank": mutation,
            "use_cnv_rerank": cnv,
            "use_metadata_rerank": metadata,
            "use_llm": False,
            "predicted_gene_effect": result.baseline["weighted_prediction"],
            "neighbour_standard_deviation": result.baseline[
                "standard_deviation"
            ],
            "mean_similarity": result.baseline["mean_similarity"],
        })
    return records


def evaluation_summary(
    runs_dir: Path,
) -> dict[str, float | int | None] | None:
    rows: list[dict[str, object]] = []
    for path in runs_dir.glob("*/evaluation.json"):
        rows.append(cast(dict[str, object], json.loads(path.read_text())))
    if not rows:
        return None
    table = pd.DataFrame(rows)
    predicted = cast(
        np.ndarray,
        table["PredictedGeneEffect"].to_numpy(dtype=float),
    )
    actual = cast(
        np.ndarray,
        table["ActualGeneEffect"].to_numpy(dtype=float),
    )
    result: dict[str, float | int | None] = {
        "n": len(table),
        "MAE": float(np.mean(np.abs(predicted - actual))),
        "RMSE": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "classification_accuracy": float(
            np.mean(table["PredictedCategory"] == table["ActualCategory"])
        ),
    }
    if len(table) >= 2:
        result["PearsonCorrelation"] = float(np.corrcoef(predicted, actual)[0, 1])
        predicted_ranks = pd.Series(predicted).rank().to_numpy(dtype=float)
        actual_ranks = pd.Series(actual).rank().to_numpy(dtype=float)
        result["SpearmanCorrelation"] = float(
            np.corrcoef(predicted_ranks, actual_ranks)[0, 1]
        )
    else:
        result["PearsonCorrelation"] = None
        result["SpearmanCorrelation"] = None
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict continuous CRISPR Gene Effect with multi-omics RAG."
    )
    parser.add_argument("target_gene", nargs="?")
    parser.add_argument("test_model_id", nargs="?")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-ablations", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_dir = Path(__file__).parent
    config_path = project_dir / "config" / "settings.yaml"
    config = cast(
        dict[str, Any],
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
    )
    target_gene = arguments.target_gene or config["input"]["target_gene"]
    test_model_id = (
        arguments.test_model_id or config["input"]["test_model_id"]
    )
    retrieval_config = cast(dict[str, Any], config["retrieval"])
    prediction_config = cast(dict[str, Any], config["prediction"])
    use_llm = bool(prediction_config["use_llm"]) and not arguments.no_llm
    run_name = f"{safe_name(target_gene)}_{safe_name(test_model_id)}"
    runs_dir = (project_dir / "config" / config["output"]["runs"]).resolve()
    run_dir = runs_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    repository = DataRepository(config_path)
    data = repository.load(target_gene, test_model_id)
    retriever = MultiOmicsRetriever(retrieval_config)
    result = retriever.retrieve(
        data,
        bool(retrieval_config["use_mutation_rerank"]),
        bool(retrieval_config["use_cnv_rerank"]),
        bool(retrieval_config["use_metadata_rerank"]),
    )
    save_retrieval(run_dir, result)
    save_json(run_dir / "input.json", {
        "TargetGene": data.target_gene,
        "TargetColumn": data.target_column,
        "TestModelID": data.test_model_id,
        "UseLLM": use_llm,
    })

    ablations: list[dict[str, object]] = []
    if bool(config["experiments"]["run_ablations"]) and not arguments.no_ablations:
        ablations = run_ablations(retriever, data)

    baseline_effect = float(result.baseline["weighted_prediction"])
    predicted_effect = baseline_effect
    uncertainty = float(result.baseline["standard_deviation"])
    confidence = deterministic_confidence(result.baseline)
    llm_adjustment = 0.0

    if use_llm:
        graph = DependencyRAGGraph(project_dir / "config")
        state_input = graph_input(data, result)
        save_prepared_prompts(run_dir, graph, state_input)
        try:
            state = graph.invoke(state_input)
        except Exception as error:
            save_json(run_dir / "error.json", {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            })
            raise
        save_graph_result(run_dir, state)
        judge = state["judge_decision"]
        llm_adjustment = judge.suggested_adjustment
        predicted_effect = judge.predicted_gene_effect
        uncertainty = max(uncertainty, judge.uncertainty)
        confidence = judge.confidence

    thresholds = cast(dict[str, float], prediction_config["thresholds"])
    predicted_category = effect_category(predicted_effect, thresholds)
    prediction = {
        "TargetGene": data.target_gene,
        "TestModelID": data.test_model_id,
        "KNNBaselineGeneEffect": baseline_effect,
        "LLMAdjustment": llm_adjustment,
        "PredictedGeneEffect": predicted_effect,
        "PredictedCategory": predicted_category,
        "Confidence": confidence,
        "Uncertainty": uncertainty,
        "TestEffectUsedBeforeEvaluation": False,
    }
    save_json(run_dir / "prediction.json", prediction)

    actual_effect = repository.actual_effect(data.target_column, test_model_id)
    actual_category = effect_category(actual_effect, thresholds)
    evaluation: dict[str, object] = {
        **prediction,
        "ActualGeneEffect": actual_effect,
        "AbsoluteError": abs(predicted_effect - actual_effect),
        "SquaredError": (predicted_effect - actual_effect) ** 2,
        "ActualCategory": actual_category,
        "CategoryCorrect": predicted_category == actual_category,
    }
    save_json(run_dir / "evaluation.json", evaluation)

    if ablations:
        for row in ablations:
            effect = cast(float, row["predicted_gene_effect"])
            row["actual_gene_effect"] = actual_effect
            row["absolute_error"] = abs(effect - actual_effect)
            row["squared_error"] = (effect - actual_effect) ** 2
        ablations.append({
            "variant": "full_multi_agent_rag_llm" if use_llm else "configured_run_no_llm",
            "use_mutation_rerank": retrieval_config["use_mutation_rerank"],
            "use_cnv_rerank": retrieval_config["use_cnv_rerank"],
            "use_metadata_rerank": retrieval_config["use_metadata_rerank"],
            "use_llm": use_llm,
            "predicted_gene_effect": predicted_effect,
            "actual_gene_effect": actual_effect,
            "absolute_error": abs(predicted_effect - actual_effect),
            "squared_error": (predicted_effect - actual_effect) ** 2,
            "neighbour_standard_deviation": result.baseline[
                "standard_deviation"
            ],
            "mean_similarity": result.baseline["mean_similarity"],
        })
        pd.DataFrame(ablations).to_csv(run_dir / "ablations.csv", index=False)

    summary = evaluation_summary(runs_dir)
    if summary is not None:
        save_json(runs_dir / "evaluation_summary.json", summary)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    print(run_dir)


if __name__ == "__main__":
    main()
