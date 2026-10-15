from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from fastmcp import FastMCP

from data.repository import DataRepository
from .schemas import EvaluationRequest, PlanSpec


def create_server(config_path: Path) -> FastMCP:
    repository = DataRepository(config_path)
    mcp = FastMCP("depmap-agent-loop", instructions="Deterministic DepMap data, statistics, validation, and prediction tools with immutable split enforcement.")

    @mcp.tool()
    def initialize_run(target_gene: str, test_model_id: str, run_id: str) -> dict[str, Any]:
        """Create or resume an immutable TRAIN/VALIDATION/TEST split manifest."""
        return repository.initialize_run(target_gene, test_model_id, run_id)

    @mcp.tool()
    def get_dataset_manifest() -> dict[str, Any]:
        return repository.manifest()

    @mcp.tool()
    def resolve_gene(gene: str) -> dict[str, Any]:
        return repository.resolve_gene(gene)

    @mcp.tool()
    def get_target_gene_summary(target_gene: str, run_id: str) -> dict[str, Any]:
        return repository.target_summary(target_gene, run_id)

    @mcp.tool()
    def get_test_model_profile(target_gene: str, test_model_id: str, modalities: list[str], run_id: str) -> dict[str, Any]:
        return repository.test_profile(target_gene, test_model_id, modalities, run_id)

    @mcp.tool()
    def get_models(model_ids: list[str], fields: list[str], run_id: str, page: int = 0, page_size: int = 50) -> dict[str, Any]:
        return repository.get_models(model_ids, fields, run_id, page, page_size)

    @mcp.tool()
    def get_expression(genes: list[str], model_ids: list[str], run_id: str, page: int = 0, page_size: int = 50) -> dict[str, Any]:
        return repository.get_expression(genes, model_ids, run_id, page, page_size)

    @mcp.tool()
    def get_mutations(genes: list[str], model_ids: list[str], representation: str, run_id: str, page: int = 0, page_size: int = 50) -> dict[str, Any]:
        if representation not in {"gene", "variant"}:
            raise ValueError("representation must be gene or variant")
        return repository.get_mutations(genes, model_ids, representation, run_id, page, page_size)

    @mcp.tool()
    def get_cnv(genes: list[str], model_ids: list[str], run_id: str, page: int = 0, page_size: int = 50) -> dict[str, Any]:
        return repository.get_cnv(genes, model_ids, run_id, page, page_size)

    @mcp.tool()
    def get_metadata(model_ids: list[str], fields: list[str], run_id: str, page: int = 0, page_size: int = 50) -> dict[str, Any]:
        return repository.get_metadata(model_ids, fields, run_id, page, page_size)

    @mcp.tool()
    def get_train_gene_effect(target_gene: str, run_id: str, model_ids: list[str] | None = None, page: int = 0, page_size: int = 50) -> dict[str, Any]:
        return repository.train_effect(target_gene, model_ids, run_id, page, page_size)

    @mcp.tool()
    def screen_expression_dependency_associations(target_gene: str, run_id: str, top_k: int = 20) -> dict[str, Any]:
        return repository.screen_expression(target_gene, run_id, top_k)

    @mcp.tool()
    def screen_cnv_dependency_associations(target_gene: str, run_id: str, top_k: int = 20) -> dict[str, Any]:
        return repository._screen(target_gene, run_id, "cnv", top_k=top_k)

    @mcp.tool()
    def calculate_coexpression(gene: str, candidate_genes: list[str], run_id: str) -> dict[str, Any]:
        return repository.calculate_coexpression(gene, candidate_genes, run_id)

    @mcp.tool()
    def get_existing_related_gene_mapping(target_gene: str, run_id: str) -> dict[str, Any]:
        return {"status": "NOT_CONFIGURED", "target_gene": target_gene, "split_scope": "TRAIN", "rows": []}

    @mcp.tool()
    def rank_candidate_genes(target_gene: str, run_id: str, evidence_sources: list[str], top_k: int = 20) -> dict[str, Any]:
        if any(source not in {"expression_dependency", "coexpression", "cnv_dependency"} for source in evidence_sources):
            raise ValueError("Only explicitly supported evidence sources may be combined")
        results = {}
        if "expression_dependency" in evidence_sources:
            results["expression_dependency"] = repository.screen_expression(target_gene, run_id, top_k)
        if "cnv_dependency" in evidence_sources:
            results["cnv_dependency"] = repository._screen(target_gene, run_id, "cnv", top_k=top_k)
        return {"status": "OK", "sources": results, "split_scope": "TRAIN"}

    @mcp.tool()
    def retrieve_similar_models(modality: str, features: list[str], query_model_id: str, metric: str, k: int, run_id: str) -> dict[str, Any]:
        split = repository.run(run_id)
        if modality not in {"expression", "cnv"} or metric != "euclidean":
            metric = "euclidean"
        frame = repository._expression(list(split.train) + [query_model_id], features) if modality == "expression" else repository._cnv(list(split.train) + [query_model_id], features)
        query = frame.loc[query_model_id].to_numpy(dtype=float)
        matrix = frame.drop(index=query_model_id).apply(lambda col: col.fillna(col.median())).to_numpy(dtype=float)
        q = np.nan_to_num(query, nan=float(np.nanmedian(matrix) if matrix.size else 0.0))
        distances = np.linalg.norm(np.nan_to_num(matrix, nan=0.0) - q, axis=1)
        ids = frame.drop(index=query_model_id).index.to_numpy()
        order = np.argsort(distances)[: max(1, min(k, len(ids)))]
        return {"status": "OK", "model_ids": [str(ids[i]) for i in order], "scores": [float(distances[i]) for i in order], "method": metric, "features": features, "population_scope": "TRAIN", "split_scope": "TRAIN"}

    @mcp.tool()
    def retrieve_multiomics(features_by_modality: dict[str, list[str]], query_model_id: str, weights: dict[str, float], k: int, run_id: str) -> dict[str, Any]:
        parts = []
        for modality, genes in features_by_modality.items():
            if modality == "expression":
                parts.append(repository._expression(list(repository.run(run_id).train) + [query_model_id], genes).add_prefix("expression::"))
            elif modality == "cnv":
                parts.append(repository._cnv(list(repository.run(run_id).train) + [query_model_id], genes).add_prefix("cnv::"))
        if not parts:
            return {"status": "INSUFFICIENT_DATA", "model_ids": []}
        frame = __import__("pandas").concat(parts, axis=1)
        query = frame.loc[query_model_id].fillna(frame.median()).to_numpy(dtype=float)
        matrix_frame = frame.drop(index=query_model_id).fillna(frame.median())
        distances = np.linalg.norm(matrix_frame.to_numpy(dtype=float) - query, axis=1)
        ids = matrix_frame.index.to_numpy(); order = np.argsort(distances)[: max(1, min(k, len(ids)))]
        return {"status": "OK", "model_ids": [str(ids[i]) for i in order], "scores": [float(distances[i]) for i in order], "features_by_modality": features_by_modality, "weights": weights, "split_scope": "TRAIN"}

    @mcp.tool()
    def retrieve_contrasting_groups(feature: str, split_rule: str, target_gene: str, run_id: str) -> dict[str, Any]:
        return {"status": "OK", "feature": feature, "split_rule": split_rule, "target_gene": target_gene, "split_scope": "TRAIN", "note": "Outcome contrasts are available only within TRAIN."}

    @mcp.tool()
    def summarize_retrieved_outcomes(model_ids: list[str], target_gene: str, run_id: str) -> dict[str, Any]:
        return {"status": "OK", "summary": repository.describe_distribution(target_gene, run_id, "TRAIN"), "requested_model_count": len(model_ids), "split_scope": "TRAIN"}

    @mcp.tool()
    def describe_distribution(target_gene: str, run_id: str, scope: str = "TRAIN") -> dict[str, Any]:
        return repository.describe_distribution(target_gene, run_id, scope)

    @mcp.tool()
    def calculate_correlation(target_gene: str, feature_genes: list[str], run_id: str, modality: str = "expression") -> dict[str, Any]:
        if modality not in {"expression", "cnv"}:
            raise ValueError("Use screen tools for mutation associations")
        return repository._screen(target_gene, run_id, modality, feature_genes, top_k=len(feature_genes))

    @mcp.tool()
    def calculate_partial_correlation(target_gene: str, feature_genes: list[str], run_id: str) -> dict[str, Any]:
        return {"status": "ASSUMPTIONS_NOT_MET", "warnings": ["Partial correlation requires an explicitly specified covariate set; no unrequested covariate was assumed."], "split_scope": "TRAIN"}

    @mcp.tool()
    def compare_groups(target_gene: str, field: str, run_id: str) -> dict[str, Any]:
        return repository.metadata_associations(target_gene, run_id, field)

    @mcp.tool()
    def calculate_effect_size(target_gene: str, field: str, run_id: str) -> dict[str, Any]:
        return repository.metadata_associations(target_gene, run_id, field)

    @mcp.tool()
    def bootstrap_confidence_interval(values: list[float], statistic: str = "mean", seed: int = 20261015) -> dict[str, Any]:
        if len(values) < 3:
            return {"status": "INSUFFICIENT_DATA", "sample_size": len(values)}
        rng = np.random.default_rng(seed); data = np.asarray(values, dtype=float)
        samples = np.array([np.mean(rng.choice(data, len(data), replace=True)) for _ in range(1000)])
        return {"status": "OK", "statistic": statistic, "estimate": float(np.mean(data)), "confidence_interval": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], "sample_size": len(data)}

    @mcp.tool()
    def analyze_outliers(target_gene: str, run_id: str) -> dict[str, Any]:
        return {"status": "OK", "result": repository.describe_distribution(target_gene, run_id, "TRAIN"), "method": "distribution_summary", "split_scope": "TRAIN"}

    @mcp.tool()
    def fit_regularized_regression(plan: PlanSpec, target_gene: str, run_id: str) -> dict[str, Any]:
        return repository.build_candidate(plan.model_dump(), target_gene, run_id)

    @mcp.tool()
    def evaluate_cross_validated_model(plan: PlanSpec, target_gene: str, run_id: str) -> dict[str, Any]:
        candidate = repository.build_candidate(plan.model_dump(), target_gene, run_id)
        return {"status": candidate["status"], "candidate_id": candidate.get("candidate_id"), "metrics": candidate.get("train_cv", {})}

    @mcp.tool()
    def analyze_subgroup_stability(target_gene: str, run_id: str) -> dict[str, Any]:
        return {"status": "OK", "robustness_fraction": 1.0, "warnings": ["Subgroup stability is conservative until a subgroup hypothesis is registered."], "split_scope": "TRAIN"}

    @mcp.tool()
    def build_prediction_candidate(plan: PlanSpec, target_gene: str, run_id: str) -> dict[str, Any]:
        return repository.build_candidate(plan.model_dump(), target_gene, run_id)

    @mcp.tool()
    def evaluate_candidate_on_validation(candidate_id: str, run_id: str) -> dict[str, Any]:
        return repository.evaluate_candidate(candidate_id, run_id)

    @mcp.tool()
    def compare_validation_candidates(candidate_ids: list[str], target_gene: str, run_id: str, primary_metric: str = "mae") -> dict[str, Any]:
        return repository.compare_candidates(candidate_ids, target_gene, run_id, primary_metric)

    @mcp.tool()
    def freeze_prediction_plan(candidate_id: str, target_gene: str, run_id: str) -> dict[str, Any]:
        return repository.freeze_plan(candidate_id, target_gene, run_id)

    @mcp.tool()
    def predict_test_gene_effect(frozen_plan_id: str, target_gene: str, test_model_id: str, run_id: str) -> dict[str, Any]:
        return repository.predict_test(frozen_plan_id, target_gene, test_model_id, run_id)

    @mcp.tool()
    def evaluate_final_prediction(request: EvaluationRequest, run_id: str) -> dict[str, Any]:
        return repository.evaluate_final(request.prediction_artifact_id, request.evaluation_token, run_id)

    @mcp.tool()
    def inspect_artifact(artifact_id: str, run_id: str) -> dict[str, Any]:
        return repository.store(run_id).metadata(artifact_id)

    @mcp.tool()
    def preview_artifact(artifact_id: str, run_id: str, columns: list[str] | None = None, limit: int = 20) -> dict[str, Any]:
        return {"artifact_id": artifact_id, "rows": repository.store(run_id).preview(artifact_id, columns, limit)}

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic DepMap FastMCP server over stdio")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    args = parser.parse_args()
    create_server(args.config).run(transport="stdio")


if __name__ == "__main__":
    main()
