from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    run_id: str
    target_gene: str
    test_model_id: str
    config_path: str
    split_manifest_id: str
    iteration_count: int
    tool_call_count: int
    validation_submission_count: int
    token_usage: int
    research_status: str
    termination_reason: str
    phase: str
    next_action: dict[str, Any]
    last_result: dict[str, Any]
    tool_call_history: list[dict[str, Any]]
    completed_action_hashes: list[str]
    artifact_ids: list[str]
    evidence_ids: list[str]
    hypotheses: list[dict[str, Any]]
    candidate_prediction_ids: list[str]
    validated_candidate_ids: list[str]
    selected_candidate_id: str | None
    frozen_plan_id: str | None
    prediction_artifact_id: str | None
    results: dict[str, Any]
    decisions: list[dict[str, Any]]
    final_report: dict[str, Any]

