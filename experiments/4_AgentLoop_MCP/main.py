from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.graph import MCPResearchAgent
from agent.report import save_report
from agent.state import ResearchState
from data.manifests import load_config


def new_run_id(gene: str, model_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(f"{gene}:{model_id}:{stamp}".encode()).hexdigest()[:10]
    return f"run_{stamp}_{suffix}"


def report_for(state: ResearchState, agent: MCPResearchAgent) -> dict[str, Any]:
    prediction = state.get("results", {}).get("predict_test_gene_effect", {})
    if not prediction.get("prediction_artifact_id"):
        raise RuntimeError(f"Run ended without a persisted prediction: {state.get('termination_reason')}")
    root = agent.config_path.parent
    trajectories = (root / agent.config["artifacts"]["trajectories"]).resolve()
    results_root = (root / agent.config["artifacts"]["results"]).resolve()
    validation = [event.get("result", {}) for event in agent.trajectory.read() if event.get("event") == "tool_call" and event.get("tool") == "evaluate_candidate_on_validation"]
    report_data = {
        "target_gene": state["target_gene"], "test_model_id": state["test_model_id"], "split_manifest_id": state.get("results", {}).get("initialize_run", {}).get("split_manifest_id", "unknown"),
        "prediction_status": state.get("research_status", "COMPLETE"), "selected_or_fallback_plan_id": prediction["frozen_plan_id"], "predicted_gene_effect": prediction["predicted_gene_effect"], "uncertainty_interval": prediction["uncertainty_interval"], "confidence": prediction["confidence"], "effect_category": prediction["effect_category"], "fallback_used": prediction["fallback_used"],
        "validation_metrics": validation, "findings": [{"evidence_id": evidence_id, "claim": "A deterministic scoped analysis was registered during research."} for evidence_id in state.get("evidence_ids", [])], "contradictory_hypotheses": [hypothesis for hypothesis in state.get("hypotheses", []) if hypothesis.get("status") in {"CONTRADICTED", "INCONCLUSIVE"}], "limitations": ["Associations are observational and TRAIN-only unless explicitly labelled validation-supported.", "Uncertainty is calibrated from deterministic cross-validation residuals; it is not a biological confidence statement."], "termination_reason": state.get("termination_reason", "PREDICTION_PERSISTED"), "evidence_ids": state.get("evidence_ids", []), "artifact_ids": state.get("artifact_ids", []) + [prediction["prediction_artifact_id"]], "trajectory_path": str(trajectories / f"{state['run_id']}.jsonl"),
    }
    json_path = results_root / f"{state['run_id']}.json"
    markdown_path = results_root / f"{state['run_id']}.md"
    report = save_report(report_data, json_path, markdown_path)
    return {"report": report.model_dump(), "json_path": str(json_path), "markdown_path": str(markdown_path)}


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = load_config(config_path)
    run_id = args.resume or new_run_id(args.gene, args.test_model_id)
    if args.resume:
        trajectory_path = (config_path.parent / config["artifacts"]["trajectories"] / f"{run_id}.jsonl").resolve()
        existing = trajectory_path.read_text(encoding="utf-8") if trajectory_path.exists() else ""
        if '"event":"run_complete"' in existing and '"status":"COMPLETE"' in existing:
            report_path = (config_path.parent / config["artifacts"]["results"] / f"{run_id}.json").resolve()
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                output = {"run_id": run_id, "resumed_without_duplicate_calls": True, "report": report, "json_path": str(report_path)}
                print(json.dumps(output, indent=2, default=str))
                return output
    agent = MCPResearchAgent(config_path, run_id, args.gene, args.test_model_id, resume=bool(args.resume))
    state = await agent.run(agent.resume_state() if args.resume else None)
    report = report_for(state, agent)
    output = {"run_id": run_id, "prediction": state.get("results", {}).get("predict_test_gene_effect", {}), **report}
    print(json.dumps(output, indent=2, default=str))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous DepMap Gene Effect research agent")
    parser.add_argument("--gene", default="ACO2")
    parser.add_argument("--test-model-id", default="ACH-000475")
    parser.add_argument("--resume")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
