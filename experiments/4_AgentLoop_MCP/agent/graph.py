from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import ToolException
from langgraph.graph import END, START, StateGraph

from research.evidence import register
from research.hypotheses import Hypothesis
from research.trajectory import Trajectory
from .report import save_report
from .state import ResearchState


class MCPResearchAgent:
    def __init__(self, config_path: Path, run_id: str, target_gene: str, test_model_id: str, resume: bool = False) -> None:
        self.config_path = config_path.resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.run_id, self.target_gene, self.test_model_id = run_id, target_gene, test_model_id
        root = self.config_path.parent
        self.trajectory = Trajectory((root / self.config["artifacts"]["trajectories"] / f"{run_id}.jsonl").resolve())
        self.start_time = time.monotonic()
        self.tool_map: dict[str, Any] = {}
        self.llm: Any = None
        python_executable = os.environ.get("AGENT_PYTHON") or sys.executable
        self.client = MultiServerMCPClient({"depmap": {"command": python_executable, "args": ["-m", "mcp_server.server", "--config", str(self.config_path)], "transport": "stdio"}}, tool_name_prefix=False)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ResearchState)
        builder.add_node("research_model", self.research_model)
        builder.add_node("execute_mcp_tools", self.execute_mcp_tools)
        builder.add_node("ingest_evidence", self.ingest_evidence)
        builder.add_node("prediction_plan_controller", self.controller)
        builder.add_edge(START, "research_model")
        builder.add_edge("research_model", "execute_mcp_tools")
        builder.add_edge("execute_mcp_tools", "ingest_evidence")
        builder.add_edge("ingest_evidence", "prediction_plan_controller")
        builder.add_conditional_edges("prediction_plan_controller", self.route, {"continue": "research_model", "finish": END})
        return builder.compile()

    async def connect(self) -> None:
        tools = await self.client.get_tools(server_name="depmap")
        self.tool_map = {tool.name: tool for tool in tools}
        self.trajectory.append("tool_discovery", {"run_id": self.run_id, "tools": sorted(self.tool_map)})

    def initial_state(self) -> ResearchState:
        return {"run_id": self.run_id, "target_gene": self.target_gene, "test_model_id": self.test_model_id, "config_path": str(self.config_path), "iteration_count": 0, "tool_call_count": 0, "validation_submission_count": 0, "token_usage": 0, "research_status": "RUNNING", "phase": "research", "tool_call_history": [], "completed_action_hashes": [], "artifact_ids": [], "evidence_ids": [], "hypotheses": [], "candidate_prediction_ids": [], "validated_candidate_ids": [], "selected_candidate_id": None, "frozen_plan_id": None, "prediction_artifact_id": None, "results": {}, "decisions": [], "final_report": {}}

    @staticmethod
    def _hash(tool: str, args: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps([tool, args], sort_keys=True, default=str).encode()).hexdigest()

    def _budget_exhausted(self, state: ResearchState) -> bool:
        research = self.config["research"]
        return state.get("iteration_count", 0) >= int(research["max_iterations"]) or state.get("tool_call_count", 0) >= int(research["max_tool_calls"]) or time.monotonic() - self.start_time >= float(research["max_seconds"])

    def _policy_action(self, state: ResearchState) -> dict[str, Any]:
        phase = state.get("phase", "research")
        results = state.get("results", {})
        if phase == "research" and "initialize_run" not in results:
            return {"tool": "initialize_run", "arguments": {"target_gene": self.target_gene, "test_model_id": self.test_model_id, "run_id": self.run_id}}
        if phase == "research" and "get_target_gene_summary" not in results:
            return {"tool": "get_target_gene_summary", "arguments": {"target_gene": self.target_gene, "run_id": self.run_id}}
        if phase == "research" and "get_test_model_profile" not in results:
            return {"tool": "get_test_model_profile", "arguments": {"target_gene": self.target_gene, "test_model_id": self.test_model_id, "modalities": ["expression", "mutation", "cnv", "metadata"], "run_id": self.run_id}}
        if phase == "research" and "screen_expression_dependency_associations" not in results:
            return {"tool": "screen_expression_dependency_associations", "arguments": {"target_gene": self.target_gene, "run_id": self.run_id, "top_k": 20}}
        expression = results.get("screen_expression_dependency_associations", {})
        rows = expression.get("rows", []) if isinstance(expression, dict) else []
        strong = [row for row in rows if abs(float(row.get("estimate") or 0.0)) >= float(self.config["research"]["minimum_effect_size"])]
        if strong and "expression_candidate" not in results:
            genes = [row["feature"] for row in strong[:12]]
            return {"tool": "build_prediction_candidate", "label": "expression_candidate", "arguments": {"target_gene": self.target_gene, "run_id": self.run_id, "plan": {"strategy": "expression_ridge", "features_by_modality": {"expression": genes}, "alpha": 10.0, "feature_selection": {"nested": True, "top_k": min(12, len(genes))}}}}
        if not strong and "screen_cnv_dependency_associations" not in results:
            return {"tool": "screen_cnv_dependency_associations", "arguments": {"target_gene": self.target_gene, "run_id": self.run_id, "top_k": 12}}
        if "baseline_candidate" not in results:
            return {"tool": "build_prediction_candidate", "label": "baseline_candidate", "arguments": {"target_gene": self.target_gene, "run_id": self.run_id, "plan": {"strategy": "multiomics_knn", "features_by_modality": {"expression": [self.target_gene], "cnv": [self.target_gene]}, "k": int(self.config["research"].get("knn_k", 10)), "alpha": 1.0, "feature_selection": {"nested": False, "top_k": 2}}}}
        candidates = state.get("candidate_prediction_ids", [])
        validated = state.get("validated_candidate_ids", [])
        if len(validated) < len(candidates):
            candidate_id = candidates[len(validated)]
            return {"tool": "evaluate_candidate_on_validation", "arguments": {"candidate_id": candidate_id, "run_id": self.run_id}}
        if "compare_validation_candidates" not in results:
            return {"tool": "compare_validation_candidates", "arguments": {"candidate_ids": candidates, "target_gene": self.target_gene, "run_id": self.run_id, "primary_metric": "mae"}}
        if "freeze_prediction_plan" not in results:
            best = results.get("compare_validation_candidates", {}).get("best_candidate_id") or (candidates[0] if candidates else "")
            return {"tool": "freeze_prediction_plan", "arguments": {"candidate_id": best, "target_gene": self.target_gene, "run_id": self.run_id}}
        if "predict_test_gene_effect" not in results:
            frozen = results.get("freeze_prediction_plan", {}).get("frozen_plan_id")
            return {"tool": "predict_test_gene_effect", "arguments": {"frozen_plan_id": frozen, "target_gene": self.target_gene, "test_model_id": self.test_model_id, "run_id": self.run_id}}
        return {"tool": "__finish__", "arguments": {}}

    def _model_action(self, state: ResearchState) -> dict[str, Any] | None:
        if self.llm is None:
            return None
        prompt_config = yaml.safe_load((self.config_path.parent / "prompts.yaml").read_text(encoding="utf-8")) or {}
        compact_state = {key: state.get(key) for key in ("target_gene", "test_model_id", "phase", "results", "candidate_prediction_ids", "validated_candidate_ids", "evidence_ids")}
        response = self.llm.invoke([SystemMessage(content=str(prompt_config.get("system", ""))), HumanMessage(content=json.dumps(compact_state, default=str))])
        calls = getattr(response, "tool_calls", [])
        if not calls:
            return None
        call = calls[0]
        name = str(call.get("name", "")); arguments = call.get("args", {})
        if name not in self.tool_map:
            return None
        return {"tool": name, "arguments": arguments}

    def research_model(self, state: ResearchState) -> ResearchState:
        if self._budget_exhausted(state):
            return {"next_action": {"tool": "__finish__", "arguments": {}}, "termination_reason": "BUDGET_EXHAUSTED", "research_status": "BASELINE_FALLBACK_SELECTED"}
        action = self._model_action(state) or self._policy_action(state)
        decision = {"current_question": "Which measured relationship can improve held-out prediction?", "selected_action": action["tool"], "reason": "The action adds a new scoped statistic or tests a frozen deterministic plan.", "evidence_ids": state.get("evidence_ids", [])[-5:], "result": "pending"}
        self.trajectory.append("decision", {"run_id": self.run_id, **decision})
        return {"next_action": action, "decisions": [*state.get("decisions", []), decision], "iteration_count": state.get("iteration_count", 0) + 1}

    async def execute_mcp_tools(self, state: ResearchState) -> ResearchState:
        action = state.get("next_action", {"tool": "__finish__", "arguments": {}})
        if action["tool"] == "__finish__":
            return {"last_result": {"status": "FINISHED"}}
        tool_name, arguments = action["tool"], action["arguments"]
        action_hash = self._hash(tool_name, arguments)
        if action_hash in state.get("completed_action_hashes", []):
            return {"last_result": {"status": "CACHED_ACTION", "tool": tool_name}}
        if tool_name not in self.tool_map:
            return {"last_result": {"status": "EXECUTION_ERROR", "error": f"MCP tool not discovered: {tool_name}"}}
        try:
            value = await self.tool_map[tool_name].ainvoke(arguments)
            # langchain-mcp-adapters may expose MCP text content as a list
            # under ``value``.  Decode the server's structured JSON here so
            # policy decisions use the deterministic result, not a wrapper.
            if isinstance(value, dict) and isinstance(value.get("value"), list):
                content = value["value"]
                if content and isinstance(content[0], dict) and isinstance(content[0].get("text"), str):
                    value = content[0]["text"]
            if isinstance(value, list) and value and isinstance(value[0], dict) and isinstance(value[0].get("text"), str):
                value = value[0]["text"]
            if isinstance(value, str):
                try: value = json.loads(value)
                except json.JSONDecodeError: value = {"status": "OK", "text": value}
            result = cast(dict[str, Any], value) if isinstance(value, dict) else {"status": "OK", "value": value}
            result["tool_call_id"] = action_hash[:16]
        except (TimeoutError, OSError, ValueError, PermissionError, ToolException) as exc:
            result = {"status": "RECOVERABLE_ERROR", "error": str(exc), "tool": tool_name, "tool_call_id": action_hash[:16]}
        compact = {key: value for key, value in result.items() if key not in {"rows", "profile"}}
        if isinstance(result.get("rows"), list):
            compact["rows"] = result["rows"][:20]
        if isinstance(result.get("profile"), dict):
            compact["profile"] = result["profile"]
        self.trajectory.append("tool_call", {"run_id": self.run_id, "tool": tool_name, "label": action.get("label"), "arguments": arguments, "result": compact, "action_hash": action_hash})
        return {"last_result": result, "tool_call_count": state.get("tool_call_count", 0) + 1, "completed_action_hashes": [*state.get("completed_action_hashes", []), action_hash]}

    def ingest_evidence(self, state: ResearchState) -> ResearchState:
        action = state.get("next_action", {})
        result = state.get("last_result", {})
        results = dict(state.get("results", {}))
        label = str(action.get("label", action.get("tool", "unknown")))
        results[label] = result
        artifacts = [*state.get("artifact_ids", [])]
        if result.get("artifact_id"):
            artifacts.append(str(result["artifact_id"]))
        candidates = list(state.get("candidate_prediction_ids", []))
        if action.get("tool") == "build_prediction_candidate" and result.get("candidate_id"):
            candidates.append(str(result["candidate_id"]))
            results[action.get("label", "candidate")] = result
        validated = list(state.get("validated_candidate_ids", []))
        if action.get("tool") == "evaluate_candidate_on_validation" and result.get("candidate_id") and result.get("status") == "OK":
            validated.append(str(result["candidate_id"]))
        evidence = list(state.get("evidence_ids", []))
        if action.get("tool") not in {"initialize_run", "get_test_model_profile", "__finish__"} and result.get("status") in {"OK", "INSUFFICIENT_DATA"}:
            hyp_id = "hypothesis_primary"
            if not any(h.get("hypothesis_id") == hyp_id for h in state.get("hypotheses", [])):
                hypothesis = Hypothesis(hyp_id, self.target_gene, self.test_model_id, "predictive_association", f"Measured molecular features are associated with {self.target_gene} dependency and may improve prediction.", motivation="TRAIN screening and controlled validation", status="UNDER_TEST")
                hypotheses = [hypothesis.as_dict()]
            else:
                hypotheses = state.get("hypotheses", [])
            record = register(hyp_id, action["tool"], result)
            if record["evidence_id"] not in evidence:
                evidence.append(record["evidence_id"])
        else:
            hypotheses = state.get("hypotheses", [])
        if action.get("tool") == "freeze_prediction_plan" and result.get("frozen_plan_id"):
            results["freeze_prediction_plan"] = result
        if action.get("tool") == "predict_test_gene_effect" and result.get("prediction_artifact_id"):
            results["predict_test_gene_effect"] = result
        return {"results": results, "artifact_ids": artifacts, "candidate_prediction_ids": candidates, "validated_candidate_ids": validated, "evidence_ids": evidence, "hypotheses": hypotheses, "validation_submission_count": len(validated)}

    def controller(self, state: ResearchState) -> ResearchState:
        if state.get("prediction_artifact_id") or state.get("results", {}).get("predict_test_gene_effect", {}).get("prediction_artifact_id"):
            return {"research_status": "COMPLETE", "termination_reason": "PREDICTION_PERSISTED", "prediction_artifact_id": state.get("results", {}).get("predict_test_gene_effect", {}).get("prediction_artifact_id")}
        return {}

    def route(self, state: ResearchState) -> str:
        if state.get("next_action", {}).get("tool") == "__finish__" or state.get("research_status") == "COMPLETE":
            return "finish"
        if state.get("termination_reason") == "BUDGET_EXHAUSTED":
            return "finish"
        return "continue"

    async def run(self, initial: ResearchState | None = None) -> ResearchState:
        state = initial or self.initial_state()
        # MultiServerMCPClient normally opens a new stdio session for each
        # tool call.  Use one explicit session for a run so the repository's
        # in-process caches are effective and the trajectory is one client
        # conversation with the server.
        async with self.client.session("depmap") as session:
            tools = await load_mcp_tools(session)
            self.tool_map = {tool.name: tool for tool in tools}
            if bool(self.config.get("models", {}).get("use_llm", False)):
                from langchain_openai import ChatOpenAI
                model_name = str(self.config.get("models", {}).get("name", "gpt-4o-mini"))
                self.llm = ChatOpenAI(model=model_name, temperature=0, api_key=os.environ.get("OPENAI_API_KEY"), timeout=60, max_retries=1).bind_tools(tools)
            self.trajectory.append("tool_discovery", {"run_id": self.run_id, "tools": sorted(self.tool_map)})
            final = await self.graph.ainvoke(state, config={"recursion_limit": int(self.config["research"]["max_iterations"]) * 5})
        final = cast(ResearchState, final)
        self.trajectory.append("run_complete", {"run_id": self.run_id, "status": final.get("research_status"), "termination_reason": final.get("termination_reason"), "prediction_artifact_id": final.get("prediction_artifact_id")})
        return final

    def resume_state(self) -> ResearchState:
        """Rehydrate only serialized state and completed action hashes."""
        state = self.initial_state()
        results: dict[str, Any] = {}
        completed: list[str] = []
        candidates: list[str] = []
        validated: list[str] = []
        artifacts: list[str] = []
        prediction_artifact: str | None = None
        for event in self.trajectory.read():
            if event.get("event") != "tool_call":
                continue
            tool = str(event.get("tool", "")); result = event.get("result", {})
            action_hash = event.get("action_hash")
            if action_hash and action_hash not in completed:
                completed.append(action_hash)
            results[str(event.get("label") or tool)] = result
            if result.get("artifact_id"):
                artifacts.append(str(result["artifact_id"]))
            if tool == "build_prediction_candidate" and result.get("candidate_id"):
                candidates.append(str(result["candidate_id"]))
            if tool == "evaluate_candidate_on_validation" and result.get("candidate_id"):
                validated.append(str(result["candidate_id"]))
            if tool == "predict_test_gene_effect":
                prediction_artifact = result.get("prediction_artifact_id")
        state.update({"results": results, "completed_action_hashes": completed, "candidate_prediction_ids": candidates, "validated_candidate_ids": validated, "artifact_ids": artifacts, "prediction_artifact_id": prediction_artifact, "iteration_count": len(completed), "tool_call_count": len(completed), "validation_submission_count": len(validated)})
        return state

