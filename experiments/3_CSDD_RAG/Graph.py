import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from langchain_core.prompt_values import PromptValue
from langgraph.graph import END, START, StateGraph

from LLM import LLMFactory
from Prompts import Prompts
from Schemas import (
    AnalystState,
    CNVEvidence,
    CNVOutput,
    CompletedGraphState,
    ExpressionEvidence,
    ExpressionOutput,
    GraphInput,
    GraphState,
    JudgeDecision,
    JudgeProposal,
    JudgeOutput,
    MetadataEvidence,
    MetadataOutput,
    MutationEvidence,
    MutationOutput,
)


class DependencyRAGGraph:
    def __init__(self, config_dir: Path):
        factory = LLMFactory(config_dir)
        self.prompts = Prompts(config_dir / "prompts.yaml")
        settings = cast(
            dict[str, Any],
            yaml.safe_load(
                (config_dir / "settings.yaml").read_text(encoding="utf-8")
            ),
        )
        prediction = cast(dict[str, Any], settings["prediction"])
        self.maximum_adjustment = float(
            prediction["max_absolute_llm_adjustment"]
        )
        self.thresholds = cast(dict[str, float], prediction["thresholds"])
        self.expression_llm = factory.create("expression", ExpressionEvidence)
        self.mutation_llm = factory.create("mutation", MutationEvidence)
        self.cnv_llm = factory.create("cnv", CNVEvidence)
        self.metadata_llm = factory.create("metadata", MetadataEvidence)
        self.judge_llm = factory.create("judge", JudgeProposal)

        builder = StateGraph(GraphState)
        builder.add_node("expression", self.expression)
        builder.add_node("mutation", self.mutation)
        builder.add_node("cnv", self.cnv)
        builder.add_node("metadata", self.metadata)
        builder.add_node("judge", self.judge)
        builder.add_edge(START, "expression")
        builder.add_edge("expression", "mutation")
        builder.add_edge("expression", "cnv")
        builder.add_edge("expression", "metadata")
        builder.add_edge(
            ["mutation", "cnv", "metadata"],
            "judge",
        )
        builder.add_edge("judge", END)
        self.graph = builder.compile()

    def _analyst_prompt(self, role: str, state: GraphState) -> PromptValue:
        field = {
            "expression": "expression_data",
            "mutation": "mutation_data",
            "cnv": "cnv_data",
            "metadata": "metadata_data",
        }[role]
        return self.prompts.analyst(
            role=role,
            gene=state["gene"],
            test_model_id=state["test_model_id"],
            baseline=state["baseline"],
            neighbours=state["neighbours"],
            modality_data=state[field],
        )

    def analyst_prompts(self, state: GraphInput) -> dict[str, str]:
        graph_state = cast(GraphState, state)
        return {
            role: self.prompts.text(self._analyst_prompt(role, graph_state))
            for role in ["expression", "mutation", "cnv", "metadata"]
        }

    def expression(self, state: GraphState) -> ExpressionOutput:
        prompt = self._analyst_prompt("expression", state)
        evidence = cast(ExpressionEvidence, self.expression_llm.invoke(prompt))
        return {
            "expression_prompt": self.prompts.text(prompt),
            "expression_evidence": evidence,
        }

    def mutation(self, state: GraphState) -> MutationOutput:
        prompt = self._analyst_prompt("mutation", state)
        evidence = cast(MutationEvidence, self.mutation_llm.invoke(prompt))
        return {
            "mutation_prompt": self.prompts.text(prompt),
            "mutation_evidence": evidence,
        }

    def cnv(self, state: GraphState) -> CNVOutput:
        prompt = self._analyst_prompt("cnv", state)
        evidence = cast(CNVEvidence, self.cnv_llm.invoke(prompt))
        return {
            "cnv_prompt": self.prompts.text(prompt),
            "cnv_evidence": evidence,
        }

    def metadata(self, state: GraphState) -> MetadataOutput:
        prompt = self._analyst_prompt("metadata", state)
        evidence = cast(MetadataEvidence, self.metadata_llm.invoke(prompt))
        return {
            "metadata_prompt": self.prompts.text(prompt),
            "metadata_evidence": evidence,
        }

    def judge(self, state: GraphState) -> JudgeOutput:
        complete = cast(AnalystState, state)
        prompt = self.prompts.judge(
            gene=complete["gene"],
            test_model_id=complete["test_model_id"],
            baseline=complete["baseline"],
            neighbours=complete["neighbours"],
            expression=complete["expression_evidence"],
            mutation=complete["mutation_evidence"],
            cnv=complete["cnv_evidence"],
            metadata=complete["metadata_evidence"],
        )
        proposal = cast(JudgeProposal, self.judge_llm.invoke(prompt))
        adjustment = float(np.clip(
            proposal.suggested_adjustment,
            -self.maximum_adjustment,
            self.maximum_adjustment,
        ))
        baseline_data = cast(dict[str, Any], json.loads(state["baseline"]))
        baseline = float(baseline_data["weighted_prediction"])
        uncertainty = max(
            proposal.uncertainty,
            float(baseline_data["standard_deviation"]),
        )
        predicted = baseline + adjustment
        category = (
            "strong_effect"
            if predicted <= self.thresholds["strong_effect"]
            else "weak_effect"
            if predicted <= self.thresholds["weak_effect"]
            else "no_effect"
        )
        decision = JudgeDecision(
            **proposal.model_dump(exclude={
                "baseline_gene_effect",
                "suggested_adjustment",
                "uncertainty",
            }),
            baseline_gene_effect=baseline,
            suggested_adjustment=adjustment,
            uncertainty=uncertainty,
            predicted_gene_effect=predicted,
            final_category=category,
        )
        return {
            "judge_prompt": self.prompts.text(prompt),
            "judge_decision": decision,
        }

    def invoke(self, state: GraphInput) -> CompletedGraphState:
        return cast(
            CompletedGraphState,
            self.graph.invoke(cast(GraphState, state)),
        )
