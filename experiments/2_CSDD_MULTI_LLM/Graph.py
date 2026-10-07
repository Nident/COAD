from pathlib import Path
from typing import cast

from langgraph.graph import END, START, StateGraph

from LLM import LLMFactory
from Prompts import Prompts
from Schemas import (
    AnalystState,
    CompletedGraphState,
    DonorOutput,
    DonorHypothesis,
    ExpressionOutput,
    ExpressionHypothesis,
    FinalVerdict,
    GraphInput,
    GraphState,
    JudgeOutput,
    MutationOutput,
    MutationHypothesis,
    NoMutationHypothesis,)


class DependencyGraph:
    def __init__(self, config_dir: Path):
        factory = LLMFactory(config_dir)
        self.prompts = Prompts(config_dir / "prompts.yaml")
        self.donor_llm = factory.create("donor", DonorHypothesis)
        self.expression_llm = factory.create("expression", ExpressionHypothesis)
        self.mutation_llm = factory.create("mutation", MutationHypothesis)
        self.judge_llm = factory.create("judge", FinalVerdict)

        builder = StateGraph(GraphState)
        builder.add_node("donor", self.donor)
        builder.add_node("expression", self.expression)
        builder.add_node("mutation", self.mutation)
        builder.add_node("judge", self.judge)
        builder.add_edge(START, "donor")
        builder.add_edge(START, "expression")
        builder.add_edge(START, "mutation")
        builder.add_edge(["donor", "expression", "mutation"], "judge")
        builder.add_edge("judge", END)
        self.graph = builder.compile()

    def analyst_prompts(self, state: GraphInput) -> dict[str, str]:
        prompts = {
            "donor": self.prompts.text(self.prompts.analyst(
                role="donor",
                gene=state["gene"],
                test_model_id=state["test_model_id"],
                train_crispr=state["train_crispr"],
                train_table=state["train_donor"],
                test_table=state["test_donor"],
            )),
            "expression": self.prompts.text(self.prompts.analyst(
                role="expression",
                gene=state["gene"],
                test_model_id=state["test_model_id"],
                train_crispr=state["train_crispr"],
                train_table=state["train_expression"],
                test_table=state["test_expression"],
            )),
        }
        if state["test_has_mutation"]:
            prompts["mutation"] = self.prompts.text(self.prompts.analyst(
                role="mutation",
                gene=state["gene"],
                test_model_id=state["test_model_id"],
                train_crispr=state["train_crispr"],
                train_table=state["train_mutation"],
                test_table=state["test_mutation"],
            ))
        return prompts

    def donor(self, state: GraphState) -> DonorOutput:
        prompt = self.prompts.analyst(
            role="donor",
            gene=state["gene"],
            test_model_id=state["test_model_id"],
            train_crispr=state["train_crispr"],
            train_table=state["train_donor"],
            test_table=state["test_donor"],
        )
        hypothesis = cast(DonorHypothesis, self.donor_llm.invoke(prompt))
        return {
            "donor_prompt": self.prompts.text(prompt),
            "donor_hypothesis": hypothesis,
        }

    def expression(self, state: GraphState) -> ExpressionOutput:
        prompt = self.prompts.analyst(
            role="expression",
            gene=state["gene"],
            test_model_id=state["test_model_id"],
            train_crispr=state["train_crispr"],
            train_table=state["train_expression"],
            test_table=state["test_expression"],
        )
        hypothesis = cast(ExpressionHypothesis, self.expression_llm.invoke(prompt))
        return {
            "expression_prompt": self.prompts.text(prompt),
            "expression_hypothesis": hypothesis,
        }

    def mutation(self, state: GraphState) -> MutationOutput:
        if not state["test_has_mutation"]:
            message = (
                "No mutations in the target gene or its related genes were "
                "detected for this cell model in the supplied data."
            )
            return {
                "mutation_prompt": message,
                "mutation_hypothesis": NoMutationHypothesis(
                    source="mutation",
                    hypothesis=message,
                    reasoning=[
                        "The supplied mutation table has no rows for the target "
                        "or related genes and held-out ModelID."
                    ],
                    predicted_category=None,
                    confidence=0.0,
                    evidence_model_ids=[],
                    limitations=[
                        "The absence of a supplied mutation record is not proof "
                        "that the cell model is genetically wild type."
                    ],
                ),
            }

        prompt = self.prompts.analyst(
            role="mutation",
            gene=state["gene"],
            test_model_id=state["test_model_id"],
            train_crispr=state["train_crispr"],
            train_table=state["train_mutation"],
            test_table=state["test_mutation"],
        )
        hypothesis = cast(MutationHypothesis, self.mutation_llm.invoke(prompt))
        return {
            "mutation_prompt": self.prompts.text(prompt),
            "mutation_hypothesis": hypothesis,
        }

    def judge(self, state: GraphState) -> JudgeOutput:
        analyst_state = cast(AnalystState, state)
        prompt = self.prompts.judge(
            gene=analyst_state["gene"],
            test_model_id=analyst_state["test_model_id"],
            donor=analyst_state["donor_hypothesis"],
            expression=analyst_state["expression_hypothesis"],
            mutation=analyst_state["mutation_hypothesis"],
        )
        verdict = cast(FinalVerdict, self.judge_llm.invoke(prompt))
        return {
            "judge_prompt": self.prompts.text(prompt),
            "verdict": verdict,
        }

    def invoke(self, state: GraphInput) -> CompletedGraphState:
        return cast(
            CompletedGraphState,
            self.graph.invoke(cast(GraphState, state)),
        )
