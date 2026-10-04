from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field


EffectCategory = Literal["no_effect", "weak_effect", "strong_effect"]


class Hypothesis(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    hypothesis: str
    reasoning: list[str]
    evidence_model_ids: list[str]
    limitations: list[str]


class DonorHypothesis(Hypothesis):
    source: Literal["donor"]
    predicted_category: EffectCategory
    confidence: float = Field(ge=0.0, le=1.0)


class ExpressionHypothesis(Hypothesis):
    source: Literal["expression"]
    predicted_category: EffectCategory
    confidence: float = Field(ge=0.0, le=1.0)


class MutationHypothesis(Hypothesis):
    source: Literal["mutation"]
    predicted_category: EffectCategory
    confidence: float = Field(ge=0.0, le=1.0)


class NoMutationHypothesis(Hypothesis):
    source: Literal["mutation"]
    predicted_category: None
    confidence: float = Field(ge=0.0, le=0.0)


MutationResult = MutationHypothesis | NoMutationHypothesis


class FinalVerdict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    gene: str
    model_id: str
    predicted_category: EffectCategory
    confidence: float = Field(ge=0.0, le=1.0)
    conclusion: str
    agreements: list[str]
    conflicts: list[str]
    decisive_evidence: list[str]


class GraphInput(TypedDict):
    gene: str
    test_model_id: str
    train_crispr: str
    train_donor: str
    train_expression: str
    train_mutation: str
    test_donor: str
    test_expression: str
    test_mutation: str
    test_has_mutation: bool


class GraphState(GraphInput):
    donor_prompt: NotRequired[str]
    expression_prompt: NotRequired[str]
    mutation_prompt: NotRequired[str]
    judge_prompt: NotRequired[str]
    donor_hypothesis: NotRequired[DonorHypothesis]
    expression_hypothesis: NotRequired[ExpressionHypothesis]
    mutation_hypothesis: NotRequired[MutationResult]
    verdict: NotRequired[FinalVerdict]


class AnalystState(GraphInput):
    donor_prompt: str
    expression_prompt: str
    mutation_prompt: str
    donor_hypothesis: DonorHypothesis
    expression_hypothesis: ExpressionHypothesis
    mutation_hypothesis: MutationResult


class CompletedGraphState(AnalystState):
    judge_prompt: str
    verdict: FinalVerdict


class DonorOutput(TypedDict):
    donor_prompt: str
    donor_hypothesis: DonorHypothesis


class ExpressionOutput(TypedDict):
    expression_prompt: str
    expression_hypothesis: ExpressionHypothesis


class MutationOutput(TypedDict):
    mutation_prompt: str
    mutation_hypothesis: MutationResult


class JudgeOutput(TypedDict):
    judge_prompt: str
    verdict: FinalVerdict
