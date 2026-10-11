from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field


EffectCategory = Literal["no_effect", "weak_effect", "strong_effect"]
Direction = Literal[
    "increase_dependency",
    "decrease_dependency",
    "none",
    "unclear",
]
CorrectionStrength = Literal["none", "weak", "moderate", "strong"]


class ModalityEvidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    conclusion: str
    direction: Direction
    correction_strength: CorrectionStrength
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_model_ids: list[str]
    evidence: list[str]
    limitations: list[str]


class ExpressionEvidence(ModalityEvidence):
    source: Literal["expression"] = "expression"


class MutationEvidence(ModalityEvidence):
    source: Literal["mutation"] = "mutation"


class CNVEvidence(ModalityEvidence):
    source: Literal["cnv"] = "cnv"


class MetadataEvidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    source: Literal["metadata"] = "metadata"
    conclusion: str
    transferability: Literal["high", "medium", "low", "unclear"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_model_ids: list[str]
    evidence: list[str]
    limitations: list[str]


class JudgeProposal(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    baseline_gene_effect: float
    suggested_adjustment: float
    uncertainty: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    decisive_evidence: list[str]


class JudgeDecision(JudgeProposal):
    predicted_gene_effect: float
    final_category: EffectCategory


class GraphInput(TypedDict):
    gene: str
    test_model_id: str
    baseline: str
    neighbours: str
    expression_data: str
    mutation_data: str
    cnv_data: str
    metadata_data: str


class GraphState(GraphInput):
    expression_prompt: NotRequired[str]
    mutation_prompt: NotRequired[str]
    cnv_prompt: NotRequired[str]
    metadata_prompt: NotRequired[str]
    judge_prompt: NotRequired[str]
    expression_evidence: NotRequired[ExpressionEvidence]
    mutation_evidence: NotRequired[MutationEvidence]
    cnv_evidence: NotRequired[CNVEvidence]
    metadata_evidence: NotRequired[MetadataEvidence]
    judge_decision: NotRequired[JudgeDecision]


class AnalystState(GraphInput):
    expression_prompt: str
    mutation_prompt: str
    cnv_prompt: str
    metadata_prompt: str
    expression_evidence: ExpressionEvidence
    mutation_evidence: MutationEvidence
    cnv_evidence: CNVEvidence
    metadata_evidence: MetadataEvidence


class CompletedGraphState(AnalystState):
    judge_prompt: str
    judge_decision: JudgeDecision


class ExpressionOutput(TypedDict):
    expression_prompt: str
    expression_evidence: ExpressionEvidence


class MutationOutput(TypedDict):
    mutation_prompt: str
    mutation_evidence: MutationEvidence


class CNVOutput(TypedDict):
    cnv_prompt: str
    cnv_evidence: CNVEvidence


class MetadataOutput(TypedDict):
    metadata_prompt: str
    metadata_evidence: MetadataEvidence


class JudgeOutput(TypedDict):
    judge_prompt: str
    judge_decision: JudgeDecision
