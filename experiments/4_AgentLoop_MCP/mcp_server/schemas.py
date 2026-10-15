from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlanSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["multiomics_knn", "expression_ridge", "cnv_ridge", "mutation_ridge", "metadata_ridge"]
    features_by_modality: dict[str, list[str]] = Field(default_factory=dict)
    alpha: float = Field(default=10.0, gt=0)
    k: int = Field(default=10, ge=1, le=100)
    feature_selection: dict[str, object] = Field(default_factory=lambda: {"nested": True, "top_k": 20})


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_artifact_id: str
    evaluation_token: str

