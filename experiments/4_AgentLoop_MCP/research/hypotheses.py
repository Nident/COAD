from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Hypothesis:
    hypothesis_id: str
    target_gene: str
    test_model_id: str
    claim_type: str
    claim: str
    features: list[str] = field(default_factory=list)
    subgroup: str | None = None
    expected_direction: str | None = None
    motivation: str = ""
    train_evidence_ids: list[str] = field(default_factory=list)
    contradictory_evidence_ids: list[str] = field(default_factory=list)
    candidate_prediction_ids: list[str] = field(default_factory=list)
    validation_evidence_ids: list[str] = field(default_factory=list)
    status: str = "PROPOSED"
    remaining_questions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

