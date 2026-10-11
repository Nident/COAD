import json
from pathlib import Path
from typing import Any, cast

import yaml
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from Schemas import (
    CNVEvidence,
    ExpressionEvidence,
    JudgeProposal,
    MetadataEvidence,
    MutationEvidence,
)


class Prompts:
    ANALYST_SCHEMAS: dict[str, type[BaseModel]] = {
        "expression": ExpressionEvidence,
        "mutation": MutationEvidence,
        "cnv": CNVEvidence,
        "metadata": MetadataEvidence,
    }

    def __init__(self, path: Path):
        self.config = cast(
            dict[str, str],
            yaml.safe_load(path.read_text(encoding="utf-8")),
        )

    @staticmethod
    def _example(role: str) -> dict[str, Any]:
        if role == "metadata":
            return {
                "source": "metadata",
                "conclusion": "One concise transferability conclusion",
                "transferability": "medium",
                "confidence": 0.5,
                "evidence_model_ids": ["ACH-XXXXXX"],
                "evidence": ["One concrete observation"],
                "limitations": ["One limitation"],
            }
        return {
            "source": role,
            "conclusion": "One concise evidence conclusion",
            "direction": "none",
            "correction_strength": "none",
            "confidence": 0.5,
            "evidence_model_ids": ["ACH-XXXXXX"],
            "evidence": ["One concrete observation"],
            "limitations": ["One limitation"],
        }

    def analyst(
        self,
        role: str,
        gene: str,
        test_model_id: str,
        baseline: str,
        neighbours: str,
        modality_data: str,
    ) -> PromptValue:
        schema = self.ANALYST_SCHEMAS[role]
        template = ChatPromptTemplate.from_messages([
            ("system", self.config["system"]),
            ("human", self.config[role]),
        ])
        return template.invoke({
            "gene": gene,
            "test_model_id": test_model_id,
            "baseline": baseline,
            "neighbours": neighbours,
            "modality_data": modality_data,
            "response_schema": json.dumps(
                schema.model_json_schema(), ensure_ascii=False
            ),
            "response_example": json.dumps(
                self._example(role), ensure_ascii=False
            ),
        })

    def judge(
        self,
        gene: str,
        test_model_id: str,
        baseline: str,
        neighbours: str,
        expression: ExpressionEvidence,
        mutation: MutationEvidence,
        cnv: CNVEvidence,
        metadata: MetadataEvidence,
    ) -> PromptValue:
        template = ChatPromptTemplate.from_messages([
            ("system", self.config["system"]),
            ("human", self.config["judge"]),
        ])
        example = {
            "baseline_gene_effect": -0.7,
            "suggested_adjustment": 0.0,
            "uncertainty": 0.2,
            "confidence": 0.7,
            "explanation": "The local evidence supports the numerical baseline.",
            "decisive_evidence": ["Top-K outcomes are consistent."],
        }
        return template.invoke({
            "gene": gene,
            "test_model_id": test_model_id,
            "baseline": baseline,
            "neighbours": neighbours,
            "expression_evidence": expression.model_dump_json(),
            "mutation_evidence": mutation.model_dump_json(),
            "cnv_evidence": cnv.model_dump_json(),
            "metadata_evidence": metadata.model_dump_json(),
            "response_schema": json.dumps(
                JudgeProposal.model_json_schema(), ensure_ascii=False
            ),
            "response_example": json.dumps(example, ensure_ascii=False),
        })

    @staticmethod
    def text(prompt: PromptValue) -> str:
        return "\n\n".join(
            f"## {message.type}\n\n{message.content}"
            for message in prompt.to_messages()
        )
