import json
from pathlib import Path

import yaml
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate

from Schemas import (
    DonorHypothesis,
    ExpressionHypothesis,
    FinalVerdict,
    MutationHypothesis,
    MutationResult,
)


class Prompts:
    ANALYST_SCHEMAS = {
        "donor": DonorHypothesis,
        "expression": ExpressionHypothesis,
        "mutation": MutationHypothesis,
    }

    def __init__(self, config_path: Path):
        self.config: dict[str, str] = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

    def analyst(
        self,
        role: str,
        gene: str,
        test_model_id: str,
        train_crispr: str,
        train_table: str,
        test_table: str,
    ) -> PromptValue:
        schema = self.ANALYST_SCHEMAS[role]
        example = {
            "source": role,
            "hypothesis": "One concise hypothesis supported by the supplied data",
            "reasoning": ["One concrete reason", "One exception or uncertainty"],
            "predicted_category": "weak_effect",
            "confidence": 0.5,
            "evidence_model_ids": ["ACH-XXXXXX"],
            "limitations": ["One limitation of the supplied evidence"],
        }
        template = ChatPromptTemplate.from_messages([
            ("system", self.config["system"]),
            ("human", self.config[role]),
        ])
        return template.invoke({
            "gene": gene,
            "test_model_id": test_model_id,
            "train_crispr": train_crispr,
            "train_table": train_table,
            "test_table": test_table,
            "response_schema": json.dumps(
                schema.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            ),
            "response_example": json.dumps(
                example,
                ensure_ascii=False,
                indent=2,
            ),
        })

    def judge(
        self,
        gene: str,
        test_model_id: str,
        donor: DonorHypothesis,
        expression: ExpressionHypothesis,
        mutation: MutationResult,
    ) -> PromptValue:
        example = {
            "gene": gene,
            "model_id": test_model_id,
            "predicted_category": "weak_effect",
            "confidence": 0.5,
            "conclusion": "One concise final verdict",
            "agreements": ["One agreement between analysts"],
            "conflicts": ["One conflict between analysts"],
            "decisive_evidence": ["The evidence that determined the verdict"],
        }
        template = ChatPromptTemplate.from_messages([
            ("system", self.config["system"]),
            ("human", self.config["judge"]),
        ])
        return template.invoke({
            "gene": gene,
            "test_model_id": test_model_id,
            "donor_hypothesis": donor.model_dump_json(indent=2),
            "expression_hypothesis": expression.model_dump_json(indent=2),
            "mutation_hypothesis": mutation.model_dump_json(indent=2),
            "response_schema": json.dumps(
                FinalVerdict.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            ),
            "response_example": json.dumps(
                example,
                ensure_ascii=False,
                indent=2,
            ),
        })

    @staticmethod
    def text(prompt: PromptValue) -> str:
        return "\n\n".join(
            f"## {message.type}\n\n{message.content}"
            for message in prompt.to_messages()
        )
