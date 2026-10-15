from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import pandas as pd

from tools.CNV import CNVTool
from tools.Expression import ExpressionTool
from tools.Metadata import MetadataTool
from tools.Mutation import MutationTool

if TYPE_CHECKING:
    from Data import TrainingData
    from Retrieval import RetrievalResult


@dataclass(frozen=True)
class ToolEvidence:
    prompt_packages: dict[str, dict[str, Any]]
    judge_summary: dict[str, dict[str, Any]]
    artifacts: dict[str, pd.DataFrame]


class MultiOmicsToolbox:
    def __init__(self, settings: dict[str, Any]):
        self.expression = ExpressionTool(settings)
        self.mutation = MutationTool(settings)
        self.cnv = CNVTool(settings)
        self.metadata = MetadataTool(settings)

    def analyze(
        self,
        data: "TrainingData",
        retrieval: "RetrievalResult",
    ) -> ToolEvidence:
        packages: dict[str, dict[str, Any]] = {}
        artifacts: dict[str, pd.DataFrame] = {}
        for name, tool in [
            ("expression", self.expression),
            ("mutation", self.mutation),
            ("cnv", self.cnv),
            ("metadata", self.metadata),
        ]:
            package, tables = tool.analyze(data, retrieval)
            packages[name] = package
            artifacts.update(tables)
        judge_summary = {
            "expression": {
                "ridge_regression": packages["expression"]["ridge_regression"],
                "strongest_associations": packages["expression"][
                    "top_train_expression_associations"
                ][:5],
            },
            "mutation": {
                "test_mutation_features": packages["mutation"][
                    "test_mutation_features"
                ],
                "ridge_regression": packages["mutation"]["ridge_regression"],
                "strongest_associations": packages["mutation"][
                    "top_train_mutation_associations"
                ][:5],
            },
            "cnv": {
                "test_missing_fraction": packages["cnv"]["test_missing_fraction"],
                "ridge_regression": packages["cnv"]["ridge_regression"],
                "strongest_associations": packages["cnv"][
                    "top_train_cnv_associations"
                ][:5],
            },
            "metadata": {
                "ridge_regression": packages["metadata"]["ridge_regression"],
                "test_group_associations": packages["metadata"][
                    "test_group_associations"
                ],
            },
        }
        return ToolEvidence(
            prompt_packages=packages,
            judge_summary=judge_summary,
            artifacts=artifacts,
        )
