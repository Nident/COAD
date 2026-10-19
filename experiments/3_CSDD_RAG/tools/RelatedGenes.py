from dataclasses import dataclass

import pandas as pd

from tools.Statistics import correlations


@dataclass(frozen=True)
class RelatedGeneResult:
    related_genes: list[str]
    all_correlations: pd.DataFrame
    selected_correlations: pd.DataFrame


class RelatedGeneTool:
    def __init__(self, top_k: int, fdr_alpha: float):
        self.top_k = top_k
        self.fdr_alpha = fdr_alpha

    def analyze(
        self,
        expression: pd.DataFrame,
        target_effect: pd.Series,
        target_column: str,
    ) -> RelatedGeneResult:
        feature_columns = [
            column for column in expression.columns
            if column != target_column
        ]
        report = correlations(expression[feature_columns], target_effect)
        significant = report.loc[report["best_fdr"] <= self.fdr_alpha]
        selected = significant.head(self.top_k)
        if len(selected) < self.top_k:
            selected = report.head(self.top_k)
        related_genes = selected["feature"].astype(str).tolist()
        return RelatedGeneResult(
            related_genes=related_genes,
            all_correlations=report,
            selected_correlations=selected.reset_index(drop=True),
        )
