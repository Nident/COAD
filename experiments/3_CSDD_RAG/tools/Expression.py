from typing import Any, TYPE_CHECKING

import pandas as pd

from tools.Statistics import ridge_summary

if TYPE_CHECKING:
    from Data import TrainingData
    from Retrieval import RetrievalResult


class ExpressionTool:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings

    def analyze(
        self,
        data: "TrainingData",
        retrieval: "RetrievalResult",
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        reference_ids = data.reference_effects.index
        train = data.expression.loc[reference_ids, data.feature_columns]
        test = data.expression.loc[[data.test_model_id], data.feature_columns]
        regression = ridge_summary(
            train,
            data.reference_effects,
            test,
            int(self.settings["regression_folds"]),
            float(self.settings["ridge_alpha"]),
        )
        regression["difference_from_knn_baseline"] = (
            float(regression["test_prediction"])
            - float(retrieval.baseline["weighted_prediction"])
        )
        limit = int(self.settings["prompt_feature_limit"])
        package = {
            "scope": "All feature selection and statistics use reference models only.",
            "selected_related_genes": data.feature_columns[1:],
            "test_missing_fraction": float(test.isna().mean(axis=1).iloc[0]),
            "top_train_expression_associations": data.selected_expression_correlations
            .head(limit).to_dict(orient="records"),
            "ridge_regression": regression,
            "test_and_top_k_context": retrieval.expression_context.to_dict(
                orient="records"
            ),
        }
        return package, {
            "all_expression_correlations": data.expression_correlations,
            "selected_expression_correlations": data.selected_expression_correlations,
        }
