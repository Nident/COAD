from typing import Any, TYPE_CHECKING

import pandas as pd

from tools.Statistics import correlations, ridge_summary

if TYPE_CHECKING:
    from Data import TrainingData
    from Retrieval import RetrievalResult


class CNVTool:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings

    def analyze(
        self,
        data: "TrainingData",
        retrieval: "RetrievalResult",
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        reference_ids = data.reference_effects.index.intersection(data.cnv.index)
        features = retrieval.cnv_features
        train = data.cnv.loc[reference_ids, features]
        outcome = data.reference_effects.loc[reference_ids]
        test = data.cnv.reindex([data.test_model_id])[features]
        associations = correlations(train, outcome)
        regression = ridge_summary(
            train,
            outcome,
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
            "scope": "CNV normalization, associations, and regression use reference models only.",
            "test_missing_fraction": float(test.isna().mean(axis=1).iloc[0]),
            "top_train_cnv_associations": associations.head(limit).to_dict(
                orient="records"
            ),
            "ridge_regression": regression,
            "test_and_top_k_context": retrieval.cnv_context.to_dict(
                orient="records"
            ),
        }
        return package, {"cnv_associations": associations}
