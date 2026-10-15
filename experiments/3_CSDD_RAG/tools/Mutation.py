from typing import Any, TYPE_CHECKING

import pandas as pd

from tools.Statistics import binary_associations, ridge_summary

if TYPE_CHECKING:
    from Data import TrainingData
    from Retrieval import RetrievalResult


class MutationTool:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings

    @staticmethod
    def _matrix(
        model_ids: list[str],
        features: list[str],
        profiles: dict[str, set[str]],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [int(feature in profiles.get(model_id, set())) for feature in features]
                for model_id in model_ids
            ],
            index=model_ids,
            columns=features,
        )

    def analyze(
        self,
        data: "TrainingData",
        retrieval: "RetrievalResult",
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        reference_ids = data.reference_effects.index.astype(str).tolist()
        features = retrieval.mutation_features
        train = self._matrix(reference_ids, features, retrieval.mutation_profiles)
        test = self._matrix(
            [data.test_model_id],
            features,
            retrieval.mutation_profiles,
        )
        associations = binary_associations(train, data.reference_effects)
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
        test_features = [
            feature for feature in features
            if int(test.at[data.test_model_id, feature]) == 1
        ]
        test_associations = associations.loc[
            associations["feature"].isin(test_features)
        ]
        prompt_associations = pd.concat([
            test_associations,
            associations.head(limit),
        ]).drop_duplicates("feature").head(limit)
        package = {
            "scope": "Mutation vocabulary, associations, and regression use reference models only.",
            "test_mutation_features": test_features,
            "selected_mutation_feature_count": len(features),
            "top_train_mutation_associations": prompt_associations.to_dict(
                orient="records"
            ),
            "ridge_regression": regression,
            "test_and_top_k_context": retrieval.mutation_context.to_dict(
                orient="records"
            ),
        }
        return package, {
            "mutation_associations": associations,
        }
