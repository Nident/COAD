from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

from Data import DataRepository, TrainingData


@dataclass(frozen=True)
class RetrievalResult:
    expression_top: pd.DataFrame
    reranked_top: pd.DataFrame
    top_k: pd.DataFrame
    baseline: dict[str, Any]
    expression_context: pd.DataFrame
    mutation_context: dict[str, list[str]]
    cnv_context: pd.DataFrame
    metadata_context: pd.DataFrame
    feature_manifest: dict[str, Any]


class MultiOmicsRetriever:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self._mutation_cache: tuple[dict[str, set[str]], list[str]] | None = None

    @staticmethod
    def _matrix(
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_imputed = imputer.fit_transform(train)
        test_imputed = imputer.transform(test)
        train_scaled = cast(np.ndarray, scaler.fit_transform(train_imputed))
        test_scaled = cast(np.ndarray, scaler.transform(test_imputed))
        return train_scaled, test_scaled

    @staticmethod
    def _similarity(
        train: np.ndarray,
        test: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        if metric == "cosine":
            return (cosine_similarity(train, test).ravel() + 1.0) / 2.0
        distance = euclidean_distances(train, test).ravel()
        return 1.0 / (1.0 + distance)

    @staticmethod
    def _true(value: object) -> bool:
        return str(value).strip().lower() in {"true", "yes", "y", "1"}

    def _mutation_profiles(
        self,
        data: TrainingData,
        reference_ids: pd.Index,
    ) -> tuple[dict[str, set[str]], list[str]]:
        reference_model_ids = cast(list[str], reference_ids.astype(str).tolist())
        reference_model_set = set(reference_model_ids)
        relevant_genes = {
            DataRepository.gene_symbol(column)
            for column in data.feature_columns
        }

        raw_profiles: dict[str, set[str]] = {}
        relevant_features: set[str] = {
            f"{gene}_MUT" for gene in relevant_genes
        }
        selected = data.mutation.loc[data.mutation["ModelID"].isin([
            *reference_model_ids,
            data.test_model_id,
        ])]
        for row in selected.itertuples(index=False):
            model_id = str(row.ModelID)
            gene = str(row.HugoSymbol)
            is_driver = any(
                self._true(getattr(row, column))
                for column in [
                    "Hotspot",
                    "OncogeneHighImpact",
                    "TumorSuppressorHighImpact",
                    "HessDriver",
                ]
            )
            is_functional = is_driver or (
                gene in relevant_genes and self._true(row.LikelyLoF)
            )
            if gene not in relevant_genes and not is_driver:
                continue
            features = raw_profiles.setdefault(model_id, set())
            gene_feature = f"{gene}_MUT"
            features.add(gene_feature)
            protein_change: object = getattr(row, "ProteinChange")
            protein_change_missing = (
                protein_change is None
                or protein_change is pd.NA
                or (
                    isinstance(protein_change, float)
                    and np.isnan(protein_change)
                )
            )
            if is_functional and not protein_change_missing:
                variant_feature = f"{gene}_{protein_change}"
                features.add(variant_feature)
                if gene in relevant_genes and model_id in reference_model_set:
                    relevant_features.add(variant_feature)

        recurrence: dict[str, int] = {}
        for model_id in reference_model_ids:
            for feature in raw_profiles.get(model_id, set()):
                recurrence[feature] = recurrence.get(feature, 0) + 1
        selected_features = relevant_features | {
            feature
            for feature, count in recurrence.items()
            if count >= self.settings["mutation_min_recurrence"]
        }
        profiles = {
            model_id: features & selected_features
            for model_id, features in raw_profiles.items()
        }
        return profiles, sorted(selected_features)

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        return len(left & right) / len(union)

    def _metadata_similarity(
        self,
        data: TrainingData,
        neighbour_id: str,
    ) -> float:
        if (
            data.test_model_id not in data.metadata.index
            or neighbour_id not in data.metadata.index
        ):
            return 0.5
        test = cast(pd.Series, data.metadata.loc[data.test_model_id])
        neighbour = cast(pd.Series, data.metadata.loc[neighbour_id])
        matches: list[float] = []
        for column in data.metadata.columns:
            test_value: object = test.at[column]
            neighbour_value: object = neighbour.at[column]
            test_missing = test_value is None or test_value is pd.NA or (
                isinstance(test_value, float) and np.isnan(test_value)
            )
            neighbour_missing = (
                neighbour_value is None
                or neighbour_value is pd.NA
                or (
                    isinstance(neighbour_value, float)
                    and np.isnan(neighbour_value)
                )
            )
            if test_missing or neighbour_missing:
                continue
            matches.append(float(test_value == neighbour_value))
        return float(np.mean(matches)) if matches else 0.5

    def retrieve(
        self,
        data: TrainingData,
        use_mutation: bool,
        use_cnv: bool,
        use_metadata: bool,
    ) -> RetrievalResult:
        reference_ids = data.reference_effects.index
        expression_train = data.expression.loc[
            reference_ids,
            data.feature_columns,
        ].apply(pd.to_numeric, errors="coerce")
        expression_test = data.expression.loc[
            [data.test_model_id],
            data.feature_columns,
        ].apply(pd.to_numeric, errors="coerce")
        expression_train_scaled, expression_test_scaled = self._matrix(
            expression_train,
            expression_test,
        )
        expression_similarity = self._similarity(
            expression_train_scaled,
            expression_test_scaled,
            self.settings["expression_metric"],
        )
        expression_top = pd.DataFrame({
            "ModelID": reference_ids,
            "ExpressionSimilarity": expression_similarity,
        }).sort_values("ExpressionSimilarity", ascending=False).head(
            self.settings["expression_candidates"]
        ).reset_index(drop=True)
        expression_top["ExpressionRank"] = np.arange(1, len(expression_top) + 1)

        candidate_ids = cast(list[str], expression_top["ModelID"].tolist())
        if self._mutation_cache is None:
            self._mutation_cache = self._mutation_profiles(data, reference_ids)
        mutation_profiles, mutation_features = self._mutation_cache
        test_mutations = mutation_profiles.get(data.test_model_id, set())
        mutation_similarity = [
            self._jaccard(test_mutations, mutation_profiles.get(model_id, set()))
            for model_id in candidate_ids
        ]

        cnv_columns = [
            column for column in data.feature_columns if column in data.cnv.columns
        ]
        cnv_reference_ids = reference_ids.intersection(data.cnv.index)
        cnv_train = data.cnv.loc[cnv_reference_ids, cnv_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if data.test_model_id in data.cnv.index:
            cnv_test = data.cnv.loc[[data.test_model_id], cnv_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
            cnv_train_scaled, cnv_test_scaled = self._matrix(cnv_train, cnv_test)
            cnv_scores = dict(zip(
                cnv_reference_ids,
                self._similarity(cnv_train_scaled, cnv_test_scaled, "cosine"),
                strict=True,
            ))
        else:
            cnv_scores = {}
        cnv_similarity = [cnv_scores.get(model_id, 0.5) for model_id in candidate_ids]
        metadata_similarity = [
            self._metadata_similarity(data, model_id)
            for model_id in candidate_ids
        ]

        reranked = expression_top.assign(
            MutationSimilarity=mutation_similarity,
            CNVSimilarity=cnv_similarity,
            MetadataSimilarity=metadata_similarity,
        )
        configured_weights = cast(dict[str, float], self.settings["weights"])
        active_weights = {
            "ExpressionSimilarity": configured_weights["expression"],
            "MutationSimilarity": configured_weights["mutation"] if use_mutation else 0.0,
            "CNVSimilarity": configured_weights["cnv"] if use_cnv else 0.0,
            "MetadataSimilarity": configured_weights["metadata"] if use_metadata else 0.0,
        }
        weight_sum = sum(active_weights.values())
        reranked["FinalSimilarity"] = sum(
            reranked[column] * weight / weight_sum
            for column, weight in active_weights.items()
        )
        reranked = reranked.sort_values(
            "FinalSimilarity",
            ascending=False,
        ).reset_index(drop=True)
        reranked["FinalRank"] = np.arange(1, len(reranked) + 1)

        selected = reranked.head(self.settings["top_k"]).copy()
        effects_by_model = cast(
            dict[str, float],
            data.reference_effects.to_dict(),
        )
        selected_ids = cast(list[str], selected["ModelID"].tolist())
        selected["ObservedTargetGeneEffect"] = [
            effects_by_model[model_id] for model_id in selected_ids
        ]
        effects = cast(
            np.ndarray,
            selected["ObservedTargetGeneEffect"].to_numpy(dtype=float),
        )
        weights = cast(
            np.ndarray,
            selected["FinalSimilarity"].to_numpy(dtype=float),
        ).clip(min=1e-8)
        weighted_prediction = float(np.average(effects, weights=weights))
        standard_deviation = float(np.std(effects, ddof=0))
        consistency_limits = self.settings["consistency_std"]
        consistency = (
            "high"
            if standard_deviation <= consistency_limits["high"]
            else "medium"
            if standard_deviation <= consistency_limits["medium"]
            else "low"
        )
        baseline: dict[str, Any] = {
            "weighted_prediction": weighted_prediction,
            "mean": float(np.mean(effects)),
            "median": float(np.median(effects)),
            "standard_deviation": standard_deviation,
            "minimum": float(np.min(effects)),
            "maximum": float(np.max(effects)),
            "mean_similarity": float(np.mean(weights)),
            "neighbour_outcome_consistency": consistency,
            "top_k": len(selected),
        }

        evidence_ids = [data.test_model_id, *cast(list[str], selected["ModelID"].tolist())]
        expression_context = data.expression.reindex(evidence_ids)[
            data.feature_columns
        ].rename_axis("ModelID").reset_index()
        cnv_context = data.cnv.reindex(evidence_ids)[cnv_columns].rename_axis(
            "ModelID"
        ).reset_index()
        metadata_context = data.metadata.reindex(evidence_ids).rename_axis(
            "ModelID"
        ).reset_index()
        mutation_context = {
            model_id: sorted(mutation_profiles.get(model_id, set()))
            for model_id in evidence_ids
        }
        feature_manifest: dict[str, Any] = {
            "target_column": data.target_column,
            "expression_features": data.feature_columns,
            "cnv_features": cnv_columns,
            "mutation_selected_features": mutation_features,
            "metadata_features": data.metadata.columns.tolist(),
            "expression_metric": self.settings["expression_metric"],
            "active_weights": active_weights,
            "reference_model_count": len(reference_ids),
            "test_model_id": data.test_model_id,
            "normalization_fit": "reference_models_only",
            "mutation_vocabulary_fit": "reference_models_only",
            "observed_effect_attached_after_reranking": True,
            "test_effect_used_for_retrieval": False,
        }
        return RetrievalResult(
            expression_top=expression_top,
            reranked_top=reranked,
            top_k=selected,
            baseline=baseline,
            expression_context=expression_context,
            mutation_context=mutation_context,
            cnv_context=cnv_context,
            metadata_context=metadata_context,
            feature_manifest=feature_manifest,
        )
