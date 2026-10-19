import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from tools.RelatedGenes import RelatedGeneResult, RelatedGeneTool


@dataclass(frozen=True)
class TrainingData:
    target_gene: str
    target_column: str
    feature_columns: list[str]
    reference_effects: pd.Series
    expression: pd.DataFrame
    mutation: pd.DataFrame
    cnv: pd.DataFrame
    metadata: pd.DataFrame
    expression_correlations: pd.DataFrame
    selected_expression_correlations: pd.DataFrame
    test_model_id: str


class DataRepository:
    MUTATION_COLUMNS = [
        "ModelID",
        "IsDefaultEntryForModel",
        "HugoSymbol",
        "ProteinChange",
        "VariantType",
        "VepImpact",
        "MolecularConsequence",
        "Hotspot",
        "OncogeneHighImpact",
        "TumorSuppressorHighImpact",
        "LikelyLoF",
        "HessDriver",
    ]

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = cast(
            dict[str, Any],
            yaml.safe_load(config_path.read_text(encoding="utf-8")),
        )
        self.data_config = cast(dict[str, str], self.config["data"])
        self.data_path = (
            config_path.parent / self.data_config["data_path"]
        ).resolve()
        self.effect_cache: dict[str, pd.Series] = {}
        self.expression_cache: pd.DataFrame | None = None
        self.related_gene_cache: dict[
            tuple[str, tuple[str, ...]],
            RelatedGeneResult,
        ] = {}

    @staticmethod
    def gene_symbol(column: str) -> str:
        return re.sub(r"\s+\(\d+\)$", "", column).strip()

    def _path(self, name: str) -> Path:
        return self.data_path / self.data_config[name]

    @staticmethod
    def _default_rows(data: pd.DataFrame) -> pd.DataFrame:
        values = data["IsDefaultEntryForModel"].astype(str).str.lower()
        return cast(
            pd.DataFrame,
            data.loc[values.isin(["yes", "true", "1"])].copy(),
        )

    @staticmethod
    def _resolve_column(columns: list[str], gene: str) -> str:
        if gene in columns:
            return gene
        symbol = DataRepository.gene_symbol(gene)
        return next(
            column
            for column in columns
            if DataRepository.gene_symbol(column) == symbol
        )

    def _expression(self) -> pd.DataFrame:
        if self.expression_cache is None:
            expression = pd.read_csv(self._path("expression"), low_memory=False)
            gene_columns = [
                column for column in expression.columns
                if re.search(r"\(\d+\)$", str(column))
            ]
            self.expression_cache = cast(pd.DataFrame, (
                self._default_rows(expression)
                .drop_duplicates("ModelID")
                .set_index("ModelID")[[*gene_columns]]
            ))
        return cast(pd.DataFrame, self.expression_cache)

    def load(
        self,
        target_gene: str,
        test_model_id: str,
        reference_model_ids: list[str] | None = None,
    ) -> TrainingData:
        effect_path = self._path("crispr")
        effect_columns = pd.read_csv(effect_path, nrows=0).columns.tolist()
        target_column = self._resolve_column(effect_columns, target_gene)
        effect = pd.read_csv(
            effect_path,
            usecols=[effect_columns[0], target_column],
        ).rename(columns={effect_columns[0]: "ModelID"})
        effect = effect.dropna(subset=[target_column]).set_index("ModelID")
        self.effect_cache[target_column] = cast(
            pd.Series,
            effect[target_column].astype(float).copy(),
        )
        reference_effect = effect.drop(index=test_model_id)
        if reference_model_ids is not None:
            reference_effect = reference_effect.loc[
                reference_effect.index.isin(reference_model_ids)
            ]
        full_expression = self._expression()
        reference_ids = reference_effect.index.intersection(full_expression.index)
        reference_effects = cast(
            pd.Series,
            reference_effect.loc[reference_ids, target_column].astype(float).copy(),
        )
        analysis_config = cast(dict[str, Any], self.config["analysis"])
        cache_key = (target_column, tuple(reference_ids.astype(str)))
        if cache_key not in self.related_gene_cache:
            selector = RelatedGeneTool(
                top_k=int(analysis_config["related_genes_top_k"]),
                fdr_alpha=float(analysis_config["fdr_alpha"]),
            )
            self.related_gene_cache[cache_key] = selector.analyze(
                full_expression.loc[reference_ids],
                reference_effects,
                target_column,
            )
        related = self.related_gene_cache[cache_key]
        feature_columns = [target_column, *related.related_genes]
        expression = cast(
            pd.DataFrame,
            full_expression[feature_columns].copy(),
        )

        cnv_path = self._path("cnv")
        cnv_header = pd.read_csv(cnv_path, nrows=0).columns.tolist()
        cnv_columns = [column for column in feature_columns if column in cnv_header]
        cnv = pd.read_csv(
            cnv_path,
            usecols=["ModelID", "IsDefaultEntryForModel", *cnv_columns],
            low_memory=False,
        )
        cnv = (
            self._default_rows(cnv)
            .drop_duplicates("ModelID")
            .set_index("ModelID")
        )

        mutation = pd.read_csv(
            self._path("mutations"),
            usecols=self.MUTATION_COLUMNS,
            low_memory=False,
        )
        mutation = self._default_rows(mutation)

        metadata_columns = cast(
            list[str],
            self.config["retrieval"]["metadata_columns"],
        )
        metadata = pd.read_csv(
            self._path("metadata"),
            usecols=["ModelID", *metadata_columns],
            low_memory=False,
        ).drop_duplicates("ModelID").set_index("ModelID")

        return TrainingData(
            target_gene=self.gene_symbol(target_column),
            target_column=target_column,
            feature_columns=feature_columns,
            reference_effects=reference_effects,
            expression=expression,
            mutation=mutation,
            cnv=cnv,
            metadata=metadata,
            expression_correlations=related.all_correlations,
            selected_expression_correlations=related.selected_correlations,
            test_model_id=test_model_id,
        )

    def actual_effect(self, target_column: str, test_model_id: str) -> float:
        if target_column in self.effect_cache:
            return float(self.effect_cache[target_column].loc[test_model_id])
        effect_columns = pd.read_csv(self._path("crispr"), nrows=0).columns
        model_id_column = str(effect_columns[0])
        effect = pd.read_csv(
            self._path("crispr"),
            usecols=[model_id_column, target_column],
        )
        selected = effect.loc[effect[model_id_column] == test_model_id]
        return float(selected[target_column].iloc[0])
