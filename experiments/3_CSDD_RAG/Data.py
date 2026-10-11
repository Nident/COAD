import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml


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

    def _relations(self) -> dict[str, list[str]]:
        relations: dict[str, list[str]] = {}
        with self._path("related_genes").open(encoding="utf-8") as file:
            for line in file:
                relations.update(cast(dict[str, list[str]], json.loads(line)))
        return relations

    def load(self, target_gene: str, test_model_id: str) -> TrainingData:
        effect_path = self._path("crispr")
        effect_columns = pd.read_csv(effect_path, nrows=0).columns.tolist()
        target_column = self._resolve_column(effect_columns, target_gene)
        effect = pd.read_csv(
            effect_path,
            usecols=[effect_columns[0], target_column],
        ).rename(columns={effect_columns[0]: "ModelID"})
        effect = effect.dropna(subset=[target_column]).set_index("ModelID")
        reference_effect = effect.drop(index=test_model_id)
        relations = self._relations()
        related_columns = relations[target_column]

        expression_path = self._path("expression")
        expression_header = pd.read_csv(
            expression_path,
            nrows=0,
        ).columns.tolist()
        feature_columns = [
            column
            for column in [target_column, *related_columns]
            if column in expression_header
        ]
        expression = pd.read_csv(
            expression_path,
            usecols=[
                "ModelID",
                "IsDefaultEntryForModel",
                *feature_columns,
            ],
            low_memory=False,
        )
        expression = (
            self._default_rows(expression)
            .drop_duplicates("ModelID")
            .set_index("ModelID")
        )

        reference_ids = reference_effect.index.intersection(expression.index)
        reference_effects = cast(
            pd.Series,
            reference_effect.loc[reference_ids, target_column].astype(float).copy(),
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
            test_model_id=test_model_id,
        )

    def actual_effect(self, target_column: str, test_model_id: str) -> float:
        effect_columns = pd.read_csv(self._path("crispr"), nrows=0).columns
        model_id_column = str(effect_columns[0])
        effect = pd.read_csv(
            self._path("crispr"),
            usecols=[model_id_column, target_column],
        )
        selected = effect.loc[effect[model_id_column] == test_model_id]
        return float(selected[target_column].iloc[0])
