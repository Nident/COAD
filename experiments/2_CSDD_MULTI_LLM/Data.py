import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import yaml


@dataclass(frozen=True)
class GeneData:
    gene: str
    train_crispr: pd.DataFrame
    train_donor: pd.DataFrame
    train_expression: pd.DataFrame
    train_mutation: pd.DataFrame
    test_ids: list[str]
    test_donor: pd.DataFrame
    test_expression: pd.DataFrame
    test_mutation: pd.DataFrame
    ground_truth: pd.DataFrame


class DataReader:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: dict[str, Any] = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
        self.data_config: dict[str, str] = self.config["data"]
        self.data_path = (
            config_path.parent / self.data_config["data_path"]
        ).resolve()

    def _data_file(self, name: str) -> Path:
        return self.data_path / self.data_config[name]

    def _config_file(self, name: str) -> Path:
        return (self.config_path.parent / self.data_config[name]).resolve()

    def read(self) -> dict[str, pd.DataFrame]:
        return {
            "crispr": pd.read_csv(self._data_file("crispr"), low_memory=False),
            "donor": pd.read_csv(self._data_file("donor"), low_memory=False),
            "expression": pd.read_csv(
                self._data_file("expression"),
                low_memory=False,
            ),
            "mutation": pd.read_csv(
                self._data_file("mutations"),
                low_memory=False,
            ),
        }

    def read_gene_relations(self) -> dict[str, list[str]]:
        relations: dict[str, list[str]] = {}
        with self._config_file("related_genes").open(encoding="utf-8") as file:
            for line in file:
                relations.update(cast(dict[str, list[str]], json.loads(line)))
        return relations


class DataPreparator:
    DONOR_COLUMNS = [
        "ModelID",
        "DepmapModelType",
        "OncotreeLineage",
        "OncotreePrimaryDisease",
        "OncotreeSubtype",
        "PrimaryOrMetastasis",
        "SampleCollectionSite",
    ]
    MUTATION_COLUMNS = [
        "ModelID",
        "HugoSymbol",
        "ProteinChange",
        "DNAChange",
        "VariantType",
        "MolecularConsequence",
        "VepImpact",
        "Hotspot",
        "AF",
        "DP",
    ]

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        gene_relations: dict[str, list[str]],
        relation_depth: int,
    ):
        self.gene_relations = gene_relations
        self.relation_depth = relation_depth
        self.crispr = data["crispr"].rename(columns={"Unnamed: 0": "ModelID"})
        model_ids = self.crispr["ModelID"]
        self.donor = cast(
            pd.DataFrame,
            data["donor"].loc[
                data["donor"]["ModelID"].isin(model_ids)
            ].copy(),
        )
        self.expression = self._default_rows(
            cast(
                pd.DataFrame,
                data["expression"].loc[
                    data["expression"]["ModelID"].isin(model_ids)
                ].copy(),
            )
        )
        self.mutation = self._default_rows(
            cast(
                pd.DataFrame,
                data["mutation"].loc[
                    data["mutation"]["ModelID"].isin(model_ids)
                ].copy(),
            )
        )

    @staticmethod
    def _default_rows(data: pd.DataFrame) -> pd.DataFrame:
        values = data["IsDefaultEntryForModel"].astype(str).str.lower()
        return cast(
            pd.DataFrame,
            data.loc[values.isin(["yes", "true", "1"])].copy(),
        )

    @staticmethod
    def _model_rows(data: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
        return cast(
            pd.DataFrame,
            data.loc[data["ModelID"].isin(model_ids)].copy(),
        )

    def genes(self, start: int, end: int) -> list[str]:
        return cast(
            list[str],
            self.crispr.columns.drop("ModelID")[start:end].tolist(),
        )

    def related_genes(self, target_gene: str) -> list[str]:
        visited = {target_gene}
        result: list[str] = []

        def collect(gene: str, depth: int) -> None:
            if depth == 0:
                return

            for related_gene in self.gene_relations.get(gene, []):
                if related_gene in visited:
                    continue
                visited.add(related_gene)
                result.append(related_gene)
                collect(related_gene, depth - 1)

        collect(target_gene, self.relation_depth)
        return result

    def prepare(
        self,
        gene: str,
        train_size: int,
        test_size: int,
        random_state: int,
        split_mode: Literal["ordered", "random"],
    ) -> GeneData:
        crispr = cast(
            pd.DataFrame,
            self.crispr.loc[
                self.crispr[gene].notna(),
                ["ModelID", gene],
            ].rename(columns={gene: "EffectCategory"}),
        )
        if split_mode == "ordered":
            train_crispr = cast(pd.DataFrame, crispr.iloc[:train_size].copy())
            test_crispr = cast(
                pd.DataFrame,
                crispr.iloc[train_size:train_size + test_size].copy(),
            )
        else:
            test_crispr = cast(
                pd.DataFrame,
                crispr.sample(n=test_size, random_state=random_state),
            )
            test_ids = cast(list[str], test_crispr["ModelID"].tolist())
            train_crispr = cast(
                pd.DataFrame,
                crispr.loc[~crispr["ModelID"].isin(test_ids)].sample(
                    n=train_size,
                    random_state=random_state,
                ),
            )
        test_ids = cast(list[str], test_crispr["ModelID"].tolist())
        train_ids = cast(list[str], train_crispr["ModelID"].tolist())
        gene_symbol = gene.split(" (", 1)[0]
        expression_column = next(
            column
            for column in self.expression.columns
            if column == gene or column.startswith(f"{gene_symbol} (")
        )
        related_genes = self.related_genes(gene)
        expression_columns = [
            expression_column,
            *[
                column
                for column in related_genes
                if column != expression_column
            ],
        ]
        expression = cast(
            pd.DataFrame,
            self.expression.loc[
                :, ["ModelID", *expression_columns]
            ].drop_duplicates(subset=["ModelID"]),
        )
        donor = cast(
            pd.DataFrame,
            self.donor.loc[:, self.DONOR_COLUMNS].drop_duplicates(
                subset=["ModelID"]
            ),
        )
        mutation_symbols = {
            column.split(" (", 1)[0].upper()
            for column in [gene, *related_genes]
        }
        mutation = cast(
            pd.DataFrame,
            self.mutation.loc[
                self.mutation["HugoSymbol"]
                .astype(str)
                .str.upper()
                .isin(mutation_symbols),
                self.MUTATION_COLUMNS,
            ].copy(),
        )

        return GeneData(
            gene=gene,
            train_crispr=train_crispr,
            train_donor=self._model_rows(donor, train_ids),
            train_expression=self._model_rows(expression, train_ids),
            train_mutation=self._model_rows(mutation, train_ids),
            test_ids=test_ids,
            test_donor=self._model_rows(donor, test_ids),
            test_expression=self._model_rows(expression, test_ids),
            test_mutation=self._model_rows(mutation, test_ids),
            ground_truth=test_crispr.reset_index(drop=True),
        )
