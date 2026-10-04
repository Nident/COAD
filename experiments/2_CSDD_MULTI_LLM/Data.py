from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, cast

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

    def read(self) -> dict[str, pd.DataFrame]:
        data = self.config["data"]
        data_path = (self.config_path.parent / data["data_path"]).resolve()
        return {
            "crispr": pd.read_csv(data_path / data["crispr"], low_memory=False),
            "donor": pd.read_csv(data_path / data["donor"], low_memory=False),
            "expression": pd.read_csv(
                data_path / data["expression"],
                low_memory=False,
            ),
            "mutation": pd.read_csv(
                data_path / data["mutations"],
                low_memory=False,
            ),
        }


class DataPreparator:
    GENE_COLUMN = re.compile(r".+ \(\d+\)$")
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

    def __init__(self, data: dict[str, pd.DataFrame]):
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

    def prepare(
        self,
        gene: str,
        train_size: int,
        test_size: int,
        random_state: int,
    ) -> GeneData:
        crispr = cast(
            pd.DataFrame,
            self.crispr.loc[
                self.crispr[gene].notna(),
                ["ModelID", gene],
            ].rename(columns={gene: "EffectCategory"}),
        )
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
        train_ids = cast(list[str], train_crispr["ModelID"].tolist())
        gene_symbol = gene.split(" (", 1)[0]
        expression_column = next(
            column
            for column in self.expression.columns
            if column == gene or column.startswith(f"{gene_symbol} (")
        )
        expression_columns = [
            column
            for column in self.expression.columns
            if self.GENE_COLUMN.fullmatch(str(column))
        ]
        expression_columns.remove(expression_column)
        expression_columns.insert(0, expression_column)
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
        mutation = cast(
            pd.DataFrame,
            self.mutation.loc[
                self.mutation["HugoSymbol"].astype(str).str.upper()
                == gene_symbol.upper(),
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
