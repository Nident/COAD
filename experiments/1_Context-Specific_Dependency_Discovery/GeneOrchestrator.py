import json
from pathlib import Path


class GeneOrchestrator:
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

    def __init__(self, data):
        self.crispr = data["crispr"].rename(columns={"Unnamed: 0": "ModelID"})
        model_ids = self.crispr["ModelID"]

        self.donor = data["donor"][data["donor"]["ModelID"].isin(model_ids)].copy()
        self.expression = data["expression"][
            data["expression"]["ModelID"].isin(model_ids)
        ].copy()
        self.mutations = data["mutations"][
            data["mutations"]["ModelID"].isin(model_ids)
        ].copy()

        self.expression = self._default_model_entries(self.expression)
        self.mutations = self._default_model_entries(self.mutations)

    @staticmethod
    def _default_model_entries(dataframe):
        """Keep the preferred sequencing entry when the source marks one."""
        column = "IsDefaultEntryForModel"
        if column not in dataframe.columns:
            return dataframe

        is_default = (
            dataframe[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"yes", "true", "1"})
        )
        return dataframe.loc[is_default].copy() if is_default.any() else dataframe

    @staticmethod
    def _available_columns(dataframe, requested_columns):
        return [column for column in requested_columns if column in dataframe.columns]

    def _expression_for_gene(self, gene):
        gene_symbol = gene.split(" (", 1)[0]
        expression_column = gene if gene in self.expression.columns else None

        if expression_column is None:
            expression_column = next(
                (
                    column
                    for column in self.expression.columns
                    if column == gene_symbol or column.startswith(f"{gene_symbol} (")
                ),
                None,
            )

        columns = ["ModelID"]
        if expression_column is not None:
            columns.append(expression_column)

        expression = self.expression[columns].copy()
        if expression_column is not None:
            expression = expression.rename(columns={expression_column: "Expression"})
        return expression.drop_duplicates(subset="ModelID", keep="first")

    def _mutations_for_gene(self, gene):
        gene_symbol = gene.split(" (", 1)[0]
        mutations = self.mutations[
            self.mutations["HugoSymbol"].astype(str).str.upper() == gene_symbol.upper()
        ]
        columns = self._available_columns(mutations, self.MUTATION_COLUMNS)
        return mutations[columns].copy()

    def prepare_gene(self, gene, train_size=None, test_size=3, random_state=42):
        crispr = (
            self.crispr[["ModelID", gene]]
            .dropna(subset=[gene])
            .rename(columns={gene: "EffectCategory"})
        )

        if len(crispr) < test_size:
            raise ValueError(
                f"Gene {gene} has {len(crispr)} labeled models; {test_size} are required"
            )

        test_crispr = crispr.sample(n=test_size, random_state=random_state)
        test_ids = test_crispr["ModelID"].tolist()

        train_crispr = crispr[~crispr["ModelID"].isin(test_ids)].copy()
        if train_size is not None and len(train_crispr) > train_size:
            train_crispr = train_crispr.sample(
                n=train_size,
                random_state=random_state,
            )

        donor_columns = self._available_columns(self.donor, self.DONOR_COLUMNS)
        donor = self.donor[donor_columns].drop_duplicates(subset="ModelID", keep="first")
        expression = self._expression_for_gene(gene)
        mutations = self._mutations_for_gene(gene)

        train_ids = set(train_crispr["ModelID"])
        test_id_set = set(test_ids)

        return {
            "gene": gene,
            "context": (
                f"Cross-model CCLE dependency analysis with {len(train_crispr)} "
                f"training ModelIDs and {len(test_crispr)} held-out ModelIDs"
            ),
            "train": {
                "crispr": train_crispr,
                "donor": donor[donor["ModelID"].isin(train_ids)].copy(),
                "expression": expression[expression["ModelID"].isin(train_ids)].copy(),
                "mutations": mutations[mutations["ModelID"].isin(train_ids)].copy(),
            },
            "test": {
                "model_ids": test_ids,
                "donor": donor[donor["ModelID"].isin(test_id_set)].copy(),
                "expression": expression[expression["ModelID"].isin(test_id_set)].copy(),
                "mutations": mutations[mutations["ModelID"].isin(test_id_set)].copy(),
            },
            "ground_truth": test_crispr.reset_index(drop=True),
        }

    def iter_genes(
        self,
        gene_start_index,
        gene_end_index,
        train_model_limit,
        test_model_limit,
        random_state,
    ):
        gene_columns = self.crispr.columns.drop("ModelID")
        selected_genes = gene_columns[gene_start_index:gene_end_index]

        for gene in selected_genes:
            yield self.prepare_gene(
                gene,
                train_size=train_model_limit,
                test_size=test_model_limit,
                random_state=random_state,
            )


def evaluate_predictions(result, ground_truth):
    expected = ground_truth.set_index("ModelID")["EffectCategory"].to_dict()
    rows = []

    for prediction in result["test_predictions"]:
        model_id = prediction["model_id"]
        actual = expected.get(model_id)
        rows.append({
            "model_id": model_id,
            "predicted_category": prediction["predicted_category"],
            "actual_category": actual,
            "correct": prediction["predicted_category"] == actual,
        })

    return rows


def main():
    from DataReader import DataReader
    from Model import Model
    from PromptConstructor import PromptConstructor

    project_dir = Path(__file__).parent
    reader = DataReader(project_dir / "config" / "settings.yaml")
    orchestrator = GeneOrchestrator(reader.read_all())
    analysis_config = reader.config["analysis"]
    prompt_constructor = PromptConstructor()
    model = Model()

    output_dir = project_dir / "results"
    prompt_dir = project_dir / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)

    for gene_data in orchestrator.iter_genes(
        gene_start_index=analysis_config["gene_start_index"],
        gene_end_index=analysis_config["gene_end_index"],
        train_model_limit=analysis_config["train_model_limit"],
        test_model_limit=analysis_config["test_model_limit"],
        random_state=analysis_config["random_state"],
    ):
        gene = gene_data["gene"]
        train = gene_data["train"]
        test = gene_data["test"]

        prompt = prompt_constructor.construct_prompt(
            gene=gene,
            context=gene_data["context"],
            train_crispr_data=train["crispr"],
            train_expression_data=train["expression"],
            train_mutation_data=train["mutations"],
            train_donor_data=train["donor"],
            test_model_ids=test["model_ids"],
            test_expression_data=test["expression"],
            test_mutation_data=test["mutations"],
            test_donor_data=test["donor"],
        )
        prompt_constructor.save_prompt_md(prompt, prompt_dir, gene)

        result = model.retry(
            prompt,
            expected_gene=gene,
            expected_model_ids=test["model_ids"],
        )
        evaluation = evaluate_predictions(result, gene_data["ground_truth"])
        safe_gene = gene.replace("/", "_")
        result_path = output_dir / f"{safe_gene}_result.json"
        result_path.write_text(
            json.dumps(
                {"analysis": result, "evaluation": evaluation},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(gene, evaluation, result_path)


if __name__ == "__main__":
    main()
