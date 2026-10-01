"""Predict target-gene dependency using donor and genome-wide expression data.

This pipeline collects the same donor, target-gene mutation, and CRISPR data as
GeneOrchestrator_1. In addition, it supplies expression values for the target
gene and every other available protein-coding gene so the model can examine the
broader transcriptional context associated with cell survival after knockout.
Every model request contains the complete selected training set and exactly one
held-out test ModelID. Each request is saved in its own output directory.
"""

import json
import re
from pathlib import Path

from GeneOrchestrator_1 import (
    GeneOrchestrator as TargetGeneOrchestrator,
    evaluate_predictions,
)


class GeneOrchestrator(TargetGeneOrchestrator):
    GENE_COLUMN_PATTERN = re.compile(r".+ \(\d+\)$")

    def _expression_for_gene(self, gene):
        """Return target-gene expression followed by all other gene expressions."""
        gene_symbol = gene.split(" (", 1)[0]
        target_column = gene if gene in self.expression.columns else next(
            (
                column
                for column in self.expression.columns
                if column == gene_symbol or column.startswith(f"{gene_symbol} (")
            ),
            None,
        )

        gene_columns = [
            column
            for column in self.expression.columns
            if self.GENE_COLUMN_PATTERN.fullmatch(str(column))
        ]
        if target_column in gene_columns:
            gene_columns.remove(target_column)
            gene_columns.insert(0, target_column)

        expression = self.expression[["ModelID", *gene_columns]].copy()
        return expression.drop_duplicates(subset="ModelID", keep="first")


def _safe_path_component(value):
    """Make a gene name or ModelID safe to use as one directory component."""
    return re.sub(r'[/\\:*?"<>|]+', "_", str(value)).strip()


def _rows_for_model(dataframe, model_id):
    """Return all rows belonging to one held-out model."""
    return dataframe[dataframe["ModelID"] == model_id].copy()


def _write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    from DataReader import DataReader
    from Model import Model
    from PromptConstructor import PromptConstructor

    project_dir = Path(__file__).parent
    reader = DataReader(project_dir / "config" / "settings.yaml")
    orchestrator = GeneOrchestrator(reader.read_all())
    analysis_config = reader.config["analysis"]
    prompt_constructor = PromptConstructor(prompt_name="user_prompt_2")
    model = Model()

    output_dir = project_dir / "results_2"
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
        train_count = len(train["crispr"])

        for test_model_id in test["model_ids"]:
            case_name = (
                f"{_safe_path_component(gene)}_train_{train_count}_"
                f"test_{_safe_path_component(test_model_id)}"
            )
            case_dir = output_dir / case_name
            error_dir = case_dir / "errors"
            case_dir.mkdir(parents=True, exist_ok=True)

            test_model_ids = [test_model_id]
            test_expression = _rows_for_model(test["expression"], test_model_id)
            test_mutations = _rows_for_model(test["mutations"], test_model_id)
            test_donor = _rows_for_model(test["donor"], test_model_id)
            ground_truth = _rows_for_model(
                gene_data["ground_truth"],
                test_model_id,
            )
            context = (
                f"Cross-model CCLE dependency analysis with {train_count} "
                "training ModelIDs and 1 held-out ModelID"
            )

            metadata = {
                "gene": gene,
                "train_model_count": train_count,
                "test_model_id": test_model_id,
                "status": "pending",
            }
            metadata_path = case_dir / "metadata.json"
            _write_json(metadata_path, metadata)

            prompt = prompt_constructor.construct_prompt(
                gene=gene,
                context=context,
                train_crispr_data=train["crispr"],
                train_expression_data=train["expression"],
                train_mutation_data=train["mutations"],
                train_donor_data=train["donor"],
                test_model_ids=test_model_ids,
                test_expression_data=test_expression,
                test_mutation_data=test_mutations,
                test_donor_data=test_donor,
            )
            prompt_constructor.save_prompt_md(prompt, case_dir, gene)

            try:
                result = model.retry(
                    prompt,
                    expected_gene=gene,
                    expected_model_ids=test_model_ids,
                    error_dir=error_dir,
                )
                evaluation = evaluate_predictions(result, ground_truth)
                _write_json(
                    case_dir / "result.json",
                    {"analysis": result, "evaluation": evaluation},
                )
                metadata["status"] = "completed"
                _write_json(metadata_path, metadata)
                print(gene, test_model_id, evaluation, case_dir)
            except Exception as error:
                metadata["status"] = "failed"
                metadata["error_type"] = type(error).__name__
                metadata["error"] = str(error)
                _write_json(metadata_path, metadata)
                print(f"{gene} {test_model_id} failed. Details: {case_dir}")


if __name__ == "__main__":
    main()
