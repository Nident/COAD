import json
from pathlib import Path

import pandas as pd
import yaml
from langchain_core.prompts import ChatPromptTemplate


class PromptConstructor:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "model.yaml"

        with Path(config_path).open(encoding="utf-8") as file:
            config = yaml.safe_load(file)["model"]

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", config["system_prompt"]),
            ("human", config["user_prompt_template"]),
        ])

    @staticmethod
    def _to_json(data):
        """Serialize every prepared row and column without pandas truncation."""
        if isinstance(data, pd.DataFrame):
            clean_data = data.astype(object).where(data.notna(), None)
            table = clean_data.to_dict(orient="split")
            table.pop("index")
            return json.dumps(
                table,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
                allow_nan=False,
            )
        if isinstance(data, pd.Series):
            return data.to_json(force_ascii=False)
        return json.dumps(data, ensure_ascii=False, default=str)

    def construct_prompt(
        self,
        gene,
        context,
        train_crispr_data,
        train_expression_data,
        train_mutation_data,
        train_donor_data,
        test_model_ids,
        test_expression_data,
        test_mutation_data,
        test_donor_data,
    ):
        test_predictions_example = [
            {
                "model_id": model_id,
                "predicted_category": "no_effect | weak_effect | strong_effect",
                "confidence": 0.0,
                "evidence": ["Evidence supporting this prediction"],
            }
            for model_id in test_model_ids
        ]

        values = {
            "gene": gene,
            "context": context,
            "train_crispr_data": train_crispr_data,
            "train_expression_data": train_expression_data,
            "train_mutation_data": train_mutation_data,
            "train_donor_data": train_donor_data,
            "test_model_ids": test_model_ids,
            "test_expression_data": test_expression_data,
            "test_mutation_data": test_mutation_data,
            "test_donor_data": test_donor_data,
            "test_case_count": len(test_model_ids),
            "test_predictions_json": json.dumps(
                test_predictions_example,
                ensure_ascii=False,
                indent=2,
            ),
        }

        return self.prompt_template.invoke({
            key: self._to_json(value)
            if key
            not in {
                "gene",
                "context",
                "test_case_count",
                "test_predictions_json",
            }
            else value
            for key, value in values.items()
        })

    def save_prompt_md(self, prompt, output_dir, gene):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = "\n\n".join(
            f"## {message.type.title()} message\n\n{message.content}"
            for message in prompt.to_messages()
        )

        safe_gene = gene.replace("/", "_")
        output_file = output_dir / f"{safe_gene}_prompt.md"
        output_file.write_text(prompt_text, encoding="utf-8")
        return output_file
