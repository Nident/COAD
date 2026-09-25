import time
from pathlib import Path
from typing import Literal

import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


EffectCategory = Literal["no_effect", "weak_effect", "strong_effect"]


class AnalysisSection(BaseModel):
    conclusion: str
    evidence: list[str]
    relevant_model_ids: list[str]


class TestPrediction(BaseModel):
    model_id: str
    predicted_category: EffectCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]


class GeneAnalysisResult(BaseModel):
    gene: str
    expression_analysis: AnalysisSection
    mutation_analysis: AnalysisSection
    integrated_conclusion: str
    test_predictions: list[TestPrediction] = Field(min_length=1)
    limitations: list[str]


class Model:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "model.yaml"

        with Path(config_path).open(encoding="utf-8") as file:
            config = yaml.safe_load(file)["model"]

        self.retry_attempts = config["retry_attempts"]
        self.retry_delay_seconds = config["retry_delay_seconds"]

        self.client = ChatOpenAI(
            model=config["name"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            timeout=config["timeout"],
            max_retries=config["max_retries"],
            streaming=config["streaming"],
            extra_body={"max_tokens": config["max_tokens"]},
        )

        self.structured_client = self.client.with_structured_output(
            GeneAnalysisResult,
            method="json_mode",
        )

    def invoke(self, prompt, expected_gene=None, expected_model_ids=None):
        """Run one request and validate the response against the Pydantic schema."""
        result = self.structured_client.invoke(prompt)
        result = result.model_dump()

        if expected_gene is not None and result["gene"] != expected_gene:
            raise ValueError(
                f"Response gene {result['gene']!r} does not match {expected_gene!r}"
            )

        if expected_model_ids is not None:
            predicted_ids = [
                prediction["model_id"] for prediction in result["test_predictions"]
            ]
            if len(set(predicted_ids)) != len(predicted_ids) or set(predicted_ids) != set(expected_model_ids):
                raise ValueError(
                    "Response predictions do not match the held-out ModelIDs"
                )

        return result

    def retry(self, prompt, expected_gene=None, expected_model_ids=None):
        """Retry transport errors and invalid JSON/schema responses."""
        last_error = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                return self.invoke(
                    prompt,
                    expected_gene=expected_gene,
                    expected_model_ids=expected_model_ids,
                )
            except Exception as error:
                last_error = error
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay_seconds)

        raise RuntimeError(
            f"Model request failed after {self.retry_attempts} attempts"
        ) from last_error


if __name__ == "__main__":
    print("Run GeneOrchestrator.py to build prompts and analyze genes.")
