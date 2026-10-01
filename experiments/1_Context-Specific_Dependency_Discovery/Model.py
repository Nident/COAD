import json
import time
import traceback
from datetime import datetime, timezone
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

    @staticmethod
    def _prompt_size(prompt):
        if hasattr(prompt, "to_messages"):
            return sum(len(str(message.content)) for message in prompt.to_messages())
        return len(str(prompt))

    @staticmethod
    def _exception_chain(error):
        chain = []
        current = error
        seen = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))
            item = {
                "type": f"{type(current).__module__}.{type(current).__name__}",
                "message": str(current),
            }

            response = getattr(current, "response", None)
            if response is not None:
                item["http_status"] = getattr(response, "status_code", None)
                try:
                    item["response_body"] = response.text[:10000]
                except Exception:
                    item["response_body"] = None

            request = getattr(current, "request", None)
            if request is not None:
                item["request_method"] = getattr(request, "method", None)
                item["request_url"] = str(getattr(request, "url", ""))

            chain.append(item)
            current = current.__cause__ or current.__context__

        return chain

    def _save_error(
        self,
        error,
        attempt,
        prompt,
        expected_gene,
        expected_model_ids,
        error_dir=None,
    ):
        timestamp = datetime.now(timezone.utc)
        safe_gene = (expected_gene or "unknown_gene").replace("/", "_")
        if error_dir is None:
            error_dir = Path(__file__).parent / "errors"
        else:
            error_dir = Path(error_dir)
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / (
            f"{timestamp.strftime('%Y%m%dT%H%M%S_%fZ')}_"
            f"{safe_gene}_attempt_{attempt}.json"
        )

        diagnostic = {
            "timestamp_utc": timestamp.isoformat(),
            "attempt": attempt,
            "maximum_attempts": self.retry_attempts,
            "gene": expected_gene,
            "test_model_count": len(expected_model_ids or []),
            "test_model_ids": list(expected_model_ids or []),
            "prompt_characters": self._prompt_size(prompt),
            "exception_chain": self._exception_chain(error),
            "traceback": traceback.format_exc(),
        }
        error_path.write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return error_path

    def retry(
        self,
        prompt,
        expected_gene=None,
        expected_model_ids=None,
        error_dir=None,
    ):
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
                error_path = self._save_error(
                    error=error,
                    attempt=attempt,
                    prompt=prompt,
                    expected_gene=expected_gene,
                    expected_model_ids=expected_model_ids,
                    error_dir=error_dir,
                )
                print(f"Model request attempt {attempt} failed. Details: {error_path}")
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay_seconds)

        raise RuntimeError(
            f"Model request failed after {self.retry_attempts} attempts"
        ) from last_error


if __name__ == "__main__":
    print("Run GeneOrchestrator_1.py or GeneOrchestrator_2.py to analyze genes.")
