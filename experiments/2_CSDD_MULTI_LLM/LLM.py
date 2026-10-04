from pathlib import Path
from typing import TypeVar

import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr


Output = TypeVar("Output", bound=BaseModel)


class LLMFactory:
    def __init__(self, config_dir: Path):
        self.config = yaml.safe_load(
            (config_dir / "models.yaml").read_text(encoding="utf-8")
        )
        self.secrets: dict[str, str] = yaml.safe_load(
            (config_dir / "secrets.yaml").read_text(encoding="utf-8")
        )["api_keys"]

    def create(self, role: str, output: type[Output]):
        defaults = self.config["defaults"]
        model = self.config["models"][role]
        client = ChatOpenAI(
            model=model["name"],
            base_url=model["base_url"],
            api_key=SecretStr(self.secrets[model["api_key"]]),
            temperature=defaults["temperature"],
            top_p=defaults["top_p"],
            timeout=defaults["timeout"],
            max_retries=defaults["max_retries"],
            streaming=defaults["streaming"],
            extra_body={"max_tokens": defaults["max_tokens"]},
        )
        return client.with_structured_output(output, method="json_mode")
