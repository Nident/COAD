from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr


Output = TypeVar("Output", bound=BaseModel)


class LLMFactory:
    def __init__(self, config_dir: Path):
        self.config = cast(
            dict[str, Any],
            yaml.safe_load(
                (config_dir / "models.yaml").read_text(encoding="utf-8")
            ) or {},
        )
        secrets = cast(
            dict[str, Any],
            yaml.safe_load(
                (config_dir / "secrets.yaml").read_text(encoding="utf-8")
            ) or {},
        )
        self.secrets = cast(dict[str, str], secrets.get("api_keys", {}))

    def create(
        self,
        role: str,
        output: type[Output],
    ) -> Runnable[PromptValue, Output]:
        defaults = cast(dict[str, Any], self.config.get("defaults", {}))
        models = cast(dict[str, dict[str, Any]], self.config.get("models", {}))
        model = models.get(role, {})
        api_key_name = model.get("api_key")
        api_key = self.secrets.get(api_key_name) if api_key_name else None
        client_config: dict[str, Any] = {
            "model_name": model.get("name"),
            "openai_api_base": model.get("base_url"),
            "openai_api_key": SecretStr(api_key) if api_key else None,
            "temperature": defaults.get("temperature"),
            "top_p": defaults.get("top_p"),
            "request_timeout": defaults.get("timeout"),
            "max_retries": defaults.get("max_retries"),
            "streaming": defaults.get("streaming"),
        }
        client_config = {
            key: value for key, value in client_config.items() if value is not None
        }
        extra_body: dict[str, Any] = {}
        if defaults.get("max_tokens") is not None:
            extra_body["max_tokens"] = defaults["max_tokens"]
        if defaults.get("thinking") is not None:
            extra_body["thinking"] = {"type": defaults["thinking"]}
        if extra_body:
            client_config["extra_body"] = extra_body
        client = ChatOpenAI.model_validate(client_config)
        runnable = client.with_structured_output(
            output,
            method="json_mode",
        ).with_retry(
            stop_after_attempt=int(defaults.get("structured_output_attempts", 1)),
        )
        return cast(Runnable[PromptValue, Output], runnable)
