from __future__ import annotations
from pydantic import BaseModel, SecretStr
import litellm
import dotenv


class LLMConfig(BaseModel):
    """
    Configuration for the LLM model.
    """

    model: str

    base_url: str | None = None

    api_key: SecretStr

    thinking_budget: int | None = None
    """
    The maximum number of tokens the model can use for reasoning. This is a limit on the model's thinking capacity.
    """

    @property
    def supports_thinking(self) -> bool:
        """
        Check if the model supports thinking. This is determined by whether a thinking budget is set.
        """
        return self.thinking_budget is not None and litellm.supports_reasoning(
            self.model
        )

    def as_kwargs(self) -> dict[str, str]:
        """
        Convert the configuration to a dictionary of keyword arguments. Can be passed to the LiteLLM completion calls.
        """
        results = {"model": self.model}
        if self.base_url:
            results["base_url"] = self.base_url
        results["api_key"] = self.api_key.get_secret_value()
        if self.supports_thinking:
            results["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        return results

    @staticmethod
    def from_dotenv() -> LLMConfig:
        """
        Create an LLMConfig instance from the environment variables.
        """
        return LLMConfig(**dotenv.dotenv_values())
