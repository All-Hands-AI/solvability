import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel

from solvability.llm import completion
from solvability.models.config import LLMConfig


class Feature(BaseModel):
    identifier: str
    description: str

    @property
    def to_tool_description_field(self) -> dict[str, Any]:
        return {
            "type": "boolean",
            "description": self.description,
        }


class EmbeddingDimension(BaseModel):
    feature_id: str
    result: bool


EmbeddingSample = dict[str, bool]


class FeatureEmbedding(BaseModel):
    samples: list[EmbeddingSample]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_latency: float | None = None

    @property
    def dimensions(self) -> list[str]:
        """
        Basis vectors for the embedding.
        """
        dims: set[str] = set()
        for sample in self.samples:
            dims.update(sample.keys())
        return list(dims)

    def coefficient(self, dimension: str) -> float | None:
        """
        Coefficient of the dimension in the embedding.
        """
        values = [1 if v else 0 for v in [sample.get(dimension) for sample in self.samples] if v is not None]
        if values:
            return sum(values) / len(values)
        return None

    def to_row(self) -> dict[str, Any]:
        return {
            "response_latency": self.response_latency,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            **{dimension: self.coefficient(dimension) for dimension in self.dimensions},
        }

    def sample_entropy(self) -> dict[str, float]:
        """
        Calculate the entropy of the samples for each dimension.
        """
        from collections import Counter
        from math import log2

        entropy = {}
        for dimension in self.dimensions:
            counts = Counter(sample.get(dimension, False) for sample in self.samples)
            total = sum(counts.values())
            if total == 0:
                entropy[dimension] = 0.0
                continue
            entropy_value = -sum((count / total) * log2(count / total) for count in counts.values() if count > 0)
            entropy[dimension] = entropy_value
        return entropy


class Featurizer(BaseModel):
    system_prompt: str
    message_prefix: str
    features: list[Feature]

    def system_message(self) -> dict[str, Any]:
        return {
            "role": "system",
            "content": self.system_prompt,
        }

    def user_message(self, issue_description: str, set_cache: bool = True) -> dict[str, Any]:
        """
        User message that captures the issue description and any other non-system prompting.

        Args:
            issue_description: The description of the issue.

            set_cache_content: Whether to set cache content for the message. If only one sample is requested, this
            should be set to False.
        """
        message: dict[str, Any] = {
            "role": "user",
            "content": f"{self.message_prefix}{issue_description}",
        }
        if set_cache:
            message["cache_control"] = {"type": "ephemeral"}
        return message

    @property
    def tool_choice(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": "call_featurizer"},
        }

    @property
    def tool_description(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "call_featurizer",
                "description": "Record the features present in the issue.",
                "parameters": {
                    "type": "object",
                    "properties": {feature.identifier: feature.to_tool_description_field for feature in self.features},
                },
            },
        }

    def embed(
        self, issue_description: str, temperature: float = 1.0, samples: int = 10, llm_config: LLMConfig | None = None
    ) -> FeatureEmbedding:
        """
        Generate an embedding for the issue description.

        Args:
            issue_description: The description of the issue.
            temperature: Sampling temperature for the model. Defaults to 1.0.
            samples: Number of samples to generate. Defaults to 10.
        """
        embedding_samples: list[dict[str, Any]] = []
        response_latency: float = 0.0
        prompt_tokens: int = 0
        completion_tokens: int = 0

        for _ in range(samples):
            start_time = time.time()
            response = completion(
                messages=[self.system_message(), self.user_message(issue_description, set_cache=(samples > 1))],
                tools=[self.tool_description],
                tool_choice=self.tool_choice,
                temperature=temperature,
                llm_config=llm_config,
            )
            stop_time = time.time()

            latency = stop_time - start_time
            features = response.choices[0].message.tool_calls[0].function.arguments # type: ignore[index, union-attr]
            embedding = json.loads(features)

            embedding_samples.append(embedding)
            prompt_tokens += response.usage.prompt_tokens # type: ignore[union-attr, attr-defined]
            completion_tokens += response.usage.completion_tokens # type: ignore[union-attr, attr-defined]
            response_latency += latency

        return FeatureEmbedding(
            samples=embedding_samples,
            response_latency=response_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def embed_batch(
        self,
        issue_descriptions: list[str],
        temperature: float = 1.0,
        samples: int = 10,
        llm_config: LLMConfig | None = None,
    ) -> list[FeatureEmbedding]:
        """
        Generate embeddings for a batch of issue descriptions.
        """
        with ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_desc = {
                executor.submit(self.embed, desc, temperature, samples, llm_config=llm_config): i
                for i, desc in enumerate(issue_descriptions)
            }

            # Collect results in order
            results: list[FeatureEmbedding] = [None] * len(issue_descriptions)  # type: ignore[list-item]
            for future in as_completed(future_to_desc):
                index = future_to_desc[future]
                results[index] = future.result()

            return results

    def feature_identifiers(self) -> list[str]:
        """
        Get the identifiers of the features.
        """
        return [feature.identifier for feature in self.features]
