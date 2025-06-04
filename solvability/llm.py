import litellm

from solvability.models.config import LLMConfig


def completion(
    *args, llm_config: LLMConfig | None = None, **kwargs
) -> litellm.ModelResponse | litellm.CustomStreamWrapper:
    if llm_config is None:
        llm_config = LLMConfig.from_dotenv()

    return litellm.completion(*args, **kwargs, **llm_config.as_kwargs())


def completion_cost(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cache_reads: int = 0,
    cache_writes: int = 0,
    llm_config: LLMConfig | None = None,
) -> float:
    """
    Compute the cost of processing the provided tokens.
    """
    if llm_config is None:
        llm_config = LLMConfig.from_dotenv()

    input_cost, output_cost = litellm.cost_calculator.cost_per_token(
        model=llm_config.model.removeprefix("litellm_proxy/"),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_creation_input_tokens=cache_reads,
        cache_read_input_tokens=cache_writes,
    )
    return input_cost + output_cost
