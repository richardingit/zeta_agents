"""Factory helpers for flexible LLM entrypoints."""
from __future__ import annotations

from agent_sdk.modules.llm.base import LLMConfig, LLMModule
from agent_sdk.modules.llm.function import FunctionLLM
from agent_sdk.modules.llm.mock import MockLLM
from agent_sdk.modules.llm.openai_compatible import OpenAICompatibleProvider


def create_llm(
    provider: str = "mock",
    model: str | None = None,
    fn=None,
    **kwargs,
) -> LLMModule:
    provider = provider.lower()
    if provider == "mock":
        responses = kwargs.pop("responses", None)
        return MockLLM(responses=responses, model=model or "mock")
    if provider == "function":
        if fn is None:
            raise ValueError("FunctionLLM requires `fn`")
        return FunctionLLM(fn, model=model or "function-llm")
    if provider in {"openai_compatible", "openai-compatible"}:
        if model is None:
            raise ValueError("OpenAICompatibleProvider requires `model`")
        return OpenAICompatibleProvider(model=model, **kwargs)
    if provider == "litellm":
        try:
            from agent_sdk.modules.llm.litellm_provider import LiteLLMProvider
        except ImportError as exc:
            raise ImportError(
                "LiteLLM support is not installed. Use `pip install agent-sdk[litellm]`."
            ) from exc
        if model is None:
            raise ValueError("LiteLLMProvider requires `model`")
        return LiteLLMProvider(model=model, **kwargs)
    raise ValueError(f"Unsupported llm provider: {provider}")


def create_llm_from_config(config: LLMConfig, fn=None) -> LLMModule:
    return create_llm(
        provider=config.provider,
        model=config.model,
        fn=fn,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_tokens=config.max_tokens,
        default_headers=config.headers,
        **config.extra,
    )

