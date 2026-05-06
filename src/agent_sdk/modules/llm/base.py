"""LLM module contracts and simple response models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import os
from typing import Any

from agent_sdk.core.types import Message, Tool, ToolCall


@dataclass
class LLMConfig:
    provider: str = "custom"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float | None = None
    headers: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        provider: str,
        model: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        **kwargs,
    ) -> "LLMConfig":
        return cls(
            provider=provider,
            model=model,
            api_key=os.getenv(api_key_env),
            base_url=os.getenv(base_url_env),
            **kwargs,
        )


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = "mock"
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMChunk:
    type: str = "text"  # text / tool_call / done / usage
    content: str = ""
    tool_call: ToolCall | None = None
    usage: dict[str, Any] | None = None
    model: str | None = None
    raw: dict[str, Any] | None = None


class LLMModule(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:
        response = await self.complete(
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            **kwargs,
        )
        if response.content:
            yield LLMChunk(type="text", content=response.content, model=response.model)
        for tool_call in response.tool_calls:
            yield LLMChunk(type="tool_call", tool_call=tool_call, model=response.model)
        if response.usage:
            yield LLMChunk(type="usage", usage=response.usage, model=response.model)
        yield LLMChunk(type="done", model=response.model)
