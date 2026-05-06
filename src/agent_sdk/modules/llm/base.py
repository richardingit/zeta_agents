"""LLM module contracts and simple response models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_sdk.core.types import Tool, ToolCall


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = "mock"
    usage: dict[str, Any] = field(default_factory=dict)


class LLMModule(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list,
        tools: list[Tool] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        ...

