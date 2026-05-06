"""Core shared types used across the SDK."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from typing import Any


def gen_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Any = None

    async def invoke(self, **kwargs):
        if self.handler is None:
            raise NotImplementedError(f"Tool '{self.name}' has no handler")
        result = self.handler(**kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentOutput:
    agent_name: str
    content: str
    messages: list[Message] = field(default_factory=list)
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    iterations: int = 0


@dataclass
class AgentStreamEvent:
    type: str  # text / tool_call / tool_result / done
    agent_name: str
    content: str = ""
    tool_call: ToolCall | None = None
    tool_result: str | None = None
    iteration: int = 0
    output: AgentOutput | None = None
