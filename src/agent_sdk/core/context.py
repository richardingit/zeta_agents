"""Runtime context passed through agents and orchestrators."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any

from agent_sdk.core.types import Message, gen_id


@dataclass
class Context:
    user_input: str = ""
    llm: Any = None
    memory: Any = None
    skills: Any = None
    tools: Any = None
    event_bus: Any = None
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: gen_id("sess_"))

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def fork(self) -> "Context":
        return Context(
            user_input=self.user_input,
            llm=self.llm,
            memory=self.memory,
            skills=self.skills,
            tools=self.tools,
            event_bus=self.event_bus,
            state=copy.deepcopy(self.state),
            metadata=copy.deepcopy(self.metadata),
            messages=copy.deepcopy(self.messages),
            session_id=self.session_id,
        )

