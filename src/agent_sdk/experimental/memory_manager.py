"""External memory preparation layer.

Use this when you want richer memory isolation than the SDK's current
`MemoryModule.recall(query, k)` interface provides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from agent_sdk.core.types import Message, Role


@dataclass(frozen=True)
class MemoryScope:
    user_id: str
    model: str
    agent_name: str
    project: str = "default"


@dataclass
class MemoryRecord:
    content: str
    user_id: str
    model: str
    agent_name: str
    project: str = "default"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class InMemoryMemoryBackend:
    """Simple backend for scoped memory retrieval.

    Designed to prove the external memory orchestration pattern before the
    SDK grows a richer internal memory model.
    """

    def __init__(self):
        self.records: list[MemoryRecord] = []

    async def add(self, record: MemoryRecord) -> None:
        self.records.append(record)

    async def recall(
        self,
        *,
        user_id: str,
        model: str,
        agent_name: str,
        project: str = "default",
        query: str = "",
        limit: int = 5,
    ) -> list[MemoryRecord]:
        matches = [
            record
            for record in self.records
            if record.user_id == user_id
            and record.model == model
            and record.agent_name == agent_name
            and record.project == project
        ]
        if query:
            lowered = query.lower()
            query_matches = [record for record in matches if lowered in record.content.lower()]
            matches = query_matches or matches
        return matches[-limit:]


class MemoryContextBuilder:
    """Builds Context.messages externally, without using the SDK memory hook."""

    def __init__(self, backend: InMemoryMemoryBackend):
        self.backend = backend

    async def prepare_messages(
        self,
        *,
        scope: MemoryScope,
        user_input: str,
        base_system_prompt: str,
        history: list[Message] | None = None,
        memory_limit: int = 5,
    ) -> list[Message]:
        history = list(history or [])
        memories = await self.backend.recall(
            user_id=scope.user_id,
            model=scope.model,
            agent_name=scope.agent_name,
            project=scope.project,
            query=user_input,
            limit=memory_limit,
        )

        system_prompt = base_system_prompt
        if memories:
            memory_block = "\n".join(f"- {record.content}" for record in memories)
            system_prompt += f"\n\n# Relevant Memory\n{memory_block}"

        messages: list[Message] = [Message(role=Role.SYSTEM, content=system_prompt)]
        messages.extend(history)

        has_matching_user = any(
            message.role == Role.USER and message.content == user_input
            for message in history
        )
        if not has_matching_user:
            messages.append(Message(role=Role.USER, content=user_input))
        return messages

