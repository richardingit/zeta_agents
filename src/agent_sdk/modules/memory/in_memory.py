"""Simple in-memory store for examples and tests."""
from __future__ import annotations

from agent_sdk.modules.memory.base import MemoryModule, MemoryRecord


class InMemoryStore(MemoryModule):
    def __init__(self):
        self.records: list[MemoryRecord] = []

    async def remember(self, content: str, metadata=None) -> None:
        self.records.append(MemoryRecord(content=content, metadata=metadata or {}))

    async def recall(self, query: str, k: int = 5) -> list[MemoryRecord]:
        if not query:
            return self.records[-k:]
        ranked = [
            record for record in self.records
            if query.lower() in record.content.lower()
        ]
        pool = ranked or self.records
        return pool[-k:]

