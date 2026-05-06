"""Memory module contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class MemoryRecord:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MemoryModule(ABC):
    @abstractmethod
    async def remember(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    @abstractmethod
    async def recall(self, query: str, k: int = 5) -> list[MemoryRecord]:
        ...

