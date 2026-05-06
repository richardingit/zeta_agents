"""Tools registry."""
from __future__ import annotations


class ToolsModule:
    def __init__(self):
        self._tools: dict[str, object] = {}

    def register(self, tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str):
        return self._tools.get(name)

