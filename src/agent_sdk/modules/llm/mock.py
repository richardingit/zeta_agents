"""Minimal mock LLM for tests and examples."""
from __future__ import annotations

from agent_sdk.modules.llm.base import LLMModule, LLMResponse


class MockLLM(LLMModule):
    def __init__(self, responses: list[str] | None = None, model: str = "mock"):
        self.responses = list(responses or ["ok"])
        self.model_name = model
        self._index = 0

    async def complete(self, messages, tools=None, model=None, temperature=0.7) -> LLMResponse:
        if self._index < len(self.responses):
            content = self.responses[self._index]
            self._index += 1
        else:
            content = self.responses[-1]
        return LLMResponse(content=content, model=model or self.model_name)

