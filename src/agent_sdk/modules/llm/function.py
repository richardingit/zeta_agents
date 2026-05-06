"""Function-backed LLM entrypoint for bring-your-own inference clients."""
from __future__ import annotations

from typing import Awaitable, Callable
import asyncio

from agent_sdk.modules.llm.base import LLMModule, LLMResponse


CompletionFn = Callable[..., LLMResponse | str | Awaitable[LLMResponse | str]]


class FunctionLLM(LLMModule):
    def __init__(self, fn: CompletionFn, model: str = "function-llm"):
        self.fn = fn
        self.model_name = model

    async def complete(
        self,
        messages,
        tools=None,
        model=None,
        temperature=0.7,
        **kwargs,
    ) -> LLMResponse:
        result = self.fn(
            messages=messages,
            tools=tools,
            model=model or self.model_name,
            temperature=temperature,
            **kwargs,
        )
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, LLMResponse):
            return result
        return LLMResponse(content=str(result), model=model or self.model_name)

