"""Function-backed LLM entrypoint for bring-your-own inference clients."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

from agent_sdk.modules.llm.base import LLMChunk, LLMModule, LLMResponse


CompletionFn = Callable[..., LLMResponse | str | Awaitable[LLMResponse | str]]
StreamFn = Callable[..., AsyncIterator[LLMChunk | LLMResponse | str] | Iterator[LLMChunk | LLMResponse | str] | Awaitable[AsyncIterator[LLMChunk | LLMResponse | str] | Iterator[LLMChunk | LLMResponse | str]]]


class FunctionLLM(LLMModule):
    def __init__(self, fn: CompletionFn, model: str = "function-llm", stream_fn: StreamFn | None = None):
        self.fn = fn
        self.model_name = model
        self.stream_fn = stream_fn

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

    async def stream(
        self,
        messages,
        tools=None,
        model=None,
        temperature=0.7,
        **kwargs,
    ):
        if self.stream_fn is None:
            async for chunk in super().stream(
                messages=messages,
                tools=tools,
                model=model,
                temperature=temperature,
                **kwargs,
            ):
                yield chunk
            return

        stream_result = self.stream_fn(
            messages=messages,
            tools=tools,
            model=model or self.model_name,
            temperature=temperature,
            **kwargs,
        )
        if asyncio.iscoroutine(stream_result):
            stream_result = await stream_result

        if hasattr(stream_result, "__aiter__"):
            async for item in stream_result:
                yield self._normalize_chunk(item, model or self.model_name)
            yield LLMChunk(type="done", model=model or self.model_name)
            return

        for item in stream_result:
            yield self._normalize_chunk(item, model or self.model_name)
        yield LLMChunk(type="done", model=model or self.model_name)

    @staticmethod
    def _normalize_chunk(item, model_name: str) -> LLMChunk:
        if isinstance(item, LLMChunk):
            return item
        if isinstance(item, LLMResponse):
            return LLMChunk(type="text", content=item.content, model=item.model)
        return LLMChunk(type="text", content=str(item), model=model_name)
