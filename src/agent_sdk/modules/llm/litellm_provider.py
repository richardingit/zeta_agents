"""LiteLLM adapter for wide provider compatibility."""
from __future__ import annotations

from agent_sdk.core.types import ToolCall
from agent_sdk.modules.llm.base import LLMConfig, LLMModule, LLMResponse


class LiteLLMProvider(LLMModule):
    def __init__(self, model: str, **kwargs):
        self.config = LLMConfig(provider="litellm", model=model, extra=kwargs)

    async def complete(
        self,
        messages,
        tools=None,
        model=None,
        temperature=0.7,
        **kwargs,
    ) -> LLMResponse:
        from litellm import acompletion

        payload = {
            "model": model or self.config.model,
            "messages": [
                {
                    "role": getattr(m.role, "value", m.role),
                    "content": m.content,
                }
                for m in messages
            ],
            "temperature": temperature,
            **self.config.extra,
            **kwargs,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters or {"type": "object", "properties": {}},
                    },
                }
                for tool in tools
            ]

        response = await acompletion(**payload)
        choice = response["choices"][0]["message"]
        tool_calls = []
        for raw_tc in choice.get("tool_calls", []) or []:
            function = raw_tc.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                import json
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            tool_calls.append(
                ToolCall(
                    id=raw_tc.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments,
                )
            )
        return LLMResponse(
            content=choice.get("content") or "",
            tool_calls=tool_calls,
            model=response.get("model", payload["model"]),
            usage=response.get("usage", {}),
        )

