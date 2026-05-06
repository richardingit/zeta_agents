"""OpenAI-compatible HTTP provider."""
from __future__ import annotations

import json
from typing import Any, Callable
from urllib import error, request

from agent_sdk.core.types import Role, ToolCall
from agent_sdk.modules.llm.base import LLMConfig, LLMModule, LLMResponse


Transport = Callable[[str, dict[str, str], dict[str, Any], float | None], dict[str, Any]]


def _default_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float | None,
) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed: {exc.code} {body}") from exc


class OpenAICompatibleProvider(LLMModule):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        max_tokens: int | None = None,
        timeout: float | None = 60.0,
        default_headers: dict[str, str] | None = None,
        transport: Transport | None = None,
    ):
        self.config = LLMConfig(
            provider="openai_compatible",
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_tokens=max_tokens,
            timeout=timeout,
            headers=default_headers or {},
        )
        self._transport = transport or _default_transport

    async def complete(
        self,
        messages,
        tools=None,
        model=None,
        temperature=0.7,
        **kwargs,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": [
                {
                    "role": m.role.value if isinstance(m.role, Role) else str(m.role),
                    "content": m.content,
                }
                for m in messages
            ],
            "temperature": temperature,
        }
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
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
        payload.update(kwargs)

        headers = {
            "Content-Type": "application/json",
            **self.config.headers,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        data = self._transport(
            f"{self.config.base_url}/chat/completions",
            headers,
            payload,
            self.config.timeout,
        )
        return self._parse_response(data, model or self.config.model or "openai-compatible")

    @staticmethod
    def _parse_response(data: dict[str, Any], fallback_model: str) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            return LLMResponse(content="", model=data.get("model", fallback_model), usage=data.get("usage", {}))

        message = choices[0].get("message", {})
        tool_calls = []
        for raw_tc in message.get("tool_calls", []) or []:
            function = raw_tc.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
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
            content=message.get("content") or "",
            tool_calls=tool_calls,
            model=data.get("model", fallback_model),
            usage=data.get("usage", {}),
        )
