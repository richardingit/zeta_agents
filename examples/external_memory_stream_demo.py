from __future__ import annotations

import asyncio

from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.experimental import (
    InMemoryMemoryBackend,
    MemoryContextBuilder,
    MemoryRecord,
    MemoryScope,
)
from agent_sdk.llm import FunctionLLM


async def main() -> None:
    async def complete(messages, **kwargs):
        return "LangGraph fits complex workflows; OpenAI Agents SDK is great for lighter orchestration."

    async def stream(messages, **kwargs):
        for token in [
            "LangGraph fits complex workflows",
            "; ",
            "OpenAI Agents SDK is great for lighter orchestration.",
        ]:
            yield token

    backend = InMemoryMemoryBackend()
    await backend.add(
        MemoryRecord(
            content="User prefers practical framework comparisons over theory.",
            user_id="u_123",
            model="demo-model",
            agent_name="advisor",
            project="zeta",
        )
    )
    await backend.add(
        MemoryRecord(
            content="Previous recommendation emphasized LangGraph for complex stateful agents.",
            user_id="u_123",
            model="demo-model",
            agent_name="advisor",
            project="zeta",
        )
    )

    memory_manager = MemoryContextBuilder(backend)
    bundle = (
        AgentBuilder("advisor")
        .with_llm(FunctionLLM(complete, stream_fn=stream))
        .with_system_prompt("You are an expert multi-agent framework advisor.")
        .build()
    )

    scope = MemoryScope(
        user_id="u_123",
        model="demo-model",
        agent_name="advisor",
        project="zeta",
    )
    messages = await memory_manager.prepare_messages(
        scope=scope,
        user_input="Compare LangGraph and OpenAI Agents SDK for my next project.",
        base_system_prompt=bundle.agent.config.system_prompt,
        history=[],
    )

    ctx = Context(
        user_input="Compare LangGraph and OpenAI Agents SDK for my next project.",
        llm=bundle.llm,
        skills=bundle.skills,
        tools=bundle.tools,
        event_bus=bundle.event_bus,
        messages=messages,
        metadata={"memory_mode": "external"},
    )

    async for event in bundle.agent.run_stream(ctx):
        if event.type == "text":
            print(event.content, end="")
    print()


if __name__ == "__main__":
    asyncio.run(main())

