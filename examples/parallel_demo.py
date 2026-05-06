from __future__ import annotations

import asyncio

from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import ParallelOrchestrator


async def main() -> None:
    optimist = (
        AgentBuilder("optimist")
        .with_llm(MockLLM(["This plan has strong momentum."]))
        .with_system_prompt("You are optimistic.")
        .build()
    )
    pessimist = (
        AgentBuilder("pessimist")
        .with_llm(MockLLM(["This plan still carries execution risk."]))
        .with_system_prompt("You are pessimistic.")
        .build()
    )

    orchestrator = ParallelOrchestrator([optimist, pessimist])
    result = await orchestrator.run(Context(user_input="Review this launch plan"))
    print(result.final_content)


if __name__ == "__main__":
    asyncio.run(main())

