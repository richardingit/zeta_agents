from __future__ import annotations

import asyncio

from agent_sdk import AgentBuilder
from agent_sdk.llm import MockLLM
from agent_sdk.memory import InMemoryStore
from agent_sdk.skills import SkillsModule


async def main() -> None:
    llm = MockLLM(responses=["Hello! How can I help?"])
    memory = InMemoryStore()
    skills = SkillsModule()

    agent = (
        AgentBuilder("assistant")
        .with_llm(llm)
        .with_system_prompt("You are a helpful assistant.")
        .with_memory(memory)
        .with_skills(skills, [])
        .build()
    )

    result = await agent.run("Say hello")
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
