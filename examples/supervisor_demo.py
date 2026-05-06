from __future__ import annotations

import asyncio

from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import SupervisorOrchestrator


async def main() -> None:
    supervisor = (
        AgentBuilder("supervisor")
        .with_llm(
            MockLLM([
                "DELEGATE: researcher: Find two popular multi-agent frameworks",
                "FINAL_ANSWER: LangGraph and OpenAI Agents SDK are two good starting points.",
            ])
        )
        .with_system_prompt("You are a supervisor.")
        .build()
    )
    researcher = (
        AgentBuilder("researcher", description="collects framework facts")
        .with_llm(MockLLM(["LangGraph and OpenAI Agents SDK are widely used."]))
        .with_system_prompt("You are a researcher.")
        .build()
    )

    orchestrator = SupervisorOrchestrator(
        supervisor_bundle=supervisor,
        workers={"researcher": researcher},
    )
    result = await orchestrator.run(Context(user_input="What frameworks should I learn first?"))
    print(result.final_content)


if __name__ == "__main__":
    asyncio.run(main())

