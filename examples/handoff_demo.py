from __future__ import annotations

import asyncio

from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import HandoffNode, HandoffOrchestrator


async def main() -> None:
    triage = (
        AgentBuilder("triage")
        .with_llm(MockLLM(["Please see billing.\nHANDOFF_TO: billing: payment issue"]))
        .with_system_prompt("You are triage.")
        .build()
    )
    billing = (
        AgentBuilder("billing")
        .with_llm(MockLLM(["Your billing issue has been resolved."]))
        .with_system_prompt("You are billing.")
        .build()
    )

    orchestrator = HandoffOrchestrator(
        entry_agent="triage",
        nodes={
            "triage": HandoffNode(triage, can_handoff_to=["billing"]),
            "billing": HandoffNode(billing, can_handoff_to=[]),
        },
    )
    result = await orchestrator.run(Context(user_input="I need help with a payment"))
    print(result.final_content)


if __name__ == "__main__":
    asyncio.run(main())

