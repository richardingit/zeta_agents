from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import HandoffNode, HandoffOrchestrator


async def test_handoff_transfers_control():
    triage = (
        AgentBuilder("triage")
        .with_llm(MockLLM(["Please see billing.\nHANDOFF_TO: billing: payment issue"]))
        .with_system_prompt("You are triage.")
        .build()
    )
    billing = (
        AgentBuilder("billing")
        .with_llm(MockLLM(["Your invoice has been updated."]))
        .with_system_prompt("You are billing.")
        .build()
    )

    orchestrator = HandoffOrchestrator(
        entry_agent="triage",
        nodes={
            "triage": HandoffNode(triage, can_handoff_to=["billing"]),
            "billing": HandoffNode(billing, can_handoff_to=[]),
        },
        max_handoffs=3,
    )

    output = await orchestrator.run(Context(user_input="I was charged twice"))

    assert output.sequence == ["triage", "billing"]
    assert "invoice" in output.final_content
