from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import ParallelOrchestrator


async def test_parallel_aggregates_outputs():
    optimist = (
        AgentBuilder("optimist")
        .with_llm(MockLLM(["This launch looks promising."]))
        .with_system_prompt("You are optimistic.")
        .build()
    )
    pessimist = (
        AgentBuilder("pessimist")
        .with_llm(MockLLM(["This launch still has risks."]))
        .with_system_prompt("You are pessimistic.")
        .build()
    )

    orchestrator = ParallelOrchestrator([optimist, pessimist])
    output = await orchestrator.run(Context(user_input="Review the launch plan"))

    assert output.sequence == ["optimist", "pessimist"]
    assert "optimist" in output.final_content.lower()
    assert "pessimist" in output.final_content.lower()
