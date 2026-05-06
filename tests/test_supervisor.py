from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import SupervisorOrchestrator


async def test_supervisor_delegate_then_finalize():
    supervisor = (
        AgentBuilder("supervisor")
        .with_llm(
            MockLLM([
                "DELEGATE: researcher: Find two agent frameworks",
                "FINAL_ANSWER: LangGraph and OpenAI Agents SDK",
            ])
        )
        .with_system_prompt("You are a supervisor.")
        .build()
    )
    researcher = (
        AgentBuilder("researcher", description="finds framework facts")
        .with_llm(MockLLM(["LangGraph and OpenAI Agents SDK are widely used."]))
        .with_system_prompt("You are a researcher.")
        .build()
    )

    orchestrator = SupervisorOrchestrator(
        supervisor_bundle=supervisor,
        workers={"researcher": researcher},
        max_rounds=4,
    )

    output = await orchestrator.run(Context(user_input="What frameworks should I study?"))

    assert output.final_content == "LangGraph and OpenAI Agents SDK"
    assert output.sequence == ["supervisor", "researcher", "supervisor"]
