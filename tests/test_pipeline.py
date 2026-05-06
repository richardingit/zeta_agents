from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import PipelineOrchestrator, PipelineStage


async def test_pipeline_runs_in_order_and_saves_state():
    researcher = (
        AgentBuilder("researcher")
        .with_llm(MockLLM(["research notes"]))
        .with_system_prompt("You are a researcher.")
        .build()
    )
    writer = (
        AgentBuilder("writer")
        .with_llm(MockLLM(["final draft"]))
        .with_system_prompt("You are a writer.")
        .build()
    )
    ctx = Context(user_input="Write a report")
    pipeline = PipelineOrchestrator([
        PipelineStage(researcher, save_to_state="research"),
        PipelineStage(writer, input_template="Use this:\n{previous_output}"),
    ])

    output = await pipeline.run(ctx)

    assert output.sequence == ["researcher", "writer"]
    assert output.final_content == "final draft"
    assert ctx.state["research"] == "research notes"
