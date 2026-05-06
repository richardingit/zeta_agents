from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import FunctionLLM, MockLLM
from agent_sdk.orchestrators import (
    HandoffNode,
    HandoffOrchestrator,
    ParallelOrchestrator,
    PipelineOrchestrator,
    PipelineStage,
    SupervisorOrchestrator,
)


async def test_pipeline_run_stream_emits_done():
    async def complete(messages, **kwargs):
        return "done"

    bundle = (
        AgentBuilder("researcher")
        .with_llm(FunctionLLM(complete))
        .with_system_prompt("You are helpful.")
        .build()
    )
    orchestrator = PipelineOrchestrator([PipelineStage(bundle)])
    events = [event async for event in orchestrator.run_stream(Context(user_input="hello"))]
    assert events[0].type == "start"
    assert events[-1].type == "done"
    assert events[-1].output is not None


async def test_supervisor_run_stream_emits_handoff():
    supervisor = (
        AgentBuilder("supervisor")
        .with_llm(MockLLM([
            "DELEGATE: worker: do the task",
            "FINAL_ANSWER: done",
        ]))
        .with_system_prompt("You are supervisor.")
        .build()
    )
    worker = (
        AgentBuilder("worker", description="does work")
        .with_llm(MockLLM(["worker result"]))
        .with_system_prompt("You are worker.")
        .build()
    )
    orchestrator = SupervisorOrchestrator(supervisor_bundle=supervisor, workers={"worker": worker})
    events = [event async for event in orchestrator.run_stream(Context(user_input="hello"))]
    assert any(event.type == "handoff" for event in events)
    assert events[-1].type == "done"


async def test_handoff_run_stream_emits_handoff():
    triage = (
        AgentBuilder("triage")
        .with_llm(MockLLM(["go\nHANDOFF_TO: billing: payment"]))
        .with_system_prompt("triage")
        .build()
    )
    billing = (
        AgentBuilder("billing")
        .with_llm(MockLLM(["billing answer"]))
        .with_system_prompt("billing")
        .build()
    )
    orchestrator = HandoffOrchestrator(
        entry_agent="triage",
        nodes={
            "triage": HandoffNode(triage, can_handoff_to=["billing"]),
            "billing": HandoffNode(billing, can_handoff_to=[]),
        },
    )
    events = [event async for event in orchestrator.run_stream(Context(user_input="hello"))]
    assert any(event.type == "handoff" for event in events)
    assert events[-1].type == "done"


async def test_parallel_run_stream_emits_done():
    a = AgentBuilder("a").with_llm(MockLLM(["A"])).with_system_prompt("a").build()
    b = AgentBuilder("b").with_llm(MockLLM(["B"])).with_system_prompt("b").build()
    orchestrator = ParallelOrchestrator([a, b])
    events = [event async for event in orchestrator.run_stream(Context(user_input="hello"))]
    assert events[0].type == "start"
    assert events[-1].type == "done"
