from agent_sdk import AgentBuilder
from agent_sdk.core import EventBus
from agent_sdk.llm import FunctionLLM


async def test_agent_run_stream_emits_text_and_done_events():
    async def complete(messages, **kwargs):
        return "hello world"

    async def stream(messages, **kwargs):
        for token in ["hello", " ", "world"]:
            yield token

    agent = (
        AgentBuilder("assistant")
        .with_llm(FunctionLLM(complete, stream_fn=stream))
        .with_system_prompt("You are helpful.")
        .build()
    )

    events = [event async for event in agent.run_stream("Say hello")]

    assert [event.content for event in events if event.type == "text"] == ["hello", " ", "world"]
    assert events[-1].type == "done"
    assert events[-1].output is not None
    assert events[-1].output.content == "hello world"


async def test_agent_run_stream_emits_event_bus_stream_events():
    seen = []
    bus = EventBus()
    bus.subscribe("llm.stream.*", lambda event: seen.append(event.type))

    async def complete(messages, **kwargs):
        return "ok"

    agent = (
        AgentBuilder("assistant")
        .with_llm(FunctionLLM(complete))
        .with_event_bus(bus)
        .with_system_prompt("You are helpful.")
        .build()
    )

    _ = [event async for event in agent.run_stream("Say hello")]

    assert "llm.stream.started" in seen
    assert "llm.stream.delta" in seen
    assert "llm.stream.completed" in seen
