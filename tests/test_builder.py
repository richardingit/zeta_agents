from agent_sdk import AgentBuilder
from agent_sdk.llm import MockLLM


def test_builder_builds_agent_bundle():
    bundle = (
        AgentBuilder("assistant")
        .with_llm(MockLLM(["hello"]))
        .with_system_prompt("You are helpful.")
        .build()
    )
    assert bundle.agent.name == "assistant"
    assert bundle.llm is not None


def test_builder_requires_llm():
    builder = AgentBuilder("assistant")
    try:
        builder.build()
    except ValueError as exc:
        assert "requires an LLM module" in str(exc)
    else:
        raise AssertionError("Expected ValueError when llm is missing")
