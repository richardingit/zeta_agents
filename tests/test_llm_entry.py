from agent_sdk import AgentBuilder
from agent_sdk.core import Context, Message, Role
from agent_sdk.llm import FunctionLLM, LLMConfig, OpenAICompatibleProvider, create_llm


async def test_function_llm_can_power_agent():
    async def complete(messages, tools=None, model=None, temperature=0.7, **kwargs):
        return "hello from function llm"

    bundle = (
        AgentBuilder("assistant")
        .with_llm(FunctionLLM(complete))
        .with_system_prompt("You are helpful.")
        .build()
    )

    result = await bundle.agent.run(Context(user_input="Say hi", llm=bundle.llm))
    assert result.content == "hello from function llm"


def test_create_llm_supports_openai_compatible():
    llm = create_llm(
        "openai_compatible",
        model="demo-model",
        api_key="test-key",
        base_url="http://localhost:8000/v1",
    )
    assert isinstance(llm, OpenAICompatibleProvider)


async def test_openai_compatible_provider_parses_response():
    captured = {}

    def fake_transport(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "model": "demo-model",
            "choices": [
                {
                    "message": {
                        "content": "gateway reply",
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

    provider = OpenAICompatibleProvider(
        model="demo-model",
        api_key="secret",
        base_url="http://localhost:8000/v1",
        transport=fake_transport,
    )
    response = await provider.complete(
        messages=[Message(role=Role.USER, content="hello")],
        temperature=0.1,
    )

    assert response.content == "gateway reply"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "demo-model"


def test_builder_with_llm_config_creates_provider():
    bundle = (
        AgentBuilder("assistant")
        .with_llm_config(
            LLMConfig(
                provider="openai_compatible",
                model="demo-model",
                api_key="secret",
                base_url="http://localhost:8000/v1",
            )
        )
        .build()
    )
    assert isinstance(bundle.llm, OpenAICompatibleProvider)


async def test_function_llm_supports_streaming():
    async def complete(messages, **kwargs):
        return "hello world"

    async def stream(messages, **kwargs):
        for token in ["hello", " ", "world"]:
            yield token

    llm = FunctionLLM(complete, stream_fn=stream)
    chunks = [chunk async for chunk in llm.stream(messages=[])]

    assert [chunk.content for chunk in chunks if chunk.type == "text"] == ["hello", " ", "world"]
    assert chunks[-1].type == "done"


async def test_openai_compatible_provider_supports_stream_transport():
    def fake_stream_transport(url, headers, payload, timeout):
        return [
            {
                "model": "demo-model",
                "choices": [{"delta": {"content": "hello"}}],
            },
            {
                "model": "demo-model",
                "choices": [{"delta": {"content": " world"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        ]

    provider = OpenAICompatibleProvider(
        model="demo-model",
        api_key="secret",
        base_url="http://localhost:8000/v1",
        stream_transport=fake_stream_transport,
    )
    chunks = [chunk async for chunk in provider.stream(messages=[Message(role=Role.USER, content="hello")])]

    assert [chunk.content for chunk in chunks if chunk.type == "text"] == ["hello", " world"]
    usage_chunks = [chunk for chunk in chunks if chunk.type == "usage"]
    assert usage_chunks[0].usage["completion_tokens"] == 2
    assert chunks[-1].type == "done"
