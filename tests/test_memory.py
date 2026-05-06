from agent_sdk.memory import InMemoryStore


async def test_memory_remember_and_recall():
    memory = InMemoryStore()
    await memory.remember("LangGraph is useful for workflow control", {"agent": "researcher"})
    await memory.remember("CrewAI is fast to prototype", {"agent": "researcher"})

    results = await memory.recall("LangGraph", k=5)

    assert len(results) == 1
    assert "LangGraph" in results[0].content
