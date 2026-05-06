from agent_sdk.core import Message, Role
from agent_sdk.experimental import InMemoryMemoryBackend, MemoryContextBuilder, MemoryRecord, MemoryScope


async def test_external_memory_manager_prepares_scoped_messages():
    backend = InMemoryMemoryBackend()
    builder = MemoryContextBuilder(backend)
    await backend.add(
        MemoryRecord(
            content="User prefers concise answers.",
            user_id="u1",
            model="m1",
            agent_name="assistant",
            project="p1",
        )
    )
    await backend.add(
        MemoryRecord(
            content="Different user memory should not leak.",
            user_id="u2",
            model="m1",
            agent_name="assistant",
            project="p1",
        )
    )

    messages = await builder.prepare_messages(
        scope=MemoryScope(user_id="u1", model="m1", agent_name="assistant", project="p1"),
        user_input="How should you answer me?",
        base_system_prompt="You are helpful.",
        history=[Message(role=Role.ASSISTANT, content="Previous reply")],
    )

    assert messages[0].role == Role.SYSTEM
    assert "User prefers concise answers." in messages[0].content
    assert "Different user memory should not leak." not in messages[0].content
    assert messages[1].content == "Previous reply"
    assert messages[-1].role == Role.USER

