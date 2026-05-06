from agent_sdk.core import Context
from agent_sdk.core.types import Message, Role
from agent_sdk.persistence import InMemoryCheckpointer


async def test_inmemory_checkpointer_round_trip():
    ctx = Context(user_input="hello", state={"step": 1}, metadata={"source": "test"})
    ctx.messages.append(Message(role=Role.USER, content="hello"))
    checkpointer = InMemoryCheckpointer()

    meta = await checkpointer.save(ctx, label="initial")
    restored = await checkpointer.restore(meta.checkpoint_id, Context())

    assert restored.user_input == "hello"
    assert restored.state == {"step": 1}
    assert restored.metadata == {"source": "test"}
    assert restored.messages[0].content == "hello"

