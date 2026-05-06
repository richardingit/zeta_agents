"""
Checkpointer - 状态持久化与时间旅行。

核心能力:
- 保存任意时刻的 Context 快照
- 通过 checkpoint_id 恢复任意快照
- 列出 session 下所有快照(支持回溯调试)
- 支持自动保存(Agent 每完成一步自动 checkpoint)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any
import time
import json
import copy

from agent_sdk.core.context import Context
from agent_sdk.core.types import Message, Role, ToolCall, gen_id


@dataclass
class CheckpointMeta:
    """Checkpoint 元数据(不包含完整快照)"""
    checkpoint_id: str
    session_id: str
    label: str = ""           # 业务可读标签,如 "after_research"
    timestamp: float = 0.0
    parent_id: str | None = None  # 上一个 checkpoint(形成链)
    metadata: dict = field(default_factory=dict)


@dataclass
class Checkpoint:
    """完整快照"""
    meta: CheckpointMeta
    # 序列化后的 Context 数据(只保存可序列化部分,模块实例不存)
    user_input: str = ""
    messages: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class Checkpointer(ABC):
    """状态持久化接口"""

    @abstractmethod
    async def save(
        self,
        ctx: Context,
        label: str = "",
        parent_id: str | None = None,
    ) -> CheckpointMeta:
        """保存当前 Context 为新的 checkpoint,返回元数据"""

    @abstractmethod
    async def load(self, checkpoint_id: str) -> Checkpoint:
        """加载指定 checkpoint"""

    @abstractmethod
    async def restore(
        self,
        checkpoint_id: str,
        target_ctx: Context,
    ) -> Context:
        """
        把 checkpoint 恢复到 target_ctx 上。
        模块实例(llm/memory 等)从 target_ctx 继承,只恢复可序列化数据。
        """

    @abstractmethod
    async def list_checkpoints(self, session_id: str) -> list[CheckpointMeta]:
        """列出某 session 的所有 checkpoint(按时间升序)"""

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        ...


def _serialize_messages(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        d = {
            "role": m.role.value,
            "content": m.content,
            "tool_call_id": m.tool_call_id,
            "name": m.name,
            "metadata": m.metadata,
            "timestamp": m.timestamp,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in m.tool_calls
            ],
        }
        out.append(d)
    return out


def _deserialize_messages(data: list[dict]) -> list[Message]:
    out = []
    for d in data:
        tcs = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in d.get("tool_calls", [])
        ]
        out.append(Message(
            role=Role(d["role"]),
            content=d["content"],
            tool_calls=tcs,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", time.time()),
        ))
    return out


class InMemoryCheckpointer(Checkpointer):
    """内存版实现 - 用于开发/测试。生产环境换 Postgres/Redis 实现。"""

    def __init__(self):
        self._checkpoints: dict[str, Checkpoint] = {}
        # session_id → list[checkpoint_id]  (有序)
        self._by_session: dict[str, list[str]] = {}

    async def save(
        self,
        ctx: Context,
        label: str = "",
        parent_id: str | None = None,
    ) -> CheckpointMeta:
        cp_id = gen_id("ckpt_")
        meta = CheckpointMeta(
            checkpoint_id=cp_id,
            session_id=ctx.session_id,
            label=label,
            timestamp=time.time(),
            parent_id=parent_id,
            metadata=copy.deepcopy(ctx.metadata),
        )
        cp = Checkpoint(
            meta=meta,
            user_input=ctx.user_input,
            messages=_serialize_messages(ctx.messages),
            state=copy.deepcopy(ctx.state),
            metadata=copy.deepcopy(ctx.metadata),
        )
        self._checkpoints[cp_id] = cp
        self._by_session.setdefault(ctx.session_id, []).append(cp_id)
        return meta

    async def load(self, checkpoint_id: str) -> Checkpoint:
        cp = self._checkpoints.get(checkpoint_id)
        if cp is None:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")
        return cp

    async def restore(
        self,
        checkpoint_id: str,
        target_ctx: Context,
    ) -> Context:
        cp = await self.load(checkpoint_id)
        # 把可序列化数据恢复到 target_ctx
        target_ctx.session_id = cp.meta.session_id
        target_ctx.user_input = cp.user_input
        target_ctx.messages = _deserialize_messages(cp.messages)
        target_ctx.state = copy.deepcopy(cp.state)
        target_ctx.metadata = copy.deepcopy(cp.metadata)
        return target_ctx

    async def list_checkpoints(self, session_id: str) -> list[CheckpointMeta]:
        ids = self._by_session.get(session_id, [])
        return [self._checkpoints[i].meta for i in ids]

    async def delete(self, checkpoint_id: str) -> bool:
        cp = self._checkpoints.pop(checkpoint_id, None)
        if cp is None:
            return False
        sess_list = self._by_session.get(cp.meta.session_id, [])
        if checkpoint_id in sess_list:
            sess_list.remove(checkpoint_id)
        return True


class JSONFileCheckpointer(Checkpointer):
    """
    JSON 文件版 - 简单的本地持久化。
    每个 checkpoint 一个文件,session 索引存一个 index.json。
    """

    def __init__(self, base_dir: str):
        from pathlib import Path
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._index_path = self._base / "index.json"
        self._index: dict[str, list[str]] = {}
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text())

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def _cp_path(self, cp_id: str):
        return self._base / f"{cp_id}.json"

    async def save(self, ctx, label="", parent_id=None) -> CheckpointMeta:
        cp_id = gen_id("ckpt_")
        meta = CheckpointMeta(
            checkpoint_id=cp_id,
            session_id=ctx.session_id,
            label=label,
            timestamp=time.time(),
            parent_id=parent_id,
            metadata=copy.deepcopy(ctx.metadata),
        )
        cp = Checkpoint(
            meta=meta,
            user_input=ctx.user_input,
            messages=_serialize_messages(ctx.messages),
            state=copy.deepcopy(ctx.state),
            metadata=copy.deepcopy(ctx.metadata),
        )
        self._cp_path(cp_id).write_text(json.dumps({
            "meta": asdict(meta),
            "user_input": cp.user_input,
            "messages": cp.messages,
            "state": cp.state,
            "metadata": cp.metadata,
        }, indent=2, default=str))
        self._index.setdefault(ctx.session_id, []).append(cp_id)
        self._save_index()
        return meta

    async def load(self, checkpoint_id: str) -> Checkpoint:
        path = self._cp_path(checkpoint_id)
        if not path.exists():
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")
        data = json.loads(path.read_text())
        meta = CheckpointMeta(**data["meta"])
        return Checkpoint(
            meta=meta,
            user_input=data["user_input"],
            messages=data["messages"],
            state=data["state"],
            metadata=data["metadata"],
        )

    async def restore(self, checkpoint_id, target_ctx) -> Context:
        cp = await self.load(checkpoint_id)
        target_ctx.session_id = cp.meta.session_id
        target_ctx.user_input = cp.user_input
        target_ctx.messages = _deserialize_messages(cp.messages)
        target_ctx.state = copy.deepcopy(cp.state)
        target_ctx.metadata = copy.deepcopy(cp.metadata)
        return target_ctx

    async def list_checkpoints(self, session_id) -> list[CheckpointMeta]:
        ids = self._index.get(session_id, [])
        result = []
        for i in ids:
            try:
                cp = await self.load(i)
                result.append(cp.meta)
            except KeyError:
                pass
        return result

    async def delete(self, checkpoint_id) -> bool:
        path = self._cp_path(checkpoint_id)
        if not path.exists():
            return False
        # 找 session 并清理索引
        try:
            cp = await self.load(checkpoint_id)
            sess_list = self._index.get(cp.meta.session_id, [])
            if checkpoint_id in sess_list:
                sess_list.remove(checkpoint_id)
                self._save_index()
        except Exception:
            pass
        path.unlink()
        return True
