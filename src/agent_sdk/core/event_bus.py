"""
EventBus - 框架的事件总线。

所有模块通过 emit/subscribe 解耦通信。
Observability、Audit、Guardrail 等模块都基于它实现。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any
from collections import defaultdict
import asyncio
import time
import logging

from agent_sdk.core.types import gen_id


# 标准事件类型常量(保持字符串,允许业务自定义扩展)
class EventType:
    # Agent 生命周期
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    # LLM 调用
    LLM_CALL_STARTED = "llm.call.started"
    LLM_CALL_COMPLETED = "llm.call.completed"
    LLM_CALL_FAILED = "llm.call.failed"

    # 工具调用
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"

    # 编排
    ORCHESTRATOR_STARTED = "orchestrator.started"
    ORCHESTRATOR_COMPLETED = "orchestrator.completed"
    AGENT_HANDOFF = "orchestrator.handoff"

    # Memory
    MEMORY_WRITTEN = "memory.written"
    MEMORY_RECALLED = "memory.recalled"

    # Checkpoint
    CHECKPOINT_SAVED = "checkpoint.saved"
    CHECKPOINT_LOADED = "checkpoint.loaded"


@dataclass
class Event:
    type: str
    session_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: gen_id("evt_"))


# 同步和异步 handler 都支持
EventHandler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """
    异步事件总线。
    
    特性:
    - 支持通配符订阅:'*' 监听所有,'agent.*' 监听 agent.* 类
    - 同步/异步 handler 都支持
    - emit 不阻塞:handler 异常被捕获并日志,不影响主流程
    - 支持中间件(在 emit 前对事件做修饰/过滤)
    """

    def __init__(self):
        # 事件类型 → handler 列表
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        # 中间件:对所有 event 生效
        self._middlewares: list[Callable[[Event], Event | None]] = []
        self._logger = logging.getLogger("agent_framework.event_bus")

    def subscribe(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        """
        订阅事件。返回 unsubscribe 函数。
        
        event_type 支持:
        - 精确匹配: "agent.started"
        - 前缀通配: "agent.*"
        - 全部:    "*"
        """
        self._handlers[event_type].append(handler)

        def unsubscribe():
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

        return unsubscribe

    def add_middleware(self, mw: Callable[[Event], Event | None]) -> None:
        """添加中间件。返回 None 则丢弃事件,返回 Event 则继续传递。"""
        self._middlewares.append(mw)

    async def emit(self, event: Event) -> None:
        """触发事件。不阻塞主流程,handler 异常会被捕获。"""
        # 走中间件
        for mw in self._middlewares:
            result = mw(event)
            if result is None:
                return
            event = result

        # 收集匹配的 handlers
        matched: list[EventHandler] = []
        matched.extend(self._handlers.get(event.type, []))
        matched.extend(self._handlers.get("*", []))
        # 通配符前缀:agent.* 匹配 agent.started, agent.completed 等
        for pattern, handlers in self._handlers.items():
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if event.type.startswith(prefix + "."):
                    matched.extend(handlers)

        # 并行执行所有 handler
        tasks = []
        for h in matched:
            tasks.append(self._safe_call(h, event))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def emit_quick(
        self, event_type: str, session_id: str = "", **payload
    ) -> None:
        """便捷方法:直接传 type + payload,自动构造 Event"""
        await self.emit(Event(type=event_type, session_id=session_id, payload=payload))

    async def _safe_call(self, handler: EventHandler, event: Event) -> None:
        """调用 handler 并捕获异常"""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            self._logger.exception(
                "EventBus handler raised exception for event %s: %s",
                event.type, e,
            )


# 全局默认 EventBus(可被覆盖)
_default_bus: EventBus | None = None


def get_default_bus() -> EventBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def set_default_bus(bus: EventBus) -> None:
    global _default_bus
    _default_bus = bus
