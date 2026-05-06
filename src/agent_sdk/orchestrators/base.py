"""
Orchestrator 模块 - 多 agent 协作编排。

提供四种内置协作模式:
- Pipeline:   流水线串行(researcher → writer → reviewer)
- Supervisor: 主管调度子 agent(动态决策)
- Handoff:    Agent 之间自由转交控制权(像 OpenAI Swarm)
- Parallel:   并行执行 + 聚合结果
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time

from agent_sdk.core.context import Context
from agent_sdk.core.types import AgentOutput
from agent_sdk.core.event_bus import EventType


@dataclass
class OrchestratorOutput:
    """Orchestrator 一次运行的总输出"""
    orchestrator_type: str
    final_content: str = ""
    agent_outputs: list[AgentOutput] = field(default_factory=list)  # 各 agent 的产出
    sequence: list[str] = field(default_factory=list)               # 执行顺序(agent name)
    metadata: dict = field(default_factory=dict)


class Orchestrator(ABC):
    """编排基类"""

    @abstractmethod
    async def run(self, ctx: Context) -> OrchestratorOutput:
        ...

    async def _emit_started(self, ctx: Context, payload: dict | None = None):
        if ctx.event_bus:
            await ctx.event_bus.emit_quick(
                EventType.ORCHESTRATOR_STARTED,
                session_id=ctx.session_id,
                orchestrator_type=type(self).__name__,
                **(payload or {}),
            )

    async def _emit_completed(self, ctx: Context, payload: dict | None = None):
        if ctx.event_bus:
            await ctx.event_bus.emit_quick(
                EventType.ORCHESTRATOR_COMPLETED,
                session_id=ctx.session_id,
                orchestrator_type=type(self).__name__,
                **(payload or {}),
            )

    async def _emit_handoff(self, ctx: Context, from_agent: str, to_agent: str, reason: str = ""):
        if ctx.event_bus:
            await ctx.event_bus.emit_quick(
                EventType.AGENT_HANDOFF,
                session_id=ctx.session_id,
                from_agent=from_agent,
                to_agent=to_agent,
                reason=reason,
            )

    @staticmethod
    def _build_sub_ctx(parent_ctx: Context, bundle) -> Context:
        """
        从父 ctx 派生子 ctx,并用 bundle 内的模块覆盖。
        这样 orchestrator 不要求外部 ctx 必须注入所有模块,
        每个 agent 用自己 bundle 里配的模块。
        """
        sub = parent_ctx.fork()
        sub.messages = []
        sub.llm = bundle.llm
        if bundle.memory:
            sub.memory = bundle.memory
        if bundle.skills:
            sub.skills = bundle.skills
        if bundle.tools:
            sub.tools = bundle.tools
        if bundle.event_bus:
            sub.event_bus = bundle.event_bus
        return sub
