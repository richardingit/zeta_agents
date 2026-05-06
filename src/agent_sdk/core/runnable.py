"""
Runnable - 编排树里的统一节点抽象。

任何"可以被编排的东西"——AgentBundle、Orchestrator、自定义子类——都通过
Runnable 暴露 (name, run(ctx), run_stream(ctx)) 三件套。这让 PipelineStage /
ParallelOrchestrator / HandoffNode / SupervisorOrchestrator.workers 可以接受
其它 Orchestrator 作为节点,从而实现编排器的任意嵌套。

设计要点
========
- 调用方(Orchestrator)负责给每个节点 fork 出隔离的 sub_ctx,Runnable 实现
  内部"不再 fork",只在 ctx 上做必要的模块注入(BundleRunnable)或直接透传
  (OrchestratorRunnable)。这样 fork 链路清晰、不会重复 fork。
- OrchestratorRunnable 把 OrchestratorOutput 摊平成 AgentOutput,把
  OrchestratorStreamEvent 翻译成 AgentStreamEvent,使 orchestrator 可以伪装
  成一个"超级 agent"塞进上层编排里。

用法
====
    pipeline = PipelineOrchestrator([
        PipelineStage(researcher_bundle),
        PipelineStage(ParallelOrchestrator([writer_a, writer_b])),  # 嵌套!
        PipelineStage(reviewer_bundle),
    ])
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Union

from agent_sdk.core.types import AgentOutput, AgentStreamEvent

if TYPE_CHECKING:
    from agent_sdk.core.context import Context
    from agent_sdk.runtime.builder import AgentBundle
    from agent_sdk.orchestrators.base import Orchestrator, OrchestratorOutput


class Runnable(ABC):
    """编排节点抽象。

    实现者必须提供 ``name`` 属性以及 ``run`` / ``run_stream`` 两种执行模式。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """节点名,用于事件 / 日志 / 可视化"""
        ...

    @abstractmethod
    async def run(self, ctx: "Context") -> AgentOutput:
        """阻塞执行,返回最终聚合产出"""
        ...

    @abstractmethod
    def run_stream(self, ctx: "Context") -> AsyncIterator[AgentStreamEvent]:
        """流式执行,逐步产出 AgentStreamEvent。最后一个事件 type='done'"""
        ...


class BundleRunnable(Runnable):
    """把 AgentBundle 适配成 Runnable。

    职责:在 ``ctx`` 上注入 bundle 携带的模块(llm / memory / skills / tools /
    event_bus),然后委托给 ``bundle.agent``。**不**对 ctx 再 fork——调用方应已
    fork 出隔离的 sub_ctx。
    """

    def __init__(self, bundle: "AgentBundle"):
        self._bundle = bundle

    @property
    def bundle(self) -> "AgentBundle":
        return self._bundle

    @property
    def name(self) -> str:
        return self._bundle.agent.name

    async def run(self, ctx: "Context") -> AgentOutput:
        return await self._bundle.agent.run(self._inject(ctx))

    async def run_stream(self, ctx: "Context") -> AsyncIterator[AgentStreamEvent]:
        async for event in self._bundle.agent.run_stream(self._inject(ctx)):
            yield event

    def _inject(self, ctx: "Context") -> "Context":
        """把 bundle 的模块覆盖到 ctx 上(原地修改,要求 ctx 是隔离的 sub_ctx)"""
        ctx.llm = self._bundle.llm
        if self._bundle.memory is not None:
            ctx.memory = self._bundle.memory
        if self._bundle.skills is not None:
            ctx.skills = self._bundle.skills
        if self._bundle.tools is not None:
            ctx.tools = self._bundle.tools
        if self._bundle.event_bus is not None:
            ctx.event_bus = self._bundle.event_bus
        return ctx


class OrchestratorRunnable(Runnable):
    """把 Orchestrator 适配成 Runnable,实现编排器的任意嵌套。

    职责:
        - ``run``      : 调用 ``orchestrator.run``,把 OrchestratorOutput 摊平成
                         AgentOutput
        - ``run_stream``: 调用 ``orchestrator.run_stream``,把
                         OrchestratorStreamEvent 翻译成 AgentStreamEvent。其中:
            * 内层 agent_event 透传(让上层 Pipeline 能看到子 agent 的真实流式)
            * 内层 orchestrator 自身的 'done' 事件转为 AgentStreamEvent('done',
              output=摊平后的 AgentOutput)
            * 其它结构性事件(start / handoff)被吞掉,因为它们对上层 agent 视角
              没有意义
    """

    def __init__(self, orchestrator: "Orchestrator"):
        self._orch = orchestrator

    @property
    def orchestrator(self) -> "Orchestrator":
        return self._orch

    @property
    def name(self) -> str:
        # Orchestrator 没有强制 name 字段,用类名作为兜底
        return getattr(self._orch, "name", type(self._orch).__name__)

    async def run(self, ctx: "Context") -> AgentOutput:
        out = await self._orch.run(ctx)
        return _orchestrator_output_to_agent_output(self.name, out)

    async def run_stream(self, ctx: "Context") -> AsyncIterator[AgentStreamEvent]:
        node_name = self.name
        async for orch_event in self._orch.run_stream(ctx):
            # 内层 agent 的细粒度事件直接透传,这样上层能看到逐 token 流
            if orch_event.agent_event is not None:
                yield orch_event.agent_event
                continue

            # 内层 orchestrator 自身完成 → 翻译成 done AgentStreamEvent
            if orch_event.type == "done":
                final_output = (
                    _orchestrator_output_to_agent_output(node_name, orch_event.output)
                    if orch_event.output is not None
                    else AgentOutput(agent_name=node_name, content=orch_event.content)
                )
                yield AgentStreamEvent(
                    type="done",
                    agent_name=node_name,
                    content=final_output.content,
                    output=final_output,
                )
                continue

            # start / handoff 等结构性事件:对上层 agent 视角无意义,丢弃


# 公共类型别名:任何能被 to_runnable 接受的输入
RunnableLike = Union["AgentBundle", "Orchestrator", Runnable]


def to_runnable(target: RunnableLike) -> Runnable:
    """工厂:统一把 AgentBundle / Orchestrator / Runnable 适配成 Runnable。

    幂等——如果已经是 Runnable,直接返回。
    """
    if isinstance(target, Runnable):
        return target

    # 延迟导入避免顶层循环依赖
    from agent_sdk.runtime.builder import AgentBundle
    from agent_sdk.orchestrators.base import Orchestrator

    if isinstance(target, AgentBundle):
        return BundleRunnable(target)
    if isinstance(target, Orchestrator):
        return OrchestratorRunnable(target)
    raise TypeError(
        f"Cannot convert {type(target).__name__} to Runnable. "
        f"Expected AgentBundle, Orchestrator, or Runnable subclass."
    )


def _orchestrator_output_to_agent_output(
    name: str, out: "OrchestratorOutput"
) -> AgentOutput:
    """OrchestratorOutput → AgentOutput 摊平规则。

    - final_content   → content
    - 各子 agent 的 messages / tool_calls 全部串接 → 上层可见
    - sequence 长度   → iterations(代表"内部走了多少步")
    """
    messages = []
    tool_calls = []
    for ag_out in out.agent_outputs:
        messages.extend(ag_out.messages)
        tool_calls.extend(ag_out.tool_calls_made)
    return AgentOutput(
        agent_name=name,
        content=out.final_content,
        messages=messages,
        tool_calls_made=tool_calls,
        iterations=len(out.sequence),
    )


__all__ = [
    "Runnable",
    "BundleRunnable",
    "OrchestratorRunnable",
    "RunnableLike",
    "to_runnable",
]
