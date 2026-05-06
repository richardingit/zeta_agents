"""
Pipeline Orchestrator - 流水线串行执行。

适用场景:
- 固定顺序的多步任务
- 每步输出作为下一步输入
- 例:researcher → writer → reviewer

每个阶段(PipelineStage)的节点可以是 AgentBundle、其它 Orchestrator,
或任意 Runnable 子类。这意味着 Pipeline 内可以嵌套别的 Pipeline / Parallel /
Supervisor / Handoff,组合出任意层级的工作流。
"""
from __future__ import annotations
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from agent_sdk.core.context import Context
from agent_sdk.core.runnable import Runnable, RunnableLike, to_runnable
from agent_sdk.core.types import AgentOutput
from agent_sdk.orchestrators.base import Orchestrator, OrchestratorOutput, OrchestratorStreamEvent


@dataclass
class PipelineStage:
    """流水线的一个阶段。

    `node` 可以是 AgentBundle、Orchestrator 或 Runnable;构造时统一转换为
    Runnable,简化下游逻辑。
    """
    node: RunnableLike
    # 上一阶段输出如何作为本阶段输入(可选,默认直接传)
    input_template: str = "{previous_output}"
    # 本阶段输出存到 ctx.state 的 key(可选)
    save_to_state: str | None = None
    # 内部:统一转换后的 Runnable(由 __post_init__ 填充)
    runnable: Runnable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.runnable = to_runnable(self.node)


class PipelineOrchestrator(Orchestrator):
    """
    按顺序执行一系列节点,每个节点看到上一个节点的输出。

    用法:
        pipeline = PipelineOrchestrator([
            PipelineStage(researcher_bundle),
            PipelineStage(writer_bundle, input_template="Based on this:\\n{previous_output}"),
            PipelineStage(reviewer_bundle),
        ])
        # 或者嵌套别的 orchestrator:
        pipeline = PipelineOrchestrator([
            PipelineStage(triage_bundle),
            PipelineStage(ParallelOrchestrator([opt_bundle, pess_bundle])),
            PipelineStage(reviewer_bundle),
        ])
    """

    def __init__(self, stages: list[PipelineStage], name: str | None = None):
        if not stages:
            raise ValueError("Pipeline requires at least one stage")
        super().__init__(name=name)
        self.stages = stages

    async def run(self, ctx: Context) -> OrchestratorOutput:
        await self._emit_started(ctx, {"stage_count": len(self.stages)})

        agent_outputs: list[AgentOutput] = []
        sequence: list[str] = []
        previous_output = ctx.user_input  # 第一阶段的输入

        for i, stage in enumerate(self.stages):
            node_name = stage.runnable.name
            sequence.append(node_name)

            # 构造本阶段输入
            current_input = stage.input_template.format(
                previous_output=previous_output,
                user_input=ctx.user_input,
                **ctx.state,
            )

            # 派生隔离 sub_ctx;模块注入由节点(BundleRunnable)内部负责
            stage_ctx = ctx.fork()
            stage_ctx.messages = []
            stage_ctx.user_input = current_input

            # 执行
            output = await stage.runnable.run(stage_ctx)
            agent_outputs.append(output)

            # 保存到 state(如果配置了)
            if stage.save_to_state:
                ctx.state[stage.save_to_state] = output.content

            # 把节点输出作为下一阶段输入
            previous_output = output.content

            # emit handoff 到下一阶段
            if i + 1 < len(self.stages):
                await self._emit_handoff(
                    ctx,
                    from_agent=node_name,
                    to_agent=self.stages[i + 1].runnable.name,
                    reason="pipeline_next_stage",
                )

        result = OrchestratorOutput(
            orchestrator_type="pipeline",
            final_content=previous_output,
            agent_outputs=agent_outputs,
            sequence=sequence,
        )
        await self._emit_completed(ctx, {"sequence": sequence})
        return result

    async def run_stream(self, ctx: Context) -> AsyncIterator[OrchestratorStreamEvent]:
        await self._emit_started(ctx, {"stage_count": len(self.stages)})
        yield OrchestratorStreamEvent(type="start", orchestrator_type="pipeline", content=ctx.user_input)

        agent_outputs: list[AgentOutput] = []
        sequence: list[str] = []
        previous_output = ctx.user_input

        for i, stage in enumerate(self.stages):
            node_name = stage.runnable.name
            sequence.append(node_name)
            current_input = stage.input_template.format(
                previous_output=previous_output,
                user_input=ctx.user_input,
                **ctx.state,
            )
            stage_ctx = ctx.fork()
            stage_ctx.messages = []
            stage_ctx.user_input = current_input

            # 流式跑节点;消费它产出的 AgentStreamEvent,转译成 OrchestratorStreamEvent
            stage_output: AgentOutput | None = None
            async for agent_event in stage.runnable.run_stream(stage_ctx):
                yield OrchestratorStreamEvent(
                    type="agent_chunk",
                    orchestrator_type="pipeline",
                    agent_name=node_name,
                    content=agent_event.content,
                    agent_event=agent_event,
                )
                if agent_event.type == "done" and agent_event.output is not None:
                    stage_output = agent_event.output

            if stage_output is None:
                # 节点没产出 done 事件就被吞了 → 兜底空 output,避免后续崩
                stage_output = AgentOutput(agent_name=node_name, content="")
            agent_outputs.append(stage_output)

            if stage.save_to_state:
                ctx.state[stage.save_to_state] = stage_output.content
            previous_output = stage_output.content

            if i + 1 < len(self.stages):
                next_name = self.stages[i + 1].runnable.name
                await self._emit_handoff(ctx, from_agent=node_name, to_agent=next_name, reason="pipeline_next_stage")
                yield OrchestratorStreamEvent(
                    type="handoff",
                    orchestrator_type="pipeline",
                    from_agent=node_name,
                    to_agent=next_name,
                    reason="pipeline_next_stage",
                )

        result = OrchestratorOutput(
            orchestrator_type="pipeline",
            final_content=previous_output,
            agent_outputs=agent_outputs,
            sequence=sequence,
        )
        await self._emit_completed(ctx, {"sequence": sequence})
        yield OrchestratorStreamEvent(
            type="done",
            orchestrator_type="pipeline",
            content=result.final_content,
            output=result,
        )
