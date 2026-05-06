"""
Pipeline Orchestrator - 流水线串行执行。

适用场景:
- 固定顺序的多步任务
- 每步输出作为下一步输入
- 例:researcher → writer → reviewer
"""
from __future__ import annotations
from collections.abc import AsyncIterator
from dataclasses import dataclass

from agent_sdk.core.context import Context
from agent_sdk.core.types import AgentOutput
from agent_sdk.runtime.builder import AgentBundle
from agent_sdk.orchestrators.base import Orchestrator, OrchestratorOutput, OrchestratorStreamEvent


@dataclass
class PipelineStage:
    """流水线的一个阶段"""
    bundle: AgentBundle
    # 上一阶段输出如何作为本阶段输入(可选,默认直接传)
    input_template: str = "{previous_output}"
    # 本阶段输出存到 ctx.state 的 key(可选)
    save_to_state: str | None = None


class PipelineOrchestrator(Orchestrator):
    """
    按顺序执行一系列 agent,每个 agent 看到上一个 agent 的输出。
    
    用法:
        pipeline = PipelineOrchestrator([
            PipelineStage(researcher_bundle),
            PipelineStage(writer_bundle, input_template="Based on this research:\\n{previous_output}\\n\\nWrite a report."),
            PipelineStage(reviewer_bundle),
        ])
        result = await pipeline.run(ctx)
    """

    def __init__(self, stages: list[PipelineStage]):
        if not stages:
            raise ValueError("Pipeline requires at least one stage")
        self.stages = stages

    async def run(self, ctx: Context) -> OrchestratorOutput:
        await self._emit_started(ctx, {"stage_count": len(self.stages)})

        agent_outputs: list[AgentOutput] = []
        sequence: list[str] = []
        previous_output = ctx.user_input  # 第一阶段的输入

        for i, stage in enumerate(self.stages):
            agent_name = stage.bundle.agent.name
            sequence.append(agent_name)

            # 构造本阶段输入
            current_input = stage.input_template.format(
                previous_output=previous_output,
                user_input=ctx.user_input,
                **ctx.state,
            )

            # 派生独立 ctx 给 agent 跑(避免消息历史污染),用 bundle 内模块覆盖
            stage_ctx = self._build_sub_ctx(ctx, stage.bundle)
            stage_ctx.user_input = current_input

            # 执行
            output = await stage.bundle.agent.run(stage_ctx)
            agent_outputs.append(output)

            # 保存到 state(如果配置了)
            if stage.save_to_state:
                ctx.state[stage.save_to_state] = output.content

            # 把 agent 输出作为下一阶段输入
            previous_output = output.content

            # emit handoff 到下一阶段
            if i + 1 < len(self.stages):
                await self._emit_handoff(
                    ctx,
                    from_agent=agent_name,
                    to_agent=self.stages[i + 1].bundle.agent.name,
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
            agent_name = stage.bundle.agent.name
            sequence.append(agent_name)
            current_input = stage.input_template.format(
                previous_output=previous_output,
                user_input=ctx.user_input,
                **ctx.state,
            )
            stage_ctx = self._build_sub_ctx(ctx, stage.bundle)
            stage_ctx.user_input = current_input

            output, events = await self._stream_agent_bundle(stage.bundle, stage_ctx)
            for event in events:
                event.orchestrator_type = "pipeline"
                yield event
            agent_outputs.append(output)

            if stage.save_to_state:
                ctx.state[stage.save_to_state] = output.content
            previous_output = output.content

            if i + 1 < len(self.stages):
                next_agent = self.stages[i + 1].bundle.agent.name
                await self._emit_handoff(ctx, from_agent=agent_name, to_agent=next_agent, reason="pipeline_next_stage")
                yield OrchestratorStreamEvent(
                    type="handoff",
                    orchestrator_type="pipeline",
                    from_agent=agent_name,
                    to_agent=next_agent,
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
