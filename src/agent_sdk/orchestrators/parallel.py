"""
Parallel Orchestrator - 并行执行多个 agent + 聚合结果。

适用场景:
- 多角度分析(乐观派 vs 悲观派 vs 中立派)
- 并行检索(多个数据源同时查)
- 投票/共识(多个 agent 独立判断后投票)
"""
from __future__ import annotations
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Callable, Awaitable
import asyncio

from agent_sdk.core.context import Context
from agent_sdk.core.types import AgentOutput
from agent_sdk.runtime.builder import AgentBundle
from agent_sdk.orchestrators.base import Orchestrator, OrchestratorOutput, OrchestratorStreamEvent


# 聚合函数签名:接收所有 agent 输出,返回最终内容
Aggregator = Callable[[list[AgentOutput]], Awaitable[str]] | Callable[[list[AgentOutput]], str]


async def default_aggregator(outputs: list[AgentOutput]) -> str:
    """默认聚合:把所有 agent 的输出拼接,带名字前缀"""
    parts = []
    for o in outputs:
        parts.append(f"## {o.agent_name}\n\n{o.content}")
    return "\n\n---\n\n".join(parts)


class ParallelOrchestrator(Orchestrator):
    """
    并行执行所有 agent,然后用 aggregator 聚合结果。
    
    用法:
        parallel = ParallelOrchestrator(
            agents=[optimist_bundle, pessimist_bundle, neutral_bundle],
            aggregator=my_voting_aggregator,
        )
    """

    def __init__(
        self,
        agents: list[AgentBundle],
        aggregator: Aggregator | None = None,
    ):
        if not agents:
            raise ValueError("Parallel requires at least one agent")
        self.agents = agents
        self.aggregator = aggregator or default_aggregator

    async def run(self, ctx: Context) -> OrchestratorOutput:
        await self._emit_started(ctx, {
            "agents": [b.agent.name for b in self.agents],
        })

        # 给每个 agent 派生独立 ctx
        async def run_one(bundle: AgentBundle) -> AgentOutput:
            sub_ctx = self._build_sub_ctx(ctx, bundle)
            return await bundle.agent.run(sub_ctx)

        # 并行执行
        outputs = await asyncio.gather(*[run_one(b) for b in self.agents])

        # 聚合
        agg_result = self.aggregator(outputs)
        if asyncio.iscoroutine(agg_result):
            final_content = await agg_result
        else:
            final_content = agg_result

        sequence = [o.agent_name for o in outputs]
        result = OrchestratorOutput(
            orchestrator_type="parallel",
            final_content=final_content,
            agent_outputs=list(outputs),
            sequence=sequence,
        )
        await self._emit_completed(ctx, {"sequence": sequence})
        return result

    async def run_stream(self, ctx: Context) -> AsyncIterator[OrchestratorStreamEvent]:
        await self._emit_started(ctx, {"agents": [b.agent.name for b in self.agents]})
        yield OrchestratorStreamEvent(type="start", orchestrator_type="parallel", content=ctx.user_input)

        queue: asyncio.Queue = asyncio.Queue()

        async def run_one(bundle: AgentBundle):
            sub_ctx = self._build_sub_ctx(ctx, bundle)
            final_output = None
            async for event in bundle.agent.run_stream(sub_ctx):
                await queue.put(
                    OrchestratorStreamEvent(
                        type="agent_chunk",
                        orchestrator_type="parallel",
                        agent_name=bundle.agent.name,
                        content=event.content,
                        agent_event=event,
                    )
                )
                if event.type == "done":
                    final_output = event.output
            await queue.put(("done_agent", bundle.agent.name, final_output))

        tasks = [asyncio.create_task(run_one(bundle)) for bundle in self.agents]
        done_agents = 0
        outputs: dict[str, AgentOutput] = {}

        while done_agents < len(self.agents):
            item = await queue.get()
            if isinstance(item, tuple) and item[0] == "done_agent":
                _, agent_name, output = item
                done_agents += 1
                if output is not None:
                    outputs[agent_name] = output
                continue
            yield item

        await asyncio.gather(*tasks)
        ordered_outputs = [outputs[b.agent.name] for b in self.agents if b.agent.name in outputs]
        agg_result = self.aggregator(ordered_outputs)
        final_content = await agg_result if asyncio.iscoroutine(agg_result) else agg_result
        sequence = [o.agent_name for o in ordered_outputs]
        result = OrchestratorOutput(
            orchestrator_type="parallel",
            final_content=final_content,
            agent_outputs=ordered_outputs,
            sequence=sequence,
        )
        await self._emit_completed(ctx, {"sequence": sequence})
        yield OrchestratorStreamEvent(
            type="done",
            orchestrator_type="parallel",
            content=result.final_content,
            output=result,
        )
