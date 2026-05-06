"""
Handoff Orchestrator - Agent 之间自由转交控制权(类似 OpenAI Swarm)。

核心机制:
- 每个 agent 知道自己可以转交给哪些 agent
- agent 通过输出 "HANDOFF_TO: <agent_name>: <reason>" 触发转交
- 没转交就视为本轮完成

适用场景:
- 客服分流(general → billing/tech/sales)
- 协作场景中 agent 自主决定下一棒交给谁
"""
from __future__ import annotations
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import re

from agent_sdk.core.context import Context
from agent_sdk.core.types import AgentOutput
from agent_sdk.runtime.builder import AgentBundle
from agent_sdk.orchestrators.base import Orchestrator, OrchestratorOutput, OrchestratorStreamEvent


HANDOFF_MARKER = "HANDOFF_TO:"


@dataclass
class HandoffNode:
    """Handoff 网络中的一个节点"""
    bundle: AgentBundle
    can_handoff_to: list[str] = field(default_factory=list)  # 允许转交的目标 agent 名


class HandoffOrchestrator(Orchestrator):
    """
    Agent 之间自由转交控制权。
    
    用法:
        orchestrator = HandoffOrchestrator(
            entry_agent="triage",
            nodes={
                "triage":  HandoffNode(triage_bundle, can_handoff_to=["billing", "tech"]),
                "billing": HandoffNode(billing_bundle, can_handoff_to=["triage"]),
                "tech":    HandoffNode(tech_bundle, can_handoff_to=["triage"]),
            },
            max_handoffs=5,
        )
    """

    def __init__(
        self,
        entry_agent: str,
        nodes: dict[str, HandoffNode],
        max_handoffs: int = 5,
    ):
        if entry_agent not in nodes:
            raise ValueError(f"entry_agent '{entry_agent}' not in nodes")
        self.entry_agent = entry_agent
        self.nodes = nodes
        self.max_handoffs = max_handoffs

    async def run(self, ctx: Context) -> OrchestratorOutput:
        await self._emit_started(ctx, {
            "entry": self.entry_agent,
            "available_nodes": list(self.nodes.keys()),
        })

        agent_outputs: list[AgentOutput] = []
        sequence: list[str] = []
        current_agent = self.entry_agent
        current_input = ctx.user_input
        handoffs = 0
        final_content = ""

        while True:
            sequence.append(current_agent)
            node = self.nodes[current_agent]

            # 给当前 agent 注入 handoff 说明
            agent_ctx = self._build_sub_ctx(ctx, node.bundle)
            agent_ctx.user_input = self._inject_handoff_instructions(
                current_input, node.can_handoff_to
            )

            output = await node.bundle.agent.run(agent_ctx)
            agent_outputs.append(output)

            # 解析是否要转交
            target, reason, clean_content = self._parse_handoff(output.content)

            if target is None:
                # 没转交 → 流程结束
                final_content = clean_content or output.content
                break

            if target not in node.can_handoff_to:
                # 转交目标不被允许 → 当前 agent 输出作为结果
                final_content = (
                    f"[Invalid handoff target '{target}' from {current_agent}]\n"
                    + (clean_content or output.content)
                )
                break

            if target not in self.nodes:
                final_content = (
                    f"[Handoff target '{target}' not found]\n"
                    + (clean_content or output.content)
                )
                break

            handoffs += 1
            if handoffs > self.max_handoffs:
                final_content = (
                    f"[Max handoffs ({self.max_handoffs}) reached]\n"
                    + (clean_content or output.content)
                )
                break

            await self._emit_handoff(
                ctx,
                from_agent=current_agent,
                to_agent=target,
                reason=reason,
            )

            # 把当前 agent 的输出(去掉 handoff 标记)作为下一个 agent 的输入
            current_input = (
                f"Handed off from {current_agent}. "
                f"Reason: {reason}\n\n"
                f"Context: {clean_content or output.content}\n\n"
                f"Original user request: {ctx.user_input}"
            )
            current_agent = target

        result = OrchestratorOutput(
            orchestrator_type="handoff",
            final_content=final_content,
            agent_outputs=agent_outputs,
            sequence=sequence,
            metadata={"handoff_count": handoffs},
        )
        await self._emit_completed(ctx, {"sequence": sequence, "handoffs": handoffs})
        return result

    async def run_stream(self, ctx: Context) -> AsyncIterator[OrchestratorStreamEvent]:
        await self._emit_started(ctx, {
            "entry": self.entry_agent,
            "available_nodes": list(self.nodes.keys()),
        })
        yield OrchestratorStreamEvent(type="start", orchestrator_type="handoff", content=ctx.user_input)

        agent_outputs: list[AgentOutput] = []
        sequence: list[str] = []
        current_agent = self.entry_agent
        current_input = ctx.user_input
        handoffs = 0
        final_content = ""

        while True:
            sequence.append(current_agent)
            node = self.nodes[current_agent]
            agent_ctx = self._build_sub_ctx(ctx, node.bundle)
            agent_ctx.user_input = self._inject_handoff_instructions(current_input, node.can_handoff_to)
            output, events = await self._stream_agent_bundle(node.bundle, agent_ctx)
            for event in events:
                event.orchestrator_type = "handoff"
                yield event
            agent_outputs.append(output)
            target, reason, clean_content = self._parse_handoff(output.content)
            if target is None:
                final_content = clean_content or output.content
                break
            if target not in node.can_handoff_to:
                final_content = f"[Invalid handoff target '{target}' from {current_agent}]\n" + (clean_content or output.content)
                break
            if target not in self.nodes:
                final_content = f"[Handoff target '{target}' not found]\n" + (clean_content or output.content)
                break
            handoffs += 1
            if handoffs > self.max_handoffs:
                final_content = f"[Max handoffs ({self.max_handoffs}) reached]\n" + (clean_content or output.content)
                break
            await self._emit_handoff(ctx, from_agent=current_agent, to_agent=target, reason=reason)
            yield OrchestratorStreamEvent(
                type="handoff",
                orchestrator_type="handoff",
                from_agent=current_agent,
                to_agent=target,
                reason=reason,
            )
            current_input = (
                f"Handed off from {current_agent}. "
                f"Reason: {reason}\n\n"
                f"Context: {clean_content or output.content}\n\n"
                f"Original user request: {ctx.user_input}"
            )
            current_agent = target

        result = OrchestratorOutput(
            orchestrator_type="handoff",
            final_content=final_content,
            agent_outputs=agent_outputs,
            sequence=sequence,
            metadata={"handoff_count": handoffs},
        )
        await self._emit_completed(ctx, {"sequence": sequence, "handoffs": handoffs})
        yield OrchestratorStreamEvent(
            type="done",
            orchestrator_type="handoff",
            content=result.final_content,
            output=result,
        )

    def _inject_handoff_instructions(
        self, user_input: str, can_handoff_to: list[str]
    ) -> str:
        if not can_handoff_to:
            return user_input
        targets = ", ".join(can_handoff_to)
        return f"""{user_input}

---
You may handle this yourself, OR if more appropriate, hand off to another agent.
Available agents to hand off to: {targets}

To hand off, end your response with EXACTLY this line (and nothing after):
HANDOFF_TO: <agent_name>: <reason>

If you can answer directly, just answer without the HANDOFF_TO line.
"""

    @staticmethod
    def _parse_handoff(content: str) -> tuple[str | None, str, str]:
        """
        解析输出,返回 (handoff_target, reason, content_without_marker)
        如果没有 handoff 标记,target 为 None。
        """
        m = re.search(
            rf"{HANDOFF_MARKER}\s*([a-zA-Z0-9_]+)\s*:\s*(.+?)(?:\n|$)",
            content,
        )
        if not m:
            return None, "", content
        target = m.group(1).strip()
        reason = m.group(2).strip()
        clean = (content[: m.start()] + content[m.end():]).strip()
        return target, reason, clean
