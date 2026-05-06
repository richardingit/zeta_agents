"""Agent 基类 - 框架最核心的抽象"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent_sdk.core.types import Message, Role, ToolCall, AgentOutput
from agent_sdk.core.event_bus import EventType

if TYPE_CHECKING:
    from agent_sdk.core.context import Context


@dataclass
class AgentConfig:
    """Agent 的配置 - 决定它'是谁'和'有什么能力'"""
    name: str
    description: str = ""
    system_prompt: str = "You are a helpful assistant."
    model: str | None = None             # 覆盖默认模型
    temperature: float = 0.7
    max_iterations: int = 10             # 工具调用循环上限

    # 该 Agent 启用哪些 skill (skill 名称列表)
    enabled_skills: list[str] = field(default_factory=list)

    # 该 Agent 启用哪些原生工具 (工具名称列表)
    enabled_tools: list[str] = field(default_factory=list)

    # 是否启用记忆
    use_memory: bool = False
    memory_recall_k: int = 5             # 召回多少条历史记忆


class Agent(ABC):
    """
    Agent 基类。
    具体的 Agent 子类只需实现 build_system_prompt 和 (可选) post_process。
    核心的工具循环、模块协作逻辑由基类完成。
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name

    async def run(self, ctx: "Context") -> AgentOutput:
        """主执行入口 - ReAct 循环"""
        # emit: agent.started
        if ctx.event_bus:
            await ctx.event_bus.emit_quick(
                EventType.AGENT_STARTED,
                session_id=ctx.session_id,
                agent_name=self.name,
                user_input=ctx.user_input,
            )

        try:
            # 1. 准备阶段:从模块拉取上下文
            await self._prepare(ctx)

            # 2. 工具调用循环
            iteration = 0
            all_tool_calls: list[ToolCall] = []

            while iteration < self.config.max_iterations:
                iteration += 1

                # emit: llm.call.started
                if ctx.event_bus:
                    await ctx.event_bus.emit_quick(
                        EventType.LLM_CALL_STARTED,
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        iteration=iteration,
                    )

                # 调用 LLM
                assert ctx.llm is not None, "LLM module is required"
                response = await ctx.llm.complete(
                    messages=ctx.messages,
                    tools=self._collect_tools(ctx),
                    model=self.config.model,
                    temperature=self.config.temperature,
                )

                # emit: llm.call.completed
                if ctx.event_bus:
                    await ctx.event_bus.emit_quick(
                        EventType.LLM_CALL_COMPLETED,
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        iteration=iteration,
                        model=response.model,
                        usage=response.usage,
                        has_tool_calls=bool(response.tool_calls),
                    )

                # 把 assistant 消息加入对话
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )
                ctx.add_message(assistant_msg)

                # 没有工具调用 → 结束
                if not response.tool_calls:
                    break

                # 执行所有工具调用
                for tc in response.tool_calls:
                    all_tool_calls.append(tc)

                    if ctx.event_bus:
                        await ctx.event_bus.emit_quick(
                            EventType.TOOL_CALL_STARTED,
                            session_id=ctx.session_id,
                            agent_name=self.name,
                            tool_name=tc.name,
                            arguments=tc.arguments,
                        )

                    tool_result = await self._execute_tool(ctx, tc)

                    if ctx.event_bus:
                        await ctx.event_bus.emit_quick(
                            EventType.TOOL_CALL_COMPLETED,
                            session_id=ctx.session_id,
                            agent_name=self.name,
                            tool_name=tc.name,
                            result_preview=str(tool_result)[:200],
                        )

                    ctx.add_message(Message(
                        role=Role.TOOL,
                        content=str(tool_result),
                        tool_call_id=tc.id,
                        name=tc.name,
                    ))

            # 3. 收尾:写入记忆 + 后处理
            await self._finalize(ctx)

            final_content = ctx.messages[-1].content if ctx.messages else ""
            output = AgentOutput(
                agent_name=self.name,
                content=final_content,
                messages=ctx.messages,
                tool_calls_made=all_tool_calls,
                iterations=iteration,
            )

            # emit: agent.completed
            if ctx.event_bus:
                await ctx.event_bus.emit_quick(
                    EventType.AGENT_COMPLETED,
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    iterations=iteration,
                    tool_calls_count=len(all_tool_calls),
                )

            return output

        except Exception as e:
            # emit: agent.failed
            if ctx.event_bus:
                await ctx.event_bus.emit_quick(
                    EventType.AGENT_FAILED,
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    error=str(e),
                )
            raise

    async def _prepare(self, ctx: "Context") -> None:
        """运行前的准备:组装 system prompt + 召回记忆"""
        system_parts = [await self.build_system_prompt(ctx)]

        # 如果有 skills 模块,注入选中的 skills
        if ctx.skills and self.config.enabled_skills:
            skill_prompts = await ctx.skills.build_prompts(
                self.config.enabled_skills, ctx
            )
            if skill_prompts:
                system_parts.append("# Available Skills\n" + skill_prompts)

        # 如果有 memory 模块且启用,召回相关记忆
        if ctx.memory and self.config.use_memory:
            memories = await ctx.memory.recall(
                ctx.user_input, k=self.config.memory_recall_k
            )
            if memories:
                mem_text = "\n".join(f"- {m.content}" for m in memories)
                system_parts.append(f"# Relevant Memories\n{mem_text}")

        # 拼装 system message
        system_msg = Message(role=Role.SYSTEM, content="\n\n".join(system_parts))

        # 如果 messages 为空,初始化对话;否则替换 system
        if not ctx.messages or ctx.messages[0].role != Role.SYSTEM:
            ctx.messages.insert(0, system_msg)
        else:
            ctx.messages[0] = system_msg

        # 加入用户输入(如果还没加)
        if ctx.user_input and not any(
            m.role == Role.USER and m.content == ctx.user_input
            for m in ctx.messages
        ):
            ctx.add_message(Message(role=Role.USER, content=ctx.user_input))

    async def _finalize(self, ctx: "Context") -> None:
        """运行后:写入记忆"""
        if ctx.memory and self.config.use_memory:
            # 把这一轮交互摘要写入记忆
            last_user = next(
                (m.content for m in reversed(ctx.messages) if m.role == Role.USER),
                "",
            )
            last_assistant = next(
                (m.content for m in reversed(ctx.messages) if m.role == Role.ASSISTANT),
                "",
            )
            if last_user and last_assistant:
                await ctx.memory.remember(
                    f"User: {last_user}\nAssistant: {last_assistant}",
                    metadata={"agent": self.name},
                )

    def _collect_tools(self, ctx: "Context") -> list:
        """收集本 Agent 可用的所有工具(原生工具 + skills 携带的工具)"""
        tools = []
        if ctx.tools:
            for tname in self.config.enabled_tools:
                t = ctx.tools.get(tname)
                if t:
                    tools.append(t)
        if ctx.skills:
            for sname in self.config.enabled_skills:
                tools.extend(ctx.skills.get_tools(sname))
        return tools

    async def _execute_tool(self, ctx: "Context", tc: ToolCall):
        """执行单个工具调用"""
        # 优先在 tools 模块查找
        if ctx.tools:
            t = ctx.tools.get(tc.name)
            if t:
                try:
                    return await t.invoke(**tc.arguments)
                except Exception as e:
                    return f"[Tool Error] {e}"
        # 再在 skills 携带的工具里查找
        if ctx.skills:
            for sname in self.config.enabled_skills:
                for t in ctx.skills.get_tools(sname):
                    if t.name == tc.name:
                        try:
                            return await t.invoke(**tc.arguments)
                        except Exception as e:
                            return f"[Tool Error] {e}"
        return f"[Tool Not Found] {tc.name}"

    @abstractmethod
    async def build_system_prompt(self, ctx: "Context") -> str:
        """子类实现:构建 system prompt 主体"""
        ...


class SimpleAgent(Agent):
    """最常用的 Agent 实现 - 直接用配置里的 system_prompt"""

    async def build_system_prompt(self, ctx: "Context") -> str:
        return self.config.system_prompt
