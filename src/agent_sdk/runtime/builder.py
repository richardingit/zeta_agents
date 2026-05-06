"""
AgentBuilder - 框架对外的核心 API。

设计理念:
    场景需求 → 配置 → AgentBuilder.build() → Agent 实例
    
用户用 Builder 描述"要什么样的 agent",而不是"怎么构造它"。
"""
from __future__ import annotations
from dataclasses import dataclass

from agent_sdk.core.agent import Agent, AgentConfig, SimpleAgent
from agent_sdk.core.context import Context
from agent_sdk.core.event_bus import EventBus
from agent_sdk.modules.llm.base import LLMConfig, LLMModule
from agent_sdk.modules.llm.factory import create_llm_from_config
from agent_sdk.modules.memory.base import MemoryModule
from agent_sdk.modules.skills.base import SkillsModule
from agent_sdk.modules.tools.base import ToolsModule


@dataclass
class AgentBundle:
    """一个 agent 加上它需要的所有模块上下文"""
    agent: Agent
    llm: LLMModule
    memory: MemoryModule | None = None
    skills: SkillsModule | None = None
    tools: ToolsModule | None = None
    event_bus: EventBus | None = None

    async def run(self, user_input: str, **state):
        ctx = Context(
            user_input=user_input,
            llm=self.llm,
            memory=self.memory,
            skills=self.skills,
            tools=self.tools,
            event_bus=self.event_bus,
            state=state,
        )
        return await self.agent.run(ctx)

    async def run_stream(self, user_input: str, **state):
        ctx = Context(
            user_input=user_input,
            llm=self.llm,
            memory=self.memory,
            skills=self.skills,
            tools=self.tools,
            event_bus=self.event_bus,
            state=state,
        )
        async for event in self.agent.run_stream(ctx):
            yield event


class AgentBuilder:
    """
    流式构造器 - 用法:
    
        agent = (AgentBuilder("researcher")
                .with_llm(llm)
                .with_system_prompt("You are a researcher.")
                .with_skills(skills_module, ["web_search", "summarize"])
                .with_memory(memory_store)
                .with_tools(tools_module, ["calculator"])
                .with_event_bus(bus)
                .build())
    """

    def __init__(self, name: str, description: str = ""):
        self._config = AgentConfig(name=name, description=description)
        self._llm: LLMModule | None = None
        self._memory: MemoryModule | None = None
        self._skills: SkillsModule | None = None
        self._tools: ToolsModule | None = None
        self._event_bus: EventBus | None = None
        self._agent_class: type[Agent] = SimpleAgent

    def with_llm(self, llm: LLMModule, model: str | None = None) -> "AgentBuilder":
        self._llm = llm
        if model:
            self._config.model = model
        return self

    def with_llm_config(self, config: LLMConfig, fn=None) -> "AgentBuilder":
        self._llm = create_llm_from_config(config, fn=fn)
        if config.model:
            self._config.model = config.model
        return self

    def with_system_prompt(self, prompt: str) -> "AgentBuilder":
        self._config.system_prompt = prompt
        return self

    def with_temperature(self, t: float) -> "AgentBuilder":
        self._config.temperature = t
        return self

    def with_max_iterations(self, n: int) -> "AgentBuilder":
        self._config.max_iterations = n
        return self

    def with_memory(self, memory: MemoryModule, recall_k: int = 5) -> "AgentBuilder":
        self._memory = memory
        self._config.use_memory = True
        self._config.memory_recall_k = recall_k
        return self

    def with_skills(
        self, skills: SkillsModule, enabled: list[str]
    ) -> "AgentBuilder":
        self._skills = skills
        self._config.enabled_skills = list(enabled)
        return self

    def with_tools(
        self, tools: ToolsModule, enabled: list[str]
    ) -> "AgentBuilder":
        self._tools = tools
        self._config.enabled_tools = list(enabled)
        return self

    def with_event_bus(self, bus: EventBus) -> "AgentBuilder":
        self._event_bus = bus
        return self

    def with_agent_class(self, cls: type[Agent]) -> "AgentBuilder":
        """高级用法:替换 Agent 实现类"""
        self._agent_class = cls
        return self

    def build(self) -> AgentBundle:
        if self._llm is None:
            raise ValueError(f"Agent '{self._config.name}' requires an LLM module")
        agent = self._agent_class(self._config)
        return AgentBundle(
            agent=agent,
            llm=self._llm,
            memory=self._memory,
            skills=self._skills,
            tools=self._tools,
            event_bus=self._event_bus,
        )
