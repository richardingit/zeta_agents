"""
Supervisor Orchestrator - 主管 agent 调度子 agent。

适用场景:
- 任务路径不固定,需要动态决策
- 主管 agent 根据当前状态选择"接下来谁来做"
- 子 agent 完成后回到主管,主管决定继续还是结束

工作流:
    1. Supervisor 收到任务
    2. Supervisor 决定派给哪个 worker(或自己回答)
    3. Worker 执行,结果返回 Supervisor
    4. Supervisor 决定继续派工 / 结束
    5. 循环直到 Supervisor 输出 FINAL_ANSWER 或达到 max_rounds
"""
from __future__ import annotations
import re

from agent_sdk.core.context import Context
from agent_sdk.core.types import AgentOutput
from agent_sdk.runtime.builder import AgentBundle
from agent_sdk.orchestrators.base import Orchestrator, OrchestratorOutput


# Supervisor 用的特殊"终止"标记,在 prompt 里告诉它怎么写
FINAL_ANSWER_PREFIX = "FINAL_ANSWER:"


class SupervisorOrchestrator(Orchestrator):
    """
    Supervisor 模式:一个 supervisor agent 调度多个 worker。
    
    Supervisor 通过特殊工具 "delegate_to_worker" 派工给 worker。
    Supervisor 通过输出 "FINAL_ANSWER: ..." 来结束流程。
    
    用法:
        supervisor = SupervisorOrchestrator(
            supervisor_bundle=supervisor_bundle,
            workers={
                "researcher": researcher_bundle,
                "writer": writer_bundle,
            },
            max_rounds=10,
        )
        result = await supervisor.run(ctx)
    """

    def __init__(
        self,
        supervisor_bundle: AgentBundle,
        workers: dict[str, AgentBundle],
        max_rounds: int = 10,
    ):
        if not workers:
            raise ValueError("Supervisor requires at least one worker")
        self.supervisor = supervisor_bundle
        self.workers = workers
        self.max_rounds = max_rounds

    async def run(self, ctx: Context) -> OrchestratorOutput:
        await self._emit_started(ctx, {
            "supervisor": self.supervisor.agent.name,
            "workers": list(self.workers.keys()),
        })

        agent_outputs: list[AgentOutput] = []
        sequence: list[str] = []
        worker_results: dict[str, str] = {}  # 累积所有 worker 产出
        final_content = ""

        # 给 supervisor 注入"delegate"工具的特殊处理:
        # 我们把 worker 列表注入到 supervisor 的 prompt,
        # supervisor 输出 "DELEGATE: <worker_name>: <task>" 我们解析后调用。
        # 这样不需要 supervisor 真的会调用工具,纯文本协议即可。

        round_num = 0
        current_request = ctx.user_input

        while round_num < self.max_rounds:
            round_num += 1
            sequence.append(self.supervisor.agent.name)

            # === Supervisor 决策 ===
            sup_ctx = self._build_sub_ctx(ctx, self.supervisor)
            sup_ctx.user_input = self._build_supervisor_prompt(current_request, worker_results)
            sup_output = await self.supervisor.agent.run(sup_ctx)
            agent_outputs.append(sup_output)

            # 解析 supervisor 输出
            decision = self._parse_decision(sup_output.content)

            if decision["type"] == "final":
                final_content = decision["content"]
                break

            elif decision["type"] == "delegate":
                worker_name = decision["worker"]
                task = decision["task"]

                if worker_name not in self.workers:
                    # 工作者不存在,把错误反馈给 supervisor
                    worker_results[f"_error_round{round_num}"] = (
                        f"No such worker '{worker_name}'. "
                        f"Available: {list(self.workers.keys())}"
                    )
                    continue

                # === Worker 执行 ===
                await self._emit_handoff(
                    ctx,
                    from_agent=self.supervisor.agent.name,
                    to_agent=worker_name,
                    reason=f"delegated: {task[:50]}",
                )
                sequence.append(worker_name)

                worker_bundle = self.workers[worker_name]
                w_ctx = self._build_sub_ctx(ctx, worker_bundle)
                w_ctx.user_input = task
                w_output = await worker_bundle.agent.run(w_ctx)
                agent_outputs.append(w_output)

                worker_results[f"{worker_name}_round{round_num}"] = w_output.content

                # 通过 memory(如有共享)传递结果
                if ctx.memory:
                    await ctx.memory.remember(
                        f"[{worker_name}] {w_output.content}",
                        metadata={"agent": worker_name, "round": round_num},
                    )
            else:
                # 解析失败,supervisor 直接给了答案
                final_content = sup_output.content
                break

        if not final_content and worker_results:
            # 达到 max_rounds 还没 final,把最后一个 worker 输出作为结果
            final_content = list(worker_results.values())[-1]

        result = OrchestratorOutput(
            orchestrator_type="supervisor",
            final_content=final_content,
            agent_outputs=agent_outputs,
            sequence=sequence,
            metadata={"rounds": round_num},
        )
        await self._emit_completed(ctx, {"rounds": round_num, "sequence": sequence})
        return result

    def _build_supervisor_prompt(
        self, original_request: str, worker_results: dict[str, str]
    ) -> str:
        """构造给 supervisor 的输入,包含原始任务和已完成的子任务结果"""
        worker_list = "\n".join(
            f"- {name}: {bundle.agent.config.description or 'no description'}"
            for name, bundle in self.workers.items()
        )

        prompt = f"""Original task: {original_request}

Available workers:
{worker_list}

You can either:
1. Delegate to a worker by responding EXACTLY in this format on a single line:
   DELEGATE: <worker_name>: <specific task for the worker>
2. Provide the final answer when you have enough info, by responding:
   FINAL_ANSWER: <your final answer>
"""
        if worker_results:
            prompt += "\n\nWork completed so far:\n"
            for k, v in worker_results.items():
                prompt += f"\n[{k}]\n{v}\n"
            prompt += "\nDecide your next action (DELEGATE or FINAL_ANSWER):"
        else:
            prompt += "\nDecide your first action (DELEGATE or FINAL_ANSWER):"
        return prompt

    def _parse_decision(self, content: str) -> dict:
        """解析 supervisor 的输出"""
        content = content.strip()

        # 检查 FINAL_ANSWER
        if FINAL_ANSWER_PREFIX in content:
            idx = content.find(FINAL_ANSWER_PREFIX)
            answer = content[idx + len(FINAL_ANSWER_PREFIX):].strip()
            return {"type": "final", "content": answer}

        # 检查 DELEGATE
        m = re.search(
            r"DELEGATE:\s*([a-zA-Z0-9_]+)\s*:\s*(.+)",
            content,
            re.DOTALL,
        )
        if m:
            return {
                "type": "delegate",
                "worker": m.group(1).strip(),
                "task": m.group(2).strip(),
            }

        return {"type": "unknown", "content": content}
