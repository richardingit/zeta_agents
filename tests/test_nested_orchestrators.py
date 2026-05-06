"""Phase 1: orchestrator 嵌套能力的回归测试。

当前 Pipeline 已支持以 Runnable / Orchestrator 作为节点。这里覆盖:
- Pipeline 内部嵌套另一个 Pipeline(双层串行)
- Pipeline 内部嵌套 Parallel(串行 → 多支并行 → 串行)
- Pipeline 流式路径下,内层 agent 的逐 token 事件能透传到上层
"""
from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import FunctionLLM, MockLLM
from agent_sdk.orchestrators import (
    ParallelOrchestrator,
    PipelineOrchestrator,
    PipelineStage,
)


def _bundle(name: str, reply: str):
    return (
        AgentBuilder(name)
        .with_llm(MockLLM([reply]))
        .with_system_prompt(f"You are {name}.")
        .build()
    )


async def test_pipeline_can_nest_another_pipeline():
    """Pipeline 套 Pipeline:外层第二阶段是一个完整的 Pipeline"""
    inner = PipelineOrchestrator([
        PipelineStage(_bundle("research", "facts gathered")),
        PipelineStage(_bundle("draft", "draft ready")),
    ])
    outer = PipelineOrchestrator([
        PipelineStage(_bundle("intake", "topic clarified")),
        PipelineStage(inner),  # 嵌套:第二阶段是另一个 pipeline
        PipelineStage(_bundle("polish", "polished output")),
    ])

    output = await outer.run(Context(user_input="Write something"))

    # 外层 sequence 看到的是 3 个节点:intake / inner / polish
    assert output.sequence == ["intake", "PipelineOrchestrator", "polish"]
    # 最终内容来自最后一个节点
    assert output.final_content == "polished output"


async def test_pipeline_can_nest_parallel():
    """Pipeline 套 Parallel:第二阶段并行跑两个 agent,结果再传给第三阶段"""
    parallel = ParallelOrchestrator([
        _bundle("optimist", "looks great"),
        _bundle("pessimist", "looks risky"),
    ])
    pipeline = PipelineOrchestrator([
        PipelineStage(_bundle("intake", "request received")),
        PipelineStage(parallel, save_to_state="opinions"),
        PipelineStage(_bundle("decision", "final call")),
    ])

    ctx = Context(user_input="Should we ship?")
    output = await pipeline.run(ctx)

    assert output.sequence == ["intake", "ParallelOrchestrator", "decision"]
    assert output.final_content == "final call"
    # save_to_state 应该拿到 parallel 的聚合输出
    assert "optimist" in ctx.state["opinions"]
    assert "pessimist" in ctx.state["opinions"]


async def test_orchestrator_explicit_name_appears_in_sequence():
    """显式给 orchestrator 传 name,嵌套 sequence 里能看到这个业务名"""
    inner = PipelineOrchestrator(
        [PipelineStage(_bundle("research", "facts"))],
        name="research_flow",
    )
    outer = PipelineOrchestrator([
        PipelineStage(_bundle("intake", "ok")),
        PipelineStage(inner),
    ])
    output = await outer.run(Context(user_input="go"))
    # 嵌套节点显示用户传的业务名,而不是默认类名
    assert output.sequence == ["intake", "research_flow"]


async def test_pipeline_stream_passes_through_inner_agent_events():
    """嵌套 pipeline 的流式调用应该把内层 agent 的逐 token 事件透传出去"""
    async def stream_tokens(messages, **kwargs):
        for token in ["hel", "lo"]:
            yield token

    streaming_bundle = (
        AgentBuilder("streamer")
        .with_llm(FunctionLLM(lambda **k: "hello", stream_fn=stream_tokens))
        .with_system_prompt("You are streamer.")
        .build()
    )
    inner = PipelineOrchestrator([PipelineStage(streaming_bundle)])
    outer = PipelineOrchestrator([
        PipelineStage(_bundle("intake", "ready")),
        PipelineStage(inner),
    ])

    events = [event async for event in outer.run_stream(Context(user_input="go"))]

    # 把 type=text 的内层 agent_event 收集起来,确认 token 流真的穿过两层 pipeline
    text_chunks = [
        ev.agent_event.content
        for ev in events
        if ev.agent_event is not None and ev.agent_event.type == "text"
    ]
    assert "hel" in text_chunks
    assert "lo" in text_chunks
    # 整个流程仍以 done 收尾
    assert events[-1].type == "done"
    assert events[-1].output is not None
    assert events[-1].output.sequence == ["intake", "PipelineOrchestrator"]
