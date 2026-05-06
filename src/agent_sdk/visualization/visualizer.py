"""
Workflow Visualizer - 把 Orchestrator 定义导出为可展示的流程图。

目标:
- 不侵入现有 orchestrator 运行逻辑
- 直接从配置对象生成节点/边
- 可导出 Mermaid 文本和独立 HTML 页面
"""
from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from agent_sdk.orchestrators.handoff import HandoffOrchestrator
from agent_sdk.orchestrators.parallel import ParallelOrchestrator
from agent_sdk.orchestrators.pipeline import PipelineOrchestrator
from agent_sdk.orchestrators.supervisor import SupervisorOrchestrator


@dataclass
class GraphNode:
    id: str
    label: str
    kind: str = "agent"
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str = ""
    kind: str = "default"


@dataclass
class FlowGraph:
    title: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_node(self, node_id: str, label: str, kind: str = "agent", **metadata):
        if not any(node.id == node_id for node in self.nodes):
            self.nodes.append(GraphNode(node_id, label, kind, metadata))

    def add_edge(self, source: str, target: str, label: str = "", kind: str = "default"):
        self.edges.append(GraphEdge(source, target, label, kind))


class OrchestratorVisualizer:
    """从现有 orchestrator 对象构建可视化图。"""

    @classmethod
    def build_graph(cls, orchestrator, title: str | None = None) -> FlowGraph:
        if isinstance(orchestrator, PipelineOrchestrator):
            return cls._build_pipeline(orchestrator, title)
        if isinstance(orchestrator, SupervisorOrchestrator):
            return cls._build_supervisor(orchestrator, title)
        if isinstance(orchestrator, HandoffOrchestrator):
            return cls._build_handoff(orchestrator, title)
        if isinstance(orchestrator, ParallelOrchestrator):
            return cls._build_parallel(orchestrator, title)
        raise TypeError(f"Unsupported orchestrator type: {type(orchestrator).__name__}")

    @staticmethod
    def _build_pipeline(orchestrator: PipelineOrchestrator, title: str | None) -> FlowGraph:
        graph = FlowGraph(title or "Pipeline Workflow", metadata={"type": "pipeline"})
        graph.add_node("start", "START", kind="terminal")
        graph.add_node("end", "END", kind="terminal")

        previous = "start"
        for index, stage in enumerate(orchestrator.stages, start=1):
            agent_name = stage.bundle.agent.name
            node_id = f"stage_{index}_{agent_name}"
            label = f"{index}. {agent_name}"
            if stage.save_to_state:
                label += f"\\n(save: {stage.save_to_state})"
            graph.add_node(node_id, label, kind="pipeline_stage")
            edge_label = ""
            if stage.input_template != "{previous_output}":
                edge_label = "templated input"
            graph.add_edge(previous, node_id, edge_label)
            previous = node_id

        graph.add_edge(previous, "end")
        return graph

    @staticmethod
    def _build_supervisor(orchestrator: SupervisorOrchestrator, title: str | None) -> FlowGraph:
        graph = FlowGraph(title or "Supervisor Workflow", metadata={"type": "supervisor"})
        graph.add_node("start", "START", kind="terminal")
        graph.add_node("end", "END", kind="terminal")

        supervisor_name = orchestrator.supervisor.agent.name
        graph.add_node("supervisor", f"Supervisor\\n{supervisor_name}", kind="supervisor")
        graph.add_edge("start", "supervisor", "user request")

        for worker_name, bundle in orchestrator.workers.items():
            worker_id = f"worker_{worker_name}"
            description = bundle.agent.config.description or "worker"
            graph.add_node(worker_id, f"{worker_name}\\n{description}", kind="worker")
            graph.add_edge("supervisor", worker_id, "delegate")
            graph.add_edge(worker_id, "supervisor", "result")

        graph.add_edge("supervisor", "end", "FINAL_ANSWER")
        return graph

    @staticmethod
    def _build_handoff(orchestrator: HandoffOrchestrator, title: str | None) -> FlowGraph:
        graph = FlowGraph(title or "Handoff Workflow", metadata={"type": "handoff"})
        graph.add_node("start", "START", kind="terminal")
        graph.add_node("end", "END", kind="terminal")

        for agent_name, node in orchestrator.nodes.items():
            label = agent_name
            if agent_name == orchestrator.entry_agent:
                label += "\\n(entry)"
            graph.add_node(f"agent_{agent_name}", label, kind="handoff_agent")

        graph.add_edge("start", f"agent_{orchestrator.entry_agent}", "entry")
        for agent_name, node in orchestrator.nodes.items():
            source = f"agent_{agent_name}"
            graph.add_edge(source, "end", "answer directly")
            for target_name in node.can_handoff_to:
                graph.add_edge(source, f"agent_{target_name}", "handoff", kind="handoff")

        return graph

    @staticmethod
    def _build_parallel(orchestrator: ParallelOrchestrator, title: str | None) -> FlowGraph:
        graph = FlowGraph(title or "Parallel Workflow", metadata={"type": "parallel"})
        graph.add_node("start", "START", kind="terminal")
        graph.add_node("aggregate", "Aggregator", kind="aggregator")
        graph.add_node("end", "END", kind="terminal")

        for index, bundle in enumerate(orchestrator.agents, start=1):
            agent_name = bundle.agent.name
            node_id = f"parallel_{index}_{agent_name}"
            graph.add_node(node_id, agent_name, kind="parallel_agent")
            graph.add_edge("start", node_id, "same input")
            graph.add_edge(node_id, "aggregate", "output")

        graph.add_edge("aggregate", "end", "merged result")
        return graph

    @staticmethod
    def to_mermaid(graph: FlowGraph) -> str:
        lines = ["flowchart TD"]
        for node in graph.nodes:
            lines.append(f'    {node.id}["{node.label}"]')
        for edge in graph.edges:
            if edge.label:
                lines.append(f"    {edge.source} -->|{edge.label}| {edge.target}")
            else:
                lines.append(f"    {edge.source} --> {edge.target}")
        return "\n".join(lines)

    @classmethod
    def export_html(cls, orchestrator, output_path: str | Path, title: str | None = None) -> Path:
        graph = cls.build_graph(orchestrator, title)
        mermaid = cls.to_mermaid(graph)
        page_title = graph.title

        html = f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{escape(page_title)}</title>
    <style>
      :root {{
        --bg: #f7f6f2;
        --card: #fffdf8;
        --ink: #1f2937;
        --muted: #6b7280;
        --line: #ded8c9;
        --accent: #1d4ed8;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, #ffffff 0, #f7f6f2 45%),
          linear-gradient(135deg, #f2efe6, #faf8f3 60%, #f4f1e8);
      }}
      .page {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 40px 24px 64px;
      }}
      .hero {{
        margin-bottom: 24px;
      }}
      h1 {{
        margin: 0 0 8px;
        font-size: clamp(32px, 5vw, 56px);
        line-height: 1;
        letter-spacing: -0.03em;
      }}
      p {{
        margin: 0;
        color: var(--muted);
        font-size: 16px;
      }}
      .panel {{
        background: color-mix(in srgb, var(--card) 90%, white);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: 0 18px 50px rgba(31, 41, 55, 0.08);
        overflow: hidden;
      }}
      .panel-head {{
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        padding: 18px 22px;
        border-bottom: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(247,246,242,0.9));
      }}
      .badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(29, 78, 216, 0.08);
        color: var(--accent);
        font-size: 13px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      }}
      .canvas {{
        padding: 20px;
        overflow: auto;
      }}
      .mermaid {{
        min-width: 720px;
      }}
      .source {{
        padding: 18px 22px 24px;
        border-top: 1px solid var(--line);
        background: rgba(244, 241, 232, 0.6);
      }}
      pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--muted);
        font-size: 13px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <h1>{escape(page_title)}</h1>
        <p>从 Orchestrator 配置自动生成的 Multi-Agent 流程图，可直接用于框架文档和调试展示。</p>
      </section>
      <section class="panel">
        <div class="panel-head">
          <strong>Workflow Diagram</strong>
          <span class="badge">{escape(graph.metadata.get("type", "workflow"))}</span>
        </div>
        <div class="canvas">
          <div class="mermaid">
{escape(mermaid)}
          </div>
        </div>
        <div class="source">
          <pre>{escape(mermaid)}</pre>
        </div>
      </section>
    </main>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{
        startOnLoad: true,
        theme: "base",
        themeVariables: {{
          primaryColor: "#fffaf0",
          primaryBorderColor: "#b9a77a",
          lineColor: "#8b7355",
          secondaryColor: "#eef2ff",
          tertiaryColor: "#f5f1e7",
          fontFamily: "Iowan Old Style, Palatino Linotype, Georgia, serif",
          primaryTextColor: "#1f2937",
        }},
        flowchart: {{
          curve: "basis",
          htmlLabels: true,
        }},
      }});
    </script>
  </body>
</html>
"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        return output
