from agent_sdk import AgentBuilder
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import PipelineOrchestrator, PipelineStage
from agent_sdk.visualization import OrchestratorVisualizer


def test_visualizer_generates_mermaid_for_pipeline(tmp_path):
    llm = MockLLM(["done"])
    bundle = AgentBuilder("researcher").with_llm(llm).build()
    pipeline = PipelineOrchestrator([PipelineStage(bundle)])

    graph = OrchestratorVisualizer.build_graph(pipeline, title="Test Flow")
    mermaid = OrchestratorVisualizer.to_mermaid(graph)
    output = tmp_path / "workflow.html"
    OrchestratorVisualizer.export_html(pipeline, output, title="Test Flow")

    assert "flowchart TD" in mermaid
    assert "researcher" in mermaid
    assert output.exists()
