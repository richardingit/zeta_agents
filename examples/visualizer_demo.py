from __future__ import annotations

from pathlib import Path

from agent_sdk import AgentBuilder
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import PipelineOrchestrator, PipelineStage
from agent_sdk.visualization import OrchestratorVisualizer


def main() -> None:
    bundle = (
        AgentBuilder("researcher")
        .with_llm(MockLLM(["done"]))
        .with_system_prompt("You are a researcher.")
        .build()
    )
    pipeline = PipelineOrchestrator([PipelineStage(bundle)])
    output = Path("workflow_ui.html")
    OrchestratorVisualizer.export_html(pipeline, output, title="Research Pipeline")
    print(f"Saved workflow UI to: {output.resolve()}")


if __name__ == "__main__":
    main()
