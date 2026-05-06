from __future__ import annotations

import asyncio

from agent_sdk import AgentBuilder
from agent_sdk.core import Context
from agent_sdk.llm import MockLLM
from agent_sdk.orchestrators import PipelineOrchestrator, PipelineStage


async def main() -> None:
    researcher = (
        AgentBuilder("researcher", description="collects facts")
        .with_llm(MockLLM(["Research complete"]))
        .with_system_prompt("You are a researcher.")
        .build()
    )
    writer = (
        AgentBuilder("writer", description="writes drafts")
        .with_llm(MockLLM(["Draft complete"]))
        .with_system_prompt("You are a writer.")
        .build()
    )
    pipeline = PipelineOrchestrator([
        PipelineStage(researcher, save_to_state="research"),
        PipelineStage(writer, input_template="Write from:\n{previous_output}"),
    ])

    result = await pipeline.run(Context(user_input="Prepare a short report"))
    print(result.final_content)


if __name__ == "__main__":
    asyncio.run(main())
