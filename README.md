# Agent SDK

A modular multi-agent Python SDK with pluggable memory, skills, tools, and orchestrators.

## What's New

### 0.1.1

- Added flexible LLM entrypoints so the SDK is not tied to a single provider stack
- Added `FunctionLLM` for plugging in custom inference functions directly
- Added `OpenAICompatibleProvider` for OpenAI-style hosted or local gateways
- Added `LLMConfig`, `create_llm()`, and `AgentBuilder.with_llm_config()`
- Added optional `LiteLLMProvider` adapter support
- Added tests for the new LLM entry layer

## Status

This repository is being converted from a framework prototype into an installable SDK.

Current stable focus for `0.1.x`:
- core agent runtime
- builder
- orchestrators
- basic memory / skills / tools modules
- visualization
- flexible llm entrypoints

## Installation

```bash
pip install -e .[dev]
```

For a regular project install:

```bash
pip install .
```

## Quickstart

```python
from agent_sdk import AgentBuilder
from agent_sdk.llm import MockLLM
from agent_sdk.memory import InMemoryStore
from agent_sdk.skills import SkillsModule

llm = MockLLM(responses=["Hello! How can I help?"])
memory = InMemoryStore()
skills = SkillsModule()

agent = (
    AgentBuilder("assistant")
    .with_llm(llm)
    .with_system_prompt("You are a helpful assistant.")
    .with_memory(memory)
    .with_skills(skills, [])
    .build()
)

result = await agent.run("Say hello")
print(result.content)
```

## Flexible LLM Entry

You can bring your own LLM backend through a unified interface.

### 1. Function-based entry

```python
from agent_sdk import AgentBuilder
from agent_sdk.llm import FunctionLLM

async def my_llm(messages, tools=None, model=None, temperature=0.7, **kwargs):
    return "Custom backend response"

agent = (
    AgentBuilder("assistant")
    .with_llm(FunctionLLM(my_llm))
    .build()
)
```

### 2. OpenAI-compatible gateway

```python
from agent_sdk.llm import OpenAICompatibleProvider

llm = OpenAICompatibleProvider(
    model="gpt-4.1-mini",
    api_key="YOUR_KEY",
    base_url="https://api.openai.com/v1",
)
```

This also works for local or self-hosted gateways such as vLLM if they expose an OpenAI-compatible API.

### 3. Config-driven entry

```python
from agent_sdk import AgentBuilder
from agent_sdk.llm import LLMConfig

agent = (
    AgentBuilder("assistant")
    .with_llm_config(
        LLMConfig(
            provider="openai_compatible",
            model="gpt-4.1-mini",
            api_key="YOUR_KEY",
            base_url="https://api.openai.com/v1",
        )
    )
    .build()
)
```

## Core Concepts

### AgentBuilder

The main entrypoint for composing agents from pluggable modules.

### Modules

- `llm`: model providers such as `MockLLM`
- `memory`: long-term memory implementations such as `InMemoryStore`
- `skills`: file-driven capability packs
- `tools`: runtime tool registry

### Orchestrators

- `PipelineOrchestrator`: fixed sequential workflows
- `SupervisorOrchestrator`: one supervisor delegates to workers
- `HandoffOrchestrator`: agents can pass control to each other
- `ParallelOrchestrator`: multiple agents run concurrently and aggregate outputs

### Visualization

Use `OrchestratorVisualizer` to export Mermaid or HTML workflow diagrams.

## Examples

The repository includes starter examples under [`examples/`](/Users/chenyanfeng/Documents/GitHub/agent-projects/examples):
- [single_agent.py](/Users/chenyanfeng/Documents/GitHub/agent-projects/examples/single_agent.py)
- [pipeline_demo.py](/Users/chenyanfeng/Documents/GitHub/agent-projects/examples/pipeline_demo.py)
- [supervisor_demo.py](/Users/chenyanfeng/Documents/GitHub/agent-projects/examples/supervisor_demo.py)
- [handoff_demo.py](/Users/chenyanfeng/Documents/GitHub/agent-projects/examples/handoff_demo.py)
- [parallel_demo.py](/Users/chenyanfeng/Documents/GitHub/agent-projects/examples/parallel_demo.py)
- [visualizer_demo.py](/Users/chenyanfeng/Documents/GitHub/agent-projects/examples/visualizer_demo.py)

Run an example with:

```bash
source .venv/bin/activate
python examples/single_agent.py
```

## Verify On Another Machine

Once the repository is pushed to GitHub, you can validate it from another computer with:

```bash
git clone <your-repo-url>
cd agent-projects
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
python examples/single_agent.py
python examples/visualizer_demo.py
```

You should see:
- all tests passing
- `single_agent.py` printing a mock assistant reply
- `visualizer_demo.py` generating `workflow_ui.html`

## Next Steps

- publish to GitHub and test from another machine
- add layered / composite memory
- add skill metadata and memory policies
- evolve memory into layered memory
