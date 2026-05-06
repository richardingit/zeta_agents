# Changelog

## 0.1.1

- Added flexible LLM entrypoints for bring-your-own model backends
- Added `FunctionLLM` for wiring custom sync or async inference functions
- Added streaming support with `LLMChunk` and `LLMModule.stream()`
- Added `FunctionLLM.stream_fn` for custom token streaming
- Added streaming support in `OpenAICompatibleProvider` with native or fallback paths
- Added `OpenAICompatibleProvider` for OpenAI-style cloud or local gateways
- Added `LLMConfig`, `create_llm()`, and `create_llm_from_config()` for config-driven provider creation
- Added `AgentBuilder.with_llm_config()` for builder-level provider injection
- Added `LiteLLMProvider` as an optional adapter entrypoint
- Added tests covering function-backed LLMs, OpenAI-compatible parsing, and config-based provider wiring

## 0.1.0

- Converted the prototype into an installable `agent_sdk` package
- Added core runtime, builder, orchestrators, persistence, and visualization packages
- Added minimal pluggable modules for llm, memory, skills, and tools
- Added smoke tests for builder, memory, pipeline, supervisor, handoff, parallel, and visualizer
- Added starter examples and packaging metadata
