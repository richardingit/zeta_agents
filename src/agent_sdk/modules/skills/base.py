"""Skills module contracts and a minimal filesystem-backed implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import importlib.util


@dataclass
class SkillDefinition:
    name: str
    prompt: str = ""
    tools: list = field(default_factory=list)


class SkillsModule:
    def __init__(self):
        self._skills: dict[str, SkillDefinition] = {}

    def load_dir(self, directory: str | Path) -> None:
        base = Path(directory)
        if not base.exists():
            return
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            prompt_path = skill_dir / "SKILL.md"
            prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
            tools = []
            tools_path = skill_dir / "tools.py"
            if tools_path.exists():
                spec = importlib.util.spec_from_file_location(f"{skill_dir.name}_tools", tools_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    tools = list(getattr(module, "TOOLS", []))
            self._skills[skill_dir.name] = SkillDefinition(skill_dir.name, prompt, tools)

    async def build_prompts(self, enabled: list[str], ctx=None) -> str:
        parts = []
        for name in enabled:
            skill = self._skills.get(name)
            if skill and skill.prompt:
                parts.append(f"## {name}\n{skill.prompt}")
        return "\n\n".join(parts)

    def get_tools(self, skill_name: str) -> list:
        skill = self._skills.get(skill_name)
        return list(skill.tools) if skill else []

