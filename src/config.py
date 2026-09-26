"""All configuration in one place, loaded from the environment.

Import `settings` everywhere; nothing else reads os.environ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _allowlist() -> list[str]:
    raw = os.getenv("TOOL_ALLOWLIST", "").strip()
    return [t.strip() for t in raw.split(",") if t.strip()]


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    # Must be served by local Ollama AND able to call tools (why: docs/DESIGN.md).
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5-coder:3b")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "300"))

    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "6"))
    # Empty list means "allow whatever the MCP server exposes".
    tool_allowlist: list[str] = field(default_factory=_allowlist)


settings = Settings()
