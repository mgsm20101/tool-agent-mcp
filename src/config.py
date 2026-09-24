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
    # Default to a model the local Ollama server actually serves AND that can
    # call tools. Two separate facts, both checked rather than assumed:
    #
    #   qwen2.5:7b-instruct  — the previous default. Lives in a second model
    #     store the running server cannot see, so every run fell through to a
    #     slow error path. That path is what the old "~120s per inference, no
    #     GPU" note in docs/results.md was really measuring; it was never a
    #     limit of this machine.
    #   gemma3:4b            — served, and rejected the request outright:
    #     "registry.ollama.ai/library/gemma3:4b does not support tools". A
    #     capable general model is not automatically a tool-calling one, and
    #     this project is nothing without tool calls.
    #   qwen2.5-coder:3b     — served, and /api/show reports
    #     capabilities: ["completion", "tools", "insert"].
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5-coder:3b")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "300"))

    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "6"))
    # Empty list means "allow whatever the MCP server exposes".
    tool_allowlist: list[str] = field(default_factory=_allowlist)


settings = Settings()
