"""Recover tool calls the serving layer left as plain-text JSON in `content`.

Everything returned here still goes through the loop's guardrails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.agent.mcp_client import ToolHost


@dataclass(frozen=True)
class _Function:
    name: str
    arguments: str


@dataclass(frozen=True)
class _RecoveredCall:
    """Shaped like an OpenAI `ChatCompletionMessageToolCall` so the loop does
    not need to know a call was recovered rather than parsed."""

    id: str
    function: _Function
    type: str = "function"


def _unwrap_schema_echo(value):
    """`{"type": "string", "value": "23*17+5"}` -> `"23*17+5"` (a small-model schema echo)."""
    if isinstance(value, dict) and "type" in value and len(value) <= 3:
        for key in ("value", "default", "example"):
            if key in value:
                return value[key]
    return value


def tool_calls_from_content(content: str | None, host: ToolHost) -> list[_RecoveredCall]:
    """Recover tool calls left as JSON text in `content` instead of `tool_calls`.

    Narrow on purpose: only JSON whose `name` is a tool the MCP server exposes is
    accepted. Recovered calls still pass the same guardrails as native ones.
    """
    if not content:
        return []

    known = {t["function"]["name"] for t in host.tools}
    calls: list[_RecoveredCall] = []

    for candidate in _json_objects(content):
        name = candidate.get("name") or candidate.get("tool") or candidate.get("function")
        if not isinstance(name, str) or name not in known:
            continue
        args = candidate.get("arguments", candidate.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        args = {k: _unwrap_schema_echo(v) for k, v in args.items()}
        calls.append(
            _RecoveredCall(
                id=f"recovered-{len(calls)}",
                function=_Function(name=name, arguments=json.dumps(args, ensure_ascii=False)),
            )
        )
    return calls


def _json_objects(text: str):
    """Every balanced top-level {...} in `text`, parsed; unparseable ones skipped."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    yield parsed
                start = None
