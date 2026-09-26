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
    """Shaped like an OpenAI `ChatCompletionMessageToolCall` so the loop below
    does not need to know a call was recovered rather than parsed."""

    id: str
    function: _Function
    type: str = "function"


def _unwrap_schema_echo(value):
    """`{"type": "string", "value": "23*17+5"}` -> `"23*17+5"`.

    Small models under a tool schema sometimes return the *schema* for an
    argument instead of the argument. Observed from `qwen2.5-coder:3b` the
    moment a system prompt insisted on tool use: it produced
    `{"expression": {"type": "string", "value": "23*17+5"}}`. The value is
    there; it is wearing its own type declaration.
    """
    if isinstance(value, dict) and "type" in value and len(value) <= 3:
        for key in ("value", "default", "example"):
            if key in value:
                return value[key]
    return value


def tool_calls_from_content(content: str | None, host: ToolHost) -> list[_RecoveredCall]:
    """Recover a tool call the serving layer left sitting in `content`.

    `/api/show` reports `capabilities: ["completion", "tools", "insert"]` for
    `qwen2.5-coder:3b`, and the model does decide to call tools correctly — it
    emitted `{"name": "calculator", "arguments": {"expression": "23*17+5"}}`
    for the obvious arithmetic question. Ollama's template for this model
    simply did not lift that out of the text into the structured `tool_calls`
    field, so the loop saw an assistant answer with no calls and stopped. Every
    one of the eight eval tasks came back `MISS tools=[] iters=1`.

    Advertised tool support is not the same as working tool support, and the
    boundary where that distinction lives is worth defending rather than
    assuming. The model's decision was right; only the transport lost it.

    Deliberately narrow: only JSON carrying a `name` that matches a tool the
    MCP server actually exposes is accepted, so ordinary prose that happens to
    contain braces is not mistaken for a call. Everything recovered here still
    goes through the same allowlist, argument-schema and repeat guardrails as a
    natively parsed call — recovery gets a call into the pipeline, it does not
    get it past the checks.
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
    """Every balanced top-level {...} in `text`, parsed, skipping what will not.

    Scanning for balanced braces rather than regex-matching because a tool call
    nests its own arguments object, and the outer braces are the ones that
    matter. Fenced code blocks are handled by this too: the fence markers sit
    outside the braces.
    """
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
