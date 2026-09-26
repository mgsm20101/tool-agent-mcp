"""MCP server exposing the agent's tools over stdio.

Three deliberately different tool shapes:
- calculator        pure computation, strict input validation
- knowledge_search  retrieval over a small policy KB (keyword overlap scoring)
- current_datetime  ambient context the model cannot know on its own

The agent spawns this file as a subprocess and talks MCP over stdio — the same
transport MCP hosts such as desktop assistants and IDEs use.

Run standalone (for inspection):  python -m src.mcp_server.server
"""

from __future__ import annotations

import ast
import json
import operator
from datetime import datetime
from pathlib import Path

# mcp 2.x name for what 1.x called FastMCP.
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("agent-tools")

KNOWLEDGE_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge.jsonl"

# --- calculator -----------------------------------------------------------------

# Safe arithmetic evaluator: walk the AST and allow only number ops. Never eval().
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (numbers and + - * / // % ** only).

    Example: "23*17+5" -> "396"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"error: {e}"
    # Render integers without a trailing .0 so answers read naturally.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# --- knowledge_search -----------------------------------------------------------


def _load_kb() -> list[dict]:
    return [
        json.loads(line)
        for line in KNOWLEDGE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@mcp.tool()
def knowledge_search(query: str) -> str:
    """Search the internal company-policy knowledge base (Arabic). Returns the
    most relevant policy entries for the query."""
    kb = _load_kb()
    q_tokens = set(query.split())
    scored = []
    for entry in kb:
        text_tokens = set(entry["title"].split()) | set(entry["text"].split())
        overlap = len(q_tokens & text_tokens)
        if overlap:
            scored.append((overlap, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return "لا توجد نتائج مطابقة في قاعدة المعرفة."
    top = [f"[{e['title']}] {e['text']}" for _, e in scored[:2]]
    return "\n\n".join(top)


# --- current_datetime ------------------------------------------------------------


@mcp.tool()
def current_datetime() -> str:
    """Return the current local date and time (ISO format) with the weekday."""
    now = datetime.now()
    weekdays_ar = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
    return f"{now.isoformat(timespec='seconds')} ({weekdays_ar[now.weekday()]})"


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
