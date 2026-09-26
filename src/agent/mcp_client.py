"""MCP client: spawn the tool server as a subprocess and talk to it over stdio.

ToolHost wraps the session lifecycle and exposes exactly two things the loop needs:
the tool definitions (converted to the OpenAI function format the LLM consumes) and
call_tool(). Used as an async context manager so the subprocess always shuts down.
"""

from __future__ import annotations

import sys

from mcp import Client, StdioServerParameters

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "src.mcp_server.server"],
)


class ToolHost:
    """Owns the MCP session (mcp 2.x `Client`); the loop only sees tools and call_tool()."""

    def __init__(self) -> None:
        self._client: Client | None = None
        self.tools: list[dict] = []  # OpenAI function-format definitions

    async def __aenter__(self) -> "ToolHost":
        self._client = Client(SERVER_PARAMS)
        await self._client.__aenter__()

        listed = await self._client.list_tools()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    # mcp 2.x name (was `inputSchema` in 1.x).
                    "parameters": t.input_schema,
                },
            }
            for t in listed.tools
        ]
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.__aexit__(*(exc or (None, None, None)))

    def tool_schema(self, name: str) -> dict | None:
        for t in self.tools:
            if t["function"]["name"] == name:
                return t["function"]["parameters"]
        return None

    async def call_tool(self, name: str, args: dict) -> str:
        assert self._client is not None, "ToolHost used outside its context"
        result = await self._client.call_tool(name, args)
        parts = [c.text for c in result.content if getattr(c, "text", None)]
        return "\n".join(parts) if parts else "(empty result)"
