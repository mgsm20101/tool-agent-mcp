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
    """Owns the MCP session; the agent loop only sees tool defs and call_tool().

    Uses mcp 2.x's `Client`, which takes the server parameters and owns the
    transport, the session and `initialize()` itself. The 1.x version of this
    file drove those three by hand through an `AsyncExitStack`:

        read, write = await stack.enter_async_context(stdio_client(PARAMS))
        session     = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

    That stopped working on 2.x — it hung in `__aenter__` and then failed on
    teardown with "attempted to exit cancel scope in a different task", because
    `stdio_client` opens an anyio task group that an exit stack does not
    necessarily unwind in the task that entered it. `Client` keeps that
    lifecycle inside one context manager, which is why it is the supported
    entry point now and why this wrapper got shorter rather than longer.
    """

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
                    # mcp 2.x renamed this from `inputSchema` to snake_case
                    # along with the rest of the model fields.
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
