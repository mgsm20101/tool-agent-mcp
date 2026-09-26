# Design note — a tool agent you can trust

Why this is built the way it is, and the failure modes each piece exists to stop.

## Why hand-write the loop

Frameworks (LangGraph, Semantic Kernel) are fine, but they hide exactly the parts an
interviewer — or an incident review — asks about: where the iteration cap lives, what
happens to a bad tool call, who sanitizes tool output. Writing think → act → observe by
hand keeps every defense visible and testable. `run_agent` is about 110 lines; the
framework would not have saved much, and it would have cost the understanding.

## The failure modes and their defenses

| Failure mode | Defense | Where |
|---|---|---|
| Infinite loop / wandering | hard iteration cap; the run ends with `stopped_by="max_iterations"` instead of hanging | `loop.py` |
| Stuck repeating the same call | identical (tool, args) twice in a row is refused — same input would produce the same observation | `guardrails.check_repeat` |
| Hallucinated arguments | every call validated against the tool's own JSON schema (required keys, types, no extras) before execution | `guardrails.check_args` |
| Calling a tool we don't trust | explicit allowlist, enforced client-side even if the server exposes more | `guardrails.check_allowlist` |
| Prompt injection via tool output | observations are scanned for instruction patterns (AR + EN); hits are quoted as untrusted data, flagged in the trace, and the system prompt tells the model tool results are data, not orders | `guardrails.scan_observation` |
| Code injection into the calculator | arithmetic is evaluated by walking the AST with an operator whitelist — `eval()` never runs | `mcp_server.server` |
| Blocked call corrupting the conversation | a blocked call still gets a `role="tool"` reply (`[blocked] reason`), so the message sequence stays valid and the model can adapt | `loop.py` |

The principle behind all of them: **the prompt asks, the code enforces.** Telling the
model "don't loop" is a wish; `max_iterations` is a guarantee.

## Why MCP, and why stdio

MCP is the open protocol agent hosts (desktop assistants, IDEs, and others) actually speak, so the
tools built here are reusable outside this repo. stdio transport keeps the demo
self-contained — the agent spawns the server as a subprocess, no ports, no daemon. The
client (`ToolHost`) discovers tools at runtime from the server and converts their schemas
to the OpenAI function format, so adding a tool to the server requires zero agent changes.

## Memory: the conversation is the scratchpad

Session memory is simply the messages list — each tool result is appended as a
`role="tool"` message, so every model turn sees everything learned so far. That is enough
for multi-step tasks within a session. Cross-session memory (persisting facts between
runs) is a deliberate non-goal of this MVP; the Capstone picks it up.

## Evaluating an agent

Generation quality metrics don't fit agents; what matters is whether the job got done
and whether the route was sane:

- **task completion** — the final answer contains the expected result (alternatives
  accepted, e.g. "3" / "ثلاثة").
- **tool-call accuracy** — exactly the expected tools were used: missing ones mean the
  model guessed instead of acting; extra ones mean it wandered.
- **avg iterations** — efficiency; a simple task burning five turns signals confusion.
- **guardrail stops** — how often a run had to be stopped rather than finishing cleanly.

The gold set is small and bilingual, including one task that genuinely needs two tools
in sequence (look up the probation policy, then compute the extension).

## Why the code looks like this

History that explains otherwise odd-looking lines. The code keeps a one-line
comment; the reasoning lives here.

### Default model: `qwen2.5-coder:3b` (`src/config.py`)

The default has to be a model the local Ollama server actually serves **and** one
that can call tools. Both were checked, not assumed:

- `qwen2.5:7b-instruct` (the earlier default) lived in a second model store the
  running server could not see, so every run fell through to a slow error path. The
  old "~120 s per inference, no GPU" note was measuring that path, not the machine.
- `gemma3:4b` is served, but Ollama rejects the request outright:
  `registry.ollama.ai/library/gemma3:4b does not support tools`. A capable general
  model is not automatically a tool-calling one.
- `qwen2.5-coder:3b` is served, and `/api/show` reports
  `capabilities: ["completion", "tools", "insert"]`.

### `ToolHost` uses mcp 2.x `Client` (`src/agent/mcp_client.py`)

The 1.x version of this file drove the transport, the session and `initialize()` by
hand through an `AsyncExitStack`:

```python
read, write = await stack.enter_async_context(stdio_client(PARAMS))
session     = await stack.enter_async_context(ClientSession(read, write))
await session.initialize()
```

On mcp 2.x that hung in `__aenter__` and then failed on teardown with "attempted to
exit cancel scope in a different task": `stdio_client` opens an anyio task group
that an exit stack does not necessarily unwind in the task that entered it. `Client`
keeps that lifecycle inside one context manager, so the wrapper got shorter. The
tool-schema field is `input_schema` because 2.x moved its model fields to snake_case.

### `MCPServer`, and `mcp>=2.0` (`src/mcp_server/server.py`, `requirements.txt`)

The old `mcp>=1.2` range resolved to 2.x, where `FastMCP` was renamed to
`MCPServer`; the server crashed on import and the client only saw
`Connection closed`. The code was migrated (a rename, not a port) and the range now
says `mcp>=2.0` so it cannot silently resolve to something untested.

### Tool-call recovery (`src/agent/recovery.py`)

`qwen2.5-coder:3b` does decide to call tools correctly: asked for `23*17+5` it
emitted `{"name": "calculator", "arguments": {"expression": "23*17+5"}}`. Ollama's
chat template for this model left that as plain text in `content` instead of lifting
it into the structured `tool_calls` field, so a loop that reads only `tool_calls`
saw an answer with no calls and stopped. Every eval task came back
`MISS tools=[] iters=1`. Advertised tool support is not working tool support; the
model's decision was right, only the transport lost it.

`tool_calls_from_content` recovers such calls, deliberately narrowly:

- only JSON carrying a `name` that matches a tool the MCP server actually exposes is
  accepted, so prose that happens to contain braces is not mistaken for a call;
- everything recovered goes through the same allowlist, argument-schema and repeat
  guardrails as a natively parsed call. Recovery gets a call into the pipeline; it
  does not get it past the checks;
- `_json_objects` scans for balanced braces rather than using a regex, because a tool
  call nests its own arguments object and the outer braces are the ones that matter
  (fenced code blocks work too: the fences sit outside the braces);
- `_unwrap_schema_echo` handles `{"expression": {"type": "string", "value": "23*17+5"}}`.
  The model produced that the moment the system prompt insisted on tool use: the
  value is there, wearing its own type declaration.
