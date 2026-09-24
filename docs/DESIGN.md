# Design note — a tool agent you can trust

Why this is built the way it is, and the failure modes each piece exists to stop.

## Why hand-write the loop

Frameworks (LangGraph, Semantic Kernel) are fine, but they hide exactly the parts an
interviewer — or an incident review — asks about: where the iteration cap lives, what
happens to a bad tool call, who sanitizes tool output. Writing think → act → observe by
hand keeps every defense visible and testable. The loop is ~120 lines; the framework
would not have saved much, and it would have cost the understanding.

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
