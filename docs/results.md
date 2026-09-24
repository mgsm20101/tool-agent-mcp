# Evaluation Results

<!-- RESULTS: filled from results/eval_<sha8>.json after the measured run -->

Pending a fresh, provenance-tracked run of `python -m src.eval.run_eval`. That run writes
`results/eval_<sha8>.json` — every per-task row, the aggregate metrics, model name, Ollama
host, timestamp, `source_commit_sha`, and `worktree_clean` — and this file gets the
aggregate table and per-task breakdown filled in from it. No score is written here without
a matching file in `results/`.

## Environment

| | |
|---|---|
| Model | `qwen2.5-coder:3b` via Ollama (OpenAI-compatible endpoint) |
| Machine | Windows 11 · 15.9 GB RAM · GTX 1050 Ti 4 GB |
| GPU | Used. Ollama offloads into the card; the share varies with whatever else on the desktop holds VRAM. |
| Guardrails | allowlist · argument-schema check · repeat-call check · observation scan |
| Max iterations | 6 |

## Findings from earlier runs, still true

None of these were visible by reading the code — they only showed up once the eval
actually ran end to end.

**1. `mcp>=1.2` installed a version the code could not import.** The range
resolved to mcp 2.x, where `FastMCP` was renamed to `MCPServer`. The server
crashed on import and the client reported only `Connection closed`. Fixed by
migrating (a rename) rather than pinning backwards, and the range now says
`mcp>=2.0` so it cannot resolve to something untested again.

**2. The 1.x client pattern hangs on 2.x.** Driving `stdio_client`,
`ClientSession` and `initialize()` by hand through an `AsyncExitStack` hung in
`__aenter__` and then failed teardown with *"attempted to exit cancel scope in a
different task"* — `stdio_client` opens an anyio task group that an exit stack
does not necessarily unwind in the task that entered it. mcp 2.x's `Client`
owns that whole lifecycle, so `mcp_client.py` got shorter.

**3. `gemma3:4b` cannot call tools at all.** Ollama rejects the request:
`registry.ollama.ai/library/gemma3:4b does not support tools`. This project had
recorded gemma3's tool-calling as an untested assumption; it is now tested, and
false.

**4. Advertised tool support is not working tool support.** `qwen2.5-coder:3b`
advertises tool support — `/api/show` reports
`capabilities: ["completion", "tools", "insert"]` — and the model does decide
to call the right tool. Asked *كم يساوي 23\*17+5؟* it emitted exactly the
right thing:

```json
{"name": "calculator", "arguments": {"expression": "23*17+5"}}
```

as **plain text in `content`**. Ollama's template for this model never lifted it
into the structured `tool_calls` field, so a loop that only reads `tool_calls`
sees an assistant answer with no calls and stops on the first iteration.

`loop.py` now recovers a call left in `content`, narrowly: only JSON naming a
tool the MCP server actually exposes is accepted, and everything recovered
goes through the same allowlist, schema and repeat guardrails as a natively
parsed call. Recovery gets a call into the pipeline; it does not get it past
the checks. It also unwraps a schema echo —
`{"type": "string", "value": "23*17+5"}` instead of `"23*17+5"` — which this
model produces the moment a system prompt insists on tool use.

## What this does not establish

* **A small gold set. No confidence intervals.** One task moves any rate by a
  large margin. Treat aggregate rates as a direction, not a precise measurement.
* **One model.** Nothing here generalises to a larger or hosted model; a
  frontier model would likely not need the recovery path at all.
* **Latency is not a benchmark.** Per-step times are in the traces, but the GPU
  share on a shared 4 GB desktop varies run to run, so no timing number here is
  a stable measurement.
* **The guardrails are unit-tested, not adversarially tested.** They are known
  to be reachable in this eval set; that is not the same as complete coverage.

## Reproduce

```bash
pip install -r requirements.txt
ollama serve
ollama pull qwen2.5-coder:3b
python -m src.eval.run_eval
```
