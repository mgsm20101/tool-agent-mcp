# Evaluation Results

Measured at commit `500d1988` on a clean tree with `qwen2.5-coder:3b` through local
Ollama — raw file [`results/eval_500d1988.json`](results/eval_500d1988.json).

| metric | value |
|---|---:|
| task completion (answer contains the expected fact) | **0.500** (4/8) |
| tool-call accuracy (exactly the expected tools) | **0.375** |
| mean iterations | 3.75 |
| runs stopped by the iteration guardrail | 3 |

| # | task | expected tools | tools used | stopped by | iters | completed |
|---:|---|---|---|---|---:|:---:|
| 1 | كم يساوي 23*17+5؟ | calculator | calculator | answer | 2 | yes |
| 2 | احسب 144 / 12 ثم أضف 88 | calculator | calculator, current_datetime, knowledge_search | max_iterations | 6 | no |
| 3 | كم يوم إجازة سنوية يستحق الموظف الجديد؟ | knowledge_search | knowledge_search, calculator | answer | 3 | no |
| 4 | كم يوم في الأسبوع يُسمح بالعمل عن بُعد؟ | knowledge_search | knowledge_search, current_datetime | max_iterations | 6 | no |
| 5 | ما حد المصروفات اليومية للسفر الداخلي؟ | knowledge_search | knowledge_search, calculator | answer | 3 | yes |
| 6 | ما هو تاريخ اليوم؟ | current_datetime | current_datetime | answer | 2 | yes |
| 7 | كم مدة فترة التجربة للموظف الجديد بالأيام؟ واحسب كم تساوي لو مُدّدت بالكامل. | knowledge_search, calculator | knowledge_search, calculator, current_datetime | max_iterations | 6 | no |
| 8 | What is 7 to the power of 3? | calculator | calculator | answer | 2 | yes |

**What failed, and how:**

* **Every single-tool calculator and date task completed in two iterations.**
* **Three runs hit the six-iteration guardrail** — including both multi-step tasks
  (divide-then-add, and look-up-then-calculate). The model picks a plausible first tool
  and then wanders into unrelated ones (`current_datetime` in an arithmetic task). The
  guardrail did its job: each ended with an explicit "could not finish" instead of a guess.
* **One confident wrong answer**: asked for new-employee annual leave, the agent searched
  the knowledge base and then answered 30 days; the policy says 21. Retrieval was not the
  failure — the answer ignored what was retrieved. This is the case a grounding check on
  the final answer would catch and a tool-use metric does not.
* **Tool-call accuracy is below completion** because the model often calls an extra,
  unneeded tool before answering correctly (the travel-expense task).

Eight tasks: one task is 12.5 points. The pattern (single-step fine, multi-step breaks) is
the finding; the rates are not a model ranking.

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

`src/agent/recovery.py` (called from `loop.py`) now recovers a call left in
`content`, narrowly: only JSON naming a
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
