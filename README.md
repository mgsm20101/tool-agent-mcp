# Tool Agent (MCP)

A multi-step tool-using agent with the ReAct loop written by hand — no agent framework.
The agent talks to its tools through a real MCP server (stdio), runs every call through
guardrails (iteration cap, argument validation, allowlist, prompt-injection defense),
records a structured trace of every step, and ships with an eval harness that measures
task completion and tool-call accuracy.

## Problem

Giving an LLM tools is easy; making the result *reliable* is the actual job. An agent
can loop forever, call tools with hallucinated arguments, or get steered by malicious
content inside a tool result. This project builds the loop from scratch so every one of
those failure modes has an explicit, testable defense — and then measures the agent
instead of demoing it.

## Architecture

```
demo / FastAPI ──▶ ReAct loop (think → act → observe)
                     │  guardrails: max-iterations · allowlist · arg validation
                     │              · repeat-call break · injection scan
                     │  memory: the conversation itself (tool results as messages)
                     ├─▶ LLM (local, OpenAI-compatible tool calling)
                     └─▶ MCP client ──(stdio)──▶ MCP server (mcp 2.x)
                                                   ├─ calculator        (safe AST eval)
                                                   ├─ knowledge_search  (policy KB)
                                                   └─ current_datetime
                     every event → structured Trace (type, tool, args, latency)

eval harness ──▶ task completion rate · tool-call accuracy · avg iterations · guardrail stops
```

- **The loop** (`src/agent/loop.py`) — one iteration per model turn: the model either
  answers (stop) or requests tool calls; calls pass through guardrails, execute over
  MCP, and the observations go back into the conversation. The messages list is the
  agent's working memory.
- **MCP server** (`src/mcp_server/server.py`) — three deliberately different tool shapes
  over stdio: pure computation with strict validation, retrieval, and ambient context.
  The calculator evaluates arithmetic by walking the AST — never `eval()`.
- **Guardrails** (`src/agent/guardrails.py`) — pure functions checked at the boundaries:
  before a call (allowlist, schema validation, identical-repeat break) and after it
  (injection patterns in tool output get neutralized and flagged in the trace).
- **Trace** (`src/schema.py`) — every model turn, tool call, observation, and guardrail
  intervention is a typed step with latency. The trace is returned by the API and printed
  by the demo; it is what makes the agent debuggable.

Design rationale is in [`docs/DESIGN.md`](docs/DESIGN.md).

## Run

```bash
python -m venv .venv && . .venv/Scripts/activate    # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env

ollama pull qwen2.5-coder:3b           # `gemma3:4b` cannot call tools at all —
                                       # Ollama rejects the request outright

python demo.py "كم يساوي 23*17+5؟"                       # one task + full trace
python demo.py "كم يوم إجازة سنوية أستحق بعد 6 سنوات؟"   # multi-tool task

python -m uvicorn src.api.main:app --port 8000            # POST /run -> answer + trace
```

> Windows notes: use `127.0.0.1` (not `localhost`) and run uvicorn via `python -m`.

## Eval

```bash
python -m src.eval.run_eval
```

Runs every task in `data/eval_set.jsonl` through the agent and scores:

- **task completion** — the final answer contains the expected result (alternatives
  accepted, e.g. "3" / "ثلاثة").
- **tool-call accuracy** — exactly the expected tools were used: missing ones mean the
  model guessed instead of acting; extra ones mean it wandered.
- **avg iterations** — efficiency; a simple task burning five turns signals confusion.
- **guardrail stops** — how often a run had to be stopped rather than finishing cleanly.

The harness refuses to run against a dirty git worktree (pass `--allow-dirty` to override)
and writes its raw output to `results/eval_<sha8>.json`: every per-task row, the aggregate
metrics, model name, Ollama host, timestamp, `source_commit_sha`, and `worktree_clean`. This
is the provenance trail — a results number without a matching file in `results/` should not
be trusted.

## Results

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

Engineering findings that only showed up when the eval ran end to end:

- `mcp>=1.2` resolves to mcp 2.x, where `FastMCP` was renamed to `MCPServer` — the
  requirements range now pins `mcp>=2.0` so it can't silently resolve to something
  untested.
- The 1.x client pattern (`stdio_client` + `ClientSession` driven by hand through an
  `AsyncExitStack`) hangs on mcp 2.x; `Client` owns the whole session lifecycle instead.
- `gemma3:4b` is rejected by Ollama for tool calls outright — advertised general
  capability is not the same as tool-calling support.
- `qwen2.5-coder:3b` does decide to call the right tool, but Ollama's chat template for
  this model leaves the call sitting as plain-text JSON in `content` instead of lifting
  it into the structured `tool_calls` field. `loop.py` has a narrow recovery path for
  exactly that: it accepts only JSON naming a tool the MCP server actually exposes, and
  everything recovered still goes through the same allowlist, schema and repeat
  guardrails as a natively parsed call.

Full narrative is in [`docs/results.md`](docs/results.md).

## Limitations

- Single 3B local model tested end to end; nothing here generalizes to a larger or
  hosted model.
- The gold eval set is small (8 tasks) and hand-written; it is a smoke test, not a
  statistically powered benchmark.
- Guardrails are unit-tested for the specific patterns they check, not adversarially
  red-teamed.
- Latency varies with whatever else is holding VRAM on a shared 4 GB GPU, so no timing
  number here is a stable benchmark.

## What this project does not prove

- That this agent design is production-ready at scale — it is a from-scratch
  demonstration of explicit guardrails, not a hardened deployment.
- That `qwen2.5-coder:3b` is a good general-purpose tool-calling model — only that it
  emits tool calls the serving layer doesn't always parse structurally, and that a
  narrow recovery path can still make use of them.
- That the eval numbers transfer to other task distributions, languages, or tool sets.
- Anything about frontier or hosted models — this project only exercises a small local
  model over Ollama.

## Reproduction

```bash
git clone <this-repo>
cd tool-agent-mcp
python -m venv .venv && . .venv/Scripts/activate    # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env

ollama serve
ollama pull qwen2.5-coder:3b

python -m src.eval.run_eval     # writes results/eval_<sha8>.json
```

Machine used for prior runs: Windows 11, 15.9 GB RAM, NVIDIA GTX 1050 Ti 4 GB — Ollama
offloads part of the model to that GPU; the exact share varies with whatever else is
using VRAM.

## Tech

| Layer | Choice | Why |
|-------|--------|-----|
| Agent loop | hand-written ReAct + native tool calling | full control over guardrails; nothing hidden |
| Tools | MCP server (mcp 2.x `MCPServer`, stdio) | the open protocol agent hosts actually speak |
| LLM | local `qwen2.5-coder:3b` via Ollama (OpenAI-compatible) | one of the two served models that can call tools at all; swappable for vLLM |
| Guardrails | pure functions at the boundaries | unit-testable without an LLM |
| Eval | task completion + tool-call accuracy + iteration stats | reliability measured, not claimed |

## Layout

```
data/
  knowledge.jsonl     small Arabic policy KB for knowledge_search
  eval_set.jsonl      gold tasks: expected tools + accepted answers
src/
  config.py            settings from env
  schema.py            TraceStep / AgentResult contracts
  mcp_server/server.py MCPServer: calculator, knowledge_search, current_datetime
  agent/
    mcp_client.py      ToolHost: spawn server, list/call tools over stdio
    guardrails.py      allowlist, arg validation, repeat break, injection scan
    loop.py            the ReAct loop
  eval/                metrics.py (pure), run_eval.py
  api/main.py          FastAPI POST /run
demo.py                one task, full colored trace
docs/                  DESIGN.md, results.md (generated)
tests/                 model-free unit tests (metrics, guardrails, recovery, tools, config)
results/               raw eval output, one file per measured run
```
