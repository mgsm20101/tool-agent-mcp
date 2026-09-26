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

## Structure

### Entry points

| Command | Reads | Writes |
|---|---|---|
| `python -m src.eval.run_eval` — **the measured run** | `data/eval_set.jsonl`, `.env`, git `HEAD` + status | `results/eval_<sha8>.json` (+ a console table) |
| `python demo.py "<task>"` — one task | `.env` | console only: the trace and the answer |
| `python -m uvicorn src.api.main:app` — `POST /run` | `.env` | the HTTP response (answer + trace); not part of the measured eval |

All three spawn `src/mcp_server/server.py` as a subprocess (it reads `data/knowledge.jsonl`)
and call the model on Ollama at `OLLAMA_HOST`.

### Task flow

```
task  (eval row / demo.py argument / POST /run body)
 └▶ src/agent/loop.py:run_agent                  one iteration per model turn, capped at MAX_ITERATIONS
     ├▶ Ollama /v1 chat.completions               the model answers, or asks for tool calls
     ├▶ src/agent/recovery.py:tool_calls_from_content   only when tool_calls came back empty
     ├▶ src/agent/guardrails.py:check_allowlist → check_args → check_repeat
     ├▶ src/agent/mcp_client.py:ToolHost.call_tool ⇄ (stdio) src/mcp_server/server.py
     ├▶ src/agent/guardrails.py:scan_observation  tool output → observation back into the messages
     └▶ src/schema.py:AgentResult                 answer + trace + iterations + stopped_by
          ├▶ src/eval/run_eval.py:score → write_results  → results/eval_<sha8>.json
          ├▶ demo.py:main                               → console
          └▶ src/api/main.py:run                        → HTTP JSON
```

### Code map

```
src/agent/loop.py            run_agent: the think → act → observe loop and the system prompt
src/agent/guardrails.py      allowlist, argument-schema check, repeat break, injection scan (pure)
src/agent/mcp_client.py      ToolHost: spawns the MCP server, lists tools, call_tool() over stdio
src/agent/recovery.py        lifts a tool call the model left as JSON text in `content`
src/mcp_server/server.py     MCP server: calculator, knowledge_search, current_datetime
src/eval/run_eval.py         eval CLI: git provenance check, runs data/eval_set.jsonl, scores, writes results/
src/api/main.py              FastAPI: GET /health, POST /run
src/schema.py                TraceStep / AgentResult contracts
src/config.py                settings from the environment (the only reader of os.environ)
src/__init__.py, src/*/__init__.py, tests/__init__.py   empty package markers
demo.py                      one task from the command line, colored trace
data/eval_set.jsonl          8 gold tasks: expected tools + accepted answer markers
data/knowledge.jsonl         small Arabic policy knowledge base for knowledge_search
results/eval_500d1988.json   raw output of the measured run (per-task rows + aggregate + provenance)
docs/DESIGN.md               design rationale, and why the code looks like this
docs/results.md              hand-written narrative of the measured run
tests/test_guardrails.py     guardrail checks
tests/test_recovery.py       tool-call recovery parser
tests/test_scoring.py        task_completed / tool_call_correct / avg
tests/test_run_eval_provenance.py   dirty-tree refusal, results file contents
tests/test_mcp_server_tools.py      the three tools, called as plain functions
tests/test_config.py         settings and allowlist parsing
tests/test_schema.py         trace / result contracts
.github/workflows/tests.yml  CI: model-free test suite
requirements.txt             runtime dependencies
requirements-ci.txt          what the tests import (no LLM, no network)
pyproject.toml               project metadata, ruff/black settings
.env.example                 the environment variables config.py reads
.gitignore, .gitattributes, LICENSE, README.md
```

### Read the code in this order

1. `src/schema.py` — what a run produces.
2. `src/agent/loop.py` — `run_agent`, the whole control flow.
3. `src/agent/guardrails.py` — what is checked before and after each tool call.
4. `src/agent/mcp_client.py`, then `src/mcp_server/server.py` — how tools are found and run.
5. `src/agent/recovery.py` — the one workaround for the serving layer.
6. `src/eval/run_eval.py` — how the published numbers are produced.
7. `docs/DESIGN.md` — why each piece exists.

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
- **Recovery** (`src/agent/recovery.py`) — when the serving layer leaves a tool call as
  JSON text in `content`, it is lifted out and then goes through the same guardrails.
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

Three simple tasks finished in two iterations with exactly the expected tool; three
runs, including both multi-step tasks, hit the six-iteration guardrail. Eight tasks, so
one task is 12.5 points: read the pattern, not the rates. Commits after `500d1988` only
move code and rewrite docs; the agent's behaviour is unchanged.
Per-task rows, failure analysis and engineering findings:
[`docs/results.md`](docs/results.md).

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
