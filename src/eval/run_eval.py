"""Agent eval: run every task in the gold set and score the runs.

Metrics:
- task completion rate   did the final answer contain the expected result?
- tool-call accuracy     were exactly the expected tools were used?
- avg iterations         how many loop turns a task takes on average
- guardrail stops        runs that ended by guardrail instead of an answer

Every run is provenance-tracked: it refuses to score against a dirty git
worktree (or outside a git repo) unless `--allow-dirty` is passed, and it
writes the full raw output — per-task rows, aggregate metrics, model, host,
timestamp, commit SHA, worktree state — to `results/eval_<sha8>.json`. A
number that only exists in a README or a Markdown table, with no matching
file in `results/`, should not be trusted.

Run:  python -m src.eval.run_eval
      python -m src.eval.run_eval --allow-dirty
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.agent.loop import run_agent
from src.config import settings
from src.provenance import check_provenance

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
console = Console(file=sys.stdout)

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "data" / "eval_set.jsonl"
RESULTS_DIR = ROOT / "results"


def _load() -> list[dict]:
    return [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]


async def _run_all(rows: list[dict]) -> list[dict]:
    outcomes = []
    for i, row in enumerate(rows, start=1):
        console.print(f"[dim]({i}/{len(rows)})[/dim] {row['task'][:60]}")
        result = await run_agent(row["task"])
        ok = task_completed(result.answer, row["answer_contains"])
        tools_ok = tool_call_correct(result.tools_used, row["expected_tools"])
        outcomes.append(
            {
                "task": row["task"],
                "expected_tools": row["expected_tools"],
                "answer_contains": row["answer_contains"],
                "completed": ok,
                "tools_ok": tools_ok,
                "iterations": result.iterations,
                "stopped_by": result.stopped_by,
                "tools_used": result.tools_used,
                "answer": result.answer,
            }
        )
        mark = "[green]ok[/green]" if ok else "[red]MISS[/red]"
        console.print(f"    -> {mark} tools={result.tools_used} iters={result.iterations}")
    return outcomes


# Scoring: pure functions, no I/O, no LLM.


def task_completed(answer: str, answer_contains: list[str]) -> bool:
    """A task counts as completed if the answer contains ANY of the accepted markers
    (alternatives cover numeral vs word forms, e.g. "3" / "ثلاثة")."""
    return any(marker in answer for marker in answer_contains)


def tool_call_correct(tools_used: list[str], expected_tools: list[str]) -> bool:
    """All expected tools were used and nothing outside the expected set was."""
    used, expected = set(tools_used), set(expected_tools)
    return expected.issubset(used) and used.issubset(expected)


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score(outcomes: list[dict]) -> dict:
    n = len(outcomes)
    return {
        "task_completion": sum(o["completed"] for o in outcomes) / n,
        "tool_call_accuracy": sum(o["tools_ok"] for o in outcomes) / n,
        "avg_iterations": avg([o["iterations"] for o in outcomes]),
        "guardrail_stops": sum(o["stopped_by"] != "answer" for o in outcomes),
    }


def write_results(
    outcomes: list[dict], scores: dict, source_commit_sha: str, worktree_clean: bool
) -> Path:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": settings.llm_model,
        "ollama_host": settings.ollama_host,
        "source_commit_sha": source_commit_sha,
        "worktree_clean": worktree_clean,
        "aggregate": scores,
        "tasks": outcomes,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{source_commit_sha[:8]}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gold eval set against the agent.")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run even with uncommitted changes; records worktree_clean=false",
    )
    args = parser.parse_args()

    source_commit_sha, worktree_clean = check_provenance(args.allow_dirty)

    rows = _load()
    outcomes = asyncio.run(_run_all(rows))
    scores = score(outcomes)

    table = Table(title="Tool Agent - Evaluation", header_style="bold")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    for k, v in scores.items():
        table.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
    console.print(table)

    out_path = write_results(outcomes, scores, source_commit_sha, worktree_clean)
    console.print(f"[green]Wrote[/green] {out_path}")


if __name__ == "__main__":
    main()
