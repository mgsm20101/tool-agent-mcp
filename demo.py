"""Run one task through the agent and print the full trace.

    python demo.py "كم يساوي 23*17+5؟"
    python demo.py "كم يوم إجازة سنوية أستحق بعد 6 سنوات خدمة؟"
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys

from rich.console import Console

from src.agent.loop import run_agent
from src.schema import StepType

# Make Arabic survive Windows consoles.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
console = Console(file=sys.stdout)

_COLORS = {
    StepType.MODEL_TURN: "cyan",
    StepType.TOOL_CALL: "yellow",
    StepType.OBSERVATION: "green",
    StepType.GUARDRAIL: "red",
    StepType.FINAL: "bold magenta",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tool-using agent on one task.")
    parser.add_argument("task", help="the task, Arabic or English")
    args = parser.parse_args()

    result = asyncio.run(run_agent(args.task))

    console.print(f"\n[bold]Task:[/bold] {args.task}\n")
    for s in result.trace:
        label = s.type.value.upper()
        tool = f" {s.tool}" if s.tool else ""
        ms = f" ({s.latency_ms:.0f}ms)" if s.latency_ms else ""
        console.print(f"[{_COLORS[s.type]}]{s.step:>2}. {label}{tool}{ms}[/] {s.content[:300]}")

    console.print(f"\n[bold]Answer:[/bold] {result.answer}")
    console.print(
        f"[dim]iterations={result.iterations} tools={result.tools_used} "
        f"stopped_by={result.stopped_by}[/dim]"
    )


if __name__ == "__main__":
    main()
