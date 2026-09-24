"""Pure agent metrics. No I/O, no LLM — unit-testable in isolation."""

from __future__ import annotations


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
