"""Contracts shared by the loop, guardrails, eval and API.

A run produces an AgentResult: the final answer plus a full Trace — one TraceStep per
thing that happened (model turn, tool call, observation, guardrail intervention). The
trace is what makes the agent debuggable and measurable instead of a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StepType(str, Enum):
    MODEL_TURN = "model_turn"        # the LLM produced text and/or tool calls
    TOOL_CALL = "tool_call"          # we executed a tool
    OBSERVATION = "observation"      # the tool's result fed back to the model
    GUARDRAIL = "guardrail"          # a guardrail intervened (blocked/stopped/sanitized)
    FINAL = "final"                  # the answer returned to the caller


@dataclass
class TraceStep:
    step: int
    type: StepType
    content: str
    tool: str | None = None
    args: dict | None = None
    latency_ms: float | None = None


@dataclass
class AgentResult:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    iterations: int = 0
    tools_used: list[str] = field(default_factory=list)
    stopped_by: str = "answer"  # "answer" | "max_iterations" | "guardrail"
