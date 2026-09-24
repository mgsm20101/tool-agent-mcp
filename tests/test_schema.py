"""Unit tests for src/schema.py: the TraceStep / AgentResult contracts shared by
the loop, guardrails, eval and API."""

from __future__ import annotations

from src.schema import AgentResult, StepType, TraceStep


class TestStepType:
    def test_is_a_string_enum_so_it_serializes_cleanly(self):
        assert StepType.TOOL_CALL == "tool_call"
        assert StepType.TOOL_CALL.value == "tool_call"

    def test_all_expected_members_exist(self):
        names = {member.name for member in StepType}
        assert names == {"MODEL_TURN", "TOOL_CALL", "OBSERVATION", "GUARDRAIL", "FINAL"}


class TestTraceStep:
    def test_required_fields_only(self):
        step = TraceStep(step=1, type=StepType.MODEL_TURN, content="hello")
        assert step.tool is None
        assert step.args is None
        assert step.latency_ms is None

    def test_all_fields(self):
        step = TraceStep(
            step=2,
            type=StepType.TOOL_CALL,
            content="calling",
            tool="calculator",
            args={"expression": "1+1"},
            latency_ms=12.5,
        )
        assert step.tool == "calculator"
        assert step.args == {"expression": "1+1"}
        assert step.latency_ms == 12.5


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult(answer="42")
        assert result.answer == "42"
        assert result.trace == []
        assert result.iterations == 0
        assert result.tools_used == []
        assert result.stopped_by == "answer"

    def test_trace_defaults_are_independent_between_instances(self):
        # dataclass field(default_factory=list) must not share state across instances
        a = AgentResult(answer="a")
        b = AgentResult(answer="b")
        a.trace.append(TraceStep(step=1, type=StepType.FINAL, content="a"))
        assert b.trace == []

    def test_stopped_by_can_be_max_iterations_or_guardrail(self):
        r1 = AgentResult(answer="", stopped_by="max_iterations")
        r2 = AgentResult(answer="", stopped_by="guardrail")
        assert r1.stopped_by == "max_iterations"
        assert r2.stopped_by == "guardrail"
