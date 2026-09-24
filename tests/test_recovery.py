"""Unit tests for the tool-call recovery path in src/agent/loop.py.

qwen2.5-coder:3b sometimes emits a correct tool call as plain-text JSON in the
message `content` instead of the structured `tool_calls` field. These tests
exercise the recovery parser directly — no LLM, no MCP subprocess, no network.
"""

from __future__ import annotations

import json

from src.agent.loop import _json_objects, _tool_calls_from_content, _unwrap_schema_echo


class _FakeHost:
    """Stands in for agent.mcp_client.ToolHost: only `.tools` is read."""

    def __init__(self, tool_names: list[str]) -> None:
        self.tools = [{"function": {"name": name}} for name in tool_names]


class TestJsonObjects:
    def test_finds_a_single_top_level_object(self):
        found = list(_json_objects('some text {"a": 1} trailing'))
        assert found == [{"a": 1}]

    def test_finds_nested_object_as_one_top_level_match(self):
        text = '{"name": "calculator", "arguments": {"expression": "1+1"}}'
        found = list(_json_objects(text))
        assert found == [{"name": "calculator", "arguments": {"expression": "1+1"}}]

    def test_ignores_unbalanced_braces(self):
        assert list(_json_objects("{unterminated")) == []

    def test_skips_invalid_json_but_keeps_scanning(self):
        text = '{not valid json} then {"ok": true}'
        found = list(_json_objects(text))
        assert found == [{"ok": True}]

    def test_returns_nothing_for_plain_prose(self):
        assert list(_json_objects("no braces here at all")) == []

    def test_handles_multiple_separate_objects(self):
        text = '{"a": 1} some words {"b": 2}'
        found = list(_json_objects(text))
        assert found == [{"a": 1}, {"b": 2}]


class TestUnwrapSchemaEcho:
    def test_unwraps_a_schema_echoed_value(self):
        echoed = {"type": "string", "value": "23*17+5"}
        assert _unwrap_schema_echo(echoed) == "23*17+5"

    def test_passes_plain_strings_through_untouched(self):
        assert _unwrap_schema_echo("23*17+5") == "23*17+5"

    def test_passes_through_a_dict_with_no_recognizable_key(self):
        d = {"type": "string", "other": "x"}
        assert _unwrap_schema_echo(d) == d

    def test_prefers_value_over_default_and_example(self):
        d = {"type": "string", "value": "v", "default": "d"}
        assert _unwrap_schema_echo(d) == "v"

    def test_does_not_unwrap_a_larger_dict(self):
        # len > 3 means this isn't a small schema-echo shape; leave it alone.
        d = {"type": "string", "value": "v", "default": "d", "example": "e"}
        assert _unwrap_schema_echo(d) == d


class TestToolCallsFromContent:
    def test_recovers_a_plain_json_tool_call(self):
        host = _FakeHost(["calculator"])
        content = 'Sure, {"name": "calculator", "arguments": {"expression": "23*17+5"}}'
        calls = _tool_calls_from_content(content, host)
        assert len(calls) == 1
        assert calls[0].function.name == "calculator"
        assert json.loads(calls[0].function.arguments) == {"expression": "23*17+5"}

    def test_ignores_tool_names_the_server_does_not_expose(self):
        host = _FakeHost(["calculator"])
        content = '{"name": "delete_everything", "arguments": {}}'
        assert _tool_calls_from_content(content, host) == []

    def test_ignores_plain_prose_with_no_json(self):
        host = _FakeHost(["calculator"])
        assert _tool_calls_from_content("the answer is 396", host) == []

    def test_returns_empty_list_for_none_content(self):
        host = _FakeHost(["calculator"])
        assert _tool_calls_from_content(None, host) == []

    def test_unwraps_schema_echoed_arguments(self):
        host = _FakeHost(["calculator"])
        content = json.dumps(
            {
                "name": "calculator",
                "arguments": {"expression": {"type": "string", "value": "23*17+5"}},
            }
        )
        calls = _tool_calls_from_content(content, host)
        assert json.loads(calls[0].function.arguments) == {"expression": "23*17+5"}

    def test_accepts_tool_alias_keys_name_tool_function(self):
        host = _FakeHost(["current_datetime"])
        content = '{"tool": "current_datetime", "arguments": {}}'
        calls = _tool_calls_from_content(content, host)
        assert len(calls) == 1
        assert calls[0].function.name == "current_datetime"

    def test_arguments_given_as_a_json_string_are_parsed(self):
        host = _FakeHost(["calculator"])
        content = '{"name": "calculator", "arguments": "{\\"expression\\": \\"1+1\\"}"}'
        calls = _tool_calls_from_content(content, host)
        assert json.loads(calls[0].function.arguments) == {"expression": "1+1"}

    def test_malformed_arguments_string_is_skipped(self):
        host = _FakeHost(["calculator"])
        content = '{"name": "calculator", "arguments": "not json"}'
        assert _tool_calls_from_content(content, host) == []

    def test_recovers_multiple_calls_from_one_message(self):
        host = _FakeHost(["calculator", "current_datetime"])
        content = (
            '{"name": "calculator", "arguments": {"expression": "1+1"}} '
            'and also {"name": "current_datetime", "arguments": {}}'
        )
        calls = _tool_calls_from_content(content, host)
        assert [c.function.name for c in calls] == ["calculator", "current_datetime"]
