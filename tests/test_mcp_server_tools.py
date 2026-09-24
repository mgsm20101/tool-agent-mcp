"""Unit tests for the MCP server's tools, called as plain Python functions.

@mcp.tool() (mcp 2.x MCPServer) registers the function but returns it
unmodified, so calculator/knowledge_search/current_datetime can be called
directly without spawning the server subprocess or talking MCP/stdio at all.
"""

from __future__ import annotations

import re

from src.mcp_server.server import calculator, current_datetime, knowledge_search


class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculator("23*17+5") == "396"

    def test_operator_precedence(self):
        assert calculator("144 / 12 + 88") == "100"

    def test_power(self):
        assert calculator("7 ** 3") == "343"

    def test_integer_results_have_no_trailing_dot_zero(self):
        assert calculator("4/2") == "2"

    def test_non_integer_float_keeps_its_decimal(self):
        assert calculator("5/2") == "2.5"

    def test_negative_numbers(self):
        assert calculator("-3 + 5") == "2"

    def test_division_by_zero_returns_an_error_string_not_an_exception(self):
        result = calculator("1/0")
        assert result.startswith("error:")

    def test_syntax_error_returns_an_error_string(self):
        result = calculator("1 +")
        assert result.startswith("error:")

    def test_rejects_non_arithmetic_code_injection(self):
        # __import__ etc. are not valid arithmetic AST nodes and must be rejected,
        # not executed - the whole point of walking the AST instead of eval().
        result = calculator("__import__('os').system('echo pwned')")
        assert result.startswith("error:")

    def test_rejects_name_lookups(self):
        result = calculator("x + 1")
        assert result.startswith("error:")


class TestKnowledgeSearch:
    def test_finds_the_vacation_policy(self):
        result = knowledge_search("الإجازة السنوية")
        assert "21" in result

    def test_finds_the_expense_policy(self):
        result = knowledge_search("المصروفات اليومية للسفر")
        assert "400" in result

    def test_no_match_returns_the_no_results_message(self):
        result = knowledge_search("zzz_no_such_topic_zzz")
        assert "لا توجد نتائج" in result

    def test_returns_at_most_two_entries(self):
        result = knowledge_search("سياسة")
        # each entry is rendered as "[title] text" separated by a blank line
        entries = [line for line in result.split("\n\n") if line.strip()]
        assert len(entries) <= 2


class TestCurrentDatetime:
    def test_includes_an_iso_formatted_timestamp_and_arabic_weekday(self):
        result = current_datetime()
        assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result)
        assert "(" in result and ")" in result
