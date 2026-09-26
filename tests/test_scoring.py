"""Unit tests for the scoring functions in src/eval/run_eval.py — pure, no I/O, no LLM."""

from __future__ import annotations

from src.eval.run_eval import avg, task_completed, tool_call_correct


class TestTaskCompleted:
    def test_returns_true_when_answer_contains_the_only_marker(self):
        assert task_completed("the result is 396", ["396"]) is True

    def test_returns_true_when_answer_contains_any_of_several_markers(self):
        # numeral vs. word form of the same answer
        assert task_completed("العدد المسموح هو ثلاثة أيام", ["3", "ثلاثة"]) is True

    def test_returns_false_when_answer_contains_none_of_the_markers(self):
        assert task_completed("لا أعرف الإجابة", ["396"]) is False

    def test_returns_false_for_empty_markers_list(self):
        assert task_completed("396", []) is False

    def test_is_case_and_substring_sensitive(self):
        assert task_completed("the answer is THREE", ["three"]) is False
        assert task_completed("343 is the cube", ["343"]) is True


class TestToolCallCorrect:
    def test_true_when_used_tools_match_expected_exactly(self):
        assert tool_call_correct(["calculator"], ["calculator"]) is True

    def test_true_when_order_differs(self):
        assert tool_call_correct(["b", "a"], ["a", "b"]) is True

    def test_false_when_a_required_tool_is_missing(self):
        assert tool_call_correct([], ["calculator"]) is False

    def test_false_when_an_extra_tool_was_used(self):
        assert tool_call_correct(["calculator", "current_datetime"], ["calculator"]) is False

    def test_true_when_both_sets_are_empty(self):
        assert tool_call_correct([], []) is True

    def test_duplicates_in_used_tools_do_not_break_equality(self):
        assert tool_call_correct(["calculator", "calculator"], ["calculator"]) is True


class TestAvg:
    def test_computes_the_mean(self):
        assert avg([1.0, 2.0, 3.0]) == 2.0

    def test_returns_zero_for_empty_list(self):
        assert avg([]) == 0.0

    def test_handles_a_single_value(self):
        assert avg([5.5]) == 5.5
