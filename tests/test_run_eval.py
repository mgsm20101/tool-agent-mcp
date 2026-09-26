"""Unit tests for scoring and the results file in src/eval/run_eval.py.

These tests never run the agent or need Ollama/MCP/network.
"""

from __future__ import annotations

from src.eval.run_eval import score, write_results


class TestScore:
    def test_aggregates_completion_and_tool_accuracy(self):
        outcomes = [
            {"completed": True, "tools_ok": True, "iterations": 2, "stopped_by": "answer"},
            {"completed": False, "tools_ok": False, "iterations": 6, "stopped_by": "max_iterations"},
        ]
        result = score(outcomes)
        assert result["task_completion"] == 0.5
        assert result["tool_call_accuracy"] == 0.5
        assert result["avg_iterations"] == 4.0
        assert result["guardrail_stops"] == 1


class TestWriteResults:
    def test_writes_a_json_file_named_after_the_short_sha(self, tmp_path, monkeypatch):
        import src.eval.run_eval as run_eval_module

        monkeypatch.setattr(run_eval_module, "RESULTS_DIR", tmp_path / "results")
        outcomes = [{"task": "t", "completed": True}]
        scores = {"task_completion": 1.0}

        out_path = write_results(outcomes, scores, "deadbeef00112233", worktree_clean=True)

        assert out_path.name == "eval_deadbeef.json"
        assert out_path.exists()
        import json

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["source_commit_sha"] == "deadbeef00112233"
        assert payload["worktree_clean"] is True
        assert payload["aggregate"] == scores
        assert payload["tasks"] == outcomes
        assert "model" in payload
        assert "ollama_host" in payload
        assert "timestamp" in payload
