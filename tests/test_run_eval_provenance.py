"""Unit tests for the git-provenance refusal logic in src/eval/run_eval.py.

subprocess.run is mocked throughout - these tests never touch the real git
repo, run the agent, or need Ollama/MCP/network.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.eval.run_eval import DirtyWorktreeError, check_git_state, score, write_results


def _fake_run(sha: str = "abc1234def5678900000000000000000000000", status: str = ""):
    def _run(cmd, **kwargs):
        result = type("Result", (), {})()
        if cmd[1:3] == ["rev-parse", "HEAD"]:
            result.returncode = 0
            result.stdout = sha + "\n"
            result.stderr = ""
        elif cmd[1:3] == ["status", "--porcelain"]:
            result.returncode = 0
            result.stdout = status
            result.stderr = ""
        else:
            raise AssertionError(f"unexpected git command: {cmd}")
        return result

    return _run


class TestCheckGitState:
    def test_clean_tree_returns_sha_and_clean_true(self):
        with patch("subprocess.run", side_effect=_fake_run(status="")):
            sha, clean = check_git_state(allow_dirty=False)
        assert sha == "abc1234def5678900000000000000000000000"
        assert clean is True

    def test_dirty_tree_without_allow_dirty_raises(self):
        with patch("subprocess.run", side_effect=_fake_run(status=" M src/config.py\n")):
            with pytest.raises(DirtyWorktreeError):
                check_git_state(allow_dirty=False)

    def test_dirty_tree_with_allow_dirty_is_recorded_not_rejected(self):
        with patch("subprocess.run", side_effect=_fake_run(status=" M src/config.py\n")):
            sha, clean = check_git_state(allow_dirty=True)
        assert clean is False
        assert sha

    def test_not_a_git_repo_raises_even_with_allow_dirty(self):
        def _run(cmd, **kwargs):
            result = type("Result", (), {})()
            result.returncode = 128
            result.stdout = ""
            result.stderr = "fatal: not a git repository"
            return result

        with patch("subprocess.run", side_effect=_run):
            with pytest.raises(DirtyWorktreeError):
                check_git_state(allow_dirty=True)


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
