"""Unit tests for the git-provenance refusal in src/provenance.py.

subprocess.run is mocked throughout - these tests never touch the real git repo.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.provenance import DirtyWorktreeError, check_provenance


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


class TestCheckProvenance:
    def test_clean_tree_returns_sha_and_clean_true(self):
        with patch("subprocess.run", side_effect=_fake_run(status="")):
            sha, clean = check_provenance(allow_dirty=False)
        assert sha == "abc1234def5678900000000000000000000000"
        assert clean is True

    def test_dirty_tree_without_allow_dirty_raises(self):
        with patch("subprocess.run", side_effect=_fake_run(status=" M src/config.py\n")):
            with pytest.raises(DirtyWorktreeError):
                check_provenance(allow_dirty=False)

    def test_dirty_tree_with_allow_dirty_is_recorded_not_rejected(self):
        with patch("subprocess.run", side_effect=_fake_run(status=" M src/config.py\n")):
            sha, clean = check_provenance(allow_dirty=True)
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
                check_provenance(allow_dirty=True)
