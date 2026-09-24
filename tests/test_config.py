"""Unit tests for src/config.py.

`Settings`'s scalar fields read os.getenv(...) as their default at class
*definition* time (module import), so they can't be exercised by monkeypatching
the environment after import without reloading the module. `_allowlist` is a
plain function and `tool_allowlist` is a `default_factory`, so both are
re-evaluated on every call / instantiation - that's what these tests cover.
"""

from __future__ import annotations

from src.config import Settings, _allowlist


class TestAllowlist:
    def test_parses_a_comma_separated_list(self, monkeypatch):
        monkeypatch.setenv("TOOL_ALLOWLIST", "calculator,knowledge_search")
        assert _allowlist() == ["calculator", "knowledge_search"]

    def test_strips_whitespace_around_entries(self, monkeypatch):
        monkeypatch.setenv("TOOL_ALLOWLIST", " calculator , knowledge_search ")
        assert _allowlist() == ["calculator", "knowledge_search"]

    def test_empty_env_var_yields_empty_list(self, monkeypatch):
        monkeypatch.setenv("TOOL_ALLOWLIST", "")
        assert _allowlist() == []

    def test_missing_env_var_yields_empty_list(self, monkeypatch):
        monkeypatch.delenv("TOOL_ALLOWLIST", raising=False)
        assert _allowlist() == []

    def test_drops_empty_entries_from_stray_commas(self, monkeypatch):
        monkeypatch.setenv("TOOL_ALLOWLIST", "calculator,,knowledge_search,")
        assert _allowlist() == ["calculator", "knowledge_search"]


class TestSettingsDefaultFactory:
    def test_tool_allowlist_default_factory_reads_current_environment(self, monkeypatch):
        monkeypatch.setenv("TOOL_ALLOWLIST", "current_datetime")
        assert Settings().tool_allowlist == ["current_datetime"]

    def test_settings_fields_can_be_overridden_explicitly(self):
        s = Settings(
            ollama_host="http://example:1234",
            llm_model="test-model",
            llm_timeout_s=1.0,
            max_iterations=3,
            tool_allowlist=["calculator"],
        )
        assert s.ollama_host == "http://example:1234"
        assert s.llm_model == "test-model"
        assert s.max_iterations == 3
        assert s.tool_allowlist == ["calculator"]

    def test_settings_is_frozen(self):
        s = Settings(tool_allowlist=[])
        try:
            s.max_iterations = 99  # type: ignore[misc]
            assert False, "expected a FrozenInstanceError"
        except AttributeError:
            pass
