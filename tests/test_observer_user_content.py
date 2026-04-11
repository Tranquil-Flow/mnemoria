"""Tests for UserContentObserver — URLs and file paths from user messages."""
from mnemoria.observers.user_content import UserContentObserver
from mnemoria import events


def _obs():
    return UserContentObserver()


def test_extracts_url():
    e = events.user_message("check https://example.com/api/docs for reference", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) >= 1
    assert "https://example.com/api/docs" in facts[0].content


def test_extracts_file_path():
    e = events.user_message("look at /Users/dev/proj/src/main.py", session_id="s1")
    facts = _obs().observe(e)
    path_facts = [f for f in facts if "/Users/dev" in f.content]
    assert len(path_facts) >= 1


def test_short_path_ignored():
    e = events.user_message("see /a.b", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0


def test_no_urls_or_paths():
    e = events.user_message("please fix the login bug", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0


def test_ignores_tool_result():
    e = events.tool_result("https://example.com", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0


def test_multiple_urls():
    e = events.user_message("compare https://a.com and https://b.com", session_id="s1")
    facts = _obs().observe(e)
    url_facts = [f for f in facts if "https://" in f.content]
    assert len(url_facts) == 2


def test_source_is_user_stated():
    e = events.user_message("see https://docs.example.com", session_id="s1")
    facts = _obs().observe(e)
    assert all(f.source == "user_stated" for f in facts)
