"""Tests for ErrorContextObserver — generic errors, URLs/paths near errors."""
from mnemoria.observers.error_context import ErrorContextObserver
from mnemoria import events


def _obs():
    return ErrorContextObserver()


def test_generic_error_line():
    e = events.tool_result("line1\nTraceback (most recent call last):\n  File x.py\nValueError: bad", session_id="s1")
    facts = _obs().observe(e)
    error_facts = [f for f in facts if f.type == "?"]
    assert len(error_facts) >= 1
    assert "Traceback" in error_facts[0].content or "ValueError" in error_facts[0].content


def test_no_error_no_facts():
    e = events.tool_result("all good\nno problems here", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0


def test_url_near_error():
    text = "Downloading from https://example.com/pkg.tar.gz\nERROR: checksum mismatch\nExpected abc"
    e = events.tool_result(text, session_id="s1")
    facts = _obs().observe(e)
    url_facts = [f for f in facts if "url" in f.target.lower() or "https://" in f.content]
    assert len(url_facts) >= 1
    assert "https://example.com/pkg.tar.gz" in url_facts[0].content


def test_url_far_from_error_ignored():
    text = "https://example.com/safe\n\n\n\n\n\n\n\nERROR: something broke"
    e = events.tool_result(text, session_id="s1")
    facts = _obs().observe(e)
    url_facts = [f for f in facts if "https://" in f.content]
    assert len(url_facts) == 0


def test_file_path_near_error():
    text = "Compiling /Users/dev/proj/src/main.py\nAssertionError: expected 5\nDone"
    e = events.tool_result(text, session_id="s1")
    facts = _obs().observe(e)
    path_facts = [f for f in facts if "/Users/dev" in f.content]
    assert len(path_facts) >= 1


def test_ignores_non_tool_result():
    e = events.user_message("ERROR: something", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0


def test_short_paths_ignored():
    text = "ERROR: fail\n/a.b"
    e = events.tool_result(text, session_id="s1")
    facts = _obs().observe(e)
    path_facts = [f for f in facts if f.target == "file"]
    assert len(path_facts) == 0
