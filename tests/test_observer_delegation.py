"""Tests for DelegationObserver."""
from mnemoria.observers.delegation import DelegationObserver
from mnemoria import events


def _obs():
    return DelegationObserver()

def test_basic_delegation():
    e = events.delegation("run pytest", "3 tests passed", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) >= 1
    d_facts = [f for f in facts if f.type == "D"]
    assert len(d_facts) == 1
    assert "run pytest" in d_facts[0].content
    assert "3 tests passed" in d_facts[0].content

def test_empty_task_and_result():
    e = events.delegation("", "", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0

def test_tool_trace_extraction():
    trace = [
        {"tool": "pytest", "success": False},
        {"tool": "git", "action": "push", "success": True},
    ]
    e = events.delegation("deploy app", "partial success", tool_trace=trace, session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) >= 2
    trace_facts = [f for f in facts if "subagent used" in f.content]
    assert len(trace_facts) == 2

def test_tool_trace_none():
    e = events.delegation("run build", "success", tool_trace=None, session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 1

def test_ignores_non_delegation():
    e = events.user_message("delegate this", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0

def test_long_task_truncated():
    long_task = "x" * 500
    e = events.delegation(long_task, "done", session_id="s1")
    facts = _obs().observe(e)
    d_facts = [f for f in facts if f.type == "D"]
    assert len(d_facts[0].content) <= 420  # D[delegation]: + 200 + " -> " + 200
