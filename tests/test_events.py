"""Tests for mnemoria.events — event constructors for observer pipeline."""
import time
from mnemoria import events


def test_user_message_basic():
    e = events.user_message("hello world", session_id="s1")
    assert e["kind"] == "user_message"
    assert e["session_id"] == "s1"
    assert e["payload"]["content"] == "hello world"
    assert isinstance(e["timestamp"], float)


def test_user_message_defaults():
    e = events.user_message("hi")
    assert e["session_id"] == ""
    assert e["timestamp"] > 0


def test_tool_result_structured():
    e = events.tool_result(
        stdout="PASSED", stderr="", command="pytest tests/",
        tool_name="pytest", exit_code=0, session_id="s1",
    )
    assert e["kind"] == "tool_result"
    assert e["payload"]["stdout"] == "PASSED"
    assert e["payload"]["tool"] == "pytest"
    assert e["payload"]["exit_code"] == 0
    assert e["payload"]["command"] == "pytest tests/"


def test_tool_result_raw_fallback():
    """When only output is provided, it should populate stdout for observer compat."""
    e = events.tool_result("error: file not found", session_id="s1")
    assert e["payload"]["output"] == "error: file not found"
    assert e["payload"]["stdout"] == "error: file not found"
    assert e["payload"]["stderr"] == ""
    assert e["payload"]["tool"] == ""
    assert e["payload"]["command"] == ""


def test_tool_result_stdout_overrides_output():
    """When stdout is explicitly provided, it takes precedence over output."""
    e = events.tool_result(output="raw", stdout="structured")
    assert e["payload"]["stdout"] == "structured"
    assert e["payload"]["output"] == "raw"


def test_agent_message():
    e = events.agent_message("I'll fix that", session_id="s2")
    assert e["kind"] == "agent_message"
    assert e["payload"]["content"] == "I'll fix that"


def test_memory_write():
    e = events.memory_write("add", "user", "prefers TypeScript", session_id="s1")
    assert e["kind"] == "memory_write"
    assert e["payload"]["action"] == "add"
    assert e["payload"]["target"] == "user"
    assert e["payload"]["content"] == "prefers TypeScript"


def test_delegation_basic():
    e = events.delegation("run tests", "all passed", child_session_id="c1", session_id="s1")
    assert e["kind"] == "delegation"
    assert e["payload"]["task"] == "run tests"
    assert e["payload"]["result"] == "all passed"
    assert e["payload"]["child_session_id"] == "c1"
    assert e["payload"]["tool_trace"] is None


def test_delegation_with_tool_trace():
    trace = [{"tool": "pytest", "success": True}, {"tool": "git", "success": False}]
    e = events.delegation("deploy", "failed", tool_trace=trace, session_id="s1")
    assert e["payload"]["tool_trace"] == trace
    assert len(e["payload"]["tool_trace"]) == 2


def test_timestamp_override():
    ts = 1700000000.0
    e = events.user_message("test", timestamp=ts)
    assert e["timestamp"] == ts
