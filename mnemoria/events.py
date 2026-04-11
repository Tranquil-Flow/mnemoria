"""Hermes-agnostic event constructors for the Mnemoria observer pipeline.

Any integration (hermes-agent, CLI, custom agent loop) can use these
factories to build event dicts that core observers understand.
"""
import time as _time


def user_message(content: str, session_id: str = "", timestamp: float = None) -> dict:
    """Build an event dict for a user message."""
    return {
        "kind": "user_message",
        "session_id": session_id,
        "timestamp": timestamp or _time.time(),
        "payload": {"content": content},
    }


def tool_result(output: str = "", *, tool_name: str = "", exit_code: int = None,
                stdout: str = "", stderr: str = "", command: str = "",
                path: str = "", cwd: str = "",
                session_id: str = "", timestamp: float = None) -> dict:
    """Build an event dict for a tool result.

    Callers with structured data should pass individual fields.
    Callers with only raw text (e.g., on_pre_compress message iteration)
    should pass ``output``; it auto-populates ``stdout`` for observer compat.
    """
    return {
        "kind": "tool_result",
        "session_id": session_id,
        "timestamp": timestamp or _time.time(),
        "payload": {
            "output": output,
            "stdout": stdout or output,
            "stderr": stderr,
            "command": command,
            "tool": tool_name,
            "exit_code": exit_code,
            "path": path,
            "cwd": cwd,
        },
    }


def agent_message(content: str, session_id: str = "", timestamp: float = None) -> dict:
    """Build an event dict for an assistant/agent message."""
    return {
        "kind": "agent_message",
        "session_id": session_id,
        "timestamp": timestamp or _time.time(),
        "payload": {"content": content},
    }


def memory_write(action: str, target: str, content: str,
                 session_id: str = "", timestamp: float = None) -> dict:
    """Build an event dict for a built-in memory write."""
    return {
        "kind": "memory_write",
        "session_id": session_id,
        "timestamp": timestamp or _time.time(),
        "payload": {"action": action, "target": target, "content": content},
    }


def delegation(task: str, result: str, child_session_id: str = "",
               tool_trace: list = None, session_id: str = "",
               timestamp: float = None) -> dict:
    """Build an event dict for a delegation outcome."""
    return {
        "kind": "delegation",
        "session_id": session_id,
        "timestamp": timestamp or _time.time(),
        "payload": {
            "task": task,
            "result": result,
            "child_session_id": child_session_id,
            "tool_trace": tool_trace,
        },
    }
