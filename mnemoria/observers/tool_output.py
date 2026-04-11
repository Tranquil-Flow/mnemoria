"""
Tool-output observers for Mnemoria v0.2.0.

Each watches tool_call / tool_result events and extracts structured facts
from deterministic patterns.
"""
from __future__ import annotations

import re
from typing import Optional

from . import PendingFact, Observer


# -------------------------------------------------------------------------- #
# PytestObserver
# -------------------------------------------------------------------------- #

class PytestObserver:
    """Extracts facts from pytest execution results."""

    name: str = "pytest"

    def observe(self, event: dict) -> list[PendingFact]:
        if event.get("kind") not in ("tool_call", "tool_result"):
            return []

        tool_name, exit_code, stdout, stderr, command = self._extract_fields(event)
        if tool_name is None:
            return []

        # Match pytest invocation
        if tool_name != "pytest" and not self._is_pytest_command(command):
            return []

        # Exit 0 = all passing; nothing to record
        if exit_code == 0:
            return []

        # Non-zero exit → check for FAILED in output
        output = (stdout or "") + (stderr or "")
        if "FAILED" not in output:
            return []

        # Extract first failure message
        first_failure = self._first_failure_message(output)
        target = self._infer_target(event, command)

        # Emit the failing test fact
        content = f"tests failing in {target}: {first_failure}"

        facts = [
            PendingFact(
                content=content,
                type="V",          # test failures are Verdict-style observations
                target=target,
                source="observed",
                provenance={
                    "extractor": self.name,
                    "exit_code": exit_code,
                    "session_id": event.get("session_id"),
                    "command": command,
                },
            )
        ]

        # Retract any provisional "tests passing" facts from the same session
        retraction = self._make_retraction(event, target)
        if retraction:
            facts.append(retraction)

        return facts

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    PYTEST_RE = re.compile(r"pytest(?:\s+|$)", re.IGNORECASE)

    def _is_pytest_command(self, command: Optional[str]) -> bool:
        return bool(command and self.PYTEST_RE.search(command))

    def _extract_fields(self, event: dict):
        payload = event.get("payload", {})
        kind = event.get("kind")

        if kind == "tool_result":
            tool_name = payload.get("tool", "")
            exit_code = payload.get("exit_code", 0)
            stdout = payload.get("stdout", "")
            stderr = payload.get("stderr", "")
            command = payload.get("command", "")
        elif kind == "tool_call":
            tool_name = payload.get("tool", "")
            exit_code = None
            stdout = ""
            stderr = ""
            command = payload.get("command", "")
        else:
            tool_name = None
            exit_code = None
            stdout = ""
            stderr = ""
            command = ""

        return tool_name, exit_code, stdout, stderr, command

    FAILURE_RE = re.compile(r"^(?:FAILED|PASSED|ERROR)[^$]*$", re.MULTILINE)

    def _first_failure_message(self, output: str) -> str:
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAILED"):
                # "FAILED test_foo.py::test_bar - AssertionError: expected 1 got 2"
                parts = stripped.split(" - ", 1)
                return parts[0] if len(parts) == 1 else parts[1]
        return "test(s) failed"

    def _infer_target(self, event: dict, command: Optional[str]) -> str:
        # Try to extract path from pytest command
        if command:
            # Grab positional args or -k/-c flags that point to a path
            parts = command.split()
            for part in parts:
                if part.endswith(".py") and "::" in part:
                    # "tests/unit/test_bar.py::test_baz" → project from dir
                    return part.split("::")[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                elif part.endswith(".py"):
                    return part.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        # Fall back to cwd in payload
        cwd = event.get("payload", {}).get("cwd", "")
        if cwd:
            return cwd.rstrip("/").rsplit("/", 1)[-1] or "tests"
        return "tests"

    def _make_retraction(self, event: dict, target: str) -> Optional[PendingFact]:
        """Return a retraction for any provisional 'tests passing' fact."""
        session_id = event.get("session_id", "")
        return PendingFact(
            content=f"tests passing in {target}",
            type="V",
            target=target,
            source="observed",
            provenance={
                "extractor": self.name,
                "retracts_session": session_id,
                "reason": "tests now failing",
            },
            retract=True,
        )


# -------------------------------------------------------------------------- #
# GitObserver
# -------------------------------------------------------------------------- #

class GitObserver:
    """Extracts facts from git command results."""

    name: str = "git"

    def observe(self, event: dict) -> list[PendingFact]:
        if event.get("kind") != "tool_result":
            return []

        payload = event.get("payload", {})
        if payload.get("tool") != "git":
            return []

        exit_code = payload.get("exit_code", 0)
        stderr = payload.get("stderr", "")
        command = payload.get("command", "")

        if not command:
            return []

        facts: list[PendingFact] = []
        session_id = event.get("session_id", "")

        # git push rejected
        if self._is_push(command):
            if exit_code != 0 and "rejected" in stderr.lower():
                branch = self._extract_branch(command)
                raw_reason = self._first_line(stderr).strip()
                # Avoid redundancy: "push rejected: push rejected" → "push to main rejected"
                if raw_reason.lower() == "rejected":
                    reason = "remote rejected the update"
                elif "rejected" in raw_reason.lower():
                    reason = raw_reason
                else:
                    reason = raw_reason
                facts.append(
                    PendingFact(
                        content=f"push to {branch} rejected: {reason}",
                        type="V",
                        target=self._repo_name(command),
                        source="observed",
                        provenance={
                            "extractor": self.name,
                            "command": command,
                            "session_id": session_id,
                        },
                    )
                )

        # git commit with non-default author
        elif self._is_commit(command):
            author = payload.get("author") or self._extract_author_from_stderr(stderr)
            if author and not self._is_default_author(author):
                repo = self._repo_name(command)
                facts.append(
                    PendingFact(
                        content=f"commits in {repo} use author {author}",
                        type="C",        # commits author is a Configuration fact
                        target=repo,
                        source="observed",
                        provenance={
                            "extractor": self.name,
                            "author": author,
                            "command": command,
                            "session_id": session_id,
                        },
                    )
                )

        # git status with untracked files → NO fact (too noisy per spec)

        return facts

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_push(self, command: str) -> bool:
        return bool(re.search(r"git\s+push", command, re.IGNORECASE))

    def _is_commit(self, command: str) -> bool:
        return bool(re.search(r"git\s+(?:commit|commit -a)", command, re.IGNORECASE))

    PUSH_BRANCH_RE = re.compile(
        r"git\s+push\s+(?:-\w+\s+)*(?:\S+\s+)?(\S+?)(?:\s+\S+)*$",
        re.IGNORECASE,
    )

    def _extract_branch(self, command: str) -> str:
        m = self.PUSH_BRANCH_RE.search(command)
        return m.group(1) if m else "main"

    COMMIT_AUTHOR_RE = re.compile(r"Author:\s*(.+)", re.IGNORECASE)

    def _extract_author_from_stderr(self, stderr: str) -> Optional[str]:
        m = self.COMMIT_AUTHOR_RE.search(stderr)
        return m.group(1).strip() if m else None

    def _is_default_author(self, author: str) -> bool:
        # If we can't determine the default, don't assume — be conservative
        default = self._global_git_author()
        return bool(default and author == default)

    def _global_git_author(self) -> Optional[str]:
        import subprocess
        try:
            result = subprocess.run(
                ["git", "config", "--global", "user.email"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    # Matches typical git URL styles: https://host/user/repo, git@host:user/repo, /path/to/repo
    REPO_NAME_RE = re.compile(
        r"(?:https://[^/]+/([^/]+/\S+)|git@[^:]+:([^/\s]+/\S+)|/([^/\s]+)(?:\.git)?/)",
        re.IGNORECASE,
    )

    def _repo_name(self, command: str) -> str:
        m = self.REPO_NAME_RE.search(command)
        if m:
            for g in (m.group(1), m.group(2), m.group(3)):
                if g:
                    return g.rstrip("/").rsplit("/", 1)[-1] or g
        return "repo"

    def _first_line(self, text: str) -> str:
        return text.strip().splitlines()[0] if text.strip() else "unknown"


# -------------------------------------------------------------------------- #
# FileObserver
# -------------------------------------------------------------------------- #

# Paths that are noise — not worth recording
_NOISE_PATTERNS = re.compile(
    r"(/|^)(\.venv|__pycache__|\.git|node_modules|\.pytest_cache|"
    r"\.mypy_cache|\.tox|\.eggs|\.nox|dist|build|\.sass-cache)"
    r"(/|$)",
    re.IGNORECASE,
)

# Config file names that, when read repeatedly, suggest project layout
_CONFIG_FILES = frozenset([
    "pyproject.toml", "setup.py", "setup.cfg", "poetry.lock",
    "requirements.txt", "Pipfile", "pyproject.toml",
    "package.json", "Cargo.toml", "go.mod", "Gemfile",
    ".env", ".env.local", "config.toml", "config.yaml", "config.yml",
])

class FileObserver:
    """Extracts facts from file read/write tool calls."""

    name: str = "file"

    def __init__(self) -> None:
        # Track reads per session to detect repetition (instance-level to avoid unbounded growth)
        self._session_file_reads: dict[str, list[str]] = {}

    def observe(self, event: dict) -> list[PendingFact]:
        if event.get("kind") not in ("tool_call", "tool_result"):
            return []

        payload = event.get("payload", {})
        tool_name = payload.get("tool", "").lower()

        # Match file tool names (generic; actual tool names vary by agent)
        is_file_tool = tool_name in (
            "read_file", "write_file", "edit_file", "create_file",
            "read", "write", "file_read", "file_write",
            "Read", "Write", "FileRead", "FileWrite",
        )

        if not is_file_tool:
            # Also check command string
            command = payload.get("command", "")
            is_file_tool = bool(re.search(
                r"(?:cat|head|tail|read|write|edit|create)\s+", command, re.IGNORECASE
            ))

        if not is_file_tool:
            return []

        path = payload.get("path") or ""
        session_id = event.get("session_id", "")

        # Skip noise paths
        if _NOISE_PATTERNS.search(path):
            return []

        # Record the read for repetition detection
        if event.get("kind") == "tool_call":
            self._session_file_reads.setdefault(session_id, []).append(path)

        # Repeated config file reads → emit a fact
        if event.get("kind") == "tool_result" and self._is_config_file(path):
            reads = self._session_file_reads.get(session_id, [])
            count = reads.count(path)
            if count >= 2:
                return self._config_fact(path, session_id)

        return []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_config_file(self, path: str) -> bool:
        name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return name in _CONFIG_FILES

    def _config_fact(self, path: str, session_id: str) -> list[PendingFact]:
        project = self._project_from_path(path)
        return [
            PendingFact(
                content=f"{project} config lives at {path}",
                type="D",          # declarative fact about project structure
                target=project,
                source="observed",
                provenance={
                    "extractor": self.name,
                    "path": path,
                    "session_id": session_id,
                },
            )
        ]

    def _project_from_path(self, path: str) -> str:
        parts = path.strip("/").split("/")
        # Assume project root is ~2 levels above config file
        if len(parts) >= 2:
            return parts[-2]
        return parts[-1] if parts else "project"


# -------------------------------------------------------------------------- #
# Convenience factory
# -------------------------------------------------------------------------- #

def all_observers() -> list[Observer]:
    """Return every built-in observer instance."""
    return [PytestObserver(), GitObserver(), FileObserver()]