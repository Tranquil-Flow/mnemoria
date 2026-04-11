# Mnemoria Hermes Plugin: Full Lifecycle Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the Mnemoria hermes-agent plugin from a passive store/recall endpoint into a full lifecycle participant that hooks into compression, delegation, built-in memory mirroring, and session management.

**Architecture:** Two files: `extract.py` (pattern matching extraction logic, independently testable) and `provider.py` (MemoryProvider hook implementations that call into extract.py). The extraction module produces MEMORY_SPEC-formatted strings that MnemoriaStore.store() can parse directly.

**Tech Stack:** Python 3.10+, mnemoria package, hermes-agent MemoryProvider ABC, threading for background ops, re for pattern matching.

**Spec:** `docs/superpowers/specs/2026-04-10-hermes-plugin-lifecycle-hooks-design.md`

**Working directory:** `/Users/evinova/Projects/hermes-agent-mnemoria-pr-draft/` (worktree on `draft/mnemoria-plugin` branch)

**Important:** The worktree's git is now functional (fixed during this session). All file paths are relative to the worktree root.

---

### Task 1: Create extract.py with ExtractedFact dataclass and extract_from_text

**Files:**
- Create: `plugins/memory/mnemoria/extract.py`
- Create: `tests/plugins/memory/test_mnemoria_extract.py`

- [ ] **Step 1: Write failing tests for extract_from_text**

```python
# tests/plugins/memory/test_mnemoria_extract.py

from plugins.memory.mnemoria.extract import ExtractedFact, extract_from_text


class TestExtractFromTextErrors:
    def test_extracts_error_from_traceback(self):
        text = "Traceback (most recent call last):\n  File test_auth.py\nAssertionError: 3 tests failed"
        facts = extract_from_text(text, source="tool_result")
        assert any("?[error]" in f.content for f in facts)

    def test_extracts_url_near_error(self):
        text = "Error: connection refused\nFailed to reach https://api.example.com:3005\nRetrying..."
        facts = extract_from_text(text, source="tool_result")
        assert any("V[url]" in f.content and "api.example.com" in f.content for f in facts)

    def test_extracts_file_path_near_error(self):
        text = "FAILED tests/test_auth.py::test_jwt - AssertionError\nError in /src/auth/jwt.py line 42"
        facts = extract_from_text(text, source="tool_result")
        assert any("V[file]" in f.content and "/src/auth/jwt.py" in f.content for f in facts)

    def test_ignores_urls_without_error_context(self):
        text = "Documentation available at https://docs.example.com\nAll tests passed."
        facts = extract_from_text(text, source="tool_result")
        url_facts = [f for f in facts if "V[url]" in f.content]
        assert len(url_facts) == 0


class TestExtractFromTextUserStatements:
    def test_extracts_always_use_directive(self):
        text = "always use TypeScript for new code"
        facts = extract_from_text(text, source="user_statement")
        assert any("C[user.pref]" in f.content and "TypeScript" in f.content for f in facts)

    def test_extracts_never_use_directive(self):
        text = "never use var in JavaScript"
        facts = extract_from_text(text, source="user_statement")
        assert any("C[user.pref]" in f.content for f in facts)

    def test_ignores_conversational_dont(self):
        text = "I don't think that's right"
        facts = extract_from_text(text, source="user_statement")
        pref_facts = [f for f in facts if "C[user.pref]" in f.content]
        assert len(pref_facts) == 0

    def test_extracts_url_from_user_message(self):
        text = "check out https://docs.example.com/api for reference"
        facts = extract_from_text(text, source="user_statement")
        assert any("V[url]" in f.content and "docs.example.com" in f.content for f in facts)


class TestExtractFromTextEdgeCases:
    def test_empty_text_returns_empty(self):
        assert extract_from_text("", source="tool_result") == []

    def test_no_patterns_returns_empty(self):
        assert extract_from_text("Everything is fine.", source="tool_result") == []

    def test_confidence_is_set(self):
        text = "Error: connection failed to https://api.example.com"
        facts = extract_from_text(text, source="tool_result")
        for f in facts:
            assert 0.0 <= f.confidence <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_extract.py -v`
Expected: ImportError — `extract` module doesn't exist yet.

- [ ] **Step 3: Implement extract.py**

```python
# plugins/memory/mnemoria/extract.py
"""Rule-based fact extraction from conversation messages and text.

Extracts high-signal facts (errors, URLs near errors, file paths near errors,
user directives) as MEMORY_SPEC-formatted strings that MnemoriaStore.store()
can parse directly.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

# -----------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------

@dataclass
class ExtractedFact:
    content: str        # MEMORY_SPEC notation, e.g. "V[url]: https://..."
    source: str         # "tool_result" or "user_statement"
    confidence: float   # 0.0-1.0

# -----------------------------------------------------------------------
# Pattern definitions
# -----------------------------------------------------------------------

_ERROR_RE = re.compile(
    r"(?:error|failed|exception|traceback|assert(?:ion)?error|FAILED)",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://\S+")

_FILE_PATH_RE = re.compile(r"(/[\w./_+-]+\.\w+)")

_USER_DIRECTIVE_RE = re.compile(
    r"((?:always use|never use|prefer to use|don\'t use|don\'t ever|always make sure)\s+.+)",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------
# Core extraction
# -----------------------------------------------------------------------

def _extract_errors(lines: List[str], text: str) -> List[ExtractedFact]:
    """Extract error facts from tool output."""
    facts = []
    for i, line in enumerate(lines):
        if _ERROR_RE.search(line):
            # Summarise: take the error line, truncate to 200 chars
            summary = line.strip()[:200]
            facts.append(ExtractedFact(
                content=f"?[error]: {summary}",
                source="tool_result",
                confidence=0.9,
            ))
            break  # one error fact per text block is enough
    return facts


def _extract_urls_near_errors(lines: List[str]) -> List[ExtractedFact]:
    """Extract URLs that appear within 3 lines of an error pattern."""
    facts = []
    error_lines = {i for i, line in enumerate(lines) if _ERROR_RE.search(line)}
    if not error_lines:
        return facts

    seen_urls = set()
    for i, line in enumerate(lines):
        # Check if this line is within 3 lines of any error line
        near_error = any(abs(i - e) <= 3 for e in error_lines)
        if not near_error:
            continue
        for match in _URL_RE.finditer(line):
            url = match.group(0).rstrip(".,;:)\"'")
            if url not in seen_urls:
                seen_urls.add(url)
                facts.append(ExtractedFact(
                    content=f"V[url]: {url}",
                    source="tool_result",
                    confidence=0.85,
                ))
    return facts


def _extract_file_paths_near_errors(lines: List[str]) -> List[ExtractedFact]:
    """Extract file paths that appear within 3 lines of an error pattern."""
    facts = []
    error_lines = {i for i, line in enumerate(lines) if _ERROR_RE.search(line)}
    if not error_lines:
        return facts

    seen_paths = set()
    for i, line in enumerate(lines):
        near_error = any(abs(i - e) <= 3 for e in error_lines)
        if not near_error:
            continue
        for match in _FILE_PATH_RE.finditer(line):
            path = match.group(1)
            # Skip very short paths and common non-file patterns
            if len(path) < 5 or path.count("/") < 1:
                continue
            if path not in seen_paths:
                seen_paths.add(path)
                facts.append(ExtractedFact(
                    content=f"V[file]: {path}",
                    source="tool_result",
                    confidence=0.85,
                ))
    return facts


def _extract_user_directives(text: str) -> List[ExtractedFact]:
    """Extract user preference/directive statements."""
    facts = []
    for match in _USER_DIRECTIVE_RE.finditer(text):
        directive = match.group(1).strip()[:200]
        facts.append(ExtractedFact(
            content=f"C[user.pref]: {directive}",
            source="user_statement",
            confidence=0.6,
        ))
    return facts


def _extract_user_urls(text: str) -> List[ExtractedFact]:
    """Extract URLs from user messages (always relevant — user shared them)."""
    facts = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:)\"'")
        facts.append(ExtractedFact(
            content=f"V[url]: {url}",
            source="user_statement",
            confidence=0.8,
        ))
    return facts


def _extract_user_file_paths(text: str) -> List[ExtractedFact]:
    """Extract file paths from user messages."""
    facts = []
    for match in _FILE_PATH_RE.finditer(text):
        path = match.group(1)
        if len(path) < 5 or path.count("/") < 1:
            continue
        facts.append(ExtractedFact(
            content=f"V[file]: {path}",
            source="user_statement",
            confidence=0.8,
        ))
    return facts


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def extract_from_text(
    text: str,
    source: str = "tool_result",
) -> List[ExtractedFact]:
    """Extract facts from a plain text string.

    For tool_result source: extracts errors, and URLs/file paths near errors.
    For user_statement source: extracts directives, URLs, and file paths.
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines()

    if source == "user_statement":
        facts = []
        facts.extend(_extract_user_directives(text))
        facts.extend(_extract_user_urls(text))
        facts.extend(_extract_user_file_paths(text))
        return facts

    # tool_result: errors + contextual URLs/paths
    facts = []
    facts.extend(_extract_errors(lines, text))
    facts.extend(_extract_urls_near_errors(lines))
    facts.extend(_extract_file_paths_near_errors(lines))
    return facts


def extract_from_messages(
    messages: List[dict],
    start_index: int = 0,
) -> tuple:
    """Extract facts from conversation messages.

    Returns (extracted_facts, new_last_index).
    Only processes messages from start_index onward.
    """
    all_facts: List[ExtractedFact] = []

    for i in range(start_index, len(messages)):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or not isinstance(content, str):
            continue

        if role == "tool":
            all_facts.extend(extract_from_text(content, source="tool_result"))
        elif role == "user":
            all_facts.extend(extract_from_text(content, source="user_statement"))

    return all_facts, len(messages)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_extract.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/extract.py tests/plugins/memory/test_mnemoria_extract.py
git commit -m "feat(mnemoria): add rule-based fact extraction module"
```

---

### Task 2: Add extract_from_messages tests

**Files:**
- Modify: `tests/plugins/memory/test_mnemoria_extract.py`

- [ ] **Step 1: Write failing tests for extract_from_messages**

Append to `tests/plugins/memory/test_mnemoria_extract.py`:

```python
from plugins.memory.mnemoria.extract import extract_from_messages


class TestExtractFromMessages:
    def test_extracts_from_tool_role(self):
        messages = [
            {"role": "user", "content": "run the tests"},
            {"role": "assistant", "content": "Running pytest..."},
            {"role": "tool", "content": "FAILED tests/test_auth.py - Error in /src/auth.py"},
        ]
        facts, last_idx = extract_from_messages(messages)
        assert last_idx == 3
        assert any("?[error]" in f.content for f in facts)

    def test_extracts_from_user_role(self):
        messages = [
            {"role": "user", "content": "always use TypeScript for new code"},
        ]
        facts, last_idx = extract_from_messages(messages)
        assert last_idx == 1
        assert any("C[user.pref]" in f.content for f in facts)

    def test_skips_assistant_role(self):
        messages = [
            {"role": "assistant", "content": "Error: something went wrong"},
        ]
        facts, last_idx = extract_from_messages(messages)
        assert len(facts) == 0
        assert last_idx == 1

    def test_respects_start_index(self):
        messages = [
            {"role": "tool", "content": "Error: first failure"},
            {"role": "tool", "content": "Error: second failure"},
        ]
        facts, last_idx = extract_from_messages(messages, start_index=1)
        assert last_idx == 2
        # Should only find the second error
        assert len([f for f in facts if "?[error]" in f.content]) == 1

    def test_empty_messages_returns_empty(self):
        facts, last_idx = extract_from_messages([])
        assert facts == []
        assert last_idx == 0

    def test_handles_non_string_content(self):
        messages = [
            {"role": "tool", "content": None},
            {"role": "tool", "content": 42},
        ]
        facts, last_idx = extract_from_messages(messages)
        assert facts == []
        assert last_idx == 2
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_extract.py -v`
Expected: All tests PASS (extract_from_messages was implemented in Task 1).

- [ ] **Step 3: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add tests/plugins/memory/test_mnemoria_extract.py
git commit -m "test(mnemoria): add extract_from_messages test coverage"
```

---

### Task 3: Add slug generation utility to extract.py

**Files:**
- Modify: `plugins/memory/mnemoria/extract.py`
- Modify: `tests/plugins/memory/test_mnemoria_extract.py`

- [ ] **Step 1: Write failing tests for content_slug**

Append to `tests/plugins/memory/test_mnemoria_extract.py`:

```python
from plugins.memory.mnemoria.extract import content_slug


class TestContentSlug:
    def test_basic_slug(self):
        assert content_slug("Python 3.12 project") == "python-3.12"

    def test_strips_common_words(self):
        assert content_slug("The user prefers dark mode") == "user-prefers"

    def test_limits_to_three_words(self):
        slug = content_slug("one two three four five six")
        assert slug.count("-") <= 2  # at most 3 words = 2 hyphens

    def test_empty_string(self):
        assert content_slug("") == "general"

    def test_only_common_words(self):
        assert content_slug("the a is are") == "general"

    def test_lowercases(self):
        assert content_slug("JWT Authentication Setup") == "jwt-authentication"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_extract.py::TestContentSlug -v`
Expected: ImportError — `content_slug` doesn't exist yet.

- [ ] **Step 3: Implement content_slug in extract.py**

Add to `plugins/memory/mnemoria/extract.py` after the imports:

```python
# -----------------------------------------------------------------------
# Slug generation for on_memory_write target discrimination
# -----------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "that",
    "this", "it", "its", "and", "or", "but", "not", "no", "so",
})


def content_slug(text: str, max_words: int = 3) -> str:
    """Generate a short slug from content for target discrimination.

    Strips common words, takes first few meaningful words, lowercases,
    joins with hyphens. Returns "general" for empty/unresolvable input.
    """
    if not text:
        return "general"
    words = re.findall(r"[\w.]+", text.lower())
    meaningful = [w for w in words if w not in _STOP_WORDS]
    if not meaningful:
        return "general"
    return "-".join(meaningful[:max_words])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_extract.py::TestContentSlug -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/extract.py tests/plugins/memory/test_mnemoria_extract.py
git commit -m "feat(mnemoria): add content_slug for on_memory_write target discrimination"
```

---

### Task 4: Refactor provider.py — initialize with context-aware setup

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Write failing test for context-aware initialize**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
def test_initialize_sets_read_only_for_cron_context():
    provider = MnemoriaMemoryProvider()
    provider.initialize("test-session", agent_context="cron", hermes_home="/tmp")
    assert provider._read_only is True


def test_initialize_sets_read_only_for_flush_context():
    provider = MnemoriaMemoryProvider()
    provider.initialize("test-session", agent_context="flush", hermes_home="/tmp")
    assert provider._read_only is True


def test_initialize_not_read_only_for_primary_context():
    provider = MnemoriaMemoryProvider()
    provider.initialize("test-session", agent_context="primary", hermes_home="/tmp")
    assert provider._read_only is False


def test_initialize_stores_profile_and_user_id():
    provider = MnemoriaMemoryProvider()
    provider.initialize(
        "test-session",
        agent_identity="coder",
        user_id="user-abc",
        platform="telegram",
        hermes_home="/tmp",
    )
    assert provider._profile == "coder"
    assert provider._user_id == "user-abc"
    assert provider._platform == "telegram"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py::test_initialize_sets_read_only_for_cron_context -v`
Expected: FAIL — `_read_only` attribute doesn't exist.

- [ ] **Step 3: Update provider.py __init__ and initialize**

In `plugins/memory/mnemoria/provider.py`, replace the `__init__` and `initialize` methods:

```python
class MnemoriaMemoryProvider(MemoryProvider):
    """Mnemoria cognitive memory system as a hermes-agent memory provider."""

    def __init__(self):
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._read_only: bool = False
        self._profile: str = ""
        self._user_id: str = ""
        self._platform: str = "cli"
        self._last_extracted_msg_index: int = 0
        self._prefetch_result = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "mnemoria"

    def is_available(self) -> bool:
        """True when the mnemoria package is importable."""
        return _UM_AVAILABLE

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))

        # Context-aware: skip writes in cron/flush mode
        agent_context = kwargs.get("agent_context", "primary")
        self._read_only = agent_context in ("cron", "flush")

        # Profile and user scoping
        self._profile = kwargs.get("agent_identity", "")
        self._user_id = kwargs.get("user_id", "")
        self._platform = kwargs.get("platform", "cli")

        # Resolve DB path: env var > profile-scoped > default
        global _DB_PATH
        env_db = os.getenv("HERMES_MNEMORIA_DB")
        if env_db:
            _DB_PATH = env_db
        elif self._profile:
            _DB_PATH = str(Path(self._hermes_home) / f"mnemoria-{self._profile}.db")
        else:
            _DB_PATH = str(Path(self._hermes_home) / "mnemoria.db")

        # Reset per-thread store to pick up new DB path
        _local.store = None

        # Warm up the per-thread store
        try:
            _store()
            logger.info("MnemoriaMemoryProvider initialized (session=%s, read_only=%s)", session_id, self._read_only)
        except Exception as exc:
            logger.error("MnemoriaMemoryProvider init failed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "feat(mnemoria): context-aware initialize with profile and user scoping"
```

---

### Task 5: Implement system_prompt_block

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
def test_system_prompt_block_includes_usage_hint():
    provider = MnemoriaMemoryProvider()
    block = provider.system_prompt_block()
    assert "[MNEMORIA MEMORY]" in block
    assert "mnemoria_write" in block
    assert "mnemoria_recall" in block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py::test_system_prompt_block_includes_usage_hint -v`
Expected: FAIL — current system_prompt_block returns empty string.

- [ ] **Step 3: Implement system_prompt_block**

Replace the `system_prompt_block` method in `plugins/memory/mnemoria/provider.py`:

```python
    def system_prompt_block(self) -> str:
        """Return usage hint + identity facts (Constraints and Decisions)."""
        header = (
            "[MNEMORIA MEMORY]\n"
            "Mnemoria is active. Use mnemoria_write to store facts (supports\n"
            "MEMORY_SPEC notation: C[target]: constraints, D[target]: decisions,\n"
            "V[target]: values). Use mnemoria_recall for semantic search,\n"
            "mnemoria_explore for multi-hop discovery."
        )

        if not _UM_AVAILABLE:
            return header

        try:
            s = _store()
            rows = s.conn.execute(
                "SELECT type, target, content FROM um_facts "
                "WHERE type IN ('C', 'D') AND status = 'active' AND importance >= 0.7 "
                "ORDER BY activation DESC LIMIT 10"
            ).fetchall()

            if not rows:
                return header

            identity_lines = ["\n\n[MNEMORIA IDENTITY]"]
            for row in rows:
                identity_lines.append(f"{row['type']}[{row['target']}]: {row['content']}")

            return header + "\n".join(identity_lines)
        except Exception as exc:
            logger.debug("system_prompt_block identity query failed: %s", exc)
            return header
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "feat(mnemoria): system_prompt_block with usage hint and identity facts"
```

---

### Task 6: Implement get_config_schema and save_config

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
import json
import os


def test_get_config_schema_returns_valid_fields():
    provider = MnemoriaMemoryProvider()
    schema = provider.get_config_schema()
    assert isinstance(schema, list)
    assert len(schema) >= 1
    keys = {field["key"] for field in schema}
    assert "db_path" in keys


def test_save_config_writes_json(tmp_path):
    provider = MnemoriaMemoryProvider()
    provider.save_config({"db_path": "/custom/path.db"}, str(tmp_path))
    config_path = tmp_path / "mnemoria.json"
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["db_path"] == "/custom/path.db"


def test_save_config_merges_with_existing(tmp_path):
    config_path = tmp_path / "mnemoria.json"
    config_path.write_text(json.dumps({"existing_key": "value"}))
    provider = MnemoriaMemoryProvider()
    provider.save_config({"db_path": "/new/path.db"}, str(tmp_path))
    data = json.loads(config_path.read_text())
    assert data["existing_key"] == "value"
    assert data["db_path"] == "/new/path.db"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py::test_get_config_schema_returns_valid_fields -v`
Expected: FAIL — method doesn't exist.

- [ ] **Step 3: Add methods to MnemoriaMemoryProvider**

Add to the class in `plugins/memory/mnemoria/provider.py`:

```python
    def get_config_schema(self):
        return [
            {
                "key": "db_path",
                "description": "SQLite database path",
                "default": "~/.hermes/mnemoria.db",
            },
            {
                "key": "profile",
                "description": "Memory profile",
                "default": "balanced",
                "choices": ["balanced"],
            },
        ]

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mnemoria.json."""
        import json as _json
        config_path = Path(hermes_home) / "mnemoria.json"
        existing = {}
        if config_path.exists():
            try:
                existing = _json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(_json.dumps(existing, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "feat(mnemoria): add get_config_schema and save_config for setup wizard"
```

---

### Task 7: Implement queue_prefetch and update prefetch to use cache

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
def test_prefetch_returns_cached_result_from_queue(monkeypatch):
    """Verify that prefetch uses cached results from queue_prefetch."""
    provider = MnemoriaMemoryProvider()
    # Simulate a cached prefetch result (list of mock scored facts)
    from unittest.mock import MagicMock
    mock_fact = MagicMock()
    mock_fact.fact.fact_type = "V"
    mock_fact.fact.target = "test"
    mock_fact.fact.content = "cached content"
    provider._prefetch_result = [mock_fact]
    result = provider.prefetch("any query")
    assert "cached content" in result
    # Cache should be cleared after use
    assert provider._prefetch_result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py::test_prefetch_returns_cached_result_from_queue -v`
Expected: FAIL — `_prefetch_result` attribute doesn't exist yet (or prefetch doesn't check it).

- [ ] **Step 3: Implement queue_prefetch and update prefetch**

In `plugins/memory/mnemoria/provider.py`, replace the `prefetch` method and add `queue_prefetch`:

```python
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memories, using cached result from queue_prefetch if available."""
        if not _UM_AVAILABLE:
            return ""
        if not query:
            return ""

        # Check for cached result from previous queue_prefetch
        results = None
        with self._prefetch_lock:
            if self._prefetch_result is not None:
                results = self._prefetch_result
                self._prefetch_result = None

        # Fall back to synchronous recall
        if results is None:
            try:
                results = _store().recall(query, top_k=8)
            except Exception as exc:
                logger.warning("MnemoriaMemoryProvider prefetch failed: %s", exc)
                return ""

        if not results:
            return ""

        lines = ["[MNEMORIA MEMORY]"]
        for r in results:
            fact = r.fact
            type_sym = _type_sym(fact.fact_type)
            target = fact.target or "general"
            lines.append(f"{type_sym}[{target}]: {fact.content}")

        return "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Start background recall for the next turn's prefetch."""
        if self._read_only or not _UM_AVAILABLE or not query:
            return

        # Wait for previous prefetch to finish
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)

        def _run():
            try:
                results = _store().recall(query, top_k=8)
                with self._prefetch_lock:
                    self._prefetch_result = results
            except Exception as exc:
                logger.debug("Mnemoria queue_prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="mnemoria-prefetch",
        )
        self._prefetch_thread.start()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "feat(mnemoria): queue_prefetch background pre-warming + cache-aware prefetch"
```

---

### Task 8: Implement on_memory_write

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
def test_on_memory_write_is_noop_when_read_only():
    provider = MnemoriaMemoryProvider()
    provider._read_only = True
    # Should not raise even without a store
    provider.on_memory_write("add", "user", "some content")


def test_on_memory_write_skips_remove_action():
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    # "remove" should be silently skipped — no store needed
    provider.on_memory_write("remove", "user", "some content")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py::test_on_memory_write_is_noop_when_read_only -v`
Expected: FAIL — method doesn't exist.

- [ ] **Step 3: Implement on_memory_write**

Add to MnemoriaMemoryProvider in `plugins/memory/mnemoria/provider.py`:

```python
    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as typed Mnemoria facts."""
        if self._read_only:
            return
        if action not in ("add", "replace") or not content or not content.strip():
            return
        if not _UM_AVAILABLE:
            return

        from .extract import content_slug

        slug = content_slug(content)
        if target == "user":
            spec = f"V[user.{slug}]: {content.strip()}"
        else:
            spec = f"V[memory.{slug}]: {content.strip()}"

        def _run():
            try:
                _store().store(spec)
                logger.debug("on_memory_write mirrored: %s", spec[:80])
            except Exception as exc:
                logger.debug("on_memory_write failed: %s", exc)

        # Join previous write thread if still running
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)

        self._write_thread = threading.Thread(
            target=_run, daemon=True, name="mnemoria-memory-write",
        )
        self._write_thread.start()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "feat(mnemoria): on_memory_write mirrors built-in memory as typed facts"
```

---

### Task 9: Implement on_delegation

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
def test_on_delegation_is_noop_when_read_only():
    provider = MnemoriaMemoryProvider()
    provider._read_only = True
    provider.on_delegation("do research", "found nothing", child_session_id="child-1")


def test_on_delegation_does_not_raise_without_store(monkeypatch):
    monkeypatch.setattr(mnemoria_provider_module, "_UM_AVAILABLE", False)
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    provider.on_delegation("do research", "found nothing", child_session_id="child-1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py::test_on_delegation_is_noop_when_read_only -v`
Expected: FAIL — method doesn't exist.

- [ ] **Step 3: Implement on_delegation**

Add to MnemoriaMemoryProvider in `plugins/memory/mnemoria/provider.py`:

```python
    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Store delegation outcomes and extract facts from subagent results."""
        if self._read_only or not _UM_AVAILABLE:
            return

        try:
            s = _store()

            # Store summary fact
            task_short = (task or "")[:200].strip()
            result_short = (result or "")[:200].strip()
            if task_short or result_short:
                summary = f"D[delegation]: {task_short} -> {result_short}"
                s.store(summary)

            # Extract facts from the result text
            if result:
                from .extract import extract_from_text
                facts = extract_from_text(result, source="tool_result")
                scope = f"delegation:{child_session_id}" if child_session_id else "global"
                for fact in facts:
                    s.store(fact.content, scope=scope)
        except Exception as exc:
            logger.debug("on_delegation failed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "feat(mnemoria): on_delegation stores outcomes and extracts from results"
```

---

### Task 10: Implement on_pre_compress

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
def test_on_pre_compress_is_noop_when_read_only():
    provider = MnemoriaMemoryProvider()
    provider._read_only = True
    result = provider.on_pre_compress([{"role": "tool", "content": "Error: something broke"}])
    assert result == ""


def test_on_pre_compress_returns_empty_string():
    """Return value is discarded upstream, so always return empty."""
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    result = provider.on_pre_compress([])
    assert result == ""


def test_on_pre_compress_advances_message_index():
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    provider._last_extracted_msg_index = 0
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": "ok"},
    ]
    # Even without a store, the index should advance
    # (extraction may fail but index tracking should be robust)
    provider.on_pre_compress(messages)
    assert provider._last_extracted_msg_index == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py::test_on_pre_compress_is_noop_when_read_only -v`
Expected: FAIL — method doesn't exist.

- [ ] **Step 3: Implement on_pre_compress**

Add to MnemoriaMemoryProvider in `plugins/memory/mnemoria/provider.py`:

```python
    def on_pre_compress(self, messages) -> str:
        """Extract facts from messages before context compression discards them."""
        if self._read_only:
            return ""

        try:
            from .extract import extract_from_messages
            facts, new_index = extract_from_messages(
                messages, start_index=self._last_extracted_msg_index
            )
            self._last_extracted_msg_index = new_index

            if facts and _UM_AVAILABLE:
                s = _store()
                for fact in facts:
                    try:
                        s.store(fact.content)
                    except Exception as exc:
                        logger.debug("on_pre_compress store failed for %s: %s", fact.content[:50], exc)

                logger.info("on_pre_compress extracted %d facts", len(facts))
        except Exception as exc:
            # Still advance the index to avoid re-processing on next call
            self._last_extracted_msg_index = len(messages) if messages else 0
            logger.debug("on_pre_compress failed: %s", exc)

        return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "feat(mnemoria): on_pre_compress extracts facts before context compression"
```

---

### Task 11: Implement on_session_end

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
def test_on_session_end_does_not_raise_without_store(monkeypatch):
    monkeypatch.setattr(mnemoria_provider_module, "_UM_AVAILABLE", False)
    provider = MnemoriaMemoryProvider()
    provider._read_only = False
    provider.on_session_end([{"role": "user", "content": "bye"}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py::test_on_session_end_does_not_raise_without_store -v`
Expected: FAIL — method doesn't exist.

- [ ] **Step 3: Implement on_session_end**

Add to MnemoriaMemoryProvider in `plugins/memory/mnemoria/provider.py`:

```python
    def on_session_end(self, messages) -> None:
        """Extract remaining facts and run consolidation at session end."""
        if not _UM_AVAILABLE:
            return

        try:
            s = _store()

            # Extract from messages not yet processed by on_pre_compress
            if not self._read_only and messages:
                from .extract import extract_from_messages
                facts, new_index = extract_from_messages(
                    messages, start_index=self._last_extracted_msg_index
                )
                self._last_extracted_msg_index = new_index

                for fact in facts:
                    try:
                        s.store(fact.content)
                    except Exception:
                        pass

                if facts:
                    logger.info("on_session_end extracted %d facts", len(facts))

            # Always consolidate at session end (even in read-only mode)
            try:
                report = s.consolidate()
                logger.info(
                    "on_session_end consolidation: promoted=%d demoted=%d pruned=%d",
                    report.get("promoted", 0),
                    report.get("demoted", 0),
                    report.get("pruned", 0),
                )
            except Exception as exc:
                logger.debug("on_session_end consolidation failed: %s", exc)
        except Exception as exc:
            logger.debug("on_session_end failed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "feat(mnemoria): on_session_end extraction + consolidation"
```

---

### Task 12: Update shutdown to wait for background threads

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`

- [ ] **Step 1: Update shutdown method**

Replace the existing `shutdown` method in `plugins/memory/mnemoria/provider.py`:

```python
    def shutdown(self) -> None:
        """Close the per-thread store connection and wait for background threads."""
        # Wait for background threads
        for t in (self._prefetch_thread, self._write_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)

        try:
            store = getattr(_local, "store", None)
            if store is not None:
                store.conn.close()
                _local.store = None
            logger.info("MnemoriaMemoryProvider shutdown complete")
        except Exception as exc:
            logger.warning("MnemoriaMemoryProvider shutdown error: %s", exc)
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py tests/plugins/memory/test_mnemoria_extract.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py
git commit -m "feat(mnemoria): shutdown waits for background prefetch and write threads"
```

---

### Task 13: Clean up provider.py — remove dead code, fix imports

**Files:**
- Modify: `plugins/memory/mnemoria/provider.py`

- [ ] **Step 1: Remove old docstring and unused class docstring benchmarks claim**

In provider.py, update the class docstring. Remove the old "Benchmarks: 97.2%" claim and the stale config env var docs. The class docstring should be:

```python
class MnemoriaMemoryProvider(MemoryProvider):
    """Mnemoria cognitive memory system as a hermes-agent memory provider.

    Full lifecycle participant: hooks into context compression, delegation,
    built-in memory mirroring, and session management.
    """
```

Also remove the now-unused `_resolve_session` function (line 61-62) since the new hooks don't use it. Keep `_SESSION_ID` as it's used by `handle_tool_call`.

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py tests/plugins/memory/test_mnemoria_extract.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/provider.py
git commit -m "chore(mnemoria): clean up dead code and stale docstrings"
```

---

### Task 14: Update README.md with new hooks documentation

**Files:**
- Modify: `plugins/memory/mnemoria/README.md`

- [ ] **Step 1: Update README**

Replace the content of `plugins/memory/mnemoria/README.md`:

```markdown
# Mnemoria Memory Plugin

Cognitive memory system for hermes-agent combining ACT-R activation scoring, typed facts with metabolic decay, Hebbian link formation, and RL Q-value reranking.

## Requirements

```bash
pip install mnemoria
# or with sentence-transformers for better semantic recall:
pip install 'mnemoria[embeddings]'
```

## Setup

```bash
hermes memory setup
# select "mnemoria" when prompted
```

Or set manually in config:

```
memory.provider = mnemoria
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HERMES_MNEMORIA_DB` | `~/.hermes/mnemoria.db` | SQLite database path |

When using hermes profiles, each profile gets its own database automatically (`~/.hermes/mnemoria-{profile}.db`).

## Lifecycle Integration

Mnemoria participates in the full agent lifecycle:

| Hook | What it does |
|------|-------------|
| **initialize** | Context-aware setup — read-only mode for cron/flush, per-profile DB scoping |
| **system_prompt_block** | Injects identity facts (Constraints/Decisions) + tool usage hints |
| **prefetch / queue_prefetch** | Background pre-warming for faster recall on next turn |
| **on_memory_write** | Mirrors MEMORY.md/USER.md writes as typed Mnemoria facts |
| **on_pre_compress** | Extracts facts from messages before context compression discards them |
| **on_delegation** | Stores delegation outcomes + extracts facts from subagent results |
| **on_session_end** | Final fact extraction + consolidation (promote/demote/prune) |

## Tools

| Tool | Description |
|------|-------------|
| `mnemoria_write` | Store a fact using plain text or MEMORY_SPEC notation |
| `mnemoria_recall` | Semantic recall with 4-signal fusion |
| `mnemoria_search` | Fast FTS5 keyword search |
| `mnemoria_reflect` | Synthesize all facts about a topic, grouped by type |
| `mnemoria_reward` | RL feedback signal for Q-value training |
| `mnemoria_explore` | Multi-hop discovery via Personalized PageRank |
| `mnemoria_stats` | Store health check (fact count, gauge %) |
| `mnemoria_consolidate` | Run maintenance (promote/demote/prune) |

## MEMORY_SPEC Notation

| Notation | Type | Decay Rate | Example |
|----------|------|-----------|---------|
| `C[t]:` | Constraint | Slow (0.3x) | `C[db.id]: UUID mandatory` |
| `D[t]:` | Decision | Medium (0.7x) | `D[auth]: JWT 7d refresh 6d` |
| `V[t]:` | Value | Normal (1.0x) | `V[api.prod]: api.example.com` |
| `?[t]:` | Unknown | Fast (2.0x) | `?[cache]: should we cache?` |

## Links

- Repo: https://github.com/Tranquil-Flow/mnemoria
- PyPI: `pip install mnemoria`
- License: AGPL-3.0
```

- [ ] **Step 2: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add plugins/memory/mnemoria/README.md
git commit -m "docs(mnemoria): update README with lifecycle hooks documentation"
```

---

### Task 15: Final integration test — full lifecycle smoke test

**Files:**
- Modify: `tests/plugins/memory/test_mnemoria_plugin.py`

- [ ] **Step 1: Add integration smoke test**

Append to `tests/plugins/memory/test_mnemoria_plugin.py`:

```python
def test_full_lifecycle_smoke():
    """Smoke test: provider can be instantiated and all hook methods exist."""
    provider = MnemoriaMemoryProvider()

    # All hooks should be callable without raising
    assert callable(provider.initialize)
    assert callable(provider.system_prompt_block)
    assert callable(provider.prefetch)
    assert callable(provider.queue_prefetch)
    assert callable(provider.on_memory_write)
    assert callable(provider.on_delegation)
    assert callable(provider.on_pre_compress)
    assert callable(provider.on_session_end)
    assert callable(provider.get_config_schema)
    assert callable(provider.save_config)
    assert callable(provider.shutdown)

    # system_prompt_block should return a non-empty string with usage hint
    block = provider.system_prompt_block()
    assert isinstance(block, str)
    assert len(block) > 0

    # get_config_schema should return a list
    schema = provider.get_config_schema()
    assert isinstance(schema, list)
    assert len(schema) > 0
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft && python -m pytest tests/plugins/memory/test_mnemoria_plugin.py tests/plugins/memory/test_mnemoria_extract.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/evinova/Projects/hermes-agent-mnemoria-pr-draft
git add tests/plugins/memory/test_mnemoria_plugin.py
git commit -m "test(mnemoria): add full lifecycle smoke test"
```
