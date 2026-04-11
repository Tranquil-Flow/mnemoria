"""Tests for MemoryWriteObserver and content_slug."""
from mnemoria.observers.memory_write import MemoryWriteObserver, content_slug
from mnemoria import events


# --- content_slug tests (migrated from plugin extract.py tests) ---

def test_content_slug_basic():
    assert content_slug("JWT authentication tokens") == "jwt-authentication-tokens"

def test_content_slug_stops():
    assert content_slug("the authentication of the system") == "authentication-system"

def test_content_slug_max_words():
    assert content_slug("one two three four five", max_words=2) == "one-two"

def test_content_slug_empty():
    assert content_slug("") == "general"

def test_content_slug_only_stops():
    assert content_slug("the a an") == "general"

def test_content_slug_dotted():
    assert content_slug("user.timezone setting") == "user.timezone-setting"


# --- MemoryWriteObserver tests ---

def _obs():
    return MemoryWriteObserver()

def test_add_user_write():
    e = events.memory_write("add", "user", "prefers TypeScript for frontend", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 1
    assert facts[0].type == "V"
    assert "user." in facts[0].target
    assert "prefers TypeScript" in facts[0].content

def test_replace_memory_write():
    e = events.memory_write("replace", "memory", "API key is abc123", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 1
    assert "memory." in facts[0].target

def test_delete_action_ignored():
    e = events.memory_write("delete", "user", "something", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0

def test_empty_content_ignored():
    e = events.memory_write("add", "user", "", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0

def test_ignores_non_memory_write():
    e = events.user_message("test", session_id="s1")
    facts = _obs().observe(e)
    assert len(facts) == 0
