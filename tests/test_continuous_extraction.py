"""Integration test for continuous extraction (Wave 8.3).

End-to-end test: feed a synthetic event stream through store_pending,
verify pending facts accumulate, verify promotion, verify retraction on
contradiction.
"""
import os
import tempfile

import pytest

from mnemoria import MnemoriaConfig, MnemoriaStore


def make_store():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    store = MnemoriaStore(cfg)
    store._promotion_every = 10000  # Disable auto-promote
    return tmpdir, store


def pending_count(store):
    return store.conn.execute(
        "SELECT COUNT(*) FROM um_pending WHERE status = 'provisional'"
    ).fetchone()[0]


def facts_count(store):
    return store.conn.execute(
        "SELECT COUNT(*) FROM um_facts WHERE status = 'active'"
    ).fetchone()[0]


def retracted_count(store):
    return store.conn.execute(
        "SELECT COUNT(*) FROM um_pending WHERE status = 'retracted'"
    ).fetchone()[0]


# ─── Synthetic event stream helpers ────────────────────────────────────────────

def feed_observed(store, session_id, tool, exit_code, stdout, timestamp=None):
    """Simulate an observed tool result event."""
    store.store_pending(
        content=f"pytest: {stdout}" if exit_code != 0 else "tests passing",
        source="observed",
        fact_type="V",
        target=f"test.{tool}",
        session_id=session_id,
        provenance={"extractor": f"{tool}_observer", "event": "tool_result"},
    )


def feed_user_statement(store, session_id, content, timestamp=None):
    """Simulate a user message event."""
    store.store_pending(
        content=content,
        source="user_stated",
        fact_type="V",
        target="preference",
        session_id=session_id,
        provenance={"extractor": "user_statement_observer", "event": "user_message"},
    )


def feed_agent_inference(store, session_id, content, timestamp=None):
    """Simulate an agent inference event."""
    store.store_pending(
        content=content,
        source="agent_inference",
        fact_type="V",
        target="inference",
        session_id=session_id,
        provenance={"extractor": "inference_observer", "event": "agent_message"},
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_pending_facts_accumulate():
    """Pending facts should accumulate as events are processed."""
    tmpdir, store = make_store()
    try:
        assert pending_count(store) == 0

        # Feed several events
        feed_user_statement(store, "s1", "I prefer using PostgreSQL for databases")
        assert pending_count(store) == 1

        feed_observed(store, "s1", "pytest", 1, "FAILED test_auth.py::test_login")
        assert pending_count(store) == 2

        feed_agent_inference(store, "s1", "User seems concerned about security")
        assert pending_count(store) == 3

        # Facts promoted via store should not be in pending
        assert facts_count(store) == 0
    finally:
        tmpdir.cleanup()


def test_observed_promotes_immediately_on_flush():
    """Observed facts should be promoted immediately when flush_pending is called."""
    tmpdir, store = make_store()
    try:
        feed_observed(store, "s1", "pytest", 1, "FAILED test_foo.py")

        assert pending_count(store) == 1
        assert facts_count(store) == 0

        stats = store.flush_pending()

        assert stats["observed_promoted"] == 1
        assert pending_count(store) == 0
        assert facts_count(store) == 1
    finally:
        tmpdir.cleanup()


def test_user_stated_promotes_immediately_on_flush():
    """User stated facts should be promoted immediately."""
    tmpdir, store = make_store()
    try:
        feed_user_statement(store, "s1", "I always use dark mode")

        assert pending_count(store) == 1

        stats = store.flush_pending()

        assert stats["user_stated_promoted"] == 1
        assert pending_count(store) == 0
        assert facts_count(store) == 1
    finally:
        tmpdir.cleanup()


def test_agent_inference_waits_for_ttl():
    """Agent inference facts should NOT promote until session is inactive."""
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()
        now = store._now()

        feed_agent_inference(store, "s1", "User likely prefers Python")

        assert pending_count(store) == 1

        # Immediate flush should NOT promote agent_inference
        stats = store.flush_pending()
        assert stats["agent_inference_promoted"] == 0
        assert pending_count(store) == 1
        assert facts_count(store) == 0

        # Advance time past TTL (600s)
        store.advance_time(601)
        stats = store.flush_pending()
        assert stats["agent_inference_promoted"] == 1
        assert pending_count(store) == 0
        assert facts_count(store) == 1
    finally:
        tmpdir.cleanup()


def test_retraction_on_contradiction():
    """When a newer fact contradicts an older pending one, the older is retracted.

    Note: retraction only applies within um_pending (same type/target/session).
    Once a fact is promoted to um_facts, subsequent pending facts can't retract it.
    """
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()

        # First: user states a preference (stays pending)
        feed_user_statement(store, "s1", "I prefer PostgreSQL for databases")
        assert pending_count(store) == 1

        # Advance time (still in same session)
        store.advance_time(10)

        # Second: user states a different preference for the SAME target
        # This should trigger retraction of the first pending fact
        feed_user_statement(store, "s1", "Actually I prefer SQLite for small projects")
        assert pending_count(store) == 2  # Now 2 pending

        stats = store.flush_pending()

        # The retraction pass should have retracted the OLD PostgreSQL fact
        # and promoted the NEW SQLite fact
        assert stats["retracted"] == 1, f"Expected retraction but got: {stats}"
        assert stats["user_stated_promoted"] == 1

        # Only the SQLite fact should be in um_facts
        assert facts_count(store) == 1
        row = store.conn.execute(
            "SELECT content FROM um_facts WHERE status = 'active'"
        ).fetchone()
        assert "SQLite" in row["content"], f"Expected SQLite, got: {row['content']}"
    finally:
        tmpdir.cleanup()


def test_event_stream_full_pipeline():
    """Full pipeline: mix of events, accumulation, promotion, retraction.

    Note: retraction only happens within um_pending for same (type, target, session).
    Once a fact is promoted to um_facts, it can't be retracted by new pending facts.
    """
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()

        # Phase 1: accumulate pending facts
        # Retraction only happens during flush_pending(), not at store time.
        # Both user_stated facts survive accumulation (same target, different updated_at).
        feed_user_statement(store, "s1", "I use macOS for development")
        feed_user_statement(store, "s1", "I prefer dark mode")
        feed_observed(store, "s1", "git", 0, "Everything clean")
        feed_observed(store, "s1", "pytest", 0, "all tests passed")
        feed_agent_inference(store, "s1", "User prefers terminal-based workflows")

        assert pending_count(store) == 5, "All 5 pending during accumulation"
        assert facts_count(store) == 0

        # Phase 2: first promotion — observed + user_stated immediately
        # During flush: macOS retracted (older), dark mode promotes (1 user_stated left after retraction)
        stats = store.flush_pending()
        assert stats["observed_promoted"] == 2, "git + pytest different targets → both promote"
        assert stats["user_stated_promoted"] == 1, "Only dark mode survives; macOS retracted during pass"
        assert stats["retracted"] == 1, "macOS retracted in same pass"
        assert stats["agent_inference_promoted"] == 0, "Should not yet promote agent_inference"
        assert pending_count(store) == 1, "Only agent_inference remains"
        assert facts_count(store) == 3, "dark mode + git + pytest"

        # Phase 3: session becomes inactive, agent_inference promotes
        store.advance_time(601)
        stats = store.flush_pending()
        assert stats["agent_inference_promoted"] == 1
        assert pending_count(store) == 0
        assert facts_count(store) == 4, "git + pytest + macOS + agent_inference"

        # Phase 4: new preference (different target) — no retraction expected
        # since the old user_stated was already promoted
        store.advance_time(10)
        feed_user_statement(store, "s1", "Actually I switched to Linux")
        assert pending_count(store) == 1
        stats = store.flush_pending()

        # No retraction (first fact already promoted), just promotion
        assert stats["retracted"] == 0
        assert stats["user_stated_promoted"] == 1
        assert facts_count(store) == 5, "4 prior + Linux"

        # Verify Linux is in facts
        rows = store.conn.execute(
            "SELECT content FROM um_facts WHERE status = 'active'"
        ).fetchall()
        contents = [r["content"] for r in rows]
        assert any("Linux" in c for c in contents), f"Expected Linux in facts: {contents}"
        # macOS is still there (dark mode was retracted before promotion, macOS was promoted first)
        assert any("macOS" in c for c in contents), f"macOS should still be there: {contents}"
    finally:
        tmpdir.cleanup()


def test_mixed_sessions_independent():
    """Facts from different sessions should not affect each other."""
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()

        # Session 1: user preference
        feed_user_statement(store, "s1", "I prefer coffee")
        # Session 2: user preference
        feed_user_statement(store, "s2", "I prefer tea")

        stats = store.flush_pending()

        assert stats["user_stated_promoted"] == 2
        assert facts_count(store) == 2

        # Verify both facts are present (um_facts doesn't have session_id,
        # so we check content)
        rows = store.conn.execute(
            "SELECT content FROM um_facts WHERE status = 'active'"
        ).fetchall()
        contents = [r["content"] for r in rows]
        assert any("coffee" in c for c in contents), f"Expected coffee in facts: {contents}"
        assert any("tea" in c for c in contents), f"Expected tea in facts: {contents}"
    finally:
        tmpdir.cleanup()


def test_crash_recovery_integration():
    """Simulate crash: pending facts survive abnormal shutdown, promote on reopen."""
    tmpdir, store = make_store()
    try:
        db_path = store._config.db_path

        # Accumulate pending facts
        feed_observed(store, "s1", "pytest", 1, "FAILED")
        feed_user_statement(store, "s1", "Remember: I prefer quiet mode")
        assert pending_count(store) == 2

        # Simulate crash: close without final commit or proper cleanup
        store._conn.execute("PRAGMA journal_mode=DELETE")
        store._conn.close()

        # Reopen — crash recovery in __init__ should promote observed + user_stated
        cfg = MnemoriaConfig.balanced()
        cfg.db_path = db_path
        cfg.enable_pressure = False
        store2 = MnemoriaStore(cfg)

        # The init-time run_promotion_pass should have promoted the observed/user_stated
        assert facts_count(store2) >= 1, (
            "Crash recovery should promote observed/user_stated pending facts"
        )
    finally:
        tmpdir.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
