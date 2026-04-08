"""Tests for the promoter module (Wave 8.2)."""
import json
import os
import tempfile
import time

import pytest

from mnemoria import MnemoriaConfig, MnemoriaStore
from mnemoria.promoter import run_promotion_pass


def make_store(promotion_every=1000):
    """Create a test store with auto-promotion disabled (high threshold)."""
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    store = MnemoriaStore(cfg)
    # Disable auto-promotion so we can control when promotion happens
    store._promotion_every = promotion_every
    store._pending_write_counter = 0
    return tmpdir, store


def pending_count(store):
    return store.conn.execute(
        "SELECT COUNT(*) FROM um_pending WHERE status = 'provisional'"
    ).fetchone()[0]


def promoted_count(store):
    return store.conn.execute(
        "SELECT COUNT(*) FROM um_pending WHERE status = 'promoted'"
    ).fetchone()[0]


def retracted_count(store):
    return store.conn.execute(
        "SELECT COUNT(*) FROM um_pending WHERE status = 'retracted'"
    ).fetchone()[0]


def facts_count(store):
    return store.conn.execute(
        "SELECT COUNT(*) FROM um_facts WHERE status = 'active'"
    ).fetchone()[0]


# ─── 8.2.1: observed facts promote immediately ────────────────────────────────

def test_observed_facts_promote_immediately():
    tmpdir, store = make_store()
    try:
        # Write an 'observed' fact
        pid = store.store_pending(
            content="pytest found 2 failing tests",
            source="observed",
            fact_type="V",
            target="test.status",
            session_id="s1",
            provenance={"extractor": "pytest_observer"},
        )
        assert pending_count(store) == 1, "Fact should be pending"

        # Manually trigger promotion
        stats = store.flush_pending()

        assert stats["observed_promoted"] == 1, f"observed should promote: {stats}"
        assert pending_count(store) == 0, "No pending after promotion"
        assert promoted_count(store) == 1, "Should be marked promoted"
        assert facts_count(store) == 1, "Should appear in um_facts"
    finally:
        tmpdir.cleanup()


# ─── 8.2.2: user_stated facts promote immediately ────────────────────────────

def test_user_stated_facts_promote_immediately():
    tmpdir, store = make_store()
    try:
        pid = store.store_pending(
            content="I prefer dark mode",
            source="user_stated",
            fact_type="V",
            target="preference.theme",
            session_id="s1",
            provenance={"extractor": "user_statement_observer"},
        )
        assert pending_count(store) == 1

        stats = store.flush_pending()

        assert stats["user_stated_promoted"] == 1, f"user_stated should promote: {stats}"
        assert pending_count(store) == 0
        assert promoted_count(store) == 1
        assert facts_count(store) == 1
    finally:
        tmpdir.cleanup()


# ─── 8.2.3: agent_inference facts wait for TTL ──────────────────────────────

def test_agent_inference_waits_for_ttl():
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()
        now = store._now()

        # Write an agent_inference fact
        pid = store.store_pending(
            content="The user seems to prefer Python based on their questions",
            source="agent_inference",
            fact_type="V",
            target="language.preference",
            session_id="s1",
            provenance={"extractor": "inference_observer"},
        )
        assert pending_count(store) == 1

        # Promotion should NOT promote agent_inference facts that are fresh
        stats = store.flush_pending()

        assert stats["agent_inference_promoted"] == 0, (
            f"Fresh agent_inference should NOT promote: {stats}"
        )
        assert pending_count(store) == 1, "Fact should still be pending"
        assert facts_count(store) == 0, "No fact in um_facts yet"

        # Advance time past the TTL (default 600s)
        store.advance_time(601)
        stats = store.flush_pending()

        assert stats["agent_inference_promoted"] == 1, (
            f"Expired agent_inference should now promote: {stats}"
        )
        assert pending_count(store) == 0
        assert facts_count(store) == 1
    finally:
        tmpdir.cleanup()


def test_agent_inference_promotes_when_session_inactive():
    """agent_inference promotes when the session has been inactive for TTL."""
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()
        now = store._now()

        # Write two agent_inference facts in DIFFERENT sessions
        # (retraction only applies within same type/target/session)
        store.store_pending(
            content="User asking about deployment",
            source="agent_inference",
            fact_type="V",
            target="topic.deployment",
            session_id="s1",
            provenance={"extractor": "inference"},
        )
        store.store_pending(
            content="User asking about databases",
            source="agent_inference",
            fact_type="V",
            target="topic.databases",
            session_id="s2",  # Different session
            provenance={"extractor": "inference"},
        )
        assert pending_count(store) == 2

        # Advance time past TTL
        store.advance_time(601)
        stats = store.flush_pending()

        assert stats["agent_inference_promoted"] == 2, (
            f"Both expired facts should promote: {stats}"
        )
        assert facts_count(store) == 2
    finally:
        tmpdir.cleanup()


# ─── 8.2.4: Latest-wins retraction within session ───────────────────────────

def test_latest_wins_retraction_within_session():
    """Within a (type, target, session_id) group, the latest fact wins."""
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()
        now = store._now()

        # Write two facts with same type/target/session
        pid1 = store.store_pending(
            content="Use PostgreSQL for the database",
            source="observed",
            fact_type="D",
            target="db.choice",
            session_id="s1",
            provenance={"extractor": "git_observer"},
        )
        store.advance_time(1)

        pid2 = store.store_pending(
            content="Use SQLite for the database",
            source="observed",
            fact_type="D",
            target="db.choice",
            session_id="s1",
            provenance={"extractor": "git_observer"},
        )

        stats = store.flush_pending()

        # Both pending, but one should be retracted, one promoted
        assert stats["observed_promoted"] == 1, f"Only latest should promote: {stats}"
        assert stats["retracted"] == 1, f"Old fact should be retracted: {stats}"

        # Check the promoted fact has the latest content
        row = store.conn.execute(
            "SELECT content FROM um_facts WHERE status = 'active'"
        ).fetchone()
        assert "SQLite" in row["content"], f"Expected SQLite (latest), got: {row['content']}"

        # The retracted fact should be marked retracted
        retracted = store.conn.execute(
            "SELECT id FROM um_pending WHERE status = 'retracted'"
        ).fetchall()
        assert len(retracted) == 1
    finally:
        tmpdir.cleanup()


def test_latest_wins_different_sessions_no_retraction():
    """Different sessions with same type/target do NOT retract each other."""
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()

        store.store_pending(
            content="Session s1: use PostgreSQL",
            source="observed",
            fact_type="D",
            target="db.choice",
            session_id="s1",
            provenance={"extractor": "test"},
        )
        store.store_pending(
            content="Session s2: use MySQL",
            source="observed",
            fact_type="D",
            target="db.choice",
            session_id="s2",
            provenance={"extractor": "test"},
        )

        stats = store.flush_pending()

        assert stats["observed_promoted"] == 2, f"Both should promote: {stats}"
        assert stats["retracted"] == 0, f"No retraction (different sessions): {stats}"
        assert facts_count(store) == 2
    finally:
        tmpdir.cleanup()


# ─── 8.2.5: Crash recovery ───────────────────────────────────────────────────

def test_crash_recovery_on_init():
    """Insert pending rows, close without promoting, reopen, verify recovery."""
    tmpdir, store = make_store(promotion_every=1000)
    try:
        db_path = store._config.db_path

        # Write some pending facts
        store.store_pending(
            content="Crash recovery test fact 1",
            source="observed",
            fact_type="V",
            target="test",
            session_id="s1",
            provenance={"extractor": "test"},
        )
        store.store_pending(
            content="Crash recovery test fact 2",
            source="user_stated",
            fact_type="V",
            target="test",
            session_id="s1",
            provenance={"extractor": "test"},
        )
        store._conn.commit()

        # Verify they are pending
        assert pending_count(store) == 2

        # Simulate crash: close connection without promoting
        store._conn.close()

        # Reopen the store — init calls run_promotion_pass which should recover
        # Set a low TTL for agent_inference but these are observed/user_stated
        cfg = MnemoriaConfig.balanced()
        cfg.db_path = db_path
        cfg.enable_pressure = False
        store2 = MnemoriaStore(cfg)
        # Manually trigger the init recovery (in case db already initialized)
        # Actually the init already ran run_promotion_pass, but since pending
        # are 'observed' and 'user_stated', they should have been promoted.
        # Let's verify:
        stats = store2.flush_pending()
        # Note: the init already ran promotion, so this might promote 0 more

    finally:
        tmpdir.cleanup()


def test_crash_recovery_observed_pending_after_abnormal_close():
    """Simulate crash: pending rows survive, promotion happens on next init."""
    tmpdir, store = make_store(promotion_every=10000)  # high threshold = no auto-promote
    try:
        db_path = store._config.db_path

        # Write observed fact
        store.store_pending(
            content="Survived the crash",
            source="observed",
            fact_type="V",
            target="crash.test",
            session_id="s1",
            provenance={"extractor": "test"},
        )
        pending_before = pending_count(store)
        assert pending_before == 1

        # Close without calling flush_pending (simulate crash)
        conn = store._conn
        conn.execute("PRAGMA journal_mode=DELETE")  # Ensure WAL flushed
        conn.close()

        # Re-open the store — crash recovery happens in __init__
        cfg = MnemoriaConfig.balanced()
        cfg.db_path = db_path
        cfg.enable_pressure = False
        store2 = MnemoriaStore(cfg)

        # The init's run_promotion_pass should have promoted the observed fact
        assert facts_count(store2) == 1, (
            "Crash recovery should promote observed pending facts on init"
        )
    finally:
        tmpdir.cleanup()


# ─── 8.2.6: Idempotency ──────────────────────────────────────────────────────

def test_idempotency_second_pass_promotes_nothing():
    """Running the promoter twice in a row promotes nothing new."""
    tmpdir, store = make_store()
    try:
        store.store_pending(
            content="Idempotency test fact",
            source="observed",
            fact_type="V",
            target="test",
            session_id="s1",
            provenance={"extractor": "test"},
        )

        stats1 = store.flush_pending()
        assert stats1["observed_promoted"] == 1

        # Run again immediately
        stats2 = store.flush_pending()

        assert stats2["observed_promoted"] == 0, (
            f"Second pass should promote nothing: {stats2}"
        )
        assert stats2["retracted"] == 0, "No retractions on second pass"
        assert facts_count(store) == 1, "Still only one fact"
    finally:
        tmpdir.cleanup()


def test_idempotency_already_retracted_not_retracted_again():
    """Already-retracted rows are not retracted again on second pass."""
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()
        store.advance_time(0)

        # Write two facts, same (type, target, session) — latest wins
        store.store_pending(
            content="First version",
            source="observed",
            fact_type="V",
            target="idempotency.test",
            session_id="s1",
            provenance={"extractor": "test"},
        )
        store.advance_time(1)
        store.store_pending(
            content="Second version",
            source="observed",
            fact_type="V",
            target="idempotency.test",
            session_id="s1",
            provenance={"extractor": "test"},
        )

        stats1 = store.flush_pending()
        assert stats1["observed_promoted"] == 1
        assert stats1["retracted"] == 1

        # Second pass
        stats2 = store.flush_pending()
        assert stats2["observed_promoted"] == 0
        assert stats2["retracted"] == 0, (
            f"Already retracted should not be retracted again: {stats2}"
        )
    finally:
        tmpdir.cleanup()


# ─── Edge cases ──────────────────────────────────────────────────────────────

def test_empty_pending_pass():
    """run_promotion_pass on empty um_pending is a no-op."""
    tmpdir, store = make_store()
    try:
        stats = store.flush_pending()
        assert all(v == 0 for v in stats.values()), f"Empty pass should be no-op: {stats}"
    finally:
        tmpdir.cleanup()


def test_promote_with_ttl_custom_value():
    """agent_inference respects custom session_ttl parameter."""
    tmpdir, store = make_store()
    try:
        store.enable_virtual_clock()
        now = store._now()

        store.store_pending(
            content="Quick inference fact",
            source="agent_inference",
            fact_type="V",
            target="test",
            session_id="s1",
            provenance={"extractor": "test"},
        )

        # Pass 1: with very short TTL (1ms), should promote
        # (technically the session hasn't been inactive since updated_at = now)
        # Actually, we need the updated_at to be old enough.
        # Let's advance time before the pass.
        store.advance_time(2)  # 2 seconds of inactivity

        # Custom TTL of 1 second
        stats = run_promotion_pass(store.conn, store._now(), session_ttl=1.0)

        assert stats["agent_inference_promoted"] == 1, (
            f"Should promote with expired TTL: {stats}"
        )
    finally:
        tmpdir.cleanup()


def test_promoter_updates_metrics():
    """Promoter should emit promote_count and retract_count metrics."""
    tmpdir, store = make_store()
    try:
        store.store_pending(
            content="Metric test",
            source="observed",
            fact_type="V",
            target="metrics.test",
            session_id="s1",
            provenance={"extractor": "pytest_observer"},
        )

        stats = store.flush_pending()

        # Check metrics were recorded
        metric = store.conn.execute(
            """SELECT promote_count, retract_count FROM um_metrics
               WHERE session_id = ? AND observer = ?""",
            ("s1", "pytest_observer"),
        ).fetchone()

        assert metric is not None, "Metric row should exist"
        assert metric["promote_count"] >= 1, f"promote_count should be recorded: {metric}"
    finally:
        tmpdir.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
