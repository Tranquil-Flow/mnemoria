"""Targeted forgetting tests for privacy-preserving erasure."""

import os
import tempfile

from mnemoria import MnemoriaConfig, MnemoriaStore


def _store_for_test(tmpdir: str) -> MnemoriaStore:
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = os.path.join(tmpdir, "mnemoria.db")
    cfg.embedding_model = "tfidf"
    cfg.enable_qvalue_reranking = True
    return MnemoriaStore(cfg)


def test_forget_removes_fact_retrieval_indexes_and_related_rows_without_leaking_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _store_for_test(tmpdir)
        secret = "the recovery phrase is velvet moon cactus"
        kept = store.store("V[public.note]: harmless public note", importance=0.9)
        forgotten = store.store(f"V[private.secret]: {secret}", importance=1.0)

        # Create all relationship rows this API must clean up.
        store.simulate_access("velvet moon cactus")
        store.conn.execute(
            "INSERT OR REPLACE INTO um_links (source_id, target_id, strength, link_type) VALUES (?, ?, 0.8, 'test')",
            (forgotten, kept),
        )
        store.conn.execute(
            "INSERT OR REPLACE INTO um_links (source_id, target_id, strength, link_type) VALUES (?, ?, 0.7, 'test')",
            (kept, forgotten),
        )
        store.conn.execute(
            "INSERT INTO um_qvalues (memory_id, q_value, update_count) VALUES (?, 0.9, 1)",
            (forgotten,),
        )
        pending_id = store.store_pending(
            secret,
            source="user_stated",
            session_id="privacy-test",
            provenance={"note": "sensitive fixture"},
        )
        store.conn.execute(
            "UPDATE um_pending SET promoted_to = ? WHERE id = ?",
            (forgotten, pending_id),
        )
        store.conn.commit()
        store.reward_memory(forgotten, 1.0)

        assert any(r.fact.id == forgotten for r in store.recall("velvet moon cactus", top_k=5))
        assert store.conn.execute(
            "SELECT rowid FROM um_facts_fts WHERE um_facts_fts MATCH ?",
            ('"velvet moon cactus"',),
        ).fetchall()

        receipt = store.forget(forgotten, reason="user requested erasure")

        assert receipt["deleted"] is True
        assert receipt["fact_id"] == forgotten
        assert receipt["content_hash"]
        assert secret not in repr(receipt)

        assert store.conn.execute("SELECT * FROM um_facts WHERE id = ?", (forgotten,)).fetchone() is None
        assert store.conn.execute("SELECT * FROM um_access_times WHERE fact_id = ?", (forgotten,)).fetchall() == []
        assert store.conn.execute(
            "SELECT * FROM um_links WHERE source_id = ? OR target_id = ?",
            (forgotten, forgotten),
        ).fetchall() == []
        assert store.conn.execute("SELECT * FROM um_qvalues WHERE memory_id = ?", (forgotten,)).fetchall() == []
        assert store.conn.execute(
            "SELECT * FROM um_pending WHERE promoted_to = ? OR content LIKE ?",
            (forgotten, "%velvet moon cactus%"),
        ).fetchall() == []
        assert store.conn.execute(
            "SELECT rowid FROM um_facts_fts WHERE um_facts_fts MATCH ?",
            ('"velvet moon cactus"',),
        ).fetchall() == []
        assert all(r.fact.id != forgotten and "velvet moon cactus" not in r.fact.content for r in store.recall("velvet moon cactus", top_k=5))

        q_rows = store._qvalue_store._conn.execute(
            "SELECT * FROM memory_qvalues WHERE memory_id = ?",
            (forgotten,),
        ).fetchall()
        assert q_rows == []

        log_row = store.conn.execute(
            "SELECT fact_id, content_hash, reason FROM um_deletion_log WHERE fact_id = ?",
            (forgotten,),
        ).fetchone()
        assert log_row is not None
        assert log_row["content_hash"] == receipt["content_hash"]
        assert log_row["reason"] == "user requested erasure"
        assert secret not in " ".join(str(v) for v in dict(log_row).values())


def test_forget_by_content_erases_matching_facts_and_returns_non_leaking_receipts():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _store_for_test(tmpdir)
        first = store.store("V[private.one]: api token alpha-forget-me")
        second = store.store("V[private.two]: api token beta-forget-me")
        kept = store.store("V[public]: keep this harmless memory")

        receipts = store.forget_by_content("forget-me", reason="rotated credentials")

        assert {r["fact_id"] for r in receipts} == {first, second}
        assert all(r["deleted"] is True for r in receipts)
        assert "alpha-forget-me" not in repr(receipts)
        assert "beta-forget-me" not in repr(receipts)
        assert store.conn.execute("SELECT * FROM um_facts WHERE id IN (?, ?)", (first, second)).fetchall() == []
        assert store.conn.execute("SELECT * FROM um_facts WHERE id = ?", (kept,)).fetchone() is not None
        assert all("forget-me" not in r.fact.content for r in store.recall("forget-me", top_k=10))


def test_forget_missing_fact_returns_false_receipt():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _store_for_test(tmpdir)

        receipt = store.forget("missing-id", reason="already gone")

        assert receipt == {
            "fact_id": "missing-id",
            "deleted": False,
            "reason": "already gone",
        }


def test_forget_by_content_empty_pattern_is_safe_noop():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = _store_for_test(tmpdir)
        kept = store.store("V[public]: keep this memory")

        receipts = store.forget_by_content("", reason="empty query")

        assert receipts == []
        assert store.conn.execute("SELECT * FROM um_facts WHERE id = ?", (kept,)).fetchone() is not None
