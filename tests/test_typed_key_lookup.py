"""v0.3.2 — typed-key query lookup boost (ss_e02 fix).

When the query is a key-lookup shape (e.g. "What X?", "Where is X?") and
overlaps with a typed fact's target, the target_match_boost in
score_candidates uses bumped weights so the typed answer reliably beats
substantive untyped distractors with strong embedding/FTS5 similarity.

Plus: the FTS5 strong-match override at store.py is suppressed when its top
candidate is untyped and a typed competitor exists with target overlap —
otherwise the override hands the untyped distractor a +50% score boost.
"""

import os
import tempfile

from mnemoria import MnemoriaConfig, MnemoriaStore
from mnemoria.retrieval import _is_key_lookup_query


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    store = MnemoriaStore(cfg)
    store.enable_virtual_clock()
    return tmpdir, store


# ── _is_key_lookup_query unit tests ────────────────────────────────────


def test_key_lookup_what_is_simple_question():
    assert _is_key_lookup_query("What port is the database on?")
    assert _is_key_lookup_query("What is the API rate limit?")
    assert _is_key_lookup_query("Where is the staging server?")
    assert _is_key_lookup_query("Which framework do we use?")
    assert _is_key_lookup_query("How many days does the backup retain?")


def test_key_lookup_rejects_long_sentence_questions():
    # Sentence-shaped LoCoMo open-domain questions should NOT trigger.
    assert not _is_key_lookup_query(
        "What did Caroline say about her family during the conversation last week?"
    )
    assert not _is_key_lookup_query(
        "Which of Melanie's paintings did she describe in the most detail to Caroline?"
    )


def test_key_lookup_rejects_non_question():
    assert not _is_key_lookup_query("What port is the database on")  # no question mark
    assert not _is_key_lookup_query("Tell me about the database port.")  # not what/which/where
    assert not _is_key_lookup_query("")
    assert not _is_key_lookup_query("?")


# ── Integration: ss_e02 reproduction ────────────────────────────────────


def test_typed_db_port_beats_untyped_postgres_distractor():
    """ss_e02 reproduction. Typed `V[db.port]: 5433` must rank above untyped
    distractor `"Database uses PostgreSQL"` for the query "What port is the
    database on?". Distractor wins on activation/FTS5 alone — needs B2's
    bumped target boost AND the strong-match override suppression to pass."""
    tmpdir, store = _make_store()
    try:
        store.store("V[db.port]: 5432", category="factual", importance=0.7)
        store.advance_time(30)
        store.store("V[db.port]: 5433", category="factual", importance=0.7)
        store.store(
            "Database uses PostgreSQL",
            category="factual", importance=0.5,
        )
        store.store(
            "Backups run nightly",
            category="factual", importance=0.5,
        )
        store.advance_time(5)

        results = store.recall("What port is the database on?", top_k=3)
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content
        assert "5433" in top1, (
            f"typed V[db.port]: 5433 should win on key-lookup query; "
            f"got top-1={top1!r}"
        )
    finally:
        tmpdir.cleanup()


def test_open_domain_query_does_not_get_bumped_boost():
    """Sentence-shaped open-domain queries must NOT trigger the bumped boost
    — that would over-promote typed facts on questions about narrative
    content. Mirror of the LoCoMo-style test in test_typed_fact_ce_filter.
    """
    tmpdir, store = _make_store()
    try:
        # A typed fact that shares a word with a long sentence-shaped query.
        store.store("V[db.port]: 5432", category="factual", importance=0.7)
        store.store(
            "Caroline mentioned the database backup plan during yesterday's standup; "
            "she's worried about restoration time after the production incident.",
            category="factual", importance=0.7,
        )
        store.advance_time(0.0001)

        # Long sentence-shaped query — _is_key_lookup_query should return False
        # (>=12 words rule), so the bumped boost doesn't fire.
        results = store.recall(
            "What did Caroline say about the database backup plan during yesterday's "
            "standup that worried her?",
            top_k=2,
        )
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content
        # The substantive narrative answer should win, not the typed `5432` value.
        assert "Caroline" in top1, (
            f"narrative answer should win on long sentence-shaped query; "
            f"got top-1={top1!r}"
        )
    finally:
        tmpdir.cleanup()


def test_strong_match_override_still_fires_for_typed_top_fts5():
    """The strong-match override must STILL fire when the dominant FTS5 fact
    IS a typed fact. Regression check on the override's original purpose.
    """
    tmpdir, store = _make_store()
    try:
        # Distinct keyword that only exists in one typed fact.
        store.store(
            "V[secret.zorblax]: zorblax-token-xyz123",
            category="factual", importance=0.7,
        )
        store.store(
            "Some unrelated content about other things.",
            category="factual", importance=0.5,
        )
        store.advance_time(0.0001)

        results = store.recall("What is the zorblax value?", top_k=2)
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content
        assert "zorblax-token-xyz123" in top1, (
            f"typed fact with unique keyword should win; got top-1={top1!r}"
        )
    finally:
        tmpdir.cleanup()
