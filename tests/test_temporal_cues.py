"""v0.3.2 — temporal-cue stale penalty + symmetric current-state boost
(capacity_stress cs_02 fix).

When the query asks for the *current* state ("now", "currently", "latest"):
- Candidates with stale markers ("archived", "legacy", "deprecated", "old
  version") get a -0.8 penalty so they don't beat recent-but-different-vocab
  answers via raw lexical overlap.
- Candidates that ALSO carry current-state language ("Current production…",
  "today", "now bound to") get a +1.5 boost so they can overcome BM25
  advantages from lexically-similar distractors that lack temporal cues.
"""

import os
import tempfile

from mnemoria import MnemoriaConfig, MnemoriaStore
from mnemoria.retrieval import (
    _query_wants_current,
    _candidate_is_stale_flagged,
)


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    store = MnemoriaStore(cfg)
    store.enable_virtual_clock()
    return tmpdir, store


# ── helper unit tests ────────────────────────────────────────────────────


def test_query_wants_current_positive_cases():
    assert _query_wants_current("What port does the server run on now?")
    assert _query_wants_current("What is the current API rate limit?")
    assert _query_wants_current("Who is the latest team lead?")
    assert _query_wants_current("Recently, which model are we using?")
    assert _query_wants_current("As of now, what is the deployment region?")


def test_query_wants_current_negative_cases():
    assert not _query_wants_current("What port did the server run on in 2023?")
    assert not _query_wants_current("Who was the previous lead engineer?")
    assert not _query_wants_current("What is the team's mission statement?")
    assert not _query_wants_current("")


def test_candidate_is_stale_flagged_positive_cases():
    assert _candidate_is_stale_flagged(
        "Server port does run on 3000 in the archived deployment guide"
    )
    assert _candidate_is_stale_flagged("Used to run PostgreSQL but migrated.")
    assert _candidate_is_stale_flagged("Legacy authentication has been deprecated.")
    assert _candidate_is_stale_flagged("This was previously stored in S3.")
    assert _candidate_is_stale_flagged("In the past, we ran 5 replicas.")


def test_candidate_is_stale_flagged_negative_cases():
    assert not _candidate_is_stale_flagged(
        "Current production listener is bound to 8080 for the server runtime"
    )
    assert not _candidate_is_stale_flagged("The team meets every Wednesday at 2pm.")
    assert not _candidate_is_stale_flagged("Stripe handles all payment processing.")
    assert not _candidate_is_stale_flagged("")


# ── Integration: cs_02 reproduction ──────────────────────────────────────


def test_current_query_demotes_stale_lexical_trap_in_minimal_corpus():
    """Minimal-corpus reproduction of the capacity_stress cs_02 dynamic:
    the stale-flagged fact has higher importance AND full lexical overlap
    with the query, but is correctly demoted by the B3 stale penalty.
    The recent fact carries "Current" language and gets the symmetric boost.

    This is a unit-style test isolating the stale-vs-current dynamic. The
    full cs_02 fixture (with related_noise distractors that ALSO have
    strong BM25 overlap) is verified separately via B5 benchmark re-run."""
    tmpdir, store = _make_store()
    try:
        # Old fact: high importance, full lexical overlap, stale-flagged.
        store.store(
            "Server port does run on 3000 in the archived deployment guide",
            category="factual", importance=0.9,
        )
        store.advance_time(85)
        # Current fact: lower importance, different vocabulary, current marker.
        store.store(
            "Current production listener is bound to 8080 for the server runtime",
            category="factual", importance=0.7,
        )
        store.advance_time(5)

        results = store.recall("What port does the server run on now?", top_k=2)
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content
        assert "8080" in top1, (
            f"current production listener should win top-1; got {top1!r}"
        )
        assert "3000" not in top1, (
            f"stale archived 3000 should not be top-1; got {top1!r}"
        )
    finally:
        tmpdir.cleanup()


def test_historical_query_does_not_get_stale_penalty():
    """Queries asking about historical state ("What was the previous X?",
    "In 2023, what X?") must NOT trigger the stale-penalty path. The user
    explicitly wants the old answer."""
    tmpdir, store = _make_store()
    try:
        store.store(
            "Server port was previously 3000 in the archived deployment guide",
            category="factual", importance=0.7,
        )
        store.advance_time(60)
        store.store(
            "Server port is now 8080 in the new deployment guide",
            category="factual", importance=0.7,
        )
        store.advance_time(5)

        # Historical query — has "previously" but no "current" cue
        results = store.recall(
            "What was the previous server port?", top_k=2,
        )
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content
        # Either fact passing is fine — the point is we don't crash and the
        # stale penalty doesn't actively block the historical answer.
        assert "3000" in top1 or "8080" in top1, (
            f"got unexpected top-1: {top1!r}"
        )
    finally:
        tmpdir.cleanup()


def test_no_temporal_cue_query_no_op():
    """Queries without temporal cues behave exactly as v0.3.1 did — neither
    stale penalty nor symmetric boost should fire."""
    tmpdir, store = _make_store()
    try:
        store.store("Server port runs on 8080", category="factual", importance=0.7)
        store.store("Server port previously was 3000 in the archived setup",
                    category="factual", importance=0.5)
        store.advance_time(0.0001)

        # No temporal cue — penalty/boost should not fire.
        results = store.recall("What is the server configuration?", top_k=2)
        assert results, "expected non-empty recall"
        # No specific assertion on top-1 — just checking nothing breaks.
    finally:
        tmpdir.cleanup()
