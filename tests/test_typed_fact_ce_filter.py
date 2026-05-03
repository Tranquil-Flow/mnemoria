"""v0.3.2 — typed-fact-aware cross-encoder pool exclusion (ss_h02 fix).

When the query terms overlap with a typed fact's target, untyped distractors
should not be allowed to compete in the CE rerank — the typed fact is the
authoritative answer for that key. See `filter_pool_for_typed_match` in
`mnemoria/retrieval.py`.

The unit-level filter test (no LLM, no embeddings) lives at the bottom; the
integration tests above exercise the full Store.recall() pipeline.
"""

import os
import tempfile

from mnemoria import MnemoriaConfig, MnemoriaStore
from mnemoria.retrieval import filter_pool_for_typed_match


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    store = MnemoriaStore(cfg)
    store.enable_virtual_clock()
    return tmpdir, store


# ── Integration: Store.recall() with typed + untyped facts ──────────────


def test_typed_fact_beats_untyped_distractor_via_ce_filter():
    """Reproduces the ss_h02 supersession failure case. With the v0.3.2 CE
    pool filter, the typed `D[frontend.framework]: …` answer must beat the
    untyped distractor `"TypeScript is required"` for the framework query."""
    tmpdir, store = _make_store()
    try:
        store.store(
            "D[frontend.framework]: Angular 12",
            category="factual", importance=0.7,
        )
        store.advance_time(30)
        store.store(
            "D[frontend.framework]: React 18 with Next.js",
            category="factual", importance=0.7,
        )
        # Substantive distractors that have nothing to do with framework choice
        # but live in the same project domain.
        for distractor in [
            "TypeScript is required",
            "Component library is internal",
            "Testing with Jest + RTL",
            "Build with Webpack 5",
            "E2E with Playwright",
        ]:
            store.store(distractor, category="factual", importance=0.5)
        store.advance_time(5)

        results = store.recall("What frontend framework do we use?", top_k=3)
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content
        assert "React 18 with Next.js" in top1, (
            f"typed fact should win after CE filter, got top-1={top1!r}"
        )
    finally:
        tmpdir.cleanup()


def test_filter_no_op_when_query_does_not_match_any_typed_target():
    """When no typed fact's target overlaps with query terms, the filter must
    be a no-op: untyped substantive answers should still rank normally."""
    tmpdir, store = _make_store()
    try:
        # A typed fact with target unrelated to the query.
        store.store("V[db.port]: 5432", category="factual", importance=0.7)
        # Plain untyped substantive answer that should win the unrelated query.
        store.store(
            "The conference room is on the third floor of building B.",
            category="factual", importance=0.7,
        )
        store.store(
            "Building B opened in 2019 and houses the engineering teams.",
            category="factual", importance=0.6,
        )
        store.advance_time(0.0001)

        results = store.recall("Where is the conference room?", top_k=3)
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content
        # The conference-room fact should win because the typed `db.port` fact
        # has no target overlap with `where conference room`.
        assert "conference room" in top1.lower(), (
            f"untyped substantive answer should win when no typed match — "
            f"got top-1={top1!r}"
        )
    finally:
        tmpdir.cleanup()


def test_filter_preserves_recall_on_pure_locomo_style_corpus():
    """LoCoMo conversation turns are all stored as plain text without typed
    notation, so the typed-fact filter must NOT trigger and must not regress
    open-domain conversational recall."""
    tmpdir, store = _make_store()
    try:
        for content, importance in [
            ("Caroline: I love painting and have been working on a sunset piece.", 0.7),
            ("Melanie: That's wonderful! What inspired the sunset theme?", 0.5),
            ("Caroline: I went hiking last weekend and the colors stayed with me.", 0.7),
            ("Caroline: I make abstract art that explores identity themes.", 0.7),
            ("Melanie: Tell me more about your process.", 0.5),
        ]:
            store.store(content, category="factual", importance=importance)
            store.advance_time(0.0001)

        results = store.recall("What kind of art does Caroline make?", top_k=3)
        assert results, "expected non-empty recall"
        # No typed facts in pool → filter is no-op → normal recall behaviour.
        # Top-1 must be one of the substantive Caroline statements.
        top_contents = [r.fact.content.lower() for r in results[:3]]
        assert any("abstract" in c or "painting" in c for c in top_contents), (
            f"expected art-related Caroline turn in top-3, got {top_contents}"
        )
    finally:
        tmpdir.cleanup()


# ── Unit tests on filter_pool_for_typed_match (no Store, no embeddings) ──


def test_filter_unit_returns_pool_unchanged_when_no_typed_facts():
    """Pure-untyped pool: filter is a no-op."""
    from mnemoria.types import MemoryFact, ScoredFact, FactType

    pool = [
        ScoredFact(
            fact=MemoryFact(id=str(i), content=f"plain fact {i}", embedding=None,
                            fact_type=FactType.VALUE, target="general"),
            score=1.0 - i * 0.01,
        )
        for i in range(5)
    ]
    out = filter_pool_for_typed_match(pool, "what is the database port?")
    assert out is pool or out == pool, "should return the unchanged pool"


def test_filter_unit_drops_untyped_when_typed_target_matches():
    """Mixed pool with a typed match: untyped facts dropped, typed kept."""
    from mnemoria.types import MemoryFact, ScoredFact, FactType

    pool = [
        ScoredFact(
            fact=MemoryFact(id="t1", content="React 18 with Next.js", embedding=None,
                            fact_type=FactType.DECISION, target="frontend.framework"),
            score=0.9,
        ),
        ScoredFact(
            fact=MemoryFact(id="u1", content="TypeScript is required", embedding=None,
                            fact_type=FactType.VALUE, target="general"),
            score=0.95,
        ),
        ScoredFact(
            fact=MemoryFact(id="u2", content="Component library is internal", embedding=None,
                            fact_type=FactType.VALUE, target="general"),
            score=0.85,
        ),
        ScoredFact(
            fact=MemoryFact(id="t2", content="Angular 12", embedding=None,
                            fact_type=FactType.DECISION, target="frontend.framework"),
            score=0.7,
        ),
    ]
    out = filter_pool_for_typed_match(pool, "What frontend framework do we use?")
    out_ids = {s.fact.id for s in out}
    assert out_ids == {"t1", "t2"}, (
        f"expected only typed facts, got {out_ids}"
    )


def test_filter_unit_no_op_when_typed_targets_dont_match_query():
    """Typed facts present but their targets don't overlap with query terms."""
    from mnemoria.types import MemoryFact, ScoredFact, FactType

    pool = [
        ScoredFact(
            fact=MemoryFact(id="t1", content="5432", embedding=None,
                            fact_type=FactType.VALUE, target="db.port"),
            score=0.9,
        ),
        ScoredFact(
            fact=MemoryFact(id="u1", content="The conference room is on the third floor", embedding=None,
                            fact_type=FactType.VALUE, target="general"),
            score=0.95,
        ),
    ]
    out = filter_pool_for_typed_match(pool, "Where is the conference room?")
    out_ids = {s.fact.id for s in out}
    assert out_ids == {"t1", "u1"}, (
        f"no typed-target overlap → no filter; got {out_ids}"
    )


def test_filter_unit_handles_dotted_target():
    """Targets like `db.port` must split into {db, port} for the overlap check."""
    from mnemoria.types import MemoryFact, ScoredFact, FactType

    pool = [
        ScoredFact(
            fact=MemoryFact(id="t1", content="5433", embedding=None,
                            fact_type=FactType.VALUE, target="db.port"),
            score=0.5,
        ),
        ScoredFact(
            fact=MemoryFact(id="u1", content="Database uses PostgreSQL", embedding=None,
                            fact_type=FactType.VALUE, target="general"),
            score=0.9,
        ),
    ]
    out = filter_pool_for_typed_match(pool, "What port is the database on?")
    out_ids = {s.fact.id for s in out}
    assert out_ids == {"t1"}, (
        f"dotted target {{db, port}} should overlap with query terms; got {out_ids}"
    )
