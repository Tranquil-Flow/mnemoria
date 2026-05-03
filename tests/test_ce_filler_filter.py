"""Cross-encoder pool filtering tests for v0.3.1.

The filler classifier in `score_candidates` correctly penalises tone-matched
supportive turns, but the cross-encoder rerank in `Store.recall()` uses raw
query-text relevance and re-ranks fluent filler back into top-1 (the
"filler-leak"). The fix filters filler-flagged candidates out of the CE pool
before scoring, while falling back to the unfiltered pool when filtering would
shrink it below `cross_encoder_min_pool`.
"""

import os
import tempfile

from mnemoria import MnemoriaConfig, MnemoriaStore


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    store = MnemoriaStore(cfg)
    store.enable_virtual_clock()
    return tmpdir, store


def test_ce_pool_filters_filler_when_substantive_alternative_exists():
    """Filler turns must not climb back to top-1 via cross-encoder rerank
    when there's a substantive answer-bearing turn in the pool."""
    tmpdir, store = _make_store()
    try:
        # A LoCoMo-style mix: substantive Caroline turn that answers the
        # question, plus several tone-matched filler turns that the cross-
        # encoder is known to score highly on personality-trait queries.
        store.store(
            "Caroline: I've been working on becoming more thoughtful about "
            "my decisions lately, especially around the adoption process.",
            category="factual", importance=0.7,
        )
        store.advance_time(0.0001)
        store.store(
            "Melanie: That must have been tough for you, Caroline. Respect "
            "for finding acceptance and helping others with what you've "
            "been through. You're so strong and inspiring.",
            category="factual", importance=0.6,
        )
        store.advance_time(0.0001)
        store.store(
            "Melanie: I'm so happy for you, Caroline. You found your true "
            "self and now you're helping others.",
            category="factual", importance=0.6,
        )
        store.advance_time(0.0001)
        store.store(
            "Melanie: Wow, Caroline, that's awesome. Giving a home to "
            "needy kids is such a loving way to build a family.",
            category="factual", importance=0.6,
        )
        store.advance_time(0.0001)
        store.store(
            "Caroline: I value being authentic and driven in everything "
            "I do — those are traits Melanie often points out about me.",
            category="factual", importance=0.7,
        )

        results = store.recall("What personality traits does Caroline have?", top_k=3)
        assert results, "expected non-empty recall"
        top1 = results[0].fact.content
        # Top-1 must NOT be a flagged filler turn.
        filler_markers = (
            "That must have been tough",
            "Wow, Caroline, that's awesome",
            "I'm so happy for you",
        )
        assert not any(m in top1 for m in filler_markers), (
            f"filler leaked to top-1 despite CE pool filter: {top1!r}"
        )
    finally:
        tmpdir.cleanup()


def test_ce_pool_falls_back_when_filter_shrinks_below_minimum():
    """If filtering would leave fewer than cross_encoder_min_pool clean
    candidates, the original (unfiltered) pool is used so we never return
    fewer results than before. Validated by behaviour: recall returns
    results even when the pool is mostly filler."""
    tmpdir, store = _make_store()
    try:
        # All-filler corpus — no substantive alternatives. The fix must not
        # crash or return empty: it should fall back to the original pool.
        for content in [
            "Thanks!",
            "Wow, that's amazing!",
            "I'm so glad to hear that.",
            "That must have been so tough.",
            "Absolutely, you're so strong.",
            "Yeah, exactly!",
        ]:
            store.store(content, category="factual", importance=0.5)
            store.advance_time(0.0001)

        results = store.recall("How are you doing?", top_k=3)
        # Must not be empty — falling back to the unfiltered pool keeps recall
        # functional on filler-heavy corpora.
        assert results, "expected non-empty recall on filler-only corpus (fallback path)"
    finally:
        tmpdir.cleanup()


def test_ce_pool_unchanged_when_no_filler_present():
    """On corpora with no flagged filler, the CE pool filter is a no-op and
    recall should behave identically to v0.3.0."""
    tmpdir, store = _make_store()
    try:
        store.store(
            "The production database port is 5432.",
            category="factual", importance=0.7,
        )
        store.advance_time(0.0001)
        store.store(
            "The staging database port is 5433.",
            category="factual", importance=0.7,
        )
        store.advance_time(0.0001)
        store.store(
            "API rate limits are 1000 requests per minute.",
            category="factual", importance=0.7,
        )
        store.advance_time(0.0001)
        store.store(
            "JWT signing keys rotate every 24 hours.",
            category="factual", importance=0.7,
        )
        store.advance_time(0.0001)
        store.store(
            "WebSocket heartbeats are sent every 30 seconds.",
            category="factual", importance=0.7,
        )

        results = store.recall("What port does the production database use?", top_k=3)
        assert results, "expected non-empty recall on substantive-only corpus"
        # Top-1 should reference production database (5432), not staging.
        top1 = results[0].fact.content
        assert "5432" in top1 or "production" in top1.lower(), (
            f"expected production database fact at top-1, got {top1!r}"
        )
    finally:
        tmpdir.cleanup()
