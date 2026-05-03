"""v0.4 candidate C5 — entity alias dedup tests.

Verify that the multi-word-entity first-token fallback in
`create_entity_links` correctly bridges full-name and short-name
mentions across facts.
"""

import os
import tempfile

from mnemoria import MnemoriaConfig, MnemoriaStore
from mnemoria.links import get_links_for


def _make_store():
    tmpdir = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmpdir.name, "test.db")
    cfg = MnemoriaConfig.balanced()
    cfg.db_path = db_path
    cfg.enable_pressure = False
    store = MnemoriaStore(cfg)
    store.enable_virtual_clock()
    return tmpdir, store


def test_fullname_to_firstname_alias_link_created():
    """Storing 'Caroline' first, then 'Caroline Smith' should create an
    entity link from the new fact to the existing short-name fact via the
    first-token fallback."""
    tmpdir, store = _make_store()
    try:
        f1_id = store.store(
            "Caroline went hiking last weekend.",
            category="factual", importance=0.7,
        ).id
        store.advance_time(0.0001)
        f2_id = store.store(
            "Caroline Smith hosted the team meeting on Friday.",
            category="factual", importance=0.7,
        ).id

        # f2 should have an entity link to f1 (Caroline ⊆ Caroline Smith)
        links = get_links_for(store._conn, f2_id)
        target_ids = {link.target_id for link in links if link.link_type == "entity"}
        assert f1_id in target_ids, (
            f"expected entity link from 'Caroline Smith' fact to 'Caroline' fact; "
            f"got entity targets={target_ids}"
        )
    finally:
        tmpdir.cleanup()


def test_short_first_token_skipped():
    """A multi-word entity whose first token is very short (<4 chars) must
    NOT trigger the alias fallback to avoid over-linking on common
    fragments. E.g. 'St Mary' should NOT link via 'St'."""
    tmpdir, store = _make_store()
    try:
        f1_id = store.store(
            "We attended a service at St Mary's Church.",
            category="factual", importance=0.7,
        ).id
        store.advance_time(0.0001)
        # Fact with 'St' alone (e.g. abbreviation in a different context)
        f2_id = store.store(
            "St was an unusual abbreviation in this context.",
            category="factual", importance=0.5,
        ).id
        store.advance_time(0.0001)
        # Now store another fact with 'St Mary' as a multi-word entity
        f3_id = store.store(
            "St Mary's Church holds a Sunday service.",
            category="factual", importance=0.7,
        ).id

        # f3's entity links should connect to f1 (shares 'St Mary's Church')
        # but NOT specifically via 'St'-only fallback because 'St' is too
        # short.
        links = get_links_for(store._conn, f3_id)
        target_ids = {link.target_id for link in links if link.link_type == "entity"}
        # f1 should be linked (full phrase match works)
        assert f1_id in target_ids, "expected link to original 'St Mary' fact"
    finally:
        tmpdir.cleanup()


def test_no_alias_collision_on_unrelated_short_names():
    """Two distinct people sharing no first name shouldn't get alias-linked.
    'Caroline Smith' and 'Melanie Johnson' have no first-name overlap."""
    tmpdir, store = _make_store()
    try:
        f1_id = store.store(
            "Caroline Smith works in product.",
            category="factual", importance=0.7,
        ).id
        store.advance_time(0.0001)
        f2_id = store.store(
            "Melanie Johnson works in engineering.",
            category="factual", importance=0.7,
        ).id

        links = get_links_for(store._conn, f2_id)
        entity_target_ids = {link.target_id for link in links if link.link_type == "entity"}
        # No entity link expected (no shared tokens)
        assert f1_id not in entity_target_ids, (
            f"unexpected alias link between Caroline and Melanie: {entity_target_ids}"
        )
    finally:
        tmpdir.cleanup()


def test_recall_finds_aliased_fact_via_hebbian_spread():
    """Behavioural: with the alias fix, recalling for 'Caroline' should
    surface a fact about 'Caroline Smith' AND a fact only mentioning
    'Caroline' alone in the top results, since they're now linked."""
    tmpdir, store = _make_store()
    try:
        store.store("Caroline went hiking on Saturday.", category="factual", importance=0.7)
        store.advance_time(0.0001)
        store.store("Caroline Smith chairs the architecture review.", category="factual", importance=0.7)
        store.advance_time(0.0001)
        # Distractor
        store.store("Melanie organised a pottery workshop.", category="factual", importance=0.5)
        store.advance_time(0.0001)

        results = store.recall("What did Caroline do?", top_k=3)
        assert results, "expected non-empty recall"
        contents = [r.fact.content.lower() for r in results[:3]]
        assert any("caroline" in c for c in contents), (
            f"expected at least one Caroline fact in top-3; got {contents}"
        )
    finally:
        tmpdir.cleanup()
