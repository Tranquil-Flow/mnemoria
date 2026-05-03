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


def test_fullname_to_firstname_creates_some_link():
    """Behavioural: storing 'Caroline' first, then 'Caroline Smith' should
    result in *some* link between the two facts. (Pre-existing
    `create_entity_links` skips a target if a temporal link to it already
    exists from the same source — so the link may materialize as
    `temporal` instead of `entity`. C5's contribution is that the
    multi-word entity's first token is now ALSO checked, expanding the
    set of facts considered for linking.)"""
    tmpdir, store = _make_store()
    try:
        f1_id = store.store(
            "Caroline went hiking last weekend.",
            category="factual", importance=0.7,
        )
        store.advance_time(0.0001)
        f2_id = store.store(
            "Caroline Smith hosted the team meeting on Friday.",
            category="factual", importance=0.7,
        )

        links = get_links_for(store._conn, f2_id)
        target_ids = {link.target_id for link in links}
        assert f1_id in target_ids, (
            f"expected at least one link from 'Caroline Smith' fact to "
            f"'Caroline' fact; got targets={target_ids}"
        )
    finally:
        tmpdir.cleanup()


def test_no_alias_collision_on_unrelated_short_names():
    """Two distinct people sharing no first name shouldn't get alias-linked."""
    tmpdir, store = _make_store()
    try:
        f1_id = store.store(
            "Caroline Smith works in product.",
            category="factual", importance=0.7,
        )
        store.advance_time(0.0001)
        f2_id = store.store(
            "Melanie Johnson works in engineering.",
            category="factual", importance=0.7,
        )

        links = get_links_for(store._conn, f2_id)
        entity_target_ids = {link.target_id for link in links if link.link_type == "entity"}
        assert f1_id not in entity_target_ids, (
            f"unexpected alias link between Caroline and Melanie: {entity_target_ids}"
        )
    finally:
        tmpdir.cleanup()


def test_recall_finds_aliased_fact_via_hebbian_spread():
    """Behavioural: with the alias fix, recalling for 'Caroline' should
    surface a Caroline-mentioning fact in the top results."""
    tmpdir, store = _make_store()
    try:
        store.store("Caroline went hiking on Saturday.", category="factual", importance=0.7)
        store.advance_time(0.0001)
        store.store("Caroline Smith chairs the architecture review.", category="factual", importance=0.7)
        store.advance_time(0.0001)
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
