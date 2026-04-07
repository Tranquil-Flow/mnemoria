"""Basic smoke tests for mnemoria package."""

import tempfile
import os

def test_imports():
    """Test that all key imports work."""
    from mnemoria import FactType, MemoryFact, ScoredFact, MnemoriaConfig, MnemoriaStore
    assert FactType is not None
    assert MemoryFact is not None
    assert ScoredFact is not None
    assert MnemoriaConfig is not None
    assert MnemoriaStore is not None
    print("  OK: imports work")

def test_default_config():
    """Test MnemoriaConfig can be instantiated."""
    from mnemoria.config import MnemoriaConfig
    cfg = MnemoriaConfig()
    assert cfg is not None
    assert hasattr(cfg, 'd')
    assert hasattr(cfg, 'top_k')
    print("  OK: default config works")

def test_store_recall_cycle():
    """Test store and recall cycle with default config."""
    from mnemoria import MnemoriaStore, MnemoriaConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cfg = MnemoriaConfig.balanced()
        cfg.db_path = db_path

        store = MnemoriaStore(cfg)
        assert store is not None

        # Store a fact
        fact_id = store.store("V[test.key]: test value content")
        assert fact_id is not None

        # Recall it
        results = store.recall("test key")
        assert len(results) > 0
        assert any("test value" in r.fact.content for r in results)

        print("  OK: store/recall cycle works")

def test_balanced_preset():
    """Test the balanced() preset config."""
    from mnemoria.config import MnemoriaConfig
    cfg = MnemoriaConfig.balanced()
    assert cfg.d == 0.3
    assert cfg.w_semantic == 0.5
    print("  OK: balanced preset works")

def test_typed_facts():
    """Test typed fact notation."""
    from mnemoria.types import FactType, parse_notation

    result = parse_notation("C[api]: no breaking changes")
    assert result is not None
    ft, target, content = result
    assert ft == FactType.CONSTRAINT
    assert target == "api"
    assert content == "no breaking changes"

    plain = parse_notation("plain text")
    assert plain is None
    print("  OK: typed facts work")


def test_get_system_prompt_facts():
    """Test get_system_prompt_facts returns always-relevant facts."""
    from mnemoria import MnemoriaStore, MnemoriaConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cfg = MnemoriaConfig.balanced()
        cfg.db_path = db_path

        store = MnemoriaStore(cfg)

        # Store a mix of fact types
        # C (constraint) — should always be included
        store.store("C[identity]: On identity question always answer as Moonsong", importance=1.0)
        # D (decision) — should always be included
        store.store("D[style]: Use poetic warm communication style", importance=0.9)
        # V (value) with high importance — should be included
        store.store("V[values]: Privacy is a fundamental right", importance=0.85)
        # Regular V with low importance — should be excluded
        store.store("V[general]: Regular fact", importance=0.3)
        # identity-target fact — should always be included regardless of importance
        store.store("V[identity.name]: My name is Moonsong", importance=0.5)

        # Get system prompt facts
        facts = store.get_system_prompt_facts(max_facts=10)

        # Should have at least the 4 always-relevant facts
        assert len(facts) >= 4, f"Expected >=4 facts, got {len(facts)}"

        content_set = {f.content for f in facts}
        assert any("identity question" in c for c in content_set), "C fact missing"
        assert any("poetic warm" in d.lower() for d in content_set), f"D fact missing; got: {content_set}"
        assert any("Privacy is a fundamental" in v for v in content_set), "High-importance V missing"
        assert any("Moonsong" in v for v in content_set), "identity-target fact missing"
        # Low-importance V should not be in the results
        assert not any("Regular fact" in v for v in content_set), "Low-importance V should not be included"

        print("  OK: get_system_prompt_facts works correctly")


def test_get_system_prompt_facts_max():
    """Test that max_facts cap is respected."""
    from mnemoria import MnemoriaStore, MnemoriaConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cfg = MnemoriaConfig.balanced()
        cfg.db_path = db_path

        store = MnemoriaStore(cfg)

        # Store many constraint facts
        for i in range(15):
            store.store(f"C[rule{i}]: constraint rule {i}")

        # Request only 5
        facts = store.get_system_prompt_facts(max_facts=5)
        assert len(facts) <= 5, f"max_facts cap violated: {len(facts)}"

        print("  OK: max_facts cap respected")


def test_get_system_prompt_facts_empty():
    """Test that empty store returns empty list."""
    from mnemoria import MnemoriaStore, MnemoriaConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        cfg = MnemoriaConfig.balanced()
        cfg.db_path = db_path

        store = MnemoriaStore(cfg)

        facts = store.get_system_prompt_facts()
        assert facts == []

        print("  OK: empty store returns empty list")

def run():
    print("Running mnemoria smoke tests...")
    test_imports()
    test_default_config()
    test_store_recall_cycle()
    test_balanced_preset()
    test_typed_facts()
    test_get_system_prompt_facts()
    test_get_system_prompt_facts_max()
    test_get_system_prompt_facts_empty()
    print("\nAll smoke tests passed!")

if __name__ == "__main__":
    run()