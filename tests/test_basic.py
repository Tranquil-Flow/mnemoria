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

def run():
    print("Running mnemoria smoke tests...")
    test_imports()
    test_default_config()
    test_store_recall_cycle()
    test_balanced_preset()
    test_typed_facts()
    print("\nAll smoke tests passed!")

if __name__ == "__main__":
    run()