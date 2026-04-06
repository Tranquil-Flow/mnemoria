"""
Unified Memory System — cognitive + structured + self-optimizing memory for AI agents.

Merges ACT-R activation scoring, typed facts with metabolic decay, Hebbian
link formation, Q-value reinforcement learning, scope lifecycle management,
and LinUCB self-optimizing retrieval into a single SQLite-backed engine.

Usage:
    from mnemoria.store import UnifiedMemoryStore
    from mnemoria.config import UnifiedMemoryConfig

    store = UnifiedMemoryStore(UnifiedMemoryConfig.balanced())
    store.store("V[api.url]: https://example.com")
    results = store.recall("What is the API URL?")
"""

from mnemoria.types import FactType, MemoryFact, ScoredFact
from mnemoria.config import UnifiedMemoryConfig
from mnemoria.store import UnifiedMemoryStore

__all__ = ["FactType", "MemoryFact", "ScoredFact", "UnifiedMemoryConfig", "UnifiedMemoryStore"]

__version__ = "0.1.0"
