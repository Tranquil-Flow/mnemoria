"""
Mnemoria cognitive memory system — cognitive + structured + self-optimizing memory for AI agents.

Merges ACT-R activation scoring, typed facts with metabolic decay, Hebbian
link formation, Q-value reinforcement learning, scope lifecycle management,
and LinUCB self-optimizing retrieval into a single SQLite-backed engine.

Usage:
    from mnemoria.store import MnemoriaStore
    from mnemoria.config import MnemoriaConfig

    store = MnemoriaStore(MnemoriaConfig.balanced())
    store.store("V[api.url]: https://example.com")
    results = store.recall("What is the API URL?")
"""

from mnemoria.types import FactType, MemoryFact, ScoredFact
from mnemoria.config import MnemoriaConfig
from mnemoria.store import MnemoriaStore

__all__ = ["FactType", "MemoryFact", "ScoredFact", "MnemoriaConfig", "MnemoriaStore"]

__version__ = "0.2.1"
