"""
Unified Memory data model — FactType, MemoryFact, MemoryLink, ScoredFact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# FactType
# ---------------------------------------------------------------------------

class FactType(Enum):
    """Semantic category of a memory fact.

    Values are the single-char notation codes used in structured input.
    """
    CONSTRAINT = 'C'
    DECISION   = 'D'
    VALUE      = 'V'
    UNKNOWN    = '?'
    DONE       = 'done'
    OBSOLETE   = 'obs'


# ---------------------------------------------------------------------------
# Metabolic rates — higher = faster decay / cheaper to evict
# ---------------------------------------------------------------------------

METABOLIC_RATES: dict[FactType, float] = {
    FactType.CONSTRAINT: 0.3,
    FactType.DECISION:   0.7,
    FactType.VALUE:      1.0,
    FactType.UNKNOWN:    2.0,
    FactType.DONE:       2.5,
    FactType.OBSOLETE:   5.0,
}

# ---------------------------------------------------------------------------
# Notation helpers
# ---------------------------------------------------------------------------

FACT_TYPE_FROM_NOTATION: dict[str, FactType] = {
    'C': FactType.CONSTRAINT,
    'D': FactType.DECISION,
    'V': FactType.VALUE,
    '?': FactType.UNKNOWN,
    '✓': FactType.DONE,
    '~': FactType.OBSOLETE,
}

NOTATION_PATTERN: re.Pattern = re.compile(
    r'^(C|D|V|\?|✓|~)\[([^\]]+)\]:\s*(.+)$'
)


def parse_notation(raw: str) -> Optional[tuple[FactType, str, str]]:
    """Parse a structured notation string into (fact_type, target, content).

    Returns None for plain (unstructured) text so callers can treat it as a
    VALUE fact with target='general'.

    Examples
    --------
    >>> parse_notation("C[api]: no breaking changes")
    (FactType.CONSTRAINT, 'api', 'no breaking changes')
    >>> parse_notation("plain text") is None
    True
    """
    m = NOTATION_PATTERN.match(raw.strip())
    if m is None:
        return None
    notation_char, target, content = m.group(1), m.group(2), m.group(3)
    fact_type = FACT_TYPE_FROM_NOTATION[notation_char]
    return fact_type, target.strip(), content.strip()


# ---------------------------------------------------------------------------
# MemoryFact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryFact:
    """A single memory fact stored in the unified memory system.

    frozen=True keeps instances hashable and prevents accidental mutation;
    use dataclasses.replace() to produce updated copies.
    """
    id:              str
    content:         str
    embedding:       Optional[np.ndarray]

    fact_type:       FactType   = FactType.VALUE
    target:          str        = 'general'
    scope_id:        Optional[str] = None

    status:          str        = 'active'
    activation:      float      = 0.0
    q_value:         float      = 0.5
    access_count:    int        = 0
    metabolic_rate:  float      = 1.0

    created_at:      float      = 0.0
    updated_at:      float      = 0.0
    last_accessed:   float      = 0.0

    source_hash:     Optional[str] = None
    superseded_by:   Optional[str] = None

    importance:      float      = 0.5
    category:        Optional[str] = None
    access_times:    list[float] = field(default_factory=list)

    layer:           str        = 'working'
    pinned:          bool       = False

    class Config:
        # Allow numpy arrays in equality checks without raising
        arbitrary_types_allowed = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryFact):
            return NotImplemented
        # Compare everything except embeddings by identity, then id
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


# ---------------------------------------------------------------------------
# MemoryLink
# ---------------------------------------------------------------------------

@dataclass
class MemoryLink:
    """Directed weighted link between two memory facts."""
    source_id:          str
    target_id:          str
    strength:           float = 0.1
    npmi:               float = 0.0
    co_occurrence_count: int  = 0
    link_type:          str   = 'hebbian'
    last_updated:       float = 0.0


# ---------------------------------------------------------------------------
# ScoredFact
# ---------------------------------------------------------------------------

@dataclass
class ScoredFact:
    """A MemoryFact paired with its retrieval score and score breakdown."""
    fact:       MemoryFact
    score:      float
    components: dict = field(default_factory=dict)
