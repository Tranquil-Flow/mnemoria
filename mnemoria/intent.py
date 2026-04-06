"""
Query Intent Classification for the Unified Memory System.

Classifies queries into intent types and shifts retrieval signal weights
accordingly. Ported from Ori-Mnemos classify.ts.

Intents:
  - episodic: "when did we...", "last time...", "yesterday..."
  - procedural: "how do I...", "steps to...", "process for..."
  - semantic: "what is...", "define...", "explain..."
  - decision: "what did we decide...", "why did we choose..."
  - value: "what is the URL/port/address..."
  - constraint: "what rule/requirement/constraint..."
"""

from __future__ import annotations

import re
from typing import Optional

from mnemoria.types import FactType


class QueryIntent:
    """Classified query intent with signal weight adjustments."""

    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"
    DECISION = "decision"
    VALUE = "value"
    CONSTRAINT = "constraint"
    GENERAL = "general"

    def __init__(self, intent: str, confidence: float = 0.5):
        self.intent = intent
        self.confidence = confidence

    @property
    def type_boost(self) -> Optional[FactType]:
        """Which fact type to boost for this intent."""
        return {
            self.VALUE: FactType.VALUE,
            self.CONSTRAINT: FactType.CONSTRAINT,
            self.DECISION: FactType.DECISION,
        }.get(self.intent)

    @property
    def q_lambda_shift(self) -> float:
        """How much to shift the Q-value lambda for this intent.
        Positive = trust Q-values more, negative = trust activation more.
        """
        return {
            self.SEMANTIC: -0.10,
            self.PROCEDURAL: 0.15,
            self.DECISION: 0.05,
            self.EPISODIC: -0.05,
        }.get(self.intent, 0.0)

    @property
    def category_boosts(self) -> set:
        """Which categories to boost via resolution boost."""
        return {
            self.PROCEDURAL: {"procedural", "causal"},
            self.DECISION: {"decision"},
            self.EPISODIC: set(),
            self.SEMANTIC: set(),
        }.get(self.intent, set())


# ─── Intent Classification Patterns ───────────────────────────

_PATTERNS = [
    # Value queries — concrete values, URLs, ports, addresses
    (QueryIntent.VALUE, [
        r"\b(?:what|which)\s+(?:is\s+the\s+)?(?:url|port|address|endpoint|host|ip|version|number|size|limit|timeout|password|key|token|secret|path|location)\b",
        r"\bwhat\s+(?:url|port|address|endpoint)\b",
        r"\b(?:url|port|address|endpoint|host)\s+(?:for|of|is)\b",
    ], 0.7),

    # Constraint queries — rules, requirements, limitations
    (QueryIntent.CONSTRAINT, [
        r"\b(?:what|which)\s+(?:is\s+the\s+)?(?:rule|constraint|requirement|limit|restriction|policy|guideline|standard|convention)\b",
        r"\b(?:must|should|required|mandatory|forbidden|prohibited|not\s+allowed)\b.*\?",
        r"\bwhat\s+(?:are\s+the\s+)?(?:rules|constraints|requirements|policies)\b",
    ], 0.7),

    # Decision queries — choices, rationale
    (QueryIntent.DECISION, [
        r"\b(?:what|why)\s+did\s+(?:we|they|you)\s+(?:decide|choose|pick|select|opt)\b",
        r"\bwhy\s+(?:did|do|does|are|is)\s+(?:we|they|you|it)\b",
        r"\bwhat\s+(?:was\s+the\s+)?(?:decision|rationale|reason)\b",
        r"\bwhy\s+(?:was|were|is)\b",
    ], 0.7),

    # Procedural queries — how to, steps, process
    (QueryIntent.PROCEDURAL, [
        r"\bhow\s+(?:do|does|can|should|to|would)\b",
        r"\b(?:steps?|process|procedure|workflow|guide|tutorial)\s+(?:for|to)\b",
        r"\bhow\s+(?:is|are|was|were)\s+\w+\s+(?:done|handled|managed|configured|deployed)\b",
    ], 0.6),

    # Episodic queries — time-specific, events
    (QueryIntent.EPISODIC, [
        r"\b(?:when|last\s+time|yesterday|today|recently|previously)\b",
        r"\bwhat\s+(?:happened|changed|was\s+(?:done|decided))\b",
        r"\bhow\s+(?:long|recently|often)\b",
    ], 0.5),

    # Semantic queries — definitions, explanations
    (QueryIntent.SEMANTIC, [
        r"\b(?:what\s+is|define|explain|describe|meaning\s+of)\b",
        r"\b(?:what\s+does|what\s+are)\b",
    ], 0.4),
]


def classify_intent(query: str) -> QueryIntent:
    """Classify a query's intent using regex patterns.

    Returns the highest-confidence matching intent, or GENERAL.
    """
    query_lower = query.lower().strip()
    best_intent = QueryIntent.GENERAL
    best_confidence = 0.0

    for intent_type, patterns, base_confidence in _PATTERNS:
        for pattern in patterns:
            if re.search(pattern, query_lower):
                if base_confidence > best_confidence:
                    best_intent = intent_type
                    best_confidence = base_confidence
                break  # Don't need to check more patterns for same intent

    return QueryIntent(best_intent, best_confidence)
