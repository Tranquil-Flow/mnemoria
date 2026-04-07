"""
Cognitive encoding — heuristic category classification and importance estimation.

No LLM calls. Pure pattern matching for speed. Every store() call runs through
this to auto-classify and score memories before they're persisted.

Categories:
  factual, preference, procedural, environment, episodic, semantic, causal

Importance: 0.0-1.0 float based on heuristic signals.
"""

from __future__ import annotations

import re
from typing import Tuple


# ─── Category Patterns ──────────────────────────────────────────────

# Order matters — first match wins. More specific patterns go first.
_CATEGORY_PATTERNS = [
    # preference — user likes/dislikes/prefers
    ("preference", [
        re.compile(r"\b(prefer|prefers|preferred|likes?|dislikes?|hates?|loves?|wants?|don'?t like|don'?t want|rather|favorite|favourite)\b", re.I),
        re.compile(r"\b(user|they|he|she|i)\s+(prefer|like|want|hate|love|dislike)", re.I),
        re.compile(r"\b(always use|never use|don'?t use|stop using|switch to)\b", re.I),
    ]),

    # procedural — how to do things
    ("procedural", [
        re.compile(r"\b(run|execute|install|command|script|steps?|how to|to do|procedure|workflow|recipe|guide)\b", re.I),
        re.compile(r"\b(pip|npm|apt|brew|docker|git|make|cargo|go build)\s", re.I),
        re.compile(r"```", re.I),  # code blocks
        re.compile(r"\b(first|then|next|finally|step \d)\b", re.I),
    ]),

    # environment — system/project config
    ("environment", [
        re.compile(r"\b(version|server|port|host|path|directory|folder|ip|url|endpoint|database|db)\b", re.I),
        re.compile(r"\b(running on|deployed|configured|installed at|located at|lives at)\b", re.I),
        re.compile(r"\b(macOS|Linux|Windows|Ubuntu|Debian|Docker|container)\b", re.I),
        re.compile(r"\b(Python|Node|Rust|Go|Java)\s+\d+\.\d+", re.I),
    ]),

    # causal — cause and effect
    ("causal", [
        re.compile(r"\b(because|causes?|caused by|leads? to|results? in|due to|therefore|consequently|if .+ then)\b", re.I),
        re.compile(r"\b(broke|breaks|broken|fixed by|solving|resolved by)\b", re.I),
    ]),

    # episodic — events, discussions, time-anchored
    ("episodic", [
        re.compile(r"\b(yesterday|last (week|time|session|month)|today|earlier|previously|we discussed|we decided|we agreed)\b", re.I),
        re.compile(r"\b(meeting|conversation|session|discussed|talked about|mentioned)\b", re.I),
    ]),

    # semantic — definitions, concepts, explanations
    ("semantic", [
        re.compile(r"\b(means|definition|concept|refers to|is a type of|also known as|i\.e\.|e\.g\.)\b", re.I),
        re.compile(r"\b(ACT-R|Hebbian|embedding|vector|neural|algorithm|architecture)\b", re.I),
    ]),

    # factual — default catch-all
    ("factual", [
        re.compile(r".*"),  # matches everything
    ]),
]


# ─── Importance Signals ──────────────────────────────────────────────

_IMPORTANCE_SIGNALS = [
    # High importance (0.85+)
    (0.90, [
        re.compile(r"\b(correc(t|tion|ted)|fix(ed)?|wrong|mistake|actually|update[ds]?)\b", re.I),
        re.compile(r"\b(CRITICAL|IMPORTANT|NEVER|ALWAYS|MUST|MUST NOT)\b"),
        re.compile(r"\b(remember this|don'?t forget|keep in mind)\b", re.I),
    ]),

    # Moderate-high importance (0.75)
    (0.75, [
        re.compile(r"\b(prefer|prefers|preferred|likes?|dislikes?|hates?)\b", re.I),
        re.compile(r"\b(name is|called|known as|goes by|my name)\b", re.I),
        re.compile(r"\b(password|key|token|secret|credential)\b", re.I),
        re.compile(r"\b(API|endpoint|database|production|deploy)\b", re.I),
    ]),

    # Moderate importance (0.65)
    (0.65, [
        re.compile(r"\b(install|configure|setup|command|run)\b", re.I),
        re.compile(r"\b(version|port|host|path|url)\b", re.I),
        re.compile(r"```"),  # code blocks
    ]),

    # Lower importance (0.45)
    (0.45, [
        re.compile(r"\b(observed|noticed|seems|appears|might)\b", re.I),
        re.compile(r"\b(generally|usually|sometimes|often)\b", re.I),
    ]),
]


def classify_category(content: str) -> str:
    """
    Classify memory content into a category using pattern matching.
    Returns one of: factual, preference, procedural, environment,
    episodic, semantic, causal.
    """
    for category, patterns in _CATEGORY_PATTERNS:
        for pattern in patterns:
            if pattern.search(content):
                return category
    return "factual"  # fallback


def estimate_importance(content: str, category: str = "factual") -> float:
    """
    Estimate importance score (0.0–1.0) based on content signals.
    
    Higher importance means the memory should be retained longer
    and weighted more in recall.
    """
    max_score = 0.45  # base importance

    for score, patterns in _IMPORTANCE_SIGNALS:
        for pattern in patterns:
            if pattern.search(content):
                max_score = max(max_score, score)
                break  # one match per tier is enough

    # Category-based adjustments
    category_boosts = {
        "preference": 0.10,
        "procedural": 0.05,
        "environment": 0.05,
        "causal": 0.05,
    }
    max_score += category_boosts.get(category, 0.0)

    # Length bonus — longer, more detailed memories are slightly more important
    word_count = len(content.split())
    if word_count > 30:
        max_score += 0.05
    elif word_count < 5:
        max_score -= 0.05

    return min(max(max_score, 0.1), 1.0)  # clamp to [0.1, 1.0]


def encode(content: str) -> Tuple[str, float]:
    """
    Full encoding pipeline: classify category and estimate importance.
    Returns (category, importance).
    """
    category = classify_category(content)
    importance = estimate_importance(content, category)
    return category, importance
