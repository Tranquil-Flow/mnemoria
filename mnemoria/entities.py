"""Minimal proper-noun extraction for retrieval and link formation.

Used by:
  * write path (B2): cross-session links between facts mentioning the same entity
  * read path (B5):  entity-seeded PPR over the link graph

This is a heuristic, not a full NER. It picks up capitalized tokens and
multi-word capitalized phrases, then strips out a small set of false-positive
sentence starters ("I", "The", "Thanks!"). For LoCoMo / LongMemEval style
conversational corpora it captures person names (Caroline, Melanie) and
acronyms (LGBTQ) without requiring a model dependency.

Upgrade path: swap ``extract_entities`` for ``gliner-small`` or an LLM call
once the verification slice shows the heuristic is the limiting factor.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set


# False-positive guard: capitalized tokens that show up at sentence starts
# but aren't entities. Kept short — over-pruning hides real entity matches.
_SENTENCE_STARTERS: frozenset[str] = frozenset({
    "I", "The", "A", "An", "We", "You", "He", "She", "It", "They",
    "This", "That", "These", "Those", "There", "Their",
    "What", "When", "Where", "Why", "How", "Who", "Which",
    "Yes", "No", "Maybe", "Sure", "Okay", "OK",
    "Hi", "Hello", "Hey", "Yeah", "Yep", "Nope",
    "Thanks", "Wow", "Oh", "Ah", "Cool", "Great", "Nice", "Sorry",
    "Please", "Just", "Actually", "Really", "Probably", "Definitely",
    "Congrats", "Awesome", "Amazing",
    "Mr", "Mrs", "Ms", "Dr",
})


# Two patterns: (1) all-caps acronyms of length >= 2 (LGBTQ, USA, MFA),
# (2) Title-case multi-word phrases (Caroline Smith, New York City).
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_TITLE_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def extract_entities(text: str) -> List[str]:
    """Return lower-cased entity strings, deduplicated, in stable order.

    Order is deterministic: first the acronyms in source order, then the
    title-case phrases in source order. Duplicates (case-insensitive) keep
    only the first occurrence.
    """
    if not text:
        return []
    seen: Set[str] = set()
    out: List[str] = []

    def _add(token: str) -> None:
        if token in _SENTENCE_STARTERS:
            return
        key = token.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(key)

    for m in _ACRONYM.finditer(text):
        _add(m.group(0))
    for m in _TITLE_PHRASE.finditer(text):
        _add(m.group(0))
    return out


def entity_overlap(query_entities: Iterable[str], fact_text: str) -> int:
    """Count how many of ``query_entities`` (already lower-cased) appear in
    ``fact_text``. Substring match — a heuristic, but fast and tolerant of
    inflection and punctuation."""
    if not fact_text:
        return 0
    haystack = fact_text.lower()
    n = 0
    for ent in query_entities:
        if ent and ent in haystack:
            n += 1
    return n
