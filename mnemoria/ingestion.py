"""
Conversation Ingestion for the Mnemoria cognitive memory system.

Automatically extracts factual claims from conversation turns and stores
them as typed facts. This is the "perception" layer — converting raw
dialogue into structured memory.

Strategies:
  1. Pattern extraction — regex-based extraction of factual statements
  2. Semantic deduplication — detect near-duplicate facts and merge them
  3. Fact scoring — rate how "memorable" a statement is
"""

from __future__ import annotations

import re
import hashlib
import math
from typing import List, Optional, Tuple, Dict

from mnemoria.types import FactType


# ─── Fact Extraction ──────────────────────────────────────────


def extract_facts(text: str) -> List[Dict]:
    """Extract factual claims from a text block.

    Returns list of dicts with keys:
      content: str — the extracted fact
      fact_type: FactType — inferred type
      target: str — inferred target identifier
      confidence: float — extraction confidence (0-1)
    """
    facts = []

    # Split into sentences
    sentences = _split_sentences(text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15 or len(sentence) > 400:
            continue

        # Score how "factual" this sentence is
        score = _factuality_score(sentence)
        if score < 0.3:
            continue

        # Classify the fact type
        fact_type = _classify_type(sentence)
        target = _extract_target(sentence)

        facts.append({
            "content": sentence,
            "fact_type": fact_type,
            "target": target,
            "confidence": score,
        })

    return facts


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling common abbreviations."""
    # Simple sentence splitter — handles periods, question marks, newlines
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


def _factuality_score(sentence: str) -> float:
    """Rate how "factual" and worth remembering a sentence is.

    High scores: concrete facts, decisions, values, rules
    Low scores: greetings, filler, opinions, meta-conversation
    """
    score = 0.5  # baseline
    lower = sentence.lower()

    # Boost: contains specific values (numbers, URLs, versions)
    if re.search(r'\d+(?:\.\d+)?', sentence):
        score += 0.1  # numbers
    if re.search(r'https?://\S+', sentence):
        score += 0.15  # URLs
    if re.search(r'v\d+\.\d+', lower):
        score += 0.1  # version numbers
    if re.search(r'\b[A-Z]{2,}\b', sentence):
        score += 0.05  # acronyms

    # Boost: factual language patterns
    factual_patterns = [
        r'\b(?:uses?|runs?|requires?|supports?|depends? on|configured|deployed|stored)\b',
        r'\b(?:must|always|never|required|mandatory|minimum|maximum)\b',
        r'\b(?:decided|chose|selected|switched to|migrated to)\b',
        r'\b(?:port|endpoint|url|database|server|api|version|timeout)\b',
    ]
    for pat in factual_patterns:
        if re.search(pat, lower):
            score += 0.08

    # Penalize: conversational/meta language
    conversational_patterns = [
        r'^(?:hi|hey|hello|thanks|thank you|ok|okay|sure|great|nice|cool)\b',
        r'^(?:i think|maybe|perhaps|not sure|i guess|hmm|well)\b',
        r'\b(?:lol|haha|btw|imo|tbh)\b',
        r'^(?:can you|could you|please|would you)\b',
        r'\?$',  # questions ending in ? (usually not facts to store)
        r'^how\s+are\s+you\b',
    ]
    for pat in conversational_patterns:
        if re.search(pat, lower):
            score -= 0.15

    # Penalize: too short or too generic
    words = sentence.split()
    if len(words) < 4:
        score -= 0.2
    if len(words) > 30:
        score -= 0.1  # Very long sentences are usually explanatory, not factual

    return max(0.0, min(1.0, score))


def _classify_type(sentence: str) -> FactType:
    """Classify a sentence into a fact type."""
    lower = sentence.lower()

    # Constraint patterns
    if re.search(r'\b(?:must|always|never|required|mandatory|forbidden|cannot|should not)\b', lower):
        return FactType.CONSTRAINT

    # Decision patterns
    if re.search(r'\b(?:decided|chose|selected|we use|switched to|went with|opted for)\b', lower):
        return FactType.DECISION

    # Unknown/question patterns
    if re.search(r'\b(?:unknown|unclear|tbd|todo|need to figure|open question)\b', lower):
        return FactType.UNKNOWN

    # Done patterns
    if re.search(r'\b(?:completed|done|finished|resolved|shipped|deployed|released)\b', lower):
        return FactType.DONE

    # Default to Value
    return FactType.VALUE


def _extract_target(sentence: str) -> str:
    """Extract a target identifier from a sentence.

    Tries to find the subject/topic of the fact.
    """
    # Look for "X uses/is/has Y" patterns
    m = re.match(r'^(?:The\s+)?(\w+(?:\s+\w+)?)\s+(?:uses?|is|has|runs?|requires?)\b', sentence, re.I)
    if m:
        target = m.group(1).lower().replace(' ', '.')
        return target[:30]

    # Look for "X: Y" patterns
    m = re.match(r'^(\w+(?:\s+\w+)?)\s*:', sentence)
    if m:
        target = m.group(1).lower().replace(' ', '.')
        return target[:30]

    return "general"


# ─── Semantic Deduplication ───────────────────────────────────


def find_near_duplicates(
    conn,
    content: str,
    embedding,
    threshold: float = 0.85,
) -> List[Tuple[str, float]]:
    """Find existing facts that are near-duplicates of new content.

    Uses embedding similarity + word overlap to detect semantic duplicates.
    Returns list of (fact_id, similarity) tuples.

    This goes beyond source_hash deduplication (exact content match) to
    catch facts with different wording but same meaning:
      "API uses JWT tokens" ≈ "JWT authentication for the API"
    """
    import numpy as np
    from mnemoria.links import cosine_similarity

    if embedding is None:
        return []

    rows = conn.execute(
        "SELECT id, content, embedding FROM um_facts "
        "WHERE status = 'active' AND embedding IS NOT NULL"
    ).fetchall()

    duplicates = []
    content_words = set(content.lower().split())

    for r in rows:
        existing_emb = np.frombuffer(r["embedding"], dtype=np.float32)
        sim = cosine_similarity(embedding, existing_emb)

        if sim >= threshold:
            # Additional word overlap check to reduce false positives
            existing_words = set(r["content"].lower().split())
            word_overlap = len(content_words & existing_words) / max(len(content_words | existing_words), 1)

            if word_overlap >= 0.4:  # At least 40% word overlap
                duplicates.append((r["id"], sim))

    return sorted(duplicates, key=lambda x: x[1], reverse=True)


def merge_duplicate_facts(
    conn,
    fact_id: str,
    duplicate_ids: List[str],
    now: float,
) -> None:
    """Merge duplicate facts into the primary fact.

    Transfers links and access history from duplicates to the primary,
    then marks duplicates as superseded.
    """
    for dup_id in duplicate_ids:
        # Transfer links
        conn.execute(
            "UPDATE um_links SET source_id = ? WHERE source_id = ?",
            (fact_id, dup_id)
        )
        conn.execute(
            "UPDATE um_links SET target_id = ? WHERE target_id = ?",
            (fact_id, dup_id)
        )

        # Transfer access times
        conn.execute(
            "UPDATE um_access_times SET fact_id = ? WHERE fact_id = ?",
            (fact_id, dup_id)
        )

        # Mark as superseded
        conn.execute(
            "UPDATE um_facts SET status='superseded', superseded_by=?, updated_at=? "
            "WHERE id=?",
            (fact_id, now, dup_id)
        )

    # Remove self-referencing links that may have been created
    conn.execute(
        "DELETE FROM um_links WHERE source_id = target_id"
    )

    conn.commit()


# ─── Memorability Scoring ─────────────────────────────────────


def compute_memorability(
    content: str,
    fact_type: FactType = FactType.VALUE,
    context_importance: float = 0.5,
) -> float:
    """Compute how memorable/important a fact is.

    Combines:
    - Content features (specificity, actionability)
    - Fact type (constraints > decisions > values)
    - Context importance (from conversation context)

    Returns float 0.0-1.0.
    """
    score = _factuality_score(content)

    # Type weighting: constraints and decisions are more important
    type_weights = {
        FactType.CONSTRAINT: 0.15,
        FactType.DECISION: 0.1,
        FactType.VALUE: 0.0,
        FactType.UNKNOWN: 0.05,
        FactType.DONE: -0.05,
        FactType.OBSOLETE: -0.1,
    }
    score += type_weights.get(fact_type, 0.0)

    # Blend with context importance
    score = 0.6 * score + 0.4 * context_importance

    return max(0.0, min(1.0, score))
