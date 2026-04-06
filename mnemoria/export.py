"""Training-data export for unified memory fine-tuning pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import sqlite3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrainingPair:
    """A single input→output training pair derived from memory interactions."""
    pair_type: str       # e.g. "task_completion", "fact_recall", "cross_context"
    input: str           # Query or context description
    output: str          # Fact content or structured response

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def export_training_pairs(
    conn: sqlite3.Connection,
    q_threshold: float = 0.0,
    access_threshold: int = 1,
    max_pairs: int = 1000,
    qvalue_db_path: str | None = None,
) -> list[TrainingPair]:
    """Extract training pairs from memory interactions.

    Generates three types of pairs:
    - task_completion : recall → store chains (high-value problem-solving traces)
    - fact_recall     : individual recall events with high Q-value facts
    - cross_context    : the same fact retrieved across multiple scopes (transfer signal)

    Parameters
    ----------
    conn            : sqlite3 connection to the unified memory DB
    q_threshold     : minimum Q-value for a fact to be included (0 = all)
    access_threshold: minimum access_count for a fact to be included (1 = all)
    max_pairs       : maximum number of pairs to return (default 1000)
    qvalue_db_path  : path to q-values DB for enriched Q-values (optional)

    Returns
    -------
    list[TrainingPair]
    """
    # Load all relevant facts into a dict keyed by id
    facts = {}
    rows = conn.execute(
        "SELECT id, content, q_value, access_count FROM um_facts"
        " WHERE q_value >= ? AND access_count >= ?",
        (q_threshold, access_threshold),
    ).fetchall()
    for row in rows:
        facts[row["id"]] = {
            "content": row["content"],
            "q_value": row["q_value"],
            "access_count": row["access_count"],
        }

    if not facts:
        return []

    # Load all interactions
    interactions = conn.execute(
        "SELECT id, query, scope, recalled_ids, followed_by_store FROM um_interactions"
    ).fetchall()

    # Build fact_id → [interactions] mapping
    fact_to_interactions: dict[str, list] = {fid: [] for fid in facts}
    for ix in interactions:
        recalled_ids = json.loads(ix["recalled_ids"])
        for fid in recalled_ids:
            if fid in facts:
                fact_to_interactions[fid].append({
                    "ix_id": ix["id"],
                    "query": ix["query"],
                    "scope": ix["scope"],
                    "followed_by_store": ix["followed_by_store"],
                })

    pairs: list[TrainingPair] = []
    seen_cross_context: set = set()

    for fid, ix_list in fact_to_interactions.items():
        if not ix_list:
            continue
        fact_content = facts[fid]["content"]
        scopes = {ix["scope"] for ix in ix_list if ix["scope"]}

        # cross_context: same fact recalled in >= 2 different scopes
        if len(scopes) >= 2:
            scope_list = sorted(scopes)
            key = (fid, tuple(scope_list))
            if key not in seen_cross_context:
                seen_cross_context.add(key)
                context_parts = []
                for ix in ix_list:
                    if ix["scope"]:
                        context_parts.append(
                            "[{scope}] {query} \u2192 {content}".format(
                                scope=ix["scope"],
                                query=ix["query"],
                                content=fact_content
                            )
                        )
                context_desc = " | ".join(context_parts)
                pairs.append(TrainingPair(
                    pair_type="cross_context",
                    input=context_desc,
                    output=fact_content,
                ))

        # fact_recall + task_completion from each interaction
        for ix in ix_list:
            pairs.append(TrainingPair(
                pair_type="fact_recall",
                input=ix["query"],
                output=fact_content,
            ))
            if ix["followed_by_store"]:
                pairs.append(TrainingPair(
                    pair_type="task_completion",
                    input=ix["query"],
                    output=fact_content,
                ))

    return pairs[:max_pairs]


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def export_to_jsonl(pairs: list[TrainingPair], path: str) -> int:
    """Write training pairs to a JSON Lines file.

    Returns the number of lines written.
    """
    out = Path(path)
    count = 0
    with out.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def export_to_openai_jsonl(pairs: list[TrainingPair], path: str) -> int:
    """Write training pairs in OpenAI fine-tuning format to a JSON Lines file.

    Each line is a JSON object with a ``messages`` key:
    ``[{"role": "user", "content": <input>}, {"role": "assistant", "content": <output>}]``

    Returns the number of lines written.
    """
    out = Path(path)
    count = 0
    with out.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            record = {
                "messages": [
                    {"role": "user",      "content": pair.input},
                    {"role": "assistant", "content": pair.output},
                ]
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_to_openai_messages(
    pairs: list[TrainingPair],
) -> list[dict]:
    """Format training pairs as OpenAI fine-tuning message lists.

    Returns a list of dicts, each with a ``messages`` key containing
    ``[{"role": "user", "content": <input>}, {"role": "assistant", "content": <output>}]``.
    """
    result = []
    for pair in pairs:
        result.append({
            "messages": [
                {"role": "user",      "content": pair.input},
                {"role": "assistant", "content": pair.output},
            ]
        })
    return result
