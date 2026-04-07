"""Q-value store for retrieval reinforcement learning.

Each memory has a Q-value (0.0-1.0) that tracks whether it was useful
after being retrieved. Updated via exponential moving average (EMA)
from explicit reward signals.

Ported from Ori-Mnemos Layer 1 (qvalue.ts).

Reference: Watkins & Dayan (1992), Q-learning.
"""
import math
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class QEntry:
    memory_id: str
    q_value: float = 0.5      # default: neutral
    update_count: int = 0
    total_retrievals: int = 0
    last_updated: float = 0.0  # timestamp
    last_retrieved: float = 0.0


class QValueStore:
    """SQLite-backed Q-value store for memory reinforcement learning."""

    ALPHA = 0.1          # EMA learning rate
    DEFAULT_Q = 0.5      # initial Q for unseen memories
    DECAY_RATE = 0.007   # ~99 day half-life

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_qvalues (
                memory_id TEXT PRIMARY KEY,
                q_value REAL DEFAULT 0.5,
                update_count INTEGER DEFAULT 0,
                total_retrievals INTEGER DEFAULT 0,
                last_updated REAL DEFAULT 0.0,
                last_retrieved REAL DEFAULT 0.0
            )
        """)
        self._conn.commit()

    def get_q(self, memory_id: str) -> float:
        """Get the current Q-value for a memory, with temporal decay."""
        row = self._conn.execute(
            "SELECT q_value, update_count, last_updated FROM memory_qvalues WHERE memory_id = ?",
            (memory_id,)
        ).fetchone()
        if row is None:
            return self.DEFAULT_Q
        q, count, last_updated = row
        if last_updated > 0 and count > 0:
            days_since = (time.time() - last_updated) / 86400
            # Q-informed decay: good memories decay slower, bad faster
            mult = 0.7 if q >= 0.7 else (1.3 if q <= 0.3 else 1.0)
            q = q * math.exp(-self.DECAY_RATE * mult * max(days_since, 0))
        return q

    def get_q_batch(self, memory_ids: List[str]) -> Dict[str, float]:
        """Get Q-values for multiple memories efficiently."""
        return {mid: self.get_q(mid) for mid in memory_ids}

    def record_retrieval(self, memory_id: str):
        """Record that a memory was retrieved (for exposure tracking)."""
        now = time.time()
        self._conn.execute("""
            INSERT INTO memory_qvalues (memory_id, total_retrievals, last_retrieved)
            VALUES (?, 1, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                total_retrievals = total_retrievals + 1,
                last_retrieved = ?
        """, (memory_id, now, now))
        self._conn.commit()

    def reward(self, memory_id: str, signal: float):
        """Apply a reward signal to a memory's Q-value.

        Args:
            memory_id: The memory to reward
            signal: Reward value. Positive = useful, negative = not useful.
                Standard signals:
                    +1.0  forward citation (user referenced this memory)
                    +0.5  update after retrieval (user edited this memory)
                    +0.6  downstream creation (user created new memory after)
                    +0.4  within-session re-recall (retrieved multiple times)
                    -0.15 dead end (retrieved in top-3 but nothing followed)
        """
        now = time.time()
        row = self._conn.execute(
            "SELECT q_value, update_count, total_retrievals FROM memory_qvalues WHERE memory_id = ?",
            (memory_id,)
        ).fetchone()

        if row is None:
            old_q, count, retrievals = self.DEFAULT_Q, 0, 1
        else:
            old_q, count, retrievals = row

        # Exposure correction: diminish reward for frequently-retrieved memories
        exposure_factor = 1.0 / max(math.sqrt(retrievals), 1.0)
        adjusted_signal = signal * exposure_factor

        # EMA update
        new_q = old_q + self.ALPHA * (adjusted_signal - old_q)
        new_q = max(0.0, min(1.0, new_q))  # clamp to [0, 1]

        self._conn.execute("""
            INSERT INTO memory_qvalues (memory_id, q_value, update_count, last_updated)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                q_value = ?,
                update_count = update_count + 1,
                last_updated = ?
        """, (memory_id, new_q, now, new_q, now))
        self._conn.commit()

    def get_total_updates(self) -> int:
        """Total Q-value updates across all memories."""
        row = self._conn.execute("SELECT COALESCE(SUM(update_count), 0) FROM memory_qvalues").fetchone()
        return row[0] if row else 0

    def reset(self):
        """Clear all Q-values."""
        self._conn.execute("DELETE FROM memory_qvalues")
        self._conn.commit()

    def close(self):
        self._conn.close()
