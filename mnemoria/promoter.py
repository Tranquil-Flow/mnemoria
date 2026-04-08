"""
Promoter -- drains provisional PendingFacts from um_pending into confirmed um_facts.

One shot, no thread/daemon. Called from MnemoriaStore.flush_pending() or
at startup to recover facts from crashed prior sessions.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

import sqlite3


def run_promotion_pass(conn: sqlite3.Connection, now: float, session_ttl: float = 600.0) -> dict:
    """
    One promotion pass over um_pending.

    Promotion rules
    ---------------
    - observed, user_stated: promote immediately (ignore TTL)
    - agent_inference: promote only when the session has been inactive
      for session_ttl seconds (no pending activity in that session)
      AND the row is not retracted
    - Within the same (type, target, session_id) group, the latest
      provisional row retracts all earlier ones.

    All operations run inside a single transaction.  Idempotent -- calling
    twice in a row promotes nothing new.

    Parameters
    ----------
    conn : sqlite3.Connection
        Live connection to the mnemoria database.
    now : float
        Current timestamp (e.g. time.time()).
    session_ttl : float
        Seconds of inactivity before an agent_inference session is
        considered stale.  Default 600 s.

    Returns
    -------
    dict
        {"observed_promoted": int, "user_stated_promoted": int,
         "agent_inference_promoted": int, "retracted": int}
    """
    stats = {
        "observed_promoted": 0,
        "user_stated_promoted": 0,
        "agent_inference_promoted": 0,
        "retracted": 0,
    }

    cursor = conn.cursor()

    # Per-observer metric accumulators for this pass
    retract_counts: dict[tuple[str, str], int] = {}
    promote_counts: dict[tuple[str, str], int] = {}

    # -- 1. Retraction pass -----------------------------------------------
    # For each (type, target, session_id) keep only the LATEST provisional
    # row; mark all others as retracted.  We use updated_at as tie-breaker.
    # First: collect per-observer counts before updating.
    cursor.execute("""
        WITH ranked AS (
            SELECT
                p.id,
                p.type,
                p.target,
                p.session_id,
                p.status,
                p.updated_at,
                p.provenance,
                ROW_NUMBER()
                    OVER (PARTITION BY p.type, p.target, p.session_id
                          ORDER BY p.updated_at DESC) AS rn
            FROM um_pending p
            WHERE p.status = 'provisional'
        )
        SELECT session_id, provenance
        FROM ranked
        WHERE rn > 1
    """)
    for row in cursor.fetchall():
        session_id = row["session_id"] or "global"
        prov_raw = row["provenance"] or "{}"
        try:
            prov = json.loads(prov_raw)
        except Exception:
            prov = {}
        observer = prov.get("extractor", prov.get("source", "unknown"))
        key = (session_id, observer)
        retract_counts[key] = retract_counts.get(key, 0) + 1

    # Now perform the actual retraction
    # Use the count from the first CTE query (which we already collected)
    # instead of cursor.rowcount (which returns -1 for CTE UPDATEs in SQLite)
    retraction_target_count = sum(retract_counts.values())
    cursor.execute("""
        WITH ranked AS (
            SELECT
                p.id,
                p.type,
                p.target,
                p.session_id,
                p.status,
                p.updated_at,
                ROW_NUMBER()
                    OVER (PARTITION BY p.type, p.target, p.session_id
                          ORDER BY p.updated_at DESC) AS rn
            FROM um_pending p
            WHERE p.status = 'provisional'
        )
        UPDATE um_pending
        SET status = 'retracted'
        WHERE id IN (
            SELECT id FROM ranked WHERE rn > 1
        )
    """)
    # cursor.rowcount returns -1 for CTE UPDATE, so use our pre-computed count
    stats["retracted"] = retraction_target_count

    # -- 2. Promote observed + user_stated immediately ----------------------
    for source in ("observed", "user_stated"):
        counts = _promote_by_source(cursor, conn, source, now)
        total_promoted = 0
        for (sid, obs), cnt in counts.items():
            promote_counts[(sid, obs)] = promote_counts.get((sid, obs), 0) + cnt
            total_promoted += cnt
        # cursor.rowcount is unreliable after statements in _promote_by_source,
        # so use the count we collected directly from the function
        stats[f"{source}_promoted"] = total_promoted

    # -- 3. Promote agent_inference only when session is TTL-expired -------
    # Find sessions that have NOT had any pending activity in the last
    # session_ttl seconds.  max_updated_at gives the last activity time.
    cursor.execute("""
        SELECT p.session_id
        FROM um_pending p
        WHERE p.source = 'agent_inference'
          AND p.status = 'provisional'
        GROUP BY p.session_id
        HAVING MAX(p.updated_at) < :now - :ttl
    """, {"now": now, "ttl": session_ttl})

    expired_sessions = [row["session_id"] for row in cursor.fetchall()]
    for session_id in expired_sessions:
        counts = _promote_by_source(cursor, conn, "agent_inference", now,
                            extra_where="AND session_id = ?",
                            extra_args=(session_id,))
        session_promoted = 0
        for (sid, obs), cnt in counts.items():
            promote_counts[(sid, obs)] = promote_counts.get((sid, obs), 0) + cnt
            session_promoted += cnt
        stats["agent_inference_promoted"] += session_promoted

    # -- Emit metrics ---------------------------------------------------------
    for (session_id, observer), count in retract_counts.items():
        _increment_metric(conn, session_id, observer, "retract_count", count)
    for (session_id, observer), count in promote_counts.items():
        _increment_metric(conn, session_id, observer, "promote_count", count)

    conn.commit()
    return stats


def _promote_by_source(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    source: str,
    now: float,
    extra_where: str = "",
    extra_args: tuple = (),
) -> dict[tuple[str, str], int]:
    """
    Promote all provisional rows of the given source to um_facts.

    Sets status = 'promoted' and records promoted_to on each row.
    """
    # Fetch rows to promote
    query = """
        SELECT id, content, type, target, scope_id, session_id, provenance
        FROM um_pending
        WHERE source = ?
          AND status = 'provisional'
    """ + extra_where

    cursor.execute(query, (source,) + extra_args)

    rows = cursor.fetchall()
    if not rows:
        return {}

    promoted_fact_ids = []
    promote_counts: dict[tuple[str, str], int] = {}

    for row in rows:
        pending_id = row["id"]
        content = row["content"]
        fact_type = row["type"]
        target = row["target"]
        scope_id = row["scope_id"]
        session_id = row["session_id"]
        provenance_raw = row["provenance"] or "{}"

        # Parse provenance JSON, augment with promotion metadata
        try:
            provenance = json.loads(provenance_raw)
        except Exception:
            provenance = {}

        provenance["source"] = source
        provenance["pending_id"] = pending_id
        provenance["session_id"] = session_id
        provenance["promoted_at"] = now

        new_fact_id = str(uuid.uuid4())

        # Insert into um_facts
        cursor.execute("""
            INSERT INTO um_facts
                (id, content, type, target, scope_id, status,
                 activation, q_value, access_count, metabolic_rate,
                 importance, category, layer, pinned,
                 created_at, updated_at, last_accessed, provenance)
            VALUES (?, ?, ?, ?, ?, 'active',
                    0.0, 0.5, 0, 1.0,
                    0.5, NULL, 'working', 0,
                    ?, ?, ?, ?)
        """, (
            new_fact_id,
            content,
            fact_type,
            target,
            scope_id,
            now,      # created_at
            now,      # updated_at
            now,      # last_accessed
            json.dumps(provenance),
        ))

        # Mark pending row as promoted
        cursor.execute("""
            UPDATE um_pending
            SET status = 'promoted', promoted_to = ?
            WHERE id = ?
        """, (new_fact_id, pending_id))

        promoted_fact_ids.append(new_fact_id)

        # Track per-observer promote count
        observer = provenance.get("extractor", provenance.get("source", "unknown"))
        key = (session_id or "global", observer)
        promote_counts[key] = promote_counts.get(key, 0) + 1

    # Note: conn.commit() is called by the caller (run_promotion_pass)
    return promote_counts


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _emit_metrics(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    stats: dict,
    now: float,
) -> None:
    """Emit promote_count and retract_count to um_metrics.

    Extracts observer name from provenance of promoted/retracted pending
    facts and increments per-(session_id, observer) counters.
    """
    # -- Retracted facts: per-observer counts --------------------------------
    # Query facts that were just marked retracted in this pass
    retracted_rows = cursor.execute("""
        SELECT session_id, provenance
        FROM um_pending
        WHERE status = 'retracted'
    """).fetchall()

    retract_counts: dict[tuple[str, str], int] = {}
    for row in retracted_rows:
        session_id = row["session_id"] or "global"
        prov_raw = row["provenance"] or "{}"
        try:
            prov = json.loads(prov_raw)
        except Exception:
            prov = {}
        observer = prov.get("extractor", prov.get("source", "unknown"))
        key = (session_id, observer)
        retract_counts[key] = retract_counts.get(key, 0) + 1

    for (session_id, observer), count in retract_counts.items():
        _increment_metric(conn, session_id, observer, "retract_count", count)

    # -- Promoted facts: per-observer counts --------------------------------
    promoted_rows = cursor.execute("""
        SELECT session_id, provenance
        FROM um_pending
        WHERE status = 'promoted'
    """).fetchall()

    promote_counts: dict[tuple[str, str], int] = {}
    for row in promoted_rows:
        session_id = row["session_id"] or "global"
        prov_raw = row["provenance"] or "{}"
        try:
            prov = json.loads(prov_raw)
        except Exception:
            prov = {}
        observer = prov.get("extractor", prov.get("source", "unknown"))
        key = (session_id, observer)
        promote_counts[key] = promote_counts.get(key, 0) + 1

    for (session_id, observer), count in promote_counts.items():
        _increment_metric(conn, session_id, observer, "promote_count", count)


def _increment_metric(
    conn: sqlite3.Connection,
    session_id: str,
    observer: str,
    counter: str,
    delta: int = 1,
) -> None:
    """Increment a counter in ``um_metrics``.

    Uses INSERT ... ON CONFLICT DO UPDATE SET count = count + delta
    so the operation is idempotent and accumulates correctly.
    """
    try:
        conn.execute(
            f"""INSERT INTO um_metrics (session_id, observer, {counter})
                VALUES (?, ?, ?)
                ON CONFLICT (session_id, observer) DO UPDATE
                SET {counter} = {counter} + ?""",
            (session_id, observer, delta, delta),
        )
    except sqlite3.OperationalError:
        # Table doesn't exist yet — skip silently
        pass
