#!/usr/bin/env python3
"""
pending.py — CLI inspector for Mnemoria um_pending table.

Shows pending facts grouped by session_id, color-coded by source,
with retract/promote actions.

Usage:
    python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db
    python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db --session <id>
    python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db --retract <pending_id>
    python -m mnemoria.scripts.pending --db ~/.hermes/mnemoria.db --promote <pending_id>
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
import json
import time
from datetime import datetime
from pathlib import Path

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[90m"

SOURCE_COLORS = {
    "observed": GREEN,
    "user_stated": CYAN,
    "agent_inference": YELLOW,
    "unknown": GRAY,
}

STATUS_COLORS = {
    "provisional": YELLOW,
    "promoted": GREEN,
    "retracted": RED,
}


def color_source(source: str) -> str:
    color = SOURCE_COLORS.get(source, GRAY)
    return f"{color}{source}{RESET}"


def color_status(status: str) -> str:
    color = STATUS_COLORS.get(status, GRAY)
    return f"{color}{status}{RESET}"


def color_type(fact_type: str) -> str:
    colors = {"C": RED, "D": BLUE, "V": GREEN, "?": YELLOW, "\u2713": CYAN, "~": MAGENTA}
    color = colors.get(fact_type, GRAY)
    return f"{color}{fact_type}{RESET}"


def format_timestamp(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate(s: str, width: int = 70) -> str:
    return s[:width] + ("..." if len(s) > width else "")


def get_pending_facts(conn: sqlite3.Connection, session_id: str | None = None,
                      source: str | None = None, status: str | None = None,
                      limit: int = 200) -> list:
    """Fetch pending facts with optional filters."""
    query = """
        SELECT id, content, type, target, source, status, session_id, created_at, updated_at, provenance
        FROM um_pending
        WHERE 1=1
    """
    params = []

    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    if source:
        query += " AND source = ?"
        params.append(source)
    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY session_id, created_at DESC LIMIT ?"
    params.append(limit)

    return conn.execute(query, params).fetchall()


def group_by_session(facts: list) -> dict:
    """Group pending facts by session_id."""
    groups = {}
    for fact in facts:
        sid = fact["session_id"]
        if sid not in groups:
            groups[sid] = []
        groups[sid].append(fact)
    return groups


def print_banner():
    print(f"{BOLD}Mnemoria Pending Facts Inspector{RESET}")
    print()


def print_stats(conn: sqlite3.Connection):
    """Print summary statistics."""
    stats = conn.execute("""
        SELECT status, COUNT(*) as count FROM um_pending GROUP BY status
    """).fetchall()

    total = sum(r["count"] for r in stats)
    print(f"{BOLD}Summary:{RESET} {total} pending fact(s) total")
    for row in stats:
        status = color_status(row["status"])
        print(f"  {status}: {row['count']}")

    by_source = conn.execute("""
        SELECT source, COUNT(*) as count FROM um_pending WHERE status='provisional' GROUP BY source
    """).fetchall()

    print(f"\n{BOLD}By source (provisional):{RESET}")
    for row in by_source:
        src = color_source(row["source"])
        print(f"  {src}: {row['count']}")
    print()


def print_pending_table(facts: list, show_session: bool = True):
    """Print pending facts in a formatted table."""
    if not facts:
        print(f"{GRAY}(no pending facts){RESET}")
        return

    # Print header
    header = f"{BOLD}{'ID':<10} {'TYPE':<6} {'SOURCE':<18} {'STATUS':<12} {'CREATED':<20} CONTENT{RESET}"
    if show_session:
        header = f"{BOLD}{'SESSION':<38} {header}{RESET}"
    print(header)
    print("-" * 120)

    for fact in facts:
        fid = fact["id"][:8]
        ftype = color_type(fact["type"])
        src = color_source(fact["source"])
        status = color_status(fact["status"])
        created = format_timestamp(fact["created_at"])
        content = truncate(fact["content"])

        line = f"{fid:<10} {ftype:<6} {src:<18} {status:<12} {created:<20} {content}"
        if show_session:
            sid = fact["session_id"][:36]
            line = f"{sid:<38} {line}"
        print(line)


def retract_pending(conn: sqlite3.Connection, pending_id: str, now: float) -> bool:
    """Retract a pending fact."""
    row = conn.execute(
        "SELECT id, status FROM um_pending WHERE id = ?", (pending_id,)
    ).fetchone()

    if not row:
        print(f"{RED}Error: Pending fact not found: {pending_id}{RESET}", file=sys.stderr)
        return False

    if row["status"] == "retracted":
        print(f"{YELLOW}Already retracted: {pending_id[:8]}{RESET}")
        return True

    if row["status"] == "promoted":
        print(f"{RED}Cannot retract: already promoted: {pending_id[:8]}{RESET}", file=sys.stderr)
        return False

    conn.execute(
        "UPDATE um_pending SET status = 'retracted', updated_at = ? WHERE id = ?",
        (now, pending_id),
    )
    conn.commit()
    print(f"{GREEN}Retracted: {pending_id[:8]}{RESET}")
    return True


def promote_pending(conn: sqlite3.Connection, pending_id: str, now: float) -> tuple[bool, str | None]:
    """Force-promote a pending fact. Returns (success, new_fact_id)."""
    row = conn.execute(
        "SELECT id, content, type, target, scope_id, session_id, source, provenance "
        "FROM um_pending WHERE id = ? AND status = 'provisional'",
        (pending_id,),
    ).fetchone()

    if not row:
        print(f"{RED}Error: Provisional pending fact not found: {pending_id}{RESET}", file=sys.stderr)
        return False, None

    provenance_raw = row["provenance"] or "{}"
    try:
        provenance = json.loads(provenance_raw)
    except Exception:
        provenance = {}

    provenance["source"] = row["source"]
    provenance["pending_id"] = pending_id
    provenance["session_id"] = row["session_id"]
    provenance["promoted_at"] = now
    provenance["forced_promotion"] = True

    new_fact_id = str(uuid.uuid4())

    conn.execute("""
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
        row["content"],
        row["type"],
        row["target"],
        row["scope_id"],
        now, now, now,
        json.dumps(provenance),
    ))

    conn.execute(
        "UPDATE um_pending SET status = 'promoted', promoted_to = ?, updated_at = ? WHERE id = ?",
        (new_fact_id, now, pending_id),
    )
    conn.commit()

    print(f"{GREEN}Promoted: {pending_id[:8]} -> {new_fact_id[:8]}{RESET}")
    return True, new_fact_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and control Mnemoria pending facts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --db ~/.hermes/mnemoria.db
  %(prog)s --db ~/.hermes/mnemoria.db --session <id>
  %(prog)s --db ~/.hermes/mnemoria.db --retract <pending_id>
  %(prog)s --db ~/.hermes/mnemoria.db --promote <pending_id>
  %(prog)s --db ~/.hermes/mnemoria.db --source observed
        """
    )
    parser.add_argument(
        "--db",
        default="~/.hermes/mnemoria.db",
        help="Path to Mnemoria SQLite DB (default: ~/.hermes/mnemoria.db)",
    )
    parser.add_argument(
        "--session",
        metavar="ID",
        help="Filter by session ID",
    )
    parser.add_argument(
        "--source",
        metavar="SOURCE",
        choices=["observed", "user_stated", "agent_inference"],
        help="Filter by source",
    )
    parser.add_argument(
        "--status",
        metavar="STATUS",
        choices=["provisional", "promoted", "retracted"],
        help="Filter by status",
    )
    parser.add_argument(
        "--retract",
        metavar="PENDING_ID",
        help="Retract a pending fact by ID",
    )
    parser.add_argument(
        "--promote",
        metavar="PENDING_ID",
        help="Force-promote a pending fact by ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of results (default: 200)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Suppress header and summary output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted table",
    )

    args = parser.parse_args()

    db_path = args.db.replace("~", str(Path.home()))
    if not Path(db_path).exists():
        print(f"{RED}Error: Database not found: {db_path}{RESET}", file=sys.stderr)
        return 1

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as exc:
        print(f"{RED}Error opening database: {exc}{RESET}", file=sys.stderr)
        return 1

    # Check schema
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='um_pending'"
    ).fetchone()
    if not row:
        print(f"{RED}Error: um_pending table not found. Is this a Mnemoria DB?{RESET}", file=sys.stderr)
        return 1

    now = time.time()

    # Handle single-action modes
    if args.retract:
        success = retract_pending(conn, args.retract, now)
        return 0 if success else 1

    if args.promote:
        success, _ = promote_pending(conn, args.promote, now)
        return 0 if success else 1

    # Default: display pending facts
    if not args.no_header:
        print_banner()
        print_stats(conn)

    facts = get_pending_facts(conn, session_id=args.session,
                             source=args.source, status=args.status,
                             limit=args.limit)

    if args.json:
        result = {
            "pending": [dict(f) for f in facts],
            "count": len(facts),
        }
        print(json.dumps(result, indent=2))
        return 0

    if args.session:
        # Show only session filter without grouping
        print(f"{BOLD}Pending facts for session {args.session[:36]}...{RESET}")
        print_pending_table(facts, show_session=False)
    else:
        # Group by session
        groups = group_by_session(facts)
        for sid, group_facts in groups.items():
            print(f"{BOLD}Session: {sid}{RESET}")
            print_pending_table(group_facts, show_session=False)
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())