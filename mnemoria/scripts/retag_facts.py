#!/usr/bin/env python3
"""
retag_facts.py — One-off interactive backfill script for Mnemoria v0.2.0.

Reads all facts with target='general', prints them interactively,
accepts a new target from stdin (or 'skip'), and writes back via
UPDATE inside a transaction with FTS triggers temporarily disabled.

Usage:
    python -m mnemoria.scripts.retag_facts --db ~/.hermes/mnemoria.db

The unsafe-virtual-table error we hit earlier is the FTS `au` trigger
firing during the bulk update. Dropping and recreating it around the
bulk update is the clean workaround.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys


# ----------------------------------------------------------------------
# Schema FTS trigger definitions (copied from schema.py to recreate them)
# ----------------------------------------------------------------------

FTS_INSERT_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS um_facts_ai AFTER INSERT ON um_facts BEGIN
    INSERT INTO um_facts_fts(rowid, content, target)
    VALUES (new.rowid, new.content, new.target);
END;
"""

FTS_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS um_facts_ad AFTER DELETE ON um_facts BEGIN
    INSERT INTO um_facts_fts(um_facts_fts, rowid, content, target)
    VALUES ('delete', old.rowid, old.content, old.target);
END;
"""

FTS_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS um_facts_au AFTER UPDATE ON um_facts BEGIN
    INSERT INTO um_facts_fts(um_facts_fts, rowid, content, target)
    VALUES ('delete', old.rowid, old.content, old.target);
    INSERT INTO um_facts_fts(rowid, content, target)
    VALUES (new.rowid, new.content, new.target);
END;
"


def run_retag(db_path: str, dry_run: bool = False, default_target: str | None = None) -> int:
    """Run the interactive retag session. Returns number of updated facts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Confirm schema
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='um_facts'").fetchone()
    if not row:
        print("ERROR: um_facts table not found. Is this a Mnemoria DB?", file=sys.stderr)
        return 1

    # Collect facts with target='general'
    facts = conn.execute(
        "SELECT id, content, type, target, created_at FROM um_facts WHERE target='general' AND status='active'"
    ).fetchall()

    if not facts:
        print(f"No facts with target='general' found in {db_path}")
        return 0

    print(f"Found {len(facts)} fact(s) with target='general'.", file=sys.stderr)
    print()

    updated = 0
    skipped = 0

    # Disable FTS update trigger before bulk update
    conn.execute("DROP TRIGGER IF EXISTS um_facts_au")

    try:
        for fact in facts:
            fid = fact["id"]
            content = fact["content"]
            ftype = fact["type"]
            created = fact["created_at"]

            # Truncate content for display (80 chars)
            display = content[:80] + ("..." if len(content) > 80 else "")
            print(f"[{fid[:8]}] type={ftype} | {display}")

            if default_target is not None:
                new_target = default_target
                print(f"  => target='{new_target}' (--default-target)")
            else:
                print("  Enter new target (or 'skip' to leave unchanged):", end=" ", flush=True)
                line = sys.stdin.readline()
                if not line:
                    break
                new_target = line.strip()
                if new_target.lower() in ("skip", "s", ""):
                    print("  SKIPPED")
                    skipped += 1
                    continue

            if dry_run:
                print(f"  [DRY RUN] Would set target='{new_target}'")
            else:
                conn.execute(
                    "UPDATE um_facts SET target=?, updated_at=? WHERE id=?",
                    (new_target, created, fid)
                )
                print(f"  => target set to '{new_target}'")

            updated += 1

        if not dry_run:
            conn.commit()
            print(f"\nCommitted {updated} update(s).")
        else:
            print(f"\nDry run — no changes made ({updated} would be updated, {skipped} skipped).")

    finally:
        # Recreate FTS update trigger
        conn.execute(FTS_UPDATE_TRIGGER)
        conn.close()

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Retag Mnemoria facts with target='general'")
    parser.add_argument(
        "--db",
        default="~/.hermes/mnemoria.db",
        help="Path to Mnemoria SQLite DB (default: ~/.hermes/mnemoria.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without making changes",
    )
    parser.add_argument(
        "--default-target",
        metavar="TARGET",
        help="Apply this target to all facts without prompting (e.g. 'hermes-agent')",
    )
    args = parser.parse_args()

    db_path = args.db.replace("~", str(__import__("os").path.expanduser("~")))
    return run_retag(db_path, dry_run=args.dry_run, default_target=args.default_target)


if __name__ == "__main__":
    sys.exit(main())
