"""
SQLite schema for unified memory (um_*).

Provides:
  SCHEMA_SQL  - DDL string covering all tables, indexes, triggers, and views.
  init_db()   - Execute SCHEMA_SQL on an existing connection.
  get_connection() - Open (or create) the DB, apply schema, return connection.
"""

from __future__ import annotations

import sqlite3

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-4096;
PRAGMA foreign_keys=ON;

-- ---------------------------------------------------------------- um_scopes ---
-- Must be created before um_facts due to FK reference
CREATE TABLE IF NOT EXISTS um_scopes (
    id               TEXT    PRIMARY KEY,
    label            TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'active',
    last_referenced  REAL,
    current_turn     INTEGER NOT NULL DEFAULT 0,
    created_at       REAL,
    closed_at        REAL
);

-- Prevent duplicate active scopes with the same label
CREATE UNIQUE INDEX IF NOT EXISTS idx_um_scopes_active_label
    ON um_scopes(label) WHERE status = 'active';

-- ---------------------------------------------------------------- um_facts ---
CREATE TABLE IF NOT EXISTS um_facts (
    id              TEXT    PRIMARY KEY,
    content         TEXT    NOT NULL,
    embedding       BLOB,
    type            TEXT    NOT NULL DEFAULT 'V',
    target          TEXT    NOT NULL DEFAULT 'general',
    scope_id        TEXT    REFERENCES um_scopes(id),
    status          TEXT    NOT NULL DEFAULT 'active',
    activation      REAL    NOT NULL DEFAULT 0.0,
    q_value         REAL    NOT NULL DEFAULT 0.5,
    access_count    INTEGER NOT NULL DEFAULT 0,
    metabolic_rate  REAL    NOT NULL DEFAULT 1.0,
    importance      REAL    NOT NULL DEFAULT 0.5,
    category        TEXT,
    layer           TEXT    NOT NULL DEFAULT 'working',
    pinned          INTEGER NOT NULL DEFAULT 0,
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    last_accessed   REAL    NOT NULL,
    source_hash     TEXT,
    superseded_by   TEXT    REFERENCES um_facts(id),
    provenance      TEXT  -- JSON: source, extractor, original pending id
);

CREATE INDEX IF NOT EXISTS idx_um_facts_type        ON um_facts(type);
CREATE INDEX IF NOT EXISTS idx_um_facts_target      ON um_facts(target);
CREATE INDEX IF NOT EXISTS idx_um_facts_scope_id    ON um_facts(scope_id);
CREATE INDEX IF NOT EXISTS idx_um_facts_status      ON um_facts(status);
CREATE INDEX IF NOT EXISTS idx_um_facts_updated_at  ON um_facts(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_um_facts_source_hash ON um_facts(source_hash);
CREATE INDEX IF NOT EXISTS idx_um_facts_layer       ON um_facts(layer);

-- ---------------------------------------------------------------- um_pending ---
CREATE TABLE IF NOT EXISTS um_pending (
    id              TEXT    PRIMARY KEY,
    content         TEXT    NOT NULL,
    type            TEXT    NOT NULL DEFAULT 'V',
    target          TEXT    NOT NULL DEFAULT 'general',
    scope_id        TEXT    REFERENCES um_scopes(id),
    session_id      TEXT    NOT NULL,
    source          TEXT    NOT NULL,  -- 'observed' | 'user_stated' | 'agent_inference'
    status          TEXT    NOT NULL DEFAULT 'provisional',  -- 'provisional' | 'promoted' | 'retracted'
    retracted_by    TEXT    REFERENCES um_pending(id),
    promoted_to     TEXT    REFERENCES um_facts(id),
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    provenance      TEXT  -- JSON: extractor name, event type, raw trigger
);

CREATE INDEX IF NOT EXISTS idx_um_pending_session   ON um_pending(session_id);
CREATE INDEX IF NOT EXISTS idx_um_pending_status    ON um_pending(status);
CREATE INDEX IF NOT EXISTS idx_um_pending_source    ON um_pending(source);
CREATE INDEX IF NOT EXISTS idx_um_pending_target    ON um_pending(type, target, session_id);
CREATE INDEX IF NOT EXISTS idx_um_pending_updated   ON um_pending(updated_at DESC);

-- ---------------------------------------------------------------- um_links ---
CREATE TABLE IF NOT EXISTS um_links (
    source_id           TEXT    NOT NULL REFERENCES um_facts(id),
    target_id           TEXT    NOT NULL REFERENCES um_facts(id),
    strength            REAL    NOT NULL DEFAULT 0.1,
    npmi                REAL    NOT NULL DEFAULT 0.0,
    co_occurrence_count INTEGER NOT NULL DEFAULT 0,
    link_type           TEXT    NOT NULL DEFAULT 'hebbian',
    last_updated        REAL,
    PRIMARY KEY (source_id, target_id)
);

-- -------------------------------------------------------- um_access_times ---
-- Separate table for ACT-R-style access history per fact
CREATE TABLE IF NOT EXISTS um_access_times (
    fact_id     TEXT    NOT NULL REFERENCES um_facts(id),
    access_time REAL    NOT NULL
);

-- ---------------------------------------------------------------- um_qvalues ---
CREATE TABLE IF NOT EXISTS um_qvalues (
    memory_id        TEXT    PRIMARY KEY,
    q_value          REAL    NOT NULL DEFAULT 0.5,
    update_count     INTEGER NOT NULL DEFAULT 0,
    total_retrievals INTEGER NOT NULL DEFAULT 0,
    last_updated     REAL,
    last_retrieved   REAL,
    reward_variance  REAL    NOT NULL DEFAULT 0.0
);

-- --------------------------------------------------------- um_facts FTS5 ---
-- content= points at um_facts so rebuilds work; rowid= maps to the SQLite rowid.
CREATE VIRTUAL TABLE IF NOT EXISTS um_facts_fts USING fts5(
    content,
    target,
    content='um_facts',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS um_facts_ai AFTER INSERT ON um_facts BEGIN
    INSERT INTO um_facts_fts(rowid, content, target)
    VALUES (new.rowid, new.content, new.target);
END;

CREATE TRIGGER IF NOT EXISTS um_facts_ad AFTER DELETE ON um_facts BEGIN
    INSERT INTO um_facts_fts(um_facts_fts, rowid, content, target)
    VALUES ('delete', old.rowid, old.content, old.target);
END;

CREATE TRIGGER IF NOT EXISTS um_facts_au AFTER UPDATE ON um_facts BEGIN
    INSERT INTO um_facts_fts(um_facts_fts, rowid, content, target)
    VALUES ('delete', old.rowid, old.content, old.target);
    INSERT INTO um_facts_fts(rowid, content, target)
    VALUES (new.rowid, new.content, new.target);
END;

-- --------------------------------------------------------------- views ---
CREATE VIEW IF NOT EXISTS um_gauge AS
    SELECT
        COALESCE(SUM(LENGTH(type) + LENGTH(target) + LENGTH(content) + 4), 0) AS used_chars,
        10000 AS max_chars
    FROM um_facts
    WHERE status IN ('active', 'cold');

CREATE VIEW IF NOT EXISTS um_hot_facts AS
    SELECT f.*
    FROM um_facts f
    JOIN um_scopes s ON f.scope_id = s.id
    WHERE f.status = 'active'
      AND s.status = 'active';

-- ---------------------------------------------------------------- um_meta ---
CREATE TABLE IF NOT EXISTS um_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    """Add provenance column if missing (for existing DBs)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(um_facts)")}
    if "provenance" not in cols:
        conn.execute("ALTER TABLE um_facts ADD COLUMN provenance TEXT")


def init_db(conn: sqlite3.Connection) -> None:
    """Execute SCHEMA_SQL against an existing connection. Idempotent."""
    conn.executescript(SCHEMA_SQL)
    _migrate_to_v2(conn)
    conn.execute("INSERT OR IGNORE INTO um_meta (key, value) VALUES ('schema_version', '2')")
    conn.commit()


def get_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at db_path, apply the unified-memory
    schema, and return the connection.

    The connection has:
      - row_factory = sqlite3.Row
      - WAL journal mode
      - Foreign keys enabled
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn
