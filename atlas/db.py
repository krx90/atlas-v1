"""SQLite schema and connection handling.

Three tables: `bars` (the daily OHLCV cache), `universe` (the screened symbol
list), and `meta` (small key/value state such as when the universe was built).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol      TEXT    NOT NULL,
    date        TEXT    NOT NULL,   -- ISO date, exchange-local session
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    trade_count REAL,
    vwap        REAL,
    PRIMARY KEY (symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_date ON bars (symbol, date DESC);

CREATE TABLE IF NOT EXISTS universe (
    symbol          TEXT PRIMARY KEY,
    name            TEXT,
    exchange        TEXT,
    fractionable    INTEGER NOT NULL DEFAULT 0,
    -- Borrowability. A symbol that cannot be borrowed is not a short
    -- candidate, so these gate what reaches top30_short.csv.
    shortable       INTEGER NOT NULL DEFAULT 0,
    easy_to_borrow  INTEGER NOT NULL DEFAULT 0,
    dollar_volume   REAL,
    last_price      REAL
);

-- Intraday bars live in their own table rather than gaining a timeframe column
-- on `bars`: the daily cache is large and expensive to rebuild, and the two are
-- keyed differently (a session date versus an instant).
CREATE TABLE IF NOT EXISTS intraday_bars (
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,   -- '5Min', '15Min', '1Hour'
    ts          TEXT    NOT NULL,   -- ISO-8601 UTC, the bar's opening instant
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL,
    trade_count REAL,
    vwap        REAL,
    PRIMARY KEY (symbol, timeframe, ts)
);

CREATE INDEX IF NOT EXISTS idx_intraday ON intraday_bars (symbol, timeframe, ts DESC);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


#: Columns added after the first release. `CREATE TABLE IF NOT EXISTS` is a
#: no-op on an existing table, so new columns have to be added explicitly --
#: otherwise an established cache (hundreds of thousands of bars, expensive to
#: refetch) would have to be thrown away to pick up a schema change.
_MIGRATIONS = {
    "universe": {
        "shortable": "INTEGER NOT NULL DEFAULT 0",
        "easy_to_borrow": "INTEGER NOT NULL DEFAULT 0",
    },
}


def _migrate(conn: sqlite3.Connection) -> list[str]:
    """Add any columns missing from an older database. Idempotent."""
    added = []
    for table, columns in _MIGRATIONS.items():
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # table was just created from _SCHEMA, already current
        for name, spec in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")
                added.append(f"{table}.{name}")
    return added


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL keeps a long scan's writes from blocking a concurrent `atlas view`.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
