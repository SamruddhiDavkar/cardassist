"""SQLite helpers."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    card_number TEXT NOT NULL          -- fake, never returned unmasked
);
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL REFERENCES customers(customer_id),
    date           TEXT NOT NULL,      -- ISO yyyy-mm-dd
    merchant       TEXT NOT NULL,
    category       TEXT NOT NULL,
    amount         REAL NOT NULL,
    status         TEXT NOT NULL       -- posted | pending | disputed
);
CREATE TABLE IF NOT EXISTS disputes (
    dispute_id     TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    reason         TEXT NOT NULL,
    created_on     TEXT NOT NULL,
    status         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS escalations (
    ticket_id   TEXT PRIMARY KEY,
    customer_id TEXT,
    reason      TEXT NOT NULL,
    created_on  TEXT NOT NULL
);
"""


@contextmanager
def connect(db_path: Path | str | None = None):
    conn = sqlite3.connect(str(db_path or config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
