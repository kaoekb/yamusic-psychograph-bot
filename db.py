import os
import sqlite3
import time
from typing import Optional

DB_PATH = os.environ.get("SQLITE_PATH", "bot.sqlite3")

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""PRAGMA journal_mode=WAL""")
    conn.execute("""PRAGMA synchronous=NORMAL""")
    conn.execute("""CREATE TABLE IF NOT EXISTS users(
        user_id     INTEGER PRIMARY KEY,
        username    TEXT,
        first_seen  INTEGER,
        last_seen   INTEGER,
        lang        TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS events(
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        ts      INTEGER,
        type    TEXT
    )""")
    return conn

CONN = _connect()

def upsert_user(tg_user) -> None:
    now = int(time.time())
    CONN.execute(
        """INSERT INTO users(user_id, username, first_seen, last_seen, lang)
           VALUES(?,?,?,?,?)
           ON CONFLICT(user_id) DO UPDATE SET
             username=excluded.username,
             last_seen=excluded.last_seen,
             lang=excluded.lang""",
        (tg_user.id, getattr(tg_user, "username", None), now, now, getattr(tg_user, "language_code", None)),
    )
    CONN.commit()

def log_event(user_id: int, etype: str) -> None:
    CONN.execute("INSERT INTO events(user_id, ts, type) VALUES(?,?,?)",
                 (user_id, int(time.time()), etype))
    CONN.commit()

def total_users() -> int:
    return CONN.execute("SELECT COUNT(*) FROM users").fetchone()[0]

def active_since(seconds: int) -> int:
    cutoff = int(time.time()) - seconds
    return CONN.execute("SELECT COUNT(*) FROM users WHERE last_seen>=?", (cutoff,)).fetchone()[0]
