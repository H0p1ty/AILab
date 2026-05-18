import sqlite3
from pathlib import Path

DB_PATH = Path("./sessions.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                role     TEXT NOT NULL,
                content  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id     TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id)
            )
        """)

_init()


def create_session(session_id: str) -> None:
    with _conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))


def session_exists(session_id: str) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return row is not None


def get_history(session_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def append_messages(session_id: str, messages: list[dict]) -> None:
    with _conn() as conn:
        conn.executemany(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, m["role"], m["content"]) for m in messages],
        )


def delete_session(session_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def register_doc(doc_id: str, session_id: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, session_id) VALUES (?, ?)", (doc_id, session_id)
        )


def get_session_docs(session_id: str) -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT doc_id FROM documents WHERE session_id = ?", (session_id,)
        ).fetchall()
    return [r["doc_id"] for r in rows]


def unregister_doc(doc_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
