"""
modules/memory.py

Long-term + short-term memory for a Jarvis-style assistant. SQLite only,
no extra packages besides `requests` (for Ollama).

Three kinds of memory:
  1. Conversation history  - every message, per session
  2. Facts about the user  - "name: Rahul", "likes: cricket" (key/value)
  3. Recall search         - finds old messages relevant to the current question

Quick start:
    from modules.memory import chat
    print(chat("Hi, my name is Rahul and I live in Kolkata"))
    print(chat("What's my name?"))           # remembers, even after restart

Lower-level API:
    add_message, get_recent, remember_fact, forget_fact, get_facts,
    search_memory, build_context, extract_facts, clear_memory
"""
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "Jarvis")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("MEMORY_DB", BASE_DIR / "memory.db"))

_lock = threading.Lock()
_conn = None
_has_fts = False


# ---------- database ----------
def _db() -> sqlite3.Connection:
    global _conn, _has_fts
    if _conn is None:
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);

            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated REAL NOT NULL
            );
            """
        )
        try:
            _conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts "
                "USING fts5(content, content='messages', content_rowid='id')"
            )
            _has_fts = True
        except sqlite3.OperationalError:
            _has_fts = False  # falls back to LIKE search
        _conn.commit()
    return _conn


# ---------- conversation history ----------
def add_message(role: str, content: str, session_id: str = "default") -> int:
    if not content or not content.strip():
        return -1
    with _lock:
        db = _db()
        cur = db.execute(
            "INSERT INTO messages (session_id, role, content, ts) VALUES (?,?,?,?)",
            (session_id, role, content.strip(), time.time()),
        )
        if _has_fts:
            db.execute(
                "INSERT INTO messages_fts(rowid, content) VALUES (?,?)",
                (cur.lastrowid, content.strip()),
            )
        db.commit()
        return cur.lastrowid


def get_recent(session_id: str = "default", limit: int = 10) -> list:
    """Last N messages as [{'role','content'}] in chronological order."""
    with _lock:
        rows = _db().execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ---------- facts about the user ----------
def remember_fact(key: str, value: str) -> None:
    key = re.sub(r"\s+", "_", key.strip().lower())
    if not key or not str(value).strip():
        return
    with _lock:
        db = _db()
        db.execute(
            "INSERT INTO facts (key, value, updated) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated",
            (key, str(value).strip(), time.time()),
        )
        db.commit()


def forget_fact(key: str) -> bool:
    key = re.sub(r"\s+", "_", key.strip().lower())
    with _lock:
        db = _db()
        cur = db.execute("DELETE FROM facts WHERE key=?", (key,))
        db.commit()
        return cur.rowcount > 0


def get_facts() -> dict:
    with _lock:
        rows = _db().execute("SELECT key, value FROM facts ORDER BY key").fetchall()
    return {r["key"]: r["value"] for r in rows}


# ---------- recall search ----------
def search_memory(query: str, limit: int = 5, exclude_session: str = None) -> list:
    """Find older messages related to the query. Returns [{'role','content','ts'}]."""
    words = [w for w in re.findall(r"\w{3,}", query.lower())][:8]
    if not words:
        return []

    with _lock:
        db = _db()
        if _has_fts:
            match = " OR ".join(f'"{w}"' for w in words)
            sql = (
                "SELECT m.role, m.content, m.ts FROM messages_fts f "
                "JOIN messages m ON m.id = f.rowid WHERE messages_fts MATCH ? "
            )
            params = [match]
            if exclude_session:
                sql += "AND m.session_id != ? "
                params.append(exclude_session)
            sql += "ORDER BY rank LIMIT ?"
            params.append(limit)
            try:
                rows = db.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                rows = []
        else:
            like = " OR ".join("content LIKE ?" for _ in words)
            sql = f"SELECT role, content, ts FROM messages WHERE ({like}) "
            params = [f"%{w}%" for w in words]
            if exclude_session:
                sql += "AND session_id != ? "
                params.append(exclude_session)
            sql += "ORDER BY id DESC LIMIT ?"
            params.append(limit)
            rows = db.execute(sql, params).fetchall()

    return [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]


# ---------- Ollama helpers ----------
def _ollama(messages: list, model: str = None, timeout: int = 120) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model or OLLAMA_MODEL, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def extract_facts(user_msg: str, model: str = None) -> dict:
    """Ask the LLM for lasting personal facts in the user's message and save them."""
    system = (
        "Extract lasting personal facts the user states about themselves "
        "(name, location, job, preferences, family, goals, schedule). "
        "Ignore questions, small talk and temporary things. "
        'Reply ONLY with a JSON object like {"name": "Rahul", "likes": "cricket"}. '
        "Reply {} if there is nothing worth remembering."
    )
    try:
        raw = _ollama(
            [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            model=model,
            timeout=60,
        )
        raw = re.sub(r"```(?:json)?", "", raw)
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {}
        facts = json.loads(raw[start : end + 1])
    except (requests.RequestException, ValueError, KeyError):
        return {}

    saved = {}
    for k, v in facts.items():
        if isinstance(v, (str, int, float)) and str(v).strip():
            remember_fact(k, str(v))
            saved[k] = str(v)
    return saved


# ---------- context builder ----------
def build_context(query: str, session_id: str = "default") -> str:
    """Text block to put in the system prompt: known facts + related old memories."""
    parts = []

    facts = get_facts()
    if facts:
        parts.append("Things you know about the user:\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items()))

    related = search_memory(query, limit=4, exclude_session=session_id)
    if related:
        lines = []
        for m in related:
            when = time.strftime("%Y-%m-%d", time.localtime(m["ts"]))
            who = "User" if m["role"] == "user" else ASSISTANT_NAME
            lines.append(f"- [{when}] {who}: {m['content'][:200]}")
        parts.append("Possibly relevant past conversations:\n" + "\n".join(lines))

    return "\n\n".join(parts)


# ---------- one-call chat with memory ----------
def chat(user_text: str, session_id: str = "default", model: str = None,
         history_limit: int = 10, learn: bool = True) -> str:
    """Send a message to Ollama with full memory, store everything, return the reply."""
    system = (
        f"You are {ASSISTANT_NAME}, a helpful, concise personal assistant. "
        f"Current time: {time.strftime('%A, %d %B %Y, %H:%M')}. "
        "Use the memory below naturally; never claim to remember things that aren't listed."
    )
    context = build_context(user_text, session_id)
    if context:
        system += "\n\n" + context

    messages = [{"role": "system", "content": system}]
    messages += get_recent(session_id, history_limit)
    messages.append({"role": "user", "content": user_text})

    reply = _ollama(messages, model=model)

    add_message("user", user_text, session_id)
    add_message("assistant", reply, session_id)

    if learn:
        threading.Thread(target=extract_facts, args=(user_text, model), daemon=True).start()

    return reply


def clear_memory(session_id: str = None, facts: bool = False) -> None:
    """Delete one session's messages (or all if None). facts=True also wipes facts."""
    with _lock:
        db = _db()
        if session_id:
            db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        else:
            db.execute("DELETE FROM messages")
        if _has_fts:
            db.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        if facts:
            db.execute("DELETE FROM facts")
        db.commit()


if __name__ == "__main__":
    print(f"Memory DB: {DB_PATH}")
    print("Type 'exit' to quit, '/facts' to see what I know, '/forget key' to delete a fact.\n")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if text == "/facts":
            print(get_facts())
        elif text.startswith("/forget "):
            print("Forgotten." if forget_fact(text[8:]) else "No such fact.")
        elif text:
            print(f"{ASSISTANT_NAME}: {chat(text)}\n")