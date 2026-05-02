"""
SQLite database for persisting chat history (sessions + messages).
Uses aiosqlite for async compatibility with FastAPI.
"""

import aiosqlite
import uuid
from datetime import datetime, timezone
from typing import Optional

from config import settings
from logging_config import logger

DB_PATH = settings.SQLITE_DB_PATH


async def init_db() -> None:
    """Create the chat history tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT NOT NULL,
                sources     TEXT,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
            """
        )
        await db.commit()
    logger.info("SQLite database initialized at %s", DB_PATH)


async def create_session(title: str) -> dict:
    """Create a new chat session and return its metadata."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        await db.commit()
    logger.info("Created chat session: %s (%s)", session_id, title)
    return {"id": session_id, "title": title, "created_at": now, "updated_at": now}


async def get_all_sessions() -> list[dict]:
    """Return all chat sessions ordered by most recent."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM chat_sessions ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_session_messages(session_id: str) -> list[dict]:
    """Return all messages for a given session."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[str] = None,
) -> dict:
    """Persist a single chat message and update the session timestamp."""
    message_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, session_id, role, content, sources, now),
        )
        await db.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        await db.commit()
    return {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": sources,
        "created_at": now,
    }
