"""SQLite storage: documents, chunks (with an FTS5 keyword index) and chat history.

SQLite is the source of truth for chunk text. The vector index (Chroma) only
holds embeddings keyed by chunk id, so it can always be rebuilt from here.
"""

import json
import re
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.logging_config import logger

SCHEMA_VERSION = 2
NEW_CHAT_TITLE = "New chat"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT NOT NULL UNIQUE,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    sources     TEXT,
    meta        TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL UNIQUE,
    content_hash  TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    char_count    INTEGER NOT NULL,
    chunk_count   INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    pk       INTEGER PRIMARY KEY AUTOINCREMENT,
    id       TEXT NOT NULL UNIQUE,
    doc_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,
    text     TEXT NOT NULL,
    heading  TEXT NOT NULL DEFAULT '',
    page     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);

-- Which documents each conversation searches. A document is stored and indexed once
-- and can be attached to any number of conversations.
CREATE TABLE IF NOT EXISTS session_documents (
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    added_at    TEXT NOT NULL,
    PRIMARY KEY (session_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_session_documents_doc ON session_documents(doc_id);
"""

# Keyword index; rowid mirrors chunks.pk. `title` holds "<file name> <section heading>".
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
USING fts5(text, title, tokenize = 'unicode61 remove_diacritics 2');
"""

# Function words carry no retrieval signal and make OR-queries match everything.
_STOPWORDS = frozenset(
    """
    a about all also am an and any are as at be been but by can could did do does for from
    had has have he her his how i if in into is it its me my no not of on or our she so
    some than that the their them then there these they this those to us was we were what
    when where which who whom why will with would you your tell give show please explain
    apa yang dan di ke dari ini itu adalah untuk dengan pada atau juga tidak saya kamu
    bagaimana siapa kapan mengapa dimana tentang tolong jelaskan
    """.split()
)
_WORD = re.compile(r"\w+", re.UNICODE)


def keyword_query(text: str, max_terms: int = 32) -> str | None:
    """Turn free text into a safe FTS5 expression: quoted terms joined with OR."""
    terms: list[str] = []
    for token in _WORD.findall(text.lower()):
        if len(token) > 1 and token not in _STOPWORDS and token not in terms:
            terms.append(token)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms[:max_terms])


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class NewChunk:
    """A chunk about to be stored."""

    text: str
    heading: str = ""
    page: int | None = None


@dataclass(frozen=True)
class ChunkRecord:
    """A stored chunk, joined with its document's file name."""

    id: str
    doc_id: str
    filename: str
    ordinal: int
    text: str
    heading: str
    page: int | None


_CHUNK_COLUMNS = "c.id, c.doc_id, d.filename, c.ordinal, c.text, c.heading, c.page"


def _chunk(row: sqlite3.Row) -> ChunkRecord:
    return ChunkRecord(*row)


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fts_enabled = True
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """One short-lived connection per operation: commit on success, roll back on error."""
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
            try:
                conn.executescript(_FTS_SCHEMA)
            except sqlite3.OperationalError as exc:  # SQLite built without FTS5
                self.fts_enabled = False
                logger.warning("Keyword search disabled (SQLite FTS5 unavailable): %s", exc)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if 0 < version < 2:
                _attach_cited_documents(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ── Documents and chunks ─────────────────────────────────────

    def add_document(
        self,
        *,
        filename: str,
        content_hash: str,
        size_bytes: int,
        char_count: int,
        chunks: Sequence[NewChunk],
    ) -> tuple[str, list[str]]:
        """Insert a document and its chunks atomically. Returns (doc_id, chunk_ids)."""
        doc_id = uuid.uuid4().hex
        chunk_ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        stem = Path(filename).stem
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, content_hash, size_bytes, char_count,"
                " chunk_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, filename, content_hash, size_bytes, char_count, len(chunks), _now()),
            )
            for ordinal, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks, strict=True)):
                cur = conn.execute(
                    "INSERT INTO chunks (id, doc_id, ordinal, text, heading, page)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, ordinal, chunk.text, chunk.heading, chunk.page),
                )
                if self.fts_enabled:
                    conn.execute(
                        "INSERT INTO chunks_fts (rowid, text, title) VALUES (?, ?, ?)",
                        (cur.lastrowid, chunk.text, f"{stem} {chunk.heading}".strip()),
                    )
        return doc_id, chunk_ids

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            if self.fts_enabled:
                conn.execute(
                    "DELETE FROM chunks_fts WHERE rowid IN"
                    " (SELECT pk FROM chunks WHERE doc_id = ?)",
                    (doc_id,),
                )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            deleted = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,)).rowcount
        return deleted > 0

    def get_document(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def get_document_by_filename(self, filename: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE filename = ?", (filename,)
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY filename COLLATE NOCASE"
            ).fetchall()
        return [dict(r) for r in rows]

    def count_chunks(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.doc_id"
                f" WHERE c.id IN ({marks})",
                list(chunk_ids),
            ).fetchall()
        return {row["id"]: _chunk(row) for row in rows}

    def iter_chunk_batches(self, batch_size: int = 64) -> Iterator[list[ChunkRecord]]:
        """Yield every chunk in insertion order, in batches (used to rebuild the vector index)."""
        last_pk = 0
        while True:
            with self._connect() as conn:
                rows = conn.execute(
                    f"SELECT c.pk, {_CHUNK_COLUMNS} FROM chunks c"
                    " JOIN documents d ON d.id = c.doc_id WHERE c.pk > ? ORDER BY c.pk LIMIT ?",
                    (last_pk, batch_size),
                ).fetchall()
            if not rows:
                return
            last_pk = rows[-1]["pk"]
            yield [ChunkRecord(*tuple(row)[1:]) for row in rows]

    def keyword_search(
        self, query: str, limit: int, doc_ids: Sequence[str] | None = None
    ) -> list[str]:
        """BM25 keyword search. Returns chunk ids, best match first."""
        expr = keyword_query(query)
        if not self.fts_enabled or expr is None:
            return []
        sql = (
            "SELECT c.id FROM chunks_fts JOIN chunks c ON c.pk = chunks_fts.rowid"
            " WHERE chunks_fts MATCH ?"
        )
        params: list = [expr]
        if doc_ids:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            params += list(doc_ids)
        # bm25() is "smaller is better"; the section title counts for less than body text.
        sql += " ORDER BY bm25(chunks_fts, 1.0, 0.5) LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [row[0] for row in conn.execute(sql, params).fetchall()]

    def leading_chunks(
        self, per_doc: int, limit: int, doc_ids: Sequence[str] | None = None
    ) -> list[ChunkRecord]:
        """The opening chunks of each document, round-robin across documents."""
        sql = (
            f"SELECT {_CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.doc_id"
            " WHERE c.ordinal < ?"
        )
        params: list = [per_doc]
        if doc_ids:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            params += list(doc_ids)
        sql += " ORDER BY c.ordinal, d.filename COLLATE NOCASE LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [_chunk(row) for row in conn.execute(sql, params).fetchall()]

    # ── Chat history ─────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def create_session(self, title: str = NEW_CHAT_TITLE) -> dict:
        """An empty conversation, so documents can be attached before the first question."""
        now = _now()
        session = {"id": uuid.uuid4().hex, "title": title, "created_at": now, "updated_at": now}
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at)"
                " VALUES (:id, :title, :created_at, :updated_at)",
                session,
            )
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation. Its documents stay in the library."""
        with self._connect() as conn:
            conn.execute("DELETE FROM session_documents WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            return conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0

    def session_document_ids(self, session_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM session_documents WHERE session_id = ? ORDER BY added_at",
                (session_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def session_documents(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT d.* FROM documents d JOIN session_documents sd ON sd.doc_id = d.id"
                " WHERE sd.session_id = ? ORDER BY d.filename COLLATE NOCASE",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def attach_documents(self, session_id: str, doc_ids: Sequence[str]) -> None:
        now = _now()
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO session_documents (session_id, doc_id, added_at)"
                " VALUES (?, ?, ?)",
                [(session_id, doc_id, now) for doc_id in doc_ids],
            )

    def detach_document(self, session_id: str, doc_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "DELETE FROM session_documents WHERE session_id = ? AND doc_id = ?",
                (session_id, doc_id),
            ).rowcount > 0

    def document_session_ids(self, doc_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id FROM session_documents WHERE doc_id = ?", (doc_id,)
            ).fetchall()
        return [row[0] for row in rows]

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        """Messages of a session in chronological order (the most recent `limit`, if given)."""
        sql = "SELECT * FROM messages WHERE session_id = ? ORDER BY seq DESC"
        params: list = [session_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_message(row) for row in reversed(rows)]

    def save_turn(
        self,
        *,
        session_id: str,
        title: str,
        question: str,
        answer: str,
        sources: list[dict],
        meta: dict,
    ) -> str:
        """Store one question/answer exchange, creating the session on its first turn.

        Returns the id of the assistant message.
        """
        now = _now()
        answer_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET title = excluded.title,"
                " updated_at = excluded.updated_at",
                (session_id, title, now, now),
            )
            insert = (
                "INSERT INTO messages (id, session_id, role, content, sources, meta, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            conn.execute(insert, (uuid.uuid4().hex, session_id, "user", question, None, None, now))
            conn.execute(
                insert,
                (answer_id, session_id, "assistant", answer, json.dumps(sources),
                 json.dumps(meta), now),
            )
        return answer_id


def _attach_cited_documents(conn: sqlite3.Connection) -> None:
    """Upgrade from version 1, where every conversation searched the whole library:
    attach to each conversation the documents its answers cited, so old chats keep
    working with the documents they were about."""
    by_name = {row[1]: row[0] for row in conn.execute("SELECT id, filename FROM documents")}
    links = set()
    for session_id, sources in conn.execute(
        "SELECT session_id, sources FROM messages WHERE role = 'assistant' AND sources IS NOT NULL"
    ):
        for source in json.loads(sources):
            doc_id = source.get("doc_id") or by_name.get(source.get("filename"))
            if doc_id in by_name.values() and source.get("cited", True):
                links.add((session_id, doc_id))
    now = _now()
    conn.executemany(
        "INSERT OR IGNORE INTO session_documents (session_id, doc_id, added_at) VALUES (?, ?, ?)",
        [(session_id, doc_id, now) for session_id, doc_id in sorted(links)],
    )
    if links:
        logger.info("Linked %d documents to existing conversations.", len(links))


def _message(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "role": row["role"],
        "content": row["content"],
        "sources": json.loads(row["sources"]) if row["sources"] else [],
        "meta": json.loads(row["meta"]) if row["meta"] else {},
        "created_at": row["created_at"],
    }
