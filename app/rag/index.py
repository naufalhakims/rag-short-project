"""The document library: ingestion, deletion, and the Chroma vector index."""

import hashlib
import re
import threading
from collections.abc import Sequence
from pathlib import Path, PurePath

from app.config import Settings
from app.db import ChunkRecord, Database
from app.ingest.chunking import chunk_blocks
from app.ingest.loaders import IngestError, load_blocks
from app.logging_config import logger
from app.rag.models import Embedder

_COLLECTION_PREFIX = "chunks-"
_ADD_BATCH = 256


def passage_text(filename: str, heading: str, text: str) -> str:
    """Chunk text prefixed with where it comes from — what gets embedded and reranked.

    A chunk like "It triples all points." is far easier to match once it carries
    its document title and section heading.
    """
    title = PurePath(filename).stem.replace("_", " ")
    header = f"{title} — {heading}" if heading else title
    return f"{header}\n{text}"


def _collection_name(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
    return (_COLLECTION_PREFIX + slug)[:63].rstrip("-")


class VectorIndex:
    """Embeddings in a persistent Chroma collection, keyed by chunk id.

    The collection is named after the embedding model, so switching models
    starts from an empty collection and `Library.sync_index` re-embeds everything.
    """

    def __init__(self, path: Path, model_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._name = _collection_name(model_name)
        self._lock = threading.Lock()
        for stale in self._client.list_collections():
            if stale.name.startswith(_COLLECTION_PREFIX) and stale.name != self._name:
                logger.info("Dropping vectors from a previous embedding model: %s", stale.name)
                self._client.delete_collection(stale.name)
        self._collection = self._open()

    def _open(self):
        return self._client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def count(self) -> int:
        return self._collection.count()

    def add(
        self, chunk_ids: Sequence[str], vectors: Sequence[Sequence[float]], doc_ids: Sequence[str]
    ) -> None:
        with self._lock:
            for start in range(0, len(chunk_ids), _ADD_BATCH):
                end = start + _ADD_BATCH
                self._collection.add(
                    ids=list(chunk_ids[start:end]),
                    embeddings=[list(v) for v in vectors[start:end]],
                    metadatas=[{"doc_id": d} for d in doc_ids[start:end]],
                )

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._collection.delete(where={"doc_id": doc_id})

    def clear(self) -> None:
        with self._lock:
            self._client.delete_collection(self._name)
            self._collection = self._open()

    def search(
        self, vector: Sequence[float], limit: int, doc_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        """Nearest chunks as (chunk_id, cosine distance), closest first."""
        limit = min(limit, self.count())
        if limit <= 0:
            return []
        where = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
        result = self._collection.query(
            query_embeddings=[list(vector)], n_results=limit, where=where, include=["distances"]
        )
        return list(zip(result["ids"][0], result["distances"][0], strict=True))


class Library:
    """Adds and removes documents, keeping SQLite and the vector index in step."""

    def __init__(self, settings: Settings, db: Database, embedder: Embedder, index: VectorIndex):
        self._settings = settings
        self._db = db
        self._embedder = embedder
        self._index = index
        self._write_lock = threading.Lock()

    def add(self, filename: str, data: bytes) -> dict:
        """Ingest a file. Re-uploading a file name replaces the earlier version.

        Returns {"document": <row>, "status": "added" | "replaced" | "unchanged"}.
        Raises IngestError for problems the user can fix.
        """
        filename = _safe_filename(filename)
        content_hash = hashlib.sha256(data).hexdigest()
        existing = self._db.get_document_by_filename(filename)
        if existing and existing["content_hash"] == content_hash:
            return {"document": existing, "status": "unchanged"}

        blocks = load_blocks(filename, data)
        chunks = chunk_blocks(blocks, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            raise IngestError("The file contains no readable text.")
        vectors = self._embedder.embed_documents(
            [passage_text(filename, c.heading, c.text) for c in chunks]
        )

        with self._write_lock:
            existing = self._db.get_document_by_filename(filename)
            if existing:
                self._remove(existing["id"])
            doc_id, chunk_ids = self._db.add_document(
                filename=filename,
                content_hash=content_hash,
                size_bytes=len(data),
                char_count=sum(len(b.text) for b in blocks),
                chunks=chunks,
            )
            try:
                self._index.add(chunk_ids, vectors, [doc_id] * len(chunk_ids))
            except Exception:
                self._remove(doc_id)  # never leave a document that cannot be searched
                raise
        logger.info("Ingested '%s': %d chunks.", filename, len(chunks))
        return {
            "document": self._db.get_document(doc_id),
            "status": "replaced" if existing else "added",
        }

    def remove(self, doc_id: str) -> bool:
        with self._write_lock:
            return self._remove(doc_id)

    def _remove(self, doc_id: str) -> bool:
        self._index.delete_document(doc_id)
        return self._db.delete_document(doc_id)

    def sync_index(self) -> int:
        """Rebuild the vector index from SQLite if the two disagree.

        This happens after changing EMBEDDING_MODEL, or if the index was deleted.
        Returns the number of chunks re-embedded.
        """
        with self._write_lock:
            expected = self._db.count_chunks()
            if self._index.count() == expected:
                return 0
            logger.info("Rebuilding the vector index for %d chunks…", expected)
            self._index.clear()
            done = 0
            for batch in self._db.iter_chunk_batches():
                vectors = self._embedder.embed_documents([_passage(c) for c in batch])
                self._index.add([c.id for c in batch], vectors, [c.doc_id for c in batch])
                done += len(batch)
            logger.info("Vector index rebuilt (%d chunks).", done)
            return done


def _passage(chunk: ChunkRecord) -> str:
    return passage_text(chunk.filename, chunk.heading, chunk.text)


def _safe_filename(filename: str) -> str:
    name = PurePath((filename or "").replace("\\", "/")).name
    name = re.sub(r"\s+", " ", name).strip()
    return name[:150] or "document.txt"
