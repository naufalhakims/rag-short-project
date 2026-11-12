"""Builds and holds the objects the app is made of (one set per process)."""

import threading
from dataclasses import dataclass, field

from app.config import Settings
from app.db import Database
from app.llm.router import LLMRouter, build_router
from app.logging_config import logger
from app.rag.index import Library, VectorIndex
from app.rag.models import Embedder, FastEmbedEmbedder, FastEmbedReranker, Reranker
from app.rag.pipeline import ChatService
from app.rag.retriever import Retriever


@dataclass
class Services:
    settings: Settings
    db: Database
    embedder: Embedder
    reranker: Reranker | None
    library: Library
    retriever: Retriever
    llm: LLMRouter
    chat: ChatService
    ready: threading.Event = field(default_factory=threading.Event)  # models loaded
    warmup_error: str | None = None


def build_services(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    llm: LLMRouter | None = None,
) -> Services:
    """Wire everything together. Tests pass in fake models and a fake LLM."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path)
    if embedder is None:
        embedder = FastEmbedEmbedder(settings.embedding_model, settings.model_cache_dir)
        if settings.reranker_model:
            reranker = FastEmbedReranker(settings.reranker_model, settings.model_cache_dir)
    index = VectorIndex(settings.vector_dir, embedder.name)
    library = Library(settings, db, embedder, index)
    retriever = Retriever(settings, db, embedder, index, reranker)
    llm = llm or build_router(settings)
    chat = ChatService(settings, db, retriever, llm)
    return Services(settings, db, embedder, reranker, library, retriever, llm, chat)


def warm_up(services: Services) -> None:
    """Load the local models and repair the vector index, off the request path."""
    try:
        services.library.sync_index()
        services.embedder.embed_query("warm up")
        if services.reranker:
            services.reranker.score("warm up", ["warm up"])
        services.ready.set()
        logger.info("Retrieval models are ready.")
    except Exception as exc:  # noqa: BLE001 — report it; requests will retry the load
        services.warmup_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Could not load the retrieval models")
