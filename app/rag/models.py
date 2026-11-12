"""Local embedding and reranking models (ONNX, via fastembed).

Models load lazily on first use and are cached under the data directory, so the
web server starts immediately and nothing is downloaded until it is needed.
"""

import math
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from app.logging_config import logger

# Windows without Developer Mode cannot create the symlinks the Hugging Face cache
# prefers; it falls back to copies, which is fine. Silence the repeated warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# fastembed logs a scary-looking ERROR on first download and then retries successfully;
# real failures still surface as exceptions, which the app logs itself.
try:
    from loguru import logger as _fastembed_log

    _fastembed_log.disable("fastembed")
except ImportError:  # loguru is a fastembed dependency; nothing to silence without it
    pass


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to the query, in 0..1."""
        ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: Path):
        self.name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                logger.info("Loading embedding model %s (first run downloads it)…", self.name)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(self.name, cache_dir=str(self._cache_dir))
                logger.info("Embedding model ready.")
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        return [vec.tolist() for vec in model.passage_embed(list(texts), batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        return next(iter(model.query_embed(text))).tolist()


class FastEmbedReranker:
    def __init__(self, model_name: str, cache_dir: Path):
        self.name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info("Loading reranker %s (first run downloads it)…", self.name)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextCrossEncoder(self.name, cache_dir=str(self._cache_dir))
                logger.info("Reranker ready.")
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        logits = self._load().rerank(query, list(passages), batch_size=16)
        return [_sigmoid(float(x)) for x in logits]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
