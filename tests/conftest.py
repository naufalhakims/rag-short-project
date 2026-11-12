"""Shared fixtures. Tests use small fake models and a fake LLM: no downloads, no network."""

import hashlib
import math
import re

import pytest

from app.config import Settings
from app.llm.base import ModelInfo
from app.llm.router import LLMRouter
from app.services import build_services

GUIDE = """The Official Guide to Gravity Ball (Season 2026)

1. Overview
Gravity Ball is a high-intensity orbital sport played in zero-gravity arenas on the Lunar Surface. Two teams of four players try to launch a magnetized obsidian sphere into the opposing team's Vortex Goal.

2. Scoring System
A standard goal is worth 2 points. A Long-Shot from beyond 20 meters is worth 5 points.

3. Key Rules and Penalties
Thruster Overheat: players cannot use continuous thrust for more than 4 seconds. Doing so results in a Dead Drift penalty of 30 seconds.

4. Famous Teams
The reigning champions are the Neptune Knights, led by their star striker Kaelen Voss.
"""

PAPER = """# Neuromorphic Swarms

## Abstract
This paper explores spiking neural networks for autonomous drone swarms coordinated by a blockchain ledger.

## Results
The hybrid architecture maintains 98% consensus accuracy even when 15% of the drones are malfunctioning. It consumed 1.2 Watts compared to 45 Watts for a GPU baseline.
"""

_WORDS = re.compile(r"\w+")
_COMMON = {"the", "a", "an", "is", "of", "to", "in", "what", "who", "how", "for", "and", "are"}


def _words(text: str) -> list[str]:
    return [w for w in _WORDS.findall(text.lower()) if w not in _COMMON]


class FakeEmbedder:
    """Bag-of-words hashing: texts that share words get similar vectors."""

    name = "fake-embedder"
    dim = 96

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _words(text):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeReranker:
    """Relevance = share of the query's words that appear in the passage."""

    name = "fake-reranker"

    def score(self, query, passages):
        wanted = set(_words(query))
        return [len(wanted & set(_words(p))) / max(1, len(wanted)) for p in passages]


class FakeProvider:
    """An LLM provider that replays a canned answer and records what it was asked."""

    preferred = ()

    def __init__(self, name="fake", reply="Kaelen Voss is the star striker [1].", errors=()):
        self.name = name
        self.default_model = f"{name}-large"
        self.fast_model = f"{name}-small"
        self.reply = reply
        self.errors = list(errors)  # raised one per call, before any output
        self.calls = []

    async def list_models(self):
        return [ModelInfo(self.default_model), ModelInfo(self.fast_model)]

    async def stream(self, model, request, info):
        self.calls.append((model, request))
        if self.errors:
            raise self.errors.pop(0)
        text = "standalone search query" if request.fast else self.reply
        for word in text.split(" "):
            yield word + " "
        info.usage = {"input_tokens": 10, "output_tokens": 5}

    async def aclose(self):
        pass


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in ("GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
                 "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None, data_dir=tmp_path / "data", chunk_size=300, chunk_overlap=40)


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def services(settings, provider):
    return build_services(
        settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm=LLMRouter([provider])
    )


@pytest.fixture
def library(services):
    """Services with the two sample documents already ingested."""
    services.library.add("gravity_ball_guide.txt", GUIDE.encode())
    services.library.add("swarm_paper.md", PAPER.encode())
    return services
