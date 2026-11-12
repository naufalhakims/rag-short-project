"""Shared types for LLM providers."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ChatRequest:
    system: str
    messages: Sequence[dict[str, str]]  # alternating user/assistant turns, ending with user
    temperature: float = 0.2
    max_tokens: int | None = None  # None = the provider's default
    fast: bool = False  # small helper call: prefer the provider's fast model


@dataclass
class StreamInfo:
    """Filled in while a response streams: who answered and what it cost."""

    provider: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_window: int | None = None


class LLMError(Exception):
    """A provider failure, classified so the router can decide what to do next.

    kind: auth | model_not_found | rate_limit | too_large | overloaded | network |
          bad_request | refusal | not_configured | unknown
    """

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.provider = provider
        self.model = model
        self.retry_after = retry_after


class Provider(Protocol):
    name: str
    default_model: str
    fast_model: str
    # Regexes (full match) for models worth trying when the configured ones are gone.
    preferred: Sequence[str]

    async def list_models(self) -> list[ModelInfo]:
        """Models this API key can use. Raises LLMError."""
        ...

    def stream(self, model: str, request: ChatRequest, info: StreamInfo) -> AsyncIterator[str]:
        """Yield the answer as text deltas. Raises LLMError."""
        ...

    async def aclose(self) -> None: ...
