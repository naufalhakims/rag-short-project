"""Chooses which provider and model answers a request, and fails over when one breaks.

Two things make this resilient to the usual ways hosted models fail:

* Model discovery — each provider is asked which models the API key can use. If the
  configured model has been retired, the next best available one is used instead.
* Failover — on a rate limit, outage or missing model the request moves on to the
  next model, then the next provider, as long as no text has been streamed yet.
"""

import asyncio
import re
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing

from app.config import PROVIDER_ORDER, Settings
from app.llm.base import ChatRequest, LLMError, ModelInfo, Provider, StreamInfo
from app.llm.openai_compat import OpenAICompatProvider, groq_extra_body, is_chat_model
from app.logging_config import logger

_MODELS_TTL = 15 * 60  # seconds a successful model listing is trusted
_MODELS_RETRY = 60  # seconds before retrying a failed listing
_MAX_INLINE_WAIT = 6.0  # longest rate-limit pause worth sitting through
_MODELS_PER_PROVIDER = 3

# Which failure to report when several attempts failed for different reasons.
_REPORT_ORDER = (
    "rate_limit", "too_large", "overloaded", "network", "auth", "model_not_found",
    "refusal", "bad_request", "unknown",
)
_KEY_NAMES = {
    "groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY",
}


def _version_key(model_id: str) -> tuple:
    """Sort key that orders "gemini-3.8-flash" after "gemini-3.1-flash"."""
    return tuple(float(n) for n in re.findall(r"\d+(?:\.\d+)?", model_id))


class LLMRouter:
    def __init__(self, providers: Sequence[Provider], *, fallback: bool = True):
        self._providers = list(providers)
        self._fallback = fallback
        # provider name -> (expiry, models or None if listing failed, the listing error)
        self._models: dict[str, tuple[float, list[ModelInfo] | None, LLMError | None]] = {}
        self._warned: set[str] = set()

    @property
    def configured(self) -> bool:
        return bool(self._providers)

    async def aclose(self) -> None:
        for provider in self._providers:
            await provider.aclose()

    # ── Model discovery ──────────────────────────────────────────

    async def _available(
        self, provider: Provider
    ) -> tuple[list[ModelInfo] | None, LLMError | None]:
        """(models the key can use, listing error). Models are None when listing failed."""
        cached = self._models.get(provider.name)
        if cached and cached[0] > time.monotonic():
            return cached[1], cached[2]
        models, error, ttl = None, None, _MODELS_RETRY
        try:
            models, ttl = await provider.list_models(), _MODELS_TTL
        except LLMError as exc:
            error = exc
        except Exception as exc:  # noqa: BLE001 — a listing failure must never break chat
            error = LLMError("unknown", str(exc), provider=provider.name)
        if error:
            logger.warning(
                "Could not list %s models (%s): %s", provider.name, error.kind, error.message
            )
        self._models[provider.name] = (time.monotonic() + ttl, models, error)
        return models, error

    async def _model_chain(self, provider: Provider, fast: bool) -> list[str]:
        """Models to try on this provider, best first."""
        first, second = provider.default_model, provider.fast_model
        if fast:
            first, second = second or first, first
        wanted = [m for m in dict.fromkeys([first, second]) if m]

        available, error = await self._available(provider)
        if available is None:
            if error and error.kind == "auth":
                return []  # the key is rejected; no model will work
            return wanted[:_MODELS_PER_PROVIDER]  # cannot verify — trust the configuration

        ids = [m.id for m in available if is_chat_model(m.id)]
        chain = [m for m in wanted if m in ids]
        for pattern in provider.preferred:
            matches = sorted(
                (i for i in ids if re.fullmatch(pattern, i)), key=_version_key, reverse=True
            )
            chain += [m for m in matches[:1] if m not in chain]
        if not chain:  # nothing we know by name: take the largest-context chat models
            by_context = {m.id: m.context_window or 0 for m in available}
            chain = sorted(ids, key=lambda i: by_context.get(i, 0), reverse=True)

        if wanted and chain and chain[0] != wanted[0]:
            note = f"{provider.name}:{wanted[0]}->{chain[0]}"
            if note not in self._warned:
                self._warned.add(note)
                logger.warning(
                    "%s model '%s' is not available to this API key; using '%s' instead.",
                    provider.name, wanted[0], chain[0],
                )
        return chain[:_MODELS_PER_PROVIDER]

    # ── Generation ───────────────────────────────────────────────

    async def stream(self, request: ChatRequest, info: StreamInfo) -> AsyncIterator[str]:
        """Stream the answer from the first provider/model that works. Raises LLMError."""
        if not self._providers:
            raise LLMError(
                "not_configured",
                "No LLM provider is configured. Add GROQ_API_KEY (free at console.groq.com)"
                " or another provider's key to the .env file, then restart.",
            )
        errors: list[LLMError] = []
        providers = self._providers if self._fallback else self._providers[:1]
        for provider in providers:
            models = await self._model_chain(provider, request.fast)
            if not models:
                _, listing_error = await self._available(provider)
                errors.append(
                    listing_error
                    or LLMError(
                        "model_not_found", "no usable chat model found", provider=provider.name
                    )
                )
            for model in models if self._fallback else models[:1]:
                waited = False
                while True:
                    started = False
                    info.model, info.usage = "", {}  # nothing carried over from a failed attempt
                    try:
                        async with aclosing(provider.stream(model, request, info)) as deltas:
                            async for delta in deltas:
                                started = True
                                yield delta
                        info.provider = provider.name
                        info.model = info.model or model
                        return
                    except LLMError as exc:
                        error = exc
                    except Exception as exc:  # noqa: BLE001 — SDK/transport surprises
                        error = LLMError("unknown", str(exc), provider=provider.name, model=model)
                    if started:
                        raise error  # mid-answer: switching models would garble the reply
                    short_wait = (
                        error.kind == "rate_limit"
                        and error.retry_after is not None
                        and error.retry_after <= _MAX_INLINE_WAIT
                    )
                    if short_wait and not waited:
                        waited = True
                        await asyncio.sleep(error.retry_after + 0.25)
                        continue
                    break
                errors.append(error)
                logger.warning(
                    "%s/%s failed (%s): %s", provider.name, model, error.kind, error.message[:200]
                )
                if error.kind == "model_not_found":
                    self._models.pop(provider.name, None)  # re-discover next time
                if error.kind == "auth":
                    break  # every model on this provider will fail the same way
        raise _summarise(errors)

    async def complete(self, request: ChatRequest) -> str:
        """Collect a whole (short) response."""
        async with aclosing(self.stream(request, StreamInfo())) as deltas:
            return "".join([delta async for delta in deltas]).strip()

    async def describe(self) -> dict:
        """Provider status for the health endpoint."""
        providers = []
        for provider in self._providers:
            chain = await self._model_chain(provider, fast=False)
            _, error = await self._available(provider)
            providers.append(
                {
                    "name": provider.name,
                    "model": chain[0] if chain else None,
                    "fallback_models": chain[1:],
                    "error": error.message if error else None,
                }
            )
        active = next((p for p in providers if p["model"] and not p["error"]), None)
        active = active or next((p for p in providers if p["model"]), None)
        return {
            "configured": self.configured,
            "provider": active["name"] if active else None,
            "model": active["model"] if active else None,
            "providers": providers,
        }


def _summarise(errors: Sequence[LLMError]) -> LLMError:
    """Turn a list of failed attempts into one error with a message a user can act on."""
    def rank(error: LLMError) -> int:
        return _REPORT_ORDER.index(error.kind) if error.kind in _REPORT_ORDER else 99

    error = min(errors, key=rank)
    who = error.provider.capitalize() or "The LLM provider"
    waits = [e.retry_after for e in errors if e.kind == "rate_limit" and e.retry_after]
    retry_after = min(waits) if waits else None
    if error.kind == "rate_limit":
        wait = f"in about {_duration(retry_after)}" if retry_after else "shortly"
        message = f"{who} rate limit reached. Try again {wait}."
    elif error.kind == "too_large":
        message = f"The request was too large for {who}'s token limit. Try a shorter question."
    elif error.kind == "overloaded":
        message = f"{who} is temporarily unavailable. Please try again in a moment."
    elif error.kind == "network":
        message = f"Could not reach {who}. Check your internet connection."
    elif error.kind == "auth":
        key = _KEY_NAMES.get(error.provider, "the API key")
        message = f"{who} rejected the API key. Check {key} in your .env file."
    elif error.kind == "model_not_found":
        if error.model:
            message = f"The model '{error.model}' is not available on {who}."
        else:
            message = f"No usable chat model was found on {who}."
    elif error.kind == "refusal":
        message = error.message
    else:
        message = f"{who} returned an error: {error.message}"
    return LLMError(
        error.kind, message, provider=error.provider, model=error.model, retry_after=retry_after
    )


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} seconds"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds / 3600)} hours"


def build_router(settings: Settings) -> LLMRouter:
    """Create the providers that have credentials, in priority order."""
    s = settings
    cap = s.llm_max_output_tokens
    available: dict[str, Provider] = {}

    if s.groq_api_key:
        available["groq"] = OpenAICompatProvider(
            "groq",
            base_url=s.groq_base_url,
            api_key=s.groq_api_key,
            default_model=s.groq_model,
            fast_model=s.groq_fast_model,
            preferred=(r"openai/gpt-oss-120b", r"openai/gpt-oss-20b", r"qwen/.+", r".*llama.*"),
            # Groq counts max_completion_tokens against the per-minute token limit
            # (8,000 on the free tier), so keep the reservation modest.
            max_output_tokens=cap or 1024,
            fast_max_output_tokens=300,
            extra_body=groq_extra_body,
            timeout=s.llm_timeout_seconds,
        )
    if s.anthropic_api_key or s.llm_provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        available["anthropic"] = AnthropicProvider(
            api_key=s.anthropic_api_key,
            default_model=s.anthropic_model,
            fast_model=s.anthropic_fast_model,
            effort=s.anthropic_effort,
            max_output_tokens=cap or 16000,
            timeout=s.llm_timeout_seconds,
        )
    if s.google_api_key:
        flash_lite, flash = r"gemini-[\d.]+-flash-lite", r"gemini-[\d.]+-flash"
        available["google"] = OpenAICompatProvider(
            "google",
            base_url=s.google_base_url,
            api_key=s.google_api_key,
            default_model=s.gemini_model,
            fast_model=s.gemini_fast_model,
            preferred=(flash_lite, flash),  # flash-lite has the roomiest free quota
            max_output_tokens=cap or 4096,
            timeout=s.llm_timeout_seconds,
        )
    if s.openai_model and (s.openai_api_key or s.llm_provider == "openai"):
        available["openai"] = OpenAICompatProvider(
            "openai",
            base_url=s.openai_base_url,
            api_key=s.openai_api_key,
            default_model=s.openai_model,
            fast_model=s.openai_fast_model,
            max_output_tokens=cap or 4096,
            timeout=s.llm_timeout_seconds,
        )

    order = list(PROVIDER_ORDER)
    if s.llm_provider in order:  # a named provider goes first
        order.remove(s.llm_provider)
        order.insert(0, s.llm_provider)
    elif s.llm_provider != "auto":
        logger.warning("Unknown LLM_PROVIDER '%s'; using auto.", s.llm_provider)
    providers = [available[name] for name in order if name in available]
    if s.llm_provider in PROVIDER_ORDER and s.llm_provider not in available:
        logger.warning(
            "LLM_PROVIDER=%s but it has no credentials; using %s.",
            s.llm_provider, providers[0].name if providers else "nothing",
        )
    return LLMRouter(providers, fallback=s.llm_fallback)
