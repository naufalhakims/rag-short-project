"""Provider for Anthropic's Claude models, using the official SDK."""

from collections.abc import AsyncIterator

import anthropic

from app.llm.base import ChatRequest, LLMError, ModelInfo, StreamInfo

# Claude's safety classifiers can decline a request ("refusal" stop reason). With this
# beta the API re-runs a declined request on Anthropic's recommended fallback model
# inside the same call, instead of returning the refusal.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5-5")


class AnthropicProvider:
    name = "anthropic"
    preferred = ()  # only ever use the models the user configured

    def __init__(
        self,
        *,
        api_key: str = "",
        default_model: str = "claude-opus-5-5",
        fast_model: str = "",
        effort: str = "medium",
        max_output_tokens: int = 16000,
        fast_max_output_tokens: int = 2000,
        timeout: float = 60.0,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.default_model = default_model
        self.fast_model = fast_model or default_model
        self._effort = effort
        self._max_output_tokens = max_output_tokens
        self._fast_max_output_tokens = fast_max_output_tokens
        # Without an explicit key the SDK resolves credentials from the environment.
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key or None, timeout=timeout, max_retries=1
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def list_models(self) -> list[ModelInfo]:
        try:
            return [ModelInfo(m.id) async for m in self._client.models.list(limit=100)]
        except anthropic.APIError as exc:
            raise self._translate(exc) from exc

    async def stream(
        self, model: str, request: ChatRequest, info: StreamInfo
    ) -> AsyncIterator[str]:
        default_cap = self._fast_max_output_tokens if request.fast else self._max_output_tokens
        base = {
            "model": model,
            "max_tokens": request.max_tokens or default_cap,
            "system": request.system,
            "messages": list(request.messages),
        }
        # Current models take an effort level instead of sampling parameters. Older
        # models reject it, so a 400 falls back to the plain request.
        tuned = {**base, "output_config": {"effort": "low" if request.fast else self._effort}}
        if model.startswith(_FALLBACK_MODELS):
            first = self._client.beta.messages.stream(
                **tuned, betas=[_FALLBACK_BETA], fallbacks="default"
            )
        else:
            first = self._client.messages.stream(**tuned)
        variants = [first, None]  # None = build the plain request lazily, only if needed

        for attempt, manager in enumerate(variants, start=1):
            manager = manager or self._client.messages.stream(**base)
            started = False
            try:
                async with manager as stream:
                    async for text in stream.text_stream:
                        started = True
                        yield text
                    final = await stream.get_final_message()
            except anthropic.BadRequestError as exc:
                if started or attempt == len(variants):
                    raise self._translate(exc, model) from exc
                continue
            except anthropic.APIError as exc:
                raise self._translate(exc, model) from exc

            if final.stop_reason == "refusal":
                # Partial text from a declined response must not be treated as an answer.
                raise LLMError(
                    "refusal", "The model declined to answer this request.",
                    provider=self.name, model=model,
                )
            info.model = final.model
            info.usage = {
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
            return

    def _translate(self, exc: anthropic.APIError, model: str = "") -> LLMError:
        retry_after = None
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            kind = "auth"
        elif isinstance(exc, anthropic.NotFoundError):
            kind = "model_not_found"
        elif isinstance(exc, anthropic.RateLimitError):
            kind = "rate_limit"
            try:
                retry_after = float(exc.response.headers.get("retry-after", ""))
            except ValueError:
                pass
        elif isinstance(exc, anthropic.BadRequestError):
            kind = "bad_request"
        elif isinstance(exc, anthropic.APIStatusError):
            kind = "overloaded" if exc.status_code >= 500 else "unknown"
        elif isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
            kind = "network"
        else:
            kind = "unknown"
        return LLMError(
            kind, getattr(exc, "message", str(exc)), provider=self.name, model=model,
            retry_after=retry_after,
        )
