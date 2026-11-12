"""Provider for any OpenAI-compatible chat completions API.

Groq, Google Gemini (its OpenAI endpoint), OpenAI, OpenRouter, Ollama and
LM Studio all speak this protocol, so one small httpx client covers them.
"""

import json
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing

import httpx

from app.llm.base import ChatRequest, LLMError, ModelInfo, StreamInfo

# Model ids that are not general chat models (speech, safety classifiers, embeddings…).
_NOT_CHAT = re.compile(
    r"whisper|tts|speech|audio|guard|embed|rerank|moderation|orpheus|imagen|image|veo"
    r"|aqa|learnlm|robotics|computer-use|live|gemma|nano-banana|deep-research",
    re.IGNORECASE,
)
_RETRY_HINT = re.compile(
    r"(?:try again|retry) in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?", re.IGNORECASE
)


def is_chat_model(model_id: str) -> bool:
    return not _NOT_CHAT.search(model_id)


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        api_key: str,
        default_model: str,
        fast_model: str = "",
        preferred: Sequence[str] = (),
        max_output_tokens: int = 4096,
        fast_max_output_tokens: int = 512,
        extra_body: Callable[[str], dict] | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.name = name
        self.default_model = default_model
        self.fast_model = fast_model
        self.preferred = tuple(preferred)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_output_tokens = max_output_tokens
        self._fast_max_output_tokens = fast_max_output_tokens
        self._extra_body = extra_body
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.get(f"{self._base_url}/models", headers=self._headers())
        except httpx.HTTPError as exc:
            raise self._network_error(exc) from exc
        if response.status_code != 200:
            raise self._http_error(response.status_code, response.text, response.headers)
        payload = response.json()
        items = payload.get("data", []) if isinstance(payload, dict) else payload
        models = []
        for item in items:
            if not isinstance(item, dict) or item.get("active") is False:
                continue
            model_id = str(item.get("id", "")).removeprefix("models/")
            if model_id:
                models.append(ModelInfo(model_id, item.get("context_window")))
        return models

    async def stream(
        self, model: str, request: ChatRequest, info: StreamInfo
    ) -> AsyncIterator[str]:
        default_cap = self._fast_max_output_tokens if request.fast else self._max_output_tokens
        body = {
            "model": model,
            "messages": [{"role": "system", "content": request.system}, *request.messages],
            "stream": True,
            "temperature": request.temperature,
            "max_completion_tokens": request.max_tokens or default_cap,
        }
        # Optional extras: token usage on the final chunk, plus provider-specific tuning.
        # If an API rejects them, the request is repeated without.
        extras = {"stream_options": {"include_usage": True}}
        if self._extra_body:
            extras.update(self._extra_body(model))
        variants = [{**body, **extras}, body]
        for attempt, payload in enumerate(variants, start=1):
            started = False
            try:
                async with aclosing(self._stream_once(payload, model, info)) as deltas:
                    async for delta in deltas:
                        started = True
                        yield delta
                return
            except LLMError as exc:
                if exc.kind != "bad_request" or started or attempt == len(variants):
                    raise

    async def _stream_once(
        self, payload: dict, model: str, info: StreamInfo
    ) -> AsyncIterator[str]:
        think = _ThinkFilter()
        try:
            async with self._client.stream(
                "POST", f"{self._base_url}/chat/completions", json=payload, headers=self._headers()
            ) as response:
                if response.status_code != 200:
                    raw = (await response.aread()).decode("utf-8", "replace")
                    raise self._http_error(response.status_code, raw, response.headers, model)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        raise self._http_error(500, json.dumps(event), {}, model)
                    info.model = event.get("model") or info.model or model
                    usage = event.get("usage") or (event.get("x_groq") or {}).get("usage")
                    if usage:
                        info.usage = {
                            "input_tokens": int(usage.get("prompt_tokens") or 0),
                            "output_tokens": int(usage.get("completion_tokens") or 0),
                        }
                    for choice in event.get("choices") or []:
                        text = think.feed((choice.get("delta") or {}).get("content") or "")
                        if text:
                            yield text
            tail = think.flush()
            if tail:
                yield tail
        except httpx.HTTPError as exc:
            raise self._network_error(exc, model) from exc

    def _network_error(self, exc: Exception, model: str = "") -> LLMError:
        what = "timed out" if isinstance(exc, httpx.TimeoutException) else "could not be reached"
        return LLMError("network", f"{self.name} {what}.", provider=self.name, model=model)

    def _http_error(
        self, status: int, raw: str, headers: Mapping[str, str], model: str = ""
    ) -> LLMError:
        message, code = _parse_error_body(raw)
        text = f"{code} {message}".lower()
        if status in (401, 403):
            kind = "auth"
        elif status == 404 or "model_not_found" in text or "decommissioned" in text:
            kind = "model_not_found"
        elif status == 429:
            kind = "rate_limit"
        elif status == 413:
            kind = "too_large"
        elif status in (400, 422):
            kind = "bad_request"
        elif status >= 500:
            kind = "overloaded"
        else:
            kind = "unknown"
        return LLMError(
            kind,
            message or f"HTTP {status}",
            provider=self.name,
            model=model,
            retry_after=_retry_after(headers, message) if kind == "rate_limit" else None,
        )


def _parse_error_body(raw: str) -> tuple[str, str]:
    """Extract (message, code) from the JSON error shapes providers return."""
    try:
        body = json.loads(raw)
    except ValueError:
        return raw.strip()[:300], ""
    if isinstance(body, list) and body:  # Gemini wraps errors in a list
        body = body[0]
    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        code = str(error.get("code") or error.get("status") or error.get("type") or "")
        return message[:500], code
    return str(error)[:300], ""


def _retry_after(headers: Mapping[str, str], message: str) -> float | None:
    """Seconds to wait, from the Retry-After header or a "try again in 7.6s" hint."""
    try:
        return float(headers.get("retry-after", ""))
    except ValueError:
        pass
    match = _RETRY_HINT.search(message)
    if not match or not any(match.groups()[:3]):
        return None
    hours, minutes, amount, unit = match.groups()
    seconds = int(hours or 0) * 3600 + int(minutes or 0) * 60
    if amount:
        seconds += float(amount) / (1000 if unit == "ms" else 1)
    return seconds


class _ThinkFilter:
    """Drops a leading <think>…</think> block that some open models emit inline."""

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._state = "start"  # start -> thinking -> passthrough

    def feed(self, text: str) -> str:
        if self._state == "passthrough":
            return text
        self._buffer += text
        if self._state == "start":
            head = self._buffer.lstrip()
            if head.startswith(self._OPEN):
                self._state = "thinking"
            elif self._OPEN.startswith(head):
                return ""  # could still become "<think>"; wait for more
            else:
                return self._release(self._buffer)
        end = self._buffer.find(self._CLOSE)
        if end == -1:
            return ""
        return self._release(self._buffer[end + len(self._CLOSE):].lstrip())

    def flush(self) -> str:
        return self._release(self._buffer) if self._state == "start" else ""

    def _release(self, text: str) -> str:
        self._state = "passthrough"
        self._buffer = ""
        return text


def groq_extra_body(model: str) -> dict:
    """Keep gpt-oss reasoning short and out of the answer stream."""
    if "gpt-oss" in model:
        return {"reasoning_effort": "low", "include_reasoning": False}
    return {}
