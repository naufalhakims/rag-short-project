"""The Claude provider, exercised against a stand-in for the SDK client (no network)."""

import asyncio
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import ChatRequest, LLMError, StreamInfo

REQUEST = ChatRequest(system="Be brief.", messages=[{"role": "user", "content": "Hi"}])


def api_error(cls, status, headers=None):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls("boom", response=response, body=None)


class StubStream:
    def __init__(self, texts, stop_reason="end_turn", error=None):
        self._texts, self._stop_reason, self._error = texts, stop_reason, error

    async def __aenter__(self):
        if self._error:
            raise self._error
        return self

    async def __aexit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        async def deltas():
            for text in self._texts:
                yield text

        return deltas()

    async def get_final_message(self):
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            model="claude-served",
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


class StubMessages:
    def __init__(self, calls, label, outcomes):
        self._calls, self._label, self._outcomes = calls, label, outcomes

    def stream(self, **kwargs):
        self._calls.append((self._label, kwargs))
        return self._outcomes.pop(0)


def make(outcomes, **kwargs):
    calls = []
    client = SimpleNamespace(
        messages=StubMessages(calls, "stable", outcomes),
        beta=SimpleNamespace(messages=StubMessages(calls, "beta", outcomes)),
    )
    return AnthropicProvider(client=client, **kwargs), calls


def run(provider, model, request=REQUEST):
    info = StreamInfo()

    async def collect():
        return [delta async for delta in provider.stream(model, request, info)]

    return asyncio.run(collect()), info


def test_current_models_get_effort_and_refusal_fallbacks():
    provider, calls = make([StubStream(["Hel", "lo"])], effort="medium")
    deltas, info = run(provider, "claude-opus-5-5")
    assert deltas == ["Hel", "lo"]
    assert info.model == "claude-served" and info.usage == {"input_tokens": 11, "output_tokens": 7}

    label, kwargs = calls[0]
    assert label == "beta"
    assert kwargs["fallbacks"] == "default" and kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["system"] == "Be brief." and kwargs["max_tokens"] == 16000
    assert "temperature" not in kwargs  # current Claude models reject sampling parameters


def test_haiku_is_called_without_the_fallback_beta_and_fast_calls_use_low_effort():
    provider, calls = make([StubStream(["ok"])])
    fast = ChatRequest(system="s", messages=REQUEST.messages, fast=True)
    run(provider, "claude-haiku-5-5", fast)
    label, kwargs = calls[0]
    assert label == "stable" and "fallbacks" not in kwargs
    assert kwargs["output_config"] == {"effort": "low"} and kwargs["max_tokens"] == 2000


def test_rejected_options_fall_back_to_a_plain_request():
    rejected = StubStream([], error=api_error(anthropic.BadRequestError, 400))
    provider, calls = make([rejected, StubStream(["plain"])])
    deltas, _ = run(provider, "claude-opus-5-5")
    assert deltas == ["plain"]
    label, kwargs = calls[1]
    assert label == "stable" and "output_config" not in kwargs and "fallbacks" not in kwargs


def test_refusal_is_an_error_not_an_answer():
    provider, _ = make([StubStream(["partial"], stop_reason="refusal")])
    with pytest.raises(LLMError) as caught:
        run(provider, "claude-opus-5-5")
    assert caught.value.kind == "refusal"


@pytest.mark.parametrize(
    "cls, status, kind",
    [
        (anthropic.AuthenticationError, 401, "auth"),
        (anthropic.NotFoundError, 404, "model_not_found"),
        (anthropic.RateLimitError, 429, "rate_limit"),
        (anthropic.InternalServerError, 529, "overloaded"),
    ],
)
def test_sdk_errors_are_classified(cls, status, kind):
    failing = StubStream([], error=api_error(cls, status, {"retry-after": "9"}))
    provider, _ = make([failing])
    with pytest.raises(LLMError) as caught:
        run(provider, "claude-opus-5-5")
    assert caught.value.kind == kind
    assert caught.value.retry_after == (9 if kind == "rate_limit" else None)
