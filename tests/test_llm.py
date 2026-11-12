"""LLM layer: the OpenAI-compatible client, model discovery, and failover."""

import asyncio
import json

import httpx
import pytest

from app.llm.base import ChatRequest, LLMError, StreamInfo
from app.llm.openai_compat import OpenAICompatProvider, _retry_after, _ThinkFilter, groq_extra_body
from app.llm.router import LLMRouter
from tests.conftest import FakeProvider

REQUEST = ChatRequest(system="You are helpful.", messages=[{"role": "user", "content": "Hi"}])


def sse(*deltas, usage=None):
    events = [{"model": "m", "choices": [{"delta": {"content": d}}]} for d in deltas]
    if usage:
        events.append({"choices": [], "usage": usage})
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def error(status, message, code="", headers=None):
    return httpx.Response(
        status, json={"error": {"message": message, "code": code}}, headers=headers or {}
    )


def make_provider(handler, name="groq", **kwargs):
    kwargs.setdefault("default_model", "big-model")
    kwargs.setdefault("fast_model", "small-model")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider(
        name, base_url="https://llm.test/v1", api_key="secret", client=client, **kwargs
    )


def collect(router_or_provider, *args):
    async def run():
        return [delta async for delta in router_or_provider.stream(*args)]

    return asyncio.run(run())


def models_response(*ids):
    return httpx.Response(200, json={"data": [{"id": i} for i in ids]})


# ── OpenAI-compatible provider ───────────────────────────────────


def test_stream_parses_deltas_usage_and_sends_the_right_request():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return sse("Hel", "lo", usage={"prompt_tokens": 12, "completion_tokens": 3})

    info = StreamInfo()
    provider = make_provider(handler)
    assert collect(provider, "big-model", REQUEST, info) == ["Hel", "lo"]
    assert info.usage == {"input_tokens": 12, "output_tokens": 3}
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "big-model" and seen["body"]["stream"] is True
    assert seen["body"]["messages"][0] == {"role": "system", "content": "You are helpful."}


@pytest.mark.parametrize(
    "response, kind",
    [
        (error(401, "bad key"), "auth"),
        (error(404, "The model `x` does not exist", "model_not_found"), "model_not_found"),
        (error(400, "The model `x` has been decommissioned", "model_decommissioned"),
         "model_not_found"),
        (error(429, "Rate limit reached. Please try again in 7.5s."), "rate_limit"),
        (error(413, "Request too large"), "too_large"),
        (error(400, "bad parameter"), "bad_request"),
        (error(503, "overloaded"), "overloaded"),
    ],
)
def test_http_errors_are_classified(response, kind):
    provider = make_provider(lambda request: response)
    with pytest.raises(LLMError) as caught:
        collect(provider, "big-model", REQUEST, StreamInfo())
    assert caught.value.kind == kind
    if kind == "rate_limit":
        assert caught.value.retry_after == 7.5


def test_gemini_style_error_list_is_understood():
    response = httpx.Response(429, json=[{"error": {"message": "Quota exceeded", "status": "X"}}])
    provider = make_provider(lambda request: response)
    with pytest.raises(LLMError, match="Quota exceeded"):
        collect(provider, "big-model", REQUEST, StreamInfo())


def test_network_failure_is_reported_as_network():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(LLMError) as caught:
        collect(make_provider(handler), "big-model", REQUEST, StreamInfo())
    assert caught.value.kind == "network"


def test_rejected_provider_extras_are_retried_without_them():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return error(400, "unknown parameter reasoning_effort")
        return sse("ok")

    provider = make_provider(handler, default_model="openai/gpt-oss-120b",
                             extra_body=groq_extra_body)
    assert collect(provider, "openai/gpt-oss-120b", REQUEST, StreamInfo()) == ["ok"]
    assert "reasoning_effort" in bodies[0] and "reasoning_effort" not in bodies[1]


def test_retry_after_parsing():
    assert _retry_after({"retry-after": "12"}, "") == 12
    assert _retry_after({}, "Please try again in 1m26.4s.") == pytest.approx(86.4)
    assert _retry_after({}, "try again in 430ms") == pytest.approx(0.43)
    assert _retry_after({}, "Please retry in 13h10m43.9s.") == pytest.approx(47443.9)
    assert _retry_after({}, "no hint here") is None


def test_think_blocks_are_stripped_from_the_stream():
    def run(*pieces):
        f = _ThinkFilter()
        return "".join(f.feed(p) for p in pieces) + f.flush()

    assert run("<thi", "nk>secret reasoning</th", "ink>\n\nThe answer", " is 4.") == "The answer is 4."
    assert run("Plain ", "answer <think> stays") == "Plain answer <think> stays"
    assert run("<") == "<"  # a lone prefix that never became a tag is kept


def test_list_models_strips_prefix_and_skips_inactive():
    payload = {"data": [{"id": "models/gemini-9-flash"}, {"id": "old", "active": False},
                        {"id": "big", "context_window": 131072}]}
    provider = make_provider(lambda request: httpx.Response(200, json=payload))
    models = asyncio.run(provider.list_models())
    assert [(m.id, m.context_window) for m in models] == [("gemini-9-flash", None), ("big", 131072)]


# ── Router: model discovery ──────────────────────────────────────


def router_with(handler, **kwargs):
    return LLMRouter([make_provider(handler, **kwargs)])


def test_retired_model_is_replaced_by_an_available_one():
    """The original bug: the configured model no longer exists on the provider."""
    asked = []

    def handler(request):
        if request.url.path.endswith("/models"):
            return models_response("whisper-large-v3", "openai/gpt-oss-120b", "llama-guard-4")
        asked.append(json.loads(request.content)["model"])
        return sse("fine")

    router = router_with(handler, default_model="llama-3.1-8b-instant", fast_model="",
                         preferred=(r"openai/gpt-oss-120b",))
    info = StreamInfo()
    assert collect(router, REQUEST, info) == ["fine"]
    assert asked == ["openai/gpt-oss-120b"]  # never even tried the retired model
    assert info.provider == "groq"


def test_unknown_catalogue_picks_largest_context_chat_model():
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [
                {"id": "tiny-chat", "context_window": 4096},
                {"id": "whisper-huge", "context_window": 999999},
                {"id": "future-chat", "context_window": 200000},
            ]})
        return sse(json.loads(request.content)["model"])

    router = router_with(handler, default_model="gone", fast_model="")
    assert collect(router, REQUEST, StreamInfo()) == ["future-chat"]


def test_newest_matching_model_is_preferred():
    def handler(request):
        if request.url.path.endswith("/models"):
            return models_response("gemini-3.1-flash-lite", "gemini-3.8-flash-lite",
                                   "gemini-3.8-flash", "gemini-embedding-2")
        return sse(json.loads(request.content)["model"])

    router = router_with(handler, name="google", default_model="", fast_model="",
                         preferred=(r"gemini-[\d.]+-flash-lite", r"gemini-[\d.]+-flash"))
    assert collect(router, REQUEST, StreamInfo()) == ["gemini-3.8-flash-lite"]


def test_fast_requests_use_the_fast_model():
    def handler(request):
        if request.url.path.endswith("/models"):
            return models_response("big-model", "small-model")
        return sse(json.loads(request.content)["model"])

    router = router_with(handler)
    fast = ChatRequest(system="s", messages=REQUEST.messages, fast=True)
    assert asyncio.run(router.complete(fast)) == "small-model"
    assert asyncio.run(router.complete(REQUEST)) == "big-model"


def test_listing_failure_still_tries_the_configured_model():
    def handler(request):
        if request.url.path.endswith("/models"):
            return error(500, "listing is down")
        return sse("still works")

    assert collect(router_with(handler), REQUEST, StreamInfo()) == ["still works"]


# ── Router: failover ─────────────────────────────────────────────


def rate_limited(retry_after=60.0):
    return LLMError("rate_limit", "slow down", provider="one", retry_after=retry_after)


def test_falls_over_to_the_next_model_then_the_next_provider():
    first = FakeProvider("one", errors=[rate_limited(), LLMError("overloaded", "busy")])
    second = FakeProvider("two", reply="from two")
    info = StreamInfo()
    assert "".join(collect(LLMRouter([first, second]), REQUEST, info)).strip() == "from two"
    assert [model for model, _ in first.calls] == ["one-large", "one-small"]
    assert info.provider == "two"


def test_auth_failure_skips_the_rest_of_that_provider():
    first = FakeProvider("one", errors=[LLMError("auth", "bad key", provider="one")])
    second = FakeProvider("two")
    collect(LLMRouter([first, second]), REQUEST, StreamInfo())
    assert len(first.calls) == 1 and len(second.calls) == 1


def test_short_rate_limit_is_waited_out_on_the_same_model():
    provider = FakeProvider("one", reply="after waiting", errors=[rate_limited(0.01)])
    assert "".join(collect(LLMRouter([provider]), REQUEST, StreamInfo())).strip() == "after waiting"
    assert [model for model, _ in provider.calls] == ["one-large", "one-large"]


def test_all_failed_reports_an_actionable_error():
    first = FakeProvider("one", errors=[rate_limited(42), rate_limited(90)])
    second = FakeProvider("two", errors=[LLMError("auth", "bad key", provider="two")])
    with pytest.raises(LLMError) as caught:
        collect(LLMRouter([first, second]), REQUEST, StreamInfo())
    assert caught.value.kind == "rate_limit"
    assert "42 seconds" in caught.value.message and caught.value.retry_after == 42


def test_fallback_can_be_disabled():
    first = FakeProvider("one", errors=[LLMError("overloaded", "busy", provider="one")])
    second = FakeProvider("two")
    with pytest.raises(LLMError):
        collect(LLMRouter([first, second], fallback=False), REQUEST, StreamInfo())
    assert len(first.calls) == 1 and second.calls == []


def test_error_after_output_started_is_not_retried():
    class Flaky(FakeProvider):
        async def stream(self, model, request, info):
            self.calls.append((model, request))
            yield "partial "
            raise LLMError("network", "dropped", provider=self.name)

    first, second = Flaky("one"), FakeProvider("two")
    with pytest.raises(LLMError, match="dropped"):
        collect(LLMRouter([first, second]), REQUEST, StreamInfo())
    assert second.calls == []


def test_no_providers_configured():
    with pytest.raises(LLMError) as caught:
        collect(LLMRouter([]), REQUEST, StreamInfo())
    assert caught.value.kind == "not_configured" and "GROQ_API_KEY" in caught.value.message


def test_describe_reports_the_resolved_model():
    def handler(request):
        return models_response("openai/gpt-oss-120b", "openai/gpt-oss-20b")

    router = router_with(handler, default_model="llama-3.1-8b-instant", fast_model="",
                         preferred=(r"openai/gpt-oss-120b", r"openai/gpt-oss-20b"))
    status = asyncio.run(router.describe())
    assert status["provider"] == "groq" and status["model"] == "openai/gpt-oss-120b"
    assert status["providers"][0]["fallback_models"] == ["openai/gpt-oss-20b"]


def test_rejected_key_skips_the_provider_without_a_chat_request():
    chat_requests = []

    def handler(request):
        if request.url.path.endswith("/models"):
            return error(401, "Invalid API Key")
        chat_requests.append(request)
        return sse("should not be reached")

    backup = FakeProvider("backup", reply="from backup")
    router = LLMRouter([make_provider(handler), backup])
    assert "".join(collect(router, REQUEST, StreamInfo())).strip() == "from backup"
    assert chat_requests == []

    with pytest.raises(LLMError) as caught:
        collect(LLMRouter([make_provider(handler)]), REQUEST, StreamInfo())
    assert caught.value.kind == "auth" and "GROQ_API_KEY" in caught.value.message
