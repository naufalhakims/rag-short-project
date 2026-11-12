"""End-to-end through the HTTP API, with fake models and a fake LLM."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.llm.base import LLMError
from app.main import create_app
from app.rag import prompts
from tests.conftest import GUIDE, PAPER


@pytest.fixture
def client(settings, services):
    with TestClient(create_app(settings, services)) as test_client:
        yield test_client


def upload(client, name, content):
    return client.post("/api/documents", files={"file": (name, content)})


def ask(client, question, **extra):
    """POST /api/chat and return the parsed event stream."""
    response = client.post("/api/chat", json={"question": question, **extra})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def by_type(events, kind):
    return [e for e in events if e["type"] == kind]


# ── Documents ────────────────────────────────────────────────────


def test_upload_list_search_delete(client):
    created = upload(client, "guide.txt", GUIDE)
    assert created.status_code == 200
    document = created.json()["document"]
    assert created.json()["status"] == "added" and document["chunk_count"] >= 2

    assert [d["filename"] for d in client.get("/api/documents").json()] == ["guide.txt"]

    found = client.get("/api/search", params={"q": "Who is Kaelen Voss?"}).json()
    assert found["mode"] == "matched" and "Kaelen Voss" in found["results"][0]["text"]

    assert client.delete(f"/api/documents/{document['id']}").status_code == 200
    assert client.delete(f"/api/documents/{document['id']}").status_code == 404
    assert client.get("/api/documents").json() == []


def test_upload_rejects_bad_files(client, settings):
    assert upload(client, "image.png", b"\x89PNG").status_code == 400
    assert "Unsupported" in upload(client, "image.png", b"\x89PNG").json()["detail"]
    assert upload(client, "empty.txt", b"").status_code == 400
    too_big = b"x" * (settings.max_upload_mb * 1024 * 1024 + 1)
    assert upload(client, "huge.txt", too_big).status_code == 413


def test_legacy_upload_path_still_works(client):
    assert client.post("/api/upload", files={"file": ("a.md", PAPER)}).status_code == 200


# ── Chat ─────────────────────────────────────────────────────────


def test_chat_streams_a_cited_answer_and_saves_it(client, provider):
    upload(client, "guide.txt", GUIDE)
    events = ask(client, "Who is Kaelen Voss?")

    assert [e["type"] for e in events[:2]] == ["session", "status"]
    sources = by_type(events, "sources")[0]
    assert sources["mode"] == "matched" and "Kaelen Voss" in sources["sources"][0]["text"]
    answer = "".join(e["text"] for e in by_type(events, "token")).strip()
    assert answer == "Kaelen Voss is the star striker [1]."
    done = events[-1]
    assert done["type"] == "done" and done["cited"] == [1]
    assert done["meta"]["provider"] == "fake" and done["meta"]["model"] == "fake-large"

    # The LLM was given the retrieved passage, numbered, alongside the question.
    _, request = provider.calls[-1]
    prompt = request.messages[-1]["content"]
    assert '<source id="1" file="guide.txt">' in prompt and "Kaelen Voss" in prompt
    assert prompt.endswith("Question: Who is Kaelen Voss?")

    # …and the exchange is in the history, with the cited source flagged.
    session_id = events[0]["session_id"]
    sessions = client.get("/api/history").json()
    assert [s["id"] for s in sessions] == [session_id]
    assert sessions[0]["title"] == "Who is Kaelen Voss?"
    user, assistant = client.get(f"/api/history/{session_id}").json()
    assert (user["role"], user["content"]) == ("user", "Who is Kaelen Voss?")
    assert assistant["content"].strip() == answer
    assert assistant["sources"][0]["cited"] is True


def test_follow_up_is_rewritten_using_the_conversation(client, provider):
    upload(client, "guide.txt", GUIDE)
    first = ask(client, "Who is Kaelen Voss?")
    session_id = first[0]["session_id"]
    assert not any(request.fast for _, request in provider.calls)  # nothing to rewrite yet

    events = ask(client, "what team are they on?", session_id=session_id)
    assert events[0]["session_id"] == session_id
    assert by_type(events, "sources")[0]["query"] == "standalone search query"

    rewrite = next(request for _, request in provider.calls if request.fast)
    assert "Who is Kaelen Voss?" in rewrite.messages[0]["content"]
    assert "what team are they on?" in rewrite.messages[0]["content"]
    # The answer request carries the earlier exchange as real chat turns.
    _, answer_request = provider.calls[-1]
    assert [m["role"] for m in answer_request.messages] == ["user", "assistant", "user"]
    assert len(client.get(f"/api/history/{session_id}").json()) == 4


def test_llm_failure_is_an_error_event_and_leaves_no_history(client, provider):
    upload(client, "guide.txt", GUIDE)
    provider.errors = [LLMError("rate_limit", "slow down", provider="fake", retry_after=30)] * 2
    events = ask(client, "Who is Kaelen Voss?")
    failure = events[-1]
    assert failure["type"] == "error" and failure["code"] == "rate_limit"
    assert failure["retry_after"] == 30 and "30 seconds" in failure["message"]
    assert client.get("/api/history").json() == []


def test_chat_with_empty_library_still_answers(client, provider):
    events = ask(client, "hello")
    assert by_type(events, "sources")[0] == {
        "type": "sources", "sources": [], "mode": "empty", "query": "hello",
    }
    assert events[-1]["type"] == "done"
    assert "no documents yet" in provider.calls[-1][1].messages[-1]["content"]


def test_chat_can_be_limited_to_documents(client):
    upload(client, "guide.txt", GUIDE)
    paper = upload(client, "paper.md", PAPER).json()["document"]
    events = ask(client, "accuracy points penalty", doc_ids=[paper["id"]])
    assert {s["filename"] for s in by_type(events, "sources")[0]["sources"]} == {"paper.md"}


def test_chat_validates_input(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "   "}).status_code == 422


def test_stopping_midway_saves_the_partial_answer(services, library):
    """Closing the stream early (the Stop button) keeps what was generated so far."""

    async def run():
        stream = services.chat.stream("Who is Kaelen Voss?")
        session_id = None
        async for event in stream:
            session_id = session_id or event.get("session_id")
            if event["type"] == "token":
                break
        await stream.aclose()
        return session_id

    session_id = asyncio.run(run())
    _, assistant = services.db.get_messages(session_id)
    assert assistant["content"].strip() == "Kaelen"
    assert assistant["meta"]["stopped"] is True


# ── History and status ───────────────────────────────────────────


def test_history_delete_and_missing_session(client):
    upload(client, "guide.txt", GUIDE)
    session_id = ask(client, "Who is Kaelen Voss?")[0]["session_id"]
    assert client.delete(f"/api/history/{session_id}").status_code == 200
    assert client.get(f"/api/history/{session_id}").status_code == 404
    assert client.delete(f"/api/history/{session_id}").status_code == 404


def test_health_reports_models_and_library(client):
    upload(client, "guide.txt", GUIDE)
    health = client.get("/api/health").json()
    assert health["llm"]["provider"] == "fake" and health["llm"]["model"] == "fake-large"
    assert health["retrieval"]["ready"] is True and health["retrieval"]["keyword_search"] is True
    assert health["library"]["documents"] == 1
    assert ".pdf" in health["upload"]["extensions"]


def test_frontend_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Pocket Librarian" in page.text


# ── Prompt helpers ───────────────────────────────────────────────


def test_cited_numbers():
    assert prompts.cited_numbers("A [1]. B [2, 3] and C [5][9]. Not [0] or [x].", 5) == [1, 2, 3, 5]
    assert prompts.cited_numbers("No citations here.", 3) == []


def test_clean_rewrite_and_history_pairing():
    assert prompts.clean_rewrite('"Gravity Ball penalties"\nextra', "orig") == "Gravity Ball penalties"
    assert prompts.clean_rewrite("   ", "orig") == "orig"
    assert prompts.clean_rewrite("x" * 500, "orig") == "orig"

    def message(role, content):
        return {"role": role, "content": content}

    history = [message("user", "orphan"), message("user", "q1"), message("assistant", "a1"),
               message("user", "q2"), message("assistant", "a2")]
    assert [m["content"] for m in prompts.paired_history(history, 1)] == ["q2", "a2"]
    assert [m["content"] for m in prompts.paired_history(history, 5)] == ["q1", "a1", "q2", "a2"]


def test_fullwidth_citations_are_normalised_even_when_split_across_chunks():
    def run(*pieces):
        normaliser = prompts.CitationNormaliser()
        return "".join(normaliser.feed(p) for p in pieces) + normaliser.flush()

    assert run("Voss is the striker【", "1", "】.") == "Voss is the striker[1]."
    assert run("Both【1†source】【2, 3】 agree") == "Both[1][2, 3] agree"
    assert run("Plain [1] stays, and 【not a citation】 too") == "Plain [1] stays, and 【not a citation】 too"
    assert run("ends mid 【1") == "ends mid 【1"


def test_model_citation_style_is_normalised_in_the_answer(client, provider):
    provider.reply = "Kaelen Voss is the star striker【1】."
    upload(client, "guide.txt", GUIDE)
    events = ask(client, "Who is Kaelen Voss?")
    assert "".join(e["text"] for e in by_type(events, "token")).strip().endswith("striker[1].")
    assert events[-1]["cited"] == [1]


def test_exotic_spaces_and_hyphens_are_made_plain():
    normaliser = prompts.CitationNormaliser()
    assert normaliser.feed("more than\u202f4\u00a0seconds, zero\u2011gravity") == "more than 4 seconds, zero-gravity"
