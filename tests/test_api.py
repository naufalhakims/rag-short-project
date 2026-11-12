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


def upload(client, name, content, session_id=None):
    params = {"session_id": session_id} if session_id else {}
    return client.post("/api/documents", files={"file": (name, content)}, params=params)


def new_chat(client, *files):
    """Create an empty conversation and upload (name, content) files into it."""
    session_id = client.post("/api/history").json()["id"]
    for name, content in files:
        assert upload(client, name, content, session_id).status_code == 200
    return session_id


def chat_files(client, session_id):
    return [d["filename"] for d in client.get(f"/api/history/{session_id}/documents").json()]


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
    chat = new_chat(client, ("guide.txt", GUIDE))
    events = ask(client, "Who is Kaelen Voss?", session_id=chat)

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
    assert session_id == chat
    sessions = client.get("/api/history").json()
    assert [s["id"] for s in sessions] == [session_id]
    assert sessions[0]["title"] == "Who is Kaelen Voss?"  # "New chat" until the first question
    user, assistant = client.get(f"/api/history/{session_id}").json()
    assert (user["role"], user["content"]) == ("user", "Who is Kaelen Voss?")
    assert assistant["content"].strip() == answer
    assert assistant["sources"][0]["cited"] is True


def test_follow_up_is_rewritten_using_the_conversation(client, provider):
    chat = new_chat(client, ("guide.txt", GUIDE))
    first = ask(client, "Who is Kaelen Voss?", session_id=chat)
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
    chat = new_chat(client, ("guide.txt", GUIDE))
    provider.errors = [LLMError("rate_limit", "slow down", provider="fake", retry_after=30)] * 2
    events = ask(client, "Who is Kaelen Voss?", session_id=chat)
    failure = events[-1]
    assert failure["type"] == "error" and failure["code"] == "rate_limit"
    assert failure["retry_after"] == 30 and "30 seconds" in failure["message"]
    assert client.get(f"/api/history/{chat}").json() == []

    # A brand-new chat whose first question fails is not saved at all.
    provider.errors = [LLMError("overloaded", "busy", provider="fake")] * 2
    assert ask(client, "Who is Kaelen Voss?")[-1]["type"] == "error"
    assert [s["id"] for s in client.get("/api/history").json()] == [chat]


def test_chat_without_documents_still_answers(client, provider):
    upload(client, "guide.txt", GUIDE)  # in the library, but not in this chat
    events = ask(client, "Who is Kaelen Voss?")
    assert by_type(events, "sources")[0] == {
        "type": "sources", "sources": [], "mode": "empty", "query": "Who is Kaelen Voss?",
    }
    assert events[-1]["type"] == "done"
    prompt = provider.calls[-1][1].messages[-1]["content"]
    assert "No documents are attached to this conversation" in prompt
    assert "<source" not in prompt


def test_chat_search_can_be_narrowed_within_its_documents(client):
    chat = new_chat(client, ("guide.txt", GUIDE), ("paper.md", PAPER))
    paper = next(d for d in client.get("/api/documents").json() if d["filename"] == "paper.md")
    events = ask(client, "accuracy points penalty", session_id=chat, doc_ids=[paper["id"]])
    assert {s["filename"] for s in by_type(events, "sources")[0]["sources"]} == {"paper.md"}


def test_new_chat_via_api_gets_the_documents_it_names(client):
    guide = upload(client, "guide.txt", GUIDE).json()["document"]
    events = ask(client, "Who is Kaelen Voss?", doc_ids=[guide["id"]])
    assert by_type(events, "sources")[0]["mode"] == "matched"
    assert chat_files(client, events[0]["session_id"]) == ["guide.txt"]


# ── Each conversation has its own documents ──────────────────────


def test_conversations_only_search_their_own_documents(client, provider):
    chat_a = new_chat(client, ("guide.txt", GUIDE))
    chat_b = new_chat(client, ("paper.md", PAPER))

    for chat, own, other in [(chat_a, "guide.txt", "paper.md"), (chat_b, "paper.md", "guide.txt")]:
        # Words from both documents, so a shared search would return both.
        events = ask(client, "Kaelen Voss consensus accuracy drones points", session_id=chat)
        files = {s["filename"] for s in by_type(events, "sources")[0]["sources"]}
        assert files == {own}
        prompt = provider.calls[-1][1].messages[-1]["content"]
        assert f'file="{own}"' in prompt and f'file="{other}"' not in prompt

    # Retrieval-only search respects the scope too.
    found = client.get("/api/search", params={"q": "Kaelen consensus", "session_id": chat_b})
    assert {r["filename"] for r in found.json()["results"]} == {"paper.md"}


def test_attach_and_detach_documents(client):
    chat = new_chat(client, ("guide.txt", GUIDE))
    paper = upload(client, "paper.md", PAPER).json()["document"]  # library only
    assert chat_files(client, chat) == ["guide.txt"]

    attached = client.post(f"/api/history/{chat}/documents", json={"doc_ids": [paper["id"]]})
    assert [d["filename"] for d in attached.json()] == ["guide.txt", "paper.md"]
    client.post(f"/api/history/{chat}/documents", json={"doc_ids": [paper["id"]]})  # twice is fine
    assert chat_files(client, chat) == ["guide.txt", "paper.md"]

    assert client.delete(f"/api/history/{chat}/documents/{paper['id']}").status_code == 200
    assert chat_files(client, chat) == ["guide.txt"]
    assert client.delete(f"/api/history/{chat}/documents/{paper['id']}").status_code == 404
    # Removing a file from a chat keeps it in the library.
    assert "paper.md" in [d["filename"] for d in client.get("/api/documents").json()]

    assert client.post(f"/api/history/{chat}/documents", json={"doc_ids": ["nope"]}).status_code == 404
    assert client.get("/api/history/missing/documents").status_code == 404
    assert upload(client, "x.txt", "x", session_id="missing").status_code == 404


def test_same_file_in_two_chats_is_stored_once(client):
    chat_a = new_chat(client, ("guide.txt", GUIDE))
    chat_b = new_chat(client)
    assert upload(client, "guide.txt", GUIDE, chat_b).json()["status"] == "unchanged"
    assert len(client.get("/api/documents").json()) == 1
    assert chat_files(client, chat_a) == chat_files(client, chat_b) == ["guide.txt"]


def test_reuploading_a_file_keeps_it_in_its_chats(client):
    chat = new_chat(client, ("guide.txt", GUIDE))
    replaced = upload(client, "guide.txt", GUIDE.replace("Kaelen Voss", "Mira Okafor"))
    assert replaced.json()["status"] == "replaced"
    attached = client.get(f"/api/history/{chat}/documents").json()
    assert [d["id"] for d in attached] == [replaced.json()["document"]["id"]]
    events = ask(client, "Who is Mira Okafor?", session_id=chat)
    assert "Mira Okafor" in by_type(events, "sources")[0]["sources"][0]["text"]


def test_deleting_a_document_removes_it_from_its_chats(client):
    chat = new_chat(client, ("guide.txt", GUIDE))
    guide = client.get(f"/api/history/{chat}/documents").json()[0]
    assert client.delete(f"/api/documents/{guide['id']}").status_code == 200
    assert chat_files(client, chat) == []


def test_deleting_a_chat_keeps_its_documents(client):
    chat = new_chat(client, ("guide.txt", GUIDE))
    assert client.delete(f"/api/history/{chat}").status_code == 200
    assert [d["filename"] for d in client.get("/api/documents").json()] == ["guide.txt"]


def test_chat_validates_input(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "   "}).status_code == 422


def test_stopping_midway_saves_the_partial_answer(services, library):
    """Closing the stream early (the Stop button) keeps what was generated so far."""

    doc_ids = [d["id"] for d in services.db.list_documents()]

    async def run():
        stream = services.chat.stream("Who is Kaelen Voss?", doc_ids=doc_ids)
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
    session_id = ask(client, "Who is Kaelen Voss?", session_id=new_chat(client))[0]["session_id"]
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
    chat = new_chat(client, ("guide.txt", GUIDE))
    events = ask(client, "Who is Kaelen Voss?", session_id=chat)
    assert "".join(e["text"] for e in by_type(events, "token")).strip().endswith("striker[1].")
    assert events[-1]["cited"] == [1]


def test_exotic_spaces_and_hyphens_are_made_plain():
    normaliser = prompts.CitationNormaliser()
    assert normaliser.feed("more than\u202f4\u00a0seconds, zero\u2011gravity") == "more than 4 seconds, zero-gravity"
