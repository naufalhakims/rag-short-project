"""@-mentions: tagging documents in a question limits the search to them."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.rag.mentions import find_mentions
from tests.conftest import GUIDE, PAPER

DOCS = [
    {"id": "g", "filename": "The Official Guide to Gravity Ball.txt"},
    {"id": "p", "filename": "paper.pdf"},
    {"id": "p2", "filename": "paper v2.pdf"},
]


# ── Parsing ──────────────────────────────────────────────────────


def test_no_mention():
    found = find_mentions("What is gravity ball?", DOCS)
    assert found.doc_ids == [] and found.text_without == "What is gravity ball?"


def test_full_name_with_spaces_and_extension():
    found = find_mentions("@The Official Guide to Gravity Ball.txt who is Kaelen?", DOCS)
    assert found.doc_ids == ["g"]
    assert found.text_without == "who is Kaelen?"


def test_name_without_extension_and_case_insensitive():
    assert find_mentions("summarise @PAPER please", DOCS).doc_ids == ["p"]


def test_longest_name_wins():
    """'@paper v2.pdf' must not be read as '@paper' followed by ' v2.pdf'."""
    found = find_mentions("compare @paper v2.pdf with @paper.pdf", DOCS)
    assert found.doc_ids == ["p2", "p"]
    assert found.filenames == ["paper v2.pdf", "paper.pdf"]
    assert found.text_without == "compare with"


def test_mention_must_end_at_a_boundary():
    assert find_mentions("@paperclip is not a document", DOCS).doc_ids == []
    assert find_mentions("what about @paper? and @paper.pdf.", DOCS).doc_ids == ["p"]


def test_unknown_mention_is_ignored():
    assert find_mentions("ping @nothing here", DOCS).doc_ids == []


def test_repeated_mention_counts_once():
    assert find_mentions("@paper and again @paper.pdf", DOCS).doc_ids == ["p"]


# ── End to end ───────────────────────────────────────────────────


@pytest.fixture
def client(settings, services):
    with TestClient(create_app(settings, services)) as test_client:
        yield test_client


def ask(client, question, session_id):
    response = client.post("/api/chat", json={"question": question, "session_id": session_id})
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def chat_with_both(client):
    session_id = client.post("/api/history").json()["id"]
    for name, content in [("guide.txt", GUIDE), ("paper.md", PAPER)]:
        client.post("/api/documents", files={"file": (name, content)}, params={"session_id": session_id})
    return session_id


def sources_of(events):
    return next(e for e in events if e["type"] == "sources")


def test_tagging_a_document_searches_only_that_document(client, provider):
    chat = chat_with_both(client)
    question = "Kaelen Voss consensus accuracy drones points"

    untagged = sources_of(ask(client, question, chat))
    assert {s["filename"] for s in untagged["sources"]} == {"guide.txt", "paper.md"}
    assert untagged["mentions"] == []

    fresh = chat_with_both(client)  # a first question, so it is not rewritten as a follow-up
    tagged = sources_of(ask(client, f"@paper.md {question}", fresh))
    assert {s["filename"] for s in tagged["sources"]} == {"paper.md"}
    assert [m["filename"] for m in tagged["mentions"]] == ["paper.md"]
    assert tagged["query"] == question  # the tag itself is not searched for

    # In a follow-up, the tag still applies and is kept out of the rewrite.
    follow_up = sources_of(ask(client, "@paper.md and the power use?", chat))
    assert {s["filename"] for s in follow_up["sources"]} == {"paper.md"}
    rewrite = next(r for _, r in reversed(provider.calls) if r.fast)
    assert "@paper.md" not in rewrite.messages[0]["content"].split("<latest_message>")[1]

    prompt = provider.calls[-1][1].messages[-1]["content"]
    assert 'tagged "paper.md" with @' in prompt
    assert 'file="guide.txt"' not in prompt


def test_tag_only_question_gets_the_documents_opening(client):
    chat = chat_with_both(client)
    events = ask(client, "summarise @guide", chat)
    found = sources_of(events)
    assert {s["filename"] for s in found["sources"]} == {"guide.txt"}
    assert events[-1]["type"] == "done"
    assert events[-1]["meta"]["mentions"][0]["filename"] == "guide.txt"


def test_tagging_a_document_from_another_chat_does_nothing(client):
    """Tags only reach the conversation's own documents."""
    chat = chat_with_both(client)
    other = client.post("/api/history").json()["id"]
    client.post("/api/documents", files={"file": ("secret.txt", "The vault code is 4417.")},
                params={"session_id": other})
    found = sources_of(ask(client, "@secret what is the vault code?", chat))
    assert found["mentions"] == []
    assert all(s["filename"] != "secret.txt" for s in found["sources"])
