"""Optional password protection, and sample-document seeding for public demos."""

import time

import pytest
from fastapi.testclient import TestClient

from app import auth
from app.main import create_app
from app.services import SAMPLE_CHAT_TITLE, seed_sample_documents

PASSWORD = "correct horse battery staple"


@pytest.fixture
def locked(settings, services, monkeypatch):
    settings.app_password = PASSWORD
    monkeypatch.setattr(auth, "_FAILED_LOGIN_DELAY", 0)
    with TestClient(create_app(settings, services)) as client:
        yield client


def login(client, password=PASSWORD):
    return client.post("/api/login", json={"password": password})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_everything_needs_a_login_when_a_password_is_set(locked):
    assert locked.get("/api/auth").json() == {"password_required": True}
    for method, path in [
        ("get", "/api/health"), ("get", "/api/documents"), ("get", "/api/history"),
        ("post", "/api/history"), ("get", "/api/search?q=x"), ("delete", "/api/documents/x"),
    ]:
        assert getattr(locked, method)(path).status_code == 401, path
    chat = locked.post("/api/chat", json={"question": "hi"})
    assert chat.status_code == 401
    upload = locked.post("/api/documents", files={"file": ("a.txt", "secret")})
    assert upload.status_code == 401
    # The page itself and its assets load, so the login screen can be shown.
    assert locked.get("/").status_code == 200
    assert locked.get("/static/app.js").status_code == 200


def test_login_gives_a_working_token(locked):
    assert login(locked, "wrong").status_code == 401
    response = login(locked)
    assert response.status_code == 200
    token = response.json()["token"]
    assert response.json()["expires_in"] == 30 * 86400
    assert locked.get("/api/documents", headers=bearer(token)).status_code == 200
    assert locked.get("/api/documents", headers=bearer(token + "0")).status_code == 401
    assert locked.get("/api/documents", headers={"Authorization": token}).status_code == 401


def test_tokens_expire_and_die_with_a_password_change():
    old = auth.issue_token(PASSWORD, now=time.time() - 31 * 86400)
    assert not auth.token_is_valid(PASSWORD, old, 30 * 86400)
    fresh = auth.issue_token(PASSWORD)
    assert auth.token_is_valid(PASSWORD, fresh, 30 * 86400)
    assert not auth.token_is_valid("a new password", fresh, 30 * 86400)
    future = auth.issue_token(PASSWORD, now=time.time() + 3600)
    assert not auth.token_is_valid(PASSWORD, future, 30 * 86400)
    assert not auth.token_is_valid(PASSWORD, "garbage", 30 * 86400)


def test_no_password_means_no_login(settings, services):
    with TestClient(create_app(settings, services)) as client:
        assert client.get("/api/auth").json() == {"password_required": False}
        assert client.get("/api/documents").status_code == 200


def test_sample_documents_are_seeded_once_into_their_own_chat(services):
    assert seed_sample_documents(services) == 2
    sessions = services.db.list_sessions()
    assert [s["title"] for s in sessions] == [SAMPLE_CHAT_TITLE]
    attached = services.db.session_documents(sessions[0]["id"])
    assert len(attached) == 2
    # A library that already has documents is left alone.
    assert seed_sample_documents(services) == 0
    assert len(services.db.list_sessions()) == 1
