"""Optional password protection.

When APP_PASSWORD is set, every /api route except login needs a session token, sent as
`Authorization: Bearer <token>`. A header (not a cookie) is used because a Hugging Face
Space shows the app inside an iframe on huggingface.co, where browsers block cookies.

Tokens are signed with the password, so they need no server-side storage, expire after
SESSION_DAYS, and all stop working when the password changes.
"""

import asyncio
import hashlib
import hmac
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.logging_config import logger

_FAILED_LOGIN_DELAY = 1.5  # seconds; slows down password guessing


def issue_token(password: str, now: float | None = None) -> str:
    issued = str(int(now if now is not None else time.time()))
    return f"{issued}.{_sign(password, issued)}"


def token_is_valid(password: str, token: str, max_age_seconds: float) -> bool:
    issued, _, signature = token.partition(".")
    try:
        age = time.time() - int(issued)
    except ValueError:
        return False
    if not 0 <= age <= max_age_seconds:
        return False
    return hmac.compare_digest(signature, _sign(password, issued))


def _sign(password: str, issued: str) -> str:
    message = f"pocket-librarian:{issued}".encode()
    return hmac.new(password.encode(), message, hashlib.sha256).hexdigest()


def require_login(request: Request) -> None:
    """Dependency for every protected route. A no-op when no password is set."""
    settings = request.app.state.services.settings
    if not settings.app_password:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token or not token_is_valid(settings.app_password, token, _max_age(settings)):
        raise HTTPException(401, "Please log in.", headers={"WWW-Authenticate": "Bearer"})


def _max_age(settings) -> float:
    return settings.session_days * 86400


class LoginBody(BaseModel):
    password: str = Field(max_length=500)


router = APIRouter(prefix="/api", tags=["Login"])


@router.get("/auth")
def auth_status(request: Request) -> dict:
    """Whether this server needs a password (and so whether to show the login screen)."""
    return {"password_required": bool(request.app.state.services.settings.app_password)}


@router.post("/login")
async def login(body: LoginBody, request: Request) -> dict:
    settings = request.app.state.services.settings
    if not settings.app_password:
        return {"token": "", "expires_in": None}
    if not hmac.compare_digest(body.password.encode(), settings.app_password.encode()):
        client = request.client.host if request.client else "?"
        logger.warning("Failed login attempt from %s", client)
        await asyncio.sleep(_FAILED_LOGIN_DELAY)
        raise HTTPException(401, "Wrong password.")
    return {"token": issue_token(settings.app_password), "expires_in": int(_max_age(settings))}
