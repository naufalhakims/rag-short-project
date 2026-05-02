"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Request Models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat query from the user."""

    question: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    session_id: Optional[str] = Field(None, description="Existing session ID to continue a conversation")


# ── Response Models ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response after a successful file upload."""

    filename: str
    chunks_created: int
    message: str


class SessionResponse(BaseModel):
    """Metadata for a single chat session."""

    id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    """A single chat message."""

    id: str
    session_id: str
    role: str
    content: str
    sources: Optional[str] = None
    created_at: str


class DocumentInfo(BaseModel):
    """Information about an uploaded document."""

    filename: str
    chunk_count: int


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    status_code: int
