"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="Continue an existing conversation")
    doc_ids: list[str] | None = Field(
        None,
        description="Narrow the search to these documents. For a new conversation "
        "(no session_id), these become its documents.",
    )


class AttachBody(BaseModel):
    doc_ids: list[str] = Field(min_length=1, max_length=200)


class DocumentOut(BaseModel):
    id: str
    filename: str
    chunk_count: int
    char_count: int
    size_bytes: int
    created_at: str


class UploadOut(BaseModel):
    document: DocumentOut
    status: Literal["added", "replaced", "unchanged"]


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class SourceOut(BaseModel):
    n: int
    chunk_id: str | None = None
    doc_id: str | None = None
    filename: str
    heading: str = ""
    page: int | None = None
    text: str = ""
    score: float = 0.0
    cited: bool = False


class MessageOut(BaseModel):
    id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    sources: list[SourceOut] = []
    meta: dict = {}
    created_at: str


class SearchOut(BaseModel):
    query: str
    mode: str
    results: list[SourceOut]
