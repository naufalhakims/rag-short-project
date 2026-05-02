"""
Personal Pocket Librarian — FastAPI Entry Point

A production-ready RAG API that lets users upload documents,
store them in a vector database, and query them with streaming AI responses.
"""

import json
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from logging_config import logger
from database.sqlite_db import init_db, create_session, get_all_sessions, get_session_messages, save_message
from services.vector_service import ingest_document, list_documents, delete_document, get_vector_store
from services.llm_service import stream_rag_response
from schemas.models import (
    ChatRequest,
    UploadResponse,
    SessionResponse,
    MessageResponse,
    DocumentInfo,
    ErrorResponse,
)


# ── Lifespan ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting Personal Pocket Librarian…")
    await init_db()
    # Warm up the vector store (triggers embedding model download on first run)
    get_vector_store()
    logger.info("All systems go.")
    yield
    logger.info("Shutting down.")


# ── App Instance ────────────────────────────────────────────────

app = FastAPI(
    title="Personal Pocket Librarian",
    description="RAG API — upload documents, ask questions, get streaming AI answers.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ───────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — log and return a clean JSON error."""
    logger.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method,
        request.url.path,
        str(exc),
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "status_code": 500},
    )


# ── Frontend ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the single-page frontend."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── File Upload ─────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".text"}


@app.post("/api/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a text or markdown file for ingestion into the vector store."""

    # Validate extension
    filename = file.filename or "unknown.txt"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Read content
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text.")

    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    # Ingest
    chunks = ingest_document(filename, content)

    return UploadResponse(
        filename=filename,
        chunks_created=chunks,
        message=f"Successfully ingested '{filename}' into {chunks} chunks.",
    )


# ── Chat / Query ────────────────────────────────────────────────

@app.post("/api/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Ask a question against the uploaded documents.
    Returns a Server-Sent Events stream of tokens.
    """
    # Ensure we have documents
    docs = list_documents()
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Please upload at least one document first.",
        )

    # Create or reuse session
    session_id = request.session_id
    if not session_id:
        session = await create_session(title=request.question[:60])
        session_id = session["id"]

    # Save user message
    await save_message(session_id=session_id, role="user", content=request.question)

    # Streaming wrapper that also saves the full response when done
    async def event_generator():
        full_response = ""
        sources = []
        async for event in stream_rag_response(request.question):
            yield event
            # Parse the event to accumulate the full response
            try:
                data_str = event.replace("data: ", "").strip()
                if data_str:
                    data = json.loads(data_str)
                    if not data.get("done"):
                        full_response += data.get("token", "")
                    else:
                        sources = data.get("sources", [])
            except (json.JSONDecodeError, KeyError):
                pass

        # Persist assistant response
        await save_message(
            session_id=session_id,
            role="assistant",
            content=full_response,
            sources=json.dumps(sources) if sources else None,
        )

    # Send initial session info, then stream
    async def wrapped_generator():
        # First event: session metadata
        yield f"data: {json.dumps({'session_id': session_id, 'type': 'session'})}\n\n"
        async for chunk in event_generator():
            yield chunk

    return StreamingResponse(
        wrapped_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Chat History ────────────────────────────────────────────────

@app.get("/api/history", response_model=list[SessionResponse], tags=["History"])
async def get_history():
    """List all chat sessions."""
    sessions = await get_all_sessions()
    return sessions


@app.get("/api/history/{session_id}", response_model=list[MessageResponse], tags=["History"])
async def get_session_history(session_id: str):
    """Get all messages for a specific chat session."""
    messages = await get_session_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found or has no messages.")
    return messages


# ── Document Management ────────────────────────────────────────

@app.get("/api/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def get_documents():
    """List all uploaded documents and their chunk counts."""
    return list_documents()


@app.delete("/api/documents/{filename}", tags=["Documents"])
async def remove_document(filename: str):
    """Remove a document and all its chunks from the vector store."""
    deleted = delete_document(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    return {"filename": filename, "chunks_deleted": deleted, "message": f"Deleted '{filename}'."}


# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
