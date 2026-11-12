"""HTTP API routes."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.ingest.loaders import SUPPORTED_EXTENSIONS, IngestError
from app.logging_config import logger
from app.rag.pipeline import source_dicts
from app.schemas import ChatBody, DocumentOut, MessageOut, SearchOut, SessionOut, UploadOut
from app.services import Services

router = APIRouter(prefix="/api")


def get_services(request: Request) -> Services:
    return request.app.state.services


# ── Status ───────────────────────────────────────────────────────


@router.get("/health", tags=["Status"])
async def health(svc: Services = Depends(get_services)) -> dict:
    """What the app is running on: LLM provider/model, retrieval models, library size."""
    documents = await run_in_threadpool(svc.db.list_documents)
    return {
        "status": "ok",
        "llm": await svc.llm.describe(),
        "retrieval": {
            "embedding_model": svc.embedder.name,
            "reranker": svc.reranker.name if svc.reranker else None,
            "keyword_search": svc.db.fts_enabled,
            "ready": svc.ready.is_set(),
            "error": svc.warmup_error,
        },
        "library": {
            "documents": len(documents),
            "chunks": sum(d["chunk_count"] for d in documents),
        },
        "upload": {
            "extensions": sorted(SUPPORTED_EXTENSIONS),
            "max_mb": svc.settings.max_upload_mb,
        },
    }


# ── Documents ────────────────────────────────────────────────────


@router.get("/documents", response_model=list[DocumentOut], tags=["Documents"])
def list_documents(svc: Services = Depends(get_services)):
    return svc.db.list_documents()


@router.post("/documents", response_model=UploadOut, tags=["Documents"])
@router.post("/upload", response_model=UploadOut, include_in_schema=False)  # v1 path
async def upload_document(file: UploadFile = File(...), svc: Services = Depends(get_services)):
    """Add a document (.txt, .md, .pdf, .docx). Re-uploading a file name replaces it."""
    limit = svc.settings.max_upload_mb * 1024 * 1024
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data += chunk
        if len(data) > limit:
            raise HTTPException(413, f"File is larger than {svc.settings.max_upload_mb} MB.")
    if not data:
        raise HTTPException(400, "The file is empty.")
    try:
        return await run_in_threadpool(svc.library.add, file.filename or "", bytes(data))
    except IngestError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion of '%s' failed", file.filename)
        raise HTTPException(500, f"Could not process this file ({type(exc).__name__}).") from exc


@router.delete("/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str, svc: Services = Depends(get_services)) -> dict:
    if not svc.library.remove(doc_id):
        raise HTTPException(404, "Document not found.")
    return {"deleted": doc_id}


@router.get("/search", response_model=SearchOut, tags=["Documents"])
def search(
    q: str = Query(min_length=1, max_length=1000),
    k: int | None = Query(None, ge=1, le=20),
    svc: Services = Depends(get_services),
):
    """Run retrieval only — handy for seeing what the LLM would be given."""
    result = svc.retriever.search(q, top_k=k)
    return {"query": q, "mode": result.mode, "results": source_dicts(result)}


# ── Chat ─────────────────────────────────────────────────────────


@router.post("/chat", tags=["Chat"])
async def chat(body: ChatBody, svc: Services = Depends(get_services)) -> StreamingResponse:
    """Ask a question. Responds with a Server-Sent Events stream of JSON events:
    `session`, `status`, `sources`, `token`…, then `done` or `error`."""
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "The question is empty.")

    async def events() -> AsyncIterator[str]:
        try:
            async for event in svc.chat.stream(question, body.session_id, body.doc_ids):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Chat pipeline crashed")
            error = {
                "type": "error",
                "code": "internal",
                "message": "Something went wrong while answering. Check the server log.",
                "retry_after": None,
            }
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Chat history ─────────────────────────────────────────────────


@router.get("/history", response_model=list[SessionOut], tags=["History"])
def list_sessions(svc: Services = Depends(get_services)):
    return svc.db.list_sessions()


@router.get("/history/{session_id}", response_model=list[MessageOut], tags=["History"])
def get_session(session_id: str, svc: Services = Depends(get_services)):
    if not svc.db.get_session(session_id):
        raise HTTPException(404, "Conversation not found.")
    return svc.db.get_messages(session_id)


@router.delete("/history/{session_id}", tags=["History"])
def delete_session(session_id: str, svc: Services = Depends(get_services)) -> dict:
    if not svc.db.delete_session(session_id):
        raise HTTPException(404, "Conversation not found.")
    return {"deleted": session_id}
