"""The chat pipeline: rewrite the question, retrieve passages, stream a cited answer."""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing

from app.config import Settings
from app.db import Database
from app.llm.base import ChatRequest, LLMError, StreamInfo
from app.llm.router import LLMRouter
from app.logging_config import logger
from app.rag import prompts
from app.rag.mentions import find_mentions
from app.rag.retriever import RetrievalResult, Retriever

_TITLE_CHARS = 60


def _title(question: str) -> str:
    text = " ".join(question.split())
    if len(text) <= _TITLE_CHARS:
        return text
    return text[:_TITLE_CHARS].rsplit(" ", 1)[0] + "…"


def source_dicts(result: RetrievalResult) -> list[dict]:
    return [
        {
            "n": number,
            "chunk_id": p.chunk.id,
            "doc_id": p.chunk.doc_id,
            "filename": p.chunk.filename,
            "heading": p.chunk.heading,
            "page": p.chunk.page,
            "text": p.chunk.text,
            "score": p.score,
            "cited": False,
        }
        for number, p in enumerate(result.passages, start=1)
    ]


class ChatService:
    def __init__(self, settings: Settings, db: Database, retriever: Retriever, llm: LLMRouter):
        self._settings = settings
        self._db = db
        self._retriever = retriever
        self._llm = llm

    async def stream(
        self,
        question: str,
        session_id: str | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> AsyncIterator[dict]:
        """Answer a question as a stream of events, searching only this conversation's
        documents.

        `doc_ids` narrows the search further for an existing conversation; for a new one
        (no `session_id`) it sets which documents the conversation gets.

        Events: session, status, sources, token*, then done or error. A turn is saved
        to the chat history only once it has an answer — failures leave no trace, and
        an answer the user stops midway is saved as far as it got.
        """
        started = time.perf_counter()
        session = self._db.get_session(session_id) if session_id else None
        history: list[dict] = []
        recent: list[dict] = []
        if session:
            recent = self._db.get_messages(session["id"], limit=4 * self._settings.history_turns)
            history = prompts.paired_history(recent, self._settings.history_turns)
            scope = self._db.session_document_ids(session["id"])
            if doc_ids is not None:
                scope = [d for d in scope if d in set(doc_ids)]
        else:
            scope = list(doc_ids or [])
        session_id = session["id"] if session else uuid.uuid4().hex

        # "@paper.pdf" in the question limits this search to the tagged documents.
        in_scope = [d for d in self._db.list_documents() if d["id"] in set(scope)]
        mentions = find_mentions(question, in_scope)
        query_text = question
        if mentions.doc_ids:
            scope = mentions.doc_ids
            query_text = mentions.text_without or question
        tagged = [
            {"id": doc_id, "filename": name}
            for doc_id, name in zip(mentions.doc_ids, mentions.filenames, strict=True)
        ]
        # A conversation created empty (to attach documents first) is named by its
        # first question.
        title = session["title"] if session and recent else _title(question)
        yield {"type": "session", "session_id": session_id, "title": title}

        # 1. Follow-up questions ("and what about its penalties?") only make sense in
        #    context, so turn them into a standalone search query first.
        search_query = query_text
        if history:
            yield {"type": "status", "stage": "rewriting"}
            search_query = await self._standalone_query(query_text, history)

        # 2. Retrieve supporting passages (CPU-bound: keep it off the event loop).
        yield {"type": "status", "stage": "searching"}
        result = await asyncio.to_thread(self._retriever.search, search_query, scope)
        sources = source_dicts(result)
        retrieved = time.perf_counter()
        yield {
            "type": "sources",
            "sources": sources,
            "mode": result.mode,
            "query": search_query,
            "mentions": tagged,
        }

        # 3. Generate the answer.
        yield {"type": "status", "stage": "generating"}
        request = ChatRequest(
            system=prompts.ANSWER_SYSTEM_PROMPT,
            messages=prompts.build_answer_messages(
                question, result, history, focus=mentions.filenames
            ),
            temperature=self._settings.llm_temperature,
        )
        info = StreamInfo()
        citations = prompts.CitationNormaliser()
        answer = ""

        def save(stopped: bool) -> tuple[str, dict, list[int]]:
            cited = prompts.cited_numbers(answer, len(sources))
            for source in sources:
                source["cited"] = source["n"] in cited
            meta = {
                "provider": info.provider,
                "model": info.model,
                "usage": info.usage,
                "mode": result.mode,
                "search_query": search_query if search_query != question else None,
                "mentions": tagged,
                "stopped": stopped,
                "timings_ms": {
                    "retrieval": round((retrieved - started) * 1000),
                    "total": round((time.perf_counter() - started) * 1000),
                },
            }
            message_id = self._db.save_turn(
                session_id=session_id, title=title, question=question,
                answer=answer, sources=sources, meta=meta,
            )
            if not session and scope:
                self._db.attach_documents(session_id, scope)
            return message_id, meta, cited

        try:
            # aclosing: if the client goes away, cancel the upstream request promptly.
            async with aclosing(self._llm.stream(request, info)) as deltas:
                async for delta in deltas:
                    delta = citations.feed(delta)
                    if delta:
                        answer += delta
                        yield {"type": "token", "text": delta}
            tail = citations.flush()
            if tail:
                answer += tail
                yield {"type": "token", "text": tail}
        except LLMError as exc:
            logger.error("Answer generation failed (%s): %s", exc.kind, exc.message)
            yield {
                "type": "error",
                "code": exc.kind,
                "message": exc.message,
                "retry_after": exc.retry_after,
            }
            return
        except (asyncio.CancelledError, GeneratorExit):
            # The client disconnected or pressed Stop: keep what was written so far.
            if answer.strip():
                save(stopped=True)
            raise

        if not answer.strip():
            yield {
                "type": "error",
                "code": "empty",
                "message": "The model returned an empty answer. Please try again.",
                "retry_after": None,
            }
            return
        message_id, meta, cited = save(stopped=False)
        logger.info(
            "Answered via %s/%s in %d ms (%d sources, %d cited).",
            info.provider, info.model, meta["timings_ms"]["total"], len(sources), len(cited),
        )
        yield {"type": "done", "message_id": message_id, "cited": cited, "meta": meta}

    async def _standalone_query(self, question: str, history: Sequence[dict]) -> str:
        request = ChatRequest(
            system=prompts.REWRITE_SYSTEM_PROMPT,
            messages=prompts.build_rewrite_messages(question, history),
            temperature=0.0,
            fast=True,
        )
        try:
            rewritten = prompts.clean_rewrite(await self._llm.complete(request), question)
        except LLMError as exc:
            # Rewriting is an optimisation; never let it block the answer.
            logger.warning("Query rewrite failed (%s); searching with the raw question.", exc.kind)
            return question
        if rewritten != question:
            logger.info("Rewrote '%s' -> '%s'", question[:60], rewritten[:80])
        return rewritten
