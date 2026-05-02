# LLM Service - Groq-based RAG chain with streaming support.
import json
from typing import AsyncGenerator

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import settings
from logging_config import logger
from services.vector_service import retrieve_context

# ── Strict RAG Prompt ───────────────────────────────────────────

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use ONLY the following pieces of retrieved context to answer the question. "
    "If the answer is not in the context, say "
    "'I do not have enough information in your uploaded files to answer this.' "
    "Do not make up information.\n\n"
    "Context:\n{context}\n"
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

# ── LLM Singleton ──────────────────────────────────────────────

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    # Lazily initialise the Groq LLM.
    global _llm
    if _llm is None:
        logger.info("Initializing Groq LLM: %s", settings.GROQ_MODEL)
        _llm = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
            streaming=True,
        )
        logger.info("Groq LLM ready.")
    return _llm


def _format_context(docs: list[Document]) -> str:
    # Combine retrieved document chunks into a single context string.
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _extract_sources(docs: list[Document]) -> list[str]:
    # Return deduplicated source filenames.
    return list({doc.metadata.get("source", "unknown") for doc in docs})


async def stream_rag_response(question: str) -> AsyncGenerator[str, None]:
    """
    Full RAG pipeline with streaming:
    1. Retrieve relevant context from the vector store.
    2. Build the prompt with context.
    3. Stream tokens from Groq.

    Yields Server-Sent Event formatted strings.
    """
    logger.info("RAG query: %s", question[:120])

    # Step 1 — Retrieve
    docs = retrieve_context(question, k=4)
    sources = _extract_sources(docs)

    if not docs:
        logger.warning("No context found for query: %s", question[:80])
        yield f"data: {json.dumps({'token': 'I do not have enough information in your uploaded files to answer this.', 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True, 'sources': []})}\n\n"
        return

    context_str = _format_context(docs)
    logger.debug("Context length: %d characters from %d chunks", len(context_str), len(docs))

    # Step 2 — Build the chain
    llm = get_llm()
    chain = _prompt | llm | StrOutputParser()

    # Step 3 — Stream tokens
    full_response = ""
    try:
        async for token in chain.astream({"context": context_str, "question": question}):
            full_response += token
            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

        # Final event with sources
        yield f"data: {json.dumps({'token': '', 'done': True, 'sources': sources})}\n\n"
        logger.info("Streaming complete — %d tokens, sources: %s", len(full_response), sources)

    except Exception as e:
        logger.error("Streaming error: %s", str(e), exc_info=True)
        yield f"data: {json.dumps({'token': f'Error: {str(e)}', 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True, 'sources': []})}\n\n"
