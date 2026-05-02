# Vector Store Service — handles document ingestion, chunking, and retrieval via ChromaDB.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from logging_config import logger

# ── Singleton Instances ─────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: Chroma | None = None

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def get_embeddings() -> HuggingFaceEmbeddings:
    # Lazily initialise the embedding model (downloads on first call).
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s ...", settings.EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")
    return _embeddings


def get_vector_store() -> Chroma:
    # Return the persistent ChromaDB vector store (creates it if needed).
    global _vector_store
    if _vector_store is None:
        logger.info(
            "Initializing ChromaDB at %s (collection: %s)",
            settings.CHROMA_PERSIST_DIR,
            settings.COLLECTION_NAME,
        )
        _vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
        logger.info("ChromaDB ready — %d existing vectors.", _vector_store._collection.count())
    return _vector_store


def ingest_document(filename: str, content: str) -> int:
    # Split a document into chunks and add them to the vector store.
    # Returns the number of chunks created.  
    logger.info("Ingesting document: %s (%d characters)", filename, len(content))

    documents = [
        Document(page_content=content, metadata={"source": filename})
    ]
    chunks = _text_splitter.split_documents(documents)
    logger.info("Created %d chunks from %s", len(chunks), filename)

    store = get_vector_store()
    store.add_documents(chunks)

    logger.info("Document '%s' stored successfully (%d chunks).", filename, len(chunks))
    return len(chunks)


def retrieve_context(query: str, k: int = 4) -> list[Document]:
    
    # Retrieve the top-k most relevant chunks for a given query.
    
    store = get_vector_store()
    results = store.similarity_search(query, k=k)
    logger.info(
        "Retrieval for '%s' returned %d chunks from sources: %s",
        query[:80],
        len(results),
        list({doc.metadata.get("source", "unknown") for doc in results}),
    )
    return results


def list_documents() -> list[dict]:
    # List all unique documents and their chunk counts.
    store = get_vector_store()
    collection = store._collection
    all_meta = collection.get()["metadatas"]

    doc_counts: dict[str, int] = {}
    for meta in all_meta:
        source = meta.get("source", "unknown")
        doc_counts[source] = doc_counts.get(source, 0) + 1

    return [
        {"filename": name, "chunk_count": count}
        for name, count in sorted(doc_counts.items())
    ]


def delete_document(filename: str) -> int:
    # Remove all chunks belonging to a specific document.
    # Returns the number of chunks deleted.
    store = get_vector_store()
    collection = store._collection

    # Find all IDs with this source
    results = collection.get(where={"source": filename})
    ids_to_delete = results["ids"]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for document '%s'.", len(ids_to_delete), filename)
    else:
        logger.warning("No chunks found for document '%s'.", filename)

    return len(ids_to_delete)
