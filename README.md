# Personal Pocket Librarian

Personal Pocket Librarian is a local Retrieval-Augmented Generation (RAG) web app. It lets you upload text or markdown documents, stores searchable chunks in a local vector database, and answers questions using only the uploaded files.

The app is built with FastAPI, LangChain, ChromaDB, SQLite, Hugging Face embeddings, and Groq-hosted chat models. The frontend is a single-page HTML app served by FastAPI.

## Features

- Upload `.txt`, `.md`, `.markdown`, and `.text` files.
- Split uploaded documents into searchable chunks.
- Store document embeddings in a persistent local ChromaDB database.
- Ask questions through a ChatGPT-style web interface.
- Stream assistant responses token by token with Server-Sent Events.
- Persist chat sessions and messages in SQLite.
- Load previous conversations from chat history.
- Start a fresh conversation with the **New chat** button.
- Show source filenames used for an answer.
- Delete uploaded documents from the vector store.

## Architecture

```text
Browser UI
  static/index.html
        |
        | HTTP / SSE
        v
FastAPI app
  main.py
        |
        +-- config.py
        |     Loads .env values
        |
        +-- services/vector_service.py
        |     Splits documents, creates embeddings, stores/retrieves Chroma vectors
        |
        +-- services/llm_service.py
        |     Retrieves context, builds strict RAG prompt, streams Groq response
        |
        +-- database/sqlite_db.py
        |     Stores chat sessions and messages in SQLite
        |
        +-- schemas/models.py
              Defines Pydantic request/response models
```

## User Flow

1. Open the web app in a browser.
2. Upload one or more supported text documents from the sidebar.
3. The backend validates the file, chunks it, embeds each chunk, and stores the vectors in ChromaDB.
4. Ask a question in the chat input.
5. The backend retrieves the most relevant chunks from uploaded documents.
6. The retrieved context and user question are sent to Groq through LangChain.
7. The answer streams back to the browser and is saved to chat history.
8. Use **New chat** to reset the current session while keeping uploaded documents available.

## Project Structure

```text
.
├── main.py                  # FastAPI app, routes, upload/chat/history endpoints
├── config.py                # Environment-driven settings
├── logging_config.py        # Logging setup
├── requirements.txt         # Python dependencies
├── static/
│   └── index.html           # Single-page web UI
├── services/
│   ├── llm_service.py       # Groq + LangChain streaming RAG pipeline
│   └── vector_service.py    # ChromaDB ingestion and retrieval logic
├── database/
│   └── sqlite_db.py         # SQLite chat history persistence
└── schemas/
    └── models.py            # Pydantic API schemas
```

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
CHROMA_PERSIST_DIR=./chroma_db
SQLITE_DB_PATH=./chat_history.db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
COLLECTION_NAME=pocket_librarian
```

`config.py` can also read a local `Groq-API.txt` fallback, but that file should never be committed.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

The embedding model may download the first time the app starts.

## API Overview

- `GET /` serves the frontend.
- `POST /api/upload` uploads and ingests a document.
- `GET /api/documents` lists uploaded documents and chunk counts.
- `DELETE /api/documents/{filename}` deletes a document from ChromaDB.
- `POST /api/chat` streams a RAG answer for a question.
- `GET /api/history` lists chat sessions.
- `GET /api/history/{session_id}` loads messages for a session.

## Documentation
<img width="2880" height="1800" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/cdce0c18-ed3b-4558-874a-d298916c65fd" />
