# Pocket Librarian

Ask questions about your own documents and get answers that cite their sources.

Pocket Librarian is a local retrieval-augmented generation (RAG) web app. You upload PDF, Word, Markdown or text files; it indexes them on your machine; and a language model answers your questions using only what the documents say, with numbered citations you can click to read the passage behind each claim.

## Features

- **Each chat has its own documents.** A chat only searches the files added to it, so topics never mix: a question in one chat cannot pull passages from another chat's files. Files are stored once and can be reused in any chat.
- **Tag documents with @.** Type `@` in a question to pick one of the chat's documents; that question then searches only the tagged documents.
- **Grounded, cited answers.** Every claim carries a `[n]` marker linked to the exact passage it came from. If the documents do not contain the answer, the app says so.
- **Hybrid retrieval.** Vector search and keyword search (BM25) run together, their results are fused, and a cross-encoder reranks the candidates.
- **Conversations that follow context.** Follow-ups such as "and what about its penalties?" are rewritten into standalone search queries before retrieval.
- **Resilient model access.** The app asks each provider which models your key can use, picks a working one if the configured model has been retired, and fails over to the next model or provider on rate limits and outages.
- **Runs on free models.** Groq is the default and Google Gemini is the backup; both have free tiers. Any OpenAI-compatible endpoint also works, including local models through Ollama. Anthropic Claude is supported as an optional paid provider.
- **Local retrieval models.** Embedding and reranking run on your CPU through ONNX. No GPU or PyTorch is needed, and document text is only sent to the LLM as the few passages relevant to a question.
- **PDF, DOCX, Markdown and text** ingestion, with page numbers and section headings kept for citations.
- **Web UI** with streaming answers, a stop button, saved conversations, a document list per chat, and drag-and-drop upload.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env            # macOS/Linux: cp .env.example .env
# edit .env and set GROQ_API_KEY (free: https://console.groq.com/keys)

python main.py
```

Open <http://127.0.0.1:8000>. The first start downloads the two retrieval models (about 150 MB) into `data/models/`; the status pill in the top right turns green when everything is ready.

`sample_docs/` contains two short documents to try it with.

## Configuration

All settings live in `.env`; see [`.env.example`](.env.example) for the full list with defaults.

### Language models

The project is set up for free models: a Groq key alone is enough, and a Google Gemini key adds a free backup. With `LLM_PROVIDER=auto` (the default) every provider that has a key is used, and the app moves on to the next one when a request fails before any text has been produced.

| Provider | Key | Default model | Cost |
| --- | --- | --- | --- |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` | Free tier, very fast. Follow-up rewriting uses `openai/gpt-oss-20b`. |
| Google Gemini | `GOOGLE_API_KEY` | newest `flash-lite` | Free tier. |
| OpenAI-compatible | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | none | Free when local (Ollama, LM Studio); otherwise depends on the service. |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | Paid, optional. Not needed to run the project. |

If a Claude key is set, it is tried before the free providers. Leave `ANTHROPIC_API_KEY` empty to stay on free models.

Set `LLM_PROVIDER=groq` (or another name) to put that provider first. Set `LLM_FALLBACK=false` to use only the first provider and its configured model.

Notes:

- **Groq free tier** allows 8,000 tokens per minute per model and counts the requested output size against it, so the app keeps Groq answers to 1,024 output tokens by default (`LLM_MAX_OUTPUT_TOKENS`). When one model is rate-limited the next one takes over.
- **Gemini free tier** has a daily request cap; when it runs out the app keeps using Groq.
- **Claude** (optional) requests on current models opt in to Anthropic's server-side refusal fallback, so a request declined by a safety classifier is retried on Anthropic's recommended fallback model.

### Retrieval

| Setting | Default | Meaning |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any dense model [fastembed](https://qdrant.github.io/fastembed/examples/Supported_Models/) supports. |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Any fastembed cross-encoder; `none` disables reranking. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `120` | Passage size in characters. Applies to new uploads. |
| `TOP_K` / `MIN_PASSAGES` | `6` / `3` | Most and fewest passages given to the LLM once something matches. |
| `RERANK_MIN_SCORE` | `0.005` | Relevance cut-off (0 to 1) below which a passage is not considered a match. |

The defaults are tuned for English. For documents mostly in other languages, switch to a multilingual embedder and turn the English-only reranker off:

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANKER_MODEL=none
```

Changing the embedding model needs no manual step: on the next start the app notices and re-embeds your library.

## How it works

```text
question ──► rewrite ──► retrieve ──────────────────────────► generate ──► cited answer
             (only for    vector search ─┐                     prompt = numbered
             follow-ups)                 ├─► fuse ─► rerank    passages + question
                          keyword search ┘   (RRF)   (cross-encoder)
```

1. **Ingest.** A file is converted to text (page by page for PDFs), split into passages of about 900 characters that prefer to start at headings and overlap slightly, embedded, and stored. SQLite holds the text and a full-text index; Chroma holds the vectors.
2. **Rewrite.** If the conversation has earlier turns, a small fast model turns the new message into a standalone search query.
3. **Retrieve.** The query runs against the vector index and the keyword index. The two rankings are merged with reciprocal rank fusion, and a cross-encoder scores the best 12 candidates against the query.
4. **Select.** Passages above the relevance cut-off are kept, up to six and never fewer than three. If nothing matches (a greeting, or "summarise my documents"), the opening passages of each document are sent instead so the model still knows what the library contains.
5. **Generate.** The model receives the numbered passages and the question, and is instructed to answer only from them and cite as it goes. The answer streams to the browser; cited passages are shown underneath.

### Documents belong to chats

Every chat has its own list of documents, shown in the sidebar under "This chat's documents", and search only ever looks inside that list.

- A file uploaded while a chat is open is added to that chat only. A new chat starts with no documents.
- Each file is stored and indexed once. "Add from your files" reuses a file you uploaded in another chat without uploading it again.
- Removing a file from a chat (the ×) keeps it for other chats. Deleting it from "Add from your files" (the bin) removes it everywhere.
- Re-uploading a file with the same name updates it in every chat that uses it.
- Deleting a chat keeps its files.

### Asking about a specific document with @

Type `@` in the question box to open a list of the chat's documents, then pick one with the mouse, or with the arrow keys and Enter or Tab. Tag as many as you like. While any document is tagged, that question searches only the tagged documents; untagged questions search all of the chat's documents as usual.

- A tag is `@` plus the file name, with or without its extension: `@paper.pdf` or `@paper`. Names with spaces work.
- A tag on its own is enough for broad requests: "summarise @paper" sends the opening of that document.
- Tags only reach the chat's own documents. A name that matches nothing is treated as ordinary text.
- The same rule applies to `POST /api/chat`; the `sources` event lists the tagged documents under `mentions`.

A conversation turn is saved only once it has an answer. Failed requests leave nothing behind, and an answer you stop midway is saved as far as it got.

## Deploy to a server

The repository includes everything needed to run the app on a Linux server with Docker:

- `Dockerfile`: builds the app with the retrieval models inside the image.
- `docker-compose.yml`: runs the app next to [Caddy](https://caddyserver.com), which serves it over HTTPS with a free, automatically renewed certificate. Documents and chats are kept in a Docker volume, so they survive restarts and updates.
- `deploy/ORACLE.md`: a step-by-step guide for Oracle Cloud's **Always Free** Arm server, from creating the account to updating and backing up.

The short version, on a server with Docker and ports 80 and 443 open:

```bash
git clone https://github.com/naufalhakims/rag-short-project.git pocket-librarian
cd pocket-librarian
# create .env with GROQ_API_KEY, APP_PASSWORD and DOMAIN (see deploy/ORACLE.md)
docker compose up -d --build
```

**Always set `APP_PASSWORD` on a public server.** Without it, anyone who finds the address can read, upload and delete documents and use up your API quota.

## Project layout

```text
main.py                  Entry point (python main.py, or uvicorn main:app)
app/
  main.py                FastAPI app and startup
  api.py                 HTTP routes
  auth.py                Optional password login
  services.py            Wires the components together
  config.py              Settings (.env)
  db.py                  SQLite: documents, passages, full-text index, chat history
  ingest/
    loaders.py           PDF, DOCX, Markdown and text to plain text
    chunking.py          Text to passages
  rag/
    models.py            Local embedding and reranking models
    index.py             Document library and Chroma vector index
    retriever.py         Hybrid search, fusion, reranking
    prompts.py           Prompts and citation handling
    mentions.py          @-tags that limit a question to specific documents
    pipeline.py          The chat pipeline (rewrite, retrieve, generate)
  llm/
    router.py            Model discovery and failover across providers
    openai_compat.py     Groq, Gemini and other OpenAI-compatible APIs
    anthropic_provider.py  Claude
static/                  Web UI (plain HTML, CSS and JavaScript; no build step)
tests/                   Test suite (fake models and LLM; runs offline)
sample_docs/             Two example documents
data/                    Created at runtime: database, vector index, models, log
```

## API

Interactive documentation is at <http://127.0.0.1:8000/docs>.

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Active provider and model, retrieval models, library size |
| `GET /api/documents` | List every uploaded document |
| `POST /api/documents?session_id=…` | Upload a file (multipart field `file`) and add it to that chat; a repeated file name replaces the earlier version |
| `DELETE /api/documents/{id}` | Delete a document from every chat |
| `GET /api/search?q=…&session_id=…` | Run retrieval only, within a chat's documents, and return the passages with scores |
| `POST /api/chat` | Ask a question; responds with a Server-Sent Events stream |
| `GET /api/history` | List conversations |
| `POST /api/history` | Start an empty conversation (to add documents before the first question) |
| `GET /api/history/{id}` | Messages of a conversation |
| `DELETE /api/history/{id}` | Delete a conversation (its documents are kept) |
| `GET /api/history/{id}/documents` | The documents a conversation searches |
| `POST /api/history/{id}/documents` | Add uploaded documents to a conversation: `{"doc_ids": ["..."]}` |
| `DELETE /api/history/{id}/documents/{doc_id}` | Remove a document from a conversation only |

`POST /api/chat` takes `{"question": "...", "session_id": "...", "doc_ids": ["..."]}` and streams JSON events: `session`, `status`, `sources`, `token` (repeated), then `done` or `error`. It searches only the conversation's documents. `doc_ids` is optional: for an existing conversation it narrows the search further, and without a `session_id` it sets the documents of the new conversation.

## Development

```bash
pip install -r requirements-dev.txt
pytest                                  # about 80 tests, no network or model downloads needed
ruff check .
uvicorn app.main:app --reload           # run with auto-restart on code changes
```

## Troubleshooting

- **"No language model" in the status pill.** No provider key was found. Add one to `.env` and restart.
- **"The model … is not available".** Providers retire models. Leave the model setting empty or at its default and the app chooses from what your key can access; `GET /api/health` shows what it picked.
- **"Rate limit reached".** Free tiers are small. Wait for the time shown, or add a second provider's key so the app can fail over.
- **A scanned PDF is rejected.** It has no text layer. Run it through OCR first.
- **"The sources do not contain…" for something you uploaded.** The file is probably in a different chat. Add it to this one with "Add from your files".
- **Answers miss something that is in a document.** Use `GET /api/search?q=…&session_id=…` to see what retrieval returns for the question. Raising `TOP_K`, or lowering `RERANK_MIN_SCORE`, sends the model more passages.
- **Starting over.** Stop the app and delete the `data/` folder.

## Screenshot

<img width="2880" height="1800" alt="Pocket Librarian web interface" src="https://github.com/user-attachments/assets/cdce0c18-ed3b-4558-874a-d298916c65fd" />
