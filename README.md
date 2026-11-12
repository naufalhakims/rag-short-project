# Pocket Librarian

Ask questions about your own documents and get answers that cite their sources.

Pocket Librarian is a local retrieval-augmented generation (RAG) web app. You upload PDF, Word, Markdown or text files; it indexes them on your machine; and a language model answers your questions using only what the documents say, with numbered citations you can click to read the passage behind each claim.

## Features

- **Grounded, cited answers.** Every claim carries a `[n]` marker linked to the exact passage it came from. If the documents do not contain the answer, the app says so.
- **Hybrid retrieval.** Vector search and keyword search (BM25) run together, their results are fused, and a cross-encoder reranks the candidates.
- **Conversations that follow context.** Follow-ups such as "and what about its penalties?" are rewritten into standalone search queries before retrieval.
- **Resilient model access.** The app asks each provider which models your key can use, picks a working one if the configured model has been retired, and fails over to the next model or provider on rate limits and outages.
- **Several providers.** Groq and Google Gemini (both have free tiers), Anthropic Claude, or any OpenAI-compatible endpoint, including local ones such as Ollama.
- **Local retrieval models.** Embedding and reranking run on your CPU through ONNX. No GPU or PyTorch is needed, and document text is only sent to the LLM as the few passages relevant to a question.
- **PDF, DOCX, Markdown and text** ingestion, with page numbers and section headings kept for citations.
- **Web UI** with streaming answers, a stop button, saved conversations, per-document search scope, and drag-and-drop upload.

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

Set a key for at least one provider. With `LLM_PROVIDER=auto` (the default) every provider that has a key is used, in the order below, and the app moves on to the next one when a request fails before any text has been produced.

| Provider | Key | Default model | Notes |
| --- | --- | --- | --- |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | Paid API. |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` | Free tier, very fast. Follow-up rewriting uses `openai/gpt-oss-20b`. |
| Google Gemini | `GOOGLE_API_KEY` | newest `flash-lite` | Free tier. |
| OpenAI-compatible | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | none | OpenAI, OpenRouter, Ollama, LM Studio and similar. |

Set `LLM_PROVIDER=groq` (or another name) to put that provider first. Set `LLM_FALLBACK=false` to use only the first provider and its configured model.

Notes:

- **Groq free tier** allows 8,000 tokens per minute per model and counts the requested output size against it, so the app keeps Groq answers to 1,024 output tokens by default (`LLM_MAX_OUTPUT_TOKENS`). When one model is rate-limited the next one takes over.
- **Claude** requests on current models opt in to Anthropic's server-side refusal fallback, so a request declined by a safety classifier is retried on Anthropic's recommended fallback model.

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

A conversation turn is saved only once it has an answer. Failed requests leave nothing behind, and an answer you stop midway is saved as far as it got.

## Project layout

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